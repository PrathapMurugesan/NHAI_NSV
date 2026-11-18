"""
Cracking processing, D2 calibration, aggregation, and severity classification.

Migrated from Notebook Cell 8 (Cracking Correction), Cell 9 (Map All Defects), and Cell 12 (Cracking Severity).
Preserves all business logic:
- D2 calibration factor: factor(bin) = min(D2_area / parquet_area, 1.0)
- Cracking % (10m) = area / 35.0 * 100
- Dual records: 'As per CA' and cloned 'As per IRC'
- 100m and 1km aggregation: crack_count-weighted mean for corrected width
- Severity classification: width < 3 -> None, 3 <= width < 6 -> Medium, width >= 6 -> High
- Visual condition and color coding for both CA and IRC standards
"""

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.constants import (
    CHILD_100M,
    DEFECT_INTERVALS,
    LANE_AREA_1000M,
    LANE_AREA_100M,
    LANE_AREA_10M,
    PARENT_LENGTH_M,
    as_float,
    as_int,
    as_text,
    first_non_null,
    lane_to_direction,
    normalize_lane,
)
from src.common.logger import logger
from config.table_schema import BASE_DEFECT_COLUMNS
from src.ingestion.bronze_loader import crack_condition_from_pct
from src.ingestion.d1_loader import D1Context
from src.ingestion.tracker_loader import get_reporting_interval, get_threshold_name
from src.common.fs_utils import path_exists, list_dir, read_parquet_df


def _col(cols: List[str], *must: str, exclude: Tuple[str, ...] = ()) -> Optional[str]:
    return next((c for c in cols if all(m in c.lower() for m in must) and not any(x in c.lower() for x in exclude)), None)


def _load_d2_file(d2_base: str, name_filter) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    if not d2_base or not path_exists(d2_base):
        return None, None, None
    files = sorted(f for f in list_dir(d2_base) if f.endswith(".parquet") and name_filter(f))
    for f in files:
        file_path = f"{d2_base}/{f}" if d2_base.startswith("s3://") else os.path.join(d2_base, f)
        try:
            df = read_parquet_df(file_path)
        except Exception:
            continue
        ch = _col(list(df.columns), "start", "chainage")
        if ch:
            df[ch] = pd.to_numeric(df[ch], errors="coerce")
            return df.dropna(subset=[ch]), ch, f
    return None, None, None


def _row_interval_m(df: pd.DataFrame, ch: str) -> str:
    tags = " ".join(df.columns).lower()
    if "1000m" in tags:
        return "1000m"
    if "100m" in tags:
        return "100m"
    end = _col(list(df.columns), "end", "chainage")
    if end is None:
        return "1000m"
    med = (pd.to_numeric(df[end], errors="coerce") - df[ch]).dropna().median()
    return "100m" if med <= 200 else "1000m"


def calibrate_cracking_with_d2(
    df_cracking_raw: pd.DataFrame,
    d2_base: str,
    ca_crack_pct: float = 10.0,
) -> pd.DataFrame:
    """
    Calibrate raw cracking backup area with D2 data.
    Factor = min(D2_area / parquet_area, 1.0)
    """
    if df_cracking_raw.empty or not path_exists(d2_base):
        return df_cracking_raw

    d2_ca, d2ch_ca, _ = _load_d2_file(d2_base, lambda f: "CA" in f)
    d2_irc, d2ch_irc, _ = _load_d2_file(d2_base, lambda f: "IRC" in f)

    if d2_ca is None and d2_irc is None:
        logger.warning("No D2 CA or IRC file found for cracking calibration.")
        return df_cracking_raw

    ca_interval = _row_interval_m(d2_ca, d2ch_ca) if d2_ca is not None else "1000m"
    irc_interval = None
    if d2_irc is not None:
        irc_interval = "100m" if any("crack" in c.lower() and "100m" in c.lower() for c in d2_irc.columns) else _row_interval_m(d2_irc, d2ch_irc)

    if d2_irc is not None and irc_interval == "100m":
        calib_src, calib_m, calib_df, calib_ch = "IRC", 100, d2_irc.copy(), d2ch_irc
    else:
        calib_src, calib_m, calib_df, calib_ch = "CA", (100 if ca_interval == "100m" else 1000), d2_ca.copy(), d2ch_ca

    calib_df["_cbin"] = (calib_df[calib_ch] // calib_m * calib_m).astype(int)
    calib_lane_area = calib_m * 3.5

    all_lanes = sorted(df_cracking_raw["Lane"].dropna().unique())
    calib_cols = {ln: _col(list(calib_df.columns), ln.lower(), "crack", exclude=("limit",)) for ln in all_lanes}

    sch_all = df_cracking_raw[["Start_Chainage", "End_Chainage"]].min(axis=1)

    for lane in all_lanes:
        col = calib_cols.get(lane)
        if col is None:
            continue
        m = df_cracking_raw["Lane"] == lane
        d2_area = (
            pd.to_numeric(calib_df[col], errors="coerce").fillna(0)
            .groupby(calib_df["_cbin"]).mean() / 100.0 * calib_lane_area
        )
        cbin = (sch_all[m] // calib_m * calib_m).astype(int)
        area = pd.to_numeric(df_cracking_raw.loc[m, "Area_sq_m"], errors="coerce")
        pq_total = area.groupby(cbin).transform("sum")
        d2_val = cbin.map(d2_area)
        factor = np.where(d2_val.notna() & (pq_total > 0), np.minimum(d2_val / pq_total, 1.0), 1.0)
        new_area = area * factor
        df_cracking_raw.loc[m, "Area_sq_m"] = new_area
        df_cracking_raw.loc[m, "Cracking"] = new_area / LANE_AREA_10M * 100.0

    df_cracking_raw["Condition"] = pd.to_numeric(df_cracking_raw["Cracking"], errors="coerce").fillna(0).apply(
        lambda v: crack_condition_from_pct(v, ca_crack_pct)
    )
    logger.info(f"Cracking calibrated against D2 {calib_src} at {calib_m}m.")
    return df_cracking_raw


def build_cracking_pdf(
    df_raw: pd.DataFrame,
    upc: str,
    zone: str,
    phase: str,
    cycle: str,
    d1_ctx: D1Context,
    tracker_ctx: Dict[str, Any],
    run_date: date,
    ca_crack_pct: float,
    irc_crack_pct: float,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build 10m cracking table with dual rows (As per CA and As per IRC)."""
    target_columns = columns or list(BASE_DEFECT_COLUMNS)
    if df_raw.empty:
        return pd.DataFrame(columns=target_columns)

    survey_date_str = run_date.strftime("%d%m%y")
    parent_m = DEFECT_INTERVALS["cracking"]["parent_m"]
    upc_slash = upc.replace("_", "/")
    run_ts = pd.Timestamp(run_date)
    now_ts = pd.Timestamp.now()

    records = []
    for idx, row in df_raw.iterrows():
        start_ch = as_int(row.get("Start_Chainage"))
        end_ch = as_int(row.get("End_Chainage"))
        if start_ch is not None and end_ch is not None and end_ch < start_ch:
            start_ch, end_ch = end_ch, start_ch

        lane_val = normalize_lane(row.get("Lane"))
        road_ctx = d1_ctx.get_road_context(start_ch, end_ch, lane=lane_val)
        crack_val = as_float(row.get("Cracking"))
        condition_val = as_text(row.get("Condition"))

        parent_start = (start_ch // parent_m) * parent_m if start_ch is not None else None
        parent_end = parent_start + parent_m if parent_start is not None else None
        defect_id_val = f"D/{survey_date_str}/N/{upc}/{start_ch if start_ch is not None else 0}/Cracking/{idx+1:04d}"
        record_id_val = f"cracking_{idx+1:06d}"
        parent_id_val = f"{upc}_{lane_val}_{parent_start}_{parent_end}" if lane_val and parent_start is not None else None

        bf_ca = 1 if crack_val is not None and crack_val > ca_crack_pct else 0
        seg_len = float(abs(end_ch - start_ch)) if start_ch is not None and end_ch is not None else 0.0

        rec_ca = {col: None for col in target_columns}
        rec_ca.update({
            "id": f"{upc}_master_cracking_{idx+1:06d}",
            "collection_name": "master_cracking",
            "defect_base": "cracking",
            "level": "child_10m",
            "defect_id": defect_id_val if bf_ca == 1 else None,
            "display_defect_id": defect_id_val if bf_ca == 1 else None,
            "source_defect_id": record_id_val,
            "record_id": record_id_val,
            "parent_id": parent_id_val,
            "upc": upc_slash,
            "upc_code": upc,
            "nh_number": first_non_null(road_ctx.get("nh_number")),
            "section_code": first_non_null(road_ctx.get("section_code")),
            "zone": zone,
            "phase": phase,
            "cycle": int(cycle),
            "source_system": "NSV",
            "sources": ["NSV"],
            "defect_type": "Cracking",
            "type_of_defect": "Cracking",
            "category": "LENGTH",
            "severity": condition_val,
            "pavement_type": first_non_null(road_ctx.get("pavement_type")),
            "pci": tracker_ctx.get("pci"),
            "lane": lane_val,
            "breach_flag": bf_ca,
            "metric_name": "cracking_pct",
            "metric_value": crack_val,
            "threshold_name": "As per CA",
            "threshold_value": ca_crack_pct,
            "total_children": 1,
            "length_m": seg_len,
            "total_length": seg_len,
            "affected_length": round(seg_len, 4) if bf_ca == 1 else 0.0,
            "affected_length_ca": round(seg_len, 4) if bf_ca == 1 else 0.0,
            "affected_length_irc": 0.0,
            "area_sqm": as_float(row.get("Area_sq_m")),
            "start_chainage": start_ch,
            "end_chainage": end_ch,
            "defect_chainage": (start_ch + end_ch) / 2 if start_ch is not None and end_ch is not None else start_ch,
            "start_lat": as_float(row.get("Start_Lat")),
            "start_long": as_float(row.get("Start_Lon")),
            "end_lat": as_float(row.get("End_Lat")),
            "end_long": as_float(row.get("End_Lon")),
            "defect_lat": as_float(row.get("Start_Lat")),
            "defect_long": as_float(row.get("Start_Lon")),
            "datetime": run_ts,
            "observed_at": run_ts,
            "survey_start_date": tracker_ctx.get("survey_start_date"),
            "survey_end_date": tracker_ctx.get("survey_end_date"),
            "image_name": as_text(row.get("Image_Name")),
            "image_urls": [as_text(row.get("Image_Name"))] if as_text(row.get("Image_Name")) else None,
            "is_duplicate": False,
            "nsv": json.dumps({"condition": condition_val, "cracking": crack_val}),
            "project_details": json.dumps({"project_name": upc, "zone": zone, "phase": phase, "cycle": cycle}),
            "created_at": now_ts,
            "updated_at": now_ts,
            "carriageway": first_non_null(road_ctx.get("carriageway")),
            "direction": first_non_null(lane_to_direction(lane_val), road_ctx.get("direction")),
            "parent_start_chainage": parent_start,
            "parent_end_chainage": parent_end,
            "reporting_interval_m": get_reporting_interval(tracker_ctx, "cracking"),
            "segment_count": 1,
            "cracking_pct": crack_val,
            "rating_ca": condition_val,
            "rating_irc": None,
            "defect_type_filter": "Cracking",
            "has_standard_defect": True,
            "standard_flags": "{}",
            "measures": json.dumps({
                "cracking_pct": crack_val,
                "crack_area_sqm": as_float(row.get("Area_sq_m")),
                "lane_area_sqm": 35.0,
                "crack_count": as_float(row.get("crack_count")),
                "avg_corrected_width_mm": as_float(row.get("avg_corrected_width_mm")),
                "avg_width_mm": as_float(row.get("avg_width_mm")),
                "calculation": "Crack area = length x corrected width; sum per 1m capped at 3.5 sqm; Cracking % = sum(10m area) / 35 x 100",
            }),
            "run_date": run_date,
            "detected_on": tracker_ctx.get("survey_end_date"),
        })
        records.append(rec_ca)

    ca_pdf = pd.DataFrame(records)

    # Clone for IRC rows
    irc_pdf = ca_pdf.copy()
    if not irc_pdf.empty:
        irc_pdf["id"] = irc_pdf["id"].str.replace("master_cracking_", "master_cracking_irc_")
        irc_pdf["record_id"] = irc_pdf["record_id"] + "_irc"
        irc_pdf["source_defect_id"] = irc_pdf["source_defect_id"] + "_irc"
        irc_pdf["threshold_name"] = "As per IRC"
        irc_pdf["threshold_value"] = irc_crack_pct
        irc_pdf["breach_flag"] = irc_pdf["metric_value"].apply(
            lambda v: 1 if v is not None and not pd.isna(v) and float(v) >= irc_crack_pct else 0
        )
        irc_pdf["affected_length"] = irc_pdf.apply(
            lambda r: round(float(r["length_m"] or 0), 4) if r["breach_flag"] == 1 else 0.0, axis=1
        )
        irc_pdf["affected_length_irc"] = irc_pdf["affected_length"]
        irc_pdf["affected_length_ca"] = 0.0
        irc_pdf.loc[irc_pdf["breach_flag"] == 0, "defect_id"] = None
        irc_pdf.loc[irc_pdf["breach_flag"] == 0, "display_defect_id"] = None

        need_id = (irc_pdf["breach_flag"] == 1) & irc_pdf["display_defect_id"].isna()
        if need_id.any():
            seqs = irc_pdf.loc[need_id, "record_id"].str.extract(r"cracking_(\d+)")[0].astype(int)
            sts = pd.to_numeric(irc_pdf.loc[need_id, "start_chainage"], errors="coerce").fillna(0).astype(int)
            new_ids = [f"D/{survey_date_str}/N/{upc}/{s}/Cracking/{q:04d}" for s, q in zip(sts, seqs)]
            irc_pdf.loc[need_id, "defect_id"] = new_ids
            irc_pdf.loc[need_id, "display_defect_id"] = new_ids

    full_pdf = pd.concat([ca_pdf, irc_pdf], ignore_index=True)
    apply_cracking_severity_and_condition(full_pdf, ca_crack_pct)
    return full_pdf


def build_cracking_agg_pdf(
    ca_pdf: pd.DataFrame,
    bin_m: int,
    level: str,
    ca_crack_pct: float,
    upc: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build aggregated 100m or 1km cracking tables using CA rows."""
    target_columns = columns or list(BASE_DEFECT_COLUMNS)
    if ca_pdf.empty:
        return pd.DataFrame(columns=target_columns)

    d = ca_pdf.copy()
    def _parse_measures(m):
        try:
            return json.loads(m) if isinstance(m, str) else {}
        except Exception:
            return {}

    m_series = d["measures"].apply(_parse_measures)
    d["_cc"] = pd.to_numeric(m_series.apply(lambda x: x.get("crack_count")), errors="coerce").fillna(0)
    d["_wc"] = pd.to_numeric(m_series.apply(lambda x: x.get("avg_corrected_width_mm")), errors="coerce")
    d["_wr"] = pd.to_numeric(m_series.apply(lambda x: x.get("avg_width_mm")), errors="coerce")
    d["_wcn"] = d["_wc"].fillna(d["_wr"]).fillna(0) * d["_cc"]
    d["_wrn"] = d["_wr"].fillna(0) * d["_cc"]
    d["_area"] = pd.to_numeric(d["area_sqm"], errors="coerce").fillna(0)
    d["_st"] = pd.to_numeric(d["start_chainage"], errors="coerce")
    d["_en"] = pd.to_numeric(d["end_chainage"], errors="coerce")
    d = d[d["_st"].notna() & d["_en"].notna()].copy()
    d["_bin"] = (d["_st"] // bin_m * bin_m).astype("int64")
    d = d.sort_values(["lane", "_st"])

    tbl_name = "master_cracking_child_100m" if level == "child_100m" else "master_cracking_parent"
    out = []

    for i, ((lane, b), g) in enumerate(d.groupby(["lane", "_bin"], sort=True)):
        first, last = g.iloc[0], g.iloc[-1]
        st, en = int(g["_st"].min()), int(g["_en"].max())
        seg_len = float(en - st)
        area = float(g["_area"].sum())
        lane_area = 3.5 * seg_len
        pct = round(area / lane_area * 100.0, 6) if lane_area > 0 else 0.0
        n = float(g["_cc"].sum())
        w_corr = round(float(g["_wcn"].sum() / n), 4) if n > 0 else 0.0
        w_raw = round(float(g["_wrn"].sum() / n), 4) if n > 0 else 0.0
        bf = 1 if pct is not None and pct > ca_crack_pct else 0
        p_start = (st // PARENT_LENGTH_M) * PARENT_LENGTH_M
        rid = f"cracking_{level}_{i+1:06d}"

        rec = {c: first.get(c) for c in target_columns}
        rec.update({
            "id": f"{upc}_{tbl_name}_{i+1:06d}",
            "collection_name": tbl_name,
            "level": level,
            "defect_id": None,
            "display_defect_id": None,
            "source_defect_id": rid,
            "record_id": rid,
            "parent_id": f"{upc}_{lane}_{p_start}_{p_start + PARENT_LENGTH_M}",
            "breach_flag": bf,
            "metric_name": "cracking_pct",
            "metric_value": pct,
            "cracking_pct": pct,
            "threshold_name": "As per CA",
            "threshold_value": ca_crack_pct,
            "total_children": int(len(g)),
            "segment_count": int(len(g)),
            "length_m": seg_len,
            "total_length": seg_len,
            "area_sqm": round(area, 6),
            "start_chainage": st,
            "end_chainage": en,
            "defect_chainage": (st + en) / 2,
            "start_lat": first.get("start_lat"),
            "start_long": first.get("start_long"),
            "end_lat": last.get("end_lat"),
            "end_long": last.get("end_long"),
            "defect_lat": first.get("start_lat"),
            "defect_long": first.get("start_long"),
            "parent_start_chainage": p_start,
            "parent_end_chainage": p_start + PARENT_LENGTH_M,
            "reporting_interval_m": bin_m,
            "nsv": json.dumps({"cracking": pct}),
            "measures": json.dumps({
                "cracking_pct": pct,
                "crack_area_sqm": round(area, 6),
                "lane_area_sqm": lane_area,
                "crack_count": n,
                "avg_corrected_width_mm": w_corr,
                "avg_width_mm": w_raw,
                "segments": int(len(g)),
            }),
            "affected_length": seg_len if bf == 1 else 0.0,
            "affected_length_ca": seg_len if bf == 1 else 0.0,
            "affected_length_irc": 0.0,
        })
        out.append(rec)

    res_df = pd.DataFrame(out)
    apply_cracking_severity_and_condition(res_df, ca_crack_pct)
    return res_df


def apply_cracking_severity_and_condition(df: pd.DataFrame, ca_crack_pct: float) -> pd.DataFrame:
    """
    Apply severity and visual condition calculations to cracking DataFrame.
    """
    if df.empty:
        return df

    def _calc_severity(m_str):
        try:
            m = json.loads(m_str) if isinstance(m_str, str) else {}
            w = m.get("avg_corrected_width_mm") or m.get("avg_width_mm") or 0.0
            w = float(w)
            if w < 3: return "None", w
            if w < 6: return "Medium", w
            return "High", w
        except Exception:
            return "None", 0.0

    def _calc_ca_cond(p):
        if pd.isna(p) or p is None: return None, None
        p_val = float(p)
        if p_val <= ca_crack_pct: return "Good", "#2a8524"
        return "Poor", "#e31a1c"

    def _calc_irc_cond(p):
        if pd.isna(p) or p is None: return None, None
        p_val = float(p)
        if p_val >= 30: return "Very Poor (>30%)", "#e31a1c"
        if p_val >= 20: return "Poor (20-30%)", "#ff7f00"
        if p_val >= 10: return "Fair (10-20%)", "#fff14b"
        if p_val >= 5: return "Good (5-10%)", "#0f6ad0"
        return "Very Good (<5%)", "#2a8524"

    out = df.copy()

    sev_list, ca_list, irc_list, cond_list, measures_list = [], [], [], [], []
    thresh_col = out["threshold_name"] if "threshold_name" in out.columns else [None] * len(out)
    metric_col = out["metric_value"] if "metric_value" in out.columns else [None] * len(out)
    meas_col = out["measures"] if "measures" in out.columns else [None] * len(out)

    for m_val, metric_p, thresh_name in zip(meas_col, metric_col, thresh_col):
        sev_label, width_val = _calc_severity(m_val)
        ca_lbl, ca_hex = _calc_ca_cond(metric_p)
        irc_lbl, irc_hex = _calc_irc_cond(metric_p)

        is_irc = "irc" in str(thresh_name or "").lower()
        cond_lbl = irc_lbl if is_irc else ca_lbl
        cond_hex = irc_hex if is_irc else ca_hex

        try:
            m = json.loads(m_val) if isinstance(m_val, str) else {}
        except Exception:
            m = {}
        m.update({
            "severity_width_mm": round(width_val, 4),
            "severity": sev_label,
            "visual_condition_ca": ca_lbl,
            "condition_color_ca": ca_hex,
            "visual_condition_irc": irc_lbl,
            "condition_color_irc": irc_hex,
            "condition_color": cond_hex,
        })

        sev_list.append(sev_label)
        ca_list.append(ca_lbl)
        irc_list.append(irc_lbl)
        cond_list.append(cond_lbl)
        measures_list.append(json.dumps(m))

    out["severity"] = sev_list
    out["rating_ca"] = ca_list
    out["rating_irc"] = irc_list
    out["road_condition"] = cond_list
    out["measures"] = measures_list
    return out
