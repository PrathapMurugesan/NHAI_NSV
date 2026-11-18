"""
Pothole processing, D2 fallback loader, image mapping, and hierarchy builder.

Migrated from Notebook Cell 5 (Pothole Fallback) & Cell 6 (Build Pothole 3-Level Hierarchy) & Cell 9.
Preserves all business logic:
- Fuzzy-matching D2 pothole sheet (score >= 0.85) when D4 is missing
- 10m backup segment indexing for image name, lat, lon, survey date
- Image filename regex indexing (*_{Lane}_{run}_{frame}.jpg)
- Parent (1000m) and Child grouping chainages
- Point chainage and coordinates assignment
- Defect ID generation: D/{SURVEY_DATE_STR}/N/{UPC}/{Start_Chainage}/Pothole/{counter:04d}
- master_pothole DataFrame creation adhering to table schema
"""

import difflib
import json
import os
import re
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.common.constants import (
    DEFECT_INTERVALS,
    PARENT_LENGTH_M,
    RAW_SEGMENT_M,
    as_float,
    as_int,
    as_text,
    first_non_null,
    lane_to_direction,
    normalize_lane,
)
from src.common.logger import logger
from config.table_schema import BASE_DEFECT_COLUMNS
from src.ingestion.d1_loader import D1Context
from src.ingestion.tracker_loader import get_reporting_interval, get_threshold_name
from src.common.fs_utils import path_exists, list_dir, read_parquet_df, walk_files

POTHOLE_SHEET_MIN_SCORE = 0.85


def _sheet_token(fname: str) -> str:
    return os.path.splitext(fname)[0].rsplit("_", 1)[-1].strip()


def _pothole_sheet_score(fname: str) -> float:
    tok = _sheet_token(fname).lower()
    tok = re.sub(r"^[a-z]{3}-?\d{2}\s*", "", tok)  # drop month prefix like 'Oct-25 '
    tok = re.sub(r"[^a-z]", "", tok)
    if not tok:
        return 0.0
    if "pothole" in tok:
        return 1.0
    if len(tok) >= 2 and "potholes".startswith(tok):
        return 0.9  # truncated name 'Po' / 'Pot'
    return difflib.SequenceMatcher(None, tok, "potholes").ratio()


def _pick(cols: List[str], *alts: Tuple[str, ...]) -> Optional[str]:
    for kws in alts:
        for c in cols:
            cl = str(c).lower().replace("\n", " ")
            if all(k in cl for k in kws):
                return c
    return None


def load_d2_potholes(d2_base: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load best fuzzy-matched D2 pothole sheet standardized to D4 columns."""
    if not d2_base or not path_exists(d2_base):
        return pd.DataFrame(), None

    cands = []
    for f in sorted(list_dir(d2_base)):
        if not f.endswith(".parquet"):
            continue
        score = _pothole_sheet_score(f)
        if score < POTHOLE_SHEET_MIN_SCORE:
            continue
        file_path = f"{d2_base}/{f}" if d2_base.startswith("s3://") else os.path.join(d2_base, f)
        try:
            df = read_parquet_df(file_path)
        except Exception:
            continue

        ch_c = _pick(list(df.columns), ("chainage",))
        ln_c = _pick(list(df.columns), ("lane",))
        if not ch_c or not ln_c:
            continue
        n_ok = pd.to_numeric(df[ch_c], errors="coerce").notna().sum()
        if n_ok:
            cands.append((score, n_ok, f, df))

    if not cands:
        return pd.DataFrame(), None

    score, n_ok, f, df = sorted(cands, key=lambda x: (-x[0], -x[1]))[0]
    cols = list(df.columns)
    c = {
        "chainage": _pick(cols, ("chainage",)),
        "from_km": _pick(cols, ("from_ch_km",), ("from", "km")),
        "from_m":  _pick(cols, ("from_ch_m",), ("from", "meter")),
        "lane":    _pick(cols, ("lane_no",), ("lane number",), ("lane",)),
        "area":    _pick(cols, ("area_sq_m",), ("area",)),
        "maxd":    _pick(cols, ("max", "depth")),
        "avgd":    _pick(cols, ("average", "depth"), ("ave", "depth"), ("avg", "depth")),
        "sev":     _pick(cols, ("severity",)),
        "type":    _pick(cols, ("type_of_defect",), ("type", "defect")),
        "lat":     _pick(cols, ("latitude",)),
        "lon":     _pick(cols, ("longitude",)),
        "img":     _pick(cols, ("image",)),
        "date":    _pick(cols, ("datetime",), ("date",)),
        "side":    _pick(cols, ("side",)),
    }

    out = df.copy()
    ch = pd.to_numeric(out[c["chainage"]], errors="coerce").astype(float)
    if c["from_km"] and c["from_m"]:
        alt = (pd.to_numeric(out[c["from_km"]], errors="coerce").astype(float) * 1000
               + pd.to_numeric(out[c["from_m"]], errors="coerce").astype(float))
        ch = ch.fillna(alt)
    if ch.dropna().lt(1000).all():
        ch = ch * 1000  # chainage was in km

    out["Start_Chainage"] = ch
    out["Lane"] = out[c["lane"]].map(normalize_lane) if c["lane"] else "Unknown"
    out = out.dropna(subset=["Start_Chainage", "Lane"]).reset_index(drop=True)
    out["Start_Chainage"] = out["Start_Chainage"].round().astype("int64")
    out["End_Chainage"] = out["Start_Chainage"]

    def _num(key: str):
        return pd.to_numeric(out[c[key]], errors="coerce").astype(float) if c[key] else np.nan

    def _txt(key: str):
        if not c[key]:
            return pd.Series([None] * len(out), dtype="object")
        return out[c[key]].astype("object").map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else None)

    out["Area_sq_m"] = _num("area")
    out["Max_Depth_mm"] = _num("maxd")
    out["Average_Depth_mm"] = _num("avgd")
    out["Severity"] = _txt("sev")
    out["Type_of_Defect"] = _txt("type").fillna("Potholes")
    out["Latitude"] = _num("lat")
    out["Longitude"] = _num("lon")
    out["Image_Name"] = _txt("img")
    out["Survey_Date"] = pd.to_datetime(out[c["date"]], errors="coerce") if c["date"] else pd.NaT
    side_from_lane = out["Lane"].map(lambda l: {"Increasing": "LHS", "Decreasing": "RHS"}.get(lane_to_direction(l)))
    out["Side"] = _txt("side").fillna(side_from_lane)
    out["Pothole_Source"] = "D2"
    logger.info(f"Loaded D2 pothole fallback: '{_sheet_token(f)}' ({len(out)} rows) from {f}")
    return out, f


def build_segment_index(df_seg: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Index 10m backup segments by lane for image name and coordinate matching."""
    seg: Dict[str, pd.DataFrame] = {}
    if df_seg is None or df_seg.empty or "Start_Chainage" not in df_seg.columns:
        return seg

    lane_col = "Source_Lane_Folder" if "Source_Lane_Folder" in df_seg.columns else "Lane"
    for ln, g in df_seg.groupby(df_seg[lane_col].map(normalize_lane), sort=False):
        if not ln:
            continue
        g = g.reset_index(drop=True)
        s = pd.to_numeric(g["Start_Chainage"], errors="coerce").astype(float)
        e = pd.to_numeric(g["End_Chainage"], errors="coerce").astype(float) if "End_Chainage" in g.columns else s + RAW_SEGMENT_M
        seg[ln] = pd.DataFrame({
            "lo": np.fmin(s, e),
            "hi": np.fmax(s, e),
            "pos": np.arange(len(g)),
            "img": g["Image_Name"].values if "Image_Name" in g.columns else None,
            "lat": pd.to_numeric(g["Start_Lat"], errors="coerce").values if "Start_Lat" in g.columns else np.nan,
            "lon": pd.to_numeric(g["Start_Lon"], errors="coerce").values if "Start_Lon" in g.columns else np.nan,
            "date": g["SurveyDate"].values if "SurveyDate" in g.columns else None,
        })
    return seg


def build_image_file_index(image_root: str) -> Dict[Tuple[str, int], Set[str]]:
    """Index (Lane, frame_number) -> set of image filenames under image_root."""
    idx: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
    if path_exists(image_root):
        all_imgs = walk_files(image_root, [".jpg", ".jpeg", ".png"])
        for full_p in all_imgs:
            f = os.path.basename(full_p)
            m = re.search(r"_([LR]\d+)_(\d+)_(\d+)\.(?:jpe?g|png)$", f, re.I)
            if m:
                idx[(m.group(1).upper(), int(m.group(3)))].add(f)
    return idx


def map_pothole_segments(
    df: pd.DataFrame,
    seg: Dict[str, pd.DataFrame],
    img_file_idx: Optional[Dict[Tuple[str, int], Set[str]]] = None,
) -> pd.DataFrame:
    """Map image name, coordinates, and survey date from 10m segments to pothole rows."""
    if img_file_idx is None:
        img_file_idx = defaultdict(set)
    rows = []
    for _, r in df.iterrows():
        res = {"img": None, "src": None, "lat": np.nan, "lon": np.nan, "date": None}
        lane_val = normalize_lane(r.get("Lane"))
        s = seg.get(lane_val)
        ch = pd.to_numeric(r.get("Start_Chainage"), errors="coerce")
        if s is not None and pd.notna(ch):
            hit = s[(s["lo"] <= ch) & (s["hi"] > ch)]
            if hit.empty:
                dist = np.minimum((s["lo"] - ch).abs(), (s["hi"] - ch).abs())
                hit = s[dist == dist.min()] if dist.min() <= RAW_SEGMENT_M else hit
            if not hit.empty:
                h = hit.iloc[0]
                res.update(lat=h["lat"], lon=h["lon"], date=h["date"])
                names = img_file_idx.get((lane_val, int(h["pos"])), set())
                if len(names) == 1:
                    res.update(img=next(iter(names)), src="segment_frame_image_file")
                elif pd.notna(h["img"]) and str(h["img"]).strip():
                    res.update(img=str(h["img"]).strip(), src="backup_cracking_segment")
                elif len(names) > 1:
                    res["src"] = "ambiguous_frame"
        rows.append(res)
    return pd.DataFrame(rows, index=df.index)


def process_potholes(
    df_pothole_raw: pd.DataFrame,
    df_cracking_raw: pd.DataFrame,
    d2_base: str,
    image_root: str,
    upc: str,
    zone: str,
    phase: str,
    cycle: str,
    d1_ctx: D1Context,
    tracker_ctx: Dict[str, Any],
    run_date: date,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Complete pothole processing pipeline with D2 fallback and schema creation.
    """
    target_columns = columns or list(BASE_DEFECT_COLUMNS)
    seg_index = build_segment_index(df_cracking_raw)
    img_file_idx = build_image_file_index(image_root)

    # Use D2 fallback if D4 has no pothole rows
    if df_pothole_raw.empty:
        df_pothole_d2, _ = load_d2_potholes(d2_base)
        if not df_pothole_d2.empty:
            m = map_pothole_segments(df_pothole_d2, seg_index, img_file_idx)
            no_img = df_pothole_d2["Image_Name"].isna()
            df_pothole_d2.loc[no_img, "Image_Name"] = m.loc[no_img, "img"]
            df_pothole_d2["Latitude"] = df_pothole_d2["Latitude"].fillna(m["lat"].astype(float))
            df_pothole_d2["Longitude"] = df_pothole_d2["Longitude"].fillna(m["lon"].astype(float))
            seg_date = pd.to_datetime(m["date"], errors="coerce")
            df_pothole_d2["Survey_Date"] = pd.to_datetime(df_pothole_d2["Survey_Date"], errors="coerce").fillna(seg_date)
            if df_pothole_d2["Survey_Date"].isna().all() and tracker_ctx.get("survey_end_date"):
                df_pothole_d2["Survey_Date"] = pd.Timestamp(tracker_ctx["survey_end_date"])
            df_pothole_raw = df_pothole_d2

    if df_pothole_raw.empty or "Start_Chainage" not in df_pothole_raw.columns:
        logger.info(f"No pothole records found for UPC {upc}.")
        return pd.DataFrame(columns=target_columns)

    if "Survey_Date" in df_pothole_raw.columns:
        sdt = pd.to_datetime(df_pothole_raw["Survey_Date"], errors="coerce").dropna()
        survey_date_str = sdt.iloc[0].strftime("%d%m%y") if len(sdt) > 0 else run_date.strftime("%d%m%y")
    else:
        survey_date_str = run_date.strftime("%d%m%y")

    parent_m = DEFECT_INTERVALS["pothole"]["parent_m"]
    upc_slash = upc.replace("_", "/")
    now_ts = pd.Timestamp.now()
    records = []

    for idx, row in df_pothole_raw.iterrows():
        start_ch = as_int(row.get("Start_Chainage"))
        end_ch = as_int(row.get("End_Chainage"))
        lane_val = normalize_lane(row.get("Lane"))
        road_ctx = d1_ctx.get_road_context(start_ch, end_ch, lane=lane_val)

        parent_start = (start_ch // parent_m) * parent_m if start_ch is not None else None
        parent_end = parent_start + parent_m if parent_start is not None else None
        defect_id_val = f"D/{survey_date_str}/N/{upc}/{start_ch if start_ch is not None else 0}/Pothole/{idx+1:04d}"
        record_id_val = f"pothole_{idx+1:06d}"
        parent_id_val = f"{upc}_{lane_val}_{parent_start}_{parent_end}" if lane_val and parent_start is not None else None

        survey_ts = pd.to_datetime(row.get("Survey_Date"), errors="coerce")
        survey_date_val = survey_ts.date() if pd.notnull(survey_ts) else tracker_ctx.get("survey_start_date")

        rec = {col: None for col in target_columns}
        rec.update({
            "id": f"{upc}_master_pothole_{idx+1:06d}",
            "collection_name": "master_pothole",
            "defect_base": "pothole",
            "level": "parent",
            "defect_id": defect_id_val,
            "display_defect_id": defect_id_val,
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
            "defect_type": "Pothole",
            "type_of_defect": as_text(row.get("Type_of_Defect", "Pothole")),
            "category": "POINT",
            "severity": as_text(row.get("Severity")),
            "pavement_type": first_non_null(road_ctx.get("pavement_type")),
            "pci": tracker_ctx.get("pci"),
            "lane": lane_val,
            "breach_flag": 1,  # Potholes always breach
            "metric_name": "area_sq_m",
            "metric_value": as_float(row.get("Area_sq_m")),
            "threshold_name": get_threshold_name(tracker_ctx, "pothole"),
            "threshold_value": 0.0,
            "total_children": 1,
            "length_m": max(end_ch - start_ch, 0) if start_ch is not None and end_ch is not None else None,
            "area_sqm": as_float(row.get("Area_sq_m")),
            "start_chainage": start_ch,
            "end_chainage": end_ch,
            "defect_chainage": start_ch,
            "start_lat": as_float(row.get("Latitude")),
            "start_long": as_float(row.get("Longitude")),
            "end_lat": as_float(row.get("Latitude")),
            "end_long": as_float(row.get("Longitude")),
            "defect_lat": as_float(row.get("Latitude")),
            "defect_long": as_float(row.get("Longitude")),
            "datetime": survey_ts if pd.notnull(survey_ts) else now_ts,
            "observed_at": survey_ts if pd.notnull(survey_ts) else now_ts,
            "survey_start_date": tracker_ctx.get("survey_start_date") or survey_date_val,
            "survey_end_date": tracker_ctx.get("survey_end_date") or survey_date_val,
            "image_name": as_text(row.get("Image_Name")),
            "image_urls": [as_text(row.get("Image_Name"))] if as_text(row.get("Image_Name")) else None,
            "is_duplicate": False,
            "nsv": json.dumps({"area_sqm": as_float(row.get("Area_sq_m")), "max_depth_mm": as_float(row.get("Max_Depth_mm"))}),
            "project_details": json.dumps({"project_name": upc, "zone": zone, "phase": phase, "cycle": cycle}),
            "created_at": now_ts,
            "updated_at": now_ts,
            "side": as_text(row.get("Side")) or ({"Increasing": "LHS", "Decreasing": "RHS"}.get(lane_to_direction(lane_val))),
            "carriageway": first_non_null(road_ctx.get("carriageway")),
            "direction": first_non_null(lane_to_direction(lane_val), road_ctx.get("direction")),
            "parent_start_chainage": parent_start,
            "parent_end_chainage": parent_end,
            "reporting_interval_m": get_reporting_interval(tracker_ctx, "pothole"),
            "segment_count": 1,
            "rating_ca": as_text(row.get("Severity")),
            "rating_irc": as_text(row.get("Severity")),
            "road_condition": "Bad",
            "defect_type_filter": "Pothole",
            "has_standard_defect": True,
            "standard_flags": "{}",
            "measures": json.dumps({"area_sqm": as_float(row.get("Area_sq_m"))}),
            "affected_length": as_float(row.get("Area_sq_m")) or 0.0,
            "total_length": as_float(row.get("Area_sq_m")) or 0.0,
            "affected_length_ca": as_float(row.get("Area_sq_m")) or 0.0,
            "affected_length_irc": as_float(row.get("Area_sq_m")) or 0.0,
            "run_date": run_date,
            "detected_on": tracker_ctx.get("survey_end_date") or survey_date_val,
        })
        records.append(rec)

    res_df = pd.DataFrame(records)
    for c in target_columns:
        if c not in res_df.columns:
            res_df[c] = None
    logger.info(f"Built master_pothole with {len(res_df)} records.")
    return res_df[target_columns]
