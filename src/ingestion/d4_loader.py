"""
Deliverable-4 (D4) pothole loader and chainage normalizer.

Migrated from Notebook Cell 4 (Load All Defect Data Sources).
Preserves all business logic:
- Filters D4 Parquets containing 'pothole' excluding other defect types
- Checks for minimum defect columns (severity, area, type, lane)
- Computes Start_Chainage from from_ch_km * 1000 + from_ch_m
- Handles bug in raw data where from_ch_m > 999 had full chainage
- Computes End_Chainage from to_ch_km and to_ch_m with fallback to Start_Chainage
- Normalizes Lane (e.g. 'R-1' -> 'R1')
- Renames columns to standard schema (Latitude, Longitude, Image_Name, Severity, etc.)
"""

import os
import pandas as pd
from typing import List, Optional
from src.common.logger import logger
from src.common.constants import normalize_lane
from src.common.fs_utils import path_exists, list_dir, read_parquet_df


_EXCLUDE_KW = ["roughness", "rut,", "crack", "raveling", "ravelling"]
_MIN_DEFECT_COLS = {"severity", "area_sq_m", "area", "type_of_defect", "lane_no"}


def _d4_pick(cols: List[str], *keywords: str) -> Optional[str]:
    for col in cols:
        cl = col.lower()
        if all(kw.lower() in cl for kw in keywords):
            return col
    return None


def load_d4_potholes(d4_base: str) -> pd.DataFrame:
    """
    Load D4 pothole parquets and normalize chainages and column names.
    Returns empty DataFrame if D4 pothole files are missing or lack defect columns.
    """
    if not d4_base or not path_exists(d4_base):
        logger.warning(f"D4 base path not found: {d4_base}")
        return pd.DataFrame()

    d4_files = [
        f for f in sorted(list_dir(d4_base))
        if f.endswith(".parquet") and "pothole" in f.lower()
        and not any(kw in f.lower() for kw in _EXCLUDE_KW)
    ]

    if not d4_files:
        logger.warning(f"No D4 pothole parquet files found in {d4_base}")
        return pd.DataFrame()

    child_frames = []
    for pf in d4_files:
        file_path = f"{d4_base}/{pf}" if d4_base.startswith("s3://") else os.path.join(d4_base, pf)
        try:
            df_part = read_parquet_df(file_path)
        except Exception as e:
            logger.warning(f"Could not read D4 pothole file {file_path}: {e}")
            continue

        cols_lower = {c.lower() for c in df_part.columns}
        if not (cols_lower & _MIN_DEFECT_COLS):
            continue
        child_frames.append(df_part)
        logger.info(f"Loaded D4 potholes from {pf} ({len(df_part)} rows)")

    if not child_frames:
        logger.warning("D4 pothole files found, but none contained defect columns.")
        return pd.DataFrame()

    df_pothole_raw = pd.concat(child_frames, ignore_index=True)
    d4cols = list(df_pothole_raw.columns)

    from_km_c = _d4_pick(d4cols, "from_ch_km") or _d4_pick(d4cols, "from", "km")
    from_m_c  = _d4_pick(d4cols, "from_ch_m")  or _d4_pick(d4cols, "from", "meter")
    to_km_c   = _d4_pick(d4cols, "to_ch_km")   or _d4_pick(d4cols, "to", "km")
    to_m_c    = _d4_pick(d4cols, "to_ch_m")    or _d4_pick(d4cols, "to", "meter")
    chain_c   = _d4_pick(d4cols, "chainage")
    lane_c    = _d4_pick(d4cols, "lane_no") or _d4_pick(d4cols, "lane no") or _d4_pick(d4cols, "lane")
    lat_c     = _d4_pick(d4cols, "latitude")
    lon_c     = _d4_pick(d4cols, "longitude")
    img_c     = _d4_pick(d4cols, "image_name") or _d4_pick(d4cols, "image name")
    sev_c     = _d4_pick(d4cols, "severity")
    deftype_c = _d4_pick(d4cols, "type_of_defect") or _d4_pick(d4cols, "type", "defect")
    area_c    = _d4_pick(d4cols, "area_sq_m") or _d4_pick(d4cols, "area", "sq")
    maxd_c    = _d4_pick(d4cols, "max_depth_mm") or _d4_pick(d4cols, "max", "depth")
    avgd_c    = _d4_pick(d4cols, "average_depth_mm") or _d4_pick(d4cols, "average", "depth")
    dt_c      = _d4_pick(d4cols, "datetime") or _d4_pick(d4cols, "date")
    side_c    = _d4_pick(d4cols, "side")

    # Compute Start_Chainage
    if from_km_c and from_m_c:
        km_vals = pd.to_numeric(df_pothole_raw[from_km_c], errors="coerce").fillna(0).astype(int)
        m_vals = pd.to_numeric(df_pothole_raw[from_m_c], errors="coerce").fillna(0).astype(int)
        computed_ch = km_vals * 1000 + m_vals
        bad_m = m_vals > 999
        if bad_m.any() and chain_c:
            ch_direct = pd.to_numeric(df_pothole_raw[chain_c], errors="coerce").fillna(0).astype(int)
            computed_ch[bad_m] = ch_direct[bad_m]
        df_pothole_raw["Start_Chainage"] = computed_ch
    elif chain_c:
        df_pothole_raw["Start_Chainage"] = pd.to_numeric(df_pothole_raw[chain_c], errors="coerce").fillna(0)
        df_pothole_raw.loc[df_pothole_raw["Start_Chainage"] < 1000, "Start_Chainage"] *= 1000
    else:
        df_pothole_raw["Start_Chainage"] = 0

    # Compute End_Chainage
    if to_km_c and to_m_c:
        to_km_vals = pd.to_numeric(df_pothole_raw[to_km_c], errors="coerce")
        to_m_vals = pd.to_numeric(df_pothole_raw[to_m_c], errors="coerce")
        to_computed = to_km_vals.fillna(0).astype(int) * 1000 + to_m_vals.fillna(0).astype(int)
        to_bad = to_m_vals.fillna(0).astype(int) > 999
        if to_bad.any() and chain_c:
            to_computed[to_bad] = pd.to_numeric(df_pothole_raw.loc[to_bad, chain_c], errors="coerce").fillna(0).astype(int)
        df_pothole_raw["End_Chainage"] = to_computed
        both_null = to_km_vals.isna() & to_m_vals.isna()
        df_pothole_raw.loc[both_null, "End_Chainage"] = df_pothole_raw.loc[both_null, "Start_Chainage"]
    else:
        df_pothole_raw["End_Chainage"] = df_pothole_raw["Start_Chainage"]

    df_pothole_raw["Lane"] = df_pothole_raw[lane_c].map(normalize_lane).fillna("Unknown") if lane_c else "Unknown"

    remap = {}
    if lat_c: remap[lat_c] = "Latitude"
    if lon_c: remap[lon_c] = "Longitude"
    if img_c: remap[img_c] = "Image_Name"
    if sev_c: remap[sev_c] = "Severity"
    if deftype_c: remap[deftype_c] = "Type_of_Defect"
    if area_c: remap[area_c] = "Area_sq_m"
    if maxd_c: remap[maxd_c] = "Max_Depth_mm"
    if avgd_c: remap[avgd_c] = "Average_Depth_mm"
    if dt_c: remap[dt_c] = "Survey_Date"
    if side_c: remap[side_c] = "Side"

    df_pothole_raw.rename(columns=remap, inplace=True)
    return df_pothole_raw
