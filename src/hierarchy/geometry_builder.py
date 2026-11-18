"""
Geometry builder module for DLM pipeline.
Generates:
  - geometry_parent (22 columns): UPC-level summary of survey geometry, length, and bounds
  - geometry_child (13 columns): Lane-level geometry segments with GPS coordinates
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import pandas as pd
import numpy as np

from src.common.constants import (
    normalize_lane,
    lane_to_direction,
    standardize_nh_number,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def build_geometry_tables(
    master_chainage_df: pd.DataFrame,
    upc: str,
    project_id: Optional[str] = None,
    nh_number: Optional[str] = None,
    section_code: Optional[str] = None,
    survey_start_date: Any = None,
    survey_end_date: Any = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build geometry_parent and geometry_child DataFrames from master_chainage.
    """
    if master_chainage_df is None or master_chainage_df.empty:
        logger.warning("master_chainage is empty; producing empty geometry tables")
        geom_p = pd.DataFrame(columns=[
            "id", "layer_id", "upc", "project_id", "nh_number", "section_code",
            "source_system", "defect_type", "lanes", "start_chainage", "end_chainage",
            "start_lat", "start_long", "end_lat", "end_long", "total_records",
            "survey_start_date", "survey_end_date", "lane_length_surveyed_km",
            "accepted_lane_length_km", "d2_status", "project_details"
        ])
        geom_c = pd.DataFrame(columns=[
            "id", "child_id", "parent_id", "lane", "nh_number", "section_code",
            "direction", "start_chainage", "end_chainage", "start_latitude",
            "start_longitude", "end_latitude", "end_longitude"
        ])
        return geom_p, geom_c

    mc = master_chainage_df.copy()
    parent_id = f"{upc}_geometry_parent"

    # Identify lanes present in master_chainage
    lane_cols = [c for c in mc.columns if c.endswith("_start_chainage") and not c.startswith(("lhs_", "rhs_", "upc_"))]
    lanes = [c.replace("_start_chainage", "").upper() for c in lane_cols]

    nh = standardize_nh_number(nh_number) if nh_number else (
        standardize_nh_number(mc["nh_number"].dropna().iloc[0]) if "nh_number" in mc and mc["nh_number"].notna().any() else "NH-Unknown"
    )
    sec = section_code or (mc["section_code"].dropna().iloc[0] if "section_code" in mc and mc["section_code"].notna().any() else "")

    child_rows = []
    total_lane_length_m = 0.0

    for lane in lanes:
        p = lane.lower()
        st_ch_col = f"{p}_start_chainage"
        en_ch_col = f"{p}_end_chainage"
        st_lat_col = f"{p}_start_lat"
        st_lng_col = f"{p}_start_long"
        en_lat_col = f"{p}_end_lat"
        en_lng_col = f"{p}_end_long"

        if st_ch_col not in mc.columns:
            continue

        lane_df = mc[mc[st_ch_col].notna()].copy()
        if lane_df.empty:
            continue

        direction = lane_to_direction(lane) or ("Increasing" if lane.startswith("L") else "Decreasing")

        for _, r in lane_df.iterrows():
            s_ch = float(r[st_ch_col])
            e_ch = float(r[en_ch_col]) if en_ch_col in r and pd.notna(r[en_ch_col]) else s_ch + 10.0
            seg_len = abs(e_ch - s_ch)
            total_lane_length_m += seg_len

            cid = f"{upc}_{lane}_{int(s_ch)}_{int(e_ch)}"
            child_rows.append({
                "id": cid,
                "child_id": cid,
                "parent_id": parent_id,
                "lane": lane,
                "nh_number": nh,
                "section_code": sec,
                "direction": direction,
                "start_chainage": s_ch,
                "end_chainage": e_ch,
                "start_latitude": float(r[st_lat_col]) if st_lat_col in r and pd.notna(r[st_lat_col]) else None,
                "start_longitude": float(r[st_lng_col]) if st_lng_col in r and pd.notna(r[st_lng_col]) else None,
                "end_latitude": float(r[en_lat_col]) if en_lat_col in r and pd.notna(r[en_lat_col]) else None,
                "end_longitude": float(r[en_lng_col]) if en_lng_col in r and pd.notna(r[en_lng_col]) else None,
            })

    geom_c = pd.DataFrame(child_rows)

    # Global min/max
    min_ch = float(mc["upc_start_chainage"].iloc[0]) if "upc_start_chainage" in mc else (geom_c["start_chainage"].min() if not geom_c.empty else 0.0)
    max_ch = float(mc["upc_end_chainage"].iloc[0]) if "upc_end_chainage" in mc else (geom_c["end_chainage"].max() if not geom_c.empty else 0.0)

    # First and last coordinates
    st_lat = geom_c["start_latitude"].dropna().iloc[0] if not geom_c.empty and geom_c["start_latitude"].notna().any() else None
    st_lng = geom_c["start_longitude"].dropna().iloc[0] if not geom_c.empty and geom_c["start_longitude"].notna().any() else None
    en_lat = geom_c["end_latitude"].dropna().iloc[-1] if not geom_c.empty and geom_c["end_latitude"].notna().any() else None
    en_lng = geom_c["end_longitude"].dropna().iloc[-1] if not geom_c.empty and geom_c["end_longitude"].notna().any() else None

    lane_km = round(total_lane_length_m / 1000.0, 3)

    geom_p = pd.DataFrame([{
        "id": parent_id,
        "layer_id": "dlm",
        "upc": upc,
        "project_id": project_id or upc,
        "nh_number": nh,
        "section_code": sec,
        "source_system": "NSV",
        "defect_type": "All",
        "lanes": ",".join(sorted(lanes)),
        "start_chainage": min_ch,
        "end_chainage": max_ch,
        "start_lat": st_lat,
        "start_long": st_lng,
        "end_lat": en_lat,
        "end_long": en_lng,
        "total_records": len(geom_c),
        "survey_start_date": survey_start_date,
        "survey_end_date": survey_end_date,
        "lane_length_surveyed_km": lane_km,
        "accepted_lane_length_km": lane_km,
        "d2_status": "Accepted",
        "project_details": "{}",
    }])

    logger.info("Built geometry_parent (1 row, %.2f km) and geometry_child (%d rows)", lane_km, len(geom_c))
    return geom_p, geom_c
