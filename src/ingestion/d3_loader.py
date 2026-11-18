"""
Deliverable-3 (D3) visual condition loader and lookups.

Migrated from Notebook Cell 4 (Load All Defect Data Sources) & Cell 7.
Preserves all business logic:
- Maps D3 Parquet files for Cracking, Roughness, Rutting, Pothole
- Normalizes lane column variations (e.g. 'Lane\nNumber')
- Computes 100m bin key for chainage joining
- Filters out NaN / null conditions
- Drops duplicate (bin_100m, Lane) keys
- Provides lookup_d3_condition function
"""

import os
import pandas as pd
from typing import Dict, Optional
from src.common.logger import logger
from src.common.constants import normalize_lane
from src.common.fs_utils import path_exists, list_dir, read_parquet_df


D3_DEFECT_MAP = {
    "crack": "Cracking",
    "roughness": "Roughness",
    "rutting": "Rutting",
    "rut": "Rutting",
    "pothole": "Pothole",
    "potholes": "Pothole",
}


def load_d3_conditions(d3_base: str) -> Dict[str, pd.DataFrame]:
    """
    Load D3 visual condition Parquets and index them by 100m bin and lane.
    """
    conditions: Dict[str, pd.DataFrame] = {}
    if not d3_base or not path_exists(d3_base):
        logger.warning(f"D3 base path not found: {d3_base}")
        return conditions

    for f in sorted(list_dir(d3_base)):
        if not f.endswith(".parquet"):
            continue
        fl = f.lower()
        defect_key = next((v for k, v in D3_DEFECT_MAP.items() if k in fl), None)
        if not defect_key:
            continue

        file_path = f"{d3_base}/{f}" if d3_base.startswith("s3://") else os.path.join(d3_base, f)
        try:
            df_d3 = read_parquet_df(file_path)
        except Exception as e:
            logger.warning(f"Could not read D3 file {file_path}: {e}")
            continue

        if "Condition" in df_d3.columns and "Start_Chainage" in df_d3.columns:
            df_d3["Start_Chainage"] = pd.to_numeric(df_d3["Start_Chainage"], errors="coerce")
            df_d3 = df_d3.dropna(subset=["Start_Chainage"])

            lane_col = next((c for c in df_d3.columns if "lane" in c.lower().replace("\n", " ")), None)
            if lane_col and lane_col != "Lane":
                df_d3["Lane"] = df_d3[lane_col].map(normalize_lane)
            elif "Lane" not in df_d3.columns:
                df_d3["Lane"] = "Unknown"
            else:
                df_d3["Lane"] = df_d3["Lane"].map(normalize_lane)

            df_d3["bin_100m"] = (df_d3["Start_Chainage"] // 100 * 100).astype(int)
            df_d3 = df_d3[df_d3["Condition"].notna() & (df_d3["Condition"].astype(str) != "nan")]

            conditions[defect_key] = df_d3[["bin_100m", "Lane", "Condition"]].drop_duplicates(
                subset=["bin_100m", "Lane"]
            ).copy()

    logger.info(f"Loaded D3 visual conditions for: {list(conditions.keys())}")
    return conditions


def build_condition_lookup(pdf: pd.DataFrame) -> pd.DataFrame:
    """Build condition lookup dataframe from backup raw df."""
    if pdf.empty or "Start_Chainage" not in pdf.columns or "Condition" not in pdf.columns:
        return pd.DataFrame(columns=["start_chainage", "end_chainage", "lane", "condition"])

    lane_col = "Lane" if "Lane" in pdf.columns else ("LANE" if "LANE" in pdf.columns else None)
    end_col = "End_Chainage" if "End_Chainage" in pdf.columns else "Start_Chainage"

    return pd.DataFrame({
        "start_chainage": pd.to_numeric(pdf["Start_Chainage"], errors="coerce"),
        "end_chainage": pd.to_numeric(pdf[end_col], errors="coerce"),
        "lane": pdf[lane_col] if lane_col else None,
        "condition": pdf["Condition"],
    }).dropna(subset=["start_chainage"])


def lookup_d3_condition(
    d3_conditions: Dict[str, pd.DataFrame],
    defect_name: str,
    start_ch: Optional[float],
    end_ch: Optional[float] = None,
    lane: Optional[str] = None,
) -> Optional[str]:
    """Lookup condition string for a defect from D3 conditions map."""
    defect_title = defect_name.title()
    df = d3_conditions.get(defect_title)
    if df is None or df.empty or start_ch is None:
        return None

    bin_val = int(start_ch // 100 * 100)
    matches = df[df["bin_100m"] == bin_val]
    if lane and not matches.empty:
        lane_n = normalize_lane(lane)
        lane_matches = matches[matches["Lane"] == lane_n]
        if not lane_matches.empty:
            matches = lane_matches

    if not matches.empty:
        val = matches.iloc[0]["Condition"]
        return str(val) if pd.notna(val) else None
    return None
