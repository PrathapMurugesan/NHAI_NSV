"""
Road condition assignment module.
Ported from Databricks DLM Extraction Pipeline (Cell 11).

Assigns road_condition dynamically:
  - CA standard: Simple binary "Good" (no breach) / "Bad" (breach)
  - IRC standard: D3 visual condition (Good/Fair/Poor from D3 deliverable)
  - Fallback: Simple binary based on breach_flag.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from src.common.constants import normalize_lane
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFECT_TYPE_MAP = {
    'cracking': 'Cracking',
    'roughness': 'Roughness',
    'rutting': 'Rutting',
    'pothole': 'Pothole',
    'ravelling': 'Ravelling',
}


def build_d3_combined_lookup(d3_conditions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine D3 condition tables across defects into a single lookup:
    columns: ['defect_type', 'bin_100m', 'Lane', 'Condition']
    """
    frames = []
    if not d3_conditions:
        return pd.DataFrame(columns=['defect_type', 'bin_100m', 'Lane', 'Condition'])

    for defect, df_d3 in d3_conditions.items():
        if df_d3 is None or df_d3.empty:
            continue
        tmp = df_d3.copy()
        tmp['defect_type'] = defect
        tmp['Lane'] = tmp['Lane'].apply(normalize_lane)
        tmp['bin_100m'] = pd.to_numeric(tmp['bin_100m'], errors='coerce').fillna(0).astype(int)
        frames.append(tmp[['defect_type', 'bin_100m', 'Lane', 'Condition']])

    if not frames:
        return pd.DataFrame(columns=['defect_type', 'bin_100m', 'Lane', 'Condition'])

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=['defect_type', 'bin_100m', 'Lane']
    )
    logger.info("Combined D3 condition lookup built: %d rows", len(combined))
    return combined


def apply_road_condition(
    output_dfs: Dict[str, pd.DataFrame],
    d3_conditions: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Add road_condition column to each defect DataFrame in output_dfs based on:
    CA -> Good / Bad
    IRC -> D3 condition (or None)
    """
    d3_lookup = build_d3_combined_lookup(d3_conditions or {})
    res = {}

    for tbl, df in output_dfs.items():
        if df is None or df.empty:
            res[tbl] = df
            continue

        defect_type = next((v for k, v in DEFECT_TYPE_MAP.items() if k in tbl.lower()), None)
        if defect_type is None:
            res[tbl] = df
            continue

        out = df.copy()
        lane_col = 'lane' if 'lane' in out.columns else 'Lane'
        ch_col = 'start_chainage' if 'start_chainage' in out.columns else 'Start_Chainage'

        out['_orig_idx'] = np.arange(len(out))
        out['_bin_100m'] = (pd.to_numeric(out[ch_col], errors='coerce').fillna(0) // 100 * 100).astype(int)
        out['_norm_lane'] = out[lane_col].apply(normalize_lane)

        if not d3_lookup.empty:
            d3_subset = d3_lookup[d3_lookup['defect_type'].str.lower() == defect_type.lower()][
                ['bin_100m', 'Lane', 'Condition']
            ].copy()
            d3_subset.rename(columns={'Lane': '_norm_lane', 'bin_100m': '_bin_100m', 'Condition': '_d3_condition'}, inplace=True)
            merged = out.merge(d3_subset, on=['_bin_100m', '_norm_lane'], how='left')
        else:
            merged = out.copy()
            merged['_d3_condition'] = None

        merged = merged.sort_values('_orig_idx').reset_index(drop=True)

        std_col = merged['threshold_name'].fillna('').astype(str).str.lower()
        if 'standard' in merged.columns:
            std_col = std_col.where(std_col != '', merged['standard'].fillna('').astype(str).str.lower())

        is_ca = std_col.str.contains('ca')
        is_irc = std_col.str.contains('irc')
        breach = pd.to_numeric(merged.get('breach_flag', 0), errors='coerce').fillna(0).astype(int) == 1

        ca_cond = np.where(breach, 'Bad', 'Good')
        fallback_cond = np.where(breach, 'Bad', 'Good')

        # IRC takes D3 condition, fallback to None
        irc_cond = merged['_d3_condition']

        road_cond = np.where(is_ca, ca_cond, np.where(is_irc, irc_cond, fallback_cond))

        merged['road_condition'] = road_cond
        merged.drop(columns=['_orig_idx', '_bin_100m', '_norm_lane', '_d3_condition'], inplace=True, errors='ignore')
        res[tbl] = merged
        logger.debug("Applied road_condition to %s (%d rows)", tbl, len(merged))

    return res
