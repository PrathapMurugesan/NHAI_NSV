"""
Master chainage builder module.
Ported from Databricks DLM Extraction Pipeline (Cell 17).

Builds master_chainage table with:
  - 1 row per 10m segment of the UPC
  - Per-lane start/end chainage and GPS coordinates
  - LHS and RHS aggregated start/end chainage, GPS, pavement type, and active lanes
  - Section metadata (nh_number, section_code, carriageway, upc_start_chainage, upc_end_chainage)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.common.constants import (
    CHILD_10M,
    DEFECTS_LIN,
    SRC_10M,
    normalize_lane,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def _lane_sort_key(lane: str):
    l = str(lane).strip().upper()
    prefix = l[0] if l else 'Z'
    num_str = l[1:] if len(l) > 1 else ''
    num = int(num_str) if num_str.isdigit() else 99
    return (prefix, num)


def build_master_chainage(
    output_dfs: Dict[str, pd.DataFrame],
    upc: str,
    upc_slash: str,
    zone: str,
    phase: str,
    cycle: int,
    child_10m: int = CHILD_10M,
) -> pd.DataFrame:
    """
    Build master_chainage table from the 10m CA rows across all linear defects.
    """
    frames = []
    gps_cols = ['start_lat', 'start_long', 'end_lat', 'end_long']
    txt_cols = ['pavement_type', 'nh_number', 'section_code', 'carriageway']

    for d in DEFECTS_LIN:
        src_tbl = SRC_10M.get(d)
        if src_tbl not in output_dfs or output_dfs[src_tbl] is None or output_dfs[src_tbl].empty:
            continue
        df = output_dfs[src_tbl]
        if 'threshold_name' in df.columns:
            df = df[df['threshold_name'] == 'As per CA']
        if df.empty:
            continue

        cols_needed = ['lane', 'start_chainage', 'end_chainage'] + [c for c in gps_cols + txt_cols if c in df.columns]
        sub = df[cols_needed].copy()
        s_col = pd.to_numeric(sub['start_chainage'], errors='coerce')
        e_col = pd.to_numeric(sub['end_chainage'], errors='coerce')
        sub['s'] = np.minimum(s_col, e_col)
        sub['e'] = np.maximum(s_col, e_col)

        for c in gps_cols:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
            else:
                sub[c] = np.nan

        for c in txt_cols:
            if c not in sub.columns:
                sub[c] = None

        frames.append(sub[['lane', 's', 'e'] + gps_cols + txt_cols])

    if not frames:
        logger.warning("No defect 10m data available to construct master_chainage")
        return pd.DataFrame()

    all_10m = pd.concat(frames, ignore_index=True)
    all_10m = all_10m.dropna(subset=['s', 'e'])
    all_10m['chainage_start'] = (np.floor(all_10m['s'] / child_10m) * child_10m).astype(int)
    all_10m['lane'] = all_10m['lane'].apply(normalize_lane)
    all_10m = all_10m.dropna(subset=['lane'])

    # Group by lane and chainage_start
    agg_dict = {
        's': 'min',
        'e': 'max',
    }
    for c in gps_cols + txt_cols:
        agg_dict[c] = 'first'

    lb = all_10m.groupby(['lane', 'chainage_start']).agg(agg_dict).reset_index()

    lanes_present = sorted(lb['lane'].unique(), key=_lane_sort_key)
    logger.info("Master chainage lanes present: %s", lanes_present)

    mc = pd.DataFrame({'chainage_start': sorted(lb['chainage_start'].unique())})

    for l in lanes_present:
        p = l.lower()
        sub_l = lb[lb['lane'] == l][['chainage_start', 's', 'e'] + gps_cols].copy()
        sub_l.rename(columns={
            's': f'{p}_start_chainage',
            'e': f'{p}_end_chainage',
            'start_lat': f'{p}_start_lat',
            'start_long': f'{p}_start_long',
            'end_lat': f'{p}_end_lat',
            'end_long': f'{p}_end_long',
        }, inplace=True)
        mc = mc.merge(sub_l, on='chainage_start', how='left')

    # Aggregations for LHS and RHS
    for side, pref in [('lhs', 'L'), ('rhs', 'R')]:
        g = lb[lb['lane'].str.upper().str.startswith(pref)]
        if g.empty:
            continue

        # Lowest start chainage row per 10m
        idx_min = g.groupby('chainage_start')['s'].idxmin()
        st = g.loc[idx_min, ['chainage_start', 's', 'start_lat', 'start_long', 'pavement_type']].copy()
        st.rename(columns={
            's': f'{side}_start_chainage',
            'start_lat': f'{side}_start_lat',
            'start_long': f'{side}_start_long',
            'pavement_type': f'{side}_pavement_type',
        }, inplace=True)

        # Highest end chainage row per 10m
        idx_max = g.groupby('chainage_start')['e'].idxmax()
        en = g.loc[idx_max, ['chainage_start', 'e', 'end_lat', 'end_long']].copy()
        en.rename(columns={
            'e': f'{side}_end_chainage',
            'end_lat': f'{side}_end_lat',
            'end_long': f'{side}_end_long',
        }, inplace=True)

        # Comma-separated lanes present in this side
        ln = g.groupby('chainage_start')['lane'].apply(
            lambda v: ','.join(sorted(v.unique(), key=_lane_sort_key))
        ).reset_index()
        ln.rename(columns={'lane': f'{side}_lanes'}, inplace=True)

        mc = mc.merge(st, on='chainage_start', how='left')
        mc = mc.merge(en, on='chainage_start', how='left')
        mc = mc.merge(ln, on='chainage_start', how='left')

    # Meta columns
    meta = lb.sort_values(['chainage_start', 'lane']).groupby('chainage_start')[
        ['nh_number', 'section_code', 'carriageway']
    ].first().reset_index()
    mc = mc.merge(meta, on='chainage_start', how='left')

    mc['chainage_end'] = mc['chainage_start'] + child_10m

    # Active lanes in segment
    lane_start_cols = [f'{l.lower()}_start_chainage' for l in lanes_present if f'{l.lower()}_start_chainage' in mc.columns]
    if lane_start_cols:
        mc['lanes_present'] = mc[lane_start_cols].notna().apply(
            lambda r: ','.join(l for l, ok in zip(lanes_present, r) if ok), axis=1
        )
    else:
        mc['lanes_present'] = ''

    mc['upc_start_chainage'] = float(lb['s'].min())
    mc['upc_end_chainage'] = float(lb['e'].max())
    mc['upc'] = upc_slash
    mc['upc_code'] = upc
    mc['zone'] = zone
    mc['phase'] = phase
    mc['cycle'] = int(cycle)
    mc['created_at'] = datetime.now(timezone.utc)

    front_cols = [
        'upc', 'upc_code', 'zone', 'phase', 'cycle',
        'chainage_start', 'chainage_end', 'upc_start_chainage', 'upc_end_chainage',
        'lanes_present', 'nh_number', 'section_code', 'carriageway'
    ]
    side_cols = [c for c in mc.columns if c.startswith(('lhs_', 'rhs_'))]
    lane_cols = [c for c in mc.columns if c not in front_cols + side_cols + ['created_at']]

    ordered_cols = front_cols + side_cols + lane_cols + ['created_at']
    mc = mc[[c for c in ordered_cols if c in mc.columns]]

    logger.info("Master chainage built: %d segments (chainage %.0f to %.0f)",
                len(mc), mc['upc_start_chainage'].iloc[0], mc['upc_end_chainage'].iloc[0])
    return mc
