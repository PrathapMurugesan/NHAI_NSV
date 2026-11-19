"""
Lane-wise 1000m table builder (master_{defect}_1000m_lane).

The side parents (LHS / RHS) pool all lanes of a side into one record per km. These tables keep the lanes
apart: one record per lane (L1, L2, ... R1, R2, ...) per 1000m, for roughness, rutting, cracking and
ravelling (no pothole). The rows come from the lane-wise 1000m aggregate that is already built for each
defect (CA values, as for the side parents); here they get their own identity and the IRC columns.

  - id / parent_id / record_id : {upc}_{lane}_{start}_{end}   (one record per lane per 1000m)
  - breach_flag / affected_length_ca : as in the lane's 1000m aggregate (CA threshold)
  - affected_length_irc / rating_irc : lane value >= IRC limit on the same record (no IRC row, as for the
    side parents); rating_irc is set for roughness and rutting only
"""

from typing import Optional
import numpy as np
import pandas as pd

from src.common.constants import (
    PARENT_LENGTH_M,
    lane_1000m_table,
    lane_to_direction,
    normalize_lane,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def build_lane_1000m(
    defect: str,
    src_df: Optional[pd.DataFrame],
    upc: str,
    irc_threshold: Optional[float] = None,
    parent_len_m: int = PARENT_LENGTH_M,
) -> pd.DataFrame:
    """Lane-wise 1000m table of a defect from its lane-wise 1000m aggregate (empty frame when there is none)."""
    table = lane_1000m_table(defect)
    if table is None or src_df is None or src_df.empty:
        return pd.DataFrame()

    df = src_df.copy()
    if 'threshold_name' in df.columns:
        df = df[df['threshold_name'] == 'As per CA'].copy()
    if df.empty:
        return pd.DataFrame()

    L = int(parent_len_m)
    df['lane'] = df['lane'].apply(normalize_lane)
    s = pd.to_numeric(df['start_chainage'], errors='coerce')
    e = pd.to_numeric(df['end_chainage'], errors='coerce')
    df['_lo'] = np.minimum(s, e)
    df['_hi'] = np.maximum(s, e)
    df = df.dropna(subset=['lane', '_lo']).copy()
    if df.empty:
        return pd.DataFrame()

    b = (np.floor(df['_lo'] / L) * L).astype(int)
    pid = pd.Series([f"{upc}_{ln}_{x}_{x + L}" for ln, x in zip(df['lane'], b)], index=df.index)
    mv = pd.to_numeric(df.get('metric_value'), errors='coerce')
    length = pd.to_numeric(df.get('length_m'), errors='coerce')
    length = length.fillna(df['_hi'] - df['_lo'])
    bf = pd.to_numeric(df.get('breach_flag'), errors='coerce').fillna(0).astype(int)

    df['id'] = [f"{upc}_{table}_{ln}_{x}" for ln, x in zip(df['lane'], b)]
    df['collection_name'] = table
    df['level'] = 'lane_1000m'
    df['standard'] = 'CA'
    df['parent_id'] = pid
    df['record_id'] = pid
    df['source_defect_id'] = pid
    df['defect_id'] = np.where(bf == 1, pid, None)
    df['display_defect_id'] = None                         # numbered with the main tables only
    df['side'] = np.where(df['lane'].str.startswith('L'), 'LHS', np.where(df['lane'].str.startswith('R'), 'RHS', None))
    df['direction'] = df['lane'].apply(lane_to_direction)
    df['start_chainage'] = df['_lo'].astype(int)
    df['end_chainage'] = df['_hi'].astype(int)
    df['length_m'] = length
    df['parent_start_chainage'] = b
    df['parent_end_chainage'] = b + L
    df['reporting_interval_m'] = L
    df['breach_flag'] = bf
    df['affected_length_ca'] = np.where(bf == 1, length, 0.0)
    df['affected_length'] = df['affected_length_ca']

    if irc_threshold is not None:
        bf_irc = (mv >= float(irc_threshold)).fillna(False)
        df['affected_length_irc'] = np.where(bf_irc, length, 0.0)
        if defect in ('roughness', 'rutting'):
            df['rating_irc'] = np.where(mv.isna(), None, np.where(bf_irc, 'Poor', 'Good'))

    df = df.drop(columns=['_lo', '_hi'])
    df = df.sort_values(['lane', 'start_chainage'], kind='stable').reset_index(drop=True)
    logger.info("Built lane-wise 1000m table %s: %d rows (%d lanes)", table, len(df), df['lane'].nunique())
    return df
