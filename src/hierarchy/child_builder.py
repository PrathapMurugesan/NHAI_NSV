"""
Child builder module for DLM hierarchy.
Ported from Databricks DLM Extraction Pipeline (Cell 14).

Constructs:
  - child_ca: lane-wise records at survey's CA reporting interval
  - child_10m: lane-wise 10m records (CA and IRC rows)
  - child_irc100: lane-wise 100m records as per IRC standard
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.common.constants import (
    CHILD_10M,
    CHILD_100M,
    PARENT_LENGTH_M,
    IRC_TV,
    TABLE_SPEC,
    has_parent,
    parent_len,
    ca_iv,
    SRC_10M,
    SRC_100,
    SRC_1KM,
    normalize_lane,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def compute_link_cols(
    df: pd.DataFrame,
    defect: str,
    upc: str,
    ca_interval_m: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series, int]:
    """
    Compute parent_id, parent_start_chainage, and parent_interval_length.
    If defect has parent (roughness, rutting): LHS/RHS parent at parent_len (1000m).
    Else (cracking, ravelling): lane-wise parent at CA reporting interval.
    """
    s_col = pd.to_numeric(df['start_chainage'], errors='coerce')
    e_col = pd.to_numeric(df['end_chainage'], errors='coerce')
    lo = np.minimum(s_col, e_col)

    if has_parent(defect):
        L = parent_len(defect) or PARENT_LENGTH_M
        b = (np.floor(lo / L) * L).astype(int)
        norm_lane = df['lane'].apply(normalize_lane)
        side = np.where(
            norm_lane.str.upper().str.startswith('L'),
            'LHS',
            np.where(norm_lane.str.upper().str.startswith('R'), 'RHS', '')
        )
        pid = upc + '_' + side + '_' + b.astype(str) + '_' + (b + L).astype(str)
        return pd.Series(pid, index=df.index), b, L
    else:
        L = ca_iv(defect, ca_interval_m)
        b = (np.floor(lo / L) * L).astype(int)
        norm_lane = df['lane'].apply(normalize_lane)
        pid = upc + '_' + norm_lane + '_' + b.astype(str) + '_' + (b + L).astype(str)
        return pd.Series(pid, index=df.index), b, L


def as_child(
    df: pd.DataFrame,
    table: str,
    level: str,
    defect: str,
    upc: str,
    ca_interval_m: Optional[int] = None,
) -> pd.DataFrame:
    """
    Format DataFrame as child: set collection_name, level, side, parent_id,
    parent_start_chainage, parent_end_chainage, and reset display_defect_id to None.
    """
    out = df.copy()
    pid, b, L = compute_link_cols(out, defect, upc, ca_interval_m)

    norm_lane = out['lane'].apply(normalize_lane)
    side = np.where(
        norm_lane.str.upper().str.startswith('L'),
        'LHS',
        np.where(norm_lane.str.upper().str.startswith('R'), 'RHS', None)
    )

    out['collection_name'] = table
    out['level'] = level
    out['side'] = side
    out['parent_id'] = pid
    out['parent_start_chainage'] = b
    out['parent_end_chainage'] = b + L
    out['display_defect_id'] = None
    return out


def build_child_ca(
    defect: str,
    table: str,
    base_df: pd.DataFrame,
    src_table_name: str,
    upc: str,
    ca_interval_m: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build CA child table at survey CA reporting interval.
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame()

    out = base_df.copy()
    if 'threshold_name' in out.columns:
        out = out[out['threshold_name'] == 'As per CA']

    iv = ca_iv(defect, ca_interval_m)
    out['reporting_interval_m'] = int(iv)

    level = 'child_as_per_ca' if has_parent(defect) else ('as_per_ca' if defect == 'cracking' else 'master')
    child_df = as_child(out, table, level, defect, upc, ca_interval_m)

    if 'id' in child_df.columns:
        child_df['id'] = child_df['id'].str.replace(src_table_name, table, regex=False)

    logger.info("Built child CA table %s: %d rows", table, len(child_df))
    return child_df


def build_child_10m(
    defect: str,
    table: str,
    src_10m_df: pd.DataFrame,
    src_table_name: str,
    upc: str,
    ca_interval_m: Optional[int] = None,
    irc_tv: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build child 10m table containing both 'As per CA' and 'As per IRC' rows.
    """
    if src_10m_df is None or src_10m_df.empty:
        return pd.DataFrame()

    df = src_10m_df.copy()
    tv = float(irc_tv if irc_tv is not None else IRC_TV.get(defect, 0.0))

    ca_rows = df[df['threshold_name'] == 'As per CA'].copy()
    irc_rows = df[df['threshold_name'] == 'As per IRC'].copy()

    # Recompute IRC threshold breach and metrics
    if not irc_rows.empty:
        mv = pd.to_numeric(irc_rows['metric_value'], errors='coerce').fillna(0.0)
        irc_rows['threshold_value'] = tv
        bf = (mv >= tv).astype(int)
        irc_rows['breach_flag'] = bf

        length = pd.to_numeric(
            irc_rows.get('length_m', irc_rows.get('total_length', 10.0)),
            errors='coerce'
        ).fillna(10.0)

        aff_len = np.where(bf == 1, length, 0.0)
        if 'affected_length' in irc_rows.columns:
            irc_rows['affected_length'] = aff_len
        if 'affected_length_irc' in irc_rows.columns:
            irc_rows['affected_length_irc'] = aff_len
        if 'affected_length_ca' in irc_rows.columns:
            irc_rows['affected_length_ca'] = 0.0

        if 'road_condition' in irc_rows.columns:
            default_cond = pd.Series(np.where(bf == 1, 'Poor', 'Good'), index=irc_rows.index)
            irc_rows['road_condition'] = irc_rows.get('rating_irc', default_cond).fillna(default_cond)

        if 'defect_id' in irc_rows.columns:
            rec_id = irc_rows.get('record_id', irc_rows.get('id'))
            irc_rows['defect_id'] = np.where(bf == 1, irc_rows['defect_id'].fillna(rec_id), None)

    combined = pd.concat([ca_rows, irc_rows], ignore_index=True)
    combined['reporting_interval_m'] = int(CHILD_10M)

    if 'id' in combined.columns:
        combined['id'] = combined['id'].str.replace(src_table_name, table, regex=False)

    child_df = as_child(combined, table, 'child_10m', defect, upc, ca_interval_m)
    logger.info("Built child 10m table %s: %d rows (CA: %d, IRC: %d)", table, len(child_df), len(ca_rows), len(irc_rows))
    return child_df


def build_child_irc100(
    defect: str,
    table: str,
    src_100m_df: pd.DataFrame,
    src_table_name: str,
    upc: str,
    ca_interval_m: Optional[int] = None,
    irc_tv: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build child 100m IRC table.
    """
    if src_100m_df is None or src_100m_df.empty:
        return pd.DataFrame()

    out = src_100m_df.copy()
    if 'threshold_name' in out.columns:
        out = out[out['threshold_name'] == 'As per CA'].copy()

    tv = float(irc_tv if irc_tv is not None else IRC_TV.get(defect, 0.0))
    mv = pd.to_numeric(out['metric_value'], errors='coerce').fillna(0.0)
    bf = (mv > tv).astype(int)

    out['reporting_interval_m'] = int(CHILD_100M)
    out['threshold_name'] = 'As per IRC'
    out['threshold_value'] = tv
    out['breach_flag'] = bf

    length = pd.to_numeric(
        out.get('length_m', out.get('total_length', 100.0)),
        errors='coerce'
    ).fillna(100.0)
    aff_len = np.where(bf == 1, length, 0.0)
    out['affected_length'] = aff_len
    out['affected_length_irc'] = aff_len
    out['affected_length_ca'] = 0.0

    if 'road_condition' in out.columns:
        default_cond = pd.Series(np.where(bf == 1, 'Poor', 'Good'), index=out.index)
        out['road_condition'] = out.get('rating_irc', default_cond).fillna(default_cond)

    if 'defect_id' in out.columns:
        rec_id = out.get('record_id', out.get('id'))
        out['defect_id'] = np.where(bf == 1, out['defect_id'].fillna(rec_id), None)

    if 'id' in out.columns:
        out['id'] = out['id'].str.replace(src_table_name, table, regex=False)

    child_df = as_child(out, table, 'child_100m_irc', defect, upc, ca_interval_m)
    logger.info("Built child IRC 100m table %s: %d rows", table, len(child_df))
    return child_df
