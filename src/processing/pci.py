"""
PCI (Pavement Condition Index) calculation module.
Ported from Databricks DLM Extraction Pipeline (Cell 10).

Calculates weighted defect scores at 10m, rolls up to 100m and 1km,
and backfills PCI scores into defect output DataFrames.

Formula:
    PCI = 100 - Weighted_Defect_Score
Weights:
    Crack=0.12, Ravel=0.10, Pothole=0.16, Patch=0.08, Rut=0.14, Rough=0.40
Normalization:
    Each defect index is normalized 0-100 via max-observed-value scaling.
"""

from typing import Dict, Optional
import os
import numpy as np
import pandas as pd

from src.common.constants import (
    CHILD_10M,
    CHILD_100M,
    PARENT_LENGTH_M,
    PCI_WEIGHTS,
    normalize_lane,
)
from src.common.logger import get_logger
from src.common.fs_utils import path_exists, list_dir, read_parquet_df
from src.ingestion.backup_loader import discover_lane_folders

logger = get_logger(__name__)


def load_patching_data(backup_base: str, child_10m: int = CHILD_10M) -> pd.DataFrame:
    """
    Load patching data from backup lane folders.
    Groups patching area by (Lane, Start_Chainage) at 10m grid.
    """
    frames = []
    if not backup_base or not path_exists(backup_base):
        logger.warning("Backup base path does not exist: %s", backup_base)
        return pd.DataFrame(columns=['Lane', 'Start_Chainage', 'patching_area'])

    lanes = discover_lane_folders(backup_base)
    for lane in lanes:
        lane_path = f"{backup_base}/Lane={lane}" if backup_base.startswith("s3://") else os.path.join(backup_base, f"Lane={lane}")
        matches = [
            f for f in sorted(list_dir(lane_path))
            if f.endswith('.parquet') and 'patch' in f.lower()
        ]
        for pf in matches:
            full_path = f"{lane_path}/{pf}" if lane_path.startswith("s3://") else os.path.join(lane_path, pf)
            try:
                part = read_parquet_df(full_path)
            except Exception as e:
                logger.warning("Could not read patching parquet %s: %s", full_path, e)
                continue

            cols_lower = {c.lower(): c for c in part.columns}
            ch_col = cols_lower.get('chainage') or cols_lower.get('start_chainage')
            area_col = cols_lower.get('area') or cols_lower.get('area_sq_m') or cols_lower.get('total_area')
            if ch_col and area_col:
                df = part[[ch_col, area_col]].copy()
                df.columns = ['Chainage', 'Area']
                df['Lane'] = normalize_lane(lane)
                df['Chainage'] = pd.to_numeric(df['Chainage'], errors='coerce')
                df['Area'] = pd.to_numeric(df['Area'], errors='coerce').fillna(0.0)
                frames.append(df)
                logger.debug("Patching: Lane=%s/%s (%d rows)", lane, pf, len(df))

    if not frames:
        logger.warning("No patching parquets found in %s", backup_base)
        return pd.DataFrame(columns=['Lane', 'Start_Chainage', 'patching_area'])

    all_patch = pd.concat(frames, ignore_index=True).dropna(subset=['Chainage'])
    all_patch['Start_Chainage'] = (all_patch['Chainage'] // child_10m * child_10m).astype(int)
    grouped = all_patch.groupby(['Lane', 'Start_Chainage']).agg(
        patching_area=('Area', 'sum')
    ).reset_index()
    logger.info("Loaded patching 10m records: %d", len(grouped))
    return grouped


def build_pci_spine(
    output_dfs: Dict[str, pd.DataFrame],
    patching_10m: pd.DataFrame,
    child_10m: int = CHILD_10M,
) -> pd.DataFrame:
    """
    Build a continuous 10m chainage spine per lane based on min/max chainages
    across defect tables and patching data.
    """
    spine_ranges = []
    defect_tables = [
        'master_cracking',
        'master_roughness_child_10m',
        'master_rutting_child_10m',
        'master_ravelling_child_10m',
        'master_pothole',
    ]

    for tbl in defect_tables:
        if tbl not in output_dfs or output_dfs[tbl].empty:
            continue
        df = output_dfs[tbl]
        lane_col = 'lane' if 'lane' in df.columns else 'Lane'
        ch_col = 'start_chainage' if 'start_chainage' in df.columns else 'Start_Chainage'
        if lane_col not in df.columns or ch_col not in df.columns:
            continue

        temp = df[[lane_col, ch_col]].copy()
        temp.columns = ['Lane', 'ch']
        temp['ch'] = pd.to_numeric(temp['ch'], errors='coerce')
        temp = temp.dropna(subset=['ch'])
        if not temp.empty:
            r = temp.groupby('Lane').agg(min_ch=('ch', 'min'), max_ch=('ch', 'max')).reset_index()
            spine_ranges.append(r)

    if not patching_10m.empty:
        r_patch = patching_10m.groupby('Lane').agg(
            min_ch=('Start_Chainage', 'min'),
            max_ch=('Start_Chainage', 'max'),
        ).reset_index()
        spine_ranges.append(r_patch)

    if not spine_ranges:
        logger.warning("No data found to construct PCI spine")
        return pd.DataFrame(columns=['Lane', 'Start_Chainage', 'End_Chainage'])

    lane_bounds = pd.concat(spine_ranges, ignore_index=True).groupby('Lane').agg(
        min_ch=('min_ch', 'min'),
        max_ch=('max_ch', 'max'),
    ).reset_index()

    spine_rows = []
    for _, lb in lane_bounds.iterrows():
        lane = lb['Lane']
        lo = int((lb['min_ch'] // child_10m) * child_10m)
        hi = int((lb['max_ch'] // child_10m) * child_10m)
        for ch in range(lo, hi + child_10m, child_10m):
            spine_rows.append({
                'Lane': lane,
                'Start_Chainage': ch,
                'End_Chainage': ch + child_10m,
            })

    spine_df = pd.DataFrame(spine_rows)
    logger.info(
        "PCI spine built: %d segments, lanes: %s",
        len(spine_df),
        sorted(spine_df['Lane'].unique().tolist()) if not spine_df.empty else [],
    )
    return spine_df


def extract_metric_for_pci(
    df: pd.DataFrame,
    value_col: str,
    alias: str,
    agg_func: str = 'max',
    child_10m: int = CHILD_10M,
) -> pd.DataFrame:
    """
    Extract single defect metric aggregated to 10m bins for 'As per CA' rows.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Lane', 'Start_Chainage', alias])

    temp = df.copy()
    lane_col = 'lane' if 'lane' in temp.columns else 'Lane'
    ch_col = 'start_chainage' if 'start_chainage' in temp.columns else 'Start_Chainage'

    if 'threshold_name' in temp.columns:
        temp = temp[temp['threshold_name'] == 'As per CA']

    if lane_col not in temp.columns or ch_col not in temp.columns or value_col not in temp.columns:
        return pd.DataFrame(columns=['Lane', 'Start_Chainage', alias])

    res = temp[[lane_col, ch_col, value_col]].copy()
    res.columns = ['Lane', 'raw_ch', 'val']
    res['Lane'] = res['Lane'].apply(normalize_lane)
    res['raw_ch'] = pd.to_numeric(res['raw_ch'], errors='coerce')
    res = res.dropna(subset=['raw_ch'])
    res['Start_Chainage'] = (res['raw_ch'] // child_10m * child_10m).astype(int)
    res['val'] = pd.to_numeric(res['val'], errors='coerce').fillna(0.0)

    grouped = res.groupby(['Lane', 'Start_Chainage']).agg(val=('val', agg_func)).reset_index()
    return grouped.rename(columns={'val': alias})


def calculate_pci(
    output_dfs: Dict[str, pd.DataFrame],
    patching_10m: Optional[pd.DataFrame] = None,
    child_10m: int = CHILD_10M,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate PCI at 10m, 100m, and 1km levels.
    Returns dictionary with keys '10m', '100m', '1km'.
    """
    if patching_10m is None:
        patching_10m = pd.DataFrame(columns=['Lane', 'Start_Chainage', 'patching_area'])

    pci_spine = build_pci_spine(output_dfs, patching_10m, child_10m=child_10m)
    if pci_spine.empty:
        empty_res = pd.DataFrame(columns=['Lane', 'Start_Chainage', 'pci_score'])
        return {'10m': empty_res, '100m': empty_res, '1km': empty_res}

    cracking_metric = extract_metric_for_pci(
        output_dfs.get('master_cracking'), 'metric_value', 'cracking_pct', 'max', child_10m
    )
    ravelling_metric = extract_metric_for_pci(
        output_dfs.get('master_ravelling_child_10m'), 'area_sqm', 'ravelling_area', 'sum', child_10m
    )
    pothole_metric = extract_metric_for_pci(
        output_dfs.get('master_pothole'), 'area_sqm', 'pothole_area', 'sum', child_10m
    )
    rutting_metric = extract_metric_for_pci(
        output_dfs.get('master_rutting_child_10m'), 'metric_value', 'rutting_mm', 'max', child_10m
    )
    roughness_metric = extract_metric_for_pci(
        output_dfs.get('master_roughness_child_10m'), 'iri_value', 'roughness_iri', 'max', child_10m
    )

    pci_df = pci_spine.copy()
    for mdf in [cracking_metric, ravelling_metric, pothole_metric,
                rutting_metric, roughness_metric, patching_10m]:
        if not mdf.empty:
            pci_df = pci_df.merge(mdf, on=['Lane', 'Start_Chainage'], how='left')

    raw_cols = ['cracking_pct', 'ravelling_area', 'pothole_area', 'rutting_mm', 'roughness_iri', 'patching_area']
    for col in raw_cols:
        if col not in pci_df.columns:
            pci_df[col] = 0.0
        else:
            pci_df[col] = pci_df[col].fillna(0.0)

    # Max-observed scaling
    max_vals = {c: float(pci_df[c].max()) for c in raw_cols}
    index_map = {
        'cracking_pct': 'crack_index',
        'ravelling_area': 'ravel_index',
        'pothole_area': 'pothole_index',
        'patching_area': 'patch_index',
        'rutting_mm': 'rut_index',
        'roughness_iri': 'roughness_index',
    }
    for raw_col, idx_col in index_map.items():
        mx = max_vals[raw_col]
        if mx > 0:
            pci_df[idx_col] = ((pci_df[raw_col] / mx) * 100.0).clip(0, 100)
        else:
            pci_df[idx_col] = 0.0

    # Weighted defect score & PCI
    pci_df['weighted_defect_score'] = (
        pci_df['crack_index'] * PCI_WEIGHTS['cracking'] +
        pci_df['ravel_index'] * PCI_WEIGHTS['ravelling'] +
        pci_df['pothole_index'] * PCI_WEIGHTS['pothole'] +
        pci_df['patch_index'] * PCI_WEIGHTS['patching'] +
        pci_df['rut_index'] * PCI_WEIGHTS['rutting'] +
        pci_df['roughness_index'] * PCI_WEIGHTS['roughness']
    ).round(4)

    pci_df['pci_10m'] = (100.0 - pci_df['weighted_defect_score']).clip(0, 100).round(4)

    # Roll up to 100m
    pci_df['_ch_100m'] = (pci_df['Start_Chainage'] // CHILD_100M) * CHILD_100M
    pci_100m = pci_df.groupby(['Lane', '_ch_100m']).agg(pci_score=('pci_10m', 'mean')).reset_index()
    pci_100m.rename(columns={'_ch_100m': 'Start_Chainage'}, inplace=True)
    pci_100m['pci_score'] = pci_100m['pci_score'].round(4)

    # Roll up to 1km
    pci_df['_ch_1km'] = (pci_df['Start_Chainage'] // PARENT_LENGTH_M) * PARENT_LENGTH_M
    pci_1km = pci_df.groupby(['Lane', '_ch_1km']).agg(pci_score=('pci_10m', 'mean')).reset_index()
    pci_1km.rename(columns={'_ch_1km': 'Start_Chainage'}, inplace=True)
    pci_1km['pci_score'] = pci_1km['pci_score'].round(4)

    pci_lookup_10m = pci_df[['Lane', 'Start_Chainage', 'pci_10m']].rename(columns={'pci_10m': 'pci_score'})
    pci_lookup_100m = pci_100m[['Lane', 'Start_Chainage', 'pci_score']].copy()
    pci_lookup_1km = pci_1km[['Lane', 'Start_Chainage', 'pci_score']].copy()

    logger.info(
        "PCI calculation complete: 10m (%d), 100m (%d), 1km (%d)",
        len(pci_lookup_10m), len(pci_lookup_100m), len(pci_lookup_1km)
    )

    return {
        '10m': pci_lookup_10m,
        '100m': pci_lookup_100m,
        '1km': pci_lookup_1km,
    }


def fill_pci_column(df: pd.DataFrame, pci_lookup: pd.DataFrame, grid_m: int) -> pd.DataFrame:
    """
    Join PCI scores onto a defect dataframe by (Lane, floored_chainage).
    """
    if df is None or df.empty or pci_lookup is None or pci_lookup.empty:
        if df is not None and 'pci' not in df.columns:
            df['pci'] = None
        return df

    out = df.copy()
    lane_col = 'lane' if 'lane' in out.columns else 'Lane'
    ch_col = 'start_chainage' if 'start_chainage' in out.columns else 'Start_Chainage'

    out['_orig_idx'] = np.arange(len(out))
    out['_pci_lane'] = out[lane_col].astype(str).str.strip().str.upper()
    out['_join_ch'] = (pd.to_numeric(out[ch_col], errors='coerce').fillna(0) // grid_m * grid_m).astype(int)

    lookup_clean = pci_lookup.copy()
    lookup_clean['_pci_lane'] = lookup_clean['Lane'].astype(str).str.strip().str.upper()
    lookup_clean['_join_ch'] = lookup_clean['Start_Chainage'].astype(int)

    merged = out.merge(
        lookup_clean[['_pci_lane', '_join_ch', 'pci_score']],
        on=['_pci_lane', '_join_ch'],
        how='left'
    ).sort_values('_orig_idx').reset_index(drop=True)

    merged['pci'] = merged['pci_score'].astype('float64')
    merged.drop(columns=['_orig_idx', '_pci_lane', '_join_ch', 'pci_score'], inplace=True, errors='ignore')
    return merged


def backfill_pci_to_outputs(
    output_dfs: Dict[str, pd.DataFrame],
    pci_lookups: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Backfill PCI column across all OUTPUT_DATAFRAMES matching their interval level.
    """
    res = {}
    for tbl_name, df in output_dfs.items():
        if df is None or df.empty:
            res[tbl_name] = df
            continue

        if 'pci' not in df.columns:
            res[tbl_name] = df
            continue

        if 'parent' in tbl_name:
            lvl = '1km'
            grid = PARENT_LENGTH_M
        elif tbl_name.endswith('_child_ca'):
            iv = pd.to_numeric(df.get('reporting_interval_m'), errors='coerce').dropna()
            iv = int(iv.iloc[0]) if len(iv) else CHILD_100M
            lvl, grid = ('1km', PARENT_LENGTH_M) if iv >= PARENT_LENGTH_M else ('100m', CHILD_100M)
        elif '100m' in tbl_name:
            lvl = '100m'
            grid = CHILD_100M
        else:
            lvl = '10m'
            grid = CHILD_10M

        res[tbl_name] = fill_pci_column(df, pci_lookups[lvl], grid)
        logger.debug("Backfilled PCI in %s at level %s", tbl_name, lvl)

    return res
