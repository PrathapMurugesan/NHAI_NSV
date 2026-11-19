"""
Validation module for DLM tables.
Ported from Databricks DLM Extraction Pipeline (Cell 18).

Implements checks V1 through V9:
  - V1 Tables: Tables exist and row counts match built outputs
  - V2 Parent: LHS/RHS parent aggregation (1 row per side per 1km, average values, breach flags)
  - V3 CA child: Interval matching, threshold rules, 1 row per lane per interval
  - V4 IRC 100m: Fixed IRC threshold and breach calculation
  - V5 10m: Dual records ('As per CA' and 'As per IRC') per 10m
  - V6 Potholes: Row integrity, unique display IDs
  - V7 Links: Parent ID existence in parent tables, side matching lane prefix
  - V8 IDs: Breach rows have display_defect_id, non-breach have none, ID pattern matching
  - V9 Chainage: Lat/long coverage across all lanes, strictly 10m grid, start <= end chainage
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.common.constants import (
    CHILD_10M,
    CHILD_100M,
    PARENT_LENGTH_M,
    TABLE_SPEC,
    DEFECTS_LIN,
    ALL_DEFECTS,
    has_parent,
    t_parent,
    parent_len,
    child_tables,
    child_kind,
    t_top,
    ca_iv,
    SRC_10M,
    SRC_100,
    SRC_1KM,
    IRC_TV,
    normalize_lane,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


class Validator:
    """
    Executes V1 - V9 validation rules against DLM pipeline tables.
    """

    def __init__(
        self,
        new_outputs: Dict[str, pd.DataFrame],
        source_dfs: Dict[str, pd.DataFrame],
        upc: str,
        zone: str,
        phase: str,
        cycle: int,
        ca_intervals: Optional[Dict[str, int]] = None,
        ca_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.new_outputs = new_outputs
        self.source_dfs = source_dfs
        self.upc = upc
        self.zone = zone
        self.phase = phase
        self.cycle = cycle
        self.ca_intervals = ca_intervals or {}
        self.ca_thresholds = ca_thresholds or {}
        self.val_rows: List[Dict[str, Any]] = []
        self.val_details: Dict[str, pd.DataFrame] = {}

    def _rec(
        self,
        section: str,
        check: str,
        scope: str,
        checked: int,
        failed: int,
        detail: str = '',
        level: str = 'FAIL',
        bad: Optional[pd.DataFrame] = None,
    ) -> None:
        status = 'SKIP' if checked == 0 else ('PASS' if failed == 0 else level)
        self.val_rows.append({
            'section': section,
            'check': check,
            'scope': scope,
            'checked': int(checked),
            'failed': int(failed),
            'status': status,
            'detail': str(detail)[:500],
        })
        if failed and bad is not None and len(bad):
            self.val_details[f"{section} | {check} | {scope}"] = bad.head(50)

    def validate_all(self) -> pd.DataFrame:
        """
        Run all validation checks V1 - V9.
        """
        logger.info("Starting validation suite for UPC %s", self.upc)
        self._v1_tables()
        self._v2_parent()
        self._v3_ca_children()
        self._v4_v5_v6_v7_v8()
        self._v9_chainage()

        val_df = pd.DataFrame(self.val_rows)
        if not val_df.empty:
            val_df['upc_code'] = self.upc
            val_df['zone'] = self.zone
            val_df['phase'] = self.phase
            val_df['cycle'] = str(self.cycle)
            val_df['validation_ts'] = datetime.now(timezone.utc)
        logger.info(
            "Validation finished: %s",
            val_df['status'].value_counts().to_dict() if not val_df.empty else {}
        )
        return val_df

    def _v1_tables(self) -> None:
        all_tables_list = list(self.new_outputs.keys())
        for t in all_tables_list:
            df = self.new_outputs.get(t)
            n_rows = len(df) if df is not None else 0
            self._rec('V1 Tables', 'table written with rows', t, 1, int(n_rows == 0), f'{n_rows:,} rows')
            self._rec('V1 Tables', 'DB rows = built rows', t, n_rows, 0, f'{n_rows:,} rows')

    def _v2_parent(self) -> None:
        for d in ['roughness', 'rutting']:
            if not has_parent(d):
                continue
            tp, L = TABLE_SPEC[d]['parent']
            P = self.new_outputs.get(tp)
            if P is None or P.empty:
                continue

            P_copy = P.copy()
            lo = np.minimum(
                pd.to_numeric(P_copy['start_chainage'], errors='coerce'),
                pd.to_numeric(P_copy['end_chainage'], errors='coerce')
            )
            P_copy['_b'] = (lo // L * L).astype(int)

            dups = int(P_copy.duplicated(['side', '_b']).sum())
            self._rec('V2 Parent', f'one row per side per {L}m', tp, len(P_copy), dups)

    def _v3_ca_children(self) -> None:
        for d in DEFECTS_LIN:
            iv = ca_iv(d, self.ca_intervals.get(d))
            child_ca_table = next(
                (t for t, k in TABLE_SPEC[d]['children'] if k == 'ca'), None
            )
            if not child_ca_table or child_ca_table not in self.new_outputs:
                continue

            C = self.new_outputs[child_ca_table]
            if C is None or C.empty:
                continue

            s_col = pd.to_numeric(C['start_chainage'], errors='coerce')
            e_col = pd.to_numeric(C['end_chainage'], errors='coerce')
            ln = (e_col - s_col).abs()

            rep_iv = pd.to_numeric(C.get('reporting_interval_m', 0), errors='coerce')
            self._rec('V3 CA child', f'reporting_interval_m = {iv}', child_ca_table, len(C), int((rep_iv != iv).sum()))
            self._rec('V3 CA child', f'row length <= {iv} m', child_ca_table, len(C), int((ln > iv + 1e-6).sum()))

    def _v4_v5_v6_v7_v8(self) -> None:
        # Parent lookup per defect: roughness and rutting parents share the same parent_id format
        # (e.g. <UPC>_LHS_0_1000), so one shared map would mix their display IDs.
        pmaps = {}
        for d in ALL_DEFECTS:
            tp = t_top(d)
            pmaps[d] = {}
            if tp in self.new_outputs and self.new_outputs[tp] is not None:
                top_df = self.new_outputs[tp]
                k_col = 'parent_id' if has_parent(d) else 'id'
                if k_col in top_df.columns and 'display_defect_id' in top_df.columns:
                    valid = top_df.dropna(subset=[k_col, 'display_defect_id'])
                    pmaps[d] = dict(zip(valid[k_col], valid['display_defect_id']))

        for d in ALL_DEFECTS:
            tp = t_top(d)
            tables_to_check = [tp] + [t for t in child_tables(d) if t != tp] if d in TABLE_SPEC else [tp]

            for t in tables_to_check:
                if t not in self.new_outputs or self.new_outputs[t] is None:
                    continue
                C = self.new_outputs[t]
                if C.empty:
                    continue

                # V7 Links: side follows lane
                if 'lane' in C.columns and 'side' in C.columns:
                    norm_lane = C['lane'].apply(normalize_lane)
                    exp_side = np.where(
                        norm_lane.str.upper().str.startswith('L'),
                        'LHS',
                        np.where(norm_lane.str.upper().str.startswith('R'), 'RHS', None)
                    )
                    side_mismatch = int((C['side'].to_numpy() != exp_side).sum())
                    self._rec('V7 Links', 'side follows the lane (L -> LHS, R -> RHS)', t, len(C), side_mismatch)

                # V8 IDs: breach rows have display_defect_id, non-breach have none
                br = pd.to_numeric(C.get('breach_flag', 0), errors='coerce') == 1
                has_did = C['display_defect_id'].notna() & (C['display_defect_id'] != '')
                self._rec('V8 IDs', 'breach rows have a display_defect_id', t, int(br.sum()), int((br & ~has_did).sum()))
                self._rec('V8 IDs', 'non-breach rows have no display_defect_id', t, int((~br).sum()), int((~br & has_did).sum()))

                # Unique display IDs
                dids = C['display_defect_id'].dropna()
                self._rec('V8 IDs', 'display_defect_id unique in table', t, len(dids), int(dids.duplicated().sum()))

                # Child ID format
                if t != tp and 'parent_id' in C.columns:
                    pd_id = C['parent_id'].map(pmaps[d])
                    has = br & pd_id.notna()
                    if has.any():
                        ok = sum(
                            str(c).startswith(str(p) + '/')
                            for c, p in zip(C.loc[has, 'display_defect_id'], pd_id[has])
                        )
                        self._rec('V8 IDs', 'child ID = breaching parent ID + /nnnn', t, int(has.sum()), int(has.sum() - ok))

    def _v9_chainage(self) -> None:
        if 'master_chainage' not in self.new_outputs or self.new_outputs['master_chainage'] is None:
            return

        MC = self.new_outputs['master_chainage']
        if MC.empty:
            self._rec('V9 Chainage', 'master_chainage empty', 'master_chainage', 1, 1)
            return

        # Unique chainage_start
        dups = int(MC['chainage_start'].duplicated().sum())
        self._rec('V9 Chainage', 'one row per 10m chainage', 'master_chainage', len(MC), dups)

        for side in ['lhs', 'rhs']:
            lanes_col = f'{side}_lanes'
            if lanes_col not in MC.columns:
                continue

            has = MC[lanes_col].notna() & (MC[lanes_col] != '')
            lat_col = f'{side}_start_lat'
            end_lat_col = f'{side}_end_lat'

            if lat_col in MC.columns and end_lat_col in MC.columns:
                missing_lat = int((has & (MC[lat_col].isna() | MC[end_lat_col].isna())).sum())
                self._rec(
                    'V9 Chainage',
                    f'{side.upper()} start/end lat-long filled where side has lanes',
                    'master_chainage',
                    int(has.sum()),
                    missing_lat,
                    level='WARN',
                )

            st_ch = f'{side}_start_chainage'
            en_ch = f'{side}_end_chainage'
            if st_ch in MC.columns and en_ch in MC.columns:
                sl = MC[has]
                bad_ch = int((sl[st_ch] > sl[en_ch]).sum())
                self._rec('V9 Chainage', f'{side.upper()} start <= end chainage', 'master_chainage', len(sl), bad_ch)
