"""
Excel exporter module for DLM tables.
Ported from Databricks DLM Extraction Pipeline (Cell 21).

Exports every DLM table into ONE multi-sheet Excel file with:
  - Formatted dates, arrays, and JSON
  - Frozen top panes and auto-filters
  - An 'index' sheet listing row/breach counts moved to the first tab position
"""

from typing import Dict, Any, List, Optional
import os
import tempfile
import shutil
from datetime import datetime
import pandas as pd
import numpy as np

from src.common.constants import ALL_DEFECTS, all_tables
from src.common.logger import get_logger

logger = get_logger(__name__)


def prepare_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format columns to ensure Excel compatibility (timezone-naive dates,
    strings for dicts/lists).
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.dt.tz_localize(None) if getattr(s.dt, 'tz', None) is not None else s
        elif s.dtype == object:
            out[c] = s.map(
                lambda v: ', '.join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray))
                else (str(v) if isinstance(v, dict) else v)
            )
    return out


def export_to_excel(
    tables: Dict[str, pd.DataFrame],
    output_path: str,
    validation_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Export all tables and validation results to a single Excel workbook.
    """
    is_s3 = output_path.startswith("s3://")
    if not is_s3:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    order_tables: List[str] = []
    for d in ALL_DEFECTS:
        for t in all_tables(d):
            if t not in order_tables:
                order_tables.append(t)
    for extra_t in ['geometry_parent', 'geometry_child', 'master_chainage']:
        if extra_t not in order_tables:
            order_tables.append(extra_t)

    all_dfs = dict(tables)
    if validation_df is not None and not validation_df.empty:
        all_dfs['dlm_validation_results'] = validation_df
        order_tables.append('dlm_validation_results')

    sort_cols_priority = ['side', 'lane', 'start_chainage', 'chainage_start', 'section', 'check']
    index_rows = []

    tmp_file = tempfile.mktemp(suffix='.xlsx')
    with pd.ExcelWriter(tmp_file, engine='openpyxl') as xw:
        for tbl in order_tables:
            if tbl not in all_dfs or all_dfs[tbl] is None:
                continue

            df = all_dfs[tbl]
            if df.empty:
                continue

            order = [c for c in sort_cols_priority if c in df.columns]
            sorted_df = df.sort_values(order) if order else df
            ready_df = prepare_for_excel(sorted_df)

            sheet_name = tbl[:31]
            ready_df.to_excel(xw, sheet_name=sheet_name, index=False)

            ws = xw.sheets[sheet_name]
            ws.freeze_panes = 'A2'
            if ws.dimensions:
                ws.auto_filter.ref = ws.dimensions

            breach_count = None
            if 'breach_flag' in ready_df.columns:
                breach_count = int(pd.to_numeric(ready_df['breach_flag'], errors='coerce').eq(1).sum())

            index_rows.append({
                'sheet': sheet_name,
                'table': tbl,
                'rows': len(ready_df),
                'breach_rows': breach_count,
                'columns': ready_df.shape[1],
            })
            logger.debug("Excel sheet written: %s (%d rows)", sheet_name, len(ready_df))

        # Write index sheet
        index_df = pd.DataFrame(index_rows)
        index_df.to_excel(xw, sheet_name='index', index=False)
        # Move index to first sheet
        xw.book.move_sheet('index', offset=-len(xw.book.sheetnames) + 1)

    if is_s3:
        import pyarrow.fs as pafs
        fs, path_in_fs = pafs.FileSystem.from_uri(output_path)
        with open(tmp_file, 'rb') as f_in:
            with fs.open_output_stream(path_in_fs) as f_out:
                f_out.write(f_in.read())
    else:
        shutil.copy(tmp_file, output_path)

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    logger.info("Saved multi-sheet Excel export to %s (%d sheets)", output_path, len(index_rows) + 1)
    return output_path
