"""
Batch validation report generator.
Ported from Databricks DLM Extraction Pipeline (Cell 20).

Compiles unified validation results for multiple UPCs into an Excel workbook:
  - Sheet 'summary': latest run per UPC from dlm_validation_results
  - Sheet 'by_upc': pivot table of PASS / WARN / FAIL / SKIP per UPC
  - Sheets 'detail_*': failing rows of the runs
"""

from typing import Dict, Any, List, Optional
import os
import tempfile
import shutil
from datetime import datetime
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


def generate_batch_validation_excel(
    all_validation_df: pd.DataFrame,
    output_excel_path: str,
    val_details: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """
    Generate multi-sheet Excel file with summary, by_upc pivot, and details.
    """
    if all_validation_df is None or all_validation_df.empty:
        logger.warning("No validation records to export in batch report")
        return ""

    os.makedirs(os.path.dirname(os.path.abspath(output_excel_path)), exist_ok=True)
    val_details = val_details or {}

    vr = all_validation_df.copy()
    by_upc = vr.pivot_table(
        index=['upc_code', 'zone', 'phase', 'cycle'],
        columns='status',
        values='check',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    tmp_file = tempfile.mktemp(suffix='.xlsx')
    with pd.ExcelWriter(tmp_file, engine='openpyxl') as xw:
        sort_order = [c for c in ['upc_code', 'status', 'section'] if c in vr.columns]
        vr.sort_values(sort_order).to_excel(xw, sheet_name='summary', index=False)
        by_upc.to_excel(xw, sheet_name='by_upc', index=False)

        for i, (k, v) in enumerate(val_details.items(), 1):
            sheet_name = f"detail_{i}"[:31]
            detail_df = v.copy()
            detail_df['check'] = k
            detail_df.to_excel(xw, sheet_name=sheet_name, index=False)

    shutil.copy(tmp_file, output_excel_path)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    logger.info("Saved batch validation Excel report to %s", output_excel_path)
    return output_excel_path
