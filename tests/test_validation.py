"""
Unit tests for validation layer: V1 through V9 checks.
"""

import unittest
import pandas as pd
import numpy as np

from src.validation.validator import Validator
from src.validation.validation_report import generate_validation_summary


class TestValidation(unittest.TestCase):

    def test_validator_execution(self):
        # Create minimal valid outputs
        parent_df = pd.DataFrame({
            'id': ['P1'],
            'parent_id': ['UPC001_LHS_0_1000'],
            'lane': ['LHS'],
            'side': ['LHS'],
            'start_chainage': [0],
            'end_chainage': [1000],
            'metric_value': [1800.0],
            'threshold_name': ['As per CA'],
            'breach_flag': [1],
            'display_defect_id': ['D/181126/N/00001'],
        })

        child_df = pd.DataFrame({
            'id': ['C1'],
            'parent_id': ['UPC001_LHS_0_1000'],
            'lane': ['L1'],
            'side': ['LHS'],
            'start_chainage': [0],
            'end_chainage': [10],
            'metric_value': [1900.0],
            'threshold_name': ['As per CA'],
            'breach_flag': [1],
            'display_defect_id': ['D/181126/N/00001/0001'],
            'reporting_interval_m': [10],
        })

        mc_df = pd.DataFrame({
            'chainage_start': [0],
            'chainage_end': [10],
            'l1_start_lat': [12.0],
            'lhs_lanes': ['L1'],
            'lhs_start_lat': [12.0],
            'lhs_end_lat': [12.0001],
            'lhs_start_chainage': [0],
            'lhs_end_chainage': [10],
        })

        new_outputs = {
            'master_roughness_parent': parent_df,
            'master_roughness_child_10m': child_df,
            'master_chainage': mc_df,
        }

        validator = Validator(
            new_outputs=new_outputs,
            source_dfs={},
            upc='UPC001',
            zone='East',
            phase='1',
            cycle=1,
            ca_intervals={'roughness': 100},
            ca_thresholds={'roughness': 2000.0},
        )

        val_df = validator.validate_all()

        self.assertFalse(val_df.empty)
        # Check that V1 Tables has PASS
        v1_status = val_df[val_df['section'] == 'V1 Tables']['status'].tolist()
        self.assertTrue(all(s == 'PASS' for s in v1_status))

        # Check summary formatting string
        summary_str = generate_validation_summary(val_df)
        self.assertIn("Validation Results Summary", summary_str)


if __name__ == "__main__":
    unittest.main()
