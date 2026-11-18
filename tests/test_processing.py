"""
Unit tests for processing layer: defect calculations, severity, and PCI.
"""

import unittest
import pandas as pd
import numpy as np

from src.processing.cracking import apply_cracking_severity_and_condition
from src.processing.pci import calculate_pci, fill_pci_column
from src.processing.ratings import compute_ca_rating_label


class TestProcessingCalculations(unittest.TestCase):

    def test_cracking_severity_and_condition(self):
        # Create test cracking dataframe
        df = pd.DataFrame({
            'lane': ['L1', 'L1', 'L1'],
            'start_chainage': [0, 10, 20],
            'end_chainage': [10, 20, 30],
            'metric_value': [2.0, 6.0, 15.0],
            'threshold_name': ['As per CA', 'As per CA', 'As per IRC'],
            'measures': [
                '{"avg_corrected_width_mm": 2.5}',
                '{"avg_corrected_width_mm": 4.5}',
                '{"avg_width_mm": 7.0}',
            ],
        })

        processed = apply_cracking_severity_and_condition(df, ca_crack_pct=5.0)

        # Width < 3 -> None, 3-6 -> Medium, >=6 -> High
        self.assertEqual(processed['severity'].iloc[0], 'None')
        self.assertEqual(processed['severity'].iloc[1], 'Medium')
        self.assertEqual(processed['severity'].iloc[2], 'High')

        # CA condition: 2.0 <= 5.0 -> Good, 6.0 > 5.0 -> Poor
        self.assertEqual(processed['rating_ca'].iloc[0], 'Good')
        self.assertEqual(processed['rating_ca'].iloc[1], 'Poor')

        # IRC condition: 15.0 -> Fair (10-20%)
        self.assertEqual(processed['rating_irc'].iloc[2], 'Fair (10-20%)')

    def test_pci_calculation(self):
        # Mock 10m defect outputs
        crack_df = pd.DataFrame({
            'lane': ['L1', 'L1'],
            'start_chainage': [0, 10],
            'end_chainage': [10, 20],
            'metric_value': [0.0, 10.0],
            'threshold_name': ['As per CA', 'As per CA'],
        })
        rough_df = pd.DataFrame({
            'lane': ['L1', 'L1'],
            'start_chainage': [0, 10],
            'end_chainage': [10, 20],
            'iri_value': [1.5, 3.0],
            'threshold_name': ['As per CA', 'As per CA'],
        })
        outputs = {
            'master_cracking': crack_df,
            'master_roughness_child_10m': rough_df,
        }

        pci_results = calculate_pci(outputs)
        pci_10m = pci_results['10m']

        self.assertFalse(pci_10m.empty)
        self.assertEqual(len(pci_10m), 2)
        # Segment 0 has lower defects than segment 1 -> segment 0 PCI >= segment 1 PCI
        score_0 = pci_10m.loc[pci_10m['Start_Chainage'] == 0, 'pci_score'].iloc[0]
        score_1 = pci_10m.loc[pci_10m['Start_Chainage'] == 10, 'pci_score'].iloc[0]
        self.assertGreater(score_0, score_1)

    def test_ca_rating_label(self):
        # Cracking: t=5.0
        self.assertEqual(compute_ca_rating_label('cracking', 3.0), 'Very Good (<5%)')
        self.assertEqual(compute_ca_rating_label('cracking', 8.0), 'Good (5-10%)')
        self.assertEqual(compute_ca_rating_label('cracking', 15.0), 'Fair (10-20%)')
        self.assertEqual(compute_ca_rating_label('cracking', 25.0), 'Poor (20-30%)')
        self.assertEqual(compute_ca_rating_label('cracking', 35.0), 'Very Poor (>30%)')

        # Roughness: t=2000, fl=1600
        self.assertEqual(compute_ca_rating_label('roughness', 1.2, bi_value=1200), 'Good (<1600mm/km)')
        self.assertEqual(compute_ca_rating_label('roughness', 1.8, bi_value=1800), 'Fair (1600-2000mm/km)')
        self.assertEqual(compute_ca_rating_label('roughness', 2.5, bi_value=2500), 'Poor (>2000mm/km)')


if __name__ == "__main__":
    unittest.main()
