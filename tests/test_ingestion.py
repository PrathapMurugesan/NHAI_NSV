"""
Unit tests for ingestion layer, normalizers, and bronze calculations.
"""

import unittest
import pandas as pd
import numpy as np

from src.common.constants import (
    normalize_lane,
    lane_to_direction,
    normalize_direction,
    standardize_nh_number,
    as_chainage_m,
    as_float,
    as_int,
)
from src.ingestion.bronze_loader import (
    crack_condition_from_pct,
    roughness_condition_from_metrics,
    rutting_condition_from_mm,
    ravelling_condition_from_pct,
)


class TestIngestionAndNormalizers(unittest.TestCase):

    def test_normalize_lane(self):
        self.assertEqual(normalize_lane("L1"), "L1")
        self.assertEqual(normalize_lane("lane_l2"), "L2")
        self.assertEqual(normalize_lane("R1_lane"), "R1")
        self.assertIsNone(normalize_lane(""))
        self.assertIsNone(normalize_lane(None))

    def test_lane_to_direction(self):
        self.assertEqual(lane_to_direction("L1"), "Increasing")
        self.assertEqual(lane_to_direction("L2"), "Increasing")
        self.assertEqual(lane_to_direction("R1"), "Decreasing")
        self.assertEqual(lane_to_direction("R2"), "Decreasing")
        self.assertIsNone(lane_to_direction("CL"))

    def test_normalize_direction(self):
        self.assertEqual(normalize_direction("inc"), "Increasing")
        self.assertEqual(normalize_direction("INCREASING"), "Increasing")
        self.assertEqual(normalize_direction("dec"), "Decreasing")
        self.assertEqual(normalize_direction("DECREASING"), "Decreasing")
        self.assertEqual(normalize_direction("unknown"), "unknown")

    def test_standardize_nh_number(self):
        self.assertEqual(standardize_nh_number("NH0027"), "NH-27")
        self.assertEqual(standardize_nh_number("NH-027"), "NH-27")
        self.assertEqual(standardize_nh_number("NH151A"), "NH-151A")
        self.assertEqual(standardize_nh_number("27"), "NH-27")
        self.assertEqual(standardize_nh_number("NH-44"), "NH-44")
        self.assertIsNone(standardize_nh_number(None))
        self.assertIsNone(standardize_nh_number(""))

    def test_chainage_helpers(self):
        self.assertEqual(as_chainage_m(500), 500)
        self.assertEqual(as_chainage_m("500"), 500)
        self.assertEqual(as_chainage_m(1.5, is_km=True), 1500)
        self.assertIsNone(as_chainage_m(None))

    def test_scalar_converters(self):
        self.assertEqual(as_float("12.34"), 12.34)
        self.assertIsNone(as_float("invalid"))
        self.assertEqual(as_int("42"), 42)
        self.assertIsNone(as_int("bad"))

    def test_condition_categorizers(self):
        # Cracking: t=5.0 -> <5 is Very Good (<5%), 5-10 is Good
        self.assertEqual(crack_condition_from_pct(2.0, 5.0), "Very Good (<5%)")
        self.assertEqual(crack_condition_from_pct(7.0, 5.0), "Good (5-10%)")
        self.assertIsNone(crack_condition_from_pct(None, 5.0))

        # Roughness BI: t=2000, 0.8*t=1600 -> <1600 is Good (<1600mm/km)
        self.assertEqual(roughness_condition_from_metrics(None, 1500, 2000), "Good (<1600mm/km)")
        self.assertEqual(roughness_condition_from_metrics(None, 2500, 2000), "Poor (>2000mm/km)")

        # Rutting: t=5.0, 0.5*t=2.5 (rounded to 2) -> <2 Good, 2-5 Fair, >5 Poor
        self.assertEqual(rutting_condition_from_mm(1.0, 5.0), "Good (<2mm)")
        self.assertEqual(rutting_condition_from_mm(3.0, 5.0), "Fair (2-5mm)")
        self.assertEqual(rutting_condition_from_mm(7.0, 5.0), "Poor (>5mm)")

        # Ravelling: t=10.0, 0.5*t=5.0 -> 0 Very Good, 1-5 Good, 5-10 Fair, >=10 Poor
        self.assertEqual(ravelling_condition_from_pct(0.0, 10.0), "Very Good (0)")
        self.assertEqual(ravelling_condition_from_pct(3.0, 10.0), "Good (1-5%)")
        self.assertEqual(ravelling_condition_from_pct(15.0, 10.0), "Poor (>=10%)")


if __name__ == "__main__":
    unittest.main()
