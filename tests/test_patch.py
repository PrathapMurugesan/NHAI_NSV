"""
Tests for patch/patch_parent_breach.py on a small local silver folder.
"""

import json
import os
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "patch"))
import patch_parent_breach as P  # noqa: E402


def _lane_table():
    return pd.DataFrame({
        'id': ['U1_lane_L1_0', 'U1_lane_L2_0'], 'lane': ['L1', 'L2'],
        'start_chainage': [0, 0], 'end_chainage': [1000, 1000], 'length_m': [1000.0, 1000.0],
        'parent_start_chainage': [0, 0], 'parent_id': ['U1_L1_0_1000', 'U1_L2_0_1000'],
        'breach_flag': [0, 0], 'defect_id': [None, None], 'road_condition': ['Good', 'Good'],
        'affected_length_ca': [0.0, 0.0], 'affected_length': [0.0, 0.0], 'affected_length_irc': [0.0, 0.0],
        'measures': [json.dumps({}), json.dumps({})],
    })


def _children_10m():
    rows = []
    for lane in ('L1', 'L2'):
        for st in range(0, 1000, 10):
            for std in ('CA', 'IRC'):
                rows.append(dict(lane=lane, start_chainage=st, end_chainage=st + 10, threshold_name=f'As per {std}',
                                 breach_flag=int(lane == 'L1' and st == 500 and std == 'CA')))
    return pd.DataFrame(rows)


class TestParentBreachPatch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.folder = os.path.join(self.tmp, "Zone=B", "Phase=2", "Cycle=1", "UPC=U1", "dlm_kpi_new")
        os.makedirs(self.folder)
        _lane_table().to_parquet(os.path.join(self.folder, "master_cracking_1000m_lane.parquet"), index=False)
        _children_10m().to_parquet(os.path.join(self.folder, "master_cracking_10m.parquet"), index=False)

    def _lane_file(self):
        return os.path.join(self.folder, "master_cracking_1000m_lane.parquet")

    def test_dry_run_writes_nothing(self):
        before = pd.read_parquet(self._lane_file())
        P.main(["--silver-root", self.tmp, "--all", "--workers", "1"])
        pd.testing.assert_frame_equal(before, pd.read_parquet(self._lane_file()))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "_patch_backup")))

    def test_apply_patches_the_lane_table_once(self):
        P.main(["--silver-root", self.tmp, "--all", "--apply", "--workers", "1"])
        out = pd.read_parquet(self._lane_file()).set_index('lane')
        self.assertEqual(out.loc['L1', 'breach_flag'], 1)                  # one breaching 10m child -> the lane breaches
        self.assertEqual(out.loc['L1', 'defect_id'], 'U1_L1_0_1000')
        self.assertEqual(out.loc['L1', 'road_condition'], 'Bad')
        self.assertEqual(out.loc['L1', 'affected_length_ca'], 1000.0)
        self.assertEqual(out.loc['L1', 'affected_length_irc'], 0.0)         # the child breaches CA only
        self.assertEqual(out.loc['L2', 'breach_flag'], 0)
        self.assertEqual(out.loc['L2', 'road_condition'], 'Good')
        m = json.loads(out.loc['L1', 'measures'])
        self.assertEqual((m['breach_from_child_ca'], m['breach_from_value_ca']), (1, 0))
        self.assertIn('patched_parent_breach', m)

        backups = [f for _, _, fs in os.walk(os.path.join(self.tmp, "_patch_backup")) for f in fs]
        self.assertEqual(backups, ["master_cracking_1000m_lane.parquet"])   # the original was kept first

        before = pd.read_parquet(self._lane_file())
        P.main(["--silver-root", self.tmp, "--all", "--apply", "--workers", "1"])   # second run changes nothing
        pd.testing.assert_frame_equal(before, pd.read_parquet(self._lane_file()))

    def test_upc_without_lane_table_is_skipped_not_failed(self):
        os.remove(self._lane_file())
        rows = P.patch_upc(self.tmp, ("B", "2", "1", "U1"), False, "unused", "t")
        lane_rows = [r for r in rows if r['table'] == 'master_cracking_1000m_lane']
        self.assertEqual(lane_rows[0]['status'], 'skipped')
        self.assertNotIn('ERROR', {r['status'] for r in rows})


if __name__ == "__main__":
    unittest.main()
