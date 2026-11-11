"""The rolling cooler (Analysis Upgrade, card A2; decision D-1, DA1).

The cooler meter reads unsynchronised with the bunch, so each reading
catches the mains ripple at a random phase and a point-by-point correction
scatters every event by it. With ``cooler_correction: 'rolling'`` each event
instead takes the mean of ``cooler_window_points`` readings centred on its
own: one reading per bunch, from the run file itself. Point by point stays
the default, and is untouched.

Built on synthetic ASDF runs driven through the real clstools.

Run from the project root:
    .venv/bin/python -m pytest tests/test_rolling_cooler.py -q
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from gui.analysis.pipeline import UPGRADE_TOGGLES, prepare_run_data

HEADER_KV = 3.0007        # the run-start snapshot the header holds
BASE_KV = 3.0             # the cooler's level over the run (/ 10000)
RIPPLE_KV = 0.33e-4       # 0.33 V at the platform
N_BUNCH = 4000
SOURCE_CFG = dict(
    mass=100.0, ref_freq=4.0e14, harmonic=2, bin_mode="Frequency",
    cooler_correction="pbp", cal_order=1, x_column="bins_center",
    yerr_mode="Poisson sqrt(y+1)", xerr_mode="None",
    bin_definition="Per scan step", pmt_gate=[3, 4], tof_gate=None,
    v_gate=None, f_gate=None, noise_filter=0, ref_shift=0.0,
)


def Write_Run(path, bunch_kv, hits):
    """One row per hit: bunch ``i`` at 0.05 s intervals, ``hits[i]`` hits,
    every hit stamped with its bunch's reading ``bunch_kv[i]``."""
    import asdf
    rows = []
    for i, (kv, k) in enumerate(zip(bunch_kv, hits)):
        step = 10.0 * (i % 40)
        for j in range(k):
            rows.append([i * 0.05 + j * 1e-5, step, i, 3 + j % 2,
                         50.0, kv])
    cal_set = np.linspace(0.0, 400.0, 21)
    asdf.AsdfFile({
        "Run": 2, "CoolerVoltage": HEADER_KV, "LaserSetpoint": 20000.0,
        "DwellTime": 0.05, "Experiment": "A2", "Date": "2026-11-11",
        "StepSize": 10.0, "ScanningRanges": [[0.0, 390.0]],
        "CalSet": cal_set, "CalReadback": cal_set / 1000.0,
        "raw": np.array(rows, dtype=float),
    }).write_to(path)


class Rolling_Cooler_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rolling_cooler_")
        rng = np.random.default_rng(11)
        t = np.arange(N_BUNCH) * 0.05
        # A slow drift of 0.2 V over the run, and the ripple caught at a
        # random phase in every reading.
        cls.drift_kv = BASE_KV + 0.2e-4 * t / t[-1]
        cls.bunch_kv = (cls.drift_kv
                        + RIPPLE_KV * np.sin(rng.uniform(0, 2 * np.pi,
                                                         N_BUNCH)))
        cls.hits = rng.integers(1, 4, N_BUNCH)
        cls.path = os.path.join(cls.tmp, "run_0002.asdf")
        Write_Run(cls.path, cls.bunch_kv, cls.hits)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Prep(self, path=None, **extra):
        cfg = dict(SOURCE_CFG)
        cfg.update(extra)
        data, _eff, meta = prepare_run_data(path or self.path, cfg)
        return data.Sorted, meta

    def _Cooler_Kv(self, path=None, **extra):
        """Each hit's cooler voltage as Compute_Voltages used it, in time
        order (V = cooler - DV_cal)."""
        frame, meta = self._Prep(path, **extra)
        frame = frame.sort_values("TS", kind="stable")
        return ((frame["V"] + frame["DV_cal"]).to_numpy() / 10000.0,
                frame["Bunch"].to_numpy().astype(int), meta)

    def test_It_Smooths_The_Ripple_And_Keeps_The_Drift(self):
        pbp, bunch, _ = self._Cooler_Kv()
        rolled, _, _ = self._Cooler_Kv(cooler_correction="rolling",
                                       cooler_window_points=100)
        truth = self.drift_kv[bunch]
        scatter_pbp = np.std(pbp - truth) * 1e4
        scatter_rolled = np.std(rolled - truth) * 1e4
        # 0.33 V amplitude at a random phase: 0.23 V rms; 100 readings
        # take it to about 0.023 V, and the drift is followed.
        self.assertAlmostEqual(scatter_pbp, 0.33 / np.sqrt(2), delta=0.02)
        self.assertLess(scatter_rolled, 0.04)
        self.assertLess(abs(np.mean(rolled - truth)) * 1e4, 0.01)

    def test_One_Reading_Per_Bunch_Not_Per_Hit(self):
        # A window wider than the run: every hit gets the run's mean
        # reading, counted once per bunch however many hits it had.
        rolled, _, meta = self._Cooler_Kv(cooler_correction="rolling",
                                          cooler_window_points=3 * N_BUNCH)
        per_bunch = np.mean(self.bunch_kv)
        per_hit = np.average(self.bunch_kv, weights=self.hits)
        self.assertNotAlmostEqual(per_bunch, per_hit, places=9)
        np.testing.assert_allclose(rolled, per_bunch, rtol=0, atol=1e-11)
        self.assertAlmostEqual(meta["cooler_v"], per_bunch * 1e4, places=6)

    def test_The_Window_Is_Centred_And_Shrinks_At_The_Ends(self):
        path = os.path.join(self.tmp, "run_0003.asdf")
        kv = BASE_KV + np.arange(6) * 1e-4          # 0, 1, ... 5 V steps
        Write_Run(path, kv, [2] * 6)
        rolled, bunch, _ = self._Cooler_Kv(path, cooler_correction="rolling",
                                           cooler_window_points=3)
        steps = (rolled - BASE_KV) * 1e4
        want = [0.5, 1.0, 2.0, 3.0, 4.0, 4.5]
        np.testing.assert_allclose(steps, np.repeat(want, 2), atol=1e-9)

    def test_Unread_Readings_Are_Left_Out(self):
        path = os.path.join(self.tmp, "run_0004.asdf")
        kv = np.array([3.0, 3.0001, 0.0, 3.0003, np.nan, 3.0005])
        Write_Run(path, kv, [1] * 6)
        rolled, _, meta = self._Cooler_Kv(path, cooler_correction="rolling",
                                          cooler_window_points=100)
        good = kv[np.isfinite(kv) & (kv != 0)]
        np.testing.assert_allclose(rolled, good.mean(), atol=1e-11)
        self.assertEqual(meta["cooler_rolling"]["readings"], 4)

    def test_One_Point_Is_Point_By_Point(self):
        pbp, _, _ = self._Cooler_Kv()
        one, _, _ = self._Cooler_Kv(cooler_correction="rolling",
                                    cooler_window_points=1)
        np.testing.assert_allclose(one, pbp, rtol=0, atol=1e-12)

    def test_Run_Metadata_Records_The_Readings(self):
        _, meta_pbp = self._Prep()
        _, meta = self._Prep(cooler_correction="rolling",
                             cooler_window_points=100)
        # Point by point keeps the header's run-start snapshot.
        self.assertAlmostEqual(meta_pbp["cooler_v"], HEADER_KV * 1e4,
                               places=6)
        self.assertNotIn("cooler_rolling", meta_pbp)
        self.assertAlmostEqual(meta["cooler_v"],
                               np.mean(self.bunch_kv) * 1e4, places=6)
        self.assertAlmostEqual(meta["cooler_v_effective"], meta["cooler_v"],
                               places=9)
        rolling = meta["cooler_rolling"]
        self.assertEqual(rolling["points"], 100)
        self.assertEqual(rolling["readings"], N_BUNCH)
        self.assertAlmostEqual(rolling["scatter_v"], 0.33 / np.sqrt(2),
                               delta=0.02)

    def test_The_Default_Window_Is_100_Points(self):
        _, meta = self._Prep(cooler_correction="rolling")
        self.assertEqual(meta["cooler_rolling"]["points"], 100)

    def test_A_Window_Below_One_Point_Is_Refused(self):
        with self.assertRaises(ValueError):
            self._Prep(cooler_correction="rolling", cooler_window_points=0)

    def test_The_Merge_Takes_The_Rolling_Cooler(self):
        from gui.analysis.merge import compute_merged_spectrum
        params = {"domain": "frequency", "bin_step_mhz": None}
        cfg = dict(SOURCE_CFG, cooler_offset_v=0.0)
        pbp = compute_merged_spectrum([self.path], cfg, params)
        rolled = compute_merged_spectrum(
            [self.path], dict(cfg, cooler_correction="rolling"), params)
        self.assertAlmostEqual(pbp["per_run"][0]["cooler_v"],
                               HEADER_KV * 1e4, places=6)
        self.assertAlmostEqual(rolled["per_run"][0]["cooler_v"],
                               np.mean(self.bunch_kv) * 1e4, places=6)

    def test_Rolling_Is_A_Toggle_Off_By_Default(self):
        self.assertIn(("cooler_correction", "pbp"), UPGRADE_TOGGLES)


if __name__ == "__main__":
    unittest.main()
