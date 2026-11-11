"""The per-PMT dead-time gate (Analysis Upgrade, card A4; WP-S0, D-3).

A count is dropped when the previous count on the same PMT, in the same
bunch, came no more than that PMT's dead time before it (ORACLE's own
definition, cls_acquisition/detectors.py). Off by default and never
applied by itself: ORACLE's suggestion is only read. The share removed is
reported per PMT, with the share of real photons it takes at the
strongest step, estimated from the other PMTs: the light reaches every PMT
alike, afterpulses only their own.

Run from the project root:
    .venv/bin/python -m pytest tests/test_dead_time_gate.py -q
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from gui.analysis.pipeline import (
    UPGRADE_TOGGLES, Suggested_Dead_Times, prepare_run_data)
from tests.test_oracle_ingestion import SOURCE_CFG

CFG = dict(SOURCE_CFG, bin_mode="Raw Voltage", pmt_gate=[1, 2, 3, 4])


def Write_Events(path, rows):
    """A run of the given (dv, bunch, channel, tof) events."""
    import asdf
    rows = np.asarray(rows, dtype=float)
    ts = rows[:, 1] * 0.005 + rows[:, 3] * 1e-6
    raw = np.column_stack([ts, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3],
                           np.full(len(rows), 3.0)])
    cal = np.linspace(0.0, 100.0, 21)
    asdf.AsdfFile({
        "Run": 1, "CoolerVoltage": 3.0, "LaserSetpoint": 20000.0,
        "DwellTime": 0.005, "Experiment": "A4", "Date": "2026-11-11",
        "StepSize": 1.0, "ScanningRanges": [[0.0, 100.0]], "CalSet": cal,
        "CalReadback": cal / 1000.0, "raw": raw[np.argsort(ts)]}).write_to(path)
    return path


def Bunched(rng, bunches=4000, rate=1.5, afterpulse=0.0):
    """Every PMT sees Poisson(rate) photons a bunch within 2 us at step 32
    (the strongest), a tenth of that at steps 31 and 33; PMT 3 afterpulses
    3 us after a count with probability ``afterpulse``."""
    rows = []
    for k in range(bunches):
        dv = 32.0 if k % 3 == 0 else 31.0 + 2 * (k % 2)
        lam = rate if dv == 32.0 else rate / 10
        for ch in (1, 2, 3, 4):
            for t in rng.uniform(51.0, 53.0, rng.poisson(lam)):
                rows.append((dv, k, ch, t))
                if ch == 3 and rng.random() < afterpulse:
                    rows.append((dv, k, ch, t + 3.0))
    return rows


class Dead_Time_Gate_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dead_time_")
        cls.hand = Write_Events(os.path.join(cls.tmp, "run_0021.asdf"), [
            (10, 0, 3, 50.0), (10, 0, 3, 53.0),      # 3 us: gone at 10 us
            (10, 0, 3, 70.0),                        # 17 us after: kept
            (10, 0, 4, 50.5), (10, 0, 4, 52.0),      # PMT 4 has no dead time
            (10, 1, 3, 51.0)])                       # a new bunch: kept
        rng = np.random.default_rng(1)
        cls.clean = Write_Events(os.path.join(cls.tmp, "run_0022.asdf"),
                                 Bunched(rng))
        cls.afterpulsing = Write_Events(os.path.join(cls.tmp, "run_0023.asdf"),
                                        Bunched(rng, afterpulse=0.3))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Prep(self, path, **extra):
        data, _eff, meta = prepare_run_data(path, dict(CFG, **extra))
        return data.Sorted, meta

    def test_A_Count_Within_Its_PMT_s_Dead_Time_Is_Dropped(self):
        events, meta = self._Prep(self.hand, dead_time_us={3: 10.0})
        kept = sorted(zip(events["Bunch"], events["TDC"], events["TOF"]))
        self.assertEqual(kept, [(0, 3, 50.0), (0, 3, 70.0), (0, 4, 50.5),
                                (0, 4, 52.0), (1, 3, 51.0)])
        three = meta["dead_time"][3]
        self.assertEqual(three["dead_time_us"], 10.0)
        self.assertEqual(three["removed"], 0.25)
        self.assertNotIn(4, meta["dead_time"])
        self.assertEqual(meta["n_events"], 6)        # what the file held

    def test_Off_Is_Byte_Identical(self):
        base, meta = self._Prep(self.hand)
        for off in ({}, None, {3: 0.0}):
            with self.subTest(dead_time_us=off):
                events, m = self._Prep(self.hand, dead_time_us=off)
                np.testing.assert_array_equal(events.to_numpy(),
                                              base.to_numpy())
                self.assertEqual(m, meta)

    def test_The_Other_PMTs_Estimate_The_Real_Photons_Lost(self):
        _events, meta = self._Prep(self.clean, dead_time_us={3: 10.0})
        three = meta["dead_time"][3]
        # 1.5 photons in 2 us: 1 - (1 - e^-1.5)/1.5 of them follow one.
        want = 1 - (1 - np.exp(-1.5)) / 1.5
        n = 4000 / 3 * 1.5
        self.assertLess(abs(three["removed_strongest"] - want),
                        3 * np.sqrt(want * (1 - want) / n))
        self.assertLess(abs(three["real_loss_strongest"]
                            - three["removed_strongest"]),
                        4 * np.sqrt(want * (1 - want) / n))

    def test_Afterpulses_Do_Not_Bias_The_Estimate(self):
        p, want = 0.3, 1 - (1 - np.exp(-1.5)) / 1.5
        _events, meta = self._Prep(self.afterpulsing,
                                   dead_time_us={1: 10.0, 3: 10.0})
        one, three = meta["dead_time"][1], meta["dead_time"][3]
        # Every afterpulse goes (3 us after its photon) and the real photons
        # as in a clean run: (want + p) / (1 + p) of PMT 3's counts.
        self.assertAlmostEqual(three["removed_strongest"],
                               (want + p) / (1 + p), delta=0.02)
        # The share of its real photons lost, from the other PMTs alone,
        # is the clean run's; PMT 1's reads PMT 3's afterpulses a little.
        self.assertAlmostEqual(three["real_loss_strongest"], want, delta=0.02)
        self.assertAlmostEqual(one["real_loss_strongest"], want, delta=0.03)

    def test_The_Merge_Takes_The_Gate_Too(self):
        from gui.analysis.merge import compute_merged_spectrum
        params = {"domain": "voltage", "bin_step_mhz": None}
        cfg = dict(CFG, cooler_offset_v=0.0)
        full = compute_merged_spectrum([self.hand], cfg, params)
        gated = compute_merged_spectrum(
            [self.hand], dict(cfg, dead_time_us={3: 10.0}), params)
        self.assertEqual(np.sum(full["y"]) - np.sum(gated["y"]), 1)

    def test_ORACLE_s_Suggestion_Is_Read_Not_Applied(self):
        path = os.path.join(self.tmp, "run_0024.asdf")
        shutil.copy(self.hand, path)
        with open(os.path.join(self.tmp, "run_0024_performance.yaml"),
                  "w") as fh:
            fh.write("detectors:\n"
                     "  channel 3: {counts: 9, suggested_dead_time_us: 10.0}\n"
                     "  channel 4: {counts: 9, suggested_dead_time_us: null}\n")
        self.assertEqual(Suggested_Dead_Times(path), {3: 10.0})
        self.assertEqual(Suggested_Dead_Times(self.hand), {})
        events, meta = self._Prep(path)
        self.assertEqual(len(events), 6)
        self.assertNotIn("dead_time", meta)

    def test_The_Gate_Is_A_Toggle_Off_By_Default(self):
        self.assertIn(("dead_time_us", None), UPGRADE_TOGGLES)


if __name__ == "__main__":
    unittest.main()
