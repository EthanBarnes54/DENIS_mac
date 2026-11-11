"""The measured laser and the systematic perturbation hooks (Analysis
Upgrade, card A3).

* WP-D2 (DA2): with ``laser_source: 'wavemeter'`` a run's laser is the
  wavemeter's mean over its own time span, from ORACLE's monitor record
  beside it; the header setpoint stays the default.
* WP-D5 (C3): the Systematics engine's perturbations, set through
  ``_config_overrides`` like the cooler offset, each move every event by
  the analytic amount: the cooler and scan dividers by ppm, the scan axis
  by an offset and a gain, the laser by MHz at the fundamental. Zero leaves
  the run byte-identical.

The frequency checks compare clstools' per-event axis with
cls_estimations.doppler, which shares its anti-collinear convention.

Run from the project root:
    .venv/bin/python -m pytest tests/test_laser_and_perturbations.py -q
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from cls_estimations.constants import CM_TO_HZ
from cls_estimations.doppler import voltage_to_frequency_array
from gui.analysis.pipeline import (
    PERTURBATION_KEYS, UPGRADE_TOGGLES, prepare_run_data)
from tests.test_oracle_ingestion import (
    COOLER_KV, LASER, SOURCE_CFG, Write_Run)

MASS = SOURCE_CFG["mass"]
HARMONIC = SOURCE_CFG["harmonic"]
# Inside the run's span (0-8 s): a ramp of 0.004 cm-1 over 8 s, plus
# readings the mean must ignore (unread, before and after the run).
SPAN_T = np.linspace(0.0, 8.0, 81)
SPAN_WN = 20000.010 + 0.0005 * SPAN_T


def Write_Monitor(path):
    rows = [(-2.0, 19999.0), (8.0 + 2.0, 20001.0), (4.05, 0.0),
            (4.15, np.nan)] + list(zip(SPAN_T, SPAN_WN))
    with open(path, "w") as fh:
        fh.write("time,round,setpoint_v,readback_v,cooler_v,"
                 "wavenumber_cm1,bunches,dropped,skipped\n")
        for t, wn in rows:
            fh.write(f"{float(t)!r},1,0,0,29885.2,{float(wn)!r},1,0,0\n")


class Wavemeter_Laser_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="wavemeter_")
        cls.oracle = os.path.join(cls.tmp, "run_0005.asdf")
        cls.legacy = os.path.join(cls.tmp, "run_0006.asdf")
        Write_Run(cls.oracle, {"Version": "0.6.0"})
        Write_Run(cls.legacy)
        Write_Monitor(os.path.join(cls.tmp, "run_0005_monitor.csv"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Prep(self, path, **extra):
        data, _eff, meta = prepare_run_data(path, dict(SOURCE_CFG, **extra))
        return data, meta

    def test_The_Wavemeter_Mean_Over_The_Run_Is_Its_Laser(self):
        data, meta = self._Prep(self.oracle, laser_source="wavemeter")
        self.assertAlmostEqual(data.Laser_set, SPAN_WN.mean(), places=9)
        self.assertAlmostEqual(meta["laser_set"], SPAN_WN.mean(), places=9)
        wm = meta["laser_wavemeter"]
        self.assertEqual(wm["readings"], len(SPAN_T))
        self.assertAlmostEqual(wm["std_cm1"], SPAN_WN.std(), places=9)
        # Slope x the run's duration (its events span 0-8 s).
        self.assertAlmostEqual(wm["drift_cm1"], 0.0005 * 8.0, places=9)
        self.assertEqual(wm["header_cm1"], LASER)

    def test_The_Header_Setpoint_Stays_The_Default(self):
        for extra in ({}, {"laser_source": "header"}):
            data, meta = self._Prep(self.oracle, **extra)
            self.assertEqual(data.Laser_set, LASER)
            self.assertEqual(meta["laser_set"], LASER)
            self.assertNotIn("laser_wavemeter", meta)

    def test_A_Run_Without_A_Monitor_Keeps_Its_Setpoint(self):
        data, meta = self._Prep(self.legacy, laser_source="wavemeter")
        self.assertEqual(data.Laser_set, LASER)
        self.assertIsNone(meta["laser_wavemeter"])

    def test_An_Override_Still_Outranks_The_Wavemeter(self):
        data, meta = self._Prep(self.oracle, laser_source="wavemeter",
                                override_enabled=True,
                                laser_override=20000.5)
        self.assertEqual(data.Laser_set, 20000.5)
        own = {self.oracle: {"laser_sp": 20000.25}}
        data, meta = self._Prep(self.oracle, laser_source="wavemeter",
                                run_overrides=own)
        self.assertEqual(data.Laser_set, 20000.25)

    def test_The_Merge_Takes_The_Wavemeter_Too(self):
        from gui.analysis.merge import compute_merged_spectrum
        params = {"domain": "frequency", "bin_step_mhz": None}
        cfg = dict(SOURCE_CFG, cooler_offset_v=0.0, laser_source="wavemeter")
        merged = compute_merged_spectrum([self.oracle], cfg, params)
        self.assertAlmostEqual(merged["per_run"][0]["laser_set"],
                               SPAN_WN.mean(), places=9)


class Perturbation_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="perturbations_")
        cls.path = os.path.join(cls.tmp, "run_0007.asdf")
        Write_Run(cls.path)
        cls.base = cls._Events(cls)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Events(self, **extra):
        data, _eff, meta = prepare_run_data(self.path,
                                            dict(SOURCE_CFG, **extra))
        frame = data.Sorted
        return (frame["V"].to_numpy(), frame["DV_cal"].to_numpy(),
                frame["F"].to_numpy(), meta)

    def _Assert_Moves(self, delta_v, laser=LASER, **extra):
        """Every event's voltage moves by ``delta_v`` and its frequency by
        what cls_estimations.doppler says that voltage (and laser) gives."""
        v0, _, f0, _ = self.base
        v1, _, f1, meta = self._Events(**extra)
        np.testing.assert_allclose(v1 - v0, delta_v, rtol=1e-9, atol=1e-9)

        def Doppler_Mhz(v, laser_cm1):
            return voltage_to_frequency_array(-v, 0.0, laser_cm1, MASS,
                                              HARMONIC)
        want = Doppler_Mhz(v1, laser) - Doppler_Mhz(v0, LASER)
        # atol 1 kHz: both sides take beta from 1 - E0^2/Etot^2, which at
        # 30 kV leaves ~100 Hz of rounding in every frequency.
        np.testing.assert_allclose((f1 - f0) / 1e6, want, rtol=1e-5,
                                   atol=1e-3)
        self.assertEqual(meta["perturbations"], extra)
        return want

    def test_The_Cooler_Divider_Scales_The_Cooler(self):
        want = self._Assert_Moves(COOLER_KV * 1e4 * 100e-6,
                                  cooler_div_ppm=100.0)
        self.assertTrue(np.all(want > 0))        # more energy, higher f

    def test_The_Scan_Divider_Scales_The_Scan_Voltage(self):
        _, cal, _, _ = self.base
        self._Assert_Moves(-cal * 1000e-6, scan_div_ppm=1000.0)

    def test_The_Scan_Axis_Gain_Scales_The_Scan_Voltage(self):
        _, cal, _, _ = self.base
        self._Assert_Moves(-cal * 1000e-6, scan_axis_gain_ppm=1000.0)

    def test_The_Scan_Axis_Offset_Shifts_The_Scan_Voltage(self):
        self._Assert_Moves(-1.0, scan_axis_offset_v=1.0)

    def test_The_Laser_Offset_Is_Mhz_At_The_Fundamental(self):
        laser = LASER + 10.0e6 / CM_TO_HZ
        want = self._Assert_Moves(0.0, laser=laser, laser_offset_mhz=10.0)
        # Harmonic x 10 MHz, Doppler-scaled by ~1 + beta at 30 kV.
        np.testing.assert_allclose(want, HARMONIC * 10.0, rtol=2e-3)

    def test_Zero_Perturbations_Leave_The_Run_Byte_Identical(self):
        v0, cal0, f0, m0 = self.base
        v1, cal1, f1, m1 = self._Events(**dict.fromkeys(PERTURBATION_KEYS,
                                                        0.0))
        for a, b in ((v0, v1), (cal0, cal1), (f0, f1)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(m0, m1)
        self.assertNotIn("perturbations", m0)

    def test_The_Keys_Are_The_Contract_s(self):
        self.assertEqual(set(PERTURBATION_KEYS), {
            "cooler_div_ppm", "scan_div_ppm", "scan_axis_offset_v",
            "scan_axis_gain_ppm", "laser_offset_mhz"})
        self.assertIn(("laser_source", "header"), UPGRADE_TOGGLES)


if __name__ == "__main__":
    unittest.main()
