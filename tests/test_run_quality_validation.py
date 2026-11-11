"""Run quality and "Validate fit" (Analysis Upgrade, card F2; WP-S5, WP-S6).

Both are Output-block options, off by default, worked out in the fit
worker and written into the fit report (so into the iteration):

* Run quality: variance/mean of each run's gated counts over its whole
  scans, per PMT (afterpulsing told apart from the ions per bunch, T1),
  ORACLE's verdict and failed checks from the run's performance report,
  the cooler scatter when the rolling cooler was used; flagged "!" when
  variance/mean is above 1.2 or a timing check failed, and shown in the
  Isotope Shifts run preview.
* Validate fit: N spectra drawn from the fit's own model, refitted the
  same way (T2), and per parameter the pulls and the coverage.

Run from the project root:
    .venv/bin/python -m pytest tests/test_run_quality_validation.py -q
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from tests.test_import_only_checked import _ensure_app
from tests.test_llh_error_correction import (
    _merged, _source_config, _voigt_config)
from tests.test_merged_fit_and_nav import _FITTER, _poly_model

STEPS = np.arange(0.0, 100.01, 5.0)
CFG = dict(mass=100.0, ref_freq=4.0e14, harmonic=2, bin_mode="Raw Voltage",
           cooler_correction="pbp", cal_order=1, x_column="bins_center",
           yerr_mode="Poisson sqrt(y+1)", xerr_mode="None",
           bin_definition="Per scan step", pmt_gate=[3, 4], tof_gate=None,
           v_gate=None, f_gate=None, noise_filter=0, ref_shift=0.0)


def Write_Scans(path, scans=30, rate=6.0, afterpulse=0.0, seed=1):
    """``scans`` sawtooth scans of 21 steps, one bunch a step, PMTs 3 and
    4 at ``rate`` a bunch; PMT 4 afterpulses with ``afterpulse``."""
    import asdf
    rng, rows = np.random.default_rng(seed), []
    for k in range(scans):
        for i, s in enumerate(STEPS):
            b = k * len(STEPS) + i
            for ch in (3, 4):
                for t in rng.uniform(51.0, 53.0, rng.poisson(rate)):
                    rows.append((b * 0.005, s, b, ch, t, 3.0))
                    if ch == 4 and rng.random() < afterpulse:
                        rows.append((b * 0.005, s, b, ch, t + 3.0, 3.0))
    cal = np.linspace(0.0, 100.0, 21)
    asdf.AsdfFile({
        "Run": 1, "CoolerVoltage": 3.0, "LaserSetpoint": 20000.0,
        "DwellTime": 0.005, "Experiment": "F2", "Date": "2026-11-11",
        "StepSize": 5.0, "BunchesPerChannel": 1,
        "ScanningRanges": [[0.0, 100.0]], "CalSet": cal,
        "CalReadback": cal / 1000.0, "raw": np.array(rows)}).write_to(path)
    return path


class Run_Quality_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="run_quality_")
        cls.clean = Write_Scans(os.path.join(cls.tmp, "run_0041.asdf"))
        cls.afterpulsing = Write_Scans(os.path.join(cls.tmp, "run_0042.asdf"),
                                       afterpulse=0.5, seed=2)
        cls.failed = os.path.join(cls.tmp, "run_0043.asdf")
        shutil.copy(cls.clean, cls.failed)
        with open(os.path.join(cls.tmp, "run_0043_performance.yaml"),
                  "w") as fh:
            fh.write("verdict: WARN\nchecks:\n  Timing:\n"
                     "  - {check: Bunch period, verdict: FAIL, message: x}\n"
                     "  Detectors:\n"
                     "  - {check: Afterpulsing, verdict: PASS, message: y}\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Fit(self, path, **output):
        from gui.analysis.fitting import _fit_single_run
        res = _fit_single_run(path, dict(CFG), _poly_model(), dict(_FITTER),
                              output_config=output)
        self.assertTrue(res.get("success"), str(res.get("error"))[:400])
        return res

    def test_Off_Adds_Nothing(self):
        res = self._Fit(self.clean)
        self.assertNotIn("run_quality", res["run_metadata"])
        self.assertNotIn("[[Run quality]]", res["report"])

    def test_A_Poisson_Run_Reads_One_And_Is_Not_Flagged(self):
        res = self._Fit(self.clean, run_quality=True)
        q = res["run_metadata"]["run_quality"]
        self.assertLess(abs(q["dispersion"] - 1.0), 3 * q["sigma"])
        self.assertEqual(set(q["pmt"]), {3, 4})
        self.assertEqual(q["afterpulsing"], [])
        self.assertFalse(q["flag"])
        self.assertIn("[[Run quality]]", res["report"])

    def test_Afterpulsing_Is_Found_And_Flagged(self):
        q = self._Fit(self.afterpulsing,
                      run_quality=True)["run_metadata"]["run_quality"]
        self.assertEqual(q["afterpulsing"], [4])
        self.assertGreater(q["dispersion"], 1.2)
        self.assertTrue(q["flag"])

    def test_A_Failed_Timing_Check_Flags_The_Run(self):
        res = self._Fit(self.failed, run_quality=True)
        q = res["run_metadata"]["run_quality"]
        self.assertEqual(q["oracle_verdict"], "WARN")
        self.assertEqual(q["failed_checks"], ["Timing: Bunch period"])
        self.assertTrue(q["flag"])
        self.assertIn("Timing: Bunch period", res["report"])


class Validate_Fit_Tests(unittest.TestCase):

    def test_Refits_From_The_Model_Give_Pulls_And_Coverage(self):
        from gui.analysis.fitting import _fit_single_run
        res = _fit_single_run(
            "merged://synthetic", _source_config(), _voigt_config(),
            dict(_FITTER), output_config={"validate_fit": 60},
            merged_data=_merged())
        self.assertTrue(res.get("success"), str(res.get("error"))[:400])
        v = res["diagnostics"]["validation"]
        self.assertEqual(v["n"], 60)
        self.assertTrue(v["params"])
        for name, p in v["params"].items():
            with self.subTest(parameter=name):
                self.assertEqual(p["n"] + v["failed"], 60)
                self.assertLess(abs(p["pull_width"] - 1.0), 0.5)
        self.assertIn("[[Validation]]", res["report"])
        # The fit's own numbers are the best fit's, not the last refit's.
        plain = _fit_single_run(
            "merged://synthetic", _source_config(), _voigt_config(),
            dict(_FITTER), output_config={}, merged_data=_merged())
        self.assertEqual(res["params_df"], plain["params_df"])
        np.testing.assert_array_equal(res["y_fit"], plain["y_fit"])


class Controls_Tests(unittest.TestCase):

    def setUp(self):
        _ensure_app()

    def test_The_Output_Block_Options_Are_Off_By_Default_And_Saved(self):
        from gui.analysis.blocks import OutputBlock
        block = OutputBlock()
        self.addCleanup(block.deleteLater)
        cfg = block.get_output_config()
        self.assertFalse(cfg["run_quality"])
        self.assertEqual(cfg["validate_fit"], 0)
        block._Run_Quality.setChecked(True)
        block._Validate_Fit.setChecked(True)
        block._Validate_N.setValue(150)
        back = OutputBlock()
        self.addCleanup(back.deleteLater)
        back.from_dict(block.to_dict())
        cfg = back.get_output_config()
        self.assertTrue(cfg["run_quality"])
        self.assertEqual(cfg["validate_fit"], 150)
        old = OutputBlock()
        self.addCleanup(old.deleteLater)
        old.from_dict({})
        self.assertEqual(old.get_output_config()["validate_fit"], 0)

    def test_The_Run_Preview_Notes_The_Quality(self):
        from gui.analysis.isotope_shift_tab import Quality_Note
        self.assertEqual(Quality_Note({"run_metadata": {}}), "")
        self.assertEqual(Quality_Note({"run_metadata": {"run_quality": {
            "dispersion": 1.4, "sigma": 0.05, "flag": True}}}),
            "   var/mean 1.40 !")
        self.assertEqual(Quality_Note({"run_metadata": {"run_quality": {
            "dispersion": 1.02, "sigma": 0.05, "flag": False}}}),
            "   var/mean 1.02")


if __name__ == "__main__":
    unittest.main()
