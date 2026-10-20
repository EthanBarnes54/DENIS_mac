"""Third external review (2026-10-08): fixes, with numerical checks.

  1  Pre-Analysis merges imported into Analysis: a frequency merge keeps
     its detuning (it was shifted ~1.1e9 MHz), a voltage merge keeps its
     calibrated axis and spread; the PA plot uses the fit's conversion.
  2  Cooler calibration: walk outward with seeds, leave out (and show)
     bound-pinned / error-less / non-converged fits and trend outliers;
     fit_line drops points without an error.
  3  Correction uncertainty follows what each fit recorded, not the
     "apply at fit time" box; no GP loaded -> the recorded sigma.
  4  A cooler override + offset <= 0 is refused, not swapped for the
     per-event readback.
  5  NIST schemes carry their species; a whole session without NIST
     state clears the previous session's; 15: numeric columns sort.
  6  Common grid keeps "None" errors as unit weights.
  7  A tied centroid is one measurement carrying every run's GP
     correction (numbers checked against the hand calculation).
  8-10 Systematic scans: no shifts from an incomplete step, a failed
     project forgets the last step, baselines over the used runs and
     steps, an empty shift tab stays empty; no mean across isotopes.
  11 Auto-Fitter shares by fit mode; 12 its time limit fires and Stop
     is immediate.
  +  Tied reference runs train the GP once; fits export lmfit's verdict.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_review_2026_10_08_round3.py -q
"""
import contextlib
import multiprocessing
import os
import threading
import time
import types
import unittest
import warnings
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMessageBox, QTableWidget)

satlas2 = __import__("pytest").importorskip("satlas2")

_APP = QApplication.instance() or QApplication([])

E_LO, E_UP, HARM = 557.1341, 37702.3054, 2


def _transition_mhz():
    from cls_estimations.constants import C_LIGHT
    return (E_UP - E_LO) * C_LIGHT * 100.0 / 1e6


def _p(value, lo=-1e12, hi=1e12, vary=True):
    return {"value": value, "vary": vary, "min": lo, "max": hi, "expr": ""}


# ── 1: merged import ─────────────────────────────────────────────────


class _Spin:
    def __init__(self, v):
        self._v = v

    def value(self):
        return self._v


class MergedImportTests(unittest.TestCase):

    def _mfe(self, x, domain, **kw):
        from gui.preanalysis_tab import MergedFileEntry
        e = MergedFileEntry("m", x, [4.0, 9.0, 5.0], domain, [], **kw)
        e.check.setChecked(True)
        self.addCleanup(e.deleteLater)
        return e

    def _import(self, mfe):
        from gui.analysis.blocks import SourceBlock
        sb = SourceBlock()
        self.addCleanup(sb.deleteLater)
        pa = types.SimpleNamespace(_e_lower=_Spin(E_LO), _e_upper=_Spin(E_UP),
                                   _harmonic=_Spin(HARM),
                                   _file_entries=[mfe])
        sb._import_files_from_preanalysis(pa)
        return [e["merged_data"] for e in sb._file_entries
                if e.get("is_merged")][0]

    def test_a_frequency_merge_keeps_its_detuning(self):
        md = self._import(self._mfe([-10.0, 0.0, 10.0], "frequency"))
        np.testing.assert_allclose(md["x"], [-10.0, 0.0, 10.0])
        self.assertEqual(md["merge_params"]["domain"], "frequency")
        np.testing.assert_allclose(md["yerr"], np.sqrt([5.0, 10.0, 6.0]))

    def test_a_legacy_absolute_merge_is_converted(self):
        t = _transition_mhz()
        md = self._import(self._mfe([t - 10.0, t, t + 10.0], "frequency"))
        np.testing.assert_allclose(md["x"], [-10.0, 0.0, 10.0], atol=1e-3)

    def test_a_voltage_merge_keeps_its_calibration(self):
        xcal = [1127.4, 1128.4, 1129.4]
        md = self._import(self._mfe([1130.0, 1131.0, 1132.0], "voltage",
                                    merged_x_cal=xcal,
                                    merged_cal_spread_v=2.04))
        np.testing.assert_allclose(md["x_cal"], xcal)
        self.assertEqual(md["cal_spread_v"], 2.04)
        self.assertEqual(md["x_unit"], "V")

    def test_the_pa_plot_uses_the_fits_conversion(self):
        from cls_estimations.constants import C_LIGHT
        from gui.analysis.merge import project_voltage_merge_to_frequency
        from gui.preanalysis_tab import PreAnalysisTab
        pa = PreAnalysisTab()
        self.addCleanup(pa.deleteLater)
        pa._e_lower.setValue(E_LO)
        pa._e_upper.setValue(E_UP)
        pa._harmonic.setValue(HARM)
        x = np.array([1130.0, 1131.0, 1132.0])
        xcal = x - 2.6
        meta = dict(merge_cooler_v=29907.412, merge_laser_sp=18555.15,
                    merge_mass_amu=69.924249, merge_harmonic=HARM)
        pa._add_merged_entry("vm", x, [4.0, 9.0, 5.0], "voltage", [],
                             merged_x_cal=xcal, **meta)

        def plotted(mode):
            pa._xaxis_combo.setCurrentIndex(pa._xaxis_combo.findText(mode))
            pa._replot()
            line = [ln for ln in pa._ax.get_lines()
                    if ln.get_label() == "vm"][0]
            return np.asarray(line.get_xdata(), dtype=float)

        np.testing.assert_allclose(plotted("Calibrated voltage"), xcal)
        mfe = [e for e in pa._file_entries if e.run_number == "vm"][0]
        want = project_voltage_merge_to_frequency(
            mfe.to_merged_data(),
            {"ref_freq": (E_UP - E_LO) * C_LIGHT * 100.0})[0]
        np.testing.assert_allclose(plotted("Frequency"), want)


# ── 4: cooler plan ───────────────────────────────────────────────────


class CoolerPlanTests(unittest.TestCase):

    def test_the_plan(self):
        from gui.analysis.pipeline import resolve_cooler
        for override, offset in ((5000, -6000), (5000, -5000)):
            with self.assertRaises(ValueError):
                resolve_cooler(override, offset, 30000)
        p = resolve_cooler(5000, -100, 30000)
        self.assertEqual((p.mode, p.constant), ("constant", 4900.0))
        p = resolve_cooler(0, -24.42, 29900)
        self.assertEqual((p.mode, p.constant), ("readback", None))
        self.assertAlmostEqual(p.effective, 29875.58)
        with self.assertRaises(ValueError):
            resolve_cooler(0, -31000, 30000)

    def test_the_fit_refuses_an_override_driven_below_zero(self):
        from tests.test_review_2026_10_08_round2 import CoolerOffsetTests
        with self.assertRaises(ValueError):
            CoolerOffsetTests._prepare(self, cooler_offset_v=-6000.0,
                                       override_enabled=True,
                                       cooler_override=5000.0)

    def test_the_preview_refuses_it_too(self):
        from gui.analysis import blocks
        reached = []

        class _D:
            VCoolDiv, VCoolOffset, Vcool_init = 10000, 0, 3.0

            def Compute_Voltages(self, cooler_correction="pbp"):
                reached.append(True)
                raise RuntimeError("stop")

        sb = blocks.SourceBlock()
        self.addCleanup(sb.deleteLater)
        sb._ovr_enable.setChecked(True)
        sb._cooler_spin.setValue(5000.0)
        sb._cooler_offset.setValue(-6000.0)
        with mock.patch.object(sb, "get_checked_files",
                               return_value=[{"path": "run_1.asdf"}]), \
                mock.patch.object(blocks, "_get_clstools",
                                  return_value=types.SimpleNamespace(
                                      CLSDataFrame=_D)), \
                mock.patch("gui.calibration.load_run_calibrated"), \
                mock.patch.object(blocks, "QMessageBox") as box:
            sb._show_preview_at_index()
        self.assertEqual(reached, [])
        self.assertIn("not a beam energy",
                      str(box.critical.call_args[0][2]))


# ── 2: cooler calibration ────────────────────────────────────────────


def _synthetic_points(extra=()):
    """Two isotopes whose lines cross zero at -25 V, 15 offsets each."""
    from cls_estimations.cooler_calibration import LITERATURE_A, ScanPoint
    pts = []
    for iso, b in (("171Yb", -0.21), ("173Yb", 0.059)):
        for dv in np.linspace(-50, 50, 15):
            pts.append(ScanPoint(run="1", isotope=iso, dv=float(dv),
                                 value=LITERATURE_A[iso] + b * (dv + 25.0),
                                 sigma=2.0))
    return pts + list(extra)


class CalibrationTests(unittest.TestCase):

    @staticmethod
    def _res(value, stderr, *, lo=-20000.0, hi=20000.0, msg="",
             converged=True):
        return {"success": True, "source_name": "Run_1",
                "converged": converged, "fit_message": msg,
                "params_df": {"Source": ["Run_1"], "Model": ["M"],
                              "Parameter": ["Al"], "Value": [value],
                              "Stderr": [stderr], "Minimum": [lo],
                              "Maximum": [hi], "Vary": [True]}}

    def test_untrustworthy_fits_are_named(self):
        from gui.analysis.cooler_scan import fit_point_problem as why
        self.assertEqual(why(self._res(12640.0, 2.0), "Al", 12640.0, 2.0),
                         "")
        self.assertIn("upper bound", why(self._res(20000.0, 3.6e-5), "Al",
                                         20000.0, 3.6e-5))
        self.assertIn("no error bar", why(self._res(12640.0, None), "Al",
                                          12640.0, float("nan")))
        self.assertIn("did not converge", why(
            self._res(12640.0, 2.0, msg="Fit aborted: number of function "
                                        "evaluations > 20000"),
            "Al", 12640.0, 2.0))
        self.assertIn("did not converge", why(
            self._res(12640.0, 2.0, converged=False), "Al", 12640.0, 2.0))

    def test_a_line_ignores_points_without_an_error(self):
        from cls_estimations.cooler_calibration import fit_line
        line = fit_line([0, 1, 2, 3], [0, 1, 2, 100], [1, 1, 1, 0])
        self.assertAlmostEqual(line.slope, 1.0)
        self.assertEqual(line.n, 3)

    def test_a_flagged_point_no_longer_moves_the_calibration(self):
        from dataclasses import replace
        from cls_estimations.cooler_calibration import calibrate
        pinned = replace(_synthetic_points()[0], value=20000.0,
                         sigma=3.6e-5)
        res = calibrate(_synthetic_points([replace(pinned, flag="at its "
                                                   "upper bound")]))
        self.assertAlmostEqual(res.dv, -25.0, places=6)
        self.assertEqual([why for _p, why in res.excluded],
                         ["at its upper bound"])
        self.assertIn("left out", res.note)

    def test_a_wrong_minimum_is_left_out(self):
        from dataclasses import replace
        from cls_estimations.cooler_calibration import calibrate
        bad = replace(_synthetic_points()[3], value=_synthetic_points()[3]
                      .value + 600.0)
        res = calibrate(_synthetic_points([bad]))
        self.assertAlmostEqual(res.dv, -25.0, places=6)
        self.assertIn("sigma off", res.excluded[0][1])

    def test_only_accepted_fits_seed_the_next_offset(self):
        from gui.analysis.cooler_scan import CoolerScan
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        proj = types.SimpleNamespace(project_name="P")
        scan._parameter = "Al"
        scan._collect(proj, "171Yb", -50.0, [self._res(20000.0, 3.6e-5)])
        self.assertEqual(scan._history, {})
        self.assertTrue(scan._points[0].flag)
        scan._collect(proj, "171Yb", 0.0, [self._res(12637.0, 2.0)])
        self.assertEqual(list(scan._history.values()),
                         [{0.0: (12637.0, 2.0)}])

    def test_seeds_extrapolate_along_the_walk(self):
        from gui.analysis import systematic_scan as ss
        from gui.analysis.cooler_scan import CoolerScan
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        scan._history = {ss.value_key("P", "Run_1", "M", "centroid"):
                         {0.0: (628.7, 0.8), -7.0: (570.2, 0.8)}}
        model = [{"type": "HFS", "name": "M",
                  "params": {"centroid": _p(628.7)}}]
        with mock.patch.object(ss, "checked_entries_of",
                               return_value=[("run_1.asdf", "Run_1")]), \
                mock.patch.object(ss, "model_configs_of",
                                  return_value=model):
            out = scan._seed_map(types.SimpleNamespace(project_name="P"),
                                 -14.0)
        self.assertAlmostEqual(
            out["run_1.asdf"][0]["params"]["centroid"]["value"], 511.7)

    def test_seeds_and_the_no_dialog_flag_reach_the_fit(self):
        from tests.test_cooler_scan import _StubProject, _spin
        from gui.analysis.cooler_scan import CoolerScan
        p = _StubProject("171", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        seen = []
        orig = p._on_fit_requested

        def spy():
            seen.append((dict(p._config_overrides), p._scan_driven))
            orig()
        p._on_fit_requested = spy
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        scan._seed_map = lambda project, dv: {"run.asdf": [{"dv": dv}]}
        scan.start([(p, "171Yb")], [0.0, 10.0])
        _spin(scan, timeout_ms=15000)
        self.assertEqual([o["model_configs_map"] for o, _d in seen],
                         [{"run.asdf": [{"dv": 0.0}]},
                          {"run.asdf": [{"dv": 10.0}]}])
        self.assertTrue(all(d for _o, d in seen))
        self.assertFalse(p._scan_driven)

    def test_the_scan_walks_outward_from_the_fitted_offset(self):
        from tests.test_cooler_scan import _StubProject, _spin
        from gui.analysis.cooler_scan import CoolerScan
        p = _StubProject("171", ["7547"], -0.21, 12636.0)
        self.addCleanup(p.deleteLater)
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        with mock.patch("gui.analysis.cooler_scan._fitted_offset",
                        return_value=-30.0):
            scan.start([(p, "171Yb")], [-50.0, -30.0, 0.0, 50.0])
            _spin(scan, timeout_ms=15000)
        self.assertEqual([dv for dv, _o in p.seen],
                         [-30.0, -50.0, 0.0, 50.0])

    def test_fits_export_lmfits_verdict(self):
        from tests.test_review_2026_10_08_round2 import _sim_fit
        r = _sim_fit({}, {"mA": (100.0, 20.0)})[0]
        self.assertIn("converged", r)
        self.assertIn("errorbars", r)


# ── 3 + 7: isotope shifts ────────────────────────────────────────────


class _StubGP:
    """Drift 10 MHz/h, posterior sd 5 MHz, correlation length 2 h."""
    is_fit = True
    _t0 = 100.0

    @staticmethod
    def g(t):
        return -150.0 + 10.0 * (np.asarray(t, dtype=float) - 100.0)

    def predict(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.g(t), np.full(t.shape, 5.0)

    def predict_interval(self, t1, t2, n=32):
        return float(self.g(0.5 * (t1 + max(t1, t2)))), 5.0

    def cov(self, a, b):
        a = np.atleast_1d(np.asarray(a, dtype=float))[:, None]
        b = np.atleast_1d(np.asarray(b, dtype=float))[None, :]
        return 25.0 * np.exp(-0.5 * ((a - b) / 2.0) ** 2)

    def cov_spans(self, sa, sb, n=32):
        return self.cov(np.atleast_2d(sa).mean(axis=1),
                        np.atleast_2d(sb).mean(axis=1))


def _result(centroid, t_hours, *, run, err=3.0, dur=0.1, applied=None,
            source=None, expr=None, n_events=1000):
    meta = {"ts_start": 3600.0 * t_hours,
            "ts_stop": 3600.0 * (t_hours + dur), "n_events": n_events}
    if applied is not None:
        meta.update({"centroid_correction_applied": True,
                     "centroid_correction_mhz": float(applied),
                     "centroid_correction_sigma_mhz": 5.0,
                     "centroid_correction_mode": "Auto"})
    src = source or f"Run_{run}"
    pdf = {"Source": [src], "Model": ["Model_1"], "Parameter": ["centroid"],
           "Value": [centroid], "Stderr": [err],
           "Expression": [expr if expr else float("nan")]}
    return {"success": True, "run_number": str(run), "run_file": "",
            "source_name": src, "params_df": pdf, "run_metadata": meta}


GP = _StubGP()
TA, TB, DUR = 1.0, 5.0, 0.1
GA, GB = float(GP.g(TA + DUR / 2)), float(GP.g(TB + DUR / 2))


def _shifts(ref, smp, *, ref_mode, gp=GP, apply_global=True,
            sample_run="Weighted Average"):
    from gui.analysis.tab import AnalysisTab
    at = AnalysisTab()
    r = at._add_project("74Ge", is_reference=True)
    s = at._add_project("70Ge")
    r._last_results, s._last_results = ref, smp
    ist = at._is_tab
    ist._ref_corr_panel._corrector = gp
    ist._ref_corr_panel._apply_global_cb.setChecked(apply_global)
    ist._refresh_projects()
    rows = {e.project_combo.currentText(): e for e in ist._entries}
    for name, a in (("74Ge", 74), ("70Ge", 70)):
        if name not in rows:
            ist._add_row(project_name=name, label=name, A=a)
    rows = {e.project_combo.currentText(): e for e in ist._entries}
    rows["74Ge"].ref_radio.setChecked(True)
    rows["74Ge"].run_combo.setCurrentText(ref_mode)
    rows["70Ge"].run_combo.setCurrentText(sample_run)
    for e in ist._entries:
        e.select_cb.setChecked(True)
    data = {d["label"]: d for d in ist.shifts_for_scan()}
    cell = ist._row_centroid(rows["70Ge"], ref_mode != "Weighted Average"
                             and True)
    at.deleteLater()
    return data, cell


def _refs(applied):
    out = []
    for i, t in enumerate((0.5, 2.0, 3.5, 4.5, 5.5)):
        g = float(GP.g(t + DUR / 2))
        out.append(_result(10.0 if applied else 10.0 + g, t, run=7900 + i,
                           err=2.0, applied=g if applied else None))
    return out


def _tied_sample(c, *, applied, nA=1000, nB=1000, tied=True):
    a = _result(c, TA, run=8001, n_events=nA,
                applied=GA if applied else None)
    b = _result(c, TB, run=8002, n_events=nB,
                applied=GB if applied else None,
                expr="Run_8001___Model_1___centroid" if tied else None)
    return [a, b]


class TieGroupTests(unittest.TestCase):

    def test_groups_follow_chains_and_a_missing_head(self):
        from cls_estimations.isotope_shift import tie_groups
        a = _result(1.0, 1, run=1)
        b = _result(1.0, 2, run=2, expr="Run_1___Model_1___centroid")
        c = _result(1.0, 3, run=3, expr="Run_2___Model_1___centroid")
        d = _result(1.0, 4, run=4)
        groups = tie_groups([c, d, b, a])
        self.assertEqual([[r["run_number"] for r in g] for g in groups],
                         [["1", "3", "2"], ["4"]])
        # Two runs tied to a source whose fit failed (not in the list):
        # still ONE parameter, so one group.
        e = _result(1.0, 5, run=5, expr="Run_1___Model_1___centroid")
        orphans = tie_groups([b, e])
        self.assertEqual([[r["run_number"] for r in g] for g in orphans],
                         [["2", "5"]])

    def test_on_the_fly_gp_takes_every_runs_correction(self):
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        for nA, nB, want_sigma in ((1000, 1000, 3.7672), (3000, 1000,
                                                           4.1102)):
            vA, vB = nA / (nA + nB), nB / (nA + nB)
            c_raw = -85.0 + vA * GA + vB * GB
            data, _cell = _shifts(_refs(False),
                                  _tied_sample(c_raw, applied=False, nA=nA,
                                               nB=nB),
                                  ref_mode=GP_REFERENCE)
            row = data["70Ge"]
            self.assertAlmostEqual(row["centroid"], -85.0, places=6)
            self.assertAlmostEqual(row["sigma_fit"], 3.0, places=6)
            self.assertAlmostEqual(row["sigma_correction"], want_sigma,
                                   places=3)

    def test_fit_time_corrections_propagate_over_every_run(self):
        data, _cell = _shifts(_refs(True), _tied_sample(-75.0, applied=True),
                              ref_mode="Weighted Average")
        self.assertAlmostEqual(data["70Ge"]["sigma_delta_nu"], 3.2551,
                               places=3)
        self.assertAlmostEqual(data["70Ge"]["cov_with_reference_mhz2"],
                               14.257, places=2)

    def test_a_pinned_tied_run_is_its_whole_group(self):
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        c_raw = -85.0 + 0.5 * (GA + GB)
        data, _cell = _shifts(_refs(False),
                              _tied_sample(c_raw, applied=False),
                              ref_mode=GP_REFERENCE, sample_run="8002")
        self.assertAlmostEqual(data["70Ge"]["centroid"], -85.0, places=6)

    def test_the_row_cell_agrees(self):
        from gui.analysis.isotope_shift_tab import GP_REFERENCE
        c_raw = -85.0 + 0.75 * GA + 0.25 * GB      # counts 3:1
        _data, cell = _shifts(_refs(False),
                              _tied_sample(c_raw, applied=False, nA=3000),
                              ref_mode=GP_REFERENCE)
        self.assertEqual(cell[:2], ("-85.0000", "3.0000"))


class PropagationTests(unittest.TestCase):

    def _sample(self):
        return [_result(-75.0, t, run=8000 + i, err=3.0,
                        applied=float(GP.g(t + DUR / 2)))
                for i, t in enumerate((2.0, 2.2))]

    def test_the_apply_box_does_not_touch_recorded_corrections(self):
        ref = [_result(10.0, t, run=7900 + i, err=2.0)
               for i, t in enumerate((1.0, 1.5, 2.5, 3.0))]
        on, _ = _shifts(ref, self._sample(), ref_mode="Weighted Average")
        off, _ = _shifts(ref, self._sample(), ref_mode="Weighted Average",
                         apply_global=False)
        self.assertGreater(on["70Ge"]["sigma_correction"], 4.0)
        self.assertAlmostEqual(off["70Ge"]["sigma_correction"],
                               on["70Ge"]["sigma_correction"])

    def test_no_gp_loaded_keeps_the_recorded_sigma(self):
        ref = [_result(10.0, t, run=7900 + i, err=2.0)
               for i, t in enumerate((1.0, 1.5, 2.5, 3.0))]
        data, _ = _shifts(ref, self._sample(), ref_mode="Weighted Average",
                          gp=None)
        # Fully correlated recorded sigmas (5 MHz each): 5 MHz.
        self.assertAlmostEqual(data["70Ge"]["sigma_correction"], 5.0,
                               places=6)


class TiedReferenceObservationTests(unittest.TestCase):

    def test_one_gp_observation_per_tied_centroid(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("74Ge")
        self.addCleanup(p.deleteLater)
        p._is_reference = True
        p._last_results = [
            _result(10.0, 1.0, run=1),
            _result(10.0, 2.0, run=2, expr="Run_1___Model_1___centroid"),
            _result(12.0, 3.0, run=3)]
        obs = p.get_reference_observations()
        self.assertEqual([o["run_number"] for o in obs], ["1", "3"])


# ── 8-10: systematic scans ───────────────────────────────────────────


class ScanTests(unittest.TestCase):

    def test_no_shifts_from_an_incomplete_step(self):
        from gui.analysis.systematic_scan import SystematicScan, new_record
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan._is_tab = mock.Mock()
        rec = new_record(10.0)
        rec["failed_runs"] = ["70Ge/8002"]
        scan._records = {10.0: rec}
        scan._later = lambda: None
        scan._collect_shifts(10.0)
        self.assertEqual(rec["shifts"], [])
        self.assertEqual(rec["values"], {})
        scan._is_tab.shifts_for_scan.assert_not_called()

    def test_a_pinned_value_never_seeds_the_next_step(self):
        from gui.analysis.systematic_scan import (
            ScanTarget, SystematicScan, new_record)
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        rec = new_record(5.0)
        rec["status"] = "ok"
        tgt = ScanTarget(project=types.SimpleNamespace(project_name="P"))
        scan._collect_params(tgt, rec, "Run_1", {
            "Parameter": ["Al", "centroid"], "Value": [20000.0, 600.0],
            "Stderr": [3.6e-5, 0.8], "Model": ["M", "M"],
            "Source": ["Run_1", "Run_1"], "Minimum": [-2e4, -1e12],
            "Maximum": [2e4, 1e12]})
        scan._history = {}
        scan._remember(rec)
        self.assertEqual(sorted(k.split("|")[3] for k in scan._history),
                         ["centroid"])

    def test_a_failed_project_forgets_the_last_step(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("70Ge")
        self.addCleanup(p.deleteLater)
        p._last_results = [{"success": True, "run_number": "1"}]
        p._scan_driven = True
        p._on_fit_error("pool died", mock.Mock())
        self.assertEqual(p._last_results, [])

    def test_the_baseline_is_over_the_runs_the_points_use(self):
        from cls_estimations.systematics import (
            ParameterBand, ScanValue, mean_over_runs)
        a = ParameterBand("70Ge", "Run_A", "centroid",
                          [ScanValue(dv, 0.0, 1.0)
                           for dv in (-10.0, 0.0, 10.0)],
                          baseline=0.0, baseline_sigma=1.0)
        b = ParameterBand("70Ge", "Run_B", "centroid",
                          [ScanValue(-10.0, 100.0, 1.0, include=False)],
                          baseline=100.0, baseline_sigma=1.0)
        m = mean_over_runs([a, b])
        self.assertEqual((m.baseline, m.max_dev), (0.0, 0.0))

    def _tab_with_shift_records(self):
        from gui.analysis.systematic_scan import new_record, value_key
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        st = at._sys_tab
        recs = {}
        for dv, shift in ((-10.0, 50.0), (0.0, 50.0), (10.0, 0.0)):
            r = new_record(dv)
            r["status"] = "ok"
            for label, v in (("70Ge", shift), ("74Ge", 0.0)):
                r["values"][value_key("Isotope shifts", label, "IS",
                                      "delta_nu")] = {"value": v,
                                                      "sigma": 1.0}
            recs[dv] = r
        # The even-grid scan's rule (nearest usable step); the
        # three-offset batch, the default since 2026-10-09, quotes the
        # common-offset step itself or nothing.
        from gui.analysis.systematics_tab import MODE_RANGE
        st._mode.setCurrentText(MODE_RANGE)
        st._records = recs
        st._excluded = {10.0}
        st._baseline_dv.setValue(10.0)
        return st

    def test_a_band_baseline_skips_an_excluded_step(self):
        st = self._tab_with_shift_records()
        runs, mean = st.bands("Isotope shifts", "IS", "delta_nu")
        by = {b.run: b for b in runs}
        self.assertEqual(by["70Ge"].baseline, 50.0)
        self.assertIsNone(mean)

    def test_no_weighted_mean_across_isotopes(self):
        st = self._tab_with_shift_records()
        rows = [r for r in st.summary_rows()
                if r["project"] == "Isotope shifts"]
        self.assertEqual(sorted(r["run"] for r in rows), ["70Ge", "74Ge"])

    def test_an_empty_shift_tab_stays_empty(self):
        from gui.analysis.systematic_scan import SystematicScan
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        scan._is_tab = mock.Mock()
        scan._gp = None
        scan._snapshot = {"projects": {}, "gp": None,
                          "is_tab": {"had": False}}
        scan._restore_snapshot()
        scan._is_tab.clear_shifts.assert_called_once_with()
        scan._is_tab.shifts_for_scan.assert_not_called()

    def test_clear_shifts_empties_the_tab(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        ist = at._is_tab
        ist._last_shift_data = [{"label": "70Ge"}]
        ist._last_log = "x"
        ist._result_table.setRowCount(2)
        ist.clear_shifts()
        self.assertEqual((ist._last_shift_data, ist._last_log,
                          ist._result_table.rowCount()), ([], "", 0))


# ── 5, 15: NIST ──────────────────────────────────────────────────────


class NistTests(unittest.TestCase):

    def _window(self):
        from tests.test_review_2026_10_08_round2 import NistRound2Tests
        w, lines, levels = NistRound2Tests._window(self)
        self.addCleanup(w.deleteLater)
        return w, lines, levels

    def test_schemes_keep_their_species(self):
        from gui.nist_asd.search import SchemeSearcher
        from gui.nist_asd.tab import NistAsdWindow
        from tests.test_nist_asd_search import _two_step_cfg
        w, lines, levels = self._window()
        w.set_results(SchemeSearcher(_two_step_cfg(), lines, levels).run())
        n = len(w._results)
        self.assertGreater(n, 0)
        w.spectrum_edit.setText("Bb I")
        with mock.patch("gui.nist_asd.tab.QMessageBox"):
            w._on_fetch_error("offline")
        d = w.to_dict()
        self.assertEqual((d["spectrum"], d["results_spectrum"]),
                         ("Bb I", "Aa I"))
        w2 = NistAsdWindow()
        self.addCleanup(w2.deleteLater)
        with mock.patch("gui.nist_asd.data.cache_info", return_value=None), \
                mock.patch("gui.nist_asd.data.load_cache",
                           side_effect=lambda sp, kind:
                           lines if kind == "lines" else levels):
            w2.from_dict(d)
            self.assertEqual(w2._loaded_spectrum, "Bb I")
            self.assertEqual(w2._results, [])
            w2.set_data(lines, levels, "Aa I")
        self.assertEqual(len(w2._results), n)

    def test_numeric_columns_sort_as_numbers(self):
        from gui.nist_asd.tab import _num_item
        t = QTableWidget(3, 1)
        self.addCleanup(t.deleteLater)
        for r, v in enumerate((100.0, 2.0, 20.0)):
            t.setItem(r, 0, _num_item(v, f"{v:g}"))
        t.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.assertEqual([t.item(r, 0).text() for r in range(3)],
                         ["2", "20", "100"])

    def test_a_whole_session_without_nist_drops_the_last_ones(self):
        from gui.main_window import MainWindow
        stub = types.SimpleNamespace(_nist_browser=mock.Mock(),
                                     _pending_nist_state={"x": 1})
        MainWindow._restore_nist(stub, {"analysis": {}}, [])
        stub._nist_browser.set_results.assert_not_called()
        MainWindow._restore_nist(
            stub, {"estimate": {}, "preanalysis": {}, "analysis": {}}, [])
        stub._nist_browser.set_results.assert_called_once_with([])
        self.assertIsNone(stub._pending_nist_state)


# ── 6: common grid errors ────────────────────────────────────────────


class CommonGridErrorTests(unittest.TestCase):

    Y = {"run_0001.asdf": np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], float),
         "run_0002.asdf": np.array([14, 15, 13, 14, 15, 13, 14, 15, 13, 14],
                                   float)}

    def _p0(self, yerr_mode, common_grid):
        from gui.analysis import fitting
        X = np.arange(10.0)

        def prepare(path, source_config, **kw):
            return (path, {"bin_mode": "Frequency",
                           "yerr_mode": source_config["yerr_mode"]}, {})

        def binned(data, cfg):
            from gui.analysis.binning import _make_yerr
            y = self.Y[data].copy()
            yerr, cb = _make_yerr(y, cfg["yerr_mode"])
            return {"x": X.copy(), "y": y, "yerr": yerr, "xerr": None,
                    "use_callable_yerr": cb,
                    "info": {"bin_mode": "Frequency", "n_bins": len(X)}}

        @contextlib.contextmanager
        def no_filter(data, excluded, asdf_path=None):
            yield None

        model = [{"type": "Polynomial", "name": "C",
                  "params": {"p0": _p(5.0)}}]
        fcfg = {"separate": False, "method": "leastsq", "llh": False,
                "scale_covar": False, "shared_params": [],
                "shared_all_params": ["p0"], "expressions": {},
                "priors": [], "common_grid": common_grid,
                "common_grid_width": 0.0}
        w = fitting.FitWorkerThread(
            run_files=list(self.Y),
            source_config={"bin_mode": "Frequency", "yerr_mode": yerr_mode},
            model_configs=model, fitter_config=fcfg, output_config={})
        out = {}
        w.all_done.connect(lambda r: out.setdefault("r", r))
        w.error.connect(lambda e: out.setdefault("e", e))
        with mock.patch("gui.analysis.pipeline.prepare_run_data", prepare), \
                mock.patch("gui.analysis.binning.compute_binned", binned), \
                mock.patch("gui.scan_filter.filter_data_for_binning",
                           no_filter):
            w.run()
        self.assertNotIn("e", out, out.get("e"))
        pdf = out["r"][0]["params_df"]
        return [v for p, v in zip(pdf["Parameter"], pdf["Value"])
                if p == "p0"][0]

    def test_none_stays_unit_weights_on_the_common_grid(self):
        self.assertAlmostEqual(self._p0("None", True), 7.75, places=4)
        self.assertAlmostEqual(self._p0("None", False), 7.75, places=4)


# ── 11, 12: Auto-Fitter ──────────────────────────────────────────────


def _two_voigts():
    def g(x, a, mu, fwhm):
        return a * np.exp(-0.5 * ((x - mu) / (fwhm / 2.3548)) ** 2)
    x = np.linspace(-500, 500, 201)
    y = np.round(g(x, 400, -100, 60) + g(x, 300, 150, 160) + 20.0)
    models = [
        {"type": "Voigt", "name": "M1", "params": {
            "A": _p(300.0, 0.0), "mu": _p(-90.0),
            "FWHMG": _p(90.0, 1, 600), "FWHML": _p(1.0, 0.0, 600, False),
            "Bkg_p0": _p(15.0)}},
        {"type": "Voigt", "name": "M2", "params": {
            "A": _p(250.0, 0.0), "mu": _p(140.0),
            "FWHMG": _p(90.0, 1, 600), "FWHML": _p(1.0, 0.0, 600, False),
            "Bkg_p0": _p(0.0, vary=False)}}]
    return [{"name": "Run_mA", "x": x, "y": y, "yerr": np.sqrt(y + 1)}], \
        models


class AutoFitterTests(unittest.TestCase):

    def test_sharing_follows_the_fit_mode(self):
        from gui.analysis.auto_fitter import _run_one_fit
        src, models = _two_voigts()
        keys = [(0, "A"), (0, "mu"), (0, "FWHMG"), (0, "Bkg_p0"),
                (1, "A"), (1, "mu"), (1, "FWHMG")]
        start = [300.0, -90.0, 90.0, 15.0, 250.0, 140.0, 90.0]
        tied = _run_one_fit(src, models, start, keys, shared_params=[],
                            shared_all_params=["FWHMG"],
                            fitter_config={"method": "leastsq",
                                           "separate": False})
        free = _run_one_fit(src, models, start, keys, shared_params=[],
                            shared_all_params=["FWHMG"],
                            fitter_config={"method": "leastsq",
                                           "separate": True})
        self.assertAlmostEqual(tied["fitted_vals"][2],
                               tied["fitted_vals"][6], places=6)
        self.assertGreater(abs(free["fitted_vals"][2]
                               - free["fitted_vals"][6]), 50.0)

    def test_the_dialog_varies_one_copy_of_an_all_share(self):
        from gui.analysis.auto_fitter import AutoFitterDialog
        src, models = _two_voigts()
        for separate, want in ((False, False), (True, True)):
            d = AutoFitterDialog(src, models, {
                "separate": separate, "method": "leastsq",
                "shared_params": [], "shared_all_params": ["FWHMG"]})
            self.addCleanup(d.deleteLater)
            self.assertEqual((1, "FWHMG") in d._vary_keys, want)

    def _worker(self, stuck, quick, n, cores):
        from gui.analysis import auto_fitter
        from tests import _slow_fit_helper
        patcher = mock.patch.object(auto_fitter, "_run_one_fit",
                                    _slow_fit_helper.slow_autofit)
        patcher.start()
        self.addCleanup(patcher.stop)
        w = auto_fitter._MultiStartWorker(
            sources_data=[{"name": "s", "x": [0.0], "y": [1.0],
                           "yerr": [1.0], "stuck": stuck, "quick": quick}],
            model_configs=[], vary_keys=[(0, "a")], center_values=[10.0],
            spread_frac=0.5, n_starts=n, n_cores=cores)
        ev = []
        w.result_ready.connect(lambda i, c, t, r: ev.append((i, r)),
                               Qt.ConnectionType.DirectConnection)
        return w, ev

    def test_a_hung_start_is_cut_off(self):
        from gui.analysis import auto_fitter
        old = auto_fitter._MultiStartWorker.START_TIMEOUT_S
        auto_fitter._MultiStartWorker.START_TIMEOUT_S = 2
        self.addCleanup(setattr, auto_fitter._MultiStartWorker,
                        "START_TIMEOUT_S", old)
        w, ev = self._worker(60, 0.3, 3, 2)
        t0 = time.monotonic()
        w.start()
        self.assertTrue(w.wait(60000))
        self.assertLess(time.monotonic() - t0, 30.0)
        by = dict(ev)
        self.assertIn("timed out", by[0]["error"])
        self.assertTrue(by[1]["success"] and by[2]["success"])
        self.assertEqual(multiprocessing.active_children(), [])

    def test_stop_ends_the_round_at_once(self):
        w, ev = self._worker(60, 60, 4, 1)
        w.start()
        time.sleep(1.5)
        t0 = time.monotonic()
        w.cancel()
        self.assertTrue(w.wait(30000))
        self.assertLess(time.monotonic() - t0, 10.0)
        self.assertEqual(ev, [])
        self.assertEqual(multiprocessing.active_children(), [])



# ── 13: the GP window average (Arda's choice, 2026-10-09) ────────────


class GaussLegendreWindowTests(unittest.TestCase):
    """The window-averaged correction is the exact mean (Gauss-Legendre),
    not 32 equally weighted points (ends over-weighted, error ~1/n)."""

    @staticmethod
    def _rc():
        from cls_estimations.reference_correction import (
            ReferenceCorrector, ReferenceObservation)
        t = np.linspace(0.0, 6.0, 14)
        obs = [ReferenceObservation(t=ti, centroid=30.0 * np.sin(1.3 * ti),
                                    sigma=1.0) for ti in t]
        rc = ReferenceCorrector(kernel="rbf", random_seed=0)
        rc.fit(obs)
        return rc

    def test_the_window_mean_and_its_sd_are_exact(self):
        rc = self._rc()
        x, w = np.polynomial.legendre.leggauss(256)
        nodes, w = 1.5 + 0.5 * x, w / 2.0
        ref = float(w @ rc.predict(nodes)[0])
        ref_sd = float(np.sqrt(w @ rc.cov(nodes, nodes) @ w))
        mean, sd = rc.predict_interval(1.0, 2.0)
        self.assertAlmostEqual(mean, ref, places=7)
        self.assertAlmostEqual(sd, ref_sd, places=7)
        old = float(np.mean(rc.predict(np.linspace(1.0, 2.0, 32))[0]))
        self.assertGreater(abs(old - ref), 1e-4)   # the rule it replaced

    def test_the_window_covariance_uses_the_same_rule(self):
        rc = self._rc()
        spans = [[1.0, 2.0], [2.5, 2.9]]
        c = rc.cov_spans(spans, spans)
        for i, (a, b) in enumerate(spans):
            self.assertAlmostEqual(float(np.sqrt(c[i, i])),
                                   rc.predict_interval(a, b)[1], places=9)


if __name__ == "__main__":
    unittest.main()
