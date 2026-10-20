"""Second external review (2026-10-08): fixes and the round-1 regressions.

Fitting
  * parameter sharing: satlas2's share calls split a name into letters;
    ties are now explicit expressions (Model and All mode, Bkg_p0);
  * a centroid tied to another source's counts once in the isotope-shift
    averages (it shrank the error by sqrt(N)); reloads keep Expression;
  * the Auto-Fitter prepares data like the fit (Model-based errors,
    "None" = unit weights, scan exclusions);
  * a run that hangs past the time limit is cut at once and the queued
    runs still run; a queued run is never timed out.
Binning / merging
  * the common grid keeps only bins a source measured, and the Binning
    dialog's matrix places each run at its own grid position;
  * measured steps survive even when every photon is gated away, and a
    spectrum without a count is refused at fit time;
  * a voltage merge records each bin's calibrated voltage and the
    frequency projection uses it; per-run cooler voltages are in volts.
Scans / sessions / UI
  * a systematic scan leaves the analysis as it found it, refuses a GP
    trained on references it does not refit, marks a step with a failed
    run partial, and bands only offsets every run reached;
  * scan iterations are never read back as the project's own fit;
  * an unrecognised file does not touch the registries; closing or
    loading never deletes a project whose fit did not stop;
  * Results: session clearing resets the viewer, a style paste merges;
  * NIST schemes follow the species; Quick Plot paste replaces;
  * StatusBarManager timers die with the bar.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_review_2026_10_08_round2.py -q
"""
import json
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import types
import unittest
import warnings
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMessageBox, QTableWidgetItem, QTreeWidgetItem,
)

satlas2 = __import__("pytest").importorskip("satlas2")

_APP = QApplication.instance() or QApplication([])


def _p(value, lo=-1e12, hi=1e12, vary=True):
    return {"value": value, "vary": vary, "min": lo, "max": hi, "expr": ""}


def _voigt():
    return [{"type": "Voigt", "name": "M1", "params": {
        "A": _p(300.0, 0.0), "mu": _p(120.0), "FWHMG": _p(90.0, 1, 600),
        "FWHML": _p(5.0, 0.0, 600), "Bkg_p0": _p(15.0)}}]


def _sim_fit(fitter_extra, spectra, models=None):
    """Real simultaneous fit through FitWorkerThread on inline spectra."""
    from gui.analysis.fitting import FitWorkerThread
    x = np.linspace(-500, 500, 101)
    merged = {}
    for name, (c, bkg) in spectra.items():
        y = 400 * np.exp(-0.5 * ((x - c) / 40) ** 2) + bkg
        merged[f"merged://{name}"] = {"merged_name": name,
                                      "x": x.tolist(), "y": y.tolist(),
                                      "x_unit": "MHz"}
    fcfg = {"separate": False, "method": "leastsq", "llh": False,
            "shared_params": [], "shared_all_params": [],
            "expressions": {}, "priors": []}
    fcfg.update(fitter_extra)
    w = FitWorkerThread(run_files=list(merged),
                        source_config={"bin_mode": "Frequency"},
                        model_configs=models or _voigt(), fitter_config=fcfg,
                        output_config={}, merged_data_map=merged)
    out = {}
    w.all_done.connect(lambda r: out.setdefault("r", r))
    w.error.connect(lambda e: out.setdefault("e", e))
    w.run()
    assert "e" not in out, out.get("e")
    return out["r"]


def _value(result, pname):
    pdf = result["params_df"]
    for p, v in zip(pdf["Parameter"], pdf["Value"]):
        if p == pname:
            return v
    raise KeyError(pname)


# ── sharing + tied centroids ─────────────────────────────────────────


class SharingTests(unittest.TestCase):

    SPECTRA = {"mA": (100.0, 20.0), "mB": (140.0, 20.0)}

    def _shared(self, key, names):
        r = _sim_fit({key: names}, self.SPECTRA)
        return {x["source_name"]: x for x in r}

    def test_model_mode_shares_a_multi_letter_name(self):
        res = self._shared("shared_params", ["mu"])
        self.assertAlmostEqual(_value(res["Run_mA"], "mu"),
                               _value(res["Run_mB"], "mu"), places=9)
        expr = res["Run_mB"]["params_df"]["Expression"]
        self.assertIn("Run_mA___M1___mu", list(expr))

    def test_all_mode_shares_too(self):
        res = self._shared("shared_all_params", ["mu"])
        self.assertAlmostEqual(_value(res["Run_mA"], "mu"),
                               _value(res["Run_mB"], "mu"), places=9)

    def test_background_name_maps_to_the_bkg_model(self):
        from gui.analysis.fitting import share_ties
        x = np.linspace(0, 1, 5)
        srcs = []
        for n in ("Run_a", "Run_b"):
            s = satlas2.Source(x, x + 1, yerr=np.ones(5), name=n)
            from gui.analysis.fitting import _build_models_on_source
            _build_models_on_source(s, _voigt() + [
                {"type": "Polynomial", "name": "P",
                 "params": {"p0": _p(1.0)}}], satlas2)
            srcs.append(s)
        ties = share_ties(srcs, ["Bkg_p0"], [])
        self.assertEqual(ties, {"Run_b___M1_bkg___p0":
                                "Run_a___M1_bkg___p0"})
        # All mode with a background name never ties a Polynomial's p0.
        ties = share_ties(srcs, [], ["Bkg_p0"])
        self.assertNotIn("Run_b___P___p0", ties)

    def test_auto_fitter_shares(self):
        from gui.analysis.auto_fitter import _run_one_fit
        x = np.linspace(-500, 500, 101)
        srcs = []
        for n, c in (("Run_mA", 100.0), ("Run_mB", 140.0)):
            y = 400 * np.exp(-0.5 * ((x - c) / 40) ** 2) + 20
            srcs.append({"name": n, "x": x, "y": y, "yerr": np.sqrt(y + 1)})
        r = _run_one_fit(srcs, _voigt(), [300, 120, 90, 5, 15],
                         [(0, "A"), (0, "mu"), (0, "FWHMG"), (0, "FWHML"),
                          (0, "Bkg_p0")], shared_params=["mu"],
                         fitter_config={"method": "leastsq"})
        self.assertTrue(r["success"])
        self.assertLess(abs(r["fitted_vals"][1] - 120.0), 2.0)


class TiedCentroidTests(unittest.TestCase):

    @staticmethod
    def _pdf(source, value, expr):
        return {"Source": [source], "Model": ["Model_1"],
                "Parameter": ["centroid"], "Value": [value],
                "Stderr": [0.5], "Expression": [expr]}

    def test_a_cross_source_tie_counts_once(self):
        from cls_estimations.isotope_shift import (
            independent_results, centroid_tie)
        head = {"source_name": "Run_mA",
                "params_df": self._pdf("Run_mA", 25.0, float("nan"))}
        tied = {"source_name": "Run_mB",
                "params_df": self._pdf("Run_mB", 25.0,
                                       "Run_mA___Model_1___centroid")}
        self.assertIsNone(centroid_tie(head["params_df"],
                                       source_name="Run_mA"))
        self.assertEqual(independent_results([head, tied]), [head])

    def test_a_within_source_tie_still_counts(self):
        from cls_estimations.isotope_shift import independent_results
        r = {"source_name": "Run_mA",
             "params_df": self._pdf("Run_mA", 25.0,
                                    "Run_mA___Model_2___centroid")}
        self.assertEqual(independent_results([r]), [r])

    # The source-text check that both averaging sites deduplicate is
    # replaced by numerical tests of the tab in
    # test_review_2026_10_08_round3.py (TieGroupShiftTests, review 3).

    def test_reload_keeps_the_expression(self):
        from gui.analysis.project import AnalysisProject
        root = tempfile.mkdtemp()
        d = os.path.join(root, "analysis", "P", "iter_001")
        os.makedirs(d)
        pd.DataFrame({"run_number": ["1"], "Parameter": ["centroid"],
                      "Value": [1.0], "Error": [0.1], "Source": ["Run_1"],
                      "Model": ["M"], "Expression": ["Run_0___M___centroid"]}
                     ).to_csv(os.path.join(d, "parameters.csv"),
                              index=False)
        p = AnalysisProject("P")
        p._session_iterations = ["iter_001"]
        with mock.patch("gui.shared_widgets.get_analysis_dir",
                        return_value=os.path.join(root, "analysis")):
            p.load_results_from_disk()
        self.assertEqual(p._last_results[0]["params_df"]["Expression"],
                         ["Run_0___M___centroid"])
        shutil.rmtree(root, ignore_errors=True)


# ── Auto-Fitter data preparation ─────────────────────────────────────


class AutoFitPrepTests(unittest.TestCase):

    def _load(self, use_callable, yerr, registry=None):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("P")
        y = np.array([0.0, 1.0, 10.0, 20.0])
        res = {"x": np.arange(4.0), "y": y, "yerr": yerr, "xerr": None,
               "use_callable_yerr": use_callable, "info": {}}
        seen = {}

        class _Filter:
            def __init__(self, data, excluded, asdf_path=None):
                seen["excluded"] = excluded

            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        reg = mock.Mock()
        reg.get.return_value = registry or set()
        with mock.patch("gui.analysis.pipeline.prepare_run_data",
                        return_value=(object(), {"bin_mode": "Frequency"},
                                      {})), \
                mock.patch("gui.analysis.binning.compute_binned",
                           return_value=res), \
                mock.patch("gui.scan_filter.filter_data_for_binning",
                           _Filter), \
                mock.patch("gui.scan_filter.get_registry",
                           return_value=reg):
            out = p._load_data_for_autofit("run_1.asdf", {}, None)
        return out, seen

    def test_model_based_errors_stay_model_based(self):
        (_x, _y, yerr), _ = self._load(True, np.sqrt)
        self.assertTrue(callable(yerr))

    def test_none_errors_are_unit_weights(self):
        (_x, _y, yerr), _ = self._load(False, None)
        np.testing.assert_array_equal(yerr, np.ones(4))

    def test_scan_exclusions_apply(self):
        _out, seen = self._load(False, np.ones(4), registry={0, 3})
        self.assertEqual(seen["excluded"], {0, 3})

    def test_the_dialog_draws_model_based_errors(self):
        from gui.analysis.auto_fitter import _display_yerr
        np.testing.assert_allclose(
            _display_yerr({"y": [0.0, 4.0], "yerr": np.sqrt}), [1.0, 2.0])


# ── worker time limit ────────────────────────────────────────────────


class TimeoutTests(unittest.TestCase):

    def _run(self, runs, cores, limit):
        from gui.analysis import fitting
        from tests import _slow_fit_helper
        patcher = mock.patch.object(fitting, "_fit_single_run",
                                    _slow_fit_helper.slow_fit)
        patcher.start()
        self.addCleanup(patcher.stop)
        old = fitting.FitWorkerThread.RUN_TIMEOUT_S
        fitting.FitWorkerThread.RUN_TIMEOUT_S = limit
        self.addCleanup(setattr, fitting.FitWorkerThread, "RUN_TIMEOUT_S",
                        old)
        w = fitting.FitWorkerThread(
            run_files=runs, source_config={}, model_configs=[],
            fitter_config={"separate": True, "method": "leastsq"},
            output_config={}, n_cores=cores)
        out = {}
        w.all_done.connect(lambda r: out.update(t=time.monotonic(), r=r),
                           Qt.ConnectionType.DirectConnection)
        t0 = time.monotonic()
        w.start()
        self.assertTrue(w.wait(90000))
        _APP.processEvents()
        return out["t"] - t0, {os.path.basename(r["run_file"]): r
                               for r in out["r"]}

    def test_a_hung_run_does_not_block_the_queue(self):
        took, res = self._run(["run_0_60s.asdf", "run_1_0.5s.asdf",
                               "run_2_0.5s.asdf"], 1, 2)
        self.assertLess(took, 30.0)
        self.assertFalse(res["run_0_60s.asdf"]["success"])
        self.assertIn("timed out", res["run_0_60s.asdf"]["error"])
        self.assertTrue(res["run_1_0.5s.asdf"]["success"])
        self.assertTrue(res["run_2_0.5s.asdf"]["success"])
        self.assertEqual(multiprocessing.active_children(), [])

    def test_a_queued_run_is_not_timed_out(self):
        _took, res = self._run(["run_0_2s.asdf", "run_1_2s.asdf"], 1, 3)
        self.assertTrue(all(r["success"] for r in res.values()))


# ── binning / merging ────────────────────────────────────────────────


class CommonGridRound2Tests(unittest.TestCase):

    def test_no_zero_inside_a_gap(self):
        from gui.analysis.binning import rebin_to_common_grid
        reb, _c, _w = rebin_to_common_grid([([0.0, 2.0], [10.0, 10.0])],
                                           bin_width=1.0)
        np.testing.assert_array_equal(reb[0][1], [10.0, 10.0])

    def test_matrix_puts_each_run_in_its_place(self):
        from gui.analysis.binning import common_grid_matrix
        Z, centers, _w, occ = common_grid_matrix(
            [([0.0, 1.0], [5.0, 5.0]), ([2.0, 3.0, 4.0], [7.0, 7.0, 7.0])],
            bin_width=1.0)
        self.assertEqual(Z.shape, (2, len(centers)))
        self.assertTrue(np.isnan(Z[0, 3]))
        self.assertEqual(Z[1, 3], 7.0)


class _Frame:
    def __init__(self, df):
        self.Sorted = df


class AllGatedAwayTests(unittest.TestCase):

    def test_steps_survive_when_no_photon_passes(self):
        from gui.analysis.binning import _per_step_frequency_bins
        df = pd.DataFrame({"DV": [0.0, 1.0, 2.0], "F": [1e8, 2e8, 3e8],
                           "TOF": [10.0, 10.0, 10.0], "TDC": [3, 3, 3]})
        x, y, _ = _per_step_frequency_bins(_Frame(df), (20, 40), [3],
                                           None, None)
        self.assertEqual(len(x), 3)
        self.assertEqual(float(y.sum()), 0.0)

    def test_a_spectrum_without_counts_is_refused(self):
        from gui.analysis.fitting import require_counts
        with self.assertRaises(ValueError):
            require_counts(np.zeros(5), "Run_1")
        require_counts(np.array([0.0, 1.0]), "Run_1")


class VoltageMergeCalibrationTests(unittest.TestCase):

    def _md(self, **extra):
        md = {"x": [1150.0, 1200.0, 1250.0], "y": [1.0, 2.0, 1.0],
              "yerr": [1.0, 1.0, 1.0], "merged_name": "m",
              "merge_metadata": {"cooler_v": 30000.0, "laser_sp": 20000.0,
                                 "mass_amu": 100.0, "harmonic": 2}}
        md.update(extra)
        return md

    def test_the_projection_uses_the_calibrated_voltage(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency
        src = {"ref_freq": 1.2e15}
        cal = [1.02 * v + 5 for v in (1150.0, 1200.0, 1250.0)]
        x_raw = project_voltage_merge_to_frequency(self._md(), src)[0]
        x_cal = project_voltage_merge_to_frequency(
            self._md(x_cal=cal), src)[0]
        x_ref = project_voltage_merge_to_frequency(
            self._md(x=cal), src)[0]
        np.testing.assert_allclose(x_cal, x_ref)
        self.assertGreater(abs(x_cal[0] - x_raw[0]), 100.0)

    def test_an_old_merge_is_flagged(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency
        w = project_voltage_merge_to_frequency(self._md(),
                                               {"ref_freq": 1e15})[3]
        self.assertIn("VOLTAGE_MERGE_UNCALIBRATED", [x["code"] for x in w])

    def test_cooler_readback_is_never_a_beam_energy(self):
        from gui.analysis.merge import project_voltage_merge_to_frequency
        md = self._md(per_run=[{"cooler_v": 3.0}, {"cooler_v": 30010.0}])
        md["merge_metadata"]["cooler_v"] = None
        x = project_voltage_merge_to_frequency(md, {"ref_freq": 1e15})[0]
        ref = self._md()
        ref["merge_metadata"]["cooler_v"] = 30010.0
        x_ref = project_voltage_merge_to_frequency(ref,
                                                   {"ref_freq": 1e15})[0]
        np.testing.assert_allclose(x, x_ref)


# ── systematic scans ─────────────────────────────────────────────────


class ScanStateTests(unittest.TestCase):

    def _setup(self):
        from tests.test_systematic_scan import ChainTests
        return ChainTests._setup(self)

    def _run(self, *a, **k):
        from tests.test_systematic_scan import ChainTests
        return ChainTests._run(self, *a, **k)

    def test_the_analysis_is_restored_after_a_scan(self):
        from tests.test_systematic_scan import _target
        ref, sample, gp = self._setup()
        sentinel = [{"success": True, "run_number": "nominal"}]
        for p in (ref, sample):
            p._last_results = sentinel
            p._last_iter_dir = "nominal_dir"
        gp.snapshot_state = mock.Mock(return_value={"tag": "pre-scan"})
        gp.restore_state = mock.Mock()
        self._run([_target(ref, "reference"), _target(sample)],
                  [-30.0, -10.0], gp=gp)
        for p in (ref, sample):
            self.assertIs(p._last_results, sentinel)
            self.assertEqual(p._last_iter_dir, "nominal_dir")
        gp.restore_state.assert_called_once_with({"tag": "pre-scan"})

    def test_a_gp_on_a_reference_the_scan_does_not_refit_is_refused(self):
        from gui.analysis.systematic_scan import SystematicScan
        from tests.test_systematic_scan import _target
        _ref, sample, gp = self._setup()
        gp.reference_project_names = lambda: {"116Sn"}
        with self.assertRaises(ValueError):
            SystematicScan().start([_target(sample)], [-10.0], gp=gp)

    def test_a_failed_run_makes_the_step_partial(self):
        from gui.analysis.systematic_scan import SystematicScan, ScanTarget
        scan = SystematicScan()
        rec = {"dv": 1.0, "status": "pending", "message": "", "values": {},
               "redchi": {}, "shifts": [], "labels": {}, "iterations": {},
               "failed_projects": []}
        scan._records = {1.0: rec}
        tgt = ScanTarget(project=types.SimpleNamespace(project_name="P"))
        scan._collect(tgt, 1.0, [
            {"success": True, "run_number": "1", "source_name": "Run_1",
             "params_df": {"Parameter": ["c"], "Value": [0.0],
                           "Stderr": [1.0], "Source": ["Run_1"],
                           "Model": ["M"]}},
            {"success": False, "run_number": "2"}])
        self.assertEqual(rec["failed_runs"], ["P/2"])

    def test_scan_iterations_are_not_the_projects_own_fit(self):
        from gui.analysis.project import AnalysisProject
        root = tempfile.mkdtemp()
        for name in ("iter_001", "sys_001_+1V_CO"):
            d = os.path.join(root, "analysis", "P", name)
            os.makedirs(d)
            pd.DataFrame({"run_number": ["1"], "Parameter": ["centroid"],
                          "Value": [1.0], "Error": [0.1]}).to_csv(
                os.path.join(d, "parameters.csv"), index=False)
        p = AnalysisProject("P")
        p._session_iterations = ["iter_001", "sys_001_+1V_CO"]
        p._scan_iterations = {"sys_001_+1V_CO"}
        with mock.patch("gui.shared_widgets.get_analysis_dir",
                        return_value=os.path.join(root, "analysis")):
            p.load_results_from_disk()
        self.assertEqual(p._loaded_iteration, "iter_001")
        self.assertEqual(p.to_dict(include_iterations=True)
                         ["scan_iterations"], ["sys_001_+1V_CO"])
        shutil.rmtree(root, ignore_errors=True)


# ── sessions / window ────────────────────────────────────────────────


class LoadGuardTests(unittest.TestCase):

    def _stub(self):
        from gui.main_window import MainWindow
        stub = types.SimpleNamespace()
        for name in ("_is_session_dict", "_looks_like_estimate",
                     "_stop_fits_for_load"):
            setattr(stub, name, types.MethodType(
                getattr(MainWindow, name), stub))
        return stub

    def test_an_unrecognised_file_leaves_the_registries_alone(self):
        from gui.main_window import MainWindow
        from gui.scan_filter import get_registry
        reg = get_registry()
        before = reg.to_dict()
        reg.from_dict({"C:/x/run_1.asdf": {"excluded": [1, 2]}})
        try:
            out = MainWindow._restore_all(self._stub(), {"unrecognized": 1},
                                          "x.yaml", mock.Mock())
            self.assertEqual(out, [])
            self.assertTrue(reg.to_dict())
        finally:
            reg.from_dict(before)

    def test_a_load_does_not_proceed_past_a_fit_that_will_not_stop(self):
        stub = self._stub()
        stub.analysis_tab = mock.Mock()
        stub.analysis_tab.running_fits.return_value = ["116Sn"]
        stub.analysis_tab.stop_all_fits.return_value = False
        with mock.patch("gui.main_window.QMessageBox") as box:
            box.StandardButton = QMessageBox.StandardButton
            box.question.return_value = QMessageBox.StandardButton.Yes
            self.assertFalse(stub._stop_fits_for_load())

    def test_close_project_keeps_a_project_whose_fit_will_not_stop(self):
        from gui.analysis.tab import AnalysisTab
        tab = AnalysisTab()
        tab._add_project("P")
        proj = tab._projects[0]
        proj.is_fitting = lambda: True
        proj.stop_fit = lambda *a, **k: False
        idx = tab._project_tabs.indexOf(proj)
        with mock.patch("gui.analysis.tab.QMessageBox") as box:
            box.StandardButton = QMessageBox.StandardButton
            box.question.return_value = QMessageBox.StandardButton.Yes
            tab._close_project(idx)
        self.assertIn(proj, tab._projects)


class ResultsRound2Tests(unittest.TestCase):

    def test_a_font_paste_keeps_the_targets_other_style(self):
        from gui.results_tab import _merge_style
        existing = {"figsize": [4, 3], "axes": [
            {"title": "B", "xlim": [5, 12], "grid": True,
             "title_size": 13}]}
        pasted = {"axes": [{"title_size": 21, "tick_size": 9}]}
        out = _merge_style(existing, pasted)
        ax = out["axes"][0]
        self.assertEqual((ax["title"], ax["xlim"], ax["grid"]),
                         ("B", [5, 12], True))
        self.assertEqual((ax["title_size"], ax["tick_size"]), (21, 9))
        self.assertEqual(out["figsize"], [4, 3])

    def test_clear_session_stops_writing_the_old_notes(self):
        from gui.results_tab import ResultsTab
        rt = ResultsTab()
        d = tempfile.mkdtemp()
        old = os.path.join(d, "_project_notes.txt")
        rt._notes_path = old
        rt._notes_dirty = False
        rt._current_tree_item = QTreeWidgetItem(["x"])
        rt._current_item_data = {"type": "notes", "path": old}
        rt.clear_session()
        self.assertIsNone(rt._notes_path)
        self.assertIsNone(rt._current_tree_item)
        rt._on_notes_text_changed()           # typing after the clear
        rt._save_notes(force=True)
        self.assertFalse(os.path.exists(old))
        shutil.rmtree(d, ignore_errors=True)


class NistRound2Tests(unittest.TestCase):

    def _window(self):
        from gui.nist_asd.tab import NistAsdWindow
        from tests.test_nist_asd_search import _system
        w = NistAsdWindow()
        w.spectrum_edit.setText("Aa I")
        lines, levels = _system()
        with mock.patch("gui.nist_asd.data.cache_info", return_value=None):
            w.set_data(lines, levels, "Aa I")
        return w, lines, levels

    def test_schemes_do_not_survive_another_species(self):
        from gui.nist_asd.search import SchemeSearcher
        from tests.test_nist_asd_search import _two_step_cfg
        w, lines, levels = self._window()
        w.set_results(SchemeSearcher(_two_step_cfg(), lines, levels).run())
        self.assertTrue(w._results)
        with mock.patch("gui.nist_asd.data.cache_info", return_value=None):
            w._on_fetch_done("Bb I", lines, levels)
        self.assertEqual(w._results, [])

    def test_a_late_search_result_is_dropped(self):
        from gui.nist_asd.search import SchemeSearcher
        from tests.test_nist_asd_search import _two_step_cfg
        w, lines, levels = self._window()
        w._search_spectrum = "Aa I"
        with mock.patch("gui.nist_asd.data.cache_info", return_value=None):
            w._on_fetch_done("Bb I", lines, levels)
        w._on_search_done(SchemeSearcher(_two_step_cfg(), lines,
                                         levels).run())
        self.assertEqual(w._results, [])


class QuickPlotPasteTests(unittest.TestCase):

    def test_a_shorter_paste_replaces(self):
        from gui.shared_widgets import QuickPlotDialog
        d = QuickPlotDialog()
        cb = QApplication.clipboard()
        cb.setText("x\ty\n" + "\n".join(f"{i}\t{i * 10}"
                                         for i in range(1, 7)))
        d._paste_clipboard()
        cb.setText("x\ty\n100\t1\n200\t2")
        d._paste_clipboard()
        self.assertEqual(len(d._data), 2)


class StatusBarTimerTests(unittest.TestCase):

    def test_timers_belong_to_the_bar(self):
        from PySide6.QtWidgets import QStatusBar
        from gui.shared_widgets import StatusBarManager
        bar = QStatusBar()
        m = StatusBarManager(bar)
        m.flash("done")
        self.assertIs(m._timer.parent(), bar)
        self.assertIs(m._flash_timer.parent(), bar)
        m._flash_timer.stop()



# ── cooler offset keeps the per-event cooler readback ────────────────


class CoolerOffsetTests(unittest.TestCase):
    """The Yb cooler offset used to switch the per-event
    cooler correction off. clstools: V = Vrfq * VCoolDiv + VCoolOffset -
    DV_cal; the offset path set VCoolDiv = 0 (one constant cooler)."""

    def _prepare(self, **cfg):
        import sys

        class _D:
            def __init__(self):
                self.VCoolDiv = 10000
                self.VCoolOffset = 0
                self.Vcool_init = 3.0
                self.Laser_set = 20000.0
                self.Size = 10
                self.at_voltages = None

            def Compute_Voltages(self, cooler_correction="pbp"):
                self.at_voltages = (self.VCoolDiv, self.VCoolOffset)

            def Compute_WL(self, **kw):
                # clstools takes Frequency_stepsize from this product only.
                self.stepsize_v = self.Vcool_init * self.VCoolDiv

            def Shift_Ref(self, **kw):
                pass

            def apply_filter(self, **kw):
                pass

        fake = types.ModuleType("clstools")
        fake.CLSDataFrame = _D
        base = {"mass": 100.0, "ref_freq": 1.2e15, "harmonic": 2,
                "bin_mode": "Frequency", "cooler_correction": "pbp"}
        base.update(cfg)
        cal = mock.Mock()
        cal.to_report_dict.return_value = {}
        with mock.patch.dict(sys.modules, {"clstools": fake}), \
                mock.patch("gui.calibration.load_run_calibrated",
                           return_value=cal):
            from gui.analysis.pipeline import prepare_run_data
            data, _eff, meta = prepare_run_data("run_1.asdf", base,
                                                calibrations={})
        return data, meta

    def test_offset_rides_on_the_per_event_readback(self):
        data, meta = self._prepare(cooler_offset_v=-20.0)
        self.assertEqual(data.at_voltages, (10000, -20.0))
        self.assertAlmostEqual(meta["cooler_v_effective"], 30000.0 - 20.0)

    def test_the_step_size_follows_the_offset(self):
        data, _ = self._prepare(cooler_offset_v=-20.0)
        self.assertAlmostEqual(data.stepsize_v, 30000.0 - 20.0, places=6)
        self.assertAlmostEqual(data.Vcool_init, 3.0)     # put back

    def test_the_source_preview_applies_the_offset(self):
        from gui.analysis import blocks
        seen = {}

        class _Stop(Exception):
            pass

        class _D:
            VCoolDiv, VCoolOffset, Vcool_init = 10000, 0, 3.0

            def Compute_Voltages(self, cooler_correction="pbp"):
                seen["v"] = (self.VCoolDiv, self.VCoolOffset)
                raise _Stop

        sb = blocks.SourceBlock()
        sb._cooler_offset.setValue(-20.0)
        fake = types.SimpleNamespace(CLSDataFrame=_D)
        with mock.patch.object(sb, "get_checked_files",
                               return_value=[{"path": "run_1.asdf"}]), \
                mock.patch.object(blocks, "_get_clstools",
                                  return_value=fake), \
                mock.patch("gui.calibration.load_run_calibrated"), \
                mock.patch.object(blocks, "QMessageBox"):
            sb._show_preview_at_index()
        self.assertEqual(seen["v"], (10000, -20.0))

    def test_offset_on_an_override_stays_a_constant(self):
        data, _ = self._prepare(cooler_offset_v=-20.0,
                                override_enabled=True,
                                cooler_override=30100.0)
        self.assertEqual(data.at_voltages, (0, 30100.0 - 20.0))

    def test_no_offset_changes_nothing(self):
        data, _ = self._prepare()
        self.assertEqual(data.at_voltages, (10000, 0))


if __name__ == "__main__":
    unittest.main()
