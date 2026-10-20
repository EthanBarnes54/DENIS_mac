"""Analysis fixes from the 2026-10-08 external review.

Fitting
  * simultaneous fits: each run carries only its own parameter rows;
  * the Auto-Fitter ranks likelihood fits by -ln L, not by lmfit's
    chisqr (the sum of SQUARED likelihood terms);
  * satlas2 dropped a lone Gaussian prior from Poisson fits;
  * standalone Polynomial / Piecewise models ignored Min/Max;
  * Conv. iter / Conv. tau reach satlas2 in both fit paths;
  * Cancel stops the worker processes, and its partial results are
    neither saved nor sent on.
Binning / merging
  * a measured step whose photons all fail the gates stays as a zero;
  * the common grid gives each source only the range it measured;
  * "From voltage std" x-errors are in MHz (were volts / 1e6), no NaN;
  * merges with a recorded domain export/reload at any width;
  * the voltage-merge projection follows ref_shift and cooler_offset_v.
GP / sessions
  * GP correction uncertainty is propagated over each run's window;
  * a session without a GP clears the previous one;
  * the Results tab forgets the previous session;
  * results read back from disk are the session's own iteration.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_review_2026_10_08_analysis.py -q
"""
import os
import shutil
import tempfile
import time
import unittest
import warnings
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

satlas2 = __import__("pytest").importorskip("satlas2")

_APP = QApplication.instance() or QApplication([])


def _p(value, lo=-1e12, hi=1e12, vary=True):
    return {"value": value, "vary": vary, "min": lo, "max": hi, "expr": ""}


# ── fitting ──────────────────────────────────────────────────────────


class SimultaneousRowsTests(unittest.TestCase):
    """Through the real simultaneous worker, with two inline spectra."""

    def test_each_run_keeps_its_own_rows(self):
        from gui.analysis.fitting import FitWorkerThread
        x = np.linspace(-500, 500, 101)
        merged = {}
        for name, c in (("mA", 100.0), ("mB", 140.0)):
            y = 400 * np.exp(-0.5 * ((x - c) / 40) ** 2) + 20
            merged[f"merged://{name}"] = {
                "merged_name": name, "x": x.tolist(), "y": y.tolist(),
                "x_unit": "MHz"}
        models = [{"type": "Voigt", "name": "M1", "params": {
            "A": _p(300.0, 0.0), "mu": _p(120.0),
            "FWHMG": _p(90.0, 1, 600), "FWHML": _p(5.0, 0.0, 600),
            "Bkg_p0": _p(15.0)}}]
        w = FitWorkerThread(
            run_files=list(merged), source_config={"bin_mode": "Frequency"},
            model_configs=models,
            fitter_config={"separate": False, "method": "leastsq",
                           "llh": False},
            output_config={}, merged_data_map=merged)
        out = {}
        w.all_done.connect(lambda r: out.setdefault("r", r))
        w.run()
        mus = {}
        for r in out["r"]:
            pdf = r["params_df"]
            self.assertEqual(set(pdf["Source"]), {r["source_name"]})
            mus[r["source_name"]] = [v for p, v in zip(pdf["Parameter"],
                                                        pdf["Value"])
                                     if p == "mu"]
        self.assertAlmostEqual(mus["Run_mA"][0], 100.0, places=2)
        self.assertAlmostEqual(mus["Run_mB"][0], 140.0, places=2)


class AutoFitterRankingTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(12345)
        self.x = np.linspace(-600, 600, 241)
        gen = satlas2.Source(self.x, np.ones_like(self.x),
                             yerr=np.ones_like(self.x), name="gen")
        gen.addModel(satlas2.Voigt(400.0, 0.0, 60.0, 30.0, name="big"))
        gen.addModel(satlas2.Voigt(120.0, 250.0, 60.0, 30.0, name="sm"))
        gen.addModel(satlas2.Polynomial([30.0], name="bkg"))
        self.y = rng.poisson(gen.evaluate(self.x)).astype(float)
        self.src = [{"name": "Run_1", "x": self.x, "y": self.y,
                     "yerr": np.sqrt(np.maximum(self.y, 1.0))}]
        self.models = [{"type": "Voigt", "name": "Voigt_1", "params": {
            "A": _p(100.0, 0.0), "mu": _p(0.0), "FWHMG": _p(60.0, 0.0),
            "FWHML": _p(30.0, 0.0), "Bkg_p0": _p(30.0)}}]
        self.keys = [(0, "A"), (0, "mu"), (0, "FWHMG"), (0, "FWHML"),
                     (0, "Bkg_p0")]

    def test_poisson_starts_ranked_by_minus_lnL(self):
        from gui.analysis.auto_fitter import _run_one_fit, _rank_key
        fcfg = {"method": "leastsq", "llh": True, "llh_method": "poisson",
                "scale_covar": False}
        good = _run_one_fit(self.src, self.models,
                            [350.0, 5.0, 55.0, 35.0, 30.0], self.keys,
                            fitter_config=fcfg)
        local = _run_one_fit(self.src, self.models,
                             [100.0, 250.0, 55.0, 35.0, 30.0], self.keys,
                             fitter_config=fcfg)
        self.assertTrue(good["success"] and local["success"])
        self.assertEqual(good["objective_name"], "-ln L")
        self.assertLess(_rank_key(good), _rank_key(local))
        # The displayed chi2 is the residual one, not lmfit's 5e7.
        self.assertLess(good["chi2"], 1e5)

    def test_chi_square_rank_key_is_chi_square(self):
        from gui.analysis.auto_fitter import _run_one_fit
        r = _run_one_fit(self.src, self.models,
                         [350.0, 5.0, 55.0, 35.0, 30.0], self.keys,
                         fitter_config={"method": "leastsq", "llh": False})
        self.assertAlmostEqual(r["objective"], r["chi2"], delta=1e-6
                               * r["chi2"])


class SinglePriorTests(unittest.TestCase):

    def test_one_prior_acts_in_a_poisson_fit(self):
        from gui.analysis.fitting import _patch_satlas2_emcee_numpy2
        _patch_satlas2_emcee_numpy2()
        rng = np.random.default_rng(1)
        x = np.arange(200.0)
        y = rng.poisson(10, 200).astype(float)
        s = satlas2.Source(x, y, yerr=np.sqrt(y + 1), name="S")
        m = satlas2.Polynomial([5.0], name="P")
        s.addModel(m)
        f = satlas2.Fitter()
        f.addSource(s)
        f.setParamPrior("S", "P", "p0", 1.0, 0.01)
        f.fit(llh=True, llh_method="poisson", method="nelder")
        self.assertLess(m.params["p0"].value, 2.0)


class PolynomialBoundsTests(unittest.TestCase):

    def test_bounds_reach_satlas2(self):
        from gui.analysis.fitting import _build_models_on_source
        x = np.linspace(0, 10, 50)
        y = np.full_like(x, 10.0)
        cfg = [{"type": "Polynomial", "name": "P", "params": {
                    "p0": _p(2.0, 0.0, 5.0), "p1": _p(0.0, -0.1, 0.1)}},
               {"type": "Piecewise Constant", "name": "W", "params": {
                    "value0": _p(2.0, 1.0, 5.0), "value1": _p(2.0, 1.0, 5.0),
                    "bound0": _p(5.0, vary=False)}}]
        src = satlas2.Source(x, y, yerr=np.ones_like(y), name="S")
        _build_models_on_source(src, cfg, satlas2)
        models = dict(src.models)
        self.assertEqual((models["P"].params["p0"].min,
                          models["P"].params["p0"].max), (0.0, 5.0))
        self.assertEqual(models["W"].params["value1"].max, 5.0)

    def test_eleven_coefficients_keep_their_order(self):
        from gui.analysis.fitting import _index_of
        keys = [f"p{i}" for i in range(12)]
        self.assertEqual(sorted(keys, key=_index_of), keys)


class McmcKwargsTests(unittest.TestCase):

    def test_convergence_settings_are_forwarded(self):
        from gui.analysis.fitting import emcee_fit_kwargs
        kw = emcee_fit_kwargs({"nwalkers": 64, "steps": 900,
                               "convergence": True, "convergence_iter": 80,
                               "convergence_tau": 0.02})
        self.assertEqual(kw, {"nwalkers": 64, "steps": 900,
                              "convergence": True, "convergence_iter": 80,
                              "convergence_tau": 0.02})

    def test_both_fit_paths_use_it(self):
        import inspect
        from gui.analysis import fitting
        for fn in (fitting._fit_single_run,
                   fitting.FitWorkerThread._run_simultaneous):
            self.assertIn("emcee_fit_kwargs(", inspect.getsource(fn))


class CancelTests(unittest.TestCase):
    """The real worker with a sleeping stand-in for the per-run fit."""

    def _worker(self, n=4, secs=5):
        from gui.analysis import fitting
        from tests import _slow_fit_helper
        self._patch = mock.patch.object(fitting, "_fit_single_run",
                                        _slow_fit_helper.slow_fit)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        files = [f"run_{i}_{secs}s.asdf" for i in range(n)]
        return fitting.FitWorkerThread(
            run_files=files, source_config={}, model_configs=[],
            fitter_config={"separate": True, "method": "leastsq"},
            output_config={}, n_cores=2)

    def test_cancel_stops_at_once(self):
        """Cancel once a fit is really running (not after a fixed sleep),
        time all_done where it is emitted, and give the fits 30 s so
        'killed' and 'waited for' are far apart (review 2: the old
        version was timing-sensitive in a full run)."""
        import multiprocessing
        import threading
        from PySide6.QtCore import Qt
        w = self._worker(n=4, secs=30)
        running = threading.Event()
        out = {}
        direct = Qt.ConnectionType.DirectConnection
        w.progress.connect(
            lambda c, t, s: running.set() if c > 0 else None, direct)
        w.all_done.connect(
            lambda r: out.update(t=time.monotonic(), r=r), direct)
        w.start()
        self.assertTrue(running.wait(60), "no fit ever started")
        t0 = time.monotonic()
        w.cancel()
        self.assertTrue(w.wait(30000))
        _APP.processEvents()
        self.assertLess(out["t"] - t0, 10.0)
        self.assertFalse(any(r.get("success") for r in out["r"]))
        self.assertEqual(multiprocessing.active_children(), [])

    def test_cancelled_results_are_not_saved(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("P")
        w = mock.Mock(_cancelled=True)
        p._fit_worker = w
        fb = mock.Mock()
        with mock.patch.object(p, "_save_results") as save:
            got = []
            p.results_ready.connect(lambda *a: got.append(a))
            p._on_all_fits_done([{"success": True, "run_number": "1"}],
                                fb, {}, worker=w)
        save.assert_not_called()
        self.assertEqual(got, [])
        self.assertIsNone(p._last_results)

    def test_a_scan_ends_when_its_fit_is_cancelled(self):
        """A cancelled fit sends no results_ready any more; a scan that
        waited for one hung. It now ends on fit_cancelled."""
        from PySide6.QtCore import Signal
        from gui.analysis.cooler_scan import CoolerScan
        from tests.test_cooler_scan import _StubProject, _spin

        class _Cancelling(_StubProject):
            fit_cancelled = Signal(str)

            def _on_fit_requested(self):
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.fit_cancelled.emit(
                    self.project_name))

        p = _Cancelling("171", ["1001"], -0.21, 12650.0)
        self.addCleanup(p.deleteLater)
        scan = CoolerScan()
        self.addCleanup(scan.deleteLater)
        scan.start([(p, "171Yb")], [-10.0, 0.0, 10.0])
        out = _spin(scan, timeout_ms=15000)
        self.assertIn("points", out)          # finished, not timed out
        self.assertEqual(out["points"], [])
        self.assertFalse(scan.busy)

    def test_a_stale_worker_is_ignored(self):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("P")
        old, new = mock.Mock(_cancelled=False), mock.Mock(_cancelled=False)
        p._fit_worker = new
        with mock.patch.object(p, "_save_results") as save:
            p._on_all_fits_done([{"success": True, "run_number": "1"}],
                                mock.Mock(), {}, worker=old)
        save.assert_not_called()


# ── binning / merging ────────────────────────────────────────────────


class _Frame:
    """What _per_step_frequency_bins reads: a per-event Sorted frame."""

    def __init__(self, df):
        self.Sorted = df


def _events():
    """20 steps x 5 photons; every photon of steps 0, 7, 12 is outside
    the ToF gate (measured, nothing counted)."""
    rows = []
    for step in range(20):
        tof = 10.0 if step in (0, 7, 12) else 30.0
        for k in range(5):
            rows.append({"DV": float(step), "F": 1e8 + step * 1e7 + k,
                         "TOF": tof, "TDC": 3})
    return pd.DataFrame(rows)


class EmptyStepTests(unittest.TestCase):

    def _bins(self, keep):
        from gui.analysis.binning import _per_step_frequency_bins
        return _per_step_frequency_bins(_Frame(_events()), (20, 40), [3],
                                        None, None, keep_empty_steps=keep)

    def test_emptied_steps_stay_as_zeros(self):
        x, y, _ = self._bins(True)
        self.assertEqual(len(x), 20)
        self.assertEqual(int((y == 0).sum()), 3)
        self.assertEqual(y.sum(), 85)
        self.assertTrue(np.all(np.diff(x) > 0))

    def test_old_behaviour_still_available(self):
        x, y, _ = self._bins(False)
        self.assertEqual(len(x), 17)

    def test_default_keeps_them(self):
        from gui.analysis import binning
        self.assertTrue(binning.KEEP_EMPTY_STEPS_DEFAULT)


class CommonGridTests(unittest.TestCase):

    def test_each_source_keeps_only_what_it_measured(self):
        from gui.analysis.binning import rebin_to_common_grid
        reb, _centers, _w = rebin_to_common_grid(
            [([0.0, 1.0], [10.0, 10.0]), ([10.0, 11.0], [10.0, 10.0])],
            bin_width=1.0)
        for cx, cy in reb:
            self.assertEqual(len(cx), 2)
            self.assertEqual(float(np.mean(cy)), 10.0)


class XerrUnitTests(unittest.TestCase):

    def test_uniform_grid_xerr_is_the_frequency_spread(self):
        from gui.analysis import binning

        class _D:
            Size = 100

            def Compute_Bins(self, **kw):
                self.Binned = pd.DataFrame({
                    "bins_center": [1e8, 1.2e8, 1.4e8],
                    "Fcount": [5, 0, 7],
                    "Fstd": [5e6, np.nan, 6e6],     # Hz; NaN = empty
                    "Vstd": [0.3, np.nan, 0.35]})

        x, y, xerr, _ = binning._uniform_grid_bins(
            _D(), {"xerr_mode": "From voltage std"}, {}, "Auto", 0, 0.0,
            (None, None, None, None))
        self.assertAlmostEqual(xerr[0], 5.0)
        self.assertTrue(np.all(np.isfinite(xerr)))
        self.assertAlmostEqual(xerr[1], 10.0)     # half the bin width


class NarrowMergeTests(unittest.TestCase):

    def _merge(self, domain):
        x = np.linspace(-100, 100, 21)
        y = np.full_like(x, 5.0)
        md = {"merged_name": "narrow", "x": x, "y": y,
              "yerr": np.sqrt(y + 1), "x_unit": "MHz", "per_run": []}
        if domain:
            md["merge_params"] = {"domain": domain}
        return md

    def test_a_recorded_frequency_merge_round_trips(self):
        from gui.analysis.merge import export_merged_asdf, load_merged_asdf
        path = os.path.join(tempfile.mkdtemp(), "m.asdf")
        export_merged_asdf(self._merge("frequency"), path)
        back = load_merged_asdf(path)
        self.assertEqual(len(back["x"]), 21)

    def test_an_unrecorded_narrow_merge_is_still_refused(self):
        from gui.analysis.merge import export_merged_asdf
        with self.assertRaises(ValueError):
            export_merged_asdf(self._merge(None),
                               os.path.join(tempfile.mkdtemp(), "m.asdf"))


class VoltageMergeProjectionTests(unittest.TestCase):

    def _x(self, **cfg):
        from gui.analysis.merge import project_voltage_merge_to_frequency
        md = {"x": [1150.0, 1200.0, 1250.0], "y": [1.0, 2.0, 1.0],
              "yerr": [1.0, 1.0, 1.0],
              "merge_metadata": {"cooler_v": 30000.0, "laser_sp": 20000.0,
                                 "mass_amu": 100.0, "harmonic": 2}}
        src = {"ref_freq": 1.2e15}
        src.update(cfg)
        return np.asarray(project_voltage_merge_to_frequency(md, src)[0])

    def test_ref_shift_moves_the_axis(self):
        np.testing.assert_allclose(self._x(ref_shift=10e6),
                                   self._x() - 10.0, atol=1e-6)

    def test_cooler_offset_moves_the_axis(self):
        self.assertGreater(abs(self._x(cooler_offset_v=-20.0)[0]
                               - self._x()[0]), 100.0)


# ── GP / sessions ────────────────────────────────────────────────────


def _corrector():
    from cls_estimations.reference_correction import (
        ReferenceCorrector, ReferenceObservation)
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 6.0, 14)
    y = -100.0 - 20.0 * t + rng.normal(0, 0.4, t.size)
    rc = ReferenceCorrector(kernel="rbf", random_seed=0)
    rc.fit([ReferenceObservation(t=ti, centroid=yi, sigma=0.5)
            for ti, yi in zip(t, y)])
    return rc


class GpWindowSigmaTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rc = _corrector()

    def test_window_sigma_equals_the_applied_corrections(self):
        from cls_estimations.isotope_shift import mixed_mode_propagated_sigma
        _, sd = self.rc.predict_interval(2.0, 2.4)
        s = mixed_mode_propagated_sigma(
            self.rc, auto_t=np.array([[2.0, 2.4]]), auto_w=np.array([1.0]),
            manual_sigmas=[], manual_w=[], total_weight=1.0)
        self.assertAlmostEqual(s, sd, places=9)

    def test_a_zero_length_span_is_the_point(self):
        point = float(np.sqrt(self.rc.cov([3.0], [3.0])[0, 0]))
        span = float(np.sqrt(self.rc.cov_spans([[3.0, 3.0]],
                                               [[3.0, 3.0]])[0, 0]))
        self.assertAlmostEqual(point, span, places=12)

    def test_the_tab_passes_spans(self):
        from gui.analysis.isotope_shift_tab import _span_hours
        self.assertEqual(_span_hours({"ts_start": 3600.0,
                                      "ts_stop": 7200.0}), (1.0, 2.0))
        self.assertEqual(_span_hours({"ts_start": 3600.0}), (1.0, 1.0))
        self.assertIsNone(_span_hours({}))


class _StubAnalysisTab:
    _projects = []


class GpResetTests(unittest.TestCase):

    def test_a_session_without_a_gp_clears_the_last_one(self):
        from gui.analysis.reference_correction_panel import (
            ReferenceCorrectionPanel)
        panel = ReferenceCorrectionPanel(_StubAnalysisTab())
        panel.from_dict({"kernel": "rbf",
                         "corrector": _corrector().to_dict()})
        self.assertIsNotNone(panel.corrector)
        for nxt in ({"kernel": "rbf", "corrector": None}, {}):
            panel.from_dict({"kernel": "rbf",
                             "corrector": _corrector().to_dict()})
            panel.from_dict(nxt)
            self.assertIsNone(panel.corrector)
            self.assertIsNone(panel._fitted_excluded)
            self.assertFalse(panel._send_btn.isEnabled())
            self.assertIsNone(panel.to_dict().get("corrector"))


class ResultsTabSessionTests(unittest.TestCase):

    def test_clear_session_forgets_everything(self):
        from gui.results_tab import ResultsTab
        rt = ResultsTab()
        rt._results_data = {"116Sn": {"iter_001": {"results": []}}}
        rt.clear_session()
        self.assertEqual(rt._results_data, {})


class IterationChoiceTests(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.pdir = os.path.join(self.root, "analysis", "115Sn_A")
        for name, c in (("iter_001", 80.0), ("iter_002", 90.0)):
            d = os.path.join(self.pdir, name)
            os.makedirs(d)
            pd.DataFrame({"run_number": ["1001"], "Parameter": ["centroid"],
                          "Value": [c], "Error": [1.0],
                          "Source": ["Run_1001"], "Model": ["M"]}).to_csv(
                os.path.join(d, "parameters.csv"), index=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _load(self, recorded):
        from gui.analysis.project import AnalysisProject
        p = AnalysisProject("115Sn_A")
        p._session_iterations = list(recorded)
        with mock.patch("gui.shared_widgets.get_analysis_dir",
                        return_value=os.path.join(self.root, "analysis")):
            p.load_results_from_disk()
        return p

    def test_the_sessions_own_iteration_is_read(self):
        p = self._load(["iter_001"])
        self.assertEqual(p._loaded_iteration, "iter_001")
        self.assertEqual(p._last_results[0]["params_df"]["Value"], [80.0])

    def test_newest_only_when_nothing_is_recorded(self):
        self.assertEqual(self._load([])._loaded_iteration, "iter_002")

    def test_old_simultaneous_csv_rows_are_split_by_source(self):
        d = os.path.join(self.pdir, "iter_003")
        os.makedirs(d)
        pd.DataFrame({"run_number": ["1", "1", "2", "2"],
                      "Parameter": ["centroid"] * 4,
                      "Value": [100.0, 140.0, 100.0, 140.0],
                      "Error": [1.0] * 4,
                      "Source": ["Run_1", "Run_2", "Run_1", "Run_2"],
                      "Model": ["M"] * 4}).to_csv(
            os.path.join(d, "parameters.csv"), index=False)
        p = self._load(["iter_003"])
        vals = {r["run_number"]: r["params_df"]["Value"]
                for r in p._last_results}
        self.assertEqual(vals, {"1": [100.0], "2": [140.0]})


if __name__ == "__main__":
    unittest.main()
