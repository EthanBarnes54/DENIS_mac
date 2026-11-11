"""The generalised systematics engine (Analysis Upgrade, card E1; WP-S3, DA3).

Each enabled source is propagated by refitting the whole chain at -1 and
+1 sigma of its perturbation (the cooler-offset scan's own engine, which
stays the default); each value's response is half the difference, sign
kept. A source shared by every run moves the published mean by the
weighted mean of the runs' responses; one independent per run adds them in
quadrature. On request, a Monte Carlo of the shared sources together.

The stub projects respond linearly to any perturbation, so every expected
number is a slope times a sigma.

Run from the project root:
    .venv/bin/python -m pytest tests/test_source_budget.py -q
"""
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from cls_estimations.systematics import (  # noqa: E402
    Combine_Runs, Draw_Sources, Systematic_Source)
from gui.analysis.systematic_scan import (  # noqa: E402
    Baseline, ScanTarget, Source_Budget_Scan, Source_Budget_Table,
    SystematicScan, value_key)

SHIFTS = "Isotope shifts"


class _Worker:
    def isRunning(self):
        return True


class _Linear_Project(QObject):
    """A 'fit' whose centroid per run is base + sum(slope x perturbation),
    statistical sigma 0.5; its Source block holds ``own_offset``."""
    results_ready = Signal(str, list, dict)

    def __init__(self, name, slopes, base=100.0, own_offset=0.0):
        super().__init__()
        self.project_name = name
        self._slopes = slopes            # {run: {key: slope}}
        self._base = base
        self._own_offset = own_offset
        self._fit_worker = _Worker()
        self._config_overrides = {}
        self._last_iter_dir = ""
        self.seen = []
        self.labels = []
        self.last = {}

    def _get_blocks_by_type(self, _cls):
        own = self._own_offset
        return [type("Src", (), {"get_source_config":
                                 lambda self: {"cooler_offset_v": own}})()]

    def _on_fit_requested(self):
        src = dict(self._config_overrides.get("source") or {})
        self.seen.append(dict(src))
        self.labels.append(
            (self._config_overrides.get("output") or {}).get("iter_label"))
        src["cooler_offset_v"] = src.get("cooler_offset_v",
                                         self._own_offset) - self._own_offset
        results = []
        for run, slopes in self._slopes.items():
            v = self._base + sum(s * float(src.get(k, 0.0))
                                 for k, s in slopes.items())
            self.last[run] = v
            results.append({
                "success": True, "run_number": run, "source_name": f"Run_{run}",
                "fit_quality": {"redchi": 1.0},
                "params_df": {"Source": [f"Run_{run}"], "Model": ["HFS_1"],
                              "Parameter": ["centroid"], "Value": [v],
                              "Stderr": [0.5]}})
        QTimer.singleShot(0, lambda: self.results_ready.emit(
            self.project_name, results, {}))


class _Shift_IS:
    """Isotope shifts from the stubs' latest fits: sample less reference,
    each the mean over its runs."""

    def __init__(self, sample, ref):
        self._sample, self._ref = sample, ref

    def shifts_for_scan(self):
        mean = lambda p: sum(p.last.values()) / len(p.last)
        return [{"label": "112Sn", "delta_nu": mean(self._sample)
                 - mean(self._ref), "sigma_delta_nu_stat": 0.5}]


def _Spin(engine, timeout_ms=20000):
    loop, out = QEventLoop(), {}
    engine.finished.connect(lambda r: (out.setdefault("result", r),
                                       loop.quit()))
    engine.failed.connect(lambda m: (out.setdefault("error", m), loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return out


def _Target(project, role="sample"):
    return ScanTarget(project=project, role=role, baseline=Baseline())


class Library_Tests(unittest.TestCase):

    def test_Shared_Adds_Linearly_Per_Run_In_Quadrature(self):
        self.assertAlmostEqual(Combine_Runs([2.0, -4.0], [0.5, 0.5],
                                            "campaign"), -1.0)
        self.assertAlmostEqual(Combine_Runs([2.0, -4.0], [0.5, 0.5], "run"),
                               math.hypot(1.0, 2.0))
        # Weighted as the published mean is: 1/sigma^2.
        self.assertAlmostEqual(Combine_Runs([2.0, 4.0], [1.0, 0.5],
                                            "campaign"), 3.6)

    def test_A_Source_Round_Trips_And_Reads_mV_As_V(self):
        src = Systematic_Source("Run calibration", 10.0, key="scan_axis_offset_v",
                                unit="mV", correlation="run")
        self.assertAlmostEqual(src.sigma, 0.01)
        self.assertEqual(Systematic_Source.from_dict(src.to_dict()), src)

    def test_Draws_Are_Shared_Per_Draw_And_Add_On_A_Key(self):
        sources = [Systematic_Source("a", 1.0, key="cooler_div_ppm",
                                     unit="ppm"),
                   Systematic_Source("b", 2.0, key="cooler_div_ppm",
                                     unit="ppm", correlation="region"),
                   Systematic_Source("c", 3.0, key="scan_axis_offset_v",
                                     correlation="run")]
        draws = Draw_Sources(sources, 4000, seed=1)
        self.assertEqual(set(draws[0]), {"cooler_div_ppm"})   # no per-run
        spread = math.sqrt(sum(d["cooler_div_ppm"]**2 for d in draws) / 4000)
        self.assertAlmostEqual(spread, math.sqrt(5.0), delta=0.1)
        self.assertEqual(draws[:3], Draw_Sources(sources, 3, seed=1))


class Engine_Tests(unittest.TestCase):

    def _Run(self, targets, sources, **kw):
        engine = Source_Budget_Scan()
        self.addCleanup(engine.deleteLater)
        engine.start(targets, sources, **kw)
        return _Spin(engine)

    def test_The_Cooler_Offset_Scan_Is_Unchanged_By_Default(self):
        p = _Linear_Project("P", {"1": {}})
        self.addCleanup(p.deleteLater)
        scan = SystematicScan()
        self.addCleanup(scan.deleteLater)
        loop = QEventLoop()
        scan.finished.connect(lambda _r: loop.quit())
        scan.start([_Target(p)], [-2.0, 3.0])
        loop.exec()
        self.assertEqual([s for s in p.seen], [{"cooler_offset_v": -2.0},
                                                {"cooler_offset_v": 3.0}])

    def test_Plus_And_Minus_One_Sigma_Per_Source(self):
        p = _Linear_Project("P", {"1": {"scan_div_ppm": 0.02},
                                  "2": {"scan_div_ppm": 0.04}})
        self.addCleanup(p.deleteLater)
        src = Systematic_Source("Scan divider", 100.0, key="scan_div_ppm",
                                unit="ppm", correlation="campaign")
        out = self._Run([_Target(p)], [src])["result"]
        d = out["deltas"]["Scan divider"]
        self.assertAlmostEqual(d[value_key("P", "Run_1", "HFS_1",
                                           "centroid")], 2.0)
        self.assertAlmostEqual(d[value_key("P", "Run_2", "HFS_1",
                                           "centroid")], 4.0)
        self.assertEqual(sorted(v.get("scan_div_ppm") for v in p.seen),
                         [-100.0, 100.0])
        # Each step has its own iterations, not one over the other.
        self.assertEqual(sorted(p.labels), ["sys_001_Scan_divider_+1s",
                                            "sys_001_Scan_divider_-1s"])

    def test_The_Table_Combines_By_Correlation(self):
        p = _Linear_Project("P", {"1": {"scan_div_ppm": 0.02,
                                        "scan_axis_offset_v": 100.0},
                                  "2": {"scan_div_ppm": 0.04,
                                        "scan_axis_offset_v": -200.0}})
        self.addCleanup(p.deleteLater)
        sources = [Systematic_Source("Scan divider", 100.0,
                                     key="scan_div_ppm", unit="ppm"),
                   Systematic_Source("Run calibration", 10.0,
                                     key="scan_axis_offset_v", unit="mV",
                                     correlation="run")]
        out = self._Run([_Target(p)], sources)["result"]
        row = next(r for r in out["table"] if r["project"] == "P")
        self.assertAlmostEqual(row["sources"]["Scan divider"], 3.0)
        self.assertAlmostEqual(row["sources"]["Run calibration"],
                               math.hypot(0.5, 1.0))
        self.assertAlmostEqual(row["total"], math.hypot(3.0, math.hypot(
            0.5, 1.0)))
        self.assertAlmostEqual(row["statistical"], 0.5 / math.sqrt(2))

    def test_The_Beam_Offset_Moves_Each_Project_s_Own(self):
        p = _Linear_Project("P", {"1": {"cooler_offset_v": 10.0}},
                            own_offset=-13.0)
        self.addCleanup(p.deleteLater)
        src = Systematic_Source("Beam offset", 8.0, key="cooler_offset_v")
        out = self._Run([_Target(p)], [src])["result"]
        self.assertEqual(sorted(s["cooler_offset_v"] for s in p.seen),
                         [-21.0, -5.0])
        self.assertAlmostEqual(out["table"][0]["sources"]["Beam offset"], 80.0)

    def test_Shifts_Cancel_A_Shared_Source_And_Not_A_Per_Run_One(self):
        slopes = {"1": {"cooler_div_ppm": 0.1, "scan_axis_offset_v": 50.0}}
        ref = _Linear_Project("116Sn", dict(slopes))
        sample = _Linear_Project("112Sn", dict(slopes))
        for p in (ref, sample):
            self.addCleanup(p.deleteLater)
        sources = [Systematic_Source("Cooler divider", 100.0,
                                     key="cooler_div_ppm", unit="ppm"),
                   Systematic_Source("Run calibration", 10.0,
                                     key="scan_axis_offset_v", unit="mV",
                                     correlation="run")]
        out = self._Run([_Target(ref, "reference"), _Target(sample)],
                        sources, is_tab=_Shift_IS(sample, ref))["result"]
        shift = next(r for r in out["table"] if r["project"] == SHIFTS)
        # One divider moves both isotopes alike: the shift does not move.
        self.assertAlmostEqual(shift["sources"]["Cooler divider"], 0.0)
        # A run's calibration is its own: needs the projects behind it.
        self.assertTrue(math.isnan(shift["sources"]["Run calibration"]))
        mapped = Source_Budget_Table(
            out["deltas"], out["sigmas"], sources,
            shift_projects={"112Sn": ("112Sn", "116Sn")})
        shift = next(r for r in mapped if r["project"] == SHIFTS)
        self.assertAlmostEqual(shift["sources"]["Run calibration"],
                               math.hypot(0.5, 0.5))

    def test_The_Monte_Carlo_Agrees_With_The_Linear_Total(self):
        p = _Linear_Project("P", {"1": {"cooler_div_ppm": 0.03,
                                        "scan_div_ppm": 0.04}})
        self.addCleanup(p.deleteLater)
        sources = [Systematic_Source("Cooler divider", 100.0,
                                     key="cooler_div_ppm", unit="ppm"),
                   Systematic_Source("Scan divider", 100.0,
                                     key="scan_div_ppm", unit="ppm")]
        out = self._Run([_Target(p)], sources, mc_draws=300, seed=2)["result"]
        row = out["table"][0]
        self.assertAlmostEqual(row["total"], 5.0)            # 3 (+) 4
        self.assertAlmostEqual(row["mc_shared"], 5.0, delta=5.0 * 0.15)
        self.assertEqual(len([s for s in p.seen
                              if "cooler_div_ppm" in s and "scan_div_ppm"
                              in s]), 300)


if __name__ == "__main__":
    unittest.main()
