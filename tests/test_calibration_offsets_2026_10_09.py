"""The calibration-offset changes of 2026-10-09.

* The voltage-calibration dialog: light tooltips like the rest of the
  app, a dark canvas on request, and plots that read better (per-point
  hover, an order colour bar, residual scales set by the points that
  count, clicks that hit the point under the pointer).
* The Yb calibration: four ways to pick the one offset every run is
  analysed at (intersection, middle of the zero crossings, their
  weighted mean, manual), with the zero crossings -- optionally widened
  by their 1 sigma -- as the systematic range.
* The Systematics tab: the three-offset batch (common offset + the two
  zero crossings), extra systematics, and how they are added up.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_calibration_offsets_2026_10_09.py -q
"""
import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QToolTip,
)

_APP = QApplication.instance() or QApplication([])

import gui.calibration as calibration  # noqa: E402
import gui.calibration_dialog as caldlg  # noqa: E402
from cls_estimations.cooler_calibration import (  # noqa: E402
    COMMON_INTERSECTION, COMMON_MANUAL, COMMON_MIDRANGE, COMMON_WEIGHTED,
    CalibrationResult, common_offset, systematic_range,
)
from cls_estimations.systematics import (  # noqa: E402
    HALF_WIDTH, LINEAR, MAX_DEV, MAX_DEV_FLAT, QUADRATURE, ExtraSystematic,
    ParameterBand, ScanValue, budget, combine, mean_over_runs, summarise,
)
from gui.analysis.systematic_scan import Baseline, value_key  # noqa: E402
from gui.analysis.systematics_tab import (  # noqa: E402
    MODE_CALIBRATION, MODE_RANGE, SystematicsTab,
)

# ── a synthetic calibration table ────────────────────────────────────
N_POINTS = 41
_SET = np.arange(N_POINTS, dtype=float)
_rng = np.random.default_rng(7)
_READ_V = -1.0 + 0.99 * _SET + _rng.normal(0.0, 0.05, N_POINTS)
_READ_V[0] += 5.0            # a supply still settling at the start
_READ_RAW = _READ_V / calibration.DEFAULT_VACC_DIV
PATH = os.path.join(os.path.abspath(os.sep), "nowhere", "run_1999.asdf")
PHYSICS = {"mass_amu": 100.0, "laser_cm": 20000.0, "cooler_v": 30000.0,
           "harmonic": 2}


def _fake_points(_path):
    return _SET.copy(), _READ_RAW.copy()


class _DialogCase(unittest.TestCase):
    """A CalibrationDialog on the synthetic table, settings untouched."""

    def setUp(self):
        for mod in (calibration, caldlg):
            p = mock.patch.object(mod, "read_cal_points", _fake_points)
            p.start()
            self.addCleanup(p.stop)
        calibration.get_registry().clear(PATH)
        self.addCleanup(calibration.get_registry().clear, PATH)

    def dialog(self, parent=None, physics=PHYSICS):
        dlg = caldlg.CalibrationDialog(PATH, run_label="run_1999",
                                       physics=physics, parent=parent)
        self.addCleanup(dlg.deleteLater)
        return dlg


def _tip_colour(widget):
    """Background of the tooltip Qt shows for *widget* (hex)."""
    QToolTip.showText(widget.mapToGlobal(QPoint(5, 5)),
                      "a probe tooltip long enough to see", widget)
    _APP.processEvents()
    labels = [w for w in _APP.topLevelWidgets()
              if w.objectName() == "qtooltip_label"]
    try:
        if not labels:
            return None
        img = labels[0].grab().toImage()
        return img.pixelColor(img.width() // 2, 2).name()
    finally:
        QToolTip.hideText()
        _APP.processEvents()


class _Win98:
    """Apply the Windows 98 theme for one test, and put the app back
    after."""

    def __enter__(self):
        from gui.theme import apply_theme
        self._sheet = _APP.styleSheet()
        self._palette = _APP.palette()
        self._font = _APP.font()
        apply_theme(_APP, "win98")
        return self

    def __exit__(self, *_exc):
        _APP.setStyleSheet(self._sheet)
        _APP.setPalette(self._palette)
        _APP.setFont(self._font)
        return False


class TooltipTests(_DialogCase):
    """The tooltips in the dialog were dark grey."""

    BUTTERMILK = "#ffffe1"

    def _selected_row(self):
        from gui.preanalysis_tab import FileEntry
        win = QMainWindow()
        self.addCleanup(win.deleteLater)
        fe = FileEntry(filepath=PATH, run_number="1999", cooler_v=30000.0,
                       date="2026-10-09", laser_sp=20000.0, mass_amu=100.0)
        win.setCentralWidget(fe)
        fe.selected = True      # FileEntry, FileEntry * { background }
        return win, fe

    def test_a_row_opens_its_dialogs_on_the_window(self):
        win, fe = self._selected_row()
        self.assertIs(fe._dialog_parent(), win)

    def test_the_calibration_dialog_is_opened_on_the_window(self):
        win, fe = self._selected_row()
        seen = {}

        class _Fake:
            def __init__(self, *a, parent=None, **k):
                seen["parent"] = parent

            def exec(self):
                return 0

        with mock.patch.object(caldlg, "CalibrationDialog", _Fake):
            fe._open_calibration_dialog()
        self.assertIs(seen["parent"], win)

    def test_tooltips_over_a_selected_rows_dialog_are_light(self):
        with _Win98():
            _win, fe = self._selected_row()
            dlg = self.dialog(parent=fe._dialog_parent())
            dlg.show()
            _APP.processEvents()
            self.assertEqual(_tip_colour(dlg._table), self.BUTTERMILK)

    def test_the_impact_box_does_not_darken_its_tooltip(self):
        with _Win98():
            dlg = self.dialog()
            dlg.show()
            _APP.processEvents()
            self.assertEqual(_tip_colour(dlg._impact), self.BUTTERMILK)

    def test_the_canvas_has_no_paragraph_tooltip(self):
        """The help moved to the card title; the canvas reads points."""
        dlg = self.dialog()
        self.assertEqual(dlg._canvas.toolTip(), "")


class DarkCanvasTests(_DialogCase):

    def test_it_follows_the_pre_analysis_switch_until_set(self):
        from gui.shared_widgets import _load_settings, _save_settings
        s = _load_settings()
        s.pop(caldlg.DARK_SETTING, None)
        s["pa_dark_plots"] = True
        _save_settings(s)
        self.assertTrue(self.dialog()._dark)

    def test_the_toggle_paints_the_canvas_black_and_is_remembered(self):
        from gui.shared_widgets import _load_settings
        dlg = self.dialog()
        dlg._dark_btn.setChecked(False)
        self.assertEqual(dlg._ax_fit.get_facecolor()[:3], (1.0, 1.0, 1.0))
        dlg._dark_btn.setChecked(True)
        self.assertEqual(dlg._ax_fit.get_facecolor()[:3], (0.0, 0.0, 0.0))
        self.assertTrue(_load_settings()[caldlg.DARK_SETTING])
        self.assertTrue(self.dialog()._dark)

    def test_early_points_stay_visible_on_black(self):
        """viridis starts near-black; the dark map starts higher."""
        dlg = self.dialog()
        dlg._dark_btn.setChecked(True)
        first = dlg._order_cmap()(0.0)
        # viridis(0) is (0.27, 0.00, 0.33): 0.60 in all.
        self.assertGreater(sum(first[:3]), 0.85)


class PlotTests(_DialogCase):

    def test_the_order_colour_bar_says_first_and_last(self):
        dlg = self.dialog()
        dlg.resize(1377, 894)
        dlg.show()
        _APP.processEvents()
        dlg._dark_btn.setChecked(not dlg._dark)
        dlg._dark_btn.setChecked(not dlg._dark)
        dlg._canvas.draw()
        labels = [t.get_text() for t in dlg._cbar.ax.get_yticklabels()]
        self.assertEqual(labels, ["first", "last"])

    def test_a_dropped_glitch_does_not_set_the_scale(self):
        """Point 0 sits 5 V off; once it is left out the bottom panel
        is scaled to the rest and the glitch sits on its edge."""
        dlg = self.dialog()
        dlg._toggle_point(0)
        lo, hi = dlg._ax_res1.get_ylim()
        self.assertLess(hi, 1.0)
        _idx, _x, y = dlg._hit[dlg._ax_res1]
        self.assertAlmostEqual(abs(y[0]), 0.92 * hi)

    def test_a_click_reaches_the_point_under_the_pointer(self):
        dlg = self.dialog()
        dlg.resize(1000, 800)
        dlg._canvas.draw()
        dlg._toggle_point(3)            # one left out already
        dlg._canvas.draw()
        ax = dlg._ax_res1
        _idx, x, y = dlg._hit[ax]
        px, py = ax.transData.transform((x[20], y[20]))
        dlg._on_click(SimpleNamespace(inaxes=ax, x=px, y=py, button=1))
        self.assertIn(20, set(dlg._result.excluded))
        self.assertIn(3, set(dlg._result.excluded))

    def test_hovering_a_point_reads_it_out_in_mhz(self):
        dlg = self.dialog()
        self.assertIsNotNone(dlg._mhz_per_v)
        self.assertGreater(dlg._mhz_per_v, 1.0)
        text = dlg._point_text(5)
        self.assertIn("Point #5", text)
        self.assertIn("from your line", text)
        self.assertIn("MHz", text)

    def test_without_physics_there_is_no_mhz_axis(self):
        dlg = self.dialog(physics={})
        self.assertIsNone(dlg._mhz_per_v)
        self.assertEqual(dlg._sec_axes, [])

    def test_redraws_do_not_pile_up_mhz_axes(self):
        dlg = self.dialog()
        for _ in range(3):
            dlg._draw()
        self.assertEqual(len(dlg._sec_axes), 2)
        self.assertEqual(len(dlg._ax_res0.child_axes), 1)
        self.assertEqual(len(dlg._ax_res1.child_axes), 1)

    def test_picking_a_row_rings_its_point(self):
        dlg = self.dialog()
        before = len(dlg._ax_fit.collections)
        dlg._table.selectRow(12)
        self.assertEqual(dlg._highlight, 12)
        self.assertEqual(len(dlg._ax_fit.collections), before + 1)


# ── the common offset ────────────────────────────────────────────────

# Two synthetic isotope lines, (A - A_lit) = a + b dV, with independent
# errors on a and b. Every expected number below is worked out from
# them by hand: 171Yb crosses zero at -30 V, 173Yb at -10 V.
LINE_171 = dict(a=-15.0, sa=1.6, b=-0.5, sb=0.04)
LINE_173 = dict(a=2.5, sa=0.6, b=0.25, sb=0.045)


def _zero(a, sa, b, sb):
    """Zero crossing -a/b, with sigma^2 = (sa/b)^2 + (a sb/b^2)^2."""
    return -a / b, math.hypot(sa / b, a * sb / b ** 2)


def _crossing(l1, l2):
    """Where two independent lines meet, x = (a2 - a1) / (b1 - b2), with
    sigma^2 = (sa1^2 + x^2 sb1^2 + sa2^2 + x^2 sb2^2) / (b1 - b2)^2."""
    db = l1["b"] - l2["b"]
    x = (l2["a"] - l1["a"]) / db
    var = (l1["sa"] ** 2 + x ** 2 * l1["sb"] ** 2
           + l2["sa"] ** 2 + x ** 2 * l2["sb"] ** 2) / db ** 2
    return x, math.sqrt(var)


Z171 = _zero(**LINE_171)                  # -30 +/- 4 V (3.2 and 2.4)
Z173 = _zero(**LINE_173)                  # -10 +/- 3 V (2.4 and 1.8)
CROSS = _crossing(LINE_171, LINE_173)     # -23.33 V
WIDE_LO, WIDE_HI = Z171[0] - Z171[1], Z173[0] + Z173[1]   # -34, -7 V


def _cal_result(z171=Z171, z173=Z173, dv=CROSS):
    """A calibration whose isotope lines are LINE_171 and LINE_173."""
    iso = [SimpleNamespace(isotope="171Yb", zero=z171[0],
                           zero_sigma=z171[1]),
           SimpleNamespace(isotope="173Yb", zero=z173[0],
                           zero_sigma=z173[1])]
    return CalibrationResult(isotopes=iso, dv=dv[0], dv_sigma=dv[1],
                             scan_lo=min(z171[0], z173[0]),
                             scan_hi=max(z171[0], z173[0]))


class CommonOffsetTests(unittest.TestCase):
    """Each way of picking the common offset, against the arithmetic
    done by hand on the synthetic lines above."""

    def test_the_range_is_the_zero_crossings(self):
        self.assertEqual(systematic_range(_cal_result()),
                         (Z171[0], Z173[0]))

    def test_widened_each_end_moves_out_by_its_own_sigma(self):
        lo, hi = systematic_range(_cal_result(), widen=True)
        self.assertAlmostEqual(lo, WIDE_LO)
        self.assertAlmostEqual(hi, WIDE_HI)

    def test_the_order_of_the_isotopes_does_not_matter(self):
        r = _cal_result(z171=Z173, z173=Z171)
        lo, hi = systematic_range(r, widen=True)
        self.assertAlmostEqual(lo, WIDE_LO)
        self.assertAlmostEqual(hi, WIDE_HI)

    def test_the_middle_of_the_widened_range(self):
        c = common_offset(_cal_result(), COMMON_MIDRANGE, widen=True)
        self.assertAlmostEqual(c.value, (WIDE_LO + WIDE_HI) / 2)     # -20.5
        self.assertAlmostEqual(c.half_range, (WIDE_HI - WIDE_LO) / 2)  # 13.5

    def test_the_weighted_mean_of_the_crossings(self):
        """Inverse-variance mean of the two crossings, its error, the
        chi2 of the pair about it and the Birge-scaled error: here
        -17.2 +/- 2.4 V, chi2 = 16, scaled 9.6 V."""
        (z1, s1), (z2, s2) = Z171, Z173
        w1, w2 = 1.0 / s1 ** 2, 1.0 / s2 ** 2
        mean = (w1 * z1 + w2 * z2) / (w1 + w2)
        sigma = 1.0 / math.sqrt(w1 + w2)
        chi2 = w1 * (z1 - mean) ** 2 + w2 * (z2 - mean) ** 2
        # For two values that is (z1 - z2)^2 / (s1^2 + s2^2).
        self.assertAlmostEqual(chi2, (z1 - z2) ** 2 / (s1 ** 2 + s2 ** 2))
        self.assertGreater(chi2, 1.0)       # so the error is inflated
        c = common_offset(_cal_result(), COMMON_WEIGHTED)
        self.assertAlmostEqual(c.value, mean)
        self.assertAlmostEqual(c.sigma, sigma)
        self.assertAlmostEqual(c.chi2, chi2)
        # One degree of freedom: scaled by sqrt(chi2 / 1).
        self.assertAlmostEqual(c.sigma_scaled, sigma * math.sqrt(chi2))

    def test_the_intersection(self):
        c = common_offset(_cal_result(), COMMON_INTERSECTION)
        self.assertEqual((c.value, c.sigma), CROSS)

    def test_the_range_does_not_depend_on_the_method(self):
        for m in (COMMON_INTERSECTION, COMMON_MIDRANGE, COMMON_WEIGHTED):
            c = common_offset(_cal_result(), m)
            self.assertEqual((c.lo, c.hi), (Z171[0], Z173[0]), m)

    def test_a_manual_value_needs_no_calibration(self):
        c = common_offset(None, COMMON_MANUAL, manual=-25.0)
        self.assertEqual(c.value, -25.0)
        self.assertIsNone(c.lo)

    def test_a_value_outside_the_range_is_said_so(self):
        c = common_offset(_cal_result(), COMMON_MANUAL, manual=-5.0)
        self.assertIn("outside", c.note)


class CoolerTabTests(unittest.TestCase):

    def _tab(self):
        from gui.analysis.cooler_calibration_tab import CoolerCalibrationTab
        tab = CoolerCalibrationTab(SimpleNamespace(_projects=[]))
        self.addCleanup(tab.deleteLater)
        tab._result = _cal_result()
        return tab

    def _pick(self, tab, method):
        tab._common.setCurrentIndex(tab._common.findData(method))

    def test_the_chosen_method_is_what_gets_applied(self):
        tab = self._tab()
        self._pick(tab, COMMON_MIDRANGE)
        tab._widen.setChecked(True)
        self.assertAlmostEqual(tab.common().value, (WIDE_LO + WIDE_HI) / 2,
                               places=3)
        self.assertTrue(tab._apply_btn.isEnabled())

    def test_manual_unlocks_its_box(self):
        tab = self._tab()
        self.assertFalse(tab._manual.isEnabled())
        self._pick(tab, COMMON_MANUAL)
        self.assertTrue(tab._manual.isEnabled())
        tab._manual.setValue(-20.0)
        self.assertEqual(tab.common().value, -20.0)

    def test_every_candidate_is_shown(self):
        tab = self._tab()
        tab._show_common()
        text = tab._result_label.text()
        for word in ("Intersection", "Middle", "Weighted mean"):
            self.assertIn(word, text)

    def test_the_plot_marks_the_common_offset_and_the_range(self):
        from matplotlib.figure import Figure
        from cls_estimations.cooler_calibration import LITERATURE_A
        from gui.analysis.cooler_calibration_tab import (
            CoolerCalibrationTab, draw_calibration)
        yb = [SimpleNamespace(project_name=f"{m}Yb_cal", is_calibration=True,
                              is_reference=False) for m in (171, 173)]
        tab = CoolerCalibrationTab(SimpleNamespace(_projects=yb))
        self.addCleanup(tab.deleteLater)
        pts = []
        for iso, slope, star in (("171Yb", -0.3, -30.0),
                                 ("173Yb", 0.1, -10.0)):
            for dv in np.linspace(-50, 50, 9):
                pts.append({"run": "1", "isotope": iso, "dv": float(dv),
                            "value": LITERATURE_A[iso] + slope * (dv - star),
                            "sigma": 0.3})
        tab.from_dict({"points": pts, "common_method": COMMON_MIDRANGE,
                       "widen": True})
        c = tab.common()
        d = tab.plot_data()
        self.assertAlmostEqual(d["common"], c.value)
        self.assertEqual((d["scan_lo"], d["scan_hi"]), (c.lo, c.hi))
        self.assertLess(c.lo, -30.0)          # widened past the crossing
        ax = draw_calibration(Figure(), d)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertTrue(any(lab.startswith("common offset") for lab in labels))
        self.assertTrue(any("widened" in lab for lab in labels))

    def test_the_choice_is_saved_and_an_old_session_keeps_the_crossing(self):
        tab = self._tab()
        self._pick(tab, COMMON_WEIGHTED)
        tab._widen.setChecked(True)
        tab._manual.setValue(-21.5)
        d = tab.to_dict()
        self.assertEqual((d["common_method"], d["widen"], d["manual_offset"]),
                         (COMMON_WEIGHTED, True, -21.5))
        other = self._tab()
        other.from_dict(d)
        self.assertEqual(other.common_method(), COMMON_WEIGHTED)
        self.assertTrue(other._widen.isChecked())
        old = self._tab()
        old.from_dict({"dv_min": -50.0})
        self.assertEqual(old.common_method(), COMMON_INTERSECTION)
        self.assertFalse(old._widen.isChecked())

    def test_applying_writes_the_chosen_offset(self):
        tab = self._tab()
        self._pick(tab, COMMON_MIDRANGE)
        seen = []
        tab.calibration_applied.connect(lambda v, s: seen.append((v, s)))
        with mock.patch.object(type(tab), "apply_offset",
                               lambda self, off, projs: seen.append(off) or 1):
            with mock.patch(
                    "gui.analysis.cooler_calibration_tab.QDialog.exec",
                    return_value=1), \
                 mock.patch("gui.analysis.cooler_calibration_tab"
                            ".QMessageBox.information"):
                tab._analysis_tab._projects = [SimpleNamespace(
                    project_name="112Sn", is_calibration=False,
                    is_reference=False)]
                tab._apply_dialog()
        self.assertAlmostEqual(seen[0], (Z171[0] + Z173[0]) / 2)


# ── systematics arithmetic ───────────────────────────────────────────

def _band(values, dvs=(-30.0, -20.0, -10.0), baseline=None, sigma=1.0,
          project="112Sn", run="", parameter="centroid"):
    pts = [ScanValue(dv=d, value=v, sigma=sigma) for d, v in zip(dvs, values)]
    base = values[1] if baseline is None else baseline
    return ParameterBand(project=project, run=run, parameter=parameter,
                         points=pts, baseline=base, baseline_sigma=sigma)


class BudgetTests(unittest.TestCase):

    def test_quadrature_and_linear(self):
        self.assertAlmostEqual(combine([3.0, 4.0], QUADRATURE), 5.0)
        self.assertAlmostEqual(combine([3.0, 4.0], LINEAR), 7.0)
        self.assertTrue(math.isnan(combine([float("nan")])))

    def test_the_max_deviation_takes_the_larger_edge(self):
        """max |edge - quoted|, signed edges kept."""
        b = _band([-120.0, -106.0, -92.5])
        self.assertAlmostEqual(b.systematic(MAX_DEV), 14.0)
        self.assertEqual(b.edge_deviations, (-14.0, 13.5))

    def test_uniform_one_sigma(self):
        b = _band([-120.0, -106.0, -92.5])
        self.assertAlmostEqual(b.systematic(MAX_DEV_FLAT),
                               14.0 / math.sqrt(3.0))

    def test_an_extra_applies_where_it_matches(self):
        e = ExtraSystematic("mass", 0.5, run="112Sn")
        self.assertTrue(e.matches("Isotope shifts", "112Sn", "delta_nu"))
        self.assertFalse(e.matches("Isotope shifts", "114Sn", "delta_nu"))
        self.assertFalse(ExtraSystematic("x", 1.0, use=False)
                         .matches("a", "b", "c"))

    def test_the_budget_adds_extras_as_chosen(self):
        b = _band([-120.0, -106.0, -92.0], sigma=3.0)    # cooler 14
        extras = [ExtraSystematic("scale", 5.5),
                  ExtraSystematic("other run", 99.0, run="114Sn")]
        q = budget(b, MAX_DEV, extras, QUADRATURE, QUADRATURE)
        self.assertEqual(q["extras"], [("scale", 5.5)])
        self.assertAlmostEqual(q["systematic"], math.hypot(14.0, 5.5))
        self.assertAlmostEqual(q["total"],
                               math.hypot(3.0, math.hypot(14.0, 5.5)))
        lin = budget(b, MAX_DEV, extras, LINEAR, LINEAR)
        self.assertAlmostEqual(lin["systematic"], 19.5)
        self.assertAlmostEqual(lin["total"], 22.5)

    def test_without_extras_the_total_is_unchanged(self):
        b = _band([-120.0, -106.0, -92.0], sigma=3.0)
        row = summarise([b], HALF_WIDTH)[0]
        self.assertAlmostEqual(row["total"], b.total(HALF_WIDTH))
        self.assertTrue(math.isnan(row["other"]))

    def test_birge_scales_the_run_average(self):
        a = _band([0.0, 0.0, 0.0], sigma=1.0, run="1")
        b = _band([10.0, 10.0, 10.0], sigma=1.0, run="2")
        plain = mean_over_runs([a, b])
        scaled = mean_over_runs([a, b], birge=True)
        # chi2/ndf = 50 about the mean of 0 and 10
        self.assertAlmostEqual(scaled.baseline_sigma,
                               plain.baseline_sigma * math.sqrt(50.0))
        self.assertAlmostEqual(scaled.points[0].sigma,
                               plain.points[0].sigma * math.sqrt(50.0))


# ── the three-offset batch in the Systematics tab ────────────────────

PROJECT = "sample_A"
MODEL = "HFS_1"
SOURCES = ("Run_1001", "Run_1002")
LO, COMMON, HI = -30.0, -22.0, -10.0     # off-centre: the edges differ


class _Stub:
    def __init__(self, name, reference=False):
        self.project_name = name
        self.is_reference = reference
        self.is_calibration = False

    def to_dict(self):
        return {"project_name": self.project_name, "blocks": []}


class _StubAnalysisTab:
    projects_changed = None

    def __init__(self, projects=(), cal_tab=None):
        self._projects = list(projects)
        self._cal_tab = cal_tab
        self._gp_tab = None
        self._is_tab = None


def _records(offsets=(LO, COMMON, HI), slope=20.0, base=-500.0):
    out = {}
    for dv in offsets:
        values = {}
        for i, src in enumerate(SOURCES):
            values[value_key(PROJECT, src, MODEL, "centroid")] = {
                "value": base + 10 * i + slope * dv, "sigma": 2.0}
        out[float(dv)] = {
            "dv": float(dv), "status": "ok", "message": "",
            "values": values, "redchi": {}, "shifts": [],
            "labels": {f"{PROJECT}|{s}": f"100{1 + i}"
                       for i, s in enumerate(SOURCES)},
            "iterations": {}}
    return out


class BatchTests(unittest.TestCase):

    def _tab(self, cal_tab=None):
        tab = SystematicsTab(_StubAnalysisTab([_Stub(PROJECT)], cal_tab))
        self.addCleanup(tab.deleteLater)
        tab._dv_min.setValue(LO)
        tab._dv_max.setValue(HI)
        tab._baseline_dv.setValue(COMMON)
        return tab

    def test_it_is_the_default_and_quotes_the_max_deviation(self):
        tab = self._tab()
        self.assertEqual(tab._mode.currentText(), MODE_CALIBRATION)
        self.assertEqual(tab._definition.currentText(), MAX_DEV)

    def test_every_run_is_fitted_three_times(self):
        self.assertEqual(self._tab().offsets(), [LO, COMMON, HI])

    def test_a_common_offset_on_an_edge_is_fitted_once(self):
        tab = self._tab()
        tab._baseline_dv.setValue(LO)
        self.assertEqual(tab.offsets(), [LO, HI])

    def test_the_quoted_value_is_the_fit_at_the_common_offset(self):
        """Not the baseline iteration: that may be from another frame."""
        tab = self._tab()
        tab._baselines[PROJECT] = Baseline(params={
            SOURCES[0]: {(MODEL, "centroid"): {"value": 12345.0,
                                               "sigma": 1.0}}})
        tab._records = _records()
        tab.compute()
        runs, mean = tab.bands(PROJECT, MODEL, "centroid")
        self.assertAlmostEqual(runs[0].baseline, -500.0 + 20.0 * COMMON)
        self.assertAlmostEqual(mean.baseline, -495.0 + 20.0 * COMMON)
        # the larger edge deviation: 20 MHz/V x 12 V
        self.assertAlmostEqual(mean.systematic(MAX_DEV),
                               20.0 * (HI - COMMON))

    def test_no_fit_at_the_common_offset_means_no_value(self):
        tab = self._tab()
        tab._records = _records()
        tab._excluded = {COMMON}
        tab.compute()
        runs, _mean = tab.bands(PROJECT, MODEL, "centroid")
        self.assertTrue(math.isnan(runs[0].baseline))

    def test_the_steps_say_what_they_are_for(self):
        tab = self._tab()
        tab._records = _records()
        tab.compute()
        status = [tab._step_table.item(r, 2).text() for r in range(3)]
        self.assertEqual(status, ["ok · low edge", "ok · quoted",
                                  "ok · high edge"])

    def test_it_takes_the_offsets_from_the_calibration_tab(self):
        cal = SimpleNamespace(_result=_cal_result(), common=lambda: common_offset(
            _cal_result(), COMMON_MIDRANGE, widen=True))
        tab = self._tab(cal_tab=cal)
        tab._range_from_calibration()
        self.assertAlmostEqual(tab._dv_min.value(), WIDE_LO, places=3)
        self.assertAlmostEqual(tab._dv_max.value(), WIDE_HI, places=3)
        self.assertAlmostEqual(tab._baseline_dv.value(),
                               (WIDE_LO + WIDE_HI) / 2, places=3)

    def test_extras_enter_the_summary_as_chosen(self):
        tab = self._tab()
        tab._records = _records()
        tab._add_extra(ExtraSystematic("scale", 5.0))
        tab._combine_sys.setCurrentText(LINEAR)
        tab.compute()
        row = [r for r in tab.summary_rows() if r["run"] == "<weighted mean>"][0]
        cooler = 20.0 * (HI - COMMON)
        self.assertAlmostEqual(row["systematic"], cooler)
        self.assertAlmostEqual(row["other"], 5.0)
        self.assertAlmostEqual(row["sys_total"], cooler + 5.0)
        self.assertIn("scale", tab._report_text())

    def test_the_birge_box_reaches_the_stat_error(self):
        tab = self._tab()
        tab._records = _records()
        tab.compute()
        plain = tab.bands(PROJECT, MODEL, "centroid")[1].baseline_sigma
        tab._birge.setChecked(True)
        scaled = tab.bands(PROJECT, MODEL, "centroid")[1].baseline_sigma
        self.assertGreater(scaled, plain)

    def test_the_new_settings_survive_a_save(self):
        tab = self._tab()
        tab._records = _records()
        tab._add_extra(ExtraSystematic("mass", 0.5, run="112Sn",
                                       parameter="delta_nu"))
        tab._combine_sys.setCurrentText(LINEAR)
        tab._birge.setChecked(True)
        tab._definition.setCurrentText(MAX_DEV_FLAT)
        d = tab.to_dict()
        other = self._tab()
        other.from_dict(d)
        self.assertEqual(other._mode.currentText(), MODE_CALIBRATION)
        self.assertEqual(other._combine_sys.currentText(), LINEAR)
        self.assertTrue(other._birge.isChecked())
        # Loading must not reset the definition to the mode's default.
        self.assertEqual(other._definition.currentText(), MAX_DEV_FLAT)
        self.assertEqual([e.to_dict() for e in other.extras()],
                         [e.to_dict() for e in tab.extras()])

    def test_an_old_session_is_a_range_scan(self):
        tab = self._tab()
        tab.from_dict({"dv_min": -30.0, "dv_max": -10.0, "steps": 5,
                       "definition": HALF_WIDTH})
        self.assertEqual(tab._mode.currentText(), MODE_RANGE)
        self.assertEqual(tab._definition.currentText(), HALF_WIDTH)
        self.assertEqual(len(tab.offsets()), 5)


if __name__ == "__main__":
    unittest.main()
