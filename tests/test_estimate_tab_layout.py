"""Estimate tab layout: one surface, three columns, embedded plots.

Date:    2026-06-02 (rewritten 2026-07-24 for the UI overhaul Phase 3)
Version: 2.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Pins the single-surface Estimate tab: no sub-tabs; a horizontal
three-column splitter (Element/Transition | Isotope list | Run column);
Run Options sit under the isotope list, and the run column stacks the
plots panel, the Voltage Range Estimator and a shortened log in a
vertical splitter; the ⛶ toggle expands the plots over the whole tab;
the plots canvas fits its column on Retina and scrolls with the wheel;
the estimator agrees with the pipeline and writes back only Laser and
Scan; isotope panels carry per-position color stripes that survive
reference toggling; and the historical ``run_tab`` / ``plots_tab``
handles keep working (main_window relies on them).

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_estimate_tab_layout -v

Depends on: gui.estimate_tab (EstimateTab, GlobalParamsWidget,
IsotopeListWidget, RunTab, PlotsTab, ISOTOPE_COLORS); PySide6.
"""

import unittest

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QSplitter, QTabWidget


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class EstimateTabStructureTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_no_subtabs_single_surface(self):
        """The Plots sub-tab is gone: the tab hosts no QTabWidget."""
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        self.assertFalse(hasattr(tab, "tabs"))
        self.assertIsNone(tab.findChild(QTabWidget))

    def test_three_column_splitter(self):
        from gui.estimate_tab import (
            EstimateTab, GlobalParamsWidget, IsotopeListWidget, RunTab)
        tab = EstimateTab()
        sp = tab.params_tab._main_splitter
        self.assertIsInstance(sp, QSplitter)
        self.assertEqual(sp.count(), 3)
        self.assertIsInstance(sp.widget(0), GlobalParamsWidget)
        self.assertIsInstance(sp.widget(1), IsotopeListWidget)
        self.assertIsInstance(sp.widget(2), RunTab)

    def test_column_width_caps(self):
        """Columns 1-2 are capped so the plots get the freed width.

        Both bands are measured from the content's layout minimum
        (font/DPI dependent), so assert the invariants rather than
        pixels: wide enough that nothing can clip, and no more than a
        small slack above that.
        """
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        gp = tab.params_tab.global_params
        need = gp.preferred_width()
        self.assertGreaterEqual(gp.minimumWidth(), need)
        self.assertLessEqual(gp.maximumWidth(), need + 60)
        il = tab.params_tab.isotope_list
        need2 = il.preferred_width()
        self.assertGreaterEqual(il.minimumWidth(), need2)
        self.assertLessEqual(il.maximumWidth(), need2 + 100)

    def test_isotope_panel_never_clips_at_column_minimum(self):
        """Regression: at the column's minimum width, the widest panel
        state (timing mode + isomer) must fit the scroll viewport --
        the TOF Gate spin and the header Remove button used to be cut
        off at the right edge (hardcoded 320-460 band, 2026-09-15)."""
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        il = tab.params_tab.isotope_list
        il.add_isotope()
        panel = il.panels[-1]
        panel.bg_tof_radio.setChecked(True)
        panel.isomer_check.setChecked(True)
        tab.resize(1600, 900)
        tab.show()
        QCoreApplication.processEvents()
        # Pin the splitter at the measured minimums.
        sp = tab.params_tab._main_splitter
        sp.setSizes([tab.params_tab.global_params.minimumWidth(),
                     il.minimumWidth(), 900])
        QCoreApplication.processEvents()
        viewport = il._scroll.viewport()
        self.assertLessEqual(panel.width(), viewport.width() + 1,
                             "isotope panel wider than the viewport")
        for w in (panel.remove_btn, panel.bg_gate, panel.bg_cont):
            right = w.mapTo(viewport, w.rect().topRight()).x()
            self.assertLessEqual(
                right, viewport.width() + 1,
                f"{w.objectName() or w.__class__.__name__} clips at "
                f"x={right} (viewport {viewport.width()})")
        tab.close()

    def test_Run_Column_Hosts_Plots_Estimator_Log(self):
        """Column 3: plots | optimiser tools | log splitter, the tools being
        the voltage range, saturation and efficiency estimators in one
        group box, a title each (2026-10-25). Run Options moved under the
        isotope list, fixed below its scroll area."""
        from gui.estimate_tab import (
            EstimateTab, Efficiency_Estimator, PlotsTab,
            Saturation_Estimator, Voltage_Range_Estimator)
        from gui.theme import Title_Tabs
        tab = EstimateTab()
        vs = tab.run_tab._vsplit
        self.assertEqual(vs.count(), 3)
        self.assertIsInstance(vs.widget(0), PlotsTab)
        tabs = vs.widget(1)
        self.assertIs(tabs, tab.params_tab.Optimiser_Tabs)
        self.assertIsInstance(tabs, Title_Tabs)
        self.assertGreater(tabs.minimumSizeHint().width(),   # never cropped
                           tabs._Layout()[-1][1].right())
        self.assertEqual([tabs.tabText(i) for i in range(tabs.count())],
                         ["Voltage Range Estimator", "Saturation Estimator",
                          "Efficiency Estimator"])
        self.assertIs(tabs.widget(0), tab.Voltage_Estimator)
        self.assertIsInstance(tabs.widget(0), Voltage_Range_Estimator)
        self.assertIsInstance(tabs.widget(1), Saturation_Estimator)
        self.assertIsInstance(tabs.widget(2), Efficiency_Estimator)
        self.assertIs(vs.widget(2), tab.run_tab._log_host)
        # Options host sits OUTSIDE the splitter, under the isotope list.
        il = tab.params_tab.isotope_list
        lay = il.layout()
        self.assertNotIn(tab.run_tab._top_host,
                         [vs.widget(i) for i in range(vs.count())])
        self.assertTrue(il.isAncestorOf(tab.run_tab._top_host))
        self.assertGreater(lay.indexOf(tab.run_tab._top_host),
                           lay.indexOf(il._scroll))
        self.assertTrue(hasattr(tab.run_tab, "load_requested"))
        self.assertTrue(
            tab.run_tab._log_host.isAncestorOf(tab.run_tab.log_text))

    def test_Run_Column_Shares_5_5_2(self):
        """Plots, estimator and log fill column 3 in a 5 : 5 : 2 split by
        default; the log keeps its original 80 px minimum (2026-10-24)."""
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        tab.resize(1500, 1100)
        tab.show()
        QApplication.processEvents()
        vs = tab.run_tab._vsplit
        sizes = vs.sizes()
        for got, want in zip(sizes, (5, 5, 2)):
            self.assertAlmostEqual(12 * got / sum(sizes), want, delta=0.15)
        self.assertEqual(tab.run_tab._log_host.minimumHeight(), 80)
        tab.close()

    def test_handles_still_forward(self):
        """main_window contracts: run_tab / plots_tab / run_requested."""
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        self.assertIs(tab.run_tab, tab.params_tab.run_tab)
        self.assertIs(tab.plots_tab, tab.params_tab.plots_panel)
        self.assertTrue(hasattr(tab, "run_requested"))
        for name in ("append_log", "get_options", "log_text", "run_btn"):
            self.assertTrue(hasattr(tab.run_tab, name))
        for name in ("set_plot_data", "set_peak_data", "display_results",
                     "refresh_current", "expand_btn"):
            self.assertTrue(hasattr(tab.plots_tab, name))

    def test_expand_toggle_hides_everything_but_plots(self):
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        tab.show()
        QApplication.processEvents()
        tab.plots_tab.expand_btn.setChecked(True)
        QApplication.processEvents()
        self.assertFalse(tab.params_tab.global_params.isVisible())
        self.assertFalse(tab.params_tab.isotope_list.isVisible())
        self.assertFalse(tab.run_tab._top_host.isVisible())
        self.assertFalse(tab.run_tab._log_host.isVisible())
        self.assertFalse(tab.Voltage_Estimator.isVisible())
        self.assertFalse(tab.params_tab.Optimiser_Tabs.isVisible())
        self.assertTrue(tab.plots_tab.isVisible())
        tab.plots_tab.expand_btn.setChecked(False)
        QApplication.processEvents()
        self.assertTrue(tab.params_tab.global_params.isVisible())
        self.assertTrue(tab.run_tab._log_host.isVisible())
        # Only the current optimiser page comes back (2026-10-25).
        self.assertTrue(tab.Voltage_Estimator.isVisible())
        self.assertFalse(tab.params_tab.Saturation_Estimator.isVisible())
        tab.hide()


class RunLoadRowTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_run_and_load_regular_buttons_in_options_row(self):
        """Run Estimation and Load Estimation are regular-sized buttons
        sharing one row INSIDE the Run Options group (the oversized
        full-width pair below the group dominated the column)."""
        from PySide6.QtWidgets import QGroupBox
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        rt = tab.run_tab
        parent = rt.run_btn.parentWidget()
        self.assertIs(parent, rt.load_btn.parentWidget())
        self.assertIsInstance(parent, QGroupBox)
        self.assertEqual(parent.title(), "Run Options")
        # Regular size: no oversized minimum height, no enlarged font.
        self.assertLess(rt.run_btn.minimumHeight(), 40)
        self.assertFalse(rt.run_btn.font().bold())
        tab.show()
        QApplication.processEvents()
        self.assertLessEqual(
            abs(rt.run_btn.geometry().y() - rt.load_btn.geometry().y()), 4)
        tab.hide()

    def test_Run_Save_Load_Buttons_Equal_And_Flush(self):
        """F2: Run / Save / Load share the row at equal widths and span
        the group's contents; Save waits for a run (2026-10-24)."""
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        rt = tab.run_tab
        tab.resize(1500, 900)
        tab.show()
        QApplication.processEvents()
        btns = (rt.run_btn, rt.Save_Btn, rt.load_btn)
        widths = [b.width() for b in btns]       # leftover px go somewhere
        self.assertLessEqual(max(widths) - min(widths), 1)
        group = rt.run_btn.parentWidget()
        inner = group.contentsRect().marginsRemoved(
            group.layout().contentsMargins())
        self.assertLessEqual(abs(rt.run_btn.x() - inner.left()), 1)
        self.assertLessEqual(abs(rt.load_btn.geometry().right()
                                 - inner.right()), 1)
        self.assertFalse(rt.Save_Btn.isEnabled())
        tab.close()


class PdfFallbackTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_pdf_fallback_renders_onto_canvas(self):
        """A loaded run without estimate_results.npz renders the saved
        PDF pages onto the matplotlib canvas (standard toolbar
        pan/zoom/save keep working), not a text hint."""
        import os
        import tempfile
        from matplotlib.figure import Figure
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        pt = tab.plots_tab
        with tempfile.TemporaryDirectory() as d:
            fig = Figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            ax.plot([0, 1], [0, 1])
            fig.savefig(os.path.join(d, "cfg_hfs_spectra_60Co.pdf"))
            pt.display_results(d, "cfg", "_60Co", "default")
            self.assertEqual(pt._stack.currentIndex(), 0)
            self.assertGreaterEqual(len(pt.figure.axes), 1)
            self.assertTrue(pt.figure.axes[0].get_images())
            # No handle may outlive the render: the temp dir removal
            # below fails on Windows if the PDF were still open.

    def test_missing_pdf_still_shows_hint(self):
        """No arrays AND no PDF on disk: the canvas hint remains."""
        import tempfile
        from gui.estimate_tab import EstimateTab
        tab = EstimateTab()
        pt = tab.plots_tab
        with tempfile.TemporaryDirectory() as d:
            pt.display_results(d, "cfg", "_60Co", "default")
            self.assertEqual(pt._stack.currentIndex(), 0)
            self.assertFalse(pt.figure.axes[0].get_images())


class PlotCanvasTests(unittest.TestCase):
    """The plots canvas in its scroll area (2026-10-22)."""

    def setUp(self):
        _ensure_app()
        import numpy as np
        from gui.estimate_tab import EstimateTab
        self.tab = EstimateTab()
        self.tab.resize(1500, 950)
        self.tab.show()
        QApplication.processEvents()
        self.pt = self.tab.plots_tab
        dv = np.linspace(-500, 500, 200)
        self.spectra = [{"label": lbl, "dV_array": dv,
                         "intensity_array": np.exp(-(dv - c) ** 2 / 800)}
                        for lbl, c in (("59Co", -100), ("60Co", 120))]

    def tearDown(self):
        self.tab.close()

    def test_Plots_Fit_Viewport_At_Pixel_Ratio_2(self):
        """On a Retina screen matplotlib doubles figure.dpi; the canvas
        came out 1000x1200 in a ~780-px column, legend and axis cut off.
        The private setter is the only way to get ratio 2 offscreen."""
        self.pt.canvas._set_device_pixel_ratio(2.0)
        self.pt.set_plot_data(self.spectra)
        self.pt.plot_selector.setCurrentIndex(1)            # Overview
        self.assertLessEqual(self.pt.canvas.width(),
                             self.pt.scroll.viewport().width())
        self.assertEqual(self.pt.canvas.height(), 600)

    def test_Wheel_Over_Plot_Scrolls(self):
        """matplotlib swallowed the wheel as a scroll_event, so neither
        a mouse wheel nor a trackpad moved a plot taller than its
        column."""
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        self.pt.set_plot_data(self.spectra * 3)             # 6 subplots
        bar = self.pt.scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(0)
        pos = QPointF(50, 50)
        ev = QWheelEvent(pos, self.pt.canvas.mapToGlobal(pos), QPoint(),
                         QPoint(0, -120), Qt.MouseButton.NoButton,
                         Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(self.pt.canvas, ev)
        self.assertGreater(bar.value(), 0)


class VoltageEstimatorTests(unittest.TestCase):
    """F1, the Voltage Range Estimator (2026-10-22)."""

    def setUp(self):
        _ensure_app()
        import os
        import yaml
        from gui.estimate_tab import EstimateTab
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "configs", "example_cobalt.yaml")) as fh:
            self.cfg = yaml.safe_load(fh)
        self.tab = EstimateTab()
        self.est = self.tab.Voltage_Estimator

    def tearDown(self):
        self.tab.close()

    def _Build(self, cfg):
        """Feed ``cfg`` as the config source and wait for the build (the
        first one is deferred a tick while satlas2 imports)."""
        import copy
        self.est.Config_Source = lambda: copy.deepcopy(cfg)
        self.est.Live = True                # as after the first Run
        self.tab.show()
        self.est._Refresh()
        for _ in range(100):
            QApplication.processEvents()
            if self.est._Cfg is not None:
                break
        self.assertIsNotNone(self.est._Cfg)

    def test_Estimator_Matches_Pipeline_Peak_Voltages(self):
        """Physics guard: every component's dV at the setpoint equals the
        pipeline's, with a setpoint and with an anchored laser."""
        import copy
        import logging
        import os
        import shutil
        import tempfile
        import yaml
        import cls_estimate
        anchored = {"harmonic": 4, "anchor": {"isotope": "59Co", "dV": 50}}
        for name, laser in (("setpoint", None), ("anchor", anchored)):
            with self.subTest(laser=name):
                cfg = copy.deepcopy(self.cfg)
                if laser:
                    cfg["laser"] = laser
                self._Build(cfg)
                est = self.est
                mine = [d for st in est._States
                        for d in est._Peak_Dvs(st, est._S)]
                tmp = tempfile.mkdtemp()
                try:
                    path = os.path.join(tmp, "cfg.yaml")
                    with open(path, "w") as fh:
                        yaml.safe_dump(cfg, fh)
                    _plots, peaks = cls_estimate.run(path, tmp, no_plot=True)
                finally:
                    # Windows holds the log file open until the handlers go.
                    logger = logging.getLogger("cls_estimate")
                    for h in list(logger.handlers):
                        h.close()
                        logger.removeHandler(h)
                    shutil.rmtree(tmp, ignore_errors=True)
                self.assertEqual(len(mine), len(peaks))
                for d, pk in zip(mine, peaks):
                    self.assertAlmostEqual(d, pk["dV"], delta=1e-9)

    def test_Estimator_Zoom_Survives_Commit_And_Rings_Mark_The_Band(self):
        """A wheel zoom on the map survives a commit's full redraw, and
        the red rings sit only on selected crossings inside +/-dV (2026-10-24)."""
        from types import SimpleNamespace
        self._Build(self.cfg)
        est = self.est
        x0, x1 = est._Map.get_xlim()
        est._On_Scroll(SimpleNamespace(inaxes=est._Map, xdata=est._S,
                                       ydata=0.0, step=1, button="up",
                                       guiEvent=None))     # one notch in
        view = (est._Map.get_xlim(), est._Map.get_ylim())
        self.assertLess(view[0][1] - view[0][0], x1 - x0)
        est.Dv_Box.setValue(300)
        est.Dv_Box.editingFinished.emit()                  # full redraw
        self.assertEqual((est._Map.get_xlim(), est._Map.get_ylim()), view)
        # Rings mark only the selected peaks (2026-10-24).
        dvs = [d for st in est._States
               for k, d in enumerate(est._Peak_Dvs(st, est._S))
               if k in est._Selected(st)]
        inside = sorted(d for d in dvs if d is not None and abs(d) <= 300)
        self.assertTrue(inside)
        self.assertEqual(sorted(est._Rings.get_offsets()[:, 1]), inside)

    def test_Estimator_Rings_Fade_Across_The_Margin(self):
        """Rings are full strength in the sweep's core and fade towards its
        edges across the margin; the readout calls those crossings "edge"
        and counts them by weight; the map shows no text until a ring is
        clicked (2026-10-24)."""
        self._Build(self.cfg)
        est = self.est
        self.assertFalse([t for t in est._Map.texts if t.get_visible()])
        est.Dv_Box.setValue(400)
        est.Dv_Box.editingFinished.emit()
        est.Margin_Box.setValue(150)
        ys = est._Rings.get_offsets()[:, 1]
        alphas = est._Rings.get_edgecolors()[:, 3]
        self.assertGreater(len(ys), 0)
        for y, a in zip(ys, alphas):
            want = min(1.0, (400 - abs(y)) / 150)
            self.assertAlmostEqual(a, max(0.1, want), places=6)
        self.assertEqual(est._Weight(0.0), 1.0)
        self.assertAlmostEqual(est._Weight(325.0), 0.5)
        self.assertEqual(est._Weight(401.0), 0.0)
        text = est.Readout.toPlainText()
        self.assertIn("selected coverage", text)
        if any(250 < abs(y) <= 400 for y in ys):
            self.assertIn(" edge", text)

    def test_Estimator_Ring_Click_Labels_Its_Dv(self):
        """A click on a red ring labels its dV beside the bar without moving
        the setpoint; the label follows the ring, and a second click on it
        or a click elsewhere on the map clears it (2026-10-25)."""
        from types import SimpleNamespace
        self._Build(self.cfg)
        est = self.est
        est.Dv_Box.setValue(300)
        est.Dv_Box.editingFinished.emit()
        est.Window_Box.setValue(400)        # rings tens of pixels apart
        self.tab.resize(1600, 1000)
        QApplication.processEvents()
        est.canvas.draw()                   # lays the axes out
        s0, label, dvs = est._S, est._Label, est._Ring_Dvs
        key = max(dvs, key=lambda k: min(
            (abs(dvs[k][0] - dvs[j][0]) for j in dvs if j != k),
            default=float("inf")))          # the ring furthest from the rest
        x, y = est._Map.transData.transform((s0, est._Ring_Dvs[key][0]))

        def click(x, y):
            ev = SimpleNamespace(inaxes=est._Map, button=1, dblclick=False,
                                 x=x, y=y)
            est._On_Press(ev)
            est._On_Release(ev)
        click(x, y + 2)
        self.assertEqual(est._S, s0)                    # no commit
        self.assertTrue(label.get_visible())
        d = est._Ring_Dvs[key][0]
        self.assertEqual(label.get_text(), f"{d:+.2f} V")
        self.assertEqual(label.xy, (s0, d))
        est._Draw_At(s0 + 1e-4)                         # follows the ring
        self.assertEqual(label.xy, (s0 + 1e-4, est._Ring_Dvs[key][0]))
        est._Draw_At(s0)
        click(x, y)
        self.assertFalse(label.get_visible())
        click(x, y)
        # Clear of the bar and the centre markers: a plain click there pans.
        bb, tol = est._Map.bbox, 12 * est.canvas.device_pixel_ratio
        far = next(px for px in range(int(bb.x0) + 5, int(bb.x1) - 5, 5)
                   if all(abs(est._Map.transData.transform((c, 0))[0] - px)
                          > tol for c in [s0] + est._Centres))
        click(far, y)
        self.assertFalse(label.get_visible())
        self.assertEqual(est._S, s0)

    def test_Estimator_Waits_For_Run(self):
        """F2: before the first Run a refresh reports the change but shows
        the placeholder without building; once live, it builds
        (2026-10-24)."""
        import copy
        from unittest import mock
        est = self.est
        est.Config_Source = lambda: copy.deepcopy(self.cfg)
        self.tab.show()
        seen = []
        est.Config_Changed.connect(lambda: seen.append(1))
        with mock.patch.object(est, "_Rebuild") as build:
            est._Refresh()
            QApplication.processEvents()
            build.assert_not_called()
            self.assertEqual(seen, [1])
            self.assertIsNone(est._Map)
            est.Live = True
            est._Refresh()
            QApplication.processEvents()
            build.assert_called_once()

    def test_Estimator_Peak_Heights_Stable(self):
        """Each in-range peak of the spectrum keeps its height as the
        setpoint moves, and the strongest reads 1; a 600-step grid let
        the heights jump (2026-10-24). Overlapping tails make a weak
        peak read above its bare intensity, so heights are compared
        between setpoints rather than with ``intens``."""
        import numpy as np
        self._Build(self.cfg)
        est = self.est
        lo, hi = est._Map.get_ylim()
        heights = []
        for s in (est._S, est._S + 4e-4):
            est._Draw_At(s)
            h = {}
            for st, line in zip(est._Shown(), est._Spectra):
                y, v = (np.asarray(a) for a in line.get_data())
                self.assertAlmostEqual(y.max(), 1.0, delta=0.02)
                for i, d in enumerate(est._Peak_Dvs(st, s)):
                    if d is not None and lo <= d <= hi:
                        h[st["label"], i] = np.interp(d, v, y)
            heights.append(h)
        common = heights[0].keys() & heights[1].keys()
        self.assertGreater(len(common), 3)
        for k in common:
            self.assertAlmostEqual(heights[0][k], heights[1][k], delta=0.02)

    def test_Estimator_Selects_Peaks_To_Measure(self):
        """By default the selection is each state's strongest "Peaks to
        measure", as the pipeline times them; ticking a component makes
        an explicit selection from there (2026-10-24)."""
        self._Build(self.cfg)
        est = self.est
        st = est._States[0]                     # 59Co, peaks_to_measure 3
        top3 = sorted(range(len(st["intens"])),
                      key=lambda k: -st["intens"][k])[:3]
        self.assertEqual(est._Selected(st), set(top3))
        self.assertTrue(est.Peaks_Btn.text().startswith("To measure"))
        weak = min(range(len(st["intens"])), key=lambda k: st["intens"][k])
        est._Toggle_Peak(st, weak, True)
        self.assertEqual(est._Selected(st), set(top3) | {weak})
        self.assertTrue(est.Peaks_Btn.text().startswith("Selected"))
        est._Pick(None)
        self.assertEqual(est._Selected(st), set(top3))

    def test_Estimator_Peaks_Named_By_Dv_And_Transition(self):
        """The Peaks menu names each component by its F transition, its dV
        at the setpoint right-aligned, no longer its MHz offset, and the
        spectrum is in its state's colour, full over the selected peaks and
        faded over the rest (2026-10-25)."""
        import re
        from fractions import Fraction
        import numpy as np
        from PySide6.QtWidgets import QMenu
        self._Build(self.cfg)
        est = self.est
        est.Show_Combo.setCurrentIndex(1)       # 59Co only
        st = est._Shown()[0]
        dvs, sel = est._Peak_Dvs(st, est._S), est._Selected(st)
        est._Fill_Peaks_Menu()
        [sub] = est.Peaks_Btn.menu().findChildren(QMenu)
        order = sorted(range(len(st["intens"])), key=lambda k: -st["intens"][k])
        self.assertEqual(
            [a.text() for a in sub.actions()],
            [f"{st['transitions'][k]}\t"
             + ("N/A" if dvs[k] is None else f"{dvs[k]:+.1f} V")
             for k in order])
        for label in st["transitions"]:         # the selection rule holds
            m = re.fullmatch(r"F = (\S+) → F′ = (\S+)", label)
            lower, upper = (Fraction(f) for f in m.groups())
            self.assertLessEqual(abs(upper - lower), 1)
        from matplotlib.colors import to_rgb
        from gui.estimate_tab import FADED_PEAK
        spectrum = est._Spectra[0]
        v = np.array([seg[1, 1] for seg in spectrum.get_segments()])
        rgb = spectrum.get_edgecolor()[:, :3]
        self.assertTrue(np.all(spectrum.get_edgecolor()[:, 3] == 1))  # opaque
        full = np.array(to_rgb(st["color"]))
        faded = 1 + FADED_PEAK * (full - 1)             # on the white axes
        for k, d in enumerate(dvs):
            if d is None or not v.min() <= d <= v.max():
                continue
            at = rgb[np.argmin(np.abs(v - d))]
            if k in sel:
                np.testing.assert_allclose(at, full, atol=1e-3)
            elif min(abs(d - dvs[j]) for j in sel) > 3 * est._Fwhm_V:
                np.testing.assert_allclose(at, faded, atol=1e-3)
        # No jumps on the drawn peaks: the colour changes smoothly point to
        # point (between far-apart peaks it may switch on the baseline).
        up = spectrum.get_data()[0] > 1e-3
        self.assertLess(np.abs(np.diff(rgb, axis=0))[up[1:] & up[:-1]].max(),
                        0.2)

    def test_Estimator_Click_And_Hover_Lines_And_Peaks(self):
        """Hovering a map line shows a pointing hand, with no outline; a
        click on the line or its spectrum peak toggles the component, as its
        Peaks tick does, but never on the bar, whose rings keep their
        labels; the reset button sets Window to auto and refits a lost view
        (2026-10-25)."""
        from matplotlib.backend_bases import MouseEvent
        from PySide6.QtCore import Qt
        self.tab.resize(1500, 950)              # a canvas wide enough to aim
        self._Build(self.cfg)
        est = self.est
        est.Show_Combo.setCurrentIndex(1)       # 59Co only
        QApplication.processEvents()
        est.canvas.draw()
        st = est._Shown()[0]
        dvs = est._Peak_Dvs(st, est._S)
        lo, hi = est._Map.get_ylim()
        to_px = est._Map.transData.transform
        # A whole pixel off the bar and the centre markers, as a mouse gives.
        bar = to_px((est._S, 0))[0]
        px = next(round(bar + o) for o in (15, 30, -15, -30, 45) if all(
            abs(bar + o - to_px((c, 0))[0]) > 8 for c in est._Centres))
        x = est._Map.transData.inverted().transform((px, 0))[0]
        ys = {j: to_px((x, e))[1] for j, e in enumerate(est._Peak_Dvs(st, x))
              if e is not None and lo < e < hi}
        free = [j for j in ys if j not in est._Selected(st)
                and dvs[j] is not None and lo < dvs[j] < hi]
        # The unselected line farthest from its neighbours.
        k = max(free, key=lambda j: min(abs(ys[j] - ys[i])
                                        for i in ys if i != j))
        py = ys[k]

        def at(name, x, y):
            return MouseEvent(name, est.canvas, x, y, button=1)
        est._On_Motion(at("motion_notify_event", px, py))
        self.assertEqual(est._Hover, (st["label"], k))
        self.assertEqual(est.canvas.cursor().shape(),
                         Qt.CursorShape.PointingHandCursor)
        lines = len(est._Map.lines)
        # On the bar a line is no target: the rings and the drag keep it.
        self.assertIsNone(est._Hit(at("motion_notify_event", round(bar), py)))
        est._On_Press(at("button_press_event", px, py))   # a map click
        est._On_Release(at("button_release_event", px, py))
        self.assertIn(k, est._Selected(st))
        # The same peak on the spectrum, at the bar: off again.
        sx, sy = est._Spec.transData.transform((0.5, dvs[k]))
        est._On_Press(at("button_press_event", sx, sy))
        self.assertNotIn(k, est._Selected(st))
        est._On_Motion(at("motion_notify_event", 1, 1))    # off the axes
        self.assertIsNone(est._Hover)
        self.assertEqual(len(est._Map.lines), lines)       # nothing drawn
        # A lost view and a set Window: the button restores both.
        est.Window_Box.setValue(500)
        est._Map.set_ylim(1e5, 2e5)
        est._Keep_View()
        est.Reset_Btn.click()
        self.assertEqual(est.Window_Box.value(), 0)
        self.assertIsNone(est._Zoom)
        self.assertLess(est._Map.get_ylim()[1], 1e5)

    def test_Estimator_Optimise_Covers_Selected_Peaks(self):
        """Optimise moves the setpoint so the selected peaks are centred on
        the beam voltage, the sweep's centre, and sets Scan ▸ Voltage range
        to the least symmetric range covering them with the margin (100 V
        by default, set with the gear), start and stop outward to whole
        steps; the readout lists them. Before 2026-10-25 the setpoint
        stayed put and the range took the width of an off-centre window,
        so with all states shown the 59Co peaks fell outside the sweep."""
        self._Build(self.cfg)
        est, gp = self.est, self.tab.params_tab.global_params
        est.Show_Combo.setCurrentIndex(1)       # 59Co only
        st = est._Shown()[0]
        step = gp.scan_step.value()
        self.assertEqual(est.Margin_Box.value(), 100.0)
        est._Optimise()
        s = est._S
        self.assertEqual(gp.setpoint_cm.value(), s)
        d = [est._Peak_Dvs(st, s)[k] for k in est._Selected(st)]
        self.assertAlmostEqual((min(d) + max(d)) / 2, 0.0, delta=0.01)
        start, stop = est._Peak_Range(s)
        for edge in (start, stop):
            self.assertAlmostEqual(edge / step, round(edge / step), places=6)
        self.assertLessEqual(start, min(d) - 100.0)
        self.assertGreater(start, min(d) - 100.0 - step)
        self.assertGreaterEqual(stop, max(d) + 100.0)
        self.assertLess(stop, max(d) + 100.0 + step)
        self.assertIn(f"[{start:.1f}, {stop:.1f}] V to scan",
                      est.Readout.toPlainText())
        half = gp.scan_range.value() / 2
        self.assertAlmostEqual(half, max(-start, stop), places=6)
        self.assertAlmostEqual(est.Dv_Box.value(), half, places=2)
        for v in d:                             # all in the sweep's core
            self.assertEqual(est._Weight(v), 1.0)
        est.Margin_Box.setValue(40.0)           # a tighter margin
        est._Optimise()
        self.assertAlmostEqual(gp.scan_range.value(), 2 * half - 120.0,
                               delta=2 * step)
        est.Show_Combo.setCurrentIndex(0)       # all states, as reported
        est._Optimise()
        dv = est.Dv_Box.value()
        for state in est._Shown():
            for k in est._Selected(state):
                self.assertLessEqual(abs(est._Peak_Dvs(state, est._S)[k]), dv)

    def test_Estimator_Follows_Stepped_Parameters(self):
        """A step (arrows, wheel, code) in a parameter box refreshes the
        estimator, not just Enter or focus-out; a half-typed number does
        not (2026-10-24)."""
        from unittest import mock
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        sp = self.tab.params_tab.global_params.setpoint_cm
        self.tab.show()                 # showing schedules its own refresh
        QApplication.processEvents()
        with mock.patch.object(self.est, "Schedule_Refresh") as refresh:
            sp.stepUp()
            self.assertTrue(refresh.called)
            refresh.reset_mock()
            sp.setValue(10920.5)
            self.assertTrue(refresh.called)
            refresh.reset_mock()
            sp.setFocus()
            sp.lineEdit().selectAll()
            QTest.keyClicks(sp.lineEdit(), "109")
            refresh.assert_not_called()
            QTest.keyClick(sp, Qt.Key.Key_Up)
            self.assertTrue(refresh.called)

    def test_Estimator_Readout_Offset_Is_Display_Only(self):
        """The readout offset moves what is drawn (lines, rings, spectrum,
        the sweep band) and the readout's dV and scan range by its signed
        amount; the axes, their limits and labels, the parsed config and
        the hyperfine model stay as they are (2026-10-24)."""
        import copy
        import numpy as np
        self._Build(self.cfg)
        est = self.est
        st, s = est._States[0], est._S
        before = est._Peak_Dvs(st, s)
        rng0 = est._Peak_Range(s)
        axes0 = (est._Map.get_xlim(), est._Map.get_ylim(),
                 est._Map.get_ylabel())
        band0 = est._Map.patches[0].get_extents().y0
        model = copy.deepcopy((est._Cfg, [dict(x) for x in est._States]))
        y0, v0 = (np.asarray(a) for a in est._Spectra[0].get_data())
        est.Offset_Box.setValue(-25.0)          # the DAQ reads 25 V low
        self.assertEqual((est._Map.get_xlim(), est._Map.get_ylim(),
                          est._Map.get_ylabel()), axes0)
        after = est._Peak_Dvs(st, s)
        for b, a in zip(before, after):
            if b is not None:
                self.assertAlmostEqual(a, b - 25.0, places=9)
        self.assertLess(est._Map.patches[0].get_extents().y0, band0)
        y1, v1 = (np.asarray(a) for a in est._Spectra[0].get_data())
        self.assertAlmostEqual(v1[np.argmax(y1)], v0[np.argmax(y0)] - 25.0,
                               delta=est._Fwhm_V / 5)
        rng1 = est._Peak_Range(s)
        self.assertAlmostEqual(rng1[0], rng0[0] - 25.0, places=6)
        self.assertAlmostEqual(rng1[1], rng0[1] - 25.0, places=6)
        self.assertIn("Offset   -25.00 V", est.Readout.toPlainText())
        self.assertEqual(repr(est._Cfg), repr(model[0]))
        for x, y in zip(est._States, model[1]):
            self.assertEqual(x["hfs"], y["hfs"])
            np.testing.assert_array_equal(x["offsets"], y["offsets"])

    def test_Estimate_States_Become_PreAnalysis_Models(self):
        """Point 8: Add Model > From Estimate lays each state of the last
        run over Pre-Analysis data: spin, J, A/B as the pipeline scaled
        them, the isotope shift as centroid, and (on request) the
        Estimate transition. Nothing to offer before a Run (2026-10-24)."""
        from unittest import mock
        from PySide6.QtWidgets import QMessageBox
        from gui.preanalysis_tab import PreAnalysisTab
        self.assertIsNone(self.est.Model_Source())     # not run yet
        self._Build(self.cfg)
        cfg, states = self.est.Model_Source()
        pa = PreAnalysisTab()
        with mock.patch.object(QMessageBox, "question",
                               return_value=QMessageBox.StandardButton.Yes):
            pa._Add_Estimate_Models(cfg, states)
        self.assertEqual((pa._e_lower.value(), pa._e_upper.value(),
                          pa._harmonic.value()), (3350.494, 47078.494, 4))
        self.assertEqual(len(pa._model_panels), len(states))
        for st, panel in zip(states, pa._model_panels):
            p = panel.get_model_params()
            self.assertEqual(p["I"], st["I"])
            self.assertEqual((p["Jl"], p["Ju"]), (5.0, 6.0))
            for key, value in zip(("Al", "Au", "Bl", "Bu"), st["hfs"]):
                self.assertAlmostEqual(p[key], value, places=2)
            self.assertAlmostEqual(p["centroid"], st["shift"], places=2)
        self.assertAlmostEqual(
            pa._model_panels[0].get_model_params()["Al"], 1056.8, places=2)
        # The menu route: "Blank model", then one entry per state.
        pa._Fill_Model_Menu()            # no window: no Estimate to offer
        acts = pa._Add_Model_Btn.menu().actions()
        self.assertEqual(acts[0].text(), "Blank model")
        self.assertFalse(acts[-1].isEnabled())
        n = len(pa._model_panels)
        acts[0].trigger()
        self.assertEqual(len(pa._model_panels), n + 1)

    def test_Estimator_Commit_Writes_Laser_And_Scan(self):
        """A setpoint commit in anchor mode switches to Setpoint mode; a
        dV of 150 writes a 300 V range; the plots are left alone."""
        from unittest import mock
        gp = self.tab.params_tab.global_params
        gp.laser_anchor_radio.setChecked(True)
        pt = self.tab.plots_tab
        with mock.patch.object(pt, "set_plot_data") as spy:
            self.est.Setpoint_Box.setValue(10920.5)
            self.est.Setpoint_Box.editingFinished.emit()
            self.est.Dv_Box.setValue(150)
            self.est.Dv_Box.editingFinished.emit()
        self.assertTrue(gp.laser_setpoint_radio.isChecked())
        self.assertEqual(gp.setpoint_cm.value(), 10920.5)
        self.assertEqual(gp.scan_range.value(), 300)
        spy.assert_not_called()
        self.assertIsNone(pt._plot_results)
        self.assertFalse(pt.edit_plot_btn.isEnabled())   # still placeholder


class IsotopeColorCodingTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_panels_get_position_colors(self):
        from gui.estimate_tab import EstimateTab, ISOTOPE_COLORS
        tab = EstimateTab()
        lst = tab.params_tab.isotope_list
        panels = [lst.add_isotope() for _ in range(3)]
        for i, p in enumerate(panels):
            self.assertEqual(p._color_index, i)
            self.assertIn(ISOTOPE_COLORS[i], p.styleSheet())

    def test_reference_border_composes_with_color(self):
        from gui.estimate_tab import EstimateTab, ISOTOPE_COLORS
        tab = EstimateTab()
        lst = tab.params_tab.isotope_list
        p0 = lst.add_isotope()
        p1 = lst.add_isotope()
        p1.ref_check.setChecked(True)
        self.assertIn("#42a5f5", p1.styleSheet())      # reference border
        self.assertIn(ISOTOPE_COLORS[1], p1.styleSheet())  # color stripe
        p1.ref_check.setChecked(False)
        self.assertNotIn("#42a5f5", p1.styleSheet())
        self.assertIn(ISOTOPE_COLORS[1], p1.styleSheet())
        # Style stays objectName-scoped so nested groups are untouched.
        self.assertIn("QGroupBox#isotopePanel", p0.styleSheet())

    def test_colors_follow_removal_renumbering(self):
        from gui.estimate_tab import EstimateTab, ISOTOPE_COLORS
        tab = EstimateTab()
        lst = tab.params_tab.isotope_list
        panels = [lst.add_isotope() for _ in range(3)]
        lst._on_remove(panels[0])
        self.assertEqual(panels[1]._color_index, 0)
        self.assertIn(ISOTOPE_COLORS[0], panels[1].styleSheet())
        self.assertEqual(panels[2]._color_index, 1)


class Saturation_Estimator_Tests(unittest.TestCase):
    """The saturation estimator beside the voltage range estimator
    (2026-10-25)."""

    P = (0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0)

    def setUp(self):
        _ensure_app()
        from gui.estimate_tab import Saturation_Estimator
        self.sat = Saturation_Estimator()

    def _Fill(self, rows):
        from PySide6.QtWidgets import QTableWidgetItem
        t = self.sat.Table
        for r, row in enumerate(rows):
            t.setRowCount(max(t.rowCount(), r + 1))
            for c, v in enumerate(row):
                t.setItem(r, c, QTableWidgetItem(str(v)))

    def _Homogeneous(self, noise=0.0, a=5000.0, ps=1.6, b0=40.0, k=18.0,
                     scans=(1,) * len(P)):
        import numpy as np
        p = np.array(self.P)
        wiggle = noise * np.sin(7 * p)          # deterministic "noise"
        rows = []                   # (power, background, peak, scans)
        for x, w, n in zip(p, wiggle, scans):
            bkg = (b0 + k * x) * (1 + w)
            rows.append((x, n * bkg,
                         n * (bkg + a * (x / ps) / (1 + x / ps) * (1 - w)), n))
        return rows

    def test_Saturation_Fit_Finds_The_Analytic_Optimum(self):
        """Homogeneous signal over a linear background, each row summed
        over its own number of scans: the fit returns the per-scan inputs,
        S/B peaks at sqrt(Psat B0 / k), and the propagated 1 sigma matches
        the analytic propagation of the joint covariance."""
        import numpy as np
        self._Fill(self._Homogeneous(scans=(1, 3, 2, 10, 5, 1, 4, 7)))
        p, b, s = self.sat._Rows()
        fit = self.sat._Fit(p, b, s)
        np.testing.assert_allclose(fit["theta"], (5000, 1.6, 40, 18),
                                   rtol=1e-6)
        self.assertAlmostEqual(fit["opt"], np.sqrt(1.6 * 40 / 18), places=6)
        self.assertTrue(self.sat.Readout.toPlainText().startswith(
            "Optimum  1.886"))
        self._Fill(self._Homogeneous(noise=0.03))
        fit = self.sat._Fit(*self.sat._Rows())
        a, ps, b0, k = fit["theta"]
        opt = np.sqrt(ps * b0 / k)
        self.assertAlmostEqual(fit["opt"], opt, places=6)
        jac = np.array([0, opt / (2 * ps), opt / (2 * b0), -opt / (2 * k)])
        self.assertAlmostEqual(fit["opt_err"],
                               np.sqrt(jac @ fit["cov"] @ jac), places=4)
        # One curve, the background-free counts, with the optimum drawn as
        # the estimator draws its setpoint: a red bar and ring.
        (ax,) = self.sat.figure.axes
        self.assertEqual(ax.lines[-1].get_xdata()[0], fit["opt"])
        ring = ax.collections[-1].get_offsets()
        self.assertEqual(len(ring), 1)
        self.assertAlmostEqual(ring[0][1], fit["signal"](fit["opt"]))

    def test_Saturation_Refits_Once_A_Row_Is_Complete(self):
        """A row counts once all four cells hold numbers >= 0, its scans a
        whole number >= 1; a bad cell turns red and its row is skipped; one
        empty row stays at the bottom; counts are taken per scan."""
        from PySide6.QtWidgets import QTableWidgetItem
        rows = self._Homogeneous()
        self._Fill(rows[:3])
        t = self.sat.Table
        self.assertEqual(t.rowCount(), 4)
        self.assertEqual(len(self.sat._Rows()[0]), 3)
        self.assertIn("Optimum", self.sat.Readout.toPlainText())
        t.setItem(3, 0, QTableWidgetItem("3"))
        t.setItem(3, 1, QTableWidgetItem("94"))      # signal still missing
        self.assertEqual(len(self.sat._Rows()[0]), 3)
        self.assertEqual(t.rowCount(), 5)
        t.setItem(3, 2, QTableWidgetItem("abc"))
        self.assertEqual(t.item(3, 2).foreground().color().name(), "#e15759")
        self.assertEqual(len(self.sat._Rows()[0]), 3)
        t.item(3, 2).setText("3500")
        self.assertEqual(len(self.sat._Rows()[0]), 3)           # no scans
        for bad in ("0", "1.5"):
            t.setItem(3, 3, QTableWidgetItem(bad))
            self.assertEqual(t.item(3, 3).foreground().color().name(),
                             "#e15759")
            self.assertEqual(len(self.sat._Rows()[0]), 3)
        t.item(3, 3).setText("2")
        _, b, s = self.sat._Rows()
        self.assertEqual(b[-1], 94 / 2)                         # per scan
        self.assertEqual(s[-1], (3500 - 94) / 2)                # peak - bkg
        self.assertEqual(len(self.sat._Rows()[0]), 4)
        self.assertIn("4 points", self.sat.Readout.toPlainText())

    def test_Saturation_Paste_Models_And_Clear(self):
        """Pasted rows (a header line is skipped) fill the table; the free
        exponent recovers an inhomogeneous line's 0.5, whose S/sqrt(B) has
        no maximum; Clear empties the table and the plot."""
        import numpy as np
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        p = np.array(self.P)
        text = "Power\tBackground\tPeak\tScans\n" + "\n".join(
            f"{x}\t{30 + 12 * x}\t"
            f"{30 + 12 * x + 3000 * (x / 2) / np.sqrt(1 + x / 2)}\t1"
            for x in p)
        QApplication.clipboard().setText(text)
        self.sat.Table.setCurrentCell(0, 0)
        QTest.keyClick(self.sat.Table, Qt.Key.Key_V,
                       Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(len(self.sat._Rows()[0]), len(p))
        self.sat.Model_Combo.setCurrentText("Free exponent")
        fit = self.sat._Fit(*self.sat._Rows())
        self.assertAlmostEqual(fit["theta"][2], 0.5, places=4)
        self.sat.Model_Combo.setCurrentText("Inhomogeneous")
        self.sat.Merit_Combo.setCurrentIndex(1)              # S / sqrt(B)
        self.assertIn("still rises", self.sat.Readout.toPlainText())
        self.sat.Clear_Btn.click()
        self.assertEqual(self.sat.Table.rowCount(), 1)
        self.assertFalse(self.sat.figure.axes)
        self.assertTrue(self.sat.Readout.toPlainText().startswith(
            "No complete measurement"))


class Efficiency_Estimator_Tests(unittest.TestCase):
    """The efficiency estimator, third of the optimiser tabs
    (2026-10-25)."""

    def setUp(self):
        _ensure_app()
        from gui.estimate_tab import Efficiency_Estimator
        self.eff = Efficiency_Estimator()

    def test_Efficiency_Is_Net_Counts_Per_Ion_On_Resonance(self):
        """The ions fluorescing over the ions arriving: 10 bunches per
        channel x 20 scans x 10 ms dwell = 2 s on the peak; 1 pA of singly
        charged ions is 1e-12 / e = 6.24e6 ions/s, so 1.25e7 arrive; 2000
        peak - 400 background = 1600 fluorescing, so 1.28e-4 (1 photon per
        7,800 ions, +/- sqrt(2000)/1600 = 2.8 %). The unit list rescales
        the current (pA by default); no current, or a peak not above the
        background, shows a dash."""
        from cls_estimations.constants import E_CHARGE
        e = self.eff
        self.assertEqual(e.Current_Unit.currentText(), "pA")
        self.assertEqual([e.Current_Unit.itemText(i) for i in range(3)],
                         ["fA", "pA", "nA"])
        self.assertEqual(e.Value.text(), "\u2014")
        self.assertIn("beam current", e.Detail.text())
        e.Peak_Box.setValue(2000)
        e.Background_Box.setValue(400)
        e.Bunches_Box.setValue(10)
        e.Scans_Box.setValue(20)
        e.Dwell_Box.setValue(10)
        e.Current_Box.setValue(1)
        self.assertEqual(e.Value.text(), "1.28 \u00d7 10\u207b\u2074")
        self.assertEqual(e.Detail.text(), "1 photon per 7,800 ions"
                         "   \u00b12.8% (counting)")
        text = e.Readout.toPlainText()
        for line in ("200 on the peak", "2 s on the peak",
                     f"{2e-12 / E_CHARGE:,.0f} arriving (6,240,000 /s, 1+)",
                     "1600 \u00b1 45 fluorescing", "800 /s, 8 per bunch",
                     "35.8 \u03c3"):
            self.assertIn(line, text)
        e.Current_Unit.setCurrentText("fA")     # 1000 times fewer ions
        self.assertEqual(e.Value.text(), "1.28 \u00d7 10\u207b\u00b9")
        self.assertTrue(e.Detail.text().startswith("1 photon per 7.8 ions"))
        e.Current_Unit.setCurrentText("nA")
        self.assertEqual(e.Value.text(), "1.28 \u00d7 10\u207b\u2077")
        e.Background_Box.setValue(2000)
        self.assertEqual(e.Value.text(), "\u2014")
        self.assertIn("exceed the background", e.Detail.text())
        # Estimator inputs stay off the global undo stack.
        self.assertTrue(e.Peak_Box._Own_Undo)

if __name__ == "__main__":
    unittest.main()
