"""Scroll-zoom, drag-pan and double-click reset on the Analysis and
Results plots (gui.shared_widgets.Enable_Plot_Navigation)."""
import unittest

from tests.test_import_only_checked import _ensure_app


class Plot_Navigation_Tests(unittest.TestCase):

    def setUp(self):
        _ensure_app()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from gui.shared_widgets import Enable_Plot_Navigation
        self.fig = Figure(figsize=(6.4, 4.8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.addCleanup(self.canvas.deleteLater)
        Enable_Plot_Navigation(self.canvas)
        self.ax = self._Draw(0.0, 10.0)

    def _Draw(self, lo, hi):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot([lo, hi], [lo, hi])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        self.canvas.draw()
        return ax

    def _Send(self, name, xy, **kw):
        from matplotlib.backend_bases import MouseEvent
        x, y = self.ax.transData.transform(xy)
        ev = MouseEvent(name, self.canvas, x, y, **kw)
        self.canvas.callbacks.process(name, ev)

    def test_Scroll_Zooms_About_The_Cursor_And_Double_Click_Resets(self):
        self._Send("scroll_event", (5.0, 5.0), button="up", step=1)
        x0, x1 = self.ax.get_xlim()
        self.assertLess(x1 - x0, 10.0)
        self.assertAlmostEqual((x0 + x1) / 2, 5.0, delta=0.05)  # 1 px
        self._Send("button_press_event", (5.0, 5.0), button=1, dblclick=True)
        self.assertEqual(self.ax.get_xlim(), (0.0, 10.0))
        self.assertEqual(self.ax.get_ylim(), (0.0, 10.0))

    def test_Drag_Pans(self):
        self._Send("button_press_event", (5.0, 5.0), button=1)
        self._Send("motion_notify_event", (3.0, 5.0))
        self._Send("button_release_event", (3.0, 5.0), button=1)
        x0, x1 = self.ax.get_xlim()
        self.assertAlmostEqual(x0, 2.0, delta=0.05)  # events are whole pixels
        self.assertAlmostEqual(x1, 12.0, delta=0.05)
        # moving the mouse after the release pans no further
        self._Send("motion_notify_event", (1.0, 5.0))
        self.assertAlmostEqual(self.ax.get_xlim()[0], x0, places=9)

    def test_After_A_Redraw_Reset_Goes_To_The_New_View(self):
        self._Send("scroll_event", (5.0, 5.0), button="up", step=1)
        self.ax.clear()                       # same axes, replotted
        self.ax.plot([20.0, 30.0], [20.0, 30.0])
        self.ax.set_xlim(20.0, 30.0)
        self.ax.set_ylim(20.0, 30.0)
        self._Send("scroll_event", (25.0, 25.0), button="up", step=1)
        self._Send("button_press_event", (25.0, 25.0), button=1,
                   dblclick=True)
        self.assertEqual(self.ax.get_xlim(), (20.0, 30.0))

    def test_A_Toolbar_Tool_In_Use_Is_Left_Alone(self):
        class _Toolbar:
            mode = "pan/zoom"
        self.canvas.toolbar = _Toolbar()
        self._Send("scroll_event", (5.0, 5.0), button="up", step=1)
        self.assertEqual(self.ax.get_xlim(), (0.0, 10.0))

    def test_The_Results_Plot_Has_It(self):
        from gui.results_tab import ResultsTab
        tab = ResultsTab()
        self.addCleanup(tab.deleteLater)
        self.assertTrue(tab._plot_canvas.callbacks.callbacks.get(
            "scroll_event"))


if __name__ == "__main__":
    unittest.main()
