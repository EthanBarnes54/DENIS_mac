"""Budget sources on the Systematics tab (Analysis Upgrade, card F1; WP-S4).

The Budget table gains source rows (C4: what a row drives, its width,
unit, correlation and kind), whose columns stay hidden until a row is a
source, so the table looks as it did. "Propagate sources" runs the
generalised engine (E1) on the ticked projects and fills the Sources tab
with each source's effect, the total and the statistical error. Typed MHz
rows work as before; an older session loads unchanged. (No "Import from
ORACLE": run files travel between PCs without what sits beside them.)

Run from the project root:
    .venv/bin/python -m pytest tests/test_budget_sources.py -q
"""
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from cls_estimations.systematics import (  # noqa: E402
    ExtraSystematic, Systematic_Source)
from tests.test_source_budget import _Linear_Project, _Target  # noqa: E402
from tests.test_systematics_tab import (  # noqa: E402
    _StubAnalysisTab, _range_tab)

SOURCE = Systematic_Source("Scan divider", 100.0, key="scan_div_ppm",
                           unit="ppm", correlation="campaign")


class Budget_Source_Tests(unittest.TestCase):

    def _Tab(self):
        tab = _range_tab(_StubAnalysisTab())
        self.addCleanup(tab.deleteLater)
        return tab

    def _Hidden(self, tab):
        return [tab._extra_table.isColumnHidden(c) for c in range(6, 11)]

    def test_The_Table_Looks_As_It_Did_Until_A_Row_Is_A_Source(self):
        tab = self._Tab()
        tab._add_extra(ExtraSystematic("Laser", 2.0))
        self.assertEqual(self._Hidden(tab), [True] * 5)
        self.assertEqual([e.name for e in tab.extras()], ["Laser"])
        self.assertEqual(tab.sources(), [])
        tab._Add_Source(SOURCE)
        self.assertEqual(self._Hidden(tab), [False] * 5)
        # A source is propagated, never summed as a typed MHz row.
        self.assertEqual([e.name for e in tab.extras()], ["Laser"])
        self.assertEqual(tab.sources(), [SOURCE])

    def test_Sources_Are_Saved_Beside_The_Typed_Rows(self):
        tab = self._Tab()
        tab._add_extra(ExtraSystematic("Laser", 2.0))
        tab._Add_Source(SOURCE)
        saved = tab.to_dict()
        self.assertEqual([e["name"] for e in saved["extras"]], ["Laser"])
        self.assertEqual(saved["sources"], [SOURCE.to_dict()])
        back = self._Tab()
        back.from_dict(saved)
        self.assertEqual(back.sources(), [SOURCE])
        self.assertEqual([e.name for e in back.extras()], ["Laser"])

    def test_An_Older_Session_Loads_As_It_Did(self):
        tab = self._Tab()
        tab._add_extra(ExtraSystematic("Laser", 2.0))
        saved = tab.to_dict()
        del saved["sources"]
        back = self._Tab()
        back.from_dict(saved)
        self.assertEqual(back.sources(), [])
        self.assertEqual(self._Hidden(back), [True] * 5)

    def test_An_Unknown_Source_Is_Shown_But_Not_Propagated(self):
        tab = self._Tab()
        tab._Add_Source(Systematic_Source("Bench table", float("nan"),
                                          key="scan_axis_gain_ppm",
                                          unit="ppm", kind="unknown"))
        src = tab.sources()[0]
        self.assertEqual(src.kind, "unknown")
        self.assertFalse(src.enabled)

    def test_Propagate_Fills_The_Sources_Tab(self):
        tab = self._Tab()
        tab._use_gp.setChecked(False)
        tab._use_is.setChecked(False)
        p = _Linear_Project("P", {"1": {"scan_div_ppm": 0.02},
                                  "2": {"scan_div_ppm": 0.04}})
        self.addCleanup(p.deleteLater)
        tab._selected_targets = lambda: [_Target(p)]
        tab._Add_Source(SOURCE)
        loop = QEventLoop()
        tab._Budget_Engine.finished.connect(lambda _r: QTimer.singleShot(
            0, loop.quit))
        QTimer.singleShot(20000, loop.quit)
        tab._Run_Budget()
        loop.exec()
        t = tab._Source_Table
        head = [t.horizontalHeaderItem(c).text()
                for c in range(t.columnCount())]
        self.assertEqual(head[:3], ["Project", "Run / isotope", "Parameter"])
        self.assertIn("Scan divider", head)
        row = {head[c]: t.item(0, c).text() for c in range(t.columnCount())}
        self.assertEqual(row["Project"], "P")
        self.assertAlmostEqual(float(row["Scan divider"]), 3.0)
        self.assertAlmostEqual(float(row["statistical"]), 0.5 / math.sqrt(2),
                               places=4)
        self.assertTrue(tab._run_btn.isEnabled())       # not left busy
        # The result is kept with the session.
        back = self._Tab()
        back.from_dict(tab.to_dict())
        self.assertEqual(back._Source_Table.rowCount(), 1)


if __name__ == "__main__":
    unittest.main()
