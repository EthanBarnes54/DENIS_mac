"""UI fixes from the 2026-10-08 external review.

1. NIST ASD: start levels and lasers typed over in the tables were
   ignored -- the search read the number the row was created with.
2. NIST ASD: the loaded lines/levels were never tied to a species, so a
   cache miss, a failed fetch or a fetch still running left species A's
   data searched under species B's name.
3. NIST ASD: the ranked results table kept the medium of the search
   after the Air/Vacuum switch, while the details pane changed.
4. Results: switching plots with the plot editor open wrote the new
   plot's style into the old plot's sidecar.
5. Quick Plot: a manually typed table never filled the X/Y selectors,
   and a blank Y appended X alone (unequal arrays, a raised error).

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_review_2026_10_08_ui.py -q
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMessageBox, QTableWidgetItem, QTreeWidgetItem,
)

from tests.test_nist_asd_search import _system, _two_step_cfg  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def _nist_window(spectrum="Xx I"):
    from gui.nist_asd.tab import NistAsdWindow
    w = NistAsdWindow()
    w.spectrum_edit.setText(spectrum)
    lines, levels = _system()
    with mock.patch("gui.nist_asd.data.cache_info", return_value=None):
        w.set_data(lines, levels, spectrum)
    return w


def _type_into(table, row, col, text):
    """What a user edit does: change the item's edit/display text."""
    idx = table.model().index(row, col)
    table.model().setData(idx, text, Qt.ItemDataRole.EditRole)


class NistEditedNumbersTests(unittest.TestCase):

    def test_typed_laser_and_start_level_are_searched(self):
        w = _nist_window()
        w.laser_table.setRowCount(0)
        w._add_laser_row("L1", 400.0, 10.0, "PUMP")
        w.start_table.setRowCount(0)
        w._add_start_row(0.0, 1.0)
        _type_into(w.laser_table, 0, 1, "250")
        _type_into(w.laser_table, 0, 2, "50")
        _type_into(w.start_table, 0, 0, "20000")
        cfg = w.collect_search_config()
        self.assertEqual(cfg.lasers[0]["center_nm"], 250.0)
        self.assertEqual(cfg.lasers[0]["range_nm"], 50.0)
        self.assertEqual(cfg.starting_levels[0]["level"], 20000.0)

    def test_text_that_is_not_a_number_is_put_back(self):
        w = _nist_window()
        w.laser_table.setRowCount(0)
        w._add_laser_row("L1", 400.0, 10.0)
        _type_into(w.laser_table, 0, 1, "abc")
        self.assertEqual(w.laser_table.item(0, 1).text(), "400")
        self.assertEqual(w.collect_search_config().lasers[0]["center_nm"],
                         400.0)

    def test_unedited_rows_keep_full_precision(self):
        w = _nist_window()
        w.start_table.setRowCount(0)
        w._add_start_row(6298.8501234, 1.0)
        self.assertEqual(w.start_table.item(0, 0).text(), "6298.850")
        self.assertEqual(w.collect_search_config().starting_levels[0]
                         ["level"], 6298.8501234)


class NistSpeciesTests(unittest.TestCase):

    def _search_started(self, w):
        with mock.patch("gui.nist_asd.tab._SearchWorker") as worker, \
                mock.patch.object(QMessageBox, "information"):
            w._run_search()
        return worker.called

    def test_cache_miss_clears_the_previous_species(self):
        w = _nist_window("Xx I")
        w.spectrum_edit.setText("Zz I")
        with mock.patch("gui.nist_asd.data.load_cache", return_value=None):
            self.assertFalse(w.load_cached_spectrum())
        self.assertTrue(w._lines.empty)
        self.assertEqual(w.levels_table.rowCount(), 0)
        self.assertIn("Zz I", w.cache_label.text())
        self.assertFalse(self._search_started(w))

    def test_search_refuses_data_of_another_species(self):
        w = _nist_window("Xx I")
        self.assertTrue(self._search_started(w))
        w.spectrum_edit.setText("Zz I")          # typed, never fetched
        self.assertFalse(self._search_started(w))

    def test_failed_fetch_keeps_old_data_but_not_its_name(self):
        w = _nist_window("Xx I")
        w.spectrum_edit.setText("Zz I")
        with mock.patch.object(QMessageBox, "warning"):
            w._on_fetch_error("boom")
        self.assertIn("Xx I", w.cache_label.text())
        self.assertFalse(self._search_started(w))

    def test_no_search_while_a_fetch_is_running(self):
        w = _nist_window("Xx I")
        w._fetch_worker = mock.Mock(isRunning=mock.Mock(return_value=True))
        self.assertFalse(self._search_started(w))

    def test_fetched_data_is_labelled_with_the_fetched_species(self):
        w = _nist_window("Xx I")
        lines, levels = _system()
        w.spectrum_edit.setText("Typed meanwhile")
        with mock.patch("gui.nist_asd.data.cache_info", return_value=None):
            w._on_fetch_done("Zz I", lines, levels)
        self.assertEqual(w._loaded_spectrum, "Zz I")
        self.assertTrue(w.cache_label.text().startswith("Zz I"))


class NistMediumTests(unittest.TestCase):

    def test_results_table_follows_the_medium(self):
        from gui.nist_asd import data as nd
        from gui.nist_asd.search import SchemeSearcher
        w = _nist_window()
        lines, levels = _system()
        ranked = SchemeSearcher(_two_step_cfg(), lines, levels).run()
        w.medium_combo.setCurrentText("Air")
        w.set_results(ranked)
        w.results_table.selectRow(1)
        w.medium_combo.setCurrentText("Vacuum")
        step = ranked[0].scheme.steps[0].transition
        want = f"{nd.wavelength_nm(step, 'vacuum'):.2f}"
        self.assertTrue(w.results_table.item(0, 2).text().startswith(want))
        self.assertEqual(w.results_table.currentRow(), 1)


class ResultsEditorTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _npz(self, name, n):
        p = os.path.join(self.tmp, name)
        np.savez(p, plot_type="tof", centers=np.arange(n, dtype=float),
                 counts=np.linspace(1, 10, n), run_num="1",
                 tof_gate=np.array([2.0, 4.0]))
        return p

    def test_switching_plots_saves_the_edit_to_the_right_plot(self):
        import gui.results_tab as R
        a, b = self._npz("tof_run_2.npz", 20), self._npz("tof_run_7.npz",
                                                         50)
        rt = R.ResultsTab()
        rt.show()
        root = QTreeWidgetItem(["proj"])
        root.setData(0, Qt.ItemDataRole.UserRole, {"project": "P"})
        it_a = QTreeWidgetItem(root, ["A"])
        it_a.setData(0, Qt.ItemDataRole.UserRole,
                     {"type": "editable_plot", "path": a})
        it_b = QTreeWidgetItem(root, ["B"])
        it_b.setData(0, Qt.ItemDataRole.UserRole,
                     {"type": "editable_plot", "path": b})
        rt._updating_checks = True
        rt._tree.addTopLevelItem(root)
        rt._updating_checks = False
        rt._tree.setCurrentItem(it_a)
        rt._edit_plot()
        dlg = rt._edit_dialog
        rt._plot_figure.axes[0].title.set_fontsize(21)    # the edit
        rt._tree.setCurrentItem(it_b)
        self.assertFalse(dlg.isVisible())
        with open(R._style_path(a), encoding="utf-8") as fh:
            style = json.load(fh)["axes"][0]
        self.assertEqual(style["xlim"][1], 19.5)          # A's range
        self.assertEqual(style["title_size"], 21)
        self.assertFalse(os.path.exists(R._style_path(b)))
        self.assertIsNone(dlg._source_path)
        rt.close()


class QuickPlotManualTests(unittest.TestCase):

    def _dialog(self, rows):
        from gui.shared_widgets import QuickPlotDialog
        d = QuickPlotDialog()
        d._input_tabs.setCurrentIndex(1)
        d._table.setRowCount(max(d._table.rowCount(), len(rows)))
        for r, vals in enumerate(rows):
            for c, v in enumerate(vals):
                d._table.setItem(r, c, QTableWidgetItem(v))
        return d

    def _points(self, d):
        ax = d._fig.axes[0]
        return len(ax.get_lines()[0].get_xdata())

    def test_typed_table_plots(self):
        d = self._dialog([("1", "2"), ("2", "4"), ("3", "6")])
        d._do_plot()
        self.assertEqual(d._x_col.count(), d._table.columnCount())
        self.assertEqual(self._points(d), 3)

    def test_blank_y_skips_the_row(self):
        d = self._dialog([("1", "2"), ("2", ""), ("3", "6")])
        d._do_plot()
        self.assertEqual(self._points(d), 2)

    def test_column_choice_survives_replotting(self):
        d = self._dialog([("1", "2"), ("2", "4"), ("3", "6")])
        d._do_plot()
        d._y_col.setCurrentIndex(0)
        chosen = d._y_col.currentText()
        d._do_plot()
        self.assertEqual(d._y_col.currentText(), chosen)

    def test_csv_with_a_blank_y_loads(self):
        from gui.shared_widgets import QuickPlotDialog
        p = os.path.join(tempfile.mkdtemp(), "d.csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("x,y\n1,2\n2,\n3,6\n4,8\n")
        d = QuickPlotDialog()
        d._load_file(p)
        self.assertEqual(self._points(d), 3)


if __name__ == "__main__":
    unittest.main()
