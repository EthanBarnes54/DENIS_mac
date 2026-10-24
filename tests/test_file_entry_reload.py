"""Pre-Analysis FileEntry per-file reload button and its handler.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

A FileEntry exposes a reload button (and reload_requested signal) so
users watching ongoing scans can pick up new events without removing and
re-adding the file. The button is hidden on a MergedFileEntry, which has
no on-disk ASDF to re-read. PreAnalysisTab._reload_file_entry re-reads
the ASDF via _load_cls_data, then refreshes the detail label and the
spectrum plot.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_file_entry_reload -v

Depends on: gui.preanalysis_tab (FileEntry, MergedFileEntry,
PreAnalysisTab); PySide6 for QApplication.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class FileEntryReloadButtonTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_file_entry_has_reload_button_and_signal(self):
        from gui.preanalysis_tab import FileEntry

        fe = FileEntry(
            filepath="/data/run_X.asdf",
            run_number="X", cooler_v=29999.9, date="2026-01-01",
            laser_sp=20000.0, mass_amu=100.0)
        self.assertTrue(hasattr(fe, "reload_btn"))
        self.assertTrue(hasattr(fe, "reload_requested"))
        # The reload button must be visible on a regular file entry.
        self.assertTrue(fe.reload_btn.isVisibleTo(fe))

    def test_merged_file_entry_hides_reload_button(self):
        """MergedFileEntry has no on-disk ASDF backing it; the
        reload button would have no consumer there. Hidden so the
        UI doesn't tease an action that would crash."""
        from gui.preanalysis_tab import MergedFileEntry

        mfe = MergedFileEntry(
            name="m", merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]), merge_domain="voltage",
            source_info=[("a", "/p/a"), ("b", "/p/b")])
        # Inherited from FileEntry but hidden in MergedFileEntry.__init__.
        self.assertTrue(hasattr(mfe, "reload_btn"))
        self.assertFalse(mfe.reload_btn.isVisibleTo(mfe))

    def test_clicking_reload_emits_signal_with_entry(self):
        from gui.preanalysis_tab import FileEntry

        fe = FileEntry(
            filepath="/data/run_X.asdf",
            run_number="X", cooler_v=29999.9, date="2026-01-01",
            laser_sp=20000.0, mass_amu=100.0)
        captured = []
        fe.reload_requested.connect(captured.append)
        fe.reload_btn.click()
        self.assertEqual(captured, [fe])


class PaReloadHandlerTests(unittest.TestCase):
    """PA's ``_reload_file_entry`` re-reads the ASDF via
    ``_load_cls_data`` then refreshes the detail label and the
    spectrum plot. Mocks the heavy clstools work to keep the test
    headless."""

    def setUp(self):
        _ensure_app()

    def test_reload_calls_load_cls_data_and_refresh(self):
        from gui.preanalysis_tab import PreAnalysisTab, FileEntry

        pa = PreAnalysisTab()
        fe = FileEntry(
            filepath="/data/run_X.asdf",
            run_number="X", cooler_v=29999.9, date="2026-01-01",
            laser_sp=20000.0, mass_amu=100.0)
        pa._file_entries.append(fe)
        # Force clstools to "available" for the path we exercise.
        # (_get_clstools() replaced the old _HAS_CLSTOOLS flag when the
        # import went lazy for startup perf; any truthy module works.)
        with patch("gui.preanalysis_tab._get_clstools",
                   return_value=object()), \
                patch.object(pa, "_load_cls_data") as mock_load, \
                patch.object(fe, "update_detail") as mock_update, \
                patch.object(pa, "_schedule_replot") as mock_replot:
            pa._reload_file_entry(fe)
            mock_load.assert_called_once_with(fe)
            mock_update.assert_called_once()
            mock_replot.assert_called_once()


class RunOverrideTests(unittest.TestCase):
    """F3: each run card carries its own Laser / Cooler ticks, which beat
    the global override rows (2026-10-24)."""

    def setUp(self):
        _ensure_app()
        from gui.preanalysis_tab import PreAnalysisTab, FileEntry
        self.pa = PreAnalysisTab()
        self.Entry = lambda: FileEntry(
            filepath="/data/run_X.asdf", run_number="X", cooler_v=29999.9,
            date="2026-01-01", laser_sp=20000.0, mass_amu=100.0)

    def test_Run_Override_Precedence(self):
        pa, fe = self.pa, self.Entry()
        for key, file_val, get in (
                ("laser", 20000.0, pa._laser_for_entry),
                ("cooler", 29999.9, pa._cooler_for_entry)):
            tick, box = fe._Overrides[key][:2]
            g_tick = getattr(pa, f"_{key}_override_enabled")
            g_box = getattr(pa, f"_{key}_override")
            g_box.setValue(123.0)
            box.setValue(456.0)
            for run_on, glob_on, want in ((False, False, file_val),
                                          (False, True, 123.0),
                                          (True, False, 456.0),
                                          (True, True, 456.0)):
                tick.setChecked(run_on)
                g_tick.setChecked(glob_on)
                self.assertAlmostEqual(get(fe), want, places=6,
                                       msg=(key, run_on, glob_on))
            tick.setChecked(False)
            g_tick.setChecked(False)
            setattr(fe, "laser_sp" if key == "laser" else "cooler_v", 0)
            self.assertAlmostEqual(get(fe), 123.0)          # fallback
            tick.setChecked(True)
            self.assertAlmostEqual(get(fe), 456.0)

    def test_Run_Override_Round_Trips(self):
        pa = self.pa
        a, b = self.Entry(), self.Entry()
        b.filepath = "/data/run_Y.asdf"
        a._Overrides["laser"][1].setValue(20000.5)
        a._Overrides["laser"][0].setChecked(True)
        pa._file_entries += [a, b]
        files = pa._build_config_dict()["preanalysis"]["files"]
        self.assertEqual([f["laser_override"] for f in files],
                         [20000.5, None])
        self.assertEqual([f["cooler_override"] for f in files], [None, None])
        pa._file_entries.clear()
        with patch("gui.preanalysis_tab.os.path.isfile", return_value=True), \
                patch.object(pa, "_load_file", side_effect=lambda p: (
                    pa._file_entries.append(self.Entry()))):
            pa._restore_from_dict({"files": files})
        a2, b2 = pa._file_entries[-2:]
        self.assertAlmostEqual(a2.Laser_Override, 20000.5)
        self.assertIsNone(a2.Cooler_Override)
        self.assertIsNone(b2.Laser_Override)
        self.assertAlmostEqual(pa._laser_for_entry(a2), 20000.5)
        self.assertAlmostEqual(pa._laser_for_entry(b2), 20000.0)

    def test_Run_Override_Hidden_On_Merged(self):
        from gui.preanalysis_tab import MergedFileEntry
        mfe = MergedFileEntry(
            name="m", merged_x=np.array([0.0, 1.0]),
            merged_y=np.array([1.0, 2.0]), merge_domain="voltage",
            source_info=[("a", "/p/a"), ("b", "/p/b")])
        self.assertFalse(mfe.Override_Row.isVisibleTo(mfe))
        fe = self.Entry()
        self.assertTrue(fe.Override_Row.isVisibleTo(fe))


if __name__ == "__main__":
    unittest.main()
