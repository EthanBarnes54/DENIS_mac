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
    """F3: each run card carries its own Laser / Cooler ticks (2026-10-24).
    Since 2026-10-26 the rows in Plot Options hold the MAIN values, taken
    from a run or typed: an unticked run uses them, a ticked run its own
    and is drawn relative to them."""

    def setUp(self):
        _ensure_app()
        from gui.preanalysis_tab import PreAnalysisTab, FileEntry
        self.pa = PreAnalysisTab()
        self.Entry = lambda: FileEntry(
            filepath="/data/run_X.asdf", run_number="X", cooler_v=29999.9,
            date="2026-01-01", laser_sp=20000.0, mass_amu=100.0)

    def test_Run_Override_Precedence(self):
        """Unticked, a run uses the main value; ticked, its own box's (its
        file's value until edited), whatever the main one is."""
        pa, fe = self.pa, self.Entry()
        for key, file_val, get in (
                ("laser", 20000.0, pa._laser_for_entry),
                ("cooler", 29999.9, pa._cooler_for_entry)):
            tick, box = fe._Overrides[key][:2]
            self.assertAlmostEqual(box.value(), file_val, places=6)
            getattr(pa, f"_{key}_override").setValue(123.0)     # typed main
            self.assertAlmostEqual(get(fe), 123.0)
            tick.setChecked(True)
            self.assertAlmostEqual(get(fe), file_val, places=6)
            box.setValue(456.0)
            self.assertAlmostEqual(get(fe), 456.0)
            tick.setChecked(False)
            self.assertAlmostEqual(get(fe), 123.0)
            # A run picked for the main value lends its file's, which a file
            # without one cannot: the typed value stays.
            setattr(fe, "laser_sp" if key == "laser" else "cooler_v", 0)
            self.assertAlmostEqual(pa._Own(fe, key), 123.0)
        # A run's own cooler is its readback's mean once loaded, not the
        # header's single reading.
        fe.cooler_v, fe.Cooler_Measured = 29999.9, 29995.0
        self.assertAlmostEqual(pa._Own(fe, "cooler"), 29995.0)

    def test_Main_Values_Come_From_A_Run_And_Frame_The_Voltage_Axes(self):
        """The main rows list the runs, the first chosen by default; a run
        at another setpoint is drawn where the same ion-frame frequency
        resonates at the main one (the pipeline's inverse), a run at the
        main values is left as is, and frequency stays each run's own."""
        import numpy as np
        from cls_estimations.doppler import resonance_voltage
        pa = self.pa
        a, b = self.Entry(), self.Entry()
        b.filepath, b.run_number, b.laser_sp = "/data/run_Y.asdf", "Y", 20000.1
        pa._file_entries += [a, b]
        cooler, laser = pa._default_cooler_laser()
        combo = pa._Main_Rows["laser"][0]
        self.assertEqual([combo.itemText(i) for i in range(combo.count())],
                         ["Custom", "Run X", "Run Y"])
        self.assertEqual((cooler, laser), (29999.9, 20000.0))     # run X
        v = np.array([-50.0, 0.0, 50.0])
        mass, h = 100.0, 2
        main = (cooler, laser)
        self.assertIs(pa._Into_Main_Frame(v, 29999.9, 20000.0, mass, h, main),
                      v)
        moved = pa._Into_Main_Frame(v, 29999.9, 20000.1, mass, h, main)
        f = pa._voltage_to_frequency(29999.9 - v, mass, h, 20000.1)
        nu = 20000.0 * h * 2.99792458e10 / 1e6
        want = [29999.9 - resonance_voltage(x, nu, mass, 1, "anti-collinear")
                for x in f]
        np.testing.assert_allclose(moved, want)
        # Same frequency at the main setpoint, where the run's peak lands.
        np.testing.assert_allclose(
            pa._voltage_to_frequency(29999.9 - moved, mass, h, 20000.0), f,
            rtol=1e-12)
        # Choosing run Y moves the frame; typing a value makes it Custom.
        combo.setCurrentIndex(2)
        self.assertEqual(pa._default_cooler_laser()[1], 20000.1)
        pa._laser_override.setValue(20000.05)
        self.assertEqual(combo.currentText(), "Custom")
        self.assertEqual(pa._default_cooler_laser()[1], 20000.05)
        # A save keeps the run each value comes from.
        combo.setCurrentIndex(2)
        cfg = pa._build_config_dict()["preanalysis"]
        self.assertEqual((cfg["laser_reference"], cfg["cooler_reference"]),
                         ("/data/run_Y.asdf", "/data/run_X.asdf"))
        # ...and a restore picks the same runs once they load.
        pa._file_entries.clear()

        def load(path):
            fe = self.Entry()
            fe.filepath = path
            fe.laser_sp = 20000.1 if path.endswith("Y.asdf") else 20000.0
            pa._file_entries.append(fe)
        with patch("gui.preanalysis_tab.os.path.isfile", return_value=True), \
                patch.object(pa, "_load_file", side_effect=load):
            pa._restore_from_dict(cfg)
        self.assertEqual(pa._default_cooler_laser(), (29999.9, 20000.1))

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
        # Unticked, b takes the main value: by default the first run's own,
        # a's ticked one.
        self.assertAlmostEqual(pa._laser_for_entry(b2), 20000.5)

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
