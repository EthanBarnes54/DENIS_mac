"""The Source block's controls for the upgrades (Analysis Upgrade, card B2).

* WP-P2: a file row's "Cooler / Laser for this run..." sets that run's own
  override (``run_overrides``), shown on the row's badge.
* D1-D3 under Advanced Data Settings: ``rolling`` in Cooler corr. with its
  points (shown only for it), a Laser combo (header / wavemeter mean), and
  "Use the run's recorded dividers" (enabled when a file records them).
* Off, the Source config and the saved block are what they always were;
  the Preview, Meta and the merge dialog follow the new settings.

Run from the project root:
    .venv/bin/python -m pytest tests/test_source_upgrade_controls.py -q
"""
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from PySide6.QtWidgets import QDialog, QDoubleSpinBox

from tests.test_import_only_checked import _ensure_app
from tests.test_oracle_ingestion import Write_Run

NEW_KEYS = ("cooler_window_points", "laser_source", "use_oracle_dividers",
            "run_overrides")


class Source_Upgrade_Control_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_app()
        cls.tmp = tempfile.mkdtemp(prefix="source_controls_")
        cls.legacy = os.path.join(cls.tmp, "run_0011.asdf")
        cls.oracle = os.path.join(cls.tmp, "run_0012.asdf")
        Write_Run(cls.legacy)
        Write_Run(cls.oracle, {"Dividers": {"Cooler": {"ratio": 10000.5}}})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Block(self, *paths):
        from gui.analysis.blocks import SourceBlock
        sb = SourceBlock()
        self.addCleanup(sb.deleteLater)
        for p in paths:
            sb._add_file_entry(p)
        return sb

    def _Round_Trip(self, sb):
        back = self._Block()
        back.from_dict(sb.to_dict())
        return back

    def test_Off_The_Config_And_The_Save_Are_As_They_Were(self):
        sb = self._Block(self.legacy)
        cfg, saved = sb.get_source_config(), sb.to_dict()
        for key in NEW_KEYS:
            self.assertNotIn(key, cfg)
            self.assertNotIn(key, saved)
        self.assertEqual(cfg["cooler_correction"], "pbp")
        self.assertFalse(sb._Use_Dividers.isEnabled())

    def test_Rolling_Shows_Its_Points_And_Round_Trips(self):
        sb = self._Block()
        self.assertTrue(sb._Cooler_Points.isHidden())
        sb._cooler_corr.setCurrentText("rolling")
        self.assertFalse(sb._Cooler_Points.isHidden())
        sb._Cooler_Points.setValue(250)
        cfg = sb.get_source_config()
        self.assertEqual((cfg["cooler_correction"],
                          cfg["cooler_window_points"]), ("rolling", 250))
        back = self._Round_Trip(sb).get_source_config()
        self.assertEqual(back["cooler_window_points"], 250)
        self.assertEqual(back["cooler_correction"], "rolling")

    def test_The_Laser_Can_Be_The_Wavemeter_Mean(self):
        sb = self._Block()
        sb._Laser_Source.setCurrentIndex(
            sb._Laser_Source.findData("wavemeter"))
        self.assertEqual(sb.get_source_config()["laser_source"], "wavemeter")
        self.assertEqual(self._Round_Trip(sb).get_source_config()
                         ["laser_source"], "wavemeter")

    def test_The_Dividers_Tick_Needs_A_File_That_Records_Them(self):
        sb = self._Block(self.legacy)
        self.assertFalse(sb._Use_Dividers.isEnabled())
        sb._add_file_entry(self.oracle)
        self.assertTrue(sb._Use_Dividers.isEnabled())
        sb._Use_Dividers.setChecked(True)
        self.assertTrue(sb.get_source_config()["use_oracle_dividers"])
        back = self._Round_Trip(sb)
        self.assertTrue(back.get_source_config()["use_oracle_dividers"])
        self.assertTrue(back._Use_Dividers.isEnabled())

    def test_A_Run_s_Own_Cooler_And_Laser_From_Its_Row(self):
        sb = self._Block(self.legacy, self.oracle)
        entry = sb._file_entries[0]

        def Fill(dialog, cooler=29990.0, laser=0.0):
            spins = dialog.findChildren(QDoubleSpinBox)
            spins[0].setValue(cooler)
            spins[1].setValue(laser)
            return QDialog.DialogCode.Accepted

        with mock.patch.object(QDialog, "exec", Fill):
            sb._Edit_Run_Override(entry)
        self.assertEqual(sb.get_source_config()["run_overrides"],
                         {self.legacy: {"cooler_v": 29990.0}})
        self.assertEqual(entry["badge_label"].text(), "[override]")
        self.assertIn("29990", entry["badge_label"].toolTip())
        back = self._Round_Trip(sb)
        self.assertEqual(back.get_source_config()["run_overrides"],
                         {self.legacy: {"cooler_v": 29990.0}})
        self.assertEqual(back._file_entries[0]["badge_label"].text(),
                         "[override]")
        # Zero in both clears it.
        with mock.patch.object(QDialog, "exec",
                               lambda d: Fill(d, cooler=0.0)):
            sb._Edit_Run_Override(entry)
        self.assertNotIn("run_overrides", sb.get_source_config())
        self.assertTrue(entry["badge_label"].isHidden())

    def test_The_Preview_Takes_The_Run_s_Own_Override_And_Rolling(self):
        from gui.analysis import blocks
        seen = {}

        class _Stop(Exception):
            pass

        class _D:
            VCoolDiv, VCoolOffset, Vcool_init, Laser_set = 10000, 0, 3.0, 1.0

            def Compute_Voltages(self, cooler_correction="pbp"):
                seen["v"] = (self.VCoolDiv, self.VCoolOffset,
                             cooler_correction)
                raise _Stop

        sb = self._Block()
        sb._cooler_corr.setCurrentText("rolling")
        sb._Laser_Source.setCurrentIndex(
            sb._Laser_Source.findData("wavemeter"))
        entry = {"path": "run_1.asdf",
                 "run_override": {"cooler_v": 29990.0}}
        fake = types.SimpleNamespace(CLSDataFrame=_D)
        with mock.patch.object(sb, "get_checked_files",
                               return_value=[entry]), \
                mock.patch.object(blocks, "_get_clstools",
                                  return_value=fake), \
                mock.patch("gui.calibration.load_run_calibrated") as load, \
                mock.patch("gui.analysis.pipeline.Rolling_Cooler",
                           return_value={"cooler_v": 29980.0}) as roll, \
                mock.patch("gui.analysis.pipeline.Wavemeter_Laser") as wm, \
                mock.patch.object(blocks, "QMessageBox"):
            sb._show_preview_at_index()
        self.assertEqual(seen["v"], (0, 29990.0, "pbp"))
        roll.assert_called_once()
        wm.assert_called_once()
        self.assertIs(load.call_args.kwargs["use_oracle_dividers"], False)

    def test_Meta_Opens_With_Rolling_Selected(self):
        sb = self._Block(self.oracle)
        sb._cooler_corr.setCurrentText("rolling")
        sb._Use_Dividers.setChecked(True)
        html = sb._build_meta_html(self.oracle)
        self.assertIn("30001.50", html)       # 3.0 x the recorded 10000.5

    def test_The_Merge_Dialog_Offers_Rolling(self):
        from gui.analysis.merge import MergeDialog
        sb = self._Block(self.legacy)
        sb._cooler_corr.setCurrentText("rolling")
        dialog = MergeDialog(sb._file_entries, sb.get_source_config())
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog._cooler_corr.currentText(), "rolling")


if __name__ == "__main__":
    unittest.main()
