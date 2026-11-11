"""The Source block's dead-time control (Analysis Upgrade, A4's control).

Under Advanced Data Settings, a dead time per PMT (0 = off) feeds
``dead_time_us``; "Suggested" fills in ORACLE's suggestion from the files'
performance reports only when pressed (D-3: never applied by itself). The
Preview applies the gate as the fit does.

Run from the project root:
    .venv/bin/python -m pytest tests/test_source_dead_time_control.py -q
"""
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from tests.test_import_only_checked import _ensure_app
from tests.test_oracle_ingestion import Write_Run


class Dead_Time_Control_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_app()
        cls.tmp = tempfile.mkdtemp(prefix="dead_time_control_")
        cls.plain = os.path.join(cls.tmp, "run_0031.asdf")
        cls.suggested = os.path.join(cls.tmp, "run_0032.asdf")
        Write_Run(cls.plain)
        Write_Run(cls.suggested)
        with open(os.path.join(cls.tmp, "run_0032_performance.yaml"),
                  "w") as fh:
            fh.write("detectors:\n"
                     "  channel 3: {counts: 9, suggested_dead_time_us: 10.0}\n")

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

    def test_Off_The_Config_And_The_Save_Are_As_They_Were(self):
        sb = self._Block(self.plain)
        self.assertNotIn("dead_time_us", sb.get_source_config())
        self.assertNotIn("dead_time_us", sb.to_dict())

    def test_A_PMT_s_Dead_Time_Reaches_The_Config_And_Is_Saved(self):
        sb = self._Block()
        sb._Dead_Times[3].setValue(10.0)
        self.assertEqual(sb.get_source_config()["dead_time_us"], {3: 10.0})
        back = self._Block()
        back.from_dict(sb.to_dict())
        self.assertEqual(back.get_source_config()["dead_time_us"], {3: 10.0})
        self.assertEqual(back._Dead_Times[4].value(), 0.0)

    def test_The_Suggestion_Is_Filled_In_Only_When_Asked(self):
        sb = self._Block(self.plain)
        self.assertFalse(sb._Dead_Suggest.isEnabled())
        sb._add_file_entry(self.suggested)
        self.assertTrue(sb._Dead_Suggest.isEnabled())
        self.assertNotIn("dead_time_us", sb.get_source_config())
        sb._Dead_Suggest.click()
        self.assertEqual(sb.get_source_config()["dead_time_us"], {3: 10.0})

    def test_The_Preview_Applies_The_Gate(self):
        from gui.analysis import blocks

        class _Stop(Exception):
            pass

        class _D:
            VCoolDiv, VCoolOffset, Vcool_init, Laser_set = 10000, 0, 3.0, 1.0

            def Compute_Voltages(self, cooler_correction="pbp"):
                raise _Stop

        sb = self._Block()
        sb._Dead_Times[2].setValue(5.0)
        with mock.patch.object(sb, "get_checked_files",
                               return_value=[{"path": "run_1.asdf"}]), \
                mock.patch.object(blocks, "_get_clstools",
                                  return_value=types.SimpleNamespace(
                                      CLSDataFrame=_D)), \
                mock.patch("gui.calibration.load_run_calibrated"), \
                mock.patch("gui.analysis.pipeline.Dead_Time_Gate") as gate, \
                mock.patch.object(blocks, "QMessageBox"):
            sb._show_preview_at_index()
        self.assertEqual(gate.call_args.args[1],
                         {1: 0.0, 2: 5.0, 3: 0.0, 4: 0.0})


if __name__ == "__main__":
    unittest.main()
