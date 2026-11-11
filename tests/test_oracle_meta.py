"""ORACLE's record in Meta, the file viewer and project files (Analysis
Upgrade, card G1; WP-O1, WP-O2).

* ``Oracle_Summary`` turns the record into (label, text) rows, whatever
  its version: every key is optional, and no record gives no rows.
* The Source block's Meta shows them in an "ORACLE" section, and shows
  nothing new for a run without a record.
* The file viewer documents ``DacCorrection``, ``DacCorrectionApplied``
  and ``ORACLE``, and notes what the record is at a glance.
* Files added to a Sample project that ORACLE tagged as reference runs
  are offered, once and never automatically, to a Reference project.

Run from the project root:
    .venv/bin/python -m pytest tests/test_oracle_meta.py -q
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from cls_estimations.asdf_contents import FIELD_DOCS, Oracle_Summary
from tests.test_import_only_checked import _ensure_app
from tests.test_oracle_ingestion import Write_Run

RECORD = {
    "Version": "0.6.0", "Rack": "hardware", "ScanType": "Triangle",
    "Rounds": 3, "InteractionRegion": "Region 2", "ReferenceRun": True,
    "Commit": {"Hash": "0123456789abcdef0123", "Dirty": True},
    "Calibration": {
        "Verdict": "WARN", "FitResidualRmsV": 0.1126,
        "Checks": [{"Check": "tracking", "Message": "|r| = 0.99998",
                    "Verdict": "PASS"},
                   {"Check": "span", "Message": "59 V of 60 V",
                    "Verdict": "WARN"}]},
    "TableFit": {"Table": "abcdef012345", "A_V": 0.0123, "B": 1.000456},
}


class Oracle_Summary_Tests(unittest.TestCase):

    def test_A_Full_Record_Gives_Every_Row(self):
        rows = dict(Oracle_Summary(RECORD))
        self.assertEqual(rows["Version"], "0.6.0")
        self.assertEqual(rows["Commit"], "0123456789  (files changed since)")
        self.assertEqual(rows["Rack"], "hardware")
        self.assertEqual(rows["Scan"], "Triangle, 3 rounds")
        self.assertEqual(rows["Region"], "Region 2")
        self.assertEqual(rows["Reference run"], "yes")
        self.assertEqual(rows["Calibration"],
                         "WARN; fit rms 0.113 V; span: 59 V of 60 V")
        self.assertEqual(rows["Table fit"], "abcdef01: V = 0.0123 + 1.00046 T")

    def test_Older_And_Odd_Records_Give_What_They_Have(self):
        self.assertEqual(Oracle_Summary(None), [])
        self.assertEqual(Oracle_Summary({}), [])
        self.assertEqual(Oracle_Summary({"Version": "0.2.0", "Rack": "simulated",
                                         "ScanType": "Triangle"}),
                         [("Version", "0.2.0"), ("Rack", "simulated"),
                          ("Scan", "Triangle")])
        why = Oracle_Summary({"TableFit": "No bench table in force."})
        self.assertEqual(why, [("Table fit", "No bench table in force.")])
        self.assertEqual(Oracle_Summary({"Commit": None, "TableFit": None,
                                         "Calibration": {}}), [])


class Meta_And_Viewer_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_app()
        cls.tmp = tempfile.mkdtemp(prefix="oracle_meta_")
        cls.legacy = os.path.join(cls.tmp, "run_0021.asdf")
        cls.oracle = os.path.join(cls.tmp, "run_0022.asdf")
        Write_Run(cls.legacy)
        Write_Run(cls.oracle, RECORD)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Meta(self, path):
        from gui.analysis.blocks import SourceBlock
        sb = SourceBlock()
        self.addCleanup(sb.deleteLater)
        sb._add_file_entry(path)
        return sb._build_meta_html(path)

    def test_Meta_Shows_The_Record_Of_An_Oracle_Run(self):
        html = self._Meta(self.oracle)
        self.assertIn("--- ORACLE ---", html)
        self.assertIn("Scan:           Triangle, 3 rounds", html)
        self.assertIn("Calibration:    WARN; fit rms 0.113 V", html)
        # After everything Meta showed before.
        self.assertGreater(html.index("--- ORACLE ---"),
                           html.index("--- General ---"))

    def test_A_Run_Without_A_Record_Has_The_Meta_It_Had(self):
        self.assertNotIn("ORACLE", self._Meta(self.legacy))

    def test_The_Viewer_Documents_The_Record_And_Notes_It(self):
        for key in ("DacCorrection", "DacCorrectionApplied", "ORACLE"):
            with self.subTest(key=key):
                self.assertTrue(FIELD_DOCS[key])
        from gui.asdf_viewer import _NOTES
        self.assertEqual(_NOTES["ORACLE"](RECORD),
                         "0.6.0, hardware, Triangle, 3 rounds")


class Reference_Runs_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_app()
        cls.tmp = tempfile.mkdtemp(prefix="oracle_reference_")
        cls.sample = os.path.join(cls.tmp, "run_0031.asdf")
        cls.tagged = os.path.join(cls.tmp, "run_0032.asdf")
        cls.legacy = os.path.join(cls.tmp, "run_0033.asdf")
        Write_Run(cls.sample, {"Version": "0.6.0", "ReferenceRun": False})
        Write_Run(cls.tagged, {"Version": "0.6.0", "ReferenceRun": True})
        Write_Run(cls.legacy)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Tab(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        return at

    def _Add(self, at, project, *paths, choice=None):
        """Add ``paths`` through the Source block's Add files button."""
        from PySide6.QtWidgets import QFileDialog, QInputDialog
        from gui.analysis.blocks import SourceBlock
        block = next(b for b in project._blocks
                     if isinstance(b, SourceBlock))
        with mock.patch.object(QFileDialog, "getOpenFileNames",
                               return_value=(list(paths), "")), \
                mock.patch.object(QInputDialog, "getItem",
                                  return_value=choice or ("", False)) as ask:
            block._add_files()
        return block, ask

    def _Paths(self, project):
        from gui.analysis.blocks import SourceBlock
        return [e["path"] for b in project._blocks
                if isinstance(b, SourceBlock) for e in b._file_entries]

    def test_It_Is_Offered_Once_And_Not_Taken_Unasked(self):
        at = self._Tab()
        fe = at._add_project("Fe56")
        _b, ask = self._Add(at, fe, self.sample, self.tagged, self.legacy)
        ask.assert_called_once()
        self.assertIn("run_0032.asdf", ask.call_args.args[2])
        self.assertNotIn("run_0031.asdf", ask.call_args.args[2])
        self.assertEqual(self._Paths(fe),
                         [self.sample, self.tagged, self.legacy])
        self.assertEqual(len(at._projects), 1)         # Cancel: nothing moved

    def test_Untagged_Files_Raise_No_Question(self):
        at = self._Tab()
        fe = at._add_project("Fe56")
        _b, ask = self._Add(at, fe, self.sample, self.legacy)
        ask.assert_not_called()

    def test_It_Moves_Them_To_A_New_Reference_Project(self):
        at = self._Tab()
        fe = at._add_project("Fe56")
        self._Add(at, fe, self.sample, self.tagged,
                  choice=("New Reference project (Reference_2)", True))
        ref = at._projects[-1]
        self.assertTrue(ref.is_reference)
        self.assertEqual(ref.project_name, "Reference_2")
        self.assertEqual(self._Paths(ref), [self.tagged])
        self.assertEqual(self._Paths(fe), [self.sample])

    def test_It_Moves_Them_To_The_Reference_Project_Chosen(self):
        at = self._Tab()
        fe = at._add_project("Fe56")
        ref = at._add_project("Fe54", is_reference=True)
        self._Add(at, fe, self.tagged, choice=("Fe54", True))
        self.assertEqual(len(at._projects), 2)
        self.assertEqual(self._Paths(ref), [self.tagged])
        self.assertEqual(self._Paths(fe), [])

    def test_A_Reference_Project_Is_Not_Asked(self):
        at = self._Tab()
        ref = at._add_project("Fe54", is_reference=True)
        _b, ask = self._Add(at, ref, self.tagged)
        ask.assert_not_called()
        self.assertEqual(self._Paths(ref), [self.tagged])


if __name__ == "__main__":
    unittest.main()
