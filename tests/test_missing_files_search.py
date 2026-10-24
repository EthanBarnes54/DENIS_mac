"""Finding missing session files: folder search, folder moves, and a
data-loss incident.

A session file once lost every run list: the old flow located one
file (the wrong one), the user cancelled the next prompt, and the loader
wrote the *pruned* session back to disk. These tests pin the fix (the
file keeps everything still missing, with a .bak of the original) and
the new ways of finding files: searching a folder (recursively or not),
same-name tie-breaks by path tail, a hand-located file applied as a
folder move to the rest, the name-mismatch guard, and a deep remap that
also reaches GP-correction keys and merge sources.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_missing_files_search.py -q
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QFileDialog, QMessageBox,
)

from gui.missing_files import (  # noqa: E402
    apply_prefix, derive_prefix, index_folder, match_in_index,
    remap_paths_inplace, suffix_score,
)
from gui.shared_widgets import convert_path_for_os  # noqa: E402

_APP = QApplication.instance() or QApplication([])

OLD = "C:/data/old_place/denis_session"


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"#ASDF")
    return path


class _TreeCase(unittest.TestCase):
    """A moved session: <tmp>/moved/denis_session/{session.yaml, Data}."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.new = os.path.join(self.tmp, "moved", "denis_session")
        self.runs = [1001, 1002, 1003]
        self.files = [_touch(os.path.join(self.new, "Data", "01",
                                          f"run_{r}.asdf"))
                      for r in self.runs]
        _touch(os.path.join(self.new, "Data", "01", "run_1010.asdf"))
        self.missing = [f"{OLD}/Data/01/run_{r}.asdf" for r in self.runs]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class HelperTests(_TreeCase):

    def test_suffix_score_counts_shared_tail(self):
        self.assertEqual(suffix_score(
            "C:/a/Data/01/run_1.asdf", r"D:\x\Data\01\run_1.asdf"), 3)
        self.assertEqual(suffix_score("C:/a/run_1.asdf", "C:/a/run_2.asdf"),
                         0)

    def test_recursive_search_finds_all(self):
        index, complete = index_folder(self.tmp, self.missing)
        self.assertTrue(complete)
        found, amb = match_in_index(self.missing, index)
        self.assertEqual(len(found), 3)
        self.assertEqual(amb, {})

    def test_flat_search_stays_in_the_folder(self):
        index, _ = index_folder(self.tmp, self.missing, recursive=False)
        self.assertEqual(index, {})
        index, _ = index_folder(os.path.dirname(self.files[0]),
                                self.missing, recursive=False)
        self.assertEqual(len(index), 3)

    def test_directory_cap_reports_incomplete(self):
        _, complete = index_folder(self.tmp, self.missing, max_dirs=2)
        self.assertFalse(complete)

    def test_same_name_resolved_by_path_tail(self):
        stray = _touch(os.path.join(self.tmp, "copies", "run_1001.asdf"))
        index, _ = index_folder(self.tmp, self.missing)
        found, amb = match_in_index(self.missing, index)
        self.assertEqual(os.path.normcase(found[self.missing[0]]),
                         os.path.normcase(self.files[0]))
        self.assertNotEqual(found[self.missing[0]], stray)

    def test_equal_ties_are_ambiguous(self):
        a = _touch(os.path.join(self.tmp, "A", "run_1999.asdf"))
        b = _touch(os.path.join(self.tmp, "B", "run_1999.asdf"))
        orig = f"{OLD}/elsewhere/run_1999.asdf"
        index, _ = index_folder(self.tmp, [orig])
        found, amb = match_in_index([orig], index)
        self.assertEqual(found, {})
        self.assertEqual(sorted(amb[orig]), sorted([a, b]))

    def test_one_located_file_gives_the_folder_move(self):
        move = derive_prefix(self.missing[0], self.files[0])
        self.assertIsNotNone(move)
        got = apply_prefix(self.missing[1:], *move)
        self.assertEqual(
            {k: os.path.normcase(v) for k, v in got.items()},
            {m: os.path.normcase(f)
             for m, f in zip(self.missing[1:], self.files[1:])})

    def test_no_move_from_a_different_file(self):
        self.assertIsNone(derive_prefix(
            self.missing[0], os.path.join(self.new, "Data", "01",
                                          "run_1010.asdf")))

    def test_move_only_maps_files_that_exist(self):
        move = derive_prefix(self.missing[0], self.files[0])
        gone = f"{OLD}/Data/01/run_1234.asdf"
        self.assertNotIn(gone, apply_prefix([gone], *move))


class DeepRemapTests(unittest.TestCase):
    """Paths outside the file lists follow the move too."""

    def test_corrections_keys_and_merge_sources(self):
        old = f"{OLD}/Data/01/run_1001.asdf"
        new = "C:/new/Data/01/run_1001.asdf"
        d = {
            "analysis": {"projects": [{"blocks": [{
                "type": "Source",
                "files": [{"path": old}],
                "merged_entries": [{"source_files": [old]}]}]}],
                "isotope_shifts": {"corrections": {"116Sn": {
                    old: {"run_number": "1001"}}}}},
        }
        remap_paths_inplace(d, {old: new})
        an = d["analysis"]
        blk = an["projects"][0]["blocks"][0]
        self.assertEqual(blk["files"][0]["path"], new)
        self.assertEqual(blk["merged_entries"][0]["source_files"], [new])
        self.assertEqual(list(an["isotope_shifts"]["corrections"]["116Sn"]),
                         [new])

    def test_Windows_Registry_Keys_Follow_A_Move_To_Posix(self):
        """A Windows session keys its calibrations by Windows' canonical
        spelling; relocated on a Mac, the key must still find its file
        entry and move with it (2026-10-22)."""
        old = "C:/Users/X/Data/01/run_1001.asdf"
        new = "/Users/x/denis_session/Data/01/run_1001.asdf"
        spec = {"mode": "coeffs", "coeffs_v": [0.35, 1.0007]}
        d = {
            "analysis": {"projects": [{"blocks": [{
                "type": "Source", "files": [{"path": old}]}]}]},
            "calibrations": {r"c:\users\x\data\01\run_1001.asdf": spec},
        }
        remap_paths_inplace(d, {old: new})
        self.assertEqual(d["calibrations"], {new: spec})


class ConvertPathTests(unittest.TestCase):
    """Auto-conversion of a path saved on another OS, per host OS."""

    def test_Paths_Convert_For_Each_Host(self):
        """macOS gets /Volumes/<share> for a UNC path and no invented /c/
        drive path (2026-10-22); Linux and Windows are as before."""
        cases = [
            ("Darwin", r"C:\Users\X\run.asdf", "C:/Users/X/run.asdf"),
            ("Darwin", r"\\srv\share\Data\run.asdf",
             "/Volumes/share/Data/run.asdf"),
            ("Darwin", "/Users/x/run.asdf", "/Users/x/run.asdf"),
            ("Linux", r"C:\Users\X\run.asdf", "/c/Users/X/run.asdf"),
            ("Linux", r"\\srv\share\run.asdf", "//srv/share/run.asdf"),
            ("Windows", "/c/Users/X/run.asdf", r"C:\Users\X\run.asdf"),
        ]
        for system, path, want in cases:
            with self.subTest(system=system, path=path), \
                    mock.patch("platform.system", return_value=system):
                self.assertEqual(convert_path_for_os(path), want)


class DialogTests(_TreeCase):

    def _dialog(self, auto=True):
        from gui.missing_files_dialog import MissingFilesDialog
        return MissingFilesDialog(self.missing,
                                  os.path.join(self.new, "session.yaml"),
                                  auto_search=auto)

    def test_moved_session_resolves_without_clicks(self):
        dlg = self._dialog()
        self.assertEqual(len(dlg.remap()), 3)
        self.assertEqual(dlg.unresolved(), [])

    def test_one_pick_resolves_the_rest(self):
        dlg = self._dialog(auto=False)
        # A second prompt (there should be none) is cancelled, and any
        # name-mismatch question answered No, so nothing can block.
        with mock.patch.object(QFileDialog, "getOpenFileName",
                               side_effect=[(self.files[0], ""),
                                            ("", "")]) as ask, \
                mock.patch.object(QMessageBox, "question",
                                  return_value=QMessageBox.StandardButton.No):
            dlg.locate(dlg.unresolved())
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(dlg.unresolved(), [])

    def test_wrong_name_is_refused_unless_confirmed(self):
        wrong = os.path.join(self.new, "Data", "01", "run_1010.asdf")
        dlg = self._dialog(auto=False)
        with mock.patch.object(QFileDialog, "getOpenFileName",
                               side_effect=[(wrong, ""), ("", "")]), \
                mock.patch.object(QMessageBox, "question",
                                  return_value=QMessageBox.StandardButton.No):
            dlg.locate(dlg.unresolved())
        self.assertEqual(dlg.remap(), {})

    def test_cancel_keeps_what_was_found(self):
        other = os.path.join(self.tmp, "elsewhere", "run_1001.asdf")
        _touch(other)
        dlg = self._dialog(auto=False)
        # A pick whose folder move resolves nothing else, then Cancel.
        with mock.patch.object(QFileDialog, "getOpenFileName",
                               side_effect=[(other, ""), ("", "")]):
            dlg.locate(dlg.unresolved())
        self.assertEqual(list(dlg.remap()), [self.missing[0]])
        self.assertEqual(len(dlg.unresolved()), 2)

    def test_search_button_path(self):
        dlg = self._dialog(auto=False)
        self.assertEqual(dlg.search(self.tmp, recursive=False), 0)
        self.assertEqual(dlg.search(self.tmp, recursive=True), 3)


class _FakeStatus:
    def __init__(self):
        self.text = ""

    def show(self, text, *a, **k):
        self.text = text


class _FakeWindow:
    """Just what MainWindow._resolve_missing_files touches."""

    def __init__(self):
        self.status = _FakeStatus()

    def _write_yaml(self, path, d):
        from gui.main_window import MainWindow
        MainWindow._write_yaml(self, path, d)


class LoadFlowDataLossTests(_TreeCase):
    """The data-loss incident, end to end through the load flow."""

    def setUp(self):
        super().setUp()
        self.yaml_path = os.path.join(self.new, "session.yaml")
        self.session = {"preanalysis": {"projects": [{
            "name": "116Sn_session",
            "files": [{"path": p, "checked": True} for p in self.missing],
        }]}}
        with open(self.yaml_path, "w") as fh:
            yaml.dump(self.session, fh)

    def _run(self, remap, accepted=True):
        from gui.main_window import MainWindow

        class _Dlg:
            def __init__(self, *a, **k):
                pass

            def exec(self):
                return (QDialog.DialogCode.Accepted if accepted
                        else QDialog.DialogCode.Rejected)

            def remap(self):
                return dict(remap)

        raw = yaml.safe_load(open(self.yaml_path))
        with mock.patch("gui.missing_files_dialog.MissingFilesDialog",
                        _Dlg), \
                mock.patch("gui.main_window.QMessageBox"):
            ok = MainWindow._resolve_missing_files(
                _FakeWindow(), raw, self.yaml_path)
        return ok, raw, yaml.safe_load(open(self.yaml_path))

    def test_partial_locate_keeps_every_entry_on_disk(self):
        ok, raw, disk = self._run({self.missing[0]: self.files[0]})
        self.assertTrue(ok)
        on_disk = [f["path"] for f in
                   disk["preanalysis"]["projects"][0]["files"]]
        self.assertEqual(on_disk, [self.files[0]] + self.missing[1:])
        # This load only carries the file that was found.
        self.assertEqual(
            [f["path"] for f in raw["preanalysis"]["projects"][0]["files"]],
            [self.files[0]])

    def test_original_is_backed_up_once(self):
        self._run({self.missing[0]: self.files[0]})
        bak = yaml.safe_load(open(self.yaml_path + ".bak"))
        self.assertEqual(bak, self.session)
        # A second rewrite must not replace the original backup.
        self._run({self.missing[1]: self.files[1]})
        self.assertEqual(yaml.safe_load(open(self.yaml_path + ".bak")),
                         self.session)

    def test_cancel_aborts_and_leaves_file_alone(self):
        ok, _raw, disk = self._run({}, accepted=False)
        self.assertFalse(ok)
        self.assertEqual(disk, self.session)
        self.assertFalse(os.path.exists(self.yaml_path + ".bak"))

    def test_nothing_found_writes_nothing(self):
        ok, raw, disk = self._run({})
        self.assertTrue(ok)
        self.assertEqual(disk, self.session)
        self.assertEqual(raw["preanalysis"]["projects"][0]["files"], [])


if __name__ == "__main__":
    unittest.main()
