"""Selective Import from Pre-Analysis (Analysis Upgrade, card B1; WP-P1).

The Source block's "Import from Pre-Analysis" asks which parts to take:
files, merged spectra, physics, gates, the main cooler/laser as the
Source override, and each run's own cooler/laser as a per-file override
(``run_overrides``, brief C2). The default ticks are today's import,
exactly; the choice is remembered with the session.

Run from the project root:
    .venv/bin/python -m pytest tests/test_selective_import.py -q
"""
import unittest
from unittest import mock
from unittest.mock import MagicMock

from PySide6.QtWidgets import QCheckBox, QDialog

from tests.test_import_only_checked import (
    _FakePA, _ensure_app, _fake_entry, _stub_add_file_entry)


def Spin(value):
    return MagicMock(value=MagicMock(return_value=value))


def Tick(on):
    cb = QCheckBox()
    cb.setChecked(on)
    return cb


def Fake_Pa():
    """Two ticked runs (one with its card's cooler and laser ticked), one
    unticked, and every control the import reads."""
    a = _fake_entry("/data/run_1.asdf")
    a.Cooler_Override, a.Laser_Override = 29950.0, 10920.25
    b = _fake_entry("/data/run_2.asdf")
    b.Cooler_Override, b.Laser_Override = None, None
    c = _fake_entry("/data/run_3.asdf", checked=False)
    c.Cooler_Override, c.Laser_Override = 29960.0, None
    pa = _FakePA([a, b, c])
    pa._e_lower, pa._e_upper, pa._harmonic = Spin(100.0), Spin(25068.5), Spin(3)
    pa._z_spin, pa._a_spin = Spin(27), Spin(59)
    pa._tof_enable, pa._tof_lo, pa._tof_hi = Tick(True), Spin(40.0), Spin(60.0)
    pa._channels = [Tick(on) for on in (True, False, True, False)]
    pa._cooler_override, pa._laser_override = Spin(29941.0), Spin(10652.717)
    return pa


class Selective_Import_Tests(unittest.TestCase):

    def setUp(self):
        _ensure_app()

    def _Import(self, sections=None):
        import gui.analysis.blocks as blocks_mod
        from gui.analysis.blocks import SourceBlock
        sb = SourceBlock()
        _stub_add_file_entry(sb)
        pa = Fake_Pa()
        with mock.patch.object(blocks_mod, "_choose_pa_project",
                               lambda parent: pa):
            if sections is None:
                sb._import_all_from_preanalysis()
            else:
                sb._import_all_from_preanalysis(sections)
        return sb

    def _State(self, sb):
        cfg = sb.get_source_config()
        keep = ("e_lower", "e_upper", "harmonic", "Z", "A", "tof_gate",
                "pmt_gate", "override_enabled", "cooler_override",
                "laser_override")
        return ([e["path"] for e in sb._file_entries],
                {k: cfg[k] for k in keep}, cfg.get("run_overrides"))

    def test_The_Default_Ticks_Are_Today_s_Import(self):
        from gui.analysis.blocks import PA_IMPORT_SECTIONS
        defaults = {key: on for key, _label, on, _tip in PA_IMPORT_SECTIONS}
        self.assertEqual(defaults, {
            "files": True, "merged": True, "physics": True, "gates": True,
            "main_override": False, "run_overrides": False})
        today = self._State(self._Import())
        self.assertEqual(self._State(self._Import(defaults)), today)
        paths, cfg, own = today
        self.assertEqual(paths, ["/data/run_1.asdf", "/data/run_2.asdf"])
        # Today the main values are copied but the Override stays off.
        self.assertEqual(cfg["cooler_override"], 29941.0)
        self.assertFalse(cfg["override_enabled"])
        self.assertIsNone(own)

    def test_Each_Section_Gates_Its_Part(self):
        off = dict.fromkeys(("files", "merged", "physics", "gates"), False)
        paths, cfg, _ = self._State(self._Import(dict(off, physics=True)))
        self.assertEqual(paths, [])
        self.assertEqual((cfg["Z"], cfg["A"], cfg["harmonic"]), (27, 59, 3))
        self.assertNotEqual(cfg["tof_gate"], [40.0, 60.0])
        self.assertEqual(cfg["pmt_gate"], [3, 4])           # the block's own
        paths, cfg, _ = self._State(self._Import(dict(off, gates=True)))
        self.assertEqual(paths, [])
        self.assertEqual(cfg["tof_gate"], [40.0, 60.0])
        self.assertEqual(cfg["pmt_gate"], [1, 3])
        self.assertEqual((cfg["Z"], cfg["harmonic"]), (1, 4))  # untouched

    def test_Nothing_Ticked_In_Pre_Analysis_Stops_Only_A_File_Import(self):
        import gui.analysis.blocks as blocks_mod
        from gui.analysis.blocks import SourceBlock
        sb = SourceBlock()
        _stub_add_file_entry(sb)
        pa = Fake_Pa()
        for fe in pa._file_entries:
            fe.check.setChecked(False)
        with mock.patch.object(blocks_mod, "_choose_pa_project",
                               lambda parent: pa), \
                mock.patch.object(blocks_mod.QMessageBox, "warning") as warn:
            sb._import_all_from_preanalysis(
                {"files": False, "merged": False, "physics": True,
                 "gates": False})
        warn.assert_not_called()
        self.assertEqual(sb.get_source_config()["Z"], 27)

    def test_The_Main_Values_Can_Be_Put_In_Force(self):
        _, cfg, _ = self._State(self._Import({"main_override": True}))
        self.assertTrue(cfg["override_enabled"])
        self.assertEqual((cfg["cooler_override"], cfg["laser_override"]),
                         (29941.0, 10652.717))

    def test_Each_Run_s_Own_Values_Become_Its_Override(self):
        sb = self._Import({"run_overrides": True})
        self.assertEqual(sb.get_source_config()["run_overrides"], {
            "/data/run_1.asdf": {"cooler_v": 29950.0, "laser_sp": 10920.25}})

    def test_Per_File_Overrides_And_The_Choice_Are_Saved(self):
        from gui.analysis.blocks import SourceBlock
        sb = self._Import({"run_overrides": True})
        sb._Pa_Import_Choice = {"files": True, "run_overrides": True}
        saved = sb.to_dict()
        files = {f["path"]: f for f in saved["files"]}
        self.assertEqual(files["/data/run_1.asdf"]["run_override"],
                         {"cooler_v": 29950.0, "laser_sp": 10920.25})
        self.assertNotIn("run_override", files["/data/run_2.asdf"])
        back = SourceBlock()
        _stub_add_file_entry(back)
        back.from_dict(saved)
        self.assertEqual(back.get_source_config()["run_overrides"],
                         sb.get_source_config()["run_overrides"])
        self.assertEqual(back._Pa_Import_Choice, sb._Pa_Import_Choice)

    def test_An_Older_Session_Has_No_Overrides_And_No_Choice(self):
        from gui.analysis.blocks import SourceBlock
        sb = self._Import()
        saved = sb.to_dict()
        self.assertNotIn("pa_import", saved)
        self.assertTrue(all("run_override" not in f for f in saved["files"]))
        back = SourceBlock()
        _stub_add_file_entry(back)
        back.from_dict(saved)
        self.assertNotIn("run_overrides", back.get_source_config())
        self.assertIsNone(back._Pa_Import_Choice)

    def test_The_Dialog_Imports_The_Ticks_And_Remembers_Them(self):
        from gui.analysis.blocks import SourceBlock
        sb = SourceBlock()
        seen, imported = [], []

        def Untick_Gates(dialog):
            ticks = {cb.text(): cb for cb in dialog.findChildren(QCheckBox)}
            seen.append({t: cb.isChecked() for t, cb in ticks.items()})
            ticks["Gates (ToF, PMT)"].setChecked(False)
            return QDialog.DialogCode.Accepted

        with mock.patch.object(QDialog, "exec", Untick_Gates), \
                mock.patch.object(sb, "_import_all_from_preanalysis",
                                  side_effect=imported.append):
            sb._Import_From_Preanalysis_Dialog()
            sb._Import_From_Preanalysis_Dialog()
        self.assertTrue(seen[0]["Gates (ToF, PMT)"])
        self.assertFalse(seen[1]["Gates (ToF, PMT)"])
        self.assertFalse(imported[0]["gates"])
        self.assertTrue(imported[0]["files"])

    def test_Cancel_Imports_Nothing(self):
        from gui.analysis.blocks import SourceBlock
        sb = SourceBlock()
        with mock.patch.object(QDialog, "exec",
                               lambda d: QDialog.DialogCode.Rejected), \
                mock.patch.object(sb, "_import_all_from_preanalysis") as imp:
            sb._Import_From_Preanalysis_Dialog()
        imp.assert_not_called()
        self.assertIsNone(sb._Pa_Import_Choice)


if __name__ == "__main__":
    unittest.main()
