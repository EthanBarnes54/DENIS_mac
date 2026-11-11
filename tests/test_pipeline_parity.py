"""Tests for the WP1 shared run-prep helper (2026-06-02 code review).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

gui.analysis.pipeline.prepare_run_data is the one place that loads and
physically interprets a run before binning. These tests pin the behaviour the
copy-pasted call sites used to get wrong, using a fake clstools CLSDataFrame
that records the calls (real clstools is not importable in CI):

* Frequency mode applies the COMPOSED Shift_Ref(ref_freq + ref_shift), never a
  bare ref_shift (the preview/diagnostics bug);
* ref_shift == 0 or apply_ref_shift=False -> no Shift_Ref (the auto-fitter
  must still get Compute_WL, just no shift);
* Raw Voltage mode skips Compute_WL and Shift_Ref entirely;
* under a cooler override, Vcool_init/VCoolDiv are fixed around Compute_WL and
  VCoolDiv is restored to 0 afterwards (the merge-backend bug);
* a virtual split intersects its [lo, hi] with the source V-gate and loads the
  parent ASDF path.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_pipeline_parity -v

Depends on: gui.analysis.pipeline.prepare_run_data (and binning helpers).
"""

import sys
import types
import unittest


class _FakeCLS:
    """Records the prep calls and snapshots cooler attrs at Compute_WL time."""

    def __init__(self):
        self.calls = []
        self.VCoolDiv = 10000
        self.VCoolOffset = 0
        self.Vcool_init = 7.0
        self.Laser_set = 12000.0
        self.Size = 100
        self.DAQTStime = 1.0
        self.TSstart = 0.0
        self.TSstop = 5.0
        self.Date = "2026-06-02"

    def Load_Run(self, path, cal_order=1):
        self.calls.append(("Load_Run", path, cal_order))

    def Compute_Voltages(self, cooler_correction="pbp"):
        self.calls.append(("Compute_Voltages", cooler_correction,
                           self.VCoolDiv, self.VCoolOffset))

    def Compute_WL(self, Mass=None, ref=None, harmonic=None):
        # Snapshot the cooler state seen by Compute_WL (the Vcool fix must be
        # active here when a cooler override is set).
        self.calls.append(("Compute_WL", Mass, ref, harmonic,
                           self.VCoolDiv, self.Vcool_init))

    def Shift_Ref(self, ref=None):
        self.calls.append(("Shift_Ref", ref))

    def apply_filter(self, filter_window=0):
        self.calls.append(("apply_filter", filter_window))


def _install_fake_clstools():
    mod = types.ModuleType("clstools")
    mod.CLSDataFrame = _FakeCLS
    sys.modules["clstools"] = mod


REF_FREQ = 5.0e14   # Hz (a rest-frame transition reference)
CFG = {
    "mass": 40.0,
    "ref_freq": REF_FREQ,
    "harmonic": 2,
    "bin_mode": "Frequency",
    "cooler_correction": "pbp",
}


def _cfg(**over):
    c = dict(CFG)
    c.update(over)
    return c


class PrepareRunDataTests(unittest.TestCase):
    def setUp(self):
        self._had = "clstools" in sys.modules
        self._orig = sys.modules.get("clstools")
        _install_fake_clstools()
        from gui.analysis import pipeline
        self.prepare = pipeline.prepare_run_data

    def tearDown(self):
        if self._had:
            sys.modules["clstools"] = self._orig
        else:
            sys.modules.pop("clstools", None)

    def _find(self, data, name):
        return [c for c in data.calls if c[0] == name]

    def test_frequency_composes_ref_shift(self):
        data, eff, meta = self.prepare("run.asdf", _cfg(ref_shift=12.0))
        wl = self._find(data, "Compute_WL")
        self.assertEqual(len(wl), 1)
        self.assertEqual(wl[0][2], REF_FREQ)            # Compute_WL ref
        sh = self._find(data, "Shift_Ref")
        self.assertEqual(len(sh), 1)
        # COMPOSED, not bare: ref_freq + ref_shift, never just ref_shift.
        self.assertAlmostEqual(sh[0][1], REF_FREQ + 12.0)
        self.assertNotAlmostEqual(sh[0][1], 12.0)

    def test_no_shift_when_ref_shift_zero(self):
        data, eff, meta = self.prepare("run.asdf", _cfg(ref_shift=0.0))
        self.assertEqual(len(self._find(data, "Compute_WL")), 1)
        self.assertEqual(self._find(data, "Shift_Ref"), [])

    def test_apply_ref_shift_false_skips_shift_keeps_wl(self):
        data, eff, meta = self.prepare(
            "run.asdf", _cfg(ref_shift=9.0), apply_ref_shift=False)
        self.assertEqual(len(self._find(data, "Compute_WL")), 1)
        self.assertEqual(self._find(data, "Shift_Ref"), [])

    def test_raw_voltage_skips_wl_and_shift(self):
        data, eff, meta = self.prepare(
            "run.asdf", _cfg(bin_mode="Raw Voltage", ref_shift=12.0))
        self.assertEqual(self._find(data, "Compute_WL"), [])
        self.assertEqual(self._find(data, "Shift_Ref"), [])

    def test_cooler_override_fixes_vcool_around_compute_wl(self):
        data, eff, meta = self.prepare(
            "run.asdf",
            _cfg(override_enabled=True, cooler_override=30000.0, ref_shift=0.0))
        wl = self._find(data, "Compute_WL")
        self.assertEqual(len(wl), 1)
        # At Compute_WL time the fix must be active: VCoolDiv=1, Vcool_init=override.
        self.assertEqual(wl[0][4], 1)            # VCoolDiv snapshot
        self.assertEqual(wl[0][5], 30000.0)      # Vcool_init snapshot
        # Restored afterwards.
        self.assertEqual(data.VCoolDiv, 0)
        self.assertEqual(data.VCoolOffset, 30000.0)

    def test_split_intersects_vgate_and_loads_parent(self):
        split = {
            "parent_path": "parent.asdf",
            "split_lo": 10.0,
            "split_hi": 20.0,
            "metadata_override": {},
        }
        data, eff, meta = self.prepare(
            "child.vasdf", _cfg(v_gate=[5.0, 15.0]),
            split_descriptor=split)
        # Loads the PARENT path, not the descriptor.
        load = self._find(data, "Load_Run")
        self.assertEqual(load[0][1], "parent.asdf")
        # Effective V-gate is the intersection [10, 15].
        self.assertEqual(list(eff["v_gate"]), [10.0, 15.0])

    def test_run_metadata_captured(self):
        data, eff, meta = self.prepare("run.asdf", _cfg())
        self.assertEqual(meta["laser_set"], 12000.0)
        self.assertEqual(meta["n_events"], 100)
        self.assertEqual(meta["date"], "2026-06-02")


def All_Toggles_Off(cfg):
    """``cfg`` with every upgrade switched off through UPGRADE_TOGGLES and
    every systematic perturbation at zero (Analysis Upgrade, WP-V1). For
    ``cooler_correction`` off means "not rolling": "mean" is an original
    method and stays."""
    from gui.analysis.pipeline import PERTURBATION_KEYS, UPGRADE_TOGGLES
    out = dict(cfg)
    for key, off in UPGRADE_TOGGLES:
        if key != "cooler_correction" or out.get(key) == "rolling":
            out[key] = off
    out.update(dict.fromkeys(PERTURBATION_KEYS, 0.0))
    return out


class Upgrades_Off_Tests(unittest.TestCase):
    """With every upgrade off, the pipeline is what it was: each fixture
    above gives the same clstools calls, cooler/laser state, effective
    config and run_metadata as with the keys absent."""

    setUp, tearDown = PrepareRunDataTests.setUp, PrepareRunDataTests.tearDown

    def test_Every_Toggle_Off_Is_The_Keys_Absent(self):
        split = {"parent_path": "parent.asdf", "split_lo": 10.0,
                 "split_hi": 20.0, "metadata_override": {}}
        cases = [("run.asdf", _cfg(ref_shift=12.0), {}),
                 ("run.asdf", _cfg(bin_mode="Raw Voltage"), {}),
                 ("run.asdf", _cfg(cooler_correction="mean"), {}),
                 ("run.asdf", _cfg(override_enabled=True,
                                   cooler_override=30000.0), {}),
                 ("run.asdf", _cfg(cooler_offset_v=-5.0), {}),
                 ("child.vasdf", _cfg(v_gate=[5.0, 15.0]),
                  {"split_descriptor": split})]
        for path, cfg, kw in cases:
            with self.subTest(cfg=cfg):
                off = All_Toggles_Off(cfg)
                self.assertGreater(len(off), len(cfg))
                d0, e0, m0 = self.prepare(path, cfg, **kw)
                d1, e1, m1 = self.prepare(path, off, **kw)
                self.assertEqual(d0.calls, d1.calls)
                for attr in ("VCoolDiv", "VCoolOffset", "Vcool_init",
                             "Laser_set"):
                    self.assertEqual(getattr(d0, attr), getattr(d1, attr))
                self.assertEqual(e0, {k: v for k, v in e1.items()
                                      if k in e0})
                self.assertEqual(m0, m1)

    def test_Mean_Stays_And_Rolling_Goes(self):
        self.assertEqual(All_Toggles_Off(_cfg(cooler_correction="mean"))
                         ["cooler_correction"], "mean")
        self.assertEqual(All_Toggles_Off(_cfg(cooler_correction="rolling"))
                         ["cooler_correction"], "pbp")


if __name__ == "__main__":
    unittest.main()
