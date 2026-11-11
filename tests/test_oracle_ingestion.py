"""ORACLE ingestion on the Analysis load path (Analysis Upgrade, card A1).

* The record reader (brief C1): ``Oracle_Record``, ``Monitor_Of`` and
  ``Sidecar_Yaml`` read what ORACLE writes in and beside a run, and give
  None for a legacy run.
* Dividers from the record (WP-D1): with ``use_oracle_dividers`` on, the
  run's recorded ratios replace clstools' defaults, so the voltages move by
  exactly the ratio's ppm. Off, or on a run without a record, every array is
  what it was.
* Per-file cooler/laser overrides (WP-D4): the Source block's override
  outranks the run's own, which outranks a split's, which outranks the file,
  in the fit and the merge alike.

Built on synthetic ASDF runs driven through the real clstools, as in
test_cooler_offset.py: the point is what clstools does with the numbers.

Run from the project root:
    .venv/bin/python -m pytest tests/test_oracle_ingestion.py -q
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from gui.analysis.binning import compute_binned
from gui.analysis.pipeline import UPGRADE_TOGGLES, prepare_run_data
from gui.calibration import (
    Monitor_Of, Oracle_Record, Sidecar_Yaml, clear_points_cache)

COOLER_KV = 3.0          # * VCoolDiv (10000) = 30 kV
LASER = 20000.0
SOURCE_CFG = dict(
    mass=100.0, ref_freq=4.0e14, harmonic=2, bin_mode="Frequency",
    cooler_correction="pbp", cal_order=1, x_column="bins_center",
    yerr_mode="Poisson sqrt(y+1)", xerr_mode="None",
    bin_definition="Per scan step", pmt_gate=[3, 4], tof_gate=None,
    v_gate=None, f_gate=None, noise_filter=0, ref_shift=0.0,
)
MONITOR = ("time,round,setpoint_v,readback_v,cooler_v,wavenumber_cm1,"
           "bunches,dropped,skipped\n"
           "10.0,1,20,19.98,29885.2,16956.17,1,0,0\n"
           "10.1,1,25,24.97,29885.6,16956.18,2,0,0\n")


def Write_Run(path, record=None):
    """A small run in the DAQ format; ``record`` goes under ``ORACLE``."""
    import asdf
    rng = np.random.default_rng(7)
    steps = np.linspace(0.0, 400.0, 81)
    rows = []
    for i, s in enumerate(steps):
        k = int(rng.poisson(20 + 200 * np.exp(-0.5 * ((s - 200) / 15) ** 2)))
        for _ in range(k):
            rows.append([i * 0.1, s, i, rng.choice([3, 4]),
                         rng.normal(50, 3), COOLER_KV])
    cal_set = np.linspace(0.0, 400.0, 21)
    tree = {
        "Run": 1, "CoolerVoltage": COOLER_KV, "LaserSetpoint": LASER,
        "DwellTime": 0.1, "Experiment": "A1", "Date": "2026-11-11",
        "StepSize": 5.0, "ScanningRanges": [[0.0, 400.0]],
        # Readbacks in monitor units (set / 1000): an identity calibration
        # at the default scan divider.
        "CalSet": cal_set, "CalReadback": cal_set / 1000.0,
        "raw": np.array(rows, dtype=float),
    }
    if record is not None:
        tree["ORACLE"] = record
    asdf.AsdfFile(tree).write_to(path)


class Oracle_Ingestion_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oracle_ingestion_")
        cls.legacy = os.path.join(cls.tmp, "run_0001.asdf")
        cls.cooler = os.path.join(cls.tmp, "run_0002.asdf")
        cls.scan = os.path.join(cls.tmp, "run_0003.asdf")
        cls.bare = os.path.join(cls.tmp, "run_0004.asdf")
        Write_Run(cls.legacy)
        Write_Run(cls.cooler, {
            "Version": "0.6.0",
            "Dividers": {"Cooler": {"ratio": 10000.5, "ppm": 100.0}},
            "TableFit": {"Table": np.arange(4.0), "A_V": 1.5}})
        Write_Run(cls.scan, {
            "Dividers": {"Scan": {"ratio": 1000.1, "ppm": 100.0}}})
        Write_Run(cls.bare, {"Version": "0.5.0"})
        with open(os.path.join(cls.tmp, "run_0002_monitor.csv"), "w") as fh:
            fh.write(MONITOR)
        for kind in ("performance", "systematics"):
            with open(os.path.join(cls.tmp, f"run_0002_{kind}.yaml"),
                      "w") as fh:
                fh.write(f"kind: {kind}\nsources: [{{source: a, volts: 0.1}}]\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        clear_points_cache()

    def _Prep(self, path, **extra):
        cfg = dict(SOURCE_CFG)
        cfg.update(extra)
        return prepare_run_data(path, cfg)

    def _Voltages(self, path, **extra):
        data, _eff, meta = self._Prep(path, **extra)
        return data.Sorted["V"].to_numpy(), data.Sorted["DV_cal"].to_numpy(), meta

    # ── the reader (C1) ──

    def test_Oracle_Record_Reads_The_Record(self):
        record = Oracle_Record(self.cooler)
        self.assertEqual(record["Version"], "0.6.0")
        self.assertEqual(record["Dividers"]["Cooler"]["ratio"], 10000.5)
        # Arrays come out as plain numpy, usable after the file is closed.
        table = record["TableFit"]["Table"]
        self.assertIsInstance(table, np.ndarray)
        np.testing.assert_array_equal(table, np.arange(4.0))

    def test_Legacy_And_Missing_Runs_Give_None(self):
        missing = os.path.join(self.tmp, "run_0099.asdf")
        for path in (self.legacy, missing):
            self.assertIsNone(Oracle_Record(path))
            self.assertIsNone(Monitor_Of(path))
            self.assertIsNone(Sidecar_Yaml(path, "performance"))
            self.assertIsNone(Sidecar_Yaml(path, "systematics"))

    def test_Record_Follows_A_Rewritten_File(self):
        path = os.path.join(self.tmp, "run_0010.asdf")
        Write_Run(path, {"Version": "0.5.0"})
        self.assertEqual(Oracle_Record(path)["Version"], "0.5.0")
        Write_Run(path, {"Version": "0.6.0"})
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.assertEqual(Oracle_Record(path)["Version"], "0.6.0")

    def test_Monitor_Of_Reads_Every_Column(self):
        monitor = Monitor_Of(self.cooler)
        self.assertEqual(set(monitor), set(MONITOR.split("\n")[0].split(",")))
        np.testing.assert_array_equal(monitor["cooler_v"], [29885.2, 29885.6])
        np.testing.assert_array_equal(monitor["wavenumber_cm1"],
                                      [16956.17, 16956.18])

    def test_Sidecar_Yaml_Reads_Both_Kinds(self):
        for kind in ("performance", "systematics"):
            sidecar = Sidecar_Yaml(self.cooler, kind)
            self.assertEqual(sidecar["kind"], kind)
            self.assertEqual(sidecar["sources"][0]["volts"], 0.1)
        with self.assertRaises(ValueError):
            Sidecar_Yaml(self.cooler, "monitor")

    # ── dividers (D1) ──

    def test_Cooler_Divider_Moves_The_Voltage_By_Its_Ppm(self):
        v_off, cal_off, meta_off = self._Voltages(self.cooler)
        v_on, cal_on, meta_on = self._Voltages(self.cooler,
                                               use_oracle_dividers=True)
        np.testing.assert_array_equal(cal_on, cal_off)
        # V = Vrfq * VCoolDiv + ... : 3.0 * 0.5 = 1.5 V, 50 ppm of 30 kV.
        np.testing.assert_allclose(v_on - v_off, COOLER_KV * 0.5, rtol=1e-9)
        self.assertAlmostEqual(meta_on["cooler_v"], COOLER_KV * 10000.5,
                               places=6)
        self.assertEqual(meta_on["dividers"],
                         {"scan": 1000.0, "cooler": 10000.5})
        self.assertNotIn("dividers", meta_off)

    def test_Scan_Divider_Scales_The_Calibration_By_Its_Ppm(self):
        v_off, cal_off, _ = self._Voltages(self.scan)
        v_on, cal_on, meta_on = self._Voltages(self.scan,
                                               use_oracle_dividers=True)
        nonzero = np.abs(cal_off) > 1.0     # DV = 0 leaves a ~1e-15 intercept
        # The volts fit reads the readbacks through the recorded ratio:
        # 1000.1 / 1000 = 100 ppm on every calibrated scan voltage.
        np.testing.assert_allclose(cal_on[nonzero] / cal_off[nonzero],
                                   1.0001, rtol=1e-9)
        np.testing.assert_allclose(v_off - v_on, cal_on - cal_off,
                                   rtol=1e-9, atol=1e-9)
        self.assertEqual(meta_on["dividers"],
                         {"scan": 1000.1, "cooler": 10000.0})

    def test_Dividers_Off_Or_Unrecorded_Is_Byte_Identical(self):
        def Spectrum(path, **extra):
            data, eff, meta = self._Prep(path, **extra)
            res = compute_binned(data, eff)
            return (data.Sorted["V"].to_numpy(), res["x"], res["y"],
                    meta)
        base = Spectrum(self.legacy)
        for path, extra in ((self.cooler, {}), (self.scan, {}),
                            (self.legacy, {"use_oracle_dividers": False}),
                            (self.legacy, {"use_oracle_dividers": True}),
                            (self.bare, {"use_oracle_dividers": True})):
            other = Spectrum(path, **extra)
            for a, b in zip(base[:3], other[:3]):
                np.testing.assert_array_equal(a, b)
            meta = dict(other[3])
            if extra.get("use_oracle_dividers"):
                self.assertEqual(meta.pop("dividers"),
                                 {"scan": 1000.0, "cooler": 10000.0})
            self.assertEqual(meta, base[3])

    # ── per-file overrides (D4) ──

    def test_A_Run_s_Own_Override_Applies_To_It_Alone(self):
        own = {self.legacy: {"cooler_v": 29990.0, "laser_sp": 15000.0}}
        data, _eff, meta = self._Prep(self.legacy, run_overrides=own)
        self.assertEqual(meta["cooler_mode"], "constant")
        self.assertEqual(meta["cooler_v_effective"], 29990.0)
        self.assertEqual(data.Laser_set, 15000.0)
        # The file's own values are still what the record says it held.
        self.assertAlmostEqual(meta["cooler_v"], COOLER_KV * 10000.0,
                               places=6)
        self.assertEqual(meta["laser_set"], LASER)
        data, _eff, meta = self._Prep(
            self.cooler, run_overrides=own)
        self.assertEqual(meta["cooler_mode"], "readback")
        self.assertEqual(data.Laser_set, LASER)

    def test_The_Source_Override_Outranks_The_Run_s_Own(self):
        own = {self.legacy: {"cooler_v": 29990.0, "laser_sp": 15000.0}}
        data, _eff, meta = self._Prep(
            self.legacy, run_overrides=own, override_enabled=True,
            cooler_override=29950.0, laser_override=16000.0)
        self.assertEqual(meta["cooler_v_effective"], 29950.0)
        self.assertEqual(data.Laser_set, 16000.0)
        # Unticked, the Source block's values do nothing.
        data, _eff, meta = self._Prep(
            self.legacy, run_overrides=own, override_enabled=False,
            cooler_override=29950.0, laser_override=16000.0)
        self.assertEqual(meta["cooler_v_effective"], 29990.0)
        self.assertEqual(data.Laser_set, 15000.0)

    def test_The_Run_s_Own_Outranks_Its_Split_s(self):
        child = os.path.join(self.tmp, "run_0001_split1.vasdf")
        split = {"parent_path": self.legacy, "split_lo": 0.0,
                 "split_hi": 400.0,
                 "metadata_override": {"cooler_v": 29970.0,
                                       "laser_sp": 19000.0}}
        cfg = dict(SOURCE_CFG,
                   run_overrides={child: {"cooler_v": 29990.0,
                                          "laser_sp": 0.0}})
        data, _eff, meta = prepare_run_data(child, cfg,
                                            split_descriptor=split)
        self.assertEqual(meta["cooler_v_effective"], 29990.0)
        # A zero is no override: the split's laser still applies.
        self.assertEqual(data.Laser_set, 19000.0)

    def test_The_Merge_Honours_The_Run_s_Own_Override(self):
        from gui.analysis.merge import compute_merged_spectrum
        params = {"domain": "frequency", "bin_step_mhz": None,
                  "cooler_correction": "pbp"}
        cfg = dict(SOURCE_CFG, override_enabled=False, cooler_override=0.0,
                   laser_override=0.0, cooler_offset_v=0.0)
        base = compute_merged_spectrum([self.legacy], cfg, params)
        own = dict(cfg, run_overrides={self.legacy: {"laser_sp": 20000.5}})
        moved = compute_merged_spectrum([self.legacy], own, params)
        self.assertEqual(base["per_run"][0]["laser_set"], LASER)
        self.assertEqual(moved["per_run"][0]["laser_set"], 20000.5)
        self.assertFalse(np.array_equal(base["x"], moved["x"]))

    def test_The_Merge_Reads_The_Recorded_Dividers_When_Asked(self):
        from gui.analysis.merge import compute_merged_spectrum
        params = {"domain": "frequency", "bin_step_mhz": None,
                  "cooler_correction": "pbp"}
        cfg = dict(SOURCE_CFG, cooler_offset_v=0.0)
        off = compute_merged_spectrum([self.cooler], cfg, params)
        on = compute_merged_spectrum(
            [self.cooler], dict(cfg, use_oracle_dividers=True), params)
        self.assertAlmostEqual(off["per_run"][0]["cooler_v"],
                               COOLER_KV * 10000.0, places=6)
        self.assertAlmostEqual(on["per_run"][0]["cooler_v"],
                               COOLER_KV * 10000.5, places=6)

    # ── the toggle list (C6) ──

    def test_Every_Toggle_Off_Is_The_Key_Absent(self):
        self.assertIn(("use_oracle_dividers", False), UPGRADE_TOGGLES)
        self.assertIn(("run_overrides", None), UPGRADE_TOGGLES)
        off = dict(UPGRADE_TOGGLES)
        for path in (self.legacy, self.cooler):
            d0, e0, m0 = self._Prep(path)
            d1, e1, m1 = self._Prep(path, **off)
            np.testing.assert_array_equal(d0.Sorted["V"].to_numpy(),
                                          d1.Sorted["V"].to_numpy())
            np.testing.assert_array_equal(d0.Sorted["F"].to_numpy(),
                                          d1.Sorted["F"].to_numpy())
            self.assertEqual(m0, m1)


if __name__ == "__main__":
    unittest.main()
