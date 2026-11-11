"""End-to-end truth on ORACLE-like simulated runs (Analysis Upgrade, VAL1).

Runs are drawn the way ORACLE's simulated rack draws them
(cls_acquisition/hardware/simulated.py, v0.6.0): three Lorentzian
resonances at 32, 47 and 71 platform volts (heights 1, 0.6, 0.35; FWHM
4 V) on a background, ions timed in their bunch, and the cooler meter read
about every 50 ms, unsynchronised with the bunches, so each reading
catches its mains tones at a random phase. The ions do not feel the
cooler, as there. The rates are scaled up so a short test has the counts.
Sawtooth and Triangle (odd rounds run backwards) scans, bunched.

* Raw Voltage: the three resonances come back within 2 sigma of their
  setpoints.
* Frequency: point by point, every event carries its reading's ripple;
  the rolling cooler takes it out, so the Gaussian width it convolves the
  line with is smaller and the centroid the same within errors. Compared
  per event: a stepped scan binned in frequency is a comb, and a line
  fitted to bins finer than its teeth depends on how they meet.
* Afterpulsing (VAL2), at ORACLE's own rates: a PMT that afterpulses
  lifts the run's variance/mean above 1, is named by the run quality, and
  the dead-time gate on it brings both back to 1.

Run from the project root:
    .venv/bin/python -m pytest tests/test_end_to_end_truth.py -q
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
from scipy.optimize import curve_fit

from cls_estimations.doppler import voltage_to_frequency_array
from gui.analysis.binning import compute_binned
from gui.analysis.pipeline import prepare_run_data

RESONANCES = ((32.0, 1.0), (47.0, 0.6), (71.0, 0.35))  # platform V, height
FWHM_V = 4.0
COOLER_V = 29884.8
COOLER_TONES = ((50.0, 0.438), (100.0, 0.014), (150.0, 0.038),
                (200.0, 0.023))                        # (Hz, V)
COOLER_NOISE_V = 0.17
TOF_US, TOF_WIDTH_US, GATE_US = 52.0, 1.5, 200.0
PERIOD_S = 0.005                     # 200 Hz, as ORACLE's tests time bunches
PEAK_PER_BUNCH, BACKGROUND_PER_BUNCH = 10.0, 0.5       # scaled up
# 0.05 V steps (0.65 MHz at the line), finer than the cooler's scatter:
# coarser steps put the events on a comb, which bins finer than its
# spacing cut into spikes.
SETPOINTS = np.arange(20.0, 85.001, 0.05)
MASS, LASER, HARMONIC = 100.0, 16956.17, 2


def Write_Oracle_Run(path, scan_type, rounds, bunches, seed, afterpulse=None,
                     per_bunch=(PEAK_PER_BUNCH, BACKGROUND_PER_BUNCH)):
    """One simulated run in ORACLE's format; returns its path.
    ``afterpulse``, {PMT: chance}: each count there may bring a second
    2-5 us later, as ORACLE's AFTERPULSE does. ``per_bunch``: the counts a
    bunch at the top of the strongest line and of background."""
    import asdf
    rng = np.random.default_rng(seed)
    order = [SETPOINTS[::-1] if scan_type == "Triangle" and k % 2
             else SETPOINTS for k in range(rounds)]
    steps = np.repeat(np.concatenate(order), bunches)   # one per bunch
    t0 = 1.8e9
    t_bunch = t0 + np.arange(len(steps)) * PERIOD_S
    t_read = t0 - 0.05 + np.cumsum(rng.uniform(
        0.04, 0.06, int(len(steps) * PERIOD_S / 0.05) + 3))
    readings = (COOLER_V + sum(a * np.sin(2 * np.pi * f * t_read)
                               for f, a in COOLER_TONES)
                + rng.normal(0.0, COOLER_NOISE_V, len(t_read)))
    shape = sum(h / (1.0 + ((steps - c) / (0.5 * FWHM_V))**2)
                for c, h in RESONANCES)
    n_ion = rng.poisson(per_bunch[0] * shape)
    n_bg = rng.poisson(per_bunch[1], len(steps))
    bunch = np.concatenate([np.repeat(np.arange(len(steps)), n_ion),
                            np.repeat(np.arange(len(steps)), n_bg)])
    tof = np.concatenate([rng.normal(TOF_US, TOF_WIDTH_US, n_ion.sum()),
                          rng.uniform(0.0, GATE_US, n_bg.sum())])
    pmt = rng.integers(1, 5, len(tof))
    for ch, chance in (afterpulse or {}).items():
        parents = (pmt == ch) & (rng.random(len(tof)) < chance)
        bunch, pmt = np.r_[bunch, bunch[parents]], np.r_[pmt, pmt[parents]]
        tof = np.r_[tof, tof[parents] + rng.uniform(2.0, 5.0, parents.sum())]
    ts = t_bunch[bunch] + tof * 1e-6
    cooler = readings[np.searchsorted(t_read, t_bunch[bunch], "right") - 1]
    order = np.argsort(ts, kind="stable")
    raw = np.column_stack([ts, steps[bunch], bunch, pmt, tof,
                           cooler / 10000.0])[order]
    cal = np.linspace(0.0, 100.0, 21)
    asdf.AsdfFile({
        "Run": 1, "Experiment": "simulated", "Date": "2026-11-11",
        "BunchesPerChannel": bunches, "StepSize": 0.05, "DwellTime": PERIOD_S,
        "MassAMU": MASS, "MassSetPoint": MASS,
        "CoolerVoltage": readings[0] / 10000.0, "LaserSetpoint": LASER,
        "ScanningRanges": [[20.0, 85.0]], "CalSet": cal,
        "CalReadback": cal / 1000.0, "DacCorrectionApplied": False,
        "DacCorrection": "", "raw": raw,
        "ORACLE": {"ScanType": scan_type, "Rounds": rounds,
                   "Dividers": {"Scan": {"ratio": 1000.0, "ppm": 100.0},
                                "Cooler": {"ratio": 10000.0, "ppm": 100.0}},
                   "CoolerRollingS": 15.0}}).write_to(path)
    return path


def Lorentzians(x, bg, *p):
    return bg + sum(a / (1.0 + ((x - c) / (0.5 * w))**2)
                    for a, c, w in zip(p[0::3], p[1::3], p[2::3]))



class End_To_End_Truth_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="e2e_truth_")
        cls.runs = {kind: Write_Oracle_Run(
            os.path.join(cls.tmp, f"run_{i + 1:04d}.asdf"), kind, rounds=3,
            bunches=4, seed=10 + i)
            for i, kind in enumerate(("Sawtooth", "Triangle"))}
        cls.cfg = dict(
            mass=MASS, harmonic=HARMONIC, cal_order=1, pmt_gate=[1, 2, 3, 4],
            tof_gate=None, v_gate=None, f_gate=None, noise_filter=0,
            x_column="bins_center", yerr_mode="Poisson sqrt(y+1)",
            xerr_mode="None", ref_shift=0.0, cooler_correction="pbp",
            ref_freq=float(voltage_to_frequency_array(
                [-(COOLER_V - 32.0)], 0.0, LASER, MASS, HARMONIC)[0]) * 1e6)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _Spectrum(self, path, **extra):
        data, eff, _meta = prepare_run_data(path, dict(self.cfg, **extra))
        res = compute_binned(data, eff)
        return np.asarray(res["x"], float), np.asarray(res["y"], float)

    def test_Raw_Voltage_Recovers_The_Three_Resonances(self):
        for kind, path in self.runs.items():
            with self.subTest(scan=kind):
                x, y = self._Spectrum(path, bin_mode="Raw Voltage")
                p0 = [y.min()]
                for (c, h), start in zip(RESONANCES, (31.0, 48.0, 70.0)):
                    p0 += [h * y.max(), start, 3.0]
                p, cov = curve_fit(Lorentzians, x, y, p0=p0,
                                   sigma=np.sqrt(y + 1), absolute_sigma=True)
                for i, (c, _h) in enumerate(RESONANCES):
                    centre, err = p[2 + 3 * i], np.sqrt(cov[2 + 3 * i,
                                                            2 + 3 * i])
                    self.assertLess(abs(centre - c), 2 * err,
                                    f"{kind}: {centre:.3f} +/- {err:.3f} V "
                                    f"against {c} V")

    def test_Rolling_Is_No_Wider_Than_Point_By_Point(self):
        """Each ion's frequency less its step's at the cooler's own level
        is the cooler correction's contribution: the Gaussian the line is
        convolved with. Compared before binning, since a binned stepped
        scan is a comb whose fits depend on how its bins meet the steps."""
        from pandas import Series
        out = {}
        for corr in ("pbp", "rolling"):
            data, _eff, _meta = prepare_run_data(
                self.runs["Sawtooth"],
                dict(self.cfg, cooler_correction=corr, bin_mode="Frequency"))
            s = data.Sorted
            ions = ((np.abs(s["DV"] - 32.0) < 2.0)
                    & (np.abs(s["TOF"] - TOF_US) < 5.0)).to_numpy()
            nominal = voltage_to_frequency_array(
                -(COOLER_V - s["DV"].to_numpy()[ions]), 0.0, LASER, MASS,
                HARMONIC) - self.cfg["ref_freq"] / 1e6
            d = s["F"].to_numpy()[ions] / 1e6 - nominal
            # The centroid's error is the readings', not the events'.
            readings = Series(s["Vrfq"].to_numpy()[ions]).nunique()
            out[corr] = (d.std(), d.mean(), readings)
        (w_p, c_p, n_p), (w_r, c_r, _n) = out["pbp"], out["rolling"]
        # 0.35 V of ripple and noise point by point: ~4.5 MHz here.
        self.assertGreater(w_p, 3.0)
        self.assertLess(w_r, w_p / 2.0)
        self.assertLess(abs(c_r - c_p), 2 * np.sqrt(2) * w_p / np.sqrt(n_p))

    def test_Dead_Time_Gate_Takes_Out_Afterpulsing(self):
        """ORACLE's rates (60 Hz at the top and 5 Hz of background, at
        200 Hz bunches), PMT 3 afterpulsing at its test's 0.5; the gate at
        5 us, the longest delay. At the raised rates above the gate would
        also take most real photons at the peaks, where a PMT sees several
        a bunch, and so leave them below Poisson."""
        from gui.analysis.fitting import _Run_Quality
        for i, kind in enumerate(("Sawtooth", "Triangle")):
            path = Write_Oracle_Run(
                os.path.join(self.tmp, f"run_{i + 11:04d}.asdf"), kind,
                rounds=4, bunches=25, seed=40 + i, afterpulse={3: 0.5},
                per_bunch=(0.3, 0.025))
            q = {}
            for name, gate in (("off", None), ("gated", {3: 5.0})):
                data, eff, meta = prepare_run_data(path, dict(
                    self.cfg, bin_mode="Raw Voltage", dead_time_us=gate))
                q[name] = _Run_Quality(data, path, eff, meta)
            with self.subTest(scan=kind):
                off, gated = q["off"], q["gated"]
                self.assertGreater(off["dispersion"] - 1, 3 * off["sigma"])
                self.assertEqual(off["afterpulsing"], [3])
                self.assertLess(abs(gated["dispersion"] - 1),
                                3 * gated["sigma"])
                self.assertEqual(gated["afterpulsing"], [])
                own = gated["pmt"][3]
                self.assertLess(abs(own["dispersion"] - 1), 3 * own["sigma"])
                # What it takes is the afterpulses, 0.5 in 1.5 of PMT 3's.
                self.assertLess(abs(meta["dead_time"][3]["removed"] - 1 / 3),
                                0.03)


if __name__ == "__main__":
    unittest.main()
