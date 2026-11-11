"""Overdispersion of a run's counts (Analysis Upgrade, card T1; WP-S1).

Variance/mean from the scan-to-scan repeats of each step, per PMT and for
all PMTs, with its uncertainty; the excess split, as ORACLE's WP7 does,
into a common part (the covariance between PMTs: the ions per bunch) and
each PMT's own (afterpulsing).

Synthetic counts with known answers: Poisson gives 1; afterpulsing with
probability p on one PMT gives (1 + 3p)/(1 + p) there and nothing common;
an ion number of relative variance v gives 1 + v lambda on every PMT,
all of it common.

Run from the project root:
    .venv/bin/python -m pytest tests/test_overdispersion.py -q
"""
import unittest

import numpy as np

from cls_estimations.statistics import Overdispersion

STEPS, SCANS, PMTS = 40, 30, (1, 2, 3, 4)


def Events(rng, rate=4.0, afterpulse=None, ion_var=0.0):
    """Per-event (step, scan, channel) labels: every step visited once a
    scan, each PMT seeing a Lorentzian-ish line on a background."""
    step, scan, chan = [], [], []
    lam = rate * (0.3 + 1.0 / (1.0 + ((np.arange(STEPS) - 20) / 4.0)**2))
    for k in range(SCANS):
        for s in range(STEPS):
            g = rng.gamma(1 / ion_var, ion_var) if ion_var else 1.0
            for c in PMTS:
                n = rng.poisson(g * lam[s])
                if afterpulse and c in afterpulse:
                    n += rng.binomial(n, afterpulse[c])
                step += [s] * n
                scan += [k] * n
                chan += [c] * n
    return np.array(step), np.array(scan), np.array(chan), lam


class Overdispersion_Tests(unittest.TestCase):

    def test_By_Hand(self):
        # Two steps, three scans, one PMT: counts (2, 4, 6) and (1, 1, 1).
        step = np.repeat([0, 0, 0, 1, 1, 1], [2, 4, 6, 1, 1, 1])
        scan = np.repeat([0, 1, 2, 0, 1, 2], [2, 4, 6, 1, 1, 1])
        out = Overdispersion(step, scan)
        # sum of squares 8 + 0 over (3 - 1)(4 + 1) = 10.
        self.assertAlmostEqual(out["all"]["dispersion"], 0.8, places=12)
        # Poisson null, delta method: (K - 1)(2 m^2 + m / K) per step.
        self.assertAlmostEqual(out["all"]["sigma"], np.sqrt(
            2 * (2 * 16 + 4 / 3) + 2 * (2 * 1 + 1 / 3)) / 10.0, places=12)
        self.assertEqual(out["all"]["dof"], 4)
        self.assertNotIn("pmt", out)

    def test_Poisson_Is_One_Everywhere(self):
        out = Overdispersion(*Events(np.random.default_rng(1))[:3])
        self.assertLess(abs(out["all"]["dispersion"] - 1.0),
                        3 * out["all"]["sigma"])
        for c in PMTS:
            p = out["pmt"][c]
            self.assertLess(abs(p["dispersion"] - 1.0), 3 * p["sigma"])
            self.assertLess(abs(p["afterpulse"]), 3 * p["sigma"])
        self.assertLess(abs(out["common_rel_var"]),
                        3 * out["common_rel_var_sigma"])

    def test_Afterpulsing_Is_One_PMT_s_Own(self):
        p = 0.5
        step, scan, chan, _lam = Events(np.random.default_rng(2),
                                        afterpulse={3: p})
        out = Overdispersion(step, scan, chan)
        three = out["pmt"][3]
        self.assertLess(abs(three["dispersion"] - (1 + 3 * p) / (1 + p)),
                        3 * three["sigma"])
        self.assertGreater(three["afterpulse"], 10 * three["sigma"])
        for c in (1, 2, 4):
            self.assertLess(abs(out["pmt"][c]["afterpulse"]),
                            3 * out["pmt"][c]["sigma"])
        self.assertLess(abs(out["common_rel_var"]),
                        3 * out["common_rel_var_sigma"])

    def test_The_Ion_Number_Is_Common_To_Every_PMT(self):
        v = 0.04
        step, scan, chan, lam = Events(np.random.default_rng(3), rate=20.0,
                                       ion_var=v)
        out = Overdispersion(step, scan, chan)
        self.assertLess(abs(out["common_rel_var"] - v),
                        3 * out["common_rel_var_sigma"])
        for c in PMTS:
            p = out["pmt"][c]
            self.assertGreater(p["common"], 5 * p["sigma"])
            self.assertLess(abs(p["afterpulse"]), 3 * p["sigma"])
        # Summed over PMTs the excess is the common part's, four-fold.
        self.assertGreater(out["all"]["dispersion"],
                           out["pmt"][1]["dispersion"])

    def test_Sigma_Is_Calibrated_At_Low_Counts(self):
        rng = np.random.default_rng(4)
        pulls = []
        for _ in range(300):
            out = Overdispersion(*Events(rng, rate=0.5)[:2])
            pulls.append((out["all"]["dispersion"] - 1.0)
                         / out["all"]["sigma"])
        self.assertLess(abs(np.mean(pulls)), 0.2)
        self.assertLess(abs(np.std(pulls) - 1.0), 0.1)

    def test_It_Needs_Two_Scans(self):
        with self.assertRaises(ValueError):
            Overdispersion(np.zeros(5, int), np.zeros(5, int))


if __name__ == "__main__":
    unittest.main()
