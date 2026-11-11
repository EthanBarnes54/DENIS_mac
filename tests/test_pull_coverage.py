"""The pull and coverage harness (Analysis Upgrade, card T2; WP-S2).

A parametric bootstrap of a fit: spectra drawn from the fitted model in
the run's bins (Poisson, or negative binomial at a measured variance/mean),
each refitted by an injected callback, and per parameter the pulls' mean
and width and the 68 % / 95 % coverage.

Known answers: the total counts "fitted" with Poisson errors pull with
width sqrt(D) for data of variance/mean D; a line fitted with errors
that are right covers 68 % and 95 %; with Poisson errors on data twice as
variable, the pulls widen by sqrt(2) and the coverage falls.

Run from the project root:
    .venv/bin/python -m pytest tests/test_pull_coverage.py -q
"""
import unittest

import numpy as np
from scipy.optimize import curve_fit

from cls_estimations.statistics import Pull_Coverage

X = np.linspace(-50.0, 50.0, 101)


def Line(x, bg, area, centre, width):
    return bg + area * np.exp(-0.5 * ((x - centre) / width)**2)


TRUTH = {"bg": 40.0, "area": 900.0, "centre": 3.0, "width": 6.0}
EXPECTED = Line(X, *TRUTH.values())


def Fit_Line(scale=1.0):
    """A refit with errors sqrt(scale * model): right when scale is the
    data's variance/mean."""
    def fit(y):
        p, cov = curve_fit(Line, X, y, p0=[30.0, 800.0, 0.0, 5.0],
                           sigma=np.sqrt(scale * np.maximum(EXPECTED, 1.0)),
                           absolute_sigma=True)
        return dict(zip(TRUTH, zip(p, np.sqrt(np.diag(cov)))))
    return fit


def Fit_Total(y):
    return {"total": (y.sum(), np.sqrt(EXPECTED.sum()))}


class Pull_Coverage_Tests(unittest.TestCase):

    def test_The_Total_Pulls_With_Width_Sqrt_D(self):
        for d in (1.0, 3.0):
            with self.subTest(dispersion=d):
                out = Pull_Coverage(EXPECTED, {"total": EXPECTED.sum()},
                                    Fit_Total, n=2000, dispersion=d, seed=1)
                p = out["params"]["total"]
                self.assertLess(abs(p["pull_mean"]), 3 * p["pull_mean_sigma"])
                self.assertLess(abs(p["pull_width"] - np.sqrt(d)),
                                3 * p["pull_width_sigma"])

    def test_Right_Errors_Cover_68_And_95_Percent(self):
        out = Pull_Coverage(EXPECTED, TRUTH, Fit_Line(), n=400, seed=2)
        self.assertEqual(out["failed"], 0)
        for name, p in out["params"].items():
            with self.subTest(parameter=name):
                self.assertEqual(p["n"], 400)
                self.assertLess(abs(p["pull_mean"]), 3 * p["pull_mean_sigma"])
                self.assertLess(abs(p["pull_width"] - 1.0),
                                3 * p["pull_width_sigma"])
                # binomial: 68 % of 400 to within 3 x 2.3 %, 95 % to 3 x 1.1 %
                self.assertLess(abs(p["coverage_68"] - 0.683), 0.07)
                self.assertLess(abs(p["coverage_95"] - 0.954), 0.035)

    def test_Poisson_Errors_On_Overdispersed_Data_Undercover(self):
        out = Pull_Coverage(EXPECTED, TRUTH, Fit_Line(), n=400,
                            dispersion=2.0, seed=3)
        p = out["params"]["centre"]
        self.assertLess(abs(p["pull_width"] - np.sqrt(2.0)),
                        3 * p["pull_width_sigma"])
        self.assertLess(p["coverage_68"], 0.6)
        # Errors scaled by the measured variance/mean set it right again.
        fixed = Pull_Coverage(EXPECTED, TRUTH, Fit_Line(scale=2.0), n=400,
                              dispersion=2.0, seed=3)["params"]["centre"]
        self.assertLess(abs(fixed["pull_width"] - 1.0),
                        3 * fixed["pull_width_sigma"])

    def test_A_Failed_Refit_Is_Counted_And_Left_Out(self):
        calls = []

        def Sometimes(y):
            calls.append(1)
            return None if len(calls) % 4 == 0 else Fit_Total(y)
        out = Pull_Coverage(EXPECTED, {"total": EXPECTED.sum()}, Sometimes,
                            n=100, seed=4)
        self.assertEqual(out["failed"], 25)
        self.assertEqual(out["params"]["total"]["n"], 75)

    def test_A_Seed_Repeats_The_Draws(self):
        a = Pull_Coverage(EXPECTED, {"total": EXPECTED.sum()}, Fit_Total,
                          n=50, dispersion=1.5, seed=5)
        b = Pull_Coverage(EXPECTED, {"total": EXPECTED.sum()}, Fit_Total,
                          n=50, dispersion=1.5, seed=5)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
