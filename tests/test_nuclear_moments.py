"""Nuclear moments by the ratio method (Analysis Upgrade, card C1; WP-M1).

mu = mu_ref (A I)/(A_ref I_ref) from each line's A and Q = Q_ref B/B_ref
from each B with J >= 1, their weighted mean, the A_u/A_l check against
the reference's, and the error budget (statistical, systematic, the
reference's, and the optional hyperfine-anomaly allowance of DA5).

Every number here is synthetic: no literature value is used (brief 8.6).

Run from the project root:
    .venv/bin/python -m pytest tests/test_nuclear_moments.py -q
"""
import unittest

import numpy as np

from cls_estimations.hfs_model import scale_A, scale_B
from cls_estimations.nuclear_moments import (
    A_Ratio_Check, Magnetic_Moment, Measured, Quadrupole_Moment, Reference)

# A synthetic reference isotope: I = 3/2, mu = 2.0(2) nm, Q = 0.5(1) b.
REF = Reference(I=1.5, mu=2.0, mu_sigma=0.02,
                A_lower=100.0, A_lower_sigma=1.0,
                A_upper=40.0, A_upper_sigma=0.2,
                Q=0.5, Q_sigma=0.005,
                B_lower=80.0, B_lower_sigma=0.8,
                B_upper=-30.0, B_upper_sigma=0.3)


class Moment_Tests(unittest.TestCase):

    def test_Mu_Inverts_Scale_A_On_Both_Lines(self):
        mu, spin = -1.3, 2.5
        a_l = scale_A(REF.A_lower, REF.mu, REF.I, mu, spin)
        a_u = scale_A(REF.A_upper, REF.mu, REF.I, mu, spin)
        out = Magnetic_Moment(Measured(a_l, 0.5), Measured(a_u, 0.2), spin,
                              REF)
        for key in ("lower", "upper", "mean"):
            self.assertAlmostEqual(out[key].value, mu, places=12)

    def test_Q_Inverts_Scale_B_On_Both_Lines(self):
        q = -0.8
        b_l = scale_B(REF.B_lower, REF.Q, q)
        b_u = scale_B(REF.B_upper, REF.Q, q)
        out = Quadrupole_Moment(Measured(b_l, 1.0), Measured(b_u, 0.5), 2.5,
                                2.0, 3.0, REF)
        for key in ("lower", "upper", "mean"):
            self.assertAlmostEqual(out[key].value, q, places=12)

    def test_One_Line_s_Budget_By_Hand(self):
        out = Magnetic_Moment(Measured(50.0, stat=0.5, syst=0.25), None,
                              2.5, REF, anomaly_pct=0.5)
        k = 2.0 * 2.5 / (100.0 * 1.5)               # mu_ref I / (A_ref I_ref)
        mu = out["lower"]
        self.assertAlmostEqual(mu.value, 50.0 * k, places=12)
        self.assertAlmostEqual(mu.stat, 0.5 * k, places=12)
        self.assertAlmostEqual(mu.syst, 0.25 * k, places=12)
        self.assertAlmostEqual(mu.ref, 50.0 * k * np.hypot(0.01, 0.01),
                               places=12)
        self.assertAlmostEqual(mu.anomaly, 50.0 * k * 0.005, places=12)
        self.assertAlmostEqual(mu.total, np.sqrt(
            mu.stat**2 + mu.syst**2 + mu.ref**2 + mu.anomaly**2), places=12)
        self.assertIsNone(out["upper"])
        self.assertEqual(out["mean"], mu)

    def test_The_Mean_Weighs_By_The_Statistical_Covariance(self):
        a_l, a_u = Measured(50.0, 0.4, 0.3), Measured(20.2, 0.1, 0.05)
        rho = 0.5
        out = Magnetic_Moment(a_l, a_u, 2.5, REF, rho=rho)
        lo, up = out["lower"], out["upper"]
        cov = np.array([[lo.stat**2, rho * lo.stat * up.stat],
                        [rho * lo.stat * up.stat, up.stat**2]])
        w = np.linalg.solve(cov, np.ones(2))
        w /= w.sum()
        mean = out["mean"]
        self.assertAlmostEqual(mean.value, w @ [lo.value, up.value],
                               places=12)
        self.assertAlmostEqual(mean.stat, np.sqrt(w @ cov @ w), places=12)
        # One source moves both lines: the systematic adds linearly.
        self.assertAlmostEqual(mean.syst, w @ [lo.syst, up.syst], places=12)
        # mu_ref's error is common; each line's A_ref error is its own.
        own = np.hypot(w[0] * lo.value * 0.01, w[1] * up.value * 0.005)
        self.assertAlmostEqual(mean.ref, np.hypot(mean.value * 0.01, own),
                               places=12)

    def test_Equal_Lines_Average(self):
        spin = 1.5
        out = Magnetic_Moment(Measured(100.0, 1.0), Measured(40.0, 0.4),
                              spin, REF)
        self.assertAlmostEqual(out["mean"].value, 2.0, places=12)
        self.assertAlmostEqual(out["mean"].stat, 0.02 / np.sqrt(2),
                               places=12)

    def test_A_Line_Without_A_Reference_Constant_Is_Left_Out(self):
        ref = Reference(I=1.5, mu=2.0, mu_sigma=0.0, A_lower=100.0,
                        A_lower_sigma=0.0, A_upper=0.0, A_upper_sigma=0.0)
        out = Magnetic_Moment(Measured(50.0, 0.5), Measured(20.0, 0.2), 1.5,
                              ref)
        self.assertIsNone(out["upper"])
        self.assertEqual(out["mean"], out["lower"])

    def test_Q_Needs_J_And_I_Of_At_Least_One(self):
        b = Measured(40.0, 1.0)
        out = Quadrupole_Moment(b, b, 2.5, 0.5, 1.0, REF)
        self.assertIsNone(out["lower"])
        self.assertEqual(out["mean"], out["upper"])
        self.assertIsNone(Quadrupole_Moment(b, b, 0.5, 2.0, 3.0, REF))
        self.assertIsNone(Quadrupole_Moment(b, b, 2.5, 0.5, 0.5, REF))
        no_q = Reference(I=1.5, mu=2.0, mu_sigma=0.0, A_lower=100.0,
                         A_lower_sigma=0.0, A_upper=40.0, A_upper_sigma=0.0)
        self.assertIsNone(Quadrupole_Moment(b, b, 2.5, 2.0, 3.0, no_q))

    def test_Mu_Needs_A_Spin(self):
        with self.assertRaises(ValueError):
            Magnetic_Moment(Measured(1.0, 0.1), None, 0.0, REF)

    def test_The_Ratio_Check_Flags_An_Anomaly(self):
        spin = 2.5
        a_l = scale_A(REF.A_lower, REF.mu, REF.I, 1.0, spin)
        a_u = scale_A(REF.A_upper, REF.mu, REF.I, 1.0, spin)
        ok = A_Ratio_Check(Measured(a_l, 0.05), Measured(a_u, 0.02), REF)
        self.assertAlmostEqual(ok["ratio"], ok["ratio_ref"], places=12)
        self.assertAlmostEqual(ok["pull"], 0.0, places=9)
        self.assertFalse(ok["flag"])
        off = A_Ratio_Check(Measured(a_l, 0.05), Measured(1.05 * a_u, 0.02),
                            REF)
        self.assertGreater(off["pull"], 3.0)
        self.assertTrue(off["flag"])

    def test_Pulls_Over_Synthetic_Sets(self):
        rng = np.random.default_rng(3)
        spin, mu, rho = 2.5, 1.7, 0.4
        true = np.array([scale_A(REF.A_lower, REF.mu, REF.I, mu, spin),
                         scale_A(REF.A_upper, REF.mu, REF.I, mu, spin)])
        sig = np.array([0.6, 0.15])
        cov = np.array([[sig[0]**2, rho * sig[0] * sig[1]],
                        [rho * sig[0] * sig[1], sig[1]**2]])
        pulls = []
        for a_l, a_u in rng.multivariate_normal(true, cov, 1000):
            m = Magnetic_Moment(Measured(a_l, sig[0]), Measured(a_u, sig[1]),
                                spin, REF, rho=rho)["mean"]
            pulls.append((m.value - mu) / m.stat)
        self.assertLess(abs(np.mean(pulls)), 0.1)
        self.assertLess(abs(np.std(pulls) - 1.0), 0.05)


if __name__ == "__main__":
    unittest.main()
