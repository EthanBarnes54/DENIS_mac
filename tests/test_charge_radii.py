"""Charge radii and the King plot (Analysis Upgrade, card C2; WP-M2).

dnu^{AA'} = K / mu_AA' + F dr2^{AA'}, mu_AA' = m_A m_A' / (m_A' - m_A),
with the masses of the species measured (DA4). Three ways to F and K:
given; a King-plot calibration on isotopes of known dr2 (York et al. 2004,
errors in both variables); or a second line's shifts. Isomer pairs have no
mass shift: dr2 = dnu / F, and they never calibrate the plot.

F, K, dr2 and the shifts here are synthetic, never literature values
(brief 8.6); the masses are the repo's own AME table.

Run from the project root:
    .venv/bin/python -m pytest tests/test_charge_radii.py -q
"""
import unittest

import numpy as np

from cls_estimations.mass_lookup import get_mass, load_mass_table
from cls_estimations.nuclear_moments import (
    ELECTRON_MASS_U, Inverse_Mass_Factor, King_Calibration,
    Radii_From_F_K, Shift_Covariance, Species_Mass, Two_Line_King,
    York_Fit)

TABLE = load_mass_table()
F_TRUE, K_TRUE = -400.0, 1.2e6          # MHz/fm^2, MHz u (synthetic)
DR2_TRUE = np.array([-0.30, -0.15, 0.12, 0.25, 0.41])   # fm^2 (synthetic)
A_PRIME = np.array([52, 53, 55, 56, 58])


def Inv_Mu(charge=1):
    m_ref = Species_Mass(26, 54, charge, TABLE)
    return np.array([Inverse_Mass_Factor(m_ref, Species_Mass(26, a, charge,
                                                             TABLE))
                     for a in A_PRIME])


class Mass_Tests(unittest.TestCase):

    def test_The_Species_Mass_Takes_Off_The_Electrons(self):
        atom = get_mass(TABLE, 26, 56)
        self.assertEqual(Species_Mass(26, 56, 0, TABLE), atom)
        self.assertAlmostEqual(Species_Mass(26, 56, 2, TABLE),
                               atom - 2 * ELECTRON_MASS_U, places=12)

    def test_The_Inverse_Mass_Factor(self):
        m, m2 = 53.9, 55.9
        self.assertAlmostEqual(Inverse_Mass_Factor(m, m2),
                               (m2 - m) / (m * m2), places=15)
        self.assertEqual(Inverse_Mass_Factor(m, m), 0.0)       # an isomer
        self.assertAlmostEqual(Inverse_Mass_Factor(m2, m),
                               -Inverse_Mass_Factor(m, m2), places=15)


class Radii_Tests(unittest.TestCase):

    def test_One_Pair_By_Hand(self):
        dnu, s_stat, s_syst = -250.0, 2.0, 1.0
        inv_mu, f, sf, k, sk, rho = 6.6e-4, -400.0, 8.0, 1.2e6, 2e4, -0.6
        out = Radii_From_F_K([dnu], [inv_mu], f, sf, k, sk, rho,
                             cov_stat=[[s_stat**2]], cov_syst=[[s_syst**2]])
        dr2 = (dnu - k * inv_mu) / f
        self.assertAlmostEqual(out["dr2"][0], dr2, places=12)
        self.assertAlmostEqual(out["stat"][0], s_stat / abs(f), places=12)
        self.assertAlmostEqual(out["syst"][0], s_syst / abs(f), places=12)
        j = np.array([-dr2 / f, -inv_mu / f])       # d/dF, d/dK
        c = np.array([[sf**2, rho * sf * sk], [rho * sf * sk, sk**2]])
        self.assertAlmostEqual(out["atomic"][0], np.sqrt(j @ c @ j),
                               places=12)
        self.assertAlmostEqual(out["total"][0], np.sqrt(
            out["stat"][0]**2 + out["syst"][0]**2 + out["atomic"][0]**2),
            places=12)

    def test_An_Isomer_Pair_Is_Its_Shift_Over_F(self):
        out = Radii_From_F_K([-80.0], [0.0], -400.0, 8.0, 1.2e6, 2e4, 0.5,
                             cov_stat=[[4.0]])
        self.assertEqual(out["dr2"][0], -80.0 / -400.0)
        self.assertAlmostEqual(out["atomic"][0], 0.2 * 8.0 / 400.0,
                               places=12)

    def test_Shifts_From_One_Reference_Are_Correlated(self):
        cov = Shift_Covariance([2.0, 3.0], ref_sigma=1.5)
        np.testing.assert_allclose(cov, [[4.0 + 2.25, 2.25],
                                         [2.25, 9.0 + 2.25]])
        out = Radii_From_F_K([-250.0, 300.0], Inv_Mu()[:2], -400.0, 0.0,
                             1.2e6, 0.0, cov_stat=cov)
        self.assertAlmostEqual(out["cov"][0, 1], 2.25 / 400.0**2, places=15)
        # Their difference no longer carries the reference's error.
        d = np.array([1.0, -1.0])
        self.assertAlmostEqual(np.sqrt(d @ out["cov"] @ d),
                               np.hypot(2.0, 3.0) / 400.0, places=12)


class King_Plot_Tests(unittest.TestCase):

    def _Synthetic(self, rng, n=5, s_nu=3.0, s_dr2=0.01):
        inv_mu = Inv_Mu()[:n]
        dr2 = DR2_TRUE[:n]
        dnu = K_TRUE * inv_mu + F_TRUE * dr2
        return (dnu + rng.normal(0, s_nu, n), np.full(n, s_nu),
                dr2 + rng.normal(0, s_dr2, n), np.full(n, s_dr2), inv_mu)

    def test_York_Recovers_An_Exact_Line(self):
        x = np.array([1.0, 2.0, 3.0, 5.0])
        fit = York_Fit(x, 4.0 - 0.5 * x, np.full(4, 0.1), np.full(4, 0.2))
        self.assertAlmostEqual(fit["slope"], -0.5, places=12)
        self.assertAlmostEqual(fit["intercept"], 4.0, places=12)
        self.assertAlmostEqual(fit["chi2"], 0.0, places=18)
        self.assertEqual(fit["dof"], 2)

    def test_York_With_Exact_X_Is_Weighted_Least_Squares(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0, 10, 8)
        sy = rng.uniform(0.5, 2.0, 8)
        y = 1.0 + 0.3 * x + rng.normal(0, sy)
        fit = York_Fit(x, y, np.full(8, 1e-12), sy)
        (b, a), cov = np.polyfit(x, y, 1, w=1 / sy, cov="unscaled")
        self.assertAlmostEqual(fit["slope"], b, places=9)
        self.assertAlmostEqual(fit["intercept"], a, places=9)
        self.assertAlmostEqual(fit["slope_sigma"], np.sqrt(cov[0, 0]),
                               places=9)
        self.assertAlmostEqual(fit["cov"], cov[0, 1], places=9)

    def test_The_Calibration_Recovers_F_And_K(self):
        fit = King_Calibration(*self._Synthetic(np.random.default_rng(2)))
        self.assertLess(abs(fit["F"] - F_TRUE), 3 * fit["F_sigma"])
        self.assertLess(abs(fit["K"] - K_TRUE), 3 * fit["K_sigma"])
        self.assertAlmostEqual(fit["rho_FK"], fit["cov_FK"] / (
            fit["F_sigma"] * fit["K_sigma"]), places=12)
        # Centred at the weighted mean of x, F and K' are uncorrelated.
        self.assertGreater(fit["K_centre_sigma"], 0.0)
        self.assertEqual(fit["dof"], 3)

    def test_Pulls_Over_Synthetic_Sets(self):
        # sigma_dr2 = 0.005 fm^2 keeps the linearised errors valid. At 0.01
        # (x errors dominating, ~8 % on mu dr2) York's ML line leans by a
        # second-order 0.12 sigma, which scipy.odr reproduces exactly: the
        # estimator's, not this code's (ledger, C2).
        rng = np.random.default_rng(4)
        pulls = []
        for _ in range(2000):
            fit = King_Calibration(*self._Synthetic(rng, s_dr2=0.005))
            pulls.append(((fit["F"] - F_TRUE) / fit["F_sigma"],
                          (fit["K"] - K_TRUE) / fit["K_sigma"]))
        pulls = np.array(pulls)
        np.testing.assert_allclose(pulls.mean(axis=0), 0.0, atol=0.05)
        np.testing.assert_allclose(pulls.std(axis=0), 1.0, atol=0.05)

    def test_Two_Points_Have_No_Degrees_Of_Freedom(self):
        dnu, s, dr2, s2, inv_mu = self._Synthetic(np.random.default_rng(5),
                                                  n=2)
        fit = King_Calibration(dnu, s, dr2, s2, inv_mu)
        self.assertEqual(fit["dof"], 0)
        self.assertTrue(np.isnan(fit["chi2_red"]))
        # The line runs through both points.
        y, x = dnu / inv_mu, dr2 / inv_mu
        np.testing.assert_allclose(fit["K"] + fit["F"] * x, y, rtol=1e-6)

    def test_Isomers_Never_Calibrate(self):
        dnu, s, dr2, s2, inv_mu = self._Synthetic(np.random.default_rng(6))
        inv_mu[2] = 0.0
        with self.assertRaises(ValueError):
            King_Calibration(dnu, s, dr2, s2, inv_mu)
        with self.assertRaises(ValueError):
            King_Calibration(dnu[:1], s[:1], dr2[:1], s2[:1], inv_mu[:1])

    def test_A_Second_Line_Gives_This_Line_s_F_And_K(self):
        rng = np.random.default_rng(7)
        inv_mu = Inv_Mu()
        f2, k2 = 250.0, -3.0e5                    # the second line (synthetic)
        dnu1 = K_TRUE * inv_mu + F_TRUE * DR2_TRUE
        dnu2 = k2 * inv_mu + f2 * DR2_TRUE
        exact = Two_Line_King(dnu1, np.full(5, 1e-6), dnu2,
                              np.full(5, 1e-6), inv_mu, f2, 0.0, k2, 0.0)
        self.assertAlmostEqual(exact["F_ratio"], F_TRUE / f2, places=6)
        self.assertAlmostEqual(exact["F"], F_TRUE, places=4)
        self.assertAlmostEqual(exact["K"] / K_TRUE, 1.0, places=6)
        noisy = Two_Line_King(dnu1 + rng.normal(0, 2.0, 5), np.full(5, 2.0),
                              dnu2 + rng.normal(0, 1.0, 5), np.full(5, 1.0),
                              inv_mu, f2, 5.0, k2, 1e4, rho2=-0.3)
        self.assertLess(abs(noisy["F"] - F_TRUE), 3 * noisy["F_sigma"])
        self.assertLess(abs(noisy["K"] - K_TRUE), 3 * noisy["K_sigma"])

    def test_Calibrate_Then_Apply(self):
        dnu, s, dr2, s2, inv_mu = self._Synthetic(np.random.default_rng(8))
        fit = King_Calibration(dnu[:3], s[:3], dr2[:3], s2[:3], inv_mu[:3])
        out = Radii_From_F_K(dnu[3:], inv_mu[3:], fit["F"], fit["F_sigma"],
                             fit["K"], fit["K_sigma"], fit["rho_FK"],
                             cov_stat=np.diag(s[3:]**2))
        self.assertTrue(np.all(np.abs(out["dr2"] - DR2_TRUE[3:])
                               < 3 * out["total"]))


if __name__ == "__main__":
    unittest.main()
