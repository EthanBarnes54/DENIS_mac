"""Nuclear moments from hyperfine constants, by the ratio method.

Date:    2026-11-11
Version: 1.0.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The inverse of ``hfs_model.scale_A`` / ``scale_B``: against a reference
isotope of the same element and line,

    mu = mu_ref (A I) / (A_ref I_ref)        from each line's A,
    Q  = Q_ref B / B_ref                     from each B with J >= 1,

and the two lines' weighted mean. The hyperfine anomaly is neglected, as
the ratio method assumes; an optional allowance (%) enters mu's budget as
its own column (DA5). A_u/A_l against the reference's is the check on that
assumption, and on the spin and the line assignment.

Every result carries its budget: statistical and systematic (the latter
from the Systematics tab, taken as fully correlated between the lines,
since one source moves both), the reference's own uncertainty (the common
mu_ref or Q_ref, and each line's own A_ref or B_ref), and the anomaly.

Charge radii from isotope shifts (W. H. King, *Isotope Shifts in Atomic
Spectra*, 1984), with dnu^{AA'} = nu^{A'} - nu^{A}:

    dnu^{AA'} = K / mu_AA' + F dr2^{AA'},   mu_AA' = m_A m_A' / (m_A' - m_A),

the masses being those of the species measured (DA4), so an isomer pair
has no mass shift and dr2 = dnu / F. F and K are given, calibrated on
isotopes of known dr2 (a straight line in the modified shifts mu dnu
against mu dr2, fitted with errors in both by York et al. 2004, Am. J.
Phys. 72, 367), or taken from a second line's shifts. Units are the
caller's: MHz, fm^2, MHz/fm^2 and MHz u, say.

Depends on: numpy, cls_estimations.mass_lookup.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Measured:
    """A fitted constant with its statistical and systematic uncertainty."""
    value: float
    stat: float = 0.0
    syst: float = 0.0


@dataclass(frozen=True)
class Moment:
    """A moment and its budget; ``total`` adds them in quadrature."""
    value: float
    stat: float
    syst: float
    ref: float
    anomaly: float = 0.0

    @property
    def total(self) -> float:
        return float(np.sqrt(self.stat**2 + self.syst**2 + self.ref**2
                             + self.anomaly**2))


@dataclass(frozen=True)
class Reference:
    """The reference isotope (brief C5); a constant of 0 is unknown and
    leaves its line out."""
    I: float
    mu: float
    mu_sigma: float
    A_lower: float
    A_lower_sigma: float
    A_upper: float
    A_upper_sigma: float
    Q: float = 0.0
    Q_sigma: float = 0.0
    B_lower: float = 0.0
    B_lower_sigma: float = 0.0
    B_upper: float = 0.0
    B_upper_sigma: float = 0.0


def _Lines(measured, x_ref, x_ref_sigma, scale, scale_rel, rho, anomaly_rel):
    """``{lower, upper, mean}``: each line's ``scale * x / x_ref`` and their
    mean, weighted by the statistical covariance (correlation ``rho``)."""
    out, used = {}, []
    for name, m, r, rs in zip(("lower", "upper"), measured, x_ref,
                              x_ref_sigma):
        if m is None or not r:
            out[name] = None
            continue
        k = scale / r
        v = k * m.value
        out[name] = Moment(v, abs(k) * m.stat, abs(k) * m.syst,
                           abs(v) * np.hypot(rs / r, scale_rel),
                           abs(v) * anomaly_rel)
        used.append((out[name], rs / r))
    if len(used) < 2:
        out["mean"] = used[0][0] if used else None
        return out
    (a, a_rel), (b, b_rel) = used
    cov = rho * a.stat * b.stat
    den = a.stat**2 + b.stat**2 - 2 * cov
    if not den > 0:
        raise ValueError("The lines need statistical uncertainties to be "
                         "weighted.")
    wa = (b.stat**2 - cov) / den
    wb = 1.0 - wa
    v = wa * a.value + wb * b.value
    own = np.hypot(wa * a.value * a_rel, wb * b.value * b_rel)
    out["mean"] = Moment(
        v, float(np.sqrt(wa**2 * a.stat**2 + wb**2 * b.stat**2
                         + 2 * wa * wb * cov)),
        abs(wa * a.syst + wb * b.syst),
        float(np.hypot(abs(v) * scale_rel, own)), abs(v) * anomaly_rel)
    return out


def Magnetic_Moment(A_lower: Optional[Measured], A_upper: Optional[Measured],
                    I: float, ref: Reference, rho: float = 0.0,
                    anomaly_pct: float = 0.0) -> dict:
    """mu from each line's A (None for a line not measured) and their mean,
    as ``{lower, upper, mean}`` Moments; ``rho`` is the fit's correlation
    between A_l and A_u."""
    if not I > 0:
        raise ValueError(f"A magnetic moment needs a spin I > 0, not {I}.")
    return _Lines((A_lower, A_upper), (ref.A_lower, ref.A_upper),
                  (ref.A_lower_sigma, ref.A_upper_sigma),
                  ref.mu * I / ref.I, ref.mu_sigma / abs(ref.mu), rho,
                  anomaly_pct / 100.0)


def Quadrupole_Moment(B_lower: Optional[Measured],
                      B_upper: Optional[Measured], I: float, J_lower: float,
                      J_upper: float, ref: Reference,
                      rho: float = 0.0) -> Optional[dict]:
    """Q from each B whose level has J >= 1, and their mean; None when no
    line gives one (I or the reference's I below 1, or no reference Q)."""
    if I < 1 or ref.I < 1 or not ref.Q:
        return None
    B = (B_lower if J_lower >= 1 else None,
         B_upper if J_upper >= 1 else None)
    out = _Lines(B, (ref.B_lower, ref.B_upper),
                 (ref.B_lower_sigma, ref.B_upper_sigma), ref.Q,
                 ref.Q_sigma / abs(ref.Q), rho, 0.0)
    return out if out["mean"] is not None else None


def A_Ratio_Check(A_lower: Measured, A_upper: Measured, ref: Reference,
                  rho: float = 0.0, limit: float = 3.0) -> dict:
    """A_u/A_l against the reference's, which it equals without an anomaly;
    its pull in combined sigma, flagged beyond ``limit``. The systematic is
    taken as moving both lines together, so only its difference counts."""
    r = A_upper.value / A_lower.value
    rel_l, rel_u = A_lower.stat / A_lower.value, A_upper.stat / A_upper.value
    stat = abs(r) * np.sqrt(max(rel_l**2 + rel_u**2
                                - 2 * rho * rel_l * rel_u, 0.0))
    syst = abs(r) * abs(A_upper.syst / A_upper.value
                        - A_lower.syst / A_lower.value)
    r_ref = ref.A_upper / ref.A_lower
    sigma_ref = abs(r_ref) * np.hypot(ref.A_upper_sigma / ref.A_upper,
                                      ref.A_lower_sigma / ref.A_lower)
    sigma = float(np.sqrt(stat**2 + syst**2 + sigma_ref**2))
    pull = (r - r_ref) / sigma if sigma > 0 else float("nan")
    return {"ratio": r, "sigma": float(np.hypot(stat, syst)),
            "ratio_ref": r_ref, "sigma_ref": float(sigma_ref),
            "pull": pull, "flag": bool(abs(pull) > limit)}


# ── Charge radii and the King plot ──────────────────────────────────

ELECTRON_MASS_U = 5.48579909065e-4       # u, CODATA 2018


def Species_Mass(Z: int, A: int, charge: int = 0, table=None) -> float:
    """The mass (u) of the species measured: the AME atomic mass less
    ``charge`` electrons (a neutral atom after the CEC keeps them all)."""
    from cls_estimations.mass_lookup import get_mass, load_mass_table
    return get_mass(table or load_mass_table(), Z, A) - charge * ELECTRON_MASS_U


def Inverse_Mass_Factor(m_A: float, m_A2: float) -> float:
    """1/mu_AA' = (m_A' - m_A) / (m_A m_A'): zero for an isomer pair."""
    return (m_A2 - m_A) / (m_A * m_A2)


def Shift_Covariance(own_sigmas, ref_sigma: float = 0.0) -> np.ndarray:
    """The covariance of shifts all taken against one reference isotope:
    each its own variance, and the reference's variance shared by all."""
    own = np.asarray(own_sigmas, dtype=float)
    return np.diag(own**2) + ref_sigma**2


def Radii_From_F_K(shifts, inv_mu, F: float, F_sigma: float, K: float,
                   K_sigma: float, rho_FK: float = 0.0, cov_stat=None,
                   cov_syst=None) -> dict:
    """dr2 = (dnu - K / mu) / F for each pair, with ``cov_stat`` and
    ``cov_syst`` the shifts' covariances (None: zero) and F, K correlated
    by ``rho_FK``. Returns each pair's dr2 and its stat / syst / atomic /
    total sigma, and the full covariance ``cov`` between the pairs."""
    nu = np.asarray(shifts, dtype=float)
    im = np.asarray(inv_mu, dtype=float)
    zero = np.zeros((len(nu), len(nu)))
    c_stat = zero if cov_stat is None else np.asarray(cov_stat, float)
    c_syst = zero if cov_syst is None else np.asarray(cov_syst, float)
    dr2 = (nu - K * im) / F
    jac = np.column_stack((-dr2 / F, -im / F))          # d/dF, d/dK
    c_fk = np.array([[F_sigma**2, rho_FK * F_sigma * K_sigma],
                     [rho_FK * F_sigma * K_sigma, K_sigma**2]])
    c_atomic = jac @ c_fk @ jac.T
    cov = (c_stat + c_syst) / F**2 + c_atomic
    return {"dr2": dr2, "stat": np.sqrt(np.diag(c_stat)) / abs(F),
            "syst": np.sqrt(np.diag(c_syst)) / abs(F),
            "atomic": np.sqrt(np.diag(c_atomic)),
            "total": np.sqrt(np.diag(cov)), "cov": cov}


def York_Fit(x, y, sx, sy, r=None, tol: float = 1e-15,
             max_iter: int = 200) -> dict:
    """The straight line y = intercept + slope x through points with errors
    in both (sx, sy; their correlations r), by York et al. 2004. Also gives
    the line's centre (the weighted mean of the fitted x), where its value
    is uncorrelated with the slope, and chi^2 with n - 2 degrees of
    freedom (chi2_red is NaN with none)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    wx, wy = 1.0 / np.asarray(sx, float)**2, 1.0 / np.asarray(sy, float)**2
    r = np.zeros_like(x) if r is None else np.asarray(r, float)
    alpha = np.sqrt(wx * wy)
    b = np.polyfit(x, y, 1)[0]
    for _ in range(max_iter):
        w = wx * wy / (wx + b * b * wy - 2 * b * r * alpha)
        xb, yb = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
        u, v = x - xb, y - yb
        beta = w * (u / wy + b * v / wx - (b * u + v) * r / alpha)
        b_new = np.sum(w * beta * v) / np.sum(w * beta * u)
        done = abs(b_new - b) <= tol * abs(b_new)
        b = b_new
        if done:
            break
    a = yb - b * xb
    x_fit = xb + beta
    centre = np.sum(w * x_fit) / np.sum(w)
    sb2 = 1.0 / np.sum(w * (x_fit - centre)**2)
    chi2 = float(np.sum(w * (y - b * x - a)**2))
    dof = len(x) - 2
    return {"slope": float(b), "intercept": float(a),
            "slope_sigma": float(np.sqrt(sb2)),
            "intercept_sigma": float(np.sqrt(1.0 / np.sum(w)
                                             + centre**2 * sb2)),
            "cov": float(-centre * sb2), "x_centre": float(centre),
            "y_centre": float(a + b * centre),
            "y_centre_sigma": float(np.sqrt(1.0 / np.sum(w))),
            "chi2": chi2, "dof": dof,
            "chi2_red": chi2 / dof if dof > 0 else float("nan")}


def _Modified(inv_mu, y, sy, x, sx):
    """The King plot's axes: each value and sigma times mu_AA'. Isomer
    pairs (no mass shift) are refused."""
    im = np.asarray(inv_mu, dtype=float)
    if np.any(im == 0):
        raise ValueError("An isomer pair has no mass shift: it is placed "
                         "with F, never used to calibrate the King plot.")
    if len(im) < 2:
        raise ValueError("A King plot needs at least two isotope pairs.")
    y, sy, x, sx = (np.asarray(v, dtype=float) for v in (y, sy, x, sx))
    return y / im, np.abs(sy / im), x / im, np.abs(sx / im)


def _F_K(F, F_sigma, K, K_sigma, cov, fit):
    """The F, K record the calibrations return."""
    rho = cov / (F_sigma * K_sigma) if F_sigma > 0 and K_sigma > 0 else 0.0
    return {"F": F, "F_sigma": F_sigma, "K": K, "K_sigma": K_sigma,
            "cov_FK": cov, "rho_FK": rho, "chi2": fit["chi2"],
            "dof": fit["dof"], "chi2_red": fit["chi2_red"]}


def King_Calibration(shifts, shift_sigma, dr2, dr2_sigma, inv_mu) -> dict:
    """F and K from pairs of known dr2 (with their sigma): the modified
    shifts mu dnu against mu dr2, fitted with errors in both. Also the
    line's centre ``x_centre`` and ``K_centre`` (its value there), which is
    uncorrelated with F. Exactly two pairs leave no degrees of freedom.

    York's is the maximum-likelihood line, and like any errors-in-variables
    fit it leans by a second-order amount when the dr2 errors dominate
    (F sigma_dr2 well above sigma_dnu): about 0.1 sigma on F at 8 %
    relative error on mu dr2, vanishing as they shrink."""
    y, sy, x, sx = _Modified(inv_mu, shifts, shift_sigma, dr2, dr2_sigma)
    fit = York_Fit(x, y, sx, sy)
    out = _F_K(fit["slope"], fit["slope_sigma"], fit["intercept"],
               fit["intercept_sigma"], fit["cov"], fit)
    out.update(x_centre=fit["x_centre"], K_centre=fit["y_centre"],
               K_centre_sigma=fit["y_centre_sigma"])
    return out


def Two_Line_King(shifts, shift_sigma, shifts2, shifts2_sigma, inv_mu,
                  F2: float, F2_sigma: float, K2: float, K2_sigma: float,
                  rho2: float = 0.0) -> dict:
    """This line's F and K from a second line's shifts for the same pairs
    and that line's F2, K2: mu dnu = (F/F2) mu dnu2 + (K - (F/F2) K2)."""
    y, sy, x, sx = _Modified(inv_mu, shifts, shift_sigma, shifts2,
                             shifts2_sigma)
    fit = York_Fit(x, y, sx, sy)
    s, c = fit["slope"], fit["intercept"]
    jac = np.array([[0.0, F2, s, 0.0], [1.0, K2, 0.0, s]])  # d/d(c,s,F2,K2)
    cin = np.zeros((4, 4))
    cin[:2, :2] = [[fit["intercept_sigma"]**2, fit["cov"]],
                   [fit["cov"], fit["slope_sigma"]**2]]
    cin[2:, 2:] = [[F2_sigma**2, rho2 * F2_sigma * K2_sigma],
                   [rho2 * F2_sigma * K2_sigma, K2_sigma**2]]
    cov = jac @ cin @ jac.T
    out = _F_K(s * F2, float(np.sqrt(cov[0, 0])), c + s * K2,
               float(np.sqrt(cov[1, 1])), float(cov[0, 1]), fit)
    out.update(F_ratio=s, F_ratio_sigma=fit["slope_sigma"], K_offset=c,
               K_offset_sigma=fit["intercept_sigma"])
    return out
