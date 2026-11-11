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

Depends on: numpy.
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
