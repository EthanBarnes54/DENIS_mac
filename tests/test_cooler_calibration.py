"""Cooler-voltage offset calibration from Yb hyperfine constants.

Every expected value here is worked out by hand from lines built for
the test: a run's line ``(A - A_lit) = a + b dV`` reproduces the
literature A at ``dV = -a/b``, with an error propagated from those of a
and b; an isotope's offset is the inverse-variance mean over its runs;
and the calibration is where the two isotope lines cross. The numbers
are synthetic, chosen so the arithmetic is easy to follow.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_cooler_calibration.py -q
"""
import math
import unittest

import numpy as np

from cls_estimations.cooler_calibration import (
    LITERATURE_A, LineFit, ScanPoint, calibrate, fit_line, intersect,
    weighted_mean,
)

#: Synthetic per-run lines: run, isotope, b (MHz/V), sigma_b,
#: a - A_lit (MHz), sigma_a. They reproduce the literature A at
#: -30, -32 and -28 V (171Yb) and at -10 and -15 V (173Yb).
RUN_LINES = [
    ("1001", "171Yb", -0.5, 0.05, -15.0, 1.5),
    ("1002", "171Yb", -0.5, 0.02, -16.0, 0.5),
    ("1003", "171Yb", -0.4, 0.02, -11.2, 0.4),
    ("1004", "173Yb", +0.1, 0.002, +1.0, 0.03),
    ("1005", "173Yb", +0.1, 0.01, +1.5, 0.3),
]

#: Two synthetic isotope lines, ``(A - A_lit) = slope * (dV - zero)``.
ZERO_171, SLOPE_171 = -30.0, -0.3
ZERO_173, SLOPE_173 = -10.0, +0.1
#: Where they cross: b1 (x - z1) = b2 (x - z2) at
#: x = (b1 z1 - b2 z2) / (b1 - b2) = (9 + 1) / -0.4 = -25 V.
CROSSING = ((SLOPE_171 * ZERO_171 - SLOPE_173 * ZERO_173)
            / (SLOPE_171 - SLOPE_173))


def _line(b, sb, a, sa):
    """A LineFit with independent intercept and slope errors (a
    diagonal covariance, as for an offset grid symmetric about zero)."""
    return LineFit(intercept=a, slope=b, cov=np.diag([sa ** 2, sb ** 2]),
                   n=15, chi2=0.0, ndf=13)


def _zero_by_hand(b, sb, a, sa):
    """``(-a/b, sigma)`` with sigma^2 = (sa/b)^2 + (a sb/b^2)^2: the
    intercept term and the slope term."""
    return -a / b, math.hypot(sa / b, a * sb / b ** 2)


def _mean_by_hand(values, sigmas):
    """Inverse-variance mean and its error."""
    w = [1.0 / s ** 2 for s in sigmas]
    return (sum(wi * v for wi, v in zip(w, values)) / sum(w),
            1.0 / math.sqrt(sum(w)))


def _points_on(line_zero, slope, *, run, isotope, a_lit, dvs=None,
               sigma=0.05):
    """Scan points lying exactly on ``(A - A_lit) = slope (dV - zero)``."""
    dvs = np.linspace(-50, 50, 15) if dvs is None else np.asarray(dvs)
    return [ScanPoint(run=run, isotope=isotope, dv=float(dv),
                      value=a_lit + slope * (dv - line_zero),
                      sigma=sigma) for dv in dvs]


class PerRunZeroCrossingTests(unittest.TestCase):
    """Per run: where the line crosses the literature value."""

    def test_every_run_crosses_where_the_algebra_says(self):
        for run, _iso, b, sb, a, sa in RUN_LINES:
            want, want_sig = _zero_by_hand(b, sb, a, sa)
            with self.subTest(run=run):
                dv, sig = _line(b, sb, a, sa).root()
                self.assertAlmostEqual(dv, want, places=9)
                self.assertAlmostEqual(sig, want_sig, places=9)

    def test_the_error_is_not_just_the_intercept_term(self):
        """Both the intercept and the slope are uncertain, and for
        run 1001 they contribute equally (3 V each). Taking only the
        first would quote 3.0 instead of 3 sqrt(2) = 4.24 V."""
        _run, _iso, b, sb, a, sa = RUN_LINES[0]
        _dv, sig = _line(b, sb, a, sa).root()
        self.assertAlmostEqual(sig, 3.0 * math.sqrt(2.0), places=9)
        self.assertGreater(sig, abs(sa / b) * 1.3)

    def test_a_flat_line_has_no_solution(self):
        """A parameter that does not respond to the cooler voltage
        cannot calibrate it."""
        self.assertIsNone(
            LineFit(0.5, 0.0, np.eye(2), 5, 0.0, 3).root())


class PerIsotopeMeanTests(unittest.TestCase):

    def test_the_per_isotope_weighted_means(self):
        for iso in ("171Yb", "173Yb"):
            rows = [r[2:] for r in RUN_LINES if r[1] == iso]
            want, want_sig = _mean_by_hand(
                *zip(*[_zero_by_hand(*row) for row in rows]))
            dv, sig = zip(*[_line(*row).root() for row in rows])
            m, s = weighted_mean(dv, sig)
            with self.subTest(isotope=iso):
                self.assertAlmostEqual(m, want, places=9)
                self.assertAlmostEqual(s, want_sig, places=9)

    def test_the_precise_run_dominates(self):
        """1004 (sigma 0.36 V) against 1005 (3.35 V): the mean must sit
        next to 1004, not midway."""
        (z4, s4), (z5, s5) = [_zero_by_hand(*r[2:]) for r in RUN_LINES[3:]]
        m, _s = weighted_mean([z4, z5], [s4, s5])
        self.assertLess(abs(m - z4), abs(m - z5))


class IntersectionTests(unittest.TestCase):

    def test_two_lines_cross_where_the_algebra_says(self):
        """Lines through -30 V (slope -0.3) and -10 V (slope +0.1)
        meet at (b1 z1 - b2 z2) / (b1 - b2) = -25 V."""
        a = fit_line([-50, 50], [SLOPE_171 * (-50 - ZERO_171),
                                 SLOPE_171 * (50 - ZERO_171)])
        b = fit_line([-50, 50], [SLOPE_173 * (-50 - ZERO_173),
                                 SLOPE_173 * (50 - ZERO_173)])
        dv, _sig = intersect(a, b)
        self.assertAlmostEqual(dv, CROSSING, places=9)
        self.assertAlmostEqual(dv, -25.0, places=9)

    def test_parallel_lines_never_cross(self):
        a = fit_line([0, 10], [0.0, 1.0])
        b = fit_line([0, 10], [5.0, 6.0])
        self.assertIsNone(intersect(a, b))

    def test_nearly_parallel_lines_are_badly_determined(self):
        """The crossing of two similar slopes is a long lever; its
        error must blow up rather than look precise."""
        pts_a = _points_on(-30.0, -0.20, run="a", isotope="X",
                           a_lit=0.0, sigma=0.5)
        steep = fit_line([p.dv for p in pts_a], [p.value for p in pts_a],
                         [p.sigma for p in pts_a])
        close = fit_line([p.dv for p in pts_a],
                         [-0.19 * (p.dv + 12.0) for p in pts_a],
                         [p.sigma for p in pts_a])
        far = fit_line([p.dv for p in pts_a],
                       [0.20 * (p.dv + 12.0) for p in pts_a],
                       [p.sigma for p in pts_a])
        self.assertGreater(intersect(steep, close)[1],
                           intersect(steep, far)[1] * 5)


class LineFitTests(unittest.TestCase):

    def test_it_recovers_a_known_line(self):
        line = fit_line([-50, -25, 0, 25, 50],
                        [-6.0, -3.5, -1.0, 1.5, 4.0])
        self.assertAlmostEqual(line.slope, 0.1, places=9)
        self.assertAlmostEqual(line.intercept, -1.0, places=9)

    def test_a_loose_point_does_not_drag_the_fit(self):
        x = [-50, -25, 0, 25, 50]
        y = [-6.0, -3.5, -1.0, 1.5, 40.0]           # last one is wild
        tight = fit_line(x, y, [0.1, 0.1, 0.1, 0.1, 100.0])
        self.assertAlmostEqual(tight.slope, 0.1, delta=0.005)

    def test_a_point_without_an_error_is_left_out(self):
        """A fit pinned at a bound has no usable error; unit weight made
        its pull depend on the unit (review 3, 2026-10-08). It is left
        out of the line -- not quietly: the calibration flags it, draws
        it and lists it (ScanPoint.flag, CalibrationResult.excluded)."""
        line = fit_line([-50, 0, 50, 25], [-6.0, -1.0, 4.0, 900.0],
                        [0.1, 0.1, 0.1, 0.0])
        self.assertEqual(line.n, 3)
        self.assertAlmostEqual(line.slope, 0.1, delta=1e-6)

    def test_without_two_errors_the_line_is_unweighted(self):
        line = fit_line([-50, 0, 50], [-6.0, -1.0, 4.0],
                        [0.0, 0.1, float("nan")])
        self.assertEqual(line.n, 3)
        self.assertAlmostEqual(line.slope, 0.1, delta=1e-6)

    def test_identical_offsets_cannot_make_a_line(self):
        with self.assertRaises(ValueError):
            fit_line([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])

    def test_the_band_is_narrowest_in_the_middle_of_the_data(self):
        pts = _points_on(-30.0, -0.2, run="r", isotope="X", a_lit=0.0)
        line = fit_line([p.dv for p in pts], [p.value for p in pts],
                        [p.sigma for p in pts])
        self.assertLess(float(line.sigma_at(0.0)),
                        float(line.sigma_at(200.0)))


class CalibrateTests(unittest.TestCase):

    def _points(self):
        a171, a173 = LITERATURE_A["171Yb"], LITERATURE_A["173Yb"]
        pts = []
        for run in ("1001", "1002", "1003"):
            pts += _points_on(ZERO_171, SLOPE_171, run=run,
                              isotope="171Yb", a_lit=a171)
        for run in ("1004", "1005"):
            pts += _points_on(ZERO_173, SLOPE_173, run=run,
                              isotope="173Yb", a_lit=a173)
        return pts

    def test_end_to_end(self):
        res = calibrate(self._points())
        self.assertEqual([s.isotope for s in res.isotopes],
                         ["171Yb", "173Yb"])
        self.assertAlmostEqual(res.dv, CROSSING, delta=0.05)
        # The systematic scan runs between the two zero crossings.
        self.assertAlmostEqual(res.scan_lo, ZERO_171, delta=0.01)
        self.assertAlmostEqual(res.scan_hi, ZERO_173, delta=0.01)
        self.assertAlmostEqual(res.scan_width, ZERO_173 - ZERO_171,
                               delta=0.02)
        # ...and the calibration sits inside it.
        self.assertLess(res.scan_lo, res.dv)
        self.assertLess(res.dv, res.scan_hi)

    def test_every_run_is_solved_too(self):
        res = calibrate(self._points())
        by_iso = {s.isotope: s for s in res.isotopes}
        self.assertEqual(len(by_iso["171Yb"].runs), 3)
        for r in by_iso["171Yb"].runs:
            self.assertAlmostEqual(r.dv, ZERO_171, delta=0.01)
        self.assertAlmostEqual(by_iso["171Yb"].mean_dv, ZERO_171,
                               delta=0.01)

    def test_the_sign_of_the_173_constant_matters(self):
        """A(173Yb+) is negative. Handing over its magnitude flips the
        slope of that isotope's line and moves the crossing."""
        good = calibrate(self._points())
        bad = calibrate(self._points(),
                        literature={"171Yb": LITERATURE_A["171Yb"],
                                    "173Yb": -LITERATURE_A["173Yb"]})
        self.assertNotAlmostEqual(good.dv, bad.dv, delta=1.0)

    def test_one_isotope_alone_cannot_calibrate(self):
        pts = [p for p in self._points() if p.isotope == "171Yb"]
        res = calibrate(pts)
        self.assertIsNone(res.dv)
        self.assertIn("two isotopes", res.note)
        # ...but its own zero crossing is still reported.
        self.assertAlmostEqual(res.isotopes[0].zero, ZERO_171, delta=0.01)

    def test_a_run_with_a_single_offset_is_skipped(self):
        pts = self._points()
        pts.append(ScanPoint("1999", "171Yb", 0.0, 1.0, 0.1))
        res = calibrate(pts)
        runs = [r.run for s in res.isotopes for r in s.runs]
        self.assertNotIn("1999", runs)

    def test_no_points_is_not_a_crash(self):
        res = calibrate([])
        self.assertIsNone(res.dv)
        self.assertEqual(res.isotopes, [])

    def test_literature_defaults_match_the_paper(self):
        """de Groote, Atoms 2024, 12, 60, Table 2."""
        self.assertAlmostEqual(LITERATURE_A["171Yb"],
                               12642.8121184682, places=9)
        self.assertAlmostEqual(LITERATURE_A["173Yb"],
                               -3497.24007985, places=7)
        self.assertLess(LITERATURE_A["173Yb"], 0)


if __name__ == "__main__":
    unittest.main()
