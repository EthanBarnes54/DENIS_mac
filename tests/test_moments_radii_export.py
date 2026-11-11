"""The Moments & Radii tab, part 2 (Analysis Upgrade, card H2; WP-M3).

The charge radii from the Isotope Shifts tab's shifts (F and K given,
from a King-plot calibration, or from a second line), with the shifts'
systematic from the Systematics tab; isomer pairs as dnu/F; the King plot
(drawn by one function for the tab and the Results tab); and the export
of moments, radii and the plot to Results, every row naming its inputs.

F, K, dr2 and the shifts are synthetic; the masses are the repo's AME table.

Run from the project root:
    .venv/bin/python -m pytest tests/test_moments_radii_export.py -q
"""
import csv
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from cls_estimations.mass_lookup import load_mass_table
from cls_estimations.nuclear_moments import (
    Inverse_Mass_Factor, King_Calibration, Radii_From_F_K, Species_Mass)
from tests.test_import_only_checked import _ensure_app

F, K = -400.0, 1.2e6                            # synthetic
DR2 = {52: -0.30, 53: -0.15, 55: 0.12, 56: 0.25}
TABLE = load_mass_table()


def Inv_Mu(a, charge=1):
    return Inverse_Mass_Factor(Species_Mass(26, 54, charge, TABLE),
                               Species_Mass(26, a, charge, TABLE))


def Shift_Rows(extra=()):
    """Isotope Shifts rows against 54Fe, on the King line exactly."""
    rows = [{"label": "54Fe", "A": 54, "delta_nu": 0.0, "is_reference": True,
             "sigma_fit": 0.6, "sigma_scatter": 0.0, "sigma_correction": 0.0,
             "sigma_voltage": 0.2}]
    for a, dr2 in sorted(DR2.items()):
        rows.append({"label": f"{a}Fe", "A": a, "is_reference": False,
                     "delta_nu": K * Inv_Mu(a) + F * dr2,
                     "sigma_delta_nu_stat": 1.0, "sigma_delta_nu_scatter": 0.0,
                     "sigma_delta_nu_correction": 0.0,
                     "sigma_delta_nu_sys": 0.5})
    return rows + list(extra)


class Radii_Tab_Tests(unittest.TestCase):

    def setUp(self):
        _ensure_app()

    def _Tab(self, rows=None, sys_rows=()):
        from gui.analysis.moments_radii_tab import MomentsRadiiTab
        source = types.SimpleNamespace(get_source_config=lambda: {"Z": 26})
        from gui.analysis.blocks import SourceBlock
        project = types.SimpleNamespace(
            project_name="Fe56", is_calibration=False, _last_results=[],
            load_results_from_disk=lambda: None,
            _get_blocks_by_type=lambda cls: [source] if cls is SourceBlock
            else [])
        analysis = types.SimpleNamespace(
            _projects=[project],
            _is_tab=types.SimpleNamespace(
                shifts_for_scan=lambda: list(rows or Shift_Rows())),
            _sys_tab=types.SimpleNamespace(
                summary_rows=lambda: list(sys_rows)))
        tab = MomentsRadiiTab(analysis_tab=analysis)
        self.addCleanup(tab.deleteLater)
        return tab

    def _Given(self, tab, sf=4.0, sk=1e4, rho=-0.5):
        tab.from_dict({"radii": {"mode": "given", "F": F, "F_sigma": sf,
                                 "K": K, "K_sigma": sk, "rho": rho,
                                 "charge": 1, "citation": "synthetic"}})

    def test_Given_F_And_K_Reproduce_The_Library(self):
        tab = self._Tab(sys_rows=[{"project": "Isotope shifts", "run": "56Fe",
                                   "parameter": "delta_nu",
                                   "sys_total": 3.0}])
        self._Given(tab)
        tab.compute()
        radii = {r["pair"]: r for r in tab.Radii()}
        self.assertEqual(set(radii), {"52Fe", "53Fe", "55Fe", "56Fe"})
        for a, dr2 in DR2.items():
            self.assertAlmostEqual(radii[f"{a}Fe"]["dr2"], dr2, places=9)
        # The library, with the shifts' covariance built as the tab does.
        rows = [r for r in Shift_Rows() if not r["is_reference"]]
        ref_var = 0.6**2
        stat = np.full((4, 4), ref_var)
        np.fill_diagonal(stat, 1.0)
        syst = np.full((4, 4), 0.2**2)
        np.fill_diagonal(syst, 0.5**2)
        scan = np.array([0, 0, 0, 3.0])              # 56Fe's, one source
        syst = syst + np.outer(scan, scan)
        want = Radii_From_F_K([r["delta_nu"] for r in rows],
                              [Inv_Mu(r["A"]) for r in rows], F, 4.0, K,
                              1e4, -0.5, cov_stat=stat, cov_syst=syst)
        for i, r in enumerate(rows):
            got = radii[r["label"]]
            for key in ("stat", "syst", "atomic", "total"):
                self.assertAlmostEqual(got[key], want[key][i], places=12)

    def test_An_Isomer_Pair_Is_Its_Shift_Over_F(self):
        iso = {"label": "53mFe", "A": 53, "is_reference": False,
               "delta_nu": Shift_Rows()[2]["delta_nu"] + 80.0,
               "sigma_delta_nu_stat": 1.2, "sigma_delta_nu_scatter": 0.0,
               "sigma_delta_nu_correction": 0.0, "sigma_delta_nu_sys": 0.5}
        tab = self._Tab(Shift_Rows([iso]))
        self._Given(tab)
        tab.compute()
        radii = {r["pair"]: r for r in tab.Radii()}
        pair = radii["53mFe − 53Fe"]
        self.assertTrue(pair["isomer"])
        self.assertAlmostEqual(pair["dr2"], 80.0 / F, places=9)
        # The reference's error cancels: 1.2 (+) 1.0 less 2 x 0.6^2.
        self.assertAlmostEqual(pair["stat"], np.sqrt(1.2**2 + 1.0 - 2 * 0.36)
                               / abs(F), places=12)

    def test_A_King_Calibration_On_Two_Pairs(self):
        tab = self._Tab()
        tab.from_dict({"radii": {"mode": "king", "charge": 1, "points": [
            {"A": 52, "value": DR2[52], "sigma": 0.01, "citation": "syn"},
            {"A": 56, "value": DR2[56], "sigma": 0.01, "citation": "syn"}]}})
        tab.compute()
        fit = tab._King
        self.assertAlmostEqual(fit["F"], F, places=6)
        self.assertEqual(fit["dof"], 0)
        radii = {r["pair"]: r for r in tab.Radii()}
        self.assertAlmostEqual(radii["55Fe"]["dr2"], DR2[55], places=6)

    def test_A_Second_Line_Gives_F_And_K(self):
        f2, k2 = 250.0, -3.0e5
        tab = self._Tab()
        tab.from_dict({"radii": {
            "mode": "two_line", "charge": 1, "F": f2, "F_sigma": 0.0,
            "K": k2, "K_sigma": 0.0, "rho": 0.0, "points": [
                {"A": a, "value": k2 * Inv_Mu(a) + f2 * dr2, "sigma": 1e-6,
                 "citation": "syn"} for a, dr2 in DR2.items()]}})
        tab.compute()
        self.assertAlmostEqual(tab._King["F"], F, places=3)
        self.assertAlmostEqual(tab._King["K"] / K, 1.0, places=6)

    def test_The_Radii_Inputs_Are_Saved(self):
        tab = self._Tab()
        tab.from_dict({"radii": {"mode": "king", "charge": 0, "points": [
            {"A": 52, "value": -0.3, "sigma": 0.01, "citation": "syn"}]}})
        saved = tab.to_dict()["radii"]
        self.assertEqual(saved["mode"], "king")
        self.assertEqual(saved["charge"], 0)
        self.assertEqual(saved["points"], [
            {"A": 52, "value": -0.3, "sigma": 0.01, "citation": "syn"}])

    def test_The_King_Plot_Draws_For_The_Tab_And_Results(self):
        from gui.analysis.moments_radii_tab import Draw_King_Plot
        tab = self._Tab()
        self._Given(tab)
        tab.compute()
        payload = tab.King_Plot_Data()
        fig = Figure()
        Draw_King_Plot(fig, payload)
        ax = fig.axes[0]
        self.assertIn("μ", ax.get_xlabel())
        self.assertEqual(len(payload["measured"]["x"]), 4)
        # Results renders the saved payload by its plot type.
        from gui.results_tab import ResultsTab
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "king_plot.npz")
        np.savez(path, plot_type="king_plot",
                 payload=np.array(payload, dtype=object))
        results = ResultsTab()
        self.addCleanup(results.deleteLater)
        results._render_from_npz(path)
        self.assertIn("μ", results._plot_figure.axes[0].get_xlabel())

    def test_Send_To_Results_Writes_The_Iteration(self):
        tab = self._Tab()
        self._Given(tab)
        tab.from_dict(dict(tab.to_dict(), citation="synthetic reference"))
        tab.compute()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        sent = []
        tab.results_ready.connect(lambda *a: sent.append(a))
        with mock.patch("gui.shared_widgets.get_analysis_dir",
                        return_value=tmp):
            tab._Send_To_Results()
        idir = os.path.join(tmp, "Moments_Radii", "iter_001")
        for name in ("radii.csv", "moments.csv", "fit_report.txt",
                     os.path.join("plots", "king_plot.png"),
                     os.path.join("plots", "king_plot.npz")):
            self.assertTrue(os.path.isfile(os.path.join(idir, name)), name)
        with open(os.path.join(idir, "radii.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 4)
        self.assertIn("F, K given", rows[0]["inputs"])
        self.assertIn("synthetic", rows[0]["inputs"])
        self.assertEqual(sent[0][0], "Moments_Radii")


if __name__ == "__main__":
    unittest.main()
