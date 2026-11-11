"""The Moments & Radii tab, part 1 (Analysis Upgrade, card H1; WP-M3).

A permanent tab in the Analysis tab's own tab bar, beside Isotope Shifts
and Systematics (never a main-window tab: Ethan, 2026-11-11), saved with
the session. Its reference inputs (pre-filled from the Estimate tab's
reference isotope) and its moments table: mu and Q per project and HFS
state from the projects' fitted A and B, with their budget.

Every physics number here is synthetic.

Run from the project root:
    .venv/bin/python -m pytest tests/test_moments_radii_tab.py -q
"""
import types
import unittest
from unittest import mock

import numpy as np

from cls_estimations.hfs_model import scale_A, scale_B
from cls_estimations.nuclear_moments import (
    Magnetic_Moment, Measured, Quadrupole_Moment, Reference)
from tests.test_import_only_checked import _ensure_app

REF = {"I": 1.5, "mu": 2.0, "mu_sigma": 0.02, "Q": 0.5, "Q_sigma": 0.005,
       "A_lower": 100.0, "A_lower_sigma": 1.0, "A_upper": 40.0,
       "A_upper_sigma": 0.2, "B_lower": 80.0, "B_lower_sigma": 0.8,
       "B_upper": -30.0, "B_upper_sigma": 0.3}
SPIN, MU, Q = 2.5, -1.3, -0.8
A_L = scale_A(REF["A_lower"], REF["mu"], REF["I"], MU, SPIN)
A_U = scale_A(REF["A_upper"], REF["mu"], REF["I"], MU, SPIN)
B_L = scale_B(REF["B_lower"], REF["Q"], Q)
B_U = scale_B(REF["B_upper"], REF["Q"], Q)


def Params_Df(source, a_l, b_l, a_u=A_U, b_u=B_U, model="HFS_1"):
    rows = [("Al", a_l, 0.5), ("Au", a_u, 0.2), ("Bl", b_l, 2.0),
            ("Bu", b_u, 1.0), ("centroid", 10.0, 0.1)]
    return {"Source": [source] * len(rows), "Model": [model] * len(rows),
            "Parameter": [r[0] for r in rows],
            "Value": [r[1] for r in rows], "Stderr": [r[2] for r in rows]}


def Model_Block(au_free=True):
    def P(value, vary=False, expr=""):
        return {"value": value, "vary": vary, "expr": expr}
    params = {"I": P(SPIN), "Jl": P(2.0), "Ju": P(3.0), "Al": P(0, True),
              "Au": P(0, au_free, "" if au_free else "Al * 0.4"),
              "Bl": P(0, True), "Bu": P(0, True)}
    return types.SimpleNamespace(get_model_config=lambda: {
        "type": "HFS", "name": "HFS_1", "params": params})


def Project(name, results, au_free=True):
    from gui.analysis.blocks import ModelBlock
    blocks = {ModelBlock: [Model_Block(au_free)]}
    return types.SimpleNamespace(
        project_name=name, is_calibration=False, _last_results=results,
        load_results_from_disk=lambda: None,
        _get_blocks_by_type=lambda cls: blocks.get(cls, []))


class Moments_Radii_Tab_Tests(unittest.TestCase):

    def setUp(self):
        _ensure_app()

    def _Tab(self, projects=(), sys_rows=()):
        from gui.analysis.moments_radii_tab import MomentsRadiiTab
        analysis = types.SimpleNamespace(
            _projects=list(projects),
            _sys_tab=types.SimpleNamespace(summary_rows=lambda: list(sys_rows)))
        tab = MomentsRadiiTab(analysis_tab=analysis)
        self.addCleanup(tab.deleteLater)
        tab.from_dict({"reference": REF, "citation": "synthetic"})
        return tab

    def test_It_Sits_In_The_Analysis_Tab_s_Own_Tab_Bar(self):
        from gui.analysis.tab import AnalysisTab
        at = AnalysisTab()
        self.addCleanup(at.deleteLater)
        bar = at._project_tabs
        titles = lambda: [bar.tabText(i) for i in range(bar.count())]
        self.assertNotIn("Moments & Radii", titles())
        at._add_project("Fe56")
        self.assertEqual(titles()[-2:], ["Systematics", "Moments & Radii"])
        self.assertIs(bar.widget(bar.count() - 1), at._mr_tab)
        at._add_project("Fe54")
        self.assertEqual(titles()[-1], "Moments & Radii")
        from PySide6.QtWidgets import QMessageBox
        with mock.patch.object(QMessageBox, "question",
                               return_value=QMessageBox.StandardButton.Yes):
            for _ in range(2):
                at._close_project(0)
        self.assertNotIn("Moments & Radii", titles())

    def test_A_Project_Never_Fitted_Is_Skipped(self):
        """A project with an HFS model but no fits yet (its results None)
        is listed without moments instead of stopping the computation."""
        tab = self._Tab([Project("53mFe", None)])
        tab.compute()
        (row,) = tab.Moments()
        self.assertEqual((row["project"], row["mu"], row["Q"]),
                         ("53mFe", None, None))

    def test_The_Moments_Of_A_Project_s_Fits(self):
        results = [
            {"success": True, "run_number": "1", "source_name": "Run_1",
             "params_df": Params_Df("Run_1", A_L + 0.3, B_L - 1.0,
                                    A_U + 0.1, B_U + 0.5)},
            {"success": True, "run_number": "2", "source_name": "Run_2",
             "params_df": Params_Df("Run_2", A_L - 0.3, B_L + 1.0,
                                    A_U - 0.1, B_U - 0.5)},
            {"success": False, "run_number": "3", "source_name": "Run_3",
             "params_df": Params_Df("Run_3", 999.0, 999.0)}]
        sys_rows = [{"project": "Fe57", "run": "<weighted mean>",
                     "parameter": "Al", "sys_total": 0.4}]
        tab = self._Tab([Project("Fe57", results)], sys_rows)
        tab.compute()
        self.assertEqual(tab._Table.rowCount(), 1)
        ref = Reference(**REF)
        a_l = Measured(A_L, 0.5 / np.sqrt(2), 0.4)
        a_u = Measured(A_U, 0.2 / np.sqrt(2), 0.0)
        mu = Magnetic_Moment(a_l, a_u, SPIN, ref)["mean"]
        q = Quadrupole_Moment(Measured(B_L, 2.0 / np.sqrt(2)),
                              Measured(B_U, 1.0 / np.sqrt(2)), SPIN, 2.0,
                              3.0, ref)["mean"]
        row = tab.Moments()[0]
        self.assertEqual((row["project"], row["state"]), ("Fe57", "HFS_1"))
        for got, want in ((row["mu"], mu), (row["Q"], q)):
            for f in ("value", "stat", "syst", "ref"):
                self.assertAlmostEqual(getattr(got, f), getattr(want, f),
                                       places=9)
        self.assertAlmostEqual(row["mu"].value, MU, places=6)
        self.assertFalse(row["check"]["flag"])
        self.assertIn(f"{MU:.6g}", [tab._Table.item(0, c).text()
                                    for c in range(tab._Table.columnCount())])

    def test_A_Shared_Constant_Counts_Once_And_A_Tied_A_u_Is_No_Line(self):
        # A simultaneous fit: both runs carry one params_df, A shared.
        shared = Params_Df("Run_1", A_L, B_L)
        for k in shared:
            shared[k] = shared[k] + [("Run_2" if k == "Source" else v)
                                     for v in shared[k]]
        results = [{"success": True, "run_number": n, "source_name": f"Run_{n}",
                    "params_df": shared} for n in ("1", "2")]
        tab = self._Tab([Project("Fe57", results, au_free=False)])
        tab.compute()
        row = tab.Moments()[0]
        self.assertAlmostEqual(row["mu"].stat, abs(MU / A_L) * 0.5, places=9)
        self.assertIsNone(row["check"])

    def test_The_Anomaly_Allowance_Is_Per_State_And_Saved(self):
        results = [{"success": True, "run_number": "1", "source_name": "Run_1",
                    "params_df": Params_Df("Run_1", A_L, B_L)}]
        tab = self._Tab([Project("Fe57", results)])
        tab.from_dict(dict(tab.to_dict(), anomaly_pct={"Fe57/HFS_1": 0.5}))
        tab.compute()
        self.assertAlmostEqual(tab.Moments()[0]["mu"].anomaly,
                               abs(MU) * 0.005, places=6)
        saved = tab.to_dict()
        self.assertEqual(saved["anomaly_pct"], {"Fe57/HFS_1": 0.5})
        self.assertEqual(saved["reference"], REF)
        self.assertEqual(saved["citation"], "synthetic")

    def test_The_Reference_Comes_From_The_Estimate_Tab(self):
        tab = self._Tab()
        estimate = {"reference": {"A": 57, "I": 0.5, "mu": 0.09, "Q": 0.0,
                                  "A_lower_MHz": 12.5, "B_lower_MHz": 0.0,
                                  "A_upper_MHz": 7.25, "B_upper_MHz": 0.0}}
        window = types.SimpleNamespace(_build_estimate_dict=lambda: estimate)
        with mock.patch.object(tab, "window", return_value=window):
            tab._From_Estimate()
        ref = tab.to_dict()["reference"]
        self.assertEqual((ref["I"], ref["mu"], ref["A_lower"],
                          ref["A_upper"]), (0.5, 0.09, 12.5, 7.25))
        self.assertEqual(ref["mu_sigma"], REF["mu_sigma"])  # sigma kept

    def test_An_Older_Session_Has_No_Moments(self):
        tab = self._Tab()
        tab.from_dict({})
        self.assertEqual(tab.to_dict()["reference"]["mu"], 0.0)
        tab.compute()
        self.assertEqual(tab._Table.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
