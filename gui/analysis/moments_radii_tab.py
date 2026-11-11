"""Moments & Radii: nuclear moments and charge radii as an analysis technique.

Date:    2026-11-11
Version: 1.0.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A permanent tab in the Analysis tab's own tab bar, beside Isotope Shifts
and Systematics and built the way they are; never a main-window tab. It
takes every isotope project's fitted hyperfine constants, with the
Systematics tab's budget for them, and gives the nuclear moments by the
ratio method against a reference isotope (cls_estimations.nuclear_moments).
The charge radii, the King plot and the export to Results follow (H2).

Depends on: PySide6, numpy, cls_estimations (isotope_shift,
nuclear_moments), gui.analysis (blocks, helpers, naming).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from cls_estimations.isotope_shift import weighted_average
from cls_estimations.nuclear_moments import (
    A_Ratio_Check, Magnetic_Moment, Measured, Quadrupole_Moment, Reference)
from gui.analysis.helpers import _make_double
from gui.analysis.naming import safe_model_name

#: The reference isotope's inputs; all but I have a sigma beside them. The
#: Estimate tab names A and B with an "_MHz" suffix.
REF_INPUTS = {"I": "I", "mu": "μ [μN]", "Q": "Q [b]",
              "A_lower": "A lower [MHz]", "A_upper": "A upper [MHz]",
              "B_lower": "B lower [MHz]", "B_upper": "B upper [MHz]"}
MU_FIELDS = ("value", "stat", "syst", "ref", "anomaly", "total")
Q_FIELDS = ("value", "stat", "syst", "ref", "total")
COLUMNS = (("Project", "State", "I", "μ [μN]") + MU_FIELDS[1:]
           + ("Anomaly [%]", "Q [b]") + Q_FIELDS[1:] + ("A_u/A_l",))
ANOMALY_COL = 3 + len(MU_FIELDS)
MEAN_RUN = "<weighted mean>"          # a Systematics summary row's mean


class MomentsRadiiTab(QGroupBox):
    """The reference inputs and the moments table (H2 adds the radii)."""

    results_ready = Signal(str, list, dict)

    def __init__(self, analysis_tab, parent=None):
        super().__init__("Nuclear moments and charge radii", parent)
        self._analysis_tab = analysis_tab
        self._anomaly = {}          # "project/state" -> allowance, %
        self._moments = []
        root = QVBoxLayout(self)

        ref_box = QGroupBox("Reference isotope")
        form = QFormLayout(ref_box)
        self._Ref = {}
        for key, label in REF_INPUTS.items():
            self._Ref[key] = _make_double(
                0.0, -1e9, 1e9, 6, 0.1,
                tooltip=f"The reference isotope's {label}.")
            if key == "I":
                form.addRow(label + ":", self._Ref[key])
                continue
            self._Ref[key + "_sigma"] = _make_double(
                0.0, 0.0, 1e9, 6, 0.1, tooltip="Its standard uncertainty.")
            row = QHBoxLayout()
            row.addWidget(self._Ref[key], 1)
            row.addWidget(QLabel("±"))
            row.addWidget(self._Ref[key + "_sigma"], 1)
            form.addRow(label + ":", row)
        self._Citation = QLineEdit(toolTip="Where these values come from.")
        form.addRow("Citation:", self._Citation)
        fill = QPushButton("From Estimate", toolTip=(
            "Take I, μ, Q, A and B from the Estimate tab's reference "
            "isotope;\nthe uncertainties stay as typed."))
        fill.clicked.connect(self._From_Estimate)
        form.addRow("", fill)
        root.addWidget(ref_box)

        mom_box = QGroupBox("Moments")
        col = QVBoxLayout(mom_box)
        top = QHBoxLayout()
        self._Status = QLabel("")
        go = QPushButton("Compute", toolTip=(
            "μ and Q of every HFS state in the isotope projects, from "
            "their fits\nand the Systematics tab's budget for A and B."))
        go.clicked.connect(self.compute)
        top.addWidget(self._Status, 1)
        top.addWidget(go)
        col.addLayout(top)
        self._Table = QTableWidget(0, len(COLUMNS))
        self._Table.setHorizontalHeaderLabels(COLUMNS)
        self._Table.verticalHeader().setVisible(False)
        self._Table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self._Table.horizontalHeader().setStretchLastSection(True)
        self._Table.setToolTip(
            "stat: the fits' errors; syst: the Systematics tab's budget;\n"
            "ref: the reference isotope's; anomaly: the allowance typed in\n"
            "Anomaly [%]. A_u/A_l: its pull from the reference's ratio.")
        self._Table.itemChanged.connect(self._On_Anomaly_Edited)
        col.addWidget(self._Table)
        root.addWidget(mom_box, 1)

    # ── inputs ───────────────────────────────────────────────
    def _Reference(self):
        return Reference(**{k: w.value() for k, w in self._Ref.items()})

    def _From_Estimate(self):
        ref = self.window()._build_estimate_dict().get("reference")
        if not ref:
            self._Status.setText("The Estimate tab has no reference isotope.")
            return
        for key in REF_INPUTS:
            name = key if key in ("I", "mu", "Q") else key + "_MHz"
            self._Ref[key].setValue(float(ref.get(name) or 0.0))

    # ── the projects' fits ───────────────────────────────────
    def _States(self):
        """(project, model config) for each HFS state of each isotope
        project."""
        from gui.analysis.blocks import ModelBlock
        for project in self._analysis_tab._projects:
            if project.is_calibration:
                continue
            if not project._last_results:
                try:            # a reopened session keeps its fits on disk
                    project.load_results_from_disk()
                except Exception:  # noqa: BLE001
                    pass
            for block in project._get_blocks_by_type(ModelBlock):
                config = block.get_model_config()
                if config["type"] == "HFS":
                    yield project, config

    @staticmethod
    def _Fitted(project, config, name, syst):
        """This state's fitted ``name`` over the project's successful fits,
        weighted; a constant a simultaneous fit shares counts once. None
        when it is not a free parameter or was never fitted."""
        par = config["params"].get(name, {})
        if not par.get("vary") or par.get("expr"):
            return None
        models = {config["name"], safe_model_name(config["name"])}
        found = {}
        for result in project._last_results:
            if not result.get("success"):
                continue
            df = result.get("params_df") or {}
            for src, model, p, v, e in zip(
                    df.get("Source", []), df.get("Model", []),
                    df.get("Parameter", []), df.get("Value", []),
                    df.get("Stderr", df.get("Error", []))):
                if (p == name and model in models and v is not None
                        and e and np.isfinite(e) and e > 0):
                    found[(src, model)] = (float(v), float(e))
        if not found:
            return None
        values, errors = zip(*sorted(set(found.values())))
        mean, stat = weighted_average(values, errors)
        return Measured(mean, stat,
                        syst.get((project.project_name, name), 0.0))

    def compute(self, *_a):
        ref = self._Reference()
        self._moments = []
        if not (ref.mu and (ref.A_lower or ref.A_upper)):
            self._Status.setText("Enter the reference's μ and an A.")
            self._Fill()
            return
        syst = {(r["project"], r["parameter"]): float(r["sys_total"])
                for r in self._analysis_tab._sys_tab.summary_rows()
                if r.get("run") == MEAN_RUN
                and np.isfinite(r.get("sys_total", np.nan))}
        for project, config in self._States():
            m = {n: self._Fitted(project, config, n, syst)
                 for n in ("Al", "Au", "Bl", "Bu")}
            spin, j_l, j_u = (float(config["params"][k]["value"])
                              for k in ("I", "Jl", "Ju"))
            key = f"{project.project_name}/{config['name']}"
            mu = (Magnetic_Moment(m["Al"], m["Au"], spin, ref,
                                  anomaly_pct=self._anomaly.get(key, 0.0))
                  ["mean"] if spin > 0 and (m["Al"] or m["Au"]) else None)
            q = Quadrupole_Moment(m["Bl"], m["Bu"], spin, j_l, j_u, ref)
            check = (A_Ratio_Check(m["Al"], m["Au"], ref)
                     if m["Al"] and m["Au"] and ref.A_lower and ref.A_upper
                     else None)
            self._moments.append({
                "project": project.project_name, "state": config["name"],
                "key": key, "I": spin, "mu": mu, "Q": q and q["mean"],
                "check": check})
        self._Status.setText(f"{len(self._moments)} state(s).")
        self._Fill()

    def Moments(self):
        """The last computed rows: project, state, I, mu, Q (Moments or
        None) and the A_u/A_l check (or None)."""
        return list(self._moments)

    # ── the table ────────────────────────────────────────────
    def _Fill(self):
        table = self._Table
        table.blockSignals(True)
        table.setRowCount(len(self._moments))
        for r, row in enumerate(self._moments):
            check = row["check"]
            cells = ([row["project"], row["state"], f"{row['I']:g}"]
                     + _Cells(row["mu"], MU_FIELDS)
                     + [f"{self._anomaly.get(row['key'], 0.0):g}"]
                     + _Cells(row["Q"], Q_FIELDS)
                     + ["—" if check is None else
                        f"{check['pull']:+.1f}σ"
                        + (" !" if check["flag"] else "")])
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c != ANOMALY_COL:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)
            if check is not None and check["flag"]:
                table.item(r, len(cells) - 1).setToolTip(
                    "A_u/A_l is more than 3σ from the reference's: a "
                    "hyperfine anomaly,\na wrong spin or a wrong line "
                    "assignment.")
        table.blockSignals(False)

    def _On_Anomaly_Edited(self, item):
        if item.column() != ANOMALY_COL:
            return
        try:
            pct = float(item.text())
        except ValueError:
            pct = 0.0
        self._anomaly[self._moments[item.row()]["key"]] = pct
        self.compute()

    def refresh_projects(self):
        """The project list changed: the last table may no longer hold."""
        self._moments = []
        self._Status.setText("")
        self._Fill()

    # ── session ──────────────────────────────────────────────
    def to_dict(self):
        return {"reference": {k: w.value() for k, w in self._Ref.items()},
                "citation": self._Citation.text(),
                "anomaly_pct": dict(self._anomaly)}

    def from_dict(self, d):
        ref = d.get("reference") or {}
        for key, widget in self._Ref.items():
            widget.setValue(float(ref.get(key, 0.0)))
        self._Citation.setText(d.get("citation", ""))
        self._anomaly = dict(d.get("anomaly_pct") or {})


def _Cells(moment, fields):
    """A Moment's fields as table text; dashes when there is none."""
    if moment is None:
        return ["—"] * len(fields)
    return [f"{getattr(moment, f):.6g}" for f in fields]
