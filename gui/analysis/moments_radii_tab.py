"""Moments & Radii: nuclear moments and charge radii as an analysis technique.

Date:    2026-11-11
Version: 1.0.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A permanent tab in the Analysis tab's own tab bar, beside Isotope Shifts
and Systematics and built the way they are; never a main-window tab. From
every isotope project's fitted hyperfine constants it gives the nuclear
moments by the ratio method against a reference isotope; from the Isotope
Shifts tab's shifts, the changes in mean-square charge radius, with F and K
given, calibrated on a King plot, or taken from a second line; both with
the Systematics tab's budget (cls_estimations.nuclear_moments). The King
plot is drawn by one function for the tab and the Results tab, and "Send
to Results" writes both tables, every row naming its inputs.

Depends on: PySide6, numpy, matplotlib, cls_estimations (isotope_shift,
mass_lookup, nuclear_moments), gui.analysis (blocks, helpers, naming).
"""

from __future__ import annotations

import csv
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

from cls_estimations.isotope_shift import weighted_average
from cls_estimations.nuclear_moments import (
    A_Ratio_Check, Inverse_Mass_Factor, King_Calibration, Magnetic_Moment,
    Measured, Quadrupole_Moment, Radii_From_F_K, Reference, Species_Mass,
    Two_Line_King)
from gui.analysis.helpers import _make_double, _make_int
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
SHIFT_SYS = ("Isotope shifts", "delta_nu")   # its rows for the shifts
RADII_MODES = {"given": "F, K given", "king": "King-plot calibration",
               "two_line": "Second line"}
RADII_FIELDS = ("dnu", "dr2", "stat", "syst", "atomic", "total")
RADII_COLS = ("Pair", "δν [MHz]", "δ⟨r²⟩ [fm²]",
              "stat", "syst", "atomic", "total")
POINT_KEYS = ("A", "value", "sigma", "citation")
FK_KEYS = ("F", "F_sigma", "K", "K_sigma", "rho")


class MomentsRadiiTab(QGroupBox):
    """Reference and radii inputs on the left; moments, radii and the King
    plot on the right."""

    results_ready = Signal(str, list, dict)

    def __init__(self, analysis_tab, parent=None):
        super().__init__("Nuclear moments and charge radii", parent)
        self._analysis_tab = analysis_tab
        self._anomaly = {}          # "project/state" -> allowance, %
        self._moments, self._radii, self._King, self._Line = [], [], None, None
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        root.addWidget(split)
        left, right = QWidget(), QWidget()
        lcol, rcol = QVBoxLayout(left), QVBoxLayout(right)
        lcol.setContentsMargins(0, 0, 0, 0)
        rcol.setContentsMargins(0, 0, 0, 0)
        split.addWidget(left)
        split.addWidget(right)

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
            self._Ref[key + "_sigma"] = _Sigma_Row(form, label,
                                                   self._Ref[key], 6)
        self._Citation = QLineEdit(toolTip="Where these values come from.")
        form.addRow("Citation:", self._Citation)
        fill = QPushButton("From Estimate", toolTip=(
            "Take I, μ, Q, A and B from the Estimate tab's reference "
            "isotope;\nthe uncertainties stay as typed."))
        fill.clicked.connect(self._From_Estimate)
        form.addRow("", fill)
        lcol.addWidget(ref_box)

        rad_box = QGroupBox("Charge radii")
        rform = QFormLayout(rad_box)
        self._Mode = QComboBox(toolTip=(
            "Where F and K come from: given; fitted to the pairs below of "
            "known δ⟨r²⟩;\nor from a second line's "
            "shifts for the pairs below, with that line's F and K."))
        for key, text in RADII_MODES.items():
            self._Mode.addItem(text, key)
        rform.addRow("F and K:", self._Mode)
        self._Rad = {}
        for key, label, decimals in (("F", "F [MHz/fm²]", 3),
                                     ("K", "K [MHz u]", 1)):
            self._Rad[key] = _make_double(0.0, -1e12, 1e12, decimals, 1.0,
                                          tooltip=f"The field or mass shift "
                                                  f"factor, {label}.")
            self._Rad[key + "_sigma"] = _Sigma_Row(rform, label,
                                                   self._Rad[key], decimals)
        self._Rad["rho"] = _make_double(0.0, -1.0, 1.0, 3, 0.1,
                                        tooltip="The correlation of F and K.")
        rform.addRow("ρ(F, K):", self._Rad["rho"])
        self._Charge = _make_int(1, 0, 20, tooltip=(
            "The charge state measured: the masses are atomic less this many "
            "electrons\n(0 for atoms neutralised in the CEC)."))
        rform.addRow("Charge:", self._Charge)
        self._Rad_Citation = QLineEdit(toolTip="Where F, K or the pairs "
                                               "come from.")
        rform.addRow("Citation:", self._Rad_Citation)
        self._Points = _Table(("A", "", "σ", "Citation"), (
            "Pairs against the Isotope Shifts reference, by the other "
            "isotope's A."))
        rform.addRow(self._Points)
        prow = QHBoxLayout()
        for text, step in (("Add pair", 1), ("Remove pair", -1)):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _c=False, s=step: self._Points.setRowCount(
                max(0, self._Points.rowCount() + s)))
            prow.addWidget(btn)
        rform.addRow(prow)
        self._Mode.currentIndexChanged.connect(self._On_Mode)
        lcol.addWidget(rad_box)
        lcol.addStretch()

        mom_box = QGroupBox("Moments")
        col = QVBoxLayout(mom_box)
        top = QHBoxLayout()
        self._Status = QLabel("")
        top.addWidget(self._Status, 1)
        for text, tip, slot in (
                ("Compute", "μ and Q of every HFS state, and the radii "
                 "from the Isotope Shifts tab's shifts,\nwith the Systematics "
                 "tab's budget.", self.compute),
                ("Send to Results", "Write the moments, the radii and the "
                 "King plot as an iteration.", self._Send_To_Results)):
            btn = QPushButton(text, toolTip=tip)
            btn.clicked.connect(slot)
            top.addWidget(btn)
        col.addLayout(top)
        self._Table = _Table(COLUMNS, (
            "stat: the fits' errors; syst: the Systematics tab's budget;\n"
            "ref: the reference isotope's; anomaly: the allowance typed in\n"
            "Anomaly [%]. A_u/A_l: its pull from the reference's ratio."))
        self._Table.itemChanged.connect(self._On_Anomaly_Edited)
        col.addWidget(self._Table)
        rcol.addWidget(mom_box, 1)

        radii_box = QGroupBox("Changes in mean-square charge radius")
        self._Radii_Table = _Table(RADII_COLS, (
            "Against the Isotope Shifts reference. stat and syst: the shifts' "
            "(the\nSystematics tab's shift systematic as one source); atomic: "
            "F and K."))
        QVBoxLayout(radii_box).addWidget(self._Radii_Table)
        rcol.addWidget(radii_box, 1)

        king_box = QGroupBox("King plot")
        self._Figure = Figure(figsize=(6.0, 4.0))
        self._Canvas = FigureCanvasQTAgg(self._Figure)
        self._Canvas.setMinimumHeight(260)
        QVBoxLayout(king_box).addWidget(self._Canvas)
        rcol.addWidget(king_box, 2)
        self._Draw()

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

    def _On_Mode(self):
        mode = self._Mode.currentData()
        for key in FK_KEYS:
            self._Rad[key].setEnabled(mode != "king")
        self._Points.setEnabled(mode != "given")
        self._Points.horizontalHeaderItem(1).setText(
            "δν₂ [MHz]" if mode == "two_line"
            else "δ⟨r²⟩ [fm²]")

    def _Point_Rows(self):
        out = []
        for r in range(self._Points.rowCount()):
            cells = [(self._Points.item(r, c) or QTableWidgetItem("")).text()
                     .strip() for c in range(len(POINT_KEYS))]
            try:
                out.append({"A": int(cells[0]), "value": float(cells[1]),
                            "sigma": float(cells[2] or 0), "citation": cells[3]})
            except ValueError:
                continue
        return out

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
        for result in project._last_results or []:   # never fitted: None
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
        self._moments, self._radii, self._King, self._Line = [], [], None, None
        sys_rows = [r for r in self._analysis_tab._sys_tab.summary_rows()
                    if np.isfinite(r.get("sys_total", np.nan))]
        notes = []
        ref = self._Reference()
        if ref.mu and (ref.A_lower or ref.A_upper):
            self._Compute_Moments(ref, {
                (r["project"], r["parameter"]): float(r["sys_total"])
                for r in sys_rows if r.get("run") == MEAN_RUN})
        else:
            notes.append("Enter the reference's μ and an A.")
        try:
            self._Compute_Radii({r["run"]: float(r["sys_total"])
                                 for r in sys_rows if (r.get("project"),
                                 r.get("parameter")) == SHIFT_SYS})
        except ValueError as exc:
            notes.append(str(exc))
        self._Status.setText("; ".join(notes) or (
            f"{len(self._moments)} state(s), {len(self._radii)} pair(s)."))
        self._Fill()

    def _Compute_Moments(self, ref, syst):
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

    def _Z(self):
        from gui.analysis.blocks import SourceBlock
        for project in self._analysis_tab._projects:
            for source in project._get_blocks_by_type(SourceBlock):
                return int(source.get_source_config()["Z"])
        raise ValueError("No project gives the element's Z.")

    def _Compute_Radii(self, scan_syst):
        """dr2 of every pair against the Isotope Shifts reference, and of
        each isomer pair (the same A) as dnu / F."""
        mode, points = self._Mode.currentData(), self._Point_Rows()
        if (len(points) < 2 if mode == "king" else not self._Rad["F"].value()):
            return
        rows = self._analysis_tab._is_tab.shifts_for_scan()
        ref = next((r for r in rows if r.get("is_reference")), None)
        rows = [r for r in rows if not r.get("is_reference")]
        if ref is None or not rows:
            raise ValueError("No isotope shifts: compute them on the "
                             "Isotope Shifts tab.")
        from cls_estimations.mass_lookup import load_mass_table
        z, q, table = self._Z(), self._Charge.value(), load_mass_table()
        m_ref = Species_Mass(z, ref["A"], q, table)
        inv_mu = np.array([Inverse_Mass_Factor(
            m_ref, Species_Mass(z, r["A"], q, table)) for r in rows])
        dnu = np.array([r["delta_nu"] for r in rows])
        # Shifts against one reference share its error; the Systematics
        # tab's shift systematic is one source, so moves them all.
        n = len(rows)
        c_stat = np.full((n, n), sum(ref.get(k, 0.0)**2 for k in (
            "sigma_fit", "sigma_scatter", "sigma_correction")))
        np.fill_diagonal(c_stat, [sum(r.get(k, 0.0)**2 for k in (
            "sigma_delta_nu_stat", "sigma_delta_nu_scatter",
            "sigma_delta_nu_correction")) for r in rows])
        c_syst = np.full((n, n), ref.get("sigma_voltage", 0.0)**2)
        np.fill_diagonal(c_syst, [r.get("sigma_delta_nu_sys", 0.0)**2
                                  for r in rows])
        scan = np.array([scan_syst.get(r["label"], 0.0) for r in rows])
        c_syst += np.outer(scan, scan)
        sigma = np.sqrt(np.diag(c_stat + c_syst))
        given = [self._Rad[k].value() for k in FK_KEYS]
        f, sf, k, sk, rho = given
        line = (f, k, sf, sk, rho * sf * sk)        # the King line drawn
        if mode != "given":
            first = {}
            for i, r in enumerate(rows):
                if inv_mu[i]:
                    first.setdefault(r["A"], i)
            use = [p for p in points if p["A"] in first]
            idx = [first[p["A"]] for p in use]
            pv, ps = (np.array([p[x] for p in use]) for x in ("value", "sigma"))
            args = (dnu[idx], sigma[idx], pv, ps, inv_mu[idx])
            fit = self._King = (King_Calibration(*args) if mode == "king" else
                                Two_Line_King(*args, *given[:4], rho2=rho))
            f, sf, k, sk, rho = (fit[x] for x in ("F", "F_sigma", "K",
                                                  "K_sigma", "rho_FK"))
            line = ((f, k, sf, sk, fit["cov_FK"]) if mode == "king" else
                    tuple(fit[x] for x in ("F_ratio", "K_offset",
                          "F_ratio_sigma", "K_offset_sigma",
                          "ratio_offset_cov")))
            points = {"x": list(pv / inv_mu[idx]),
                      "y": list(dnu[idx] / inv_mu[idx]),
                      "sx": list(np.abs(ps / inv_mu[idx])),
                      "sy": list(np.abs(sigma[idx] / inv_mu[idx])),
                      "labels": [rows[i]["label"] for i in idx]}
        self._Line = dict(zip(("slope", "intercept", "slope_sigma",
                               "intercept_sigma", "cov"), line))
        self._Line["points"] = points if mode != "given" else {}
        out = Radii_From_F_K(dnu, inv_mu, f, sf, k, sk, rho, c_stat, c_syst)
        self._radii = [dict(pair=r["label"], isomer=not inv_mu[i],
                            inv_mu=inv_mu[i], sigma_dnu=sigma[i], dnu=dnu[i],
                            **{x: out[x][i] for x in RADII_FIELDS[1:]})
                       for i, r in enumerate(rows)]
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(r["A"], []).append(i)
        for base, *others in (g for g in groups.values() if len(g) > 1):
            for j in others:
                d = np.zeros(n)
                d[[j, base]] = 1.0, -1.0
                pair = Radii_From_F_K([d @ dnu], [0.0], f, sf, k, sk, rho,
                                      [[d @ c_stat @ d]], [[d @ c_syst @ d]])
                self._radii.append(dict(
                    pair=f"{rows[j]['label']} − {rows[base]['label']}",
                    isomer=True, inv_mu=0.0, dnu=d @ dnu,
                    sigma_dnu=float(np.sqrt(d @ (c_stat + c_syst) @ d)),
                    **{x: pair[x][0] for x in RADII_FIELDS[1:]}))

    def Moments(self):
        """The last computed rows: project, state, I, mu, Q (Moments or
        None) and the A_u/A_l check (or None)."""
        return list(self._moments)

    def Radii(self):
        """The last computed pairs: pair, isomer, dnu, dr2 and its stat /
        syst / atomic / total sigma."""
        return list(self._radii)

    # ── the tables and the plot ──────────────────────────────
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
            _Set_Row(table, r, cells, editable=ANOMALY_COL)
            if check is not None and check["flag"]:
                table.item(r, len(cells) - 1).setToolTip(
                    "A_u/A_l is more than 3σ from the reference's: a "
                    "hyperfine anomaly,\na wrong spin or a wrong line "
                    "assignment.")
        table.blockSignals(False)
        self._Radii_Table.setRowCount(len(self._radii))
        for r, row in enumerate(self._radii):
            _Set_Row(self._Radii_Table, r, [row["pair"]] + [
                f"{row[x]:.6g}" for x in RADII_FIELDS])
            if row["isomer"]:
                self._Radii_Table.item(r, 0).setToolTip(
                    "An isomer pair: no mass shift, so δ⟨r²"
                    "⟩ = δν/F; not on the King plot.")
        self._Draw()

    def King_Plot_Data(self):
        """What the King plot shows, as saved for the Results tab."""
        if self._Line is None:
            return {}
        two = self._Mode.currentData() == "two_line"
        meas = [r for r in self._radii if not r["isomer"]]
        d = {"line": dict(self._Line, label=(
                 "York fit" if self._King else "F, K given")),
             "xlabel": ("μ δν₂ [MHz u]" if two else
                        "μ δ⟨r²⟩ [u fm²]"),
             "ylabel": "μ δν [MHz u]",
             "points": self._Line.get("points", {})}
        d["line"].pop("points", None)
        if not two:
            d["measured"] = {
                "x": [r["dr2"] / r["inv_mu"] for r in meas],
                "y": [r["dnu"] / r["inv_mu"] for r in meas],
                "sx": [r["total"] / abs(r["inv_mu"]) for r in meas],
                "sy": [r["sigma_dnu"] / abs(r["inv_mu"]) for r in meas],
                "labels": [r["pair"] for r in meas]}
        return d

    def _Draw(self):
        Draw_King_Plot(self._Figure, self.King_Plot_Data())
        self._Canvas.draw_idle()

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
        """The project list changed: the last results may no longer hold."""
        self._moments, self._radii, self._King, self._Line = [], [], None, None
        self._Status.setText("")
        self._Fill()

    # ── Results ──────────────────────────────────────────────
    def _Send_To_Results(self):
        from gui.shared_widgets import get_analysis_dir
        project = "Moments_Radii"
        pdir = os.path.join(get_analysis_dir(), project)
        done = [int(n[5:]) for n in (os.listdir(pdir) if os.path.isdir(pdir)
                                     else []) if n[5:].isdigit()]
        iter_name = f"iter_{max(done, default=0) + 1:03d}"
        idir = os.path.join(pdir, iter_name)
        os.makedirs(os.path.join(idir, "plots"), exist_ok=True)
        cites = "; ".join(filter(None, [self._Rad_Citation.text()] + [
            p["citation"] for p in self._Point_Rows()]))
        rad_src = f"{self._Mode.currentText()}: {cites or 'uncited'}"
        ref_src = f"reference: {self._Citation.text() or 'uncited'}"
        with open(os.path.join(idir, "moments.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["project", "state", "I"]
                       + [f"mu_{f}" for f in MU_FIELDS]
                       + [f"Q_{f}" for f in Q_FIELDS]
                       + ["A_ratio_pull", "inputs"])
            for m in self._moments:
                pct = self._anomaly.get(m["key"], 0.0)
                w.writerow([m["project"], m["state"], m["I"]]
                           + _Cells(m["mu"], MU_FIELDS)
                           + _Cells(m["Q"], Q_FIELDS)
                           + [m["check"]["pull"] if m["check"] else "",
                              f"{ref_src}; anomaly {pct:g} %"])
        with open(os.path.join(idir, "radii.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "isomer", "delta_nu_mhz", "dr2_fm2", "stat",
                        "syst", "atomic", "total", "inputs"])
            for r in self._radii:
                w.writerow([r["pair"], int(r["isomer"])]
                           + [f"{r[x]:.8g}" for x in RADII_FIELDS]
                           + [rad_src])
        line = self._Line or {}
        report = "\n".join([
            "Moments and charge radii", ref_src, rad_src,
            f"line: slope {line.get('slope', float('nan')):.6g} +/- "
            f"{line.get('slope_sigma', float('nan')):.3g}, intercept "
            f"{line.get('intercept', float('nan')):.6g} +/- "
            f"{line.get('intercept_sigma', float('nan')):.3g}"
            + (f", chi2/dof {self._King['chi2']:.3g}/{self._King['dof']}"
               if self._King else "")])
        with open(os.path.join(idir, "fit_report.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(report)
        plots = os.path.join(idir, "plots")
        self._Figure.savefig(os.path.join(plots, "king_plot.png"), dpi=150,
                             bbox_inches="tight")
        np.savez(os.path.join(plots, "king_plot.npz"), plot_type="king_plot",
                 payload=np.array(self.King_Plot_Data(), dtype=object))
        self.results_ready.emit(project, [{
            "success": True, "run_number": "moments_radii", "run_file": "",
            "report": report, "params_df": {}, "metadata_df": {}, "x": [],
            "y": [], "yerr": [], "y_fit": [], "x_smooth": [],
            "y_fit_smooth": [], "residuals": [], "diagnostics": {},
            "fwhm": {}, "peak_positions": {}, "fit_quality": {},
            "run_metadata": {}, "harmonic": 0}],
            {"report": True, "params_csv": False, "metadata_csv": False,
             "fit_plots": True, "iter_label": iter_name,
             "iter_name": iter_name})
        self._Status.setText(f"Written to {idir} and sent to Results.")

    # ── session ──────────────────────────────────────────────
    def to_dict(self):
        return {"reference": {k: w.value() for k, w in self._Ref.items()},
                "citation": self._Citation.text(),
                "anomaly_pct": dict(self._anomaly),
                "radii": dict({k: w.value() for k, w in self._Rad.items()},
                              mode=self._Mode.currentData(),
                              charge=self._Charge.value(),
                              citation=self._Rad_Citation.text(),
                              points=self._Point_Rows())}

    def from_dict(self, d):
        ref = d.get("reference") or {}
        for key, widget in self._Ref.items():
            widget.setValue(float(ref.get(key, 0.0)))
        self._Citation.setText(d.get("citation", ""))
        self._anomaly = dict(d.get("anomaly_pct") or {})
        rad = d.get("radii") or {}
        self._Mode.setCurrentIndex(max(0, self._Mode.findData(
            rad.get("mode", "given"))))
        for key, widget in self._Rad.items():
            widget.setValue(float(rad.get(key, 0.0)))
        self._Charge.setValue(int(rad.get("charge", 1)))
        self._Rad_Citation.setText(rad.get("citation", ""))
        points = rad.get("points") or []
        self._Points.setRowCount(len(points))
        for r, p in enumerate(points):
            for c, key in enumerate(POINT_KEYS):
                self._Points.setItem(r, c, QTableWidgetItem(str(p.get(key, ""))))
        self._On_Mode()


def Draw_King_Plot(fig, d):
    """The King plot: the pairs the line was fitted to, the measured pairs
    placed with F and K, and the line with its 1-sigma band. Shared by the
    tab and the Results tab, so what is saved is what was on screen."""
    fig.clear()
    ax = fig.add_subplot(111)
    xs = [x for key in ("points", "measured")
          for x in (d.get(key) or {}).get("x", [])] if d else []
    if not xs:
        ax.text(0.5, 0.5, "No King plot yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, alpha=0.6)
        ax.set_axis_off()
        return
    line = d["line"]
    pad = 0.05 * (max(xs) - min(xs) or abs(max(xs)) or 1.0)
    grid = np.linspace(min(xs) - pad, max(xs) + pad, 200)
    y = line["intercept"] + line["slope"] * grid
    band = np.sqrt(np.maximum(line["intercept_sigma"]**2
                              + grid**2 * line["slope_sigma"]**2
                              + 2 * grid * line["cov"], 0.0))
    ax.fill_between(grid, y - band, y + band, color="#1f77b4", alpha=0.15,
                    lw=0, label="1σ band")
    ax.plot(grid, y, color="#1f77b4", lw=1.2, label=line.get("label", ""))
    named = set()          # a calibration pair that is also measured: once
    ys = [v for key in ("points", "measured")
          for v in (d.get(key) or {}).get("y", [])]
    span = ((max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0)
    placed = []            # pairs that coincide (55Fe, 53mFe): labels stack
    for key, fmt, color, label in (("points", "s", "#c0392b", "fitted pairs"),
                                   ("measured", "o", "#222222", "measured")):
        p = d.get(key) or {}
        if not p.get("x"):
            continue
        ax.errorbar(p["x"], p["y"], xerr=p.get("sx"), yerr=p.get("sy"),
                    fmt=fmt, ms=5, capsize=2, color=color, label=label)
        for x, yv, text in zip(p["x"], p["y"], p.get("labels", [])):
            if text in named:
                continue
            named.add(text)
            near = sum(abs(x - px) < 0.02 * span[0]
                       and abs(yv - py) < 0.02 * span[1] for px, py in placed)
            placed.append((x, yv))
            ax.annotate(text, (x, yv), textcoords="offset points",
                        xytext=(4, 4 + 10 * near), fontsize=8)
    ax.set_xlabel(d.get("xlabel", ""))
    ax.set_ylabel(d.get("ylabel", ""))
    ax.legend(fontsize=8)


def _Sigma_Row(form, label, value, decimals):
    """``value`` with a sigma box beside it, as one form row; the sigma."""
    sigma = _make_double(0.0, 0.0, 1e12, decimals, 0.1,
                         tooltip="Its standard uncertainty.")
    row = QHBoxLayout()
    row.addWidget(value, 1)
    row.addWidget(QLabel("±"))
    row.addWidget(sigma, 1)
    form.addRow(label + ":", row)
    return sigma


def _Table(columns, tooltip):
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    table.setToolTip(tooltip)
    return table


def _Set_Row(table, r, cells, editable=None):
    for c, text in enumerate(cells):
        item = QTableWidgetItem(text)
        if c != editable:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(r, c, item)


def _Cells(moment, fields):
    """A Moment's fields as table text; dashes when there is none."""
    if moment is None:
        return ["—"] * len(fields)
    return [f"{getattr(moment, f):.6g}" for f in fields]
