"""Cooler-voltage systematics: set the scan up, watch it, band it.

The Yb calibration hands over an interval rather than a number. This
tab repeats the analysis across that interval and reports how far each
extracted parameter moves -- the cooler-voltage systematic.

The work happens in gui/analysis/systematic_scan.py (which drives each
project's own fit) and cls_estimations/systematics.py (which does the
arithmetic). What lives here is the setup, the step list with its
flags, the seed editor for steps that misbehaved, and the tables and
plot that go to Results.

Note this is NOT the systematic already reported by the isotope-shift
tab: that one bands the run-to-run jitter of the logged cooler voltage
(isotope_shift.cooler_voltage_systematic). This one bands the
calibration offset itself. They are different numbers and both belong
in a budget.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QStyledItemDelegate, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from cls_estimations.systematics import (
    ALL, COMBINATIONS, DEFAULT_DEFINITION, DEFINITIONS, MAX_DEV,
    QUADRATURE, ExtraSystematic, ParameterBand, ScanValue, budget,
    mean_over_runs, quality_flag, scan_offsets, summarise,
    trend_outliers,
)
from gui.analysis.systematic_scan import (
    Baseline, SEED_CONTINUATION, SEED_MODES, ScanTarget, SystematicScan,
    config_differences, read_baseline, split_key, value_key,
)

#: Where the offsets come from. The calibration mode is the usual
#: three-offset batch: every run at the common offset (the numbers
#: quoted) and at the two zero crossings (the systematic) -- three fits
#: per run. The range mode walks an even grid between two offsets.
MODE_CALIBRATION = "Common offset + the two zero crossings"
MODE_RANGE = "Even grid over a range"
OFFSET_MODES = (MODE_CALIBRATION, MODE_RANGE)

#: Two offsets closer than this are the same fit.
SAME_DV = 1e-6

#: Columns of the step table.
STEP_COLS = ("Use", "Offset (V)", "Status", "Flag", "Value", "Error")
SUMMARY_COLS = ("Project", "Run", "Parameter", "N", "Value", "stat",
                "min", "max", "band", "low edge Δ", "high edge Δ",
                "cooler sys", "other sys", "sys total", "total", "dP/dV")
#: Columns of the extra-systematics table.
EXTRA_COLS = ("Use", "Name", "Project", "Run / isotope", "Parameter",
              "Value (MHz)")


def _wrap(text, width=54):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width))


def _ro(text):
    item = QTableWidgetItem(str(text))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


def _clear_cell_widgets(table):
    """Replacing a cell widget leaves the old one parented to the
    viewport, painting at its old geometry (cooler calibration tab,
    2026-09-24). Take them out before rebuilding."""
    for r in range(table.rowCount()):
        for c in range(table.columnCount()):
            old = table.cellWidget(r, c)
            if old is not None:
                table.removeCellWidget(r, c)
                old.setParent(None)
                old.deleteLater()
    table.setRowCount(0)


class _ChoiceDelegate(QStyledItemDelegate):
    """An editable combo for a filter cell: "all", or a name the scan
    has seen -- typed names are allowed (a project not scanned yet)."""

    def __init__(self, choices_fn, parent=None):
        super().__init__(parent)
        self._choices_fn = choices_fn

    def createEditor(self, parent, _option, index):
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.addItems([ALL] + [c for c in self._choices_fn(index.column())
                                if c and c != ALL])
        return combo

    def setEditorData(self, editor, index):
        editor.setCurrentText(str(index.data() or ALL))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText().strip() or ALL)


def draw_systematics(fig, d):
    """Draw one parameter against the assumed cooler offset.

    Shared by the live panel and the copy written to Results, so what
    gets saved is what was on screen.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    d = d or {}
    runs = d.get("runs") or []
    base = d.get("baseline", float("nan"))
    delta = bool(d.get("delta"))

    def _y(vals):
        if delta and math.isfinite(base):
            return [v - base for v in vals]
        return list(vals)

    if not runs and not (d.get("mean") or {}).get("dv"):
        ax.text(0.5, 0.5, "No scan yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, alpha=0.6)
        ax.set_axis_off()
        return

    for r in runs:
        ax.errorbar(r.get("dv", []), _y(r.get("value", [])),
                    yerr=r.get("sigma", []), fmt="o-", ms=4, lw=1.0,
                    capsize=2, alpha=0.75, label=r.get("label", ""))
        ex_dv = r.get("excluded_dv") or []
        if ex_dv:
            ax.plot(ex_dv, _y(r.get("excluded_value", [])), "x",
                    ms=9, color="#c0392b", zorder=5,
                    label="excluded" if r is runs[0] else None)

    mean = d.get("mean") or {}
    if mean.get("dv"):
        ax.errorbar(mean["dv"], _y(mean.get("value", [])),
                    yerr=mean.get("sigma", []), fmt="s-", ms=6, lw=2.0,
                    capsize=3, color="#1f77b4", zorder=6,
                    label="weighted mean")

    if math.isfinite(base):
        ax.axhline(0.0 if delta else base, ls="--", lw=1.0,
                   color="#555555", zorder=1,
                   label=d.get("baseline_label") or "baseline fit")
        bs = d.get("baseline_sigma", float("nan"))
        if math.isfinite(bs) and bs > 0:
            centre = 0.0 if delta else base
            ax.axhspan(centre - bs, centre + bs, color="#555555",
                       alpha=0.12, zorder=0)

    band = d.get("band")
    if band and all(math.isfinite(v) for v in band):
        lo, hi = band
        if delta and math.isfinite(base):
            lo, hi = lo - base, hi - base
        ax.axhspan(lo, hi, color="#f39c12", alpha=0.12, zorder=0,
                   label="scanned band")

    quoted = d.get("quoted_dv")
    if quoted is not None and math.isfinite(quoted):
        ax.axvline(quoted, color="#6a1b9a", ls=":", lw=1.2, zorder=1,
                   label=f"common offset {quoted:+.2f} V (quoted)")

    ax.set_xlabel("Assumed cooler offset (V)")
    unit = d.get("unit", "MHz")
    name = d.get("parameter", "parameter")
    ax.set_ylabel(f"{'delta ' if delta else ''}{name} ({unit})")
    title = d.get("title", "")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, ncol=2, framealpha=0.85)
    fig.tight_layout()


class SeedDialog(QDialog):
    """Starting values for one step, for one project.

    The whole point of the scan is that some offsets are hard: a
    centroid hundreds of MHz from where the baseline left it does not
    always find its way back. This is where the user nudges it and
    re-runs that step alone.
    """

    def __init__(self, dv, project, sources, baseline, overrides,
                 method="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Seeds for {project} at {dv:+.3f} V")
        self._dv = float(dv)
        self._project = project
        self._baseline = baseline
        self._overrides = dict(overrides or {})
        self._sources = list(sources)

        lay = QVBoxLayout(self)
        note = QLabel(
            f"Starting values for the fits at {dv:+.3f} V. Blank means "
            f"'as the scan would have it' -- the neighbouring step's "
            f"answer, or the baseline iteration.")
        note.setWordWrap(True)
        lay.addWidget(note)

        row = QHBoxLayout()
        row.addWidget(QLabel("Run:"))
        self._src = QComboBox()
        self._src.addItem("All runs")
        self._src.addItems(self._sources)
        self._src.currentTextChanged.connect(self._reload)
        row.addWidget(self._src, 1)
        row.addWidget(QLabel("Fit method:"))
        self._method = QComboBox()
        self._method.addItems(["(project default)", "leastsq",
                               "least_squares", "nelder", "powell",
                               "cobyla", "slsqp"])
        if method:
            idx = self._method.findText(method)
            if idx >= 0:
                self._method.setCurrentIndex(idx)
        self._method.setToolTip(_wrap(
            "A different minimiser for this step only. Nelder-Mead is "
            "slower but far less fussy about where it starts."))
        row.addWidget(self._method)
        lay.addLayout(row)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Model", "Parameter", "Baseline", "Start", "Min", "Max"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self._table, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        clear = buttons.addButton("Clear seeds",
                                  QDialogButtonBox.ButtonRole.ResetRole)
        clear.clicked.connect(self._clear)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._reload()
        self.resize(560, 420)

    def _current_source(self):
        text = self._src.currentText()
        return "" if text == "All runs" else text

    def _params(self):
        src = self._current_source()
        if src:
            return sorted(self._baseline.params.get(src, {}))
        return self._baseline.parameters()

    def _baseline_text(self, model, pname):
        """What the baseline iteration fitted, so the user can see
        what they are overriding."""
        src = self._current_source()
        if src:
            v = self._baseline.value(src, model, pname)
            return "" if v is None else f"{v:g}"
        seen = {round(self._baseline.value(s, model, pname), 6)
                for s in self._sources
                if self._baseline.value(s, model, pname) is not None}
        if not seen:
            return ""
        if len(seen) == 1:
            return f"{seen.pop():g}"
        return "varies"

    def _reload(self):
        src = self._current_source()
        rows = self._params()
        self._table.setRowCount(0)
        self._table.setRowCount(len(rows))
        for r, (model, pname) in enumerate(rows):
            self._table.setItem(r, 0, _ro(model))
            self._table.setItem(r, 1, _ro(pname))
            self._table.setItem(r, 2,
                                _ro(self._baseline_text(model, pname)))
            key = value_key(self._project, src, model, pname)
            ov = self._overrides.get(key) or {}
            for col, field in ((3, "value"), (4, "min"), (5, "max")):
                v = ov.get(field)
                self._table.setItem(
                    r, col,
                    QTableWidgetItem("" if v is None else f"{v:g}"))

    def _clear(self):
        src = self._current_source()
        for model, pname in self._params():
            self._overrides.pop(
                value_key(self._project, src, model, pname), None)
        self._reload()

    def _harvest(self):
        src = self._current_source()
        for r in range(self._table.rowCount()):
            model = self._table.item(r, 0).text()
            pname = self._table.item(r, 1).text()
            key = value_key(self._project, src, model, pname)
            entry = {}
            for col, field in ((3, "value"), (4, "min"), (5, "max")):
                item = self._table.item(r, col)
                text = (item.text() if item else "").strip()
                if not text:
                    continue
                try:
                    entry[field] = float(text)
                except ValueError:
                    continue
            if entry:
                self._overrides[key] = entry
            else:
                self._overrides.pop(key, None)

    def accept(self):
        self._harvest()
        super().accept()

    @property
    def overrides(self):
        return dict(self._overrides)

    @property
    def method(self):
        text = self._method.currentText()
        return "" if text.startswith("(") else text


class SystematicsTab(QGroupBox):
    """The Systematics tab: one scan at a time, saved with the session."""

    results_ready = Signal(str, list, dict)

    def __init__(self, analysis_tab, parent=None):
        super().__init__("Cooler-voltage systematics", parent)
        self._analysis_tab = analysis_tab
        self._scan = SystematicScan(self)
        self._scan.progress.connect(self._on_progress)
        self._scan.step_done.connect(self._on_step_done)
        self._scan.step_failed.connect(self._on_step_failed)
        self._scan.finished.connect(self._on_finished)
        self._scan.failed.connect(self._on_failed)
        self._scan.log.connect(self._append_log)

        self._records = {}          # dv -> record
        self._excluded = set()      # dv values kept out of the band
        self._seed_overrides = {}   # dv -> {key: {value/min/max}}
        self._step_methods = {}     # dv -> fitter method for that step
        self._baselines = {}        # project -> Baseline
        self._targets_cache = []
        self._col_width = None
        self._loading = False       # from_dict: no mode-switch defaults
        self._build_ui()

    # ── layout ───────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        root.addWidget(self._main_splitter)

        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(0, 0, 0, 0)
        self._main_splitter.addWidget(left)

        col.addWidget(QLabel("Projects and their final fits:"))
        self._proj_table = QTableWidget(0, 5)
        self._proj_table.setHorizontalHeaderLabels(
            ["Use", "Project", "Role", "Baseline iteration", "Runs"])
        self._proj_table.verticalHeader().setVisible(False)
        self._proj_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._proj_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self._proj_table.horizontalHeader().setStretchLastSection(True)
        self._proj_table.setMaximumHeight(180)
        self._proj_table.setToolTip(_wrap(
            "Pick the iteration whose numbers you mean to publish. Its "
            "fitted values seed the scan, and its parameters are the "
            "ones that get a systematic. The scan fits with the "
            "project as it stands now -- if that has moved on since "
            "the iteration, the differences are listed before it runs."))
        col.addWidget(self._proj_table)

        refresh = QPushButton("Refresh projects")
        refresh.clicked.connect(self.refresh_projects)
        col.addWidget(refresh)

        rng = QGroupBox("Offsets")
        rl = QVBoxLayout(rng)
        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Fit at:"))
        self._mode = QComboBox()
        self._mode.addItems(OFFSET_MODES)
        self._mode.setToolTip(_wrap(
            "Common offset + the two zero crossings: every run is "
            "fitted three times -- at the common offset, whose results "
            "are the values you quote, and at each zero crossing, "
            "whose deviation from them is the cooler-voltage "
            "systematic. Even grid: walk a range in equal steps, to see how "
            "a parameter depends on the offset."))
        self._mode.currentTextChanged.connect(self._on_mode_changed)
        row0.addWidget(self._mode, 1)
        rl.addLayout(row0)
        row = QHBoxLayout()
        self._lbl_from = QLabel("from")
        row.addWidget(self._lbl_from)
        self._dv_min = self._spin(-10.0)
        row.addWidget(self._dv_min)
        self._lbl_to = QLabel("to")
        row.addWidget(self._lbl_to)
        self._dv_max = self._spin(10.0)
        row.addWidget(self._dv_max)
        self._lbl_v = QLabel("V,")
        row.addWidget(self._lbl_v)
        self._steps = QSpinBox()
        self._steps.setRange(2, 101)
        self._steps.setValue(10)
        self._steps.setToolTip(_wrap(
            "How many offsets to repeat the analysis at. Every project "
            "is refitted at each one, so this multiplies the time: 10 "
            "points over 5 projects of 5 runs is 250 fits."))
        row.addWidget(self._steps)
        self._lbl_points = QLabel("points")
        row.addWidget(self._lbl_points)
        row.addStretch()
        rl.addLayout(row)
        row2 = QHBoxLayout()
        from_cal = QPushButton("From calibration")
        from_cal.setToolTip(_wrap(
            "Take the common offset and the two zero crossings from "
            "the Cooler Calibration tab, as chosen there (how the "
            "common offset is picked, and whether the range is "
            "widened by the crossings' errors)."))
        from_cal.clicked.connect(self._range_from_calibration)
        row2.addWidget(from_cal)
        self._lbl_base = QLabel("baseline at")
        row2.addWidget(self._lbl_base)
        self._baseline_dv = self._spin(0.0)
        self._baseline_dv.setToolTip(_wrap(
            "Common offset: the offset every run is analysed at -- the "
            "fits there give the values you quote, and the scan starts "
            "there and works outward. Even grid: the offset the "
            "baseline iterations were fitted at."))
        row2.addWidget(self._baseline_dv)
        row2.addWidget(QLabel("V"))
        row2.addStretch()
        rl.addLayout(row2)
        for w in (self._dv_min, self._dv_max, self._baseline_dv):
            w.valueChanged.connect(self._update_offsets_label)
        self._steps.valueChanged.connect(self._update_offsets_label)
        col.addWidget(rng)
        # Outside the group box: both themes draw the frame at the
        # layout's edge, and a label on that line comes out clipped.
        self._offsets_label = QLabel("")
        # The themes' muted-note colour. palette(mid) is near-black in
        # both dark themes: the line was all but invisible.
        self._offsets_label.setObjectName("sectionNote")
        self._offsets_label.setWordWrap(True)
        # Clear of both group frames: a group box's rect extends above
        # its visible frame (margin-top carries the title), so a label
        # placed flush against one comes out looking struck through.
        self._offsets_label.setContentsMargins(2, 4, 2, 4)
        col.addWidget(self._offsets_label)

        how = QGroupBox("How")
        hl = QVBoxLayout(how)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Seeds:"))
        self._seed_mode = QComboBox()
        self._seed_mode.addItems(SEED_MODES)
        self._seed_mode.setToolTip(_wrap(
            "A centroid moves of order 10 MHz per volt, so at the far "
            "end of the range the baseline value is hundreds of MHz "
            "out -- outside a line width, and the fit does not come "
            "back. Continuation starts each step from the neighbouring "
            "offset's answer and carries the trend."))
        r1.addWidget(self._seed_mode, 1)
        hl.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Label:"))
        self._label = QLineEdit("sys_001")
        self._label.setToolTip(_wrap(
            "Names the Results folders: sys_001_-30V_CO and so on, one "
            "per offset, grouped together. Re-running a step writes "
            "its own folder again instead of piling up."))
        r2.addWidget(self._label, 1)
        hl.addLayout(r2)
        self._light = QCheckBox("Light output (no plots per step)")
        self._light.setChecked(True)
        self._light.setToolTip(_wrap(
            "Reports and CSVs are what you read when a step misbehaves; "
            "the plot sets are what fill the disk. 10 offsets over 5 "
            "projects is 50 iterations either way."))
        hl.addWidget(self._light)
        self._use_gp = QCheckBox("Refit the reference GP at each offset")
        self._use_gp.setChecked(True)
        self._use_gp.setToolTip(_wrap(
            "The cooler offset acts on the voltages, before the Doppler "
            "conversion; the GP correction acts in the rest frame "
            "after it. Moving the offset therefore moves the reference "
            "centroids the GP is trained on, so the GP has to be "
            "refitted per offset -- otherwise the samples are corrected "
            "into a frame that belongs to a different offset."))
        hl.addWidget(self._use_gp)
        self._use_is = QCheckBox("Recompute isotope shifts at each offset")
        self._use_is.setChecked(True)
        hl.addWidget(self._use_is)
        col.addWidget(how)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Run scan")
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 4px 12px; }")
        self._run_btn.clicked.connect(self._start_scan)
        btns.addWidget(self._run_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._scan.stop)
        btns.addWidget(self._stop_btn)
        btns.addStretch()
        col.addLayout(btns)

        btns2 = QHBoxLayout()
        self._rerun_btn = QPushButton("Re-run ticked steps")
        self._rerun_btn.setToolTip(_wrap(
            "Fits the ticked offsets again, with whatever seeds you "
            "set for them, and overwrites their folders."))
        self._rerun_btn.clicked.connect(self._rerun_selected)
        btns2.addWidget(self._rerun_btn)
        self._seed_btn = QPushButton("Edit seeds...")
        self._seed_btn.clicked.connect(self._edit_seeds)
        btns2.addWidget(self._seed_btn)
        btns2.addStretch()
        col.addLayout(btns2)

        self._status = QLabel("(no scan yet)")
        self._status.setWordWrap(True)
        col.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        col.addWidget(self._progress)

        res = QGroupBox("Systematic")
        rsl = QVBoxLayout(res)
        drow = QHBoxLayout()
        drow.addWidget(QLabel("Quote:"))
        self._definition = QComboBox()
        self._definition.addItems(DEFINITIONS)
        self._definition.setCurrentText(DEFAULT_DEFINITION)
        self._definition.setToolTip(_wrap(
            "A uniform scan across an interval is not a probability "
            "distribution, so there is no one right answer. Every "
            "definition is in the summary table; this picks the one "
            "the report quotes."))
        self._definition.currentTextChanged.connect(self.compute)
        drow.addWidget(self._definition, 1)
        rsl.addLayout(drow)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Add systematics:"))
        self._combine_sys = QComboBox()
        self._combine_sys.addItems(COMBINATIONS)
        self._combine_sys.setToolTip(_wrap(
            "How the cooler-offset term and the other systematics "
            "(Budget tab) are added up. In quadrature: they are "
            "independent. Linear sum: they may be fully correlated, "
            "the conservative bound."))
        self._combine_sys.currentTextChanged.connect(self.compute)
        crow.addWidget(self._combine_sys, 1)
        crow.addWidget(QLabel("with stat:"))
        self._combine_total = QComboBox()
        self._combine_total.addItems(COMBINATIONS)
        self._combine_total.setToolTip(_wrap(
            "How the total systematic and the statistical error make "
            "the total. Quadrature is the usual choice; both are "
            "always listed separately as well."))
        self._combine_total.currentTextChanged.connect(self.compute)
        crow.addWidget(self._combine_total, 1)
        rsl.addLayout(crow)
        self._birge = QCheckBox(
            "Scale the run average's error by √χ²/ν when above 1")
        self._birge.setToolTip(_wrap(
            "When the runs scatter more than their own errors allow, "
            "inflate the weighted mean's error by the Birge ratio -- "
            "a common choice for the statistical error. Off: "
            "the plain inverse-variance error."))
        self._birge.toggled.connect(self.compute)
        rsl.addWidget(self._birge)
        self._headline = QLabel("--")
        self._headline.setWordWrap(True)
        rsl.addWidget(self._headline)
        arow = QHBoxLayout()
        self._send_btn = QPushButton("Send to Results")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_to_results)
        arow.addWidget(self._send_btn)
        arow.addStretch()
        rsl.addLayout(arow)
        col.addWidget(res)
        col.addStretch()
        left.setMinimumWidth(480)

        right = QTabWidget()
        self._main_splitter.addWidget(right)

        plot_page = QWidget()
        pl = QVBoxLayout(plot_page)
        pl.setContentsMargins(0, 0, 0, 0)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Project:"))
        self._plot_project = QComboBox()
        self._plot_project.setMaximumWidth(260)
        self._plot_project.currentTextChanged.connect(self._on_plot_project)
        prow.addWidget(self._plot_project)
        prow.addWidget(QLabel("Parameter:"))
        self._plot_param = QComboBox()
        self._plot_param.setMaximumWidth(260)
        self._plot_param.currentTextChanged.connect(self.compute)
        prow.addWidget(self._plot_param)
        self._delta_cb = QCheckBox("Show change from baseline")
        self._delta_cb.toggled.connect(self.compute)
        prow.addWidget(self._delta_cb)
        prow.addStretch()
        pl.addLayout(prow)
        self._figure = Figure(figsize=(7.0, 5.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(300)
        pl.addWidget(self._canvas, 1)
        right.addTab(plot_page, "Plot")

        self._step_table = QTableWidget(0, len(STEP_COLS))
        self._step_table.setHorizontalHeaderLabels(list(STEP_COLS))
        self._step_table.verticalHeader().setVisible(False)
        self._step_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._step_table.setToolTip(_wrap(
            "One row per offset. Untick a step to keep it out of the "
            "band without losing it; tick it and press Re-run to fit "
            "it again."))
        self._step_table.cellDoubleClicked.connect(self._toggle_excluded)
        right.addTab(self._step_table, "Steps")

        self._sum_table = QTableWidget(0, len(SUMMARY_COLS))
        self._sum_table.setHorizontalHeaderLabels(list(SUMMARY_COLS))
        self._sum_table.verticalHeader().setVisible(False)
        self._sum_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        sum_page = QWidget()
        sl = QVBoxLayout(sum_page)
        sl.setContentsMargins(0, 0, 0, 0)
        self._per_run_cb = QCheckBox("Show each run as well as the mean")
        self._per_run_cb.toggled.connect(self.compute)
        sl.addWidget(self._per_run_cb)
        sl.addWidget(self._sum_table, 1)
        right.addTab(sum_page, "Summary")

        budget_page = QWidget()
        bl = QVBoxLayout(budget_page)
        bl.setContentsMargins(4, 4, 4, 4)
        bnote = QLabel(
            "Systematics the scan does not measure -- the scan-voltage "
            "calibration, the laser frequency, a mass, a line-shape "
            "choice... -- in the quantity's own unit (MHz). Each row "
            "applies where its Project, Run / isotope and Parameter "
            "match; 'all' matches everything. They are added to the "
            "cooler-offset term as set under Systematic on the left.")
        bnote.setWordWrap(True)
        bl.addWidget(bnote)
        self._extra_table = QTableWidget(0, len(EXTRA_COLS))
        self._extra_table.setHorizontalHeaderLabels(list(EXTRA_COLS))
        self._extra_table.verticalHeader().setVisible(False)
        hh = self._extra_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c, w in ((0, 44), (2, 170), (3, 150), (4, 150), (5, 110)):
            self._extra_table.setColumnWidth(c, w)
        self._extra_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._extra_delegate = _ChoiceDelegate(self._extra_choices,
                                               self._extra_table)
        for c in (2, 3, 4):
            self._extra_table.setItemDelegateForColumn(
                c, self._extra_delegate)
        self._extra_table.itemChanged.connect(self._on_extra_changed)
        bl.addWidget(self._extra_table, 1)
        brow = QHBoxLayout()
        add = QPushButton("Add")
        add.setToolTip("Add a systematic contribution.")
        add.clicked.connect(lambda: self._add_extra())
        brow.addWidget(add)
        rem = QPushButton("Remove")
        rem.setToolTip("Remove the selected rows.")
        rem.clicked.connect(self._remove_extras)
        brow.addWidget(rem)
        brow.addStretch()
        bl.addLayout(brow)
        right.addTab(budget_page, "Budget")

        self._grid_table = QTableWidget(0, 1)
        self._grid_table.verticalHeader().setVisible(False)
        self._grid_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        right.addTab(self._grid_table, "Grid")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        right.addTab(self._log, "Log")

        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([540, 1160])
        self._on_mode_changed(self._mode.currentText())
        self._update_offsets_label()
        self.refresh_projects()

    @staticmethod
    def _spin(value, lo=-5000.0, hi=5000.0, decimals=3):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(value)
        s.setSingleStep(1.0)
        return s

    def showEvent(self, ev):
        super().showEvent(ev)
        self._fit_project_columns()

    def changeEvent(self, ev):
        super().changeEvent(ev)
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.Type.FontChange:
            self._col_width = None
            self._fit_project_columns()

    # ── projects ─────────────────────────────────────────────
    def scannable_projects(self):
        """Every ordinary project: calibration projects measure the
        offset, they are not corrected by it."""
        return [p for p in getattr(self._analysis_tab, "_projects", [])
                if not getattr(p, "is_calibration", False)]

    @staticmethod
    def _iterations(project_name):
        from gui.shared_widgets import get_analysis_dir
        pdir = os.path.join(get_analysis_dir(), project_name)
        if not os.path.isdir(pdir):
            return []
        return sorted(d for d in os.listdir(pdir)
                      if d.startswith("iter_")
                      and os.path.isdir(os.path.join(pdir, d)))

    def refresh_projects(self):
        kept = {}
        for r in range(self._proj_table.rowCount()):
            name = self._proj_table.item(r, 1).text()
            kept[name] = (self._proj_table.cellWidget(r, 0).isChecked(),
                          self._proj_table.cellWidget(r, 3).currentText())
        projects = self.scannable_projects()
        _clear_cell_widgets(self._proj_table)
        self._proj_table.setRowCount(len(projects))
        for row, p in enumerate(projects):
            name = p.project_name
            use, chosen = kept.get(name, (True, ""))
            cb = QCheckBox()
            cb.setChecked(use)
            self._proj_table.setCellWidget(row, 0, cb)
            self._proj_table.setItem(row, 1, _ro(name))
            self._proj_table.setItem(
                row, 2,
                _ro("reference" if getattr(p, "is_reference", False)
                    else "sample"))
            combo = QComboBox()
            iters = self._iterations(name)
            combo.addItems(iters or ["(no iteration on disk)"])
            if chosen and chosen in iters:
                combo.setCurrentText(chosen)
            elif iters:
                combo.setCurrentIndex(len(iters) - 1)
            combo.currentTextChanged.connect(
                lambda _t, r=row: self._on_baseline_changed(r))
            self._proj_table.setCellWidget(row, 3, combo)
            self._proj_table.setItem(row, 4, _ro(""))
            self._on_baseline_changed(row)
        self._fit_project_columns()
        self._refresh_plot_choices()

    def _on_baseline_changed(self, row):
        name = self._proj_table.item(row, 1).text()
        combo = self._proj_table.cellWidget(row, 3)
        iter_name = combo.currentText() if combo else ""
        base = self._read_baseline(name, iter_name)
        self._baselines[name] = base
        self._proj_table.setItem(
            row, 4, _ro(str(len(base.params)) if base.params else "-"))
        self._refresh_plot_choices()

    @staticmethod
    def _read_baseline(project_name, iter_name):
        from gui.shared_widgets import get_analysis_dir
        if not iter_name or iter_name.startswith("("):
            return Baseline()
        return read_baseline(os.path.join(get_analysis_dir(),
                                          project_name, iter_name))

    def _fit_project_columns(self):
        from PySide6.QtGui import QFontMetrics
        if self._proj_table.rowCount() == 0:
            return
        fm = QFontMetrics(self._proj_table.font())
        names = [self._proj_table.item(r, 1).text()
                 for r in range(self._proj_table.rowCount())] or ["Project"]
        if self._col_width is None:
            probe = QComboBox(self._proj_table)
            probe.setFont(self._proj_table.font())
            probe.addItem("iter_001_-10.00V_CO")
            self._col_width = probe.sizeHint().width() + 8
            probe.setParent(None)
            probe.deleteLater()
        widths = [
            max(34, fm.horizontalAdvance("Use") + 16),
            max(fm.horizontalAdvance(max(names, key=len)) + 18,
                fm.horizontalAdvance("Project") + 18),
            fm.horizontalAdvance("reference") + 18,
            max(self._col_width,
                fm.horizontalAdvance("Baseline iteration") + 18),
            fm.horizontalAdvance("Runs") + 18,
        ]
        for c, w in enumerate(widths):
            self._proj_table.setColumnWidth(c, w)
        left = self._main_splitter.widget(0)
        if left is not None:
            left.setMinimumWidth(max(480, sum(widths) + 60))

    # ── setup ────────────────────────────────────────────────
    def is_calibration_mode(self):
        return self._mode.currentText() == MODE_CALIBRATION

    def _apply_mode_labels(self):
        cal = self.is_calibration_mode()
        self._lbl_from.setText("zero crossings" if cal else "from")
        self._lbl_to.setText("and" if cal else "to")
        self._lbl_v.setText("V" if cal else "V,")
        self._steps.setVisible(not cal)
        self._lbl_points.setVisible(not cal)
        self._lbl_base.setText("common offset (quoted)" if cal
                               else "baseline at")
        self._update_offsets_label()

    def _on_mode_changed(self, _text=None):
        """The user switched modes (or the tab was just built)."""
        self._apply_mode_labels()
        if self._loading:
            return
        if self.is_calibration_mode():
            # The usual rule: the larger deviation of the two zero
            # crossings from the common offset, quoted as +/-.
            self._definition.setCurrentText(MAX_DEV)
        if hasattr(self, "_sum_table"):
            self.compute()

    def offsets(self):
        if self.is_calibration_mode():
            # The common offset and the two crossings, once each.
            out = []
            for v in sorted((self._dv_min.value(), self._baseline_dv.value(),
                             self._dv_max.value())):
                if not out or abs(v - out[-1]) > SAME_DV:
                    out.append(float(v))
            return out
        try:
            return scan_offsets(self._dv_min.value(), self._dv_max.value(),
                                self._steps.value())
        except ValueError:
            return []

    def _update_offsets_label(self):
        offs = self.offsets()
        if not offs:
            self._offsets_label.setText("(range is empty)")
            return
        if self.is_calibration_mode():
            lo = min(self._dv_min.value(), self._dv_max.value())
            hi = max(self._dv_min.value(), self._dv_max.value())
            common = self._baseline_dv.value()
            text = (f"every run at {common:+.2f} V (quoted) and at "
                    f"{lo:+.2f} / {hi:+.2f} V (systematic): "
                    f"{len(offs)} fits per run")
            if not lo - SAME_DV <= common <= hi + SAME_DV:
                text += " -- the common offset is OUTSIDE the range"
            self._offsets_label.setText(text)
            return
        step = (offs[1] - offs[0]) if len(offs) > 1 else 0.0
        self._offsets_label.setText(
            f"{offs[0]:+.2f} .. {offs[-1]:+.2f} V, {len(offs)} points, "
            f"{step:.2f} V apart")

    def _range_from_calibration(self):
        cal = getattr(self._analysis_tab, "_cal_tab", None)
        res = getattr(cal, "_result", None) if cal is not None else None
        common = (cal.common() if cal is not None
                  and hasattr(cal, "common") else None)
        lo = common.lo if common is not None else None
        hi = common.hi if common is not None else None
        if lo is None and res is not None and getattr(res, "scan_range",
                                                      None):
            lo, hi = res.scan_lo, res.scan_hi
        if lo is None:
            QMessageBox.information(
                self, "Systematics",
                "No calibration in this session yet. Run the Cooler "
                "Calibration tab first, or type the offsets in.")
            return
        self._dv_min.setValue(float(lo))
        self._dv_max.setValue(float(hi))
        if common is not None and common.value is not None:
            self._baseline_dv.setValue(float(common.value))
        elif res is not None and res.dv is not None:
            self._baseline_dv.setValue(float(res.dv))
        self._update_offsets_label()

    def _selected_targets(self):
        out = []
        for r in range(self._proj_table.rowCount()):
            cb = self._proj_table.cellWidget(r, 0)
            if cb is None or not cb.isChecked():
                continue
            name = self._proj_table.item(r, 1).text()
            project = next((p for p in self.scannable_projects()
                            if p.project_name == name), None)
            if project is None:
                continue
            role = ("reference" if getattr(project, "is_reference", False)
                    else "sample")
            out.append(ScanTarget(project=project, role=role,
                                  baseline=self._baselines.get(
                                      name, Baseline())))
        return out

    def _gp_panel(self):
        if not self._use_gp.isChecked():
            return None
        panel = getattr(self._analysis_tab, "_gp_tab", None)
        if panel is None or not hasattr(panel, "start_gp_fit_for_scan"):
            return None
        if getattr(panel, "corrector", None) is None:
            return None
        return panel

    def _is_tab(self):
        if not self._use_is.isChecked():
            return None
        tab = getattr(self._analysis_tab, "_is_tab", None)
        if tab is None or not hasattr(tab, "shifts_for_scan"):
            return None
        return tab

    def _confirm_chain(self):
        """Say what the chain will NOT do before it runs.

        A ticked box that quietly does nothing is worse than an
        unticked one: a scan whose GP leg silently did not run reports
        a systematic on uncorrected centroids and looks identical to
        one that worked.
        """
        missing = []
        if self._use_gp.isChecked() and self._gp_panel() is None:
            missing.append(
                "- No GP is fitted, so the reference correction will "
                "NOT be refitted per offset and the samples will be "
                "fitted uncorrected.")
        if self._use_is.isChecked() and self._is_tab() is None:
            missing.append(
                "- The isotope-shift tab cannot be reached, so no "
                "shifts will be recorded.")
        if not missing:
            return True
        return QMessageBox.question(
            self, "Part of the chain will be skipped",
            "\n".join(missing) + "\n\nRun the scan anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes

    def _warn_about_drift(self, targets):
        """Say so when a project has moved on since its baseline."""
        lines = []
        for t in targets:
            snap = t.baseline.config
            if not snap:
                continue
            try:
                diff = config_differences(t.project.to_dict(), snap)
            except Exception:                            # noqa: BLE001
                continue
            if diff:
                lines.append(f"{t.name}: " + "; ".join(diff[:4])
                             + (" ..." if len(diff) > 4 else ""))
        if not lines:
            return True
        return QMessageBox.question(
            self, "Projects have changed since their baseline",
            "The scan fits with each project as it stands now, not as "
            "its baseline iteration had it:\n\n" + "\n".join(lines)
            + "\n\nRun anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes

    # ── running ──────────────────────────────────────────────
    def _start_scan(self, offsets=None, keep_records=False):
        if self._scan.busy:
            return
        targets = self._selected_targets()
        if not targets:
            QMessageBox.warning(self, "Systematics",
                                "Tick at least one project.")
            return
        offs = list(offsets) if offsets else self.offsets()
        if not offs:
            QMessageBox.warning(self, "Systematics",
                                "The offset range is empty.")
            return
        if not self._confirm_chain():
            return
        if not keep_records and not self._warn_about_drift(targets):
            return
        if not keep_records:
            self._records = {}
            self._excluded = set()
        self._targets_cache = targets
        self._busy(True)
        self._append_log(
            f"scan {self._label.text()}: {len(targets)} project(s) x "
            f"{len(offs)} offsets, seeds: "
            f"{self._seed_mode.currentText()}")
        try:
            self._scan.start(
                targets, offs,
                label=self._label.text().strip() or "sys_001",
                seed_mode=self._seed_mode.currentText(),
                baseline_dv=self._baseline_dv.value(),
                gp=self._gp_panel(), is_tab=self._is_tab(),
                light_output=self._light.isChecked(),
                records=dict(self._records) if keep_records else None,
                seed_overrides=self._seed_overrides,
                fitter_overrides=self._step_methods)
        except (ValueError, RuntimeError) as exc:
            self._busy(False)
            QMessageBox.warning(self, "Systematics", str(exc))

    def _rerun_selected(self):
        offs = self._ticked_steps()
        if not offs:
            QMessageBox.information(
                self, "Systematics",
                "Tick the steps to fit again in the Steps tab.")
            return
        self._start_scan(offsets=offs, keep_records=True)

    def _ticked_steps(self):
        out = []
        for r in range(self._step_table.rowCount()):
            cb = self._step_table.cellWidget(r, 0)
            item = self._step_table.item(r, 1)
            if cb is not None and cb.isChecked() and item is not None:
                try:
                    out.append(float(item.text()))
                except ValueError:
                    pass
        return out

    def _busy(self, on):
        self._run_btn.setEnabled(not on)
        self._rerun_btn.setEnabled(not on)
        self._stop_btn.setEnabled(on)
        self._progress.setVisible(on)

    def _on_progress(self, done, total, what):
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(done)
        self._status.setText(f"{done}/{total}  {what}")

    def _on_step_done(self, dv, record):
        self._records[float(dv)] = record
        missing = record.get("failed_projects") or []
        self._append_log(
            f"{dv:+.3f} V: {record.get('status', 'ok')}"
            + (f" -- no fit from {', '.join(missing)}" if missing else ""))
        self.compute()

    def _on_step_failed(self, dv, message):
        rec = self._scan.records.get(float(dv))
        if rec is not None:
            self._records[float(dv)] = rec
        self._append_log(f"{dv:+.3f} V: FAILED -- {message}")
        self.compute()

    def _on_finished(self, records):
        for rec in records:
            self._records[float(rec["dv"])] = rec
        self._busy(False)
        ok = sum(1 for r in self._records.values()
                 if r.get("status") == "ok")
        part = sum(1 for r in self._records.values()
                   if r.get("status") == "partial")
        bad = sum(1 for r in self._records.values()
                  if r.get("status") == "failed")
        self._status.setText(
            f"Scan finished: {len(self._records)} offsets, {ok} ok"
            + (f", {part} partial" if part else "")
            + (f", {bad} failed" if bad else ""))
        self.compute()

    def _on_failed(self, message):
        self._busy(False)
        self._status.setText(f"Scan aborted: {message}")
        self._append_log(f"ABORTED: {message}")
        QMessageBox.warning(self, "Systematics", message)

    def _append_log(self, text):
        self._log.appendPlainText(str(text))

    # ── what was measured ────────────────────────────────────
    def _keys(self):
        """Every (project, source, model, parameter) the scan saw."""
        keys = set()
        for rec in self._records.values():
            keys |= set(rec.get("values") or {})
        return sorted(keys)

    def _refresh_plot_choices(self):
        projects = sorted({split_key(k)[0] for k in self._keys()})
        if not projects:
            projects = [t.project_name for t in self.scannable_projects()]
        self._set_combo(self._plot_project, projects)
        self._refresh_param_choices()

    def _refresh_param_choices(self):
        project = self._plot_project.currentText()
        names = sorted({f"{split_key(k)[2]} / {split_key(k)[3]}"
                        for k in self._keys()
                        if split_key(k)[0] == project})
        if not names:
            base = self._baselines.get(project)
            if base is not None:
                names = sorted(f"{m} / {p}" for m, p in base.parameters())
        # The centroid is what a scan is nearly always about, and it
        # is not what an alphabetical list opens on.
        preferred = next((n for n in names
                          if n.rsplit("/", 1)[-1].strip() == "centroid"),
                         None)
        self._set_combo(self._plot_param, names, preferred)

    @staticmethod
    def _set_combo(combo, items, preferred=None):
        current = combo.currentText()
        blocked = combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        idx = combo.findText(current)
        if idx < 0 and preferred:
            idx = combo.findText(preferred)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(blocked)

    def _on_plot_project(self, _text):
        self._refresh_param_choices()
        self.compute()

    def _selected_key_parts(self):
        text = self._plot_param.currentText()
        if " / " not in text:
            return "", ""
        model, pname = text.split(" / ", 1)
        return model, pname

    def _band_for(self, project, source, model, pname):
        """One run's band, straight from the records."""
        key = value_key(project, source, model, pname)
        points = []
        for dv in sorted(self._records):
            rec = self._records[dv]
            v = (rec.get("values") or {}).get(key)
            if v is None:
                continue
            points.append(ScanValue(
                dv=float(dv), value=float(v.get("value", float("nan"))),
                sigma=float(v.get("sigma", float("nan"))),
                include=float(dv) not in self._excluded))
        base = self._baselines.get(project, Baseline())
        if self.is_calibration_mode():
            # The quoted value is this batch's own fit at the common
            # offset: same project state, same seeds as the two edge
            # fits it is compared with. An iteration fitted earlier --
            # maybe at another offset -- would be compared with fits it
            # never shared a frame with. No usable fit there, no value.
            want = self._baseline_dv.value()
            here = [p for p in points
                    if abs(p.dv - want) <= SAME_DV and p.usable]
            value = here[0].value if here else float("nan")
            sigma = here[0].sigma if here else float("nan")
            return ParameterBand(
                project=project, run=base.run_of_source.get(source, source),
                parameter=pname, points=points, baseline=float(value),
                baseline_sigma=float(sigma))
        value = base.value(source, model, pname)
        sigma = base.sigma(source, model, pname)
        if value is None and points:
            # Nothing in any iteration knows this quantity: the
            # isotope shifts are computed from the fits, not fitted.
            # Its baseline is the step at the calibrated offset.
            # Among the points the band uses: an excluded step (the
            # natural response to a partial one) was the baseline every
            # deviation was measured from (review 3, 2026-10-08).
            want = self._baseline_dv.value()
            usable = [p for p in points if p.usable] or points
            near = min(usable, key=lambda p: abs(p.dv - want))
            value, sigma = near.value, near.sigma
        return ParameterBand(
            project=project, run=base.run_of_source.get(source, source),
            parameter=pname, points=points,
            baseline=float(value if value is not None else float("nan")),
            baseline_sigma=float(sigma))

    def _sources_for(self, project):
        return sorted({split_key(k)[1] for k in self._keys()
                       if split_key(k)[0] == project})

    def bands(self, project=None, model=None, pname=None):
        """Per-run bands plus their weighted mean."""
        project = project or self._plot_project.currentText()
        if model is None or pname is None:
            model, pname = self._selected_key_parts()
        if not (project and model and pname):
            return [], None
        runs = [self._band_for(project, src, model, pname)
                for src in self._sources_for(project)]
        runs = [b for b in runs if b.points]
        if project == SystematicScan.SHIFT_PROJECT:
            # One band per isotope, the reference among them: a weighted
            # mean across isotopes is no physical quantity (review 3).
            return runs, None
        return runs, mean_over_runs(runs, project, pname,
                                    birge=self._birge.isChecked())

    # ── display ──────────────────────────────────────────────
    def compute(self, *_a):
        self._refresh_param_choices()
        self._populate_steps()
        self._populate_summary()
        self._populate_grid()
        self._render()
        self._send_btn.setEnabled(bool(self._records))

    def plot_data(self):
        runs, mean = self.bands()
        model, pname = self._selected_key_parts()
        definition = self._definition.currentText()
        out = {
            "title": (f"{self._plot_project.currentText()} - {pname} "
                      f"vs assumed cooler offset"),
            "parameter": pname, "unit": "MHz",
            "delta": self._delta_cb.isChecked(),
            "runs": [], "mean": {},
            "baseline": float("nan"), "baseline_sigma": float("nan"),
            "band": None, "definition": definition,
            "systematic": float("nan"),
            "quoted_dv": (self._baseline_dv.value()
                          if self.is_calibration_mode() else None),
            "baseline_label": ("quoted value (fit at the common offset)"
                               if self.is_calibration_mode()
                               else "baseline fit"),
        }
        for b in runs:
            used = [p for p in b.points if p.usable]
            skipped = [p for p in b.points if not p.usable]
            out["runs"].append({
                "label": b.run or b.project,
                "dv": [p.dv for p in used],
                "value": [p.value for p in used],
                "sigma": [0.0 if not math.isfinite(p.sigma) else p.sigma
                          for p in used],
                "excluded_dv": [p.dv for p in skipped
                                if math.isfinite(p.value)],
                "excluded_value": [p.value for p in skipped
                                   if math.isfinite(p.value)],
            })
        if mean is not None and mean.n:
            out["mean"] = {
                "dv": [p.dv for p in mean.used],
                "value": [p.value for p in mean.used],
                "sigma": [0.0 if not math.isfinite(p.sigma) else p.sigma
                          for p in mean.used],
            }
            out["baseline"] = mean.baseline
            out["baseline_sigma"] = mean.baseline_sigma
            out["band"] = (mean.lo, mean.hi)
            out["systematic"] = mean.systematic(definition)
        return out

    def _render(self):
        data = self.plot_data()
        draw_systematics(self._figure, data)
        self._canvas.draw_idle()
        definition = self._definition.currentText()
        runs, mean = self.bands()
        if mean is None or not mean.n:
            self._headline.setText("--")
            return
        _model, pname = self._selected_key_parts()
        bud = self._budget(mean, run_label="<weighted mean>")
        other = (f" +/- {bud['other']:.3g} (other)"
                 if math.isfinite(bud["other"]) else "")
        lo_dev, hi_dev = mean.edge_deviations
        self._headline.setText(
            f"{pname}: {mean.baseline:.4g} +/- {mean.baseline_sigma:.3g} "
            f"(stat) +/- {bud['cooler']:.3g} (cooler V){other} MHz\n"
            f"systematic {bud['systematic']:.3g}, total "
            f"{bud['total']:.3g}; edges {lo_dev:+.3g} / {hi_dev:+.3g}; "
            f"band {mean.lo:.4g} .. {mean.hi:.4g} over {mean.n} offsets; "
            f"sensitivity {mean.slope:.3g} MHz/V")

    def _budget(self, band, run_label=None):
        return budget(band, self._definition.currentText(),
                      self.extras(), self._combine_sys.currentText(),
                      self._combine_total.currentText(),
                      run_label=run_label)

    def _populate_steps(self):
        project = self._plot_project.currentText()
        model, pname = self._selected_key_parts()
        runs, mean = self.bands(project, model, pname)
        flagged = {}
        for b in runs:
            for i in trend_outliers(b):
                flagged[b.points[i].dv] = "off trend"
        by_dv = {p.dv: p for p in (mean.points if mean else [])}
        ticked = set(self._ticked_steps())

        _clear_cell_widgets(self._step_table)
        self._step_table.setRowCount(len(self._records))
        for row, dv in enumerate(sorted(self._records)):
            rec = self._records[dv]
            cb = QCheckBox()
            cb.setChecked(dv in ticked)
            self._step_table.setCellWidget(row, 0, cb)
            self._step_table.setItem(row, 1, _ro(f"{dv:g}"))
            status = rec.get("status", "?")
            role = self._step_role(dv)
            if role:
                status += f" · {role}"
            if dv in self._excluded:
                status += " (excluded)"
            self._step_table.setItem(row, 2, _ro(status))
            point = by_dv.get(dv)
            # Judge the number the row actually shows: the weighted
            # mean at this offset against the baseline's.
            flag = flagged.get(dv, "")
            if not flag:
                flag = quality_flag(
                    success=rec.get("status") != "failed",
                    sigma=point.sigma if point else float("nan"),
                    baseline_sigma=(mean.baseline_sigma if mean
                                    else float("nan")))
            if not flag and rec.get("failed_projects"):
                flag = "no " + ", ".join(rec["failed_projects"])
            self._step_table.setItem(row, 3, _ro(flag))
            self._step_table.setItem(
                row, 4, _ro("" if point is None else f"{point.value:.6g}"))
            self._step_table.setItem(
                row, 5, _ro("" if point is None else f"{point.sigma:.3g}"))
            if rec.get("status") == "failed" or flag:
                for c in range(1, len(STEP_COLS)):
                    item = self._step_table.item(row, c)
                    if item is not None:
                        item.setForeground(Qt.GlobalColor.red)

    def _toggle_excluded(self, row, _col):
        item = self._step_table.item(row, 1)
        if item is None:
            return
        try:
            dv = float(item.text())
        except ValueError:
            return
        if dv in self._excluded:
            self._excluded.discard(dv)
        else:
            self._excluded.add(dv)
        self.compute()

    def summary_rows(self):
        definition = self._definition.currentText()
        rows = []
        for project in sorted({split_key(k)[0] for k in self._keys()}):
            params = sorted({(split_key(k)[2], split_key(k)[3])
                             for k in self._keys()
                             if split_key(k)[0] == project})
            for model, pname in params:
                runs, mean = self.bands(project, model, pname)
                # Isotope shifts are per isotope by nature: always listed.
                kw = dict(extras=self.extras(),
                          combine_sys=self._combine_sys.currentText(),
                          combine_total=self._combine_total.currentText())
                if (self._per_run_cb.isChecked()
                        or project == SystematicScan.SHIFT_PROJECT):
                    rows.extend(summarise(runs, definition, **kw))
                if mean is not None:
                    rows.extend(summarise([mean], definition, **kw))
        return rows

    def _populate_summary(self):
        rows = self.summary_rows()
        self._sum_table.setRowCount(len(rows))

        def f(v, spec):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return ""
            return "" if not math.isfinite(v) else format(v, spec)

        for r, row in enumerate(rows):
            cells = [row["project"], row["run"], row["parameter"],
                     row["n"],
                     f(row["baseline"], ".6g"), f(row["stat"], ".3g"),
                     f(row["min"], ".6g"), f(row["max"], ".6g"),
                     f(row["band_width"], ".4g"),
                     f(row["edge_lo"], "+.4g"), f(row["edge_hi"], "+.4g"),
                     f(row["systematic"], ".4g"), f(row["other"], ".4g"),
                     f(row["sys_total"], ".4g"), f(row["total"], ".4g"),
                     f(row["slope"], ".4g")]
            for c, text in enumerate(cells):
                item = _ro(text)
                if SUMMARY_COLS[c] == "other sys" and row.get("extras"):
                    item.setToolTip(row["extras"])
                self._sum_table.setItem(r, c, item)

    def _populate_grid(self):
        project = self._plot_project.currentText()
        model, pname = self._selected_key_parts()
        runs, mean = self.bands(project, model, pname)
        offsets = sorted(self._records)
        self._grid_table.setColumnCount(1 + len(offsets))
        self._grid_table.setHorizontalHeaderLabels(
            ["Run"] + [f"{dv:g} V" for dv in offsets])
        bands = list(runs) + ([mean] if mean is not None else [])
        self._grid_table.setRowCount(len(bands))
        for r, b in enumerate(bands):
            self._grid_table.setItem(
                r, 0, _ro(b.run or "weighted mean"))
            by_dv = {p.dv: p for p in b.points}
            for c, dv in enumerate(offsets, start=1):
                p = by_dv.get(dv)
                self._grid_table.setItem(
                    r, c, _ro("" if p is None else f"{p.value:.6g}"))

    def _step_role(self, dv):
        """What a step is for in the three-offset batch, or ""."""
        if not self.is_calibration_mode():
            return ""
        if abs(dv - self._baseline_dv.value()) <= SAME_DV:
            return "quoted"
        lo = min(self._dv_min.value(), self._dv_max.value())
        hi = max(self._dv_min.value(), self._dv_max.value())
        if abs(dv - lo) <= SAME_DV:
            return "low edge"
        if abs(dv - hi) <= SAME_DV:
            return "high edge"
        return "not in the batch"

    # ── other systematics (Budget tab) ───────────────────────
    def extras(self):
        """The rows of the Budget table as :class:`ExtraSystematic`."""
        out = []
        t = self._extra_table
        for r in range(t.rowCount()):
            cells = [t.item(r, c) for c in range(len(EXTRA_COLS))]
            if any(c is None for c in cells):
                continue
            try:
                value = float(cells[5].text().strip())
            except ValueError:
                value = float("nan")
            out.append(ExtraSystematic(
                name=cells[1].text().strip() or f"row {r + 1}",
                value=value,
                project=cells[2].text().strip() or ALL,
                run=cells[3].text().strip() or ALL,
                parameter=cells[4].text().strip() or ALL,
                use=cells[0].checkState() == Qt.CheckState.Checked))
        return out

    def _extra_choices(self, column):
        """What a filter cell offers: the names the scan has seen."""
        keys = [split_key(k) for k in self._keys()]
        if column == 2:
            return sorted({k[0] for k in keys}
                          | {t.project_name
                             for t in self.scannable_projects()})
        if column == 3:
            runs = set()
            for rec in self._records.values():
                runs |= {str(v) for v in (rec.get("labels") or {}).values()}
            return sorted(runs | {"<weighted mean>"})
        if column == 4:
            return sorted({k[3] for k in keys})
        return []

    def _add_extra(self, entry=None):
        e = entry or ExtraSystematic(name="", value=0.0)
        t = self._extra_table
        blocked = t.blockSignals(True)
        r = t.rowCount()
        t.insertRow(r)
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemFlag.ItemIsUserCheckable
                     | Qt.ItemFlag.ItemIsEnabled
                     | Qt.ItemFlag.ItemIsSelectable)
        use.setCheckState(Qt.CheckState.Checked if e.use
                          else Qt.CheckState.Unchecked)
        t.setItem(r, 0, use)
        value = "" if not math.isfinite(e.value) else f"{e.value:g}"
        for c, text in ((1, e.name), (2, e.project), (3, e.run),
                        (4, e.parameter), (5, value)):
            t.setItem(r, c, QTableWidgetItem(str(text)))
        t.blockSignals(blocked)
        if entry is None:
            t.setCurrentCell(r, 1)
            t.editItem(t.item(r, 1))
        self._on_extra_changed()

    def _remove_extras(self):
        t = self._extra_table
        rows = sorted({i.row() for i in t.selectedIndexes()}, reverse=True)
        for r in rows:
            t.removeRow(r)
        if rows:
            self._on_extra_changed()

    def _on_extra_changed(self, item=None):
        if item is not None and item.column() == 5:
            # A value that is not a number takes no part; say so.
            try:
                float(item.text().strip())
                ok = True
            except ValueError:
                ok = False
            blocked = self._extra_table.blockSignals(True)
            item.setForeground(Qt.GlobalColor.red if not ok
                               else self._extra_table.palette().text())
            self._extra_table.blockSignals(blocked)
        if not self._loading:
            self.compute()

    # ── seeds ────────────────────────────────────────────────
    def _edit_seeds(self):
        offs = self._ticked_steps()
        if len(offs) != 1:
            QMessageBox.information(
                self, "Systematics",
                "Tick exactly one step in the Steps tab to edit its "
                "starting values.")
            return
        dv = offs[0]
        targets = self._selected_targets()
        if not targets:
            return
        project = (self._plot_project.currentText()
                   or targets[0].name)
        base = self._baselines.get(project, Baseline())
        dlg = SeedDialog(dv, project, base.sources, base,
                         self._seed_overrides.get(dv, {}),
                         self._step_methods.get(dv, ""), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if dlg.overrides:
            self._seed_overrides[dv] = dlg.overrides
        else:
            self._seed_overrides.pop(dv, None)
        if dlg.method:
            self._step_methods[dv] = dlg.method
        else:
            self._step_methods.pop(dv, None)
        self._append_log(
            f"{dv:+.3f} V: {len(dlg.overrides)} seed(s) set"
            + (f", method {dlg.method}" if dlg.method else ""))

    # ── Results ──────────────────────────────────────────────
    def _report_text(self):
        definition = self._definition.currentText()
        offs = sorted(self._records)
        lines = ["Cooler-voltage systematic", "=" * 40,
                 f"Label: {self._label.text()}",
                 f"Fitted at: {self._mode.currentText()}"]
        if self.is_calibration_mode():
            lo = min(self._dv_min.value(), self._dv_max.value())
            hi = max(self._dv_min.value(), self._dv_max.value())
            lines += [
                f"Common offset (values quoted): "
                f"{self._baseline_dv.value():+.3f} V",
                f"Zero crossings (systematic): {lo:+.3f} and {hi:+.3f} V"]
        else:
            lines += [
                f"Offsets: {min(offs):+.3f} .. {max(offs):+.3f} V in "
                f"{len(offs)} steps" if offs else "Offsets: none",
                f"Baseline offset: {self._baseline_dv.value():+.3f} V"]
        lines += [f"Seeds: {self._seed_mode.currentText()}",
                  f"Cooler-offset systematic: {definition}",
                  f"Systematics added: "
                  f"{self._combine_sys.currentText().lower()}; total with "
                  f"stat: {self._combine_total.currentText().lower()}",
                  "Run average's error: "
                  + ("scaled by sqrt(chi2/ndf) when above 1"
                     if self._birge.isChecked() else "inverse variance"),
                  ""]
        extras = [e for e in self.extras() if e.use]
        if extras:
            lines.append("Other systematics:")
            for e in extras:
                lines.append(
                    f"  {e.name}: {e.value:g} (project {e.project}, run "
                    f"{e.run}, parameter {e.parameter})")
            lines.append("")
        if self._excluded:
            lines.append("Excluded offsets: "
                         + ", ".join(f"{v:g}" for v in
                                     sorted(self._excluded)))
        failed = [dv for dv, r in self._records.items()
                  if r.get("status") == "failed"]
        if failed:
            lines.append("Failed steps: "
                         + ", ".join(f"{v:g}" for v in sorted(failed)))
        lines += ["", "This is the systematic from the cooler-voltage "
                  "CALIBRATION OFFSET. It is a different number from "
                  "the run-to-run voltage jitter reported by the "
                  "isotope-shift tab; both belong in a budget.", ""]
        for row in self.summary_rows():
            lines.append(
                f"{row['project']:<14} {row['run']:<16} "
                f"{row['parameter']:<10} {row['baseline']:>14.6g} "
                f"+/- {row['stat']:<10.3g} cooler {row['systematic']:<10.4g}"
                f"other {row['other']:<10.4g} sys {row['sys_total']:<10.4g}"
                f"tot {row['total']:<10.4g} edges {row['edge_lo']:+.4g} / "
                f"{row['edge_hi']:+.4g}  dP/dV {row['slope']:.4g}")
        return "\n".join(lines)

    def _send_to_results(self):
        if not self._records:
            return
        from gui.shared_widgets import get_analysis_dir
        base = get_analysis_dir()
        project_name = "Systematics"
        pdir = os.path.join(base, project_name)
        iter_num = 1
        if os.path.isdir(pdir):
            existing = sorted(d for d in os.listdir(pdir)
                              if os.path.isdir(os.path.join(pdir, d)))
            if existing:
                try:
                    iter_num = int(existing[-1].split("_")[-1]) + 1
                except ValueError:
                    iter_num = len(existing) + 1
        iter_name = f"iter_{iter_num:03d}"
        idir = os.path.join(pdir, iter_name)
        plots = os.path.join(idir, "plots")
        os.makedirs(plots, exist_ok=True)

        report = self._report_text()
        with open(os.path.join(idir, "fit_report.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(report)

        with open(os.path.join(idir, "systematics_summary.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            rows = self.summary_rows()
            w = csv.writer(fh)
            head = ["project", "run", "parameter", "unit", "n",
                    "baseline", "stat", "min", "max", "band_width",
                    "half_width", "std", "max_dev", "rms_dev",
                    "max_dev_flat", "edge_lo", "edge_hi",
                    "systematic", "other", "extras", "sys_total",
                    "total", "slope", "slope_sigma"]
            w.writerow(head)
            for row in rows:
                w.writerow([row.get(k, "") for k in head])

        with open(os.path.join(idir, "scan_points.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["cooler_offset_v", "status", "project", "source",
                        "run", "model", "parameter", "value", "sigma",
                        "included"])
            for dv in sorted(self._records):
                rec = self._records[dv]
                labels = rec.get("labels") or {}
                for key, v in (rec.get("values") or {}).items():
                    proj, src, model, pname = split_key(key)
                    w.writerow([
                        f"{dv:.6f}", rec.get("status", ""), proj, src,
                        labels.get(f"{proj}|{src}", ""), model, pname,
                        f"{v.get('value', float('nan')):.8g}",
                        f"{v.get('sigma', float('nan')):.6g}",
                        int(dv not in self._excluded)])

        png = os.path.join(plots, "systematics.png")
        try:
            self._figure.savefig(png, dpi=150, bbox_inches="tight")
        except Exception:                                # noqa: BLE001
            pass
        np.savez(os.path.join(plots, "systematics.npz"),
                 plot_type="systematics",
                 payload=np.array(self.plot_data(), dtype=object))

        results = [{
            "success": True, "run_number": "systematics",
            "run_file": "", "report": report,
            "params_df": {}, "metadata_df": {},
            "x": [], "y": [], "yerr": [], "y_fit": [],
            "x_smooth": [], "y_fit_smooth": [], "residuals": [],
            "diagnostics": {}, "fwhm": {}, "peak_positions": {},
            "fit_quality": {}, "run_metadata": {}, "harmonic": 0,
        }]
        self.results_ready.emit(
            project_name, results,
            {"report": True, "params_csv": False, "metadata_csv": False,
             "fit_plots": True, "iter_label": iter_name,
             "iter_name": iter_name})
        QMessageBox.information(
            self, "Systematics",
            f"Systematics written to:\n{idir}\n\nAlso sent to the "
            f"Results tab as '{project_name}'.")

    # ── save / load ──────────────────────────────────────────
    def ui_layout(self):
        from gui.ui_layout import collect
        return collect(main=getattr(self, "_main_splitter", None))

    def apply_ui_layout(self, d):
        from gui.ui_layout import restore
        restore(d, main=getattr(self, "_main_splitter", None))

    def to_dict(self):
        rows = []
        for r in range(self._proj_table.rowCount()):
            combo = self._proj_table.cellWidget(r, 3)
            rows.append({
                "project": self._proj_table.item(r, 1).text(),
                "use": self._proj_table.cellWidget(r, 0).isChecked(),
                "baseline": combo.currentText() if combo else "",
            })
        return {
            "offset_mode": self._mode.currentText(),
            "dv_min": self._dv_min.value(),
            "dv_max": self._dv_max.value(),
            "steps": self._steps.value(),
            "baseline_dv": self._baseline_dv.value(),
            "combine_sys": self._combine_sys.currentText(),
            "combine_total": self._combine_total.currentText(),
            "birge": self._birge.isChecked(),
            "extras": [e.to_dict() for e in self.extras()],
            "seed_mode": self._seed_mode.currentText(),
            "label": self._label.text(),
            "light": self._light.isChecked(),
            "use_gp": self._use_gp.isChecked(),
            "use_is": self._use_is.isChecked(),
            "definition": self._definition.currentText(),
            "rows": rows,
            "excluded": sorted(self._excluded),
            "seed_overrides": {f"{k:g}": v for k, v
                               in self._seed_overrides.items()},
            "step_methods": {f"{k:g}": v for k, v
                             in self._step_methods.items()},
            # The scan is the expensive part -- 250 fits is not
            # something to repeat because a session was reopened.
            "records": [self._records[dv] for dv in sorted(self._records)],
        }

    def from_dict(self, d):
        if not d:
            return
        self._loading = True
        try:
            self._from_dict(d)
        finally:
            self._loading = False
        self._apply_mode_labels()
        self.compute()

    def _from_dict(self, d):
        # A session from before the three-offset batch ran a range scan.
        mode = str(d.get("offset_mode", MODE_RANGE))
        self._mode.setCurrentText(mode if mode in OFFSET_MODES
                                  else MODE_RANGE)
        for combo, key in ((self._combine_sys, "combine_sys"),
                           (self._combine_total, "combine_total")):
            v = str(d.get(key, QUADRATURE))
            combo.setCurrentText(v if v in COMBINATIONS else QUADRATURE)
        self._birge.setChecked(bool(d.get("birge", False)))
        blocked = self._extra_table.blockSignals(True)
        self._extra_table.setRowCount(0)
        self._extra_table.blockSignals(blocked)
        for e in d.get("extras") or []:
            if isinstance(e, dict):
                self._add_extra(ExtraSystematic.from_dict(e))
        self._dv_min.setValue(float(d.get("dv_min", -10.0)))
        self._dv_max.setValue(float(d.get("dv_max", 10.0)))
        self._steps.setValue(int(d.get("steps", 10)))
        self._baseline_dv.setValue(float(d.get("baseline_dv", 0.0)))
        mode = str(d.get("seed_mode", SEED_CONTINUATION))
        if mode in SEED_MODES:
            self._seed_mode.setCurrentText(mode)
        self._label.setText(str(d.get("label", "sys_001")))
        self._light.setChecked(bool(d.get("light", True)))
        self._use_gp.setChecked(bool(d.get("use_gp", True)))
        self._use_is.setChecked(bool(d.get("use_is", True)))
        definition = str(d.get("definition", DEFAULT_DEFINITION))
        if definition in DEFINITIONS:
            self._definition.setCurrentText(definition)

        self.refresh_projects()
        wanted = {r.get("project"): r for r in (d.get("rows") or [])}
        for row in range(self._proj_table.rowCount()):
            name = self._proj_table.item(row, 1).text()
            saved = wanted.get(name)
            if not saved:
                continue
            self._proj_table.cellWidget(row, 0).setChecked(
                bool(saved.get("use", True)))
            combo = self._proj_table.cellWidget(row, 3)
            if combo is not None and saved.get("baseline"):
                idx = combo.findText(str(saved["baseline"]))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._on_baseline_changed(row)

        self._excluded = {float(v) for v in (d.get("excluded") or [])}
        self._seed_overrides = {float(k): v for k, v
                                in (d.get("seed_overrides") or {}).items()}
        self._step_methods = {float(k): v for k, v
                              in (d.get("step_methods") or {}).items()}
        self._records = {float(rec.get("dv", 0.0)): rec
                         for rec in (d.get("records") or [])}
        self.compute()
