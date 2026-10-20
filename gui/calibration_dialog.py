"""Voltage-calibration diagnostics: see the fit, fix it, share the fix.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Two dialogs, reachable from the file row's right-click menu in both
Pre-Analysis and Analysis:

``CalibrationDialog``
    One run. Readback-vs-set with the fitted polynomial on top, residuals
    underneath, and -- the part that actually decides anything -- what the
    choice costs in **MHz**. Volts of residual mean nothing to a
    spectroscopist; a 47 MHz centroid shift against a 300 MHz isotope
    shift means everything. Points are clickable, and the calibration can
    instead be borrowed from another run or typed in by hand.

``CalibrationOverviewDialog``
    Every loaded run at a glance, worst first. This is what makes the
    app's "change nothing by default, but flag it" policy practical: with
    a campaign of hundreds of runs, nobody is going to open a diagnostic
    for each one, so the triage has to come to them.

Both write to the ``gui.calibration`` registry, so a fix made in
Pre-Analysis is the fix the Analysis fit uses, and vice versa.

Depends on: gui.calibration, matplotlib, PySide6.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem,
    QToolButton, QToolTip, QVBoxLayout, QWidget,
)

from gui.calibration import (
    DEFAULT_ORDER,
    DEFAULT_VACC_DIV,
    FLAG_MIN_IMPACT_V,
    MAX_ORDER,
    CalibrationError,
    calibration_shift_v,
    canonical_path,
    describe_spec,
    diagnose,
    fit_calibration,
    get_registry,
    read_beam_header,
    read_cal_points,
    resolve_calibration,
    shift_mhz_over_scan,
    suspect_outliers,
)
from gui.dialog_style import make_plot_card, style_dialog
from gui.shared_widgets import _load_settings, _save_settings, lucide_icon

# Rejection rules, in the order the combo shows them.
_RULES: List[Tuple[str, str]] = [
    ("none", "Keep every point"),
    ("first_n", "Drop the first N points"),
    ("sigma", "Drop residual outliers (n·σ)"),
    ("manual", "Only the points I unticked"),
]


def _fmt_mhz(v: float) -> str:
    return f"{v:+.1f} MHz" if abs(v) >= 0.05 else "negligible"


#: Plot colours for the light and the dark canvas. Dark is the
#: Pre-Analysis dark canvas (black axes, white ink) so the two read as
#: one app; the order colour map starts higher there, because viridis'
#: first colours are near-black and vanish on black (Arda, 2026-10-09).
_PLOT_THEMES = {
    False: dict(fig="white", ax="white", fg="#202020", dim="#616161",
                grid="#c8c8c8", zero="#888888", file="#8a8a8a",
                yours="#1e88e5", band_file="#9e9e9e", band_file_a=0.15,
                band_yours="#1e88e5", band_yours_a=0.12,
                excluded="#e53935", ring="#ff6f00", cmap_lo=0.0),
    True: dict(fig="#0d0d0d", ax="black", fg="#f0f0f0", dim="#a8a8a8",
               grid="#3a3a3a", zero="#9a9a9a", file="#b8b8b8",
               yours="#4fc3f7", band_file="#bdbdbd", band_file_a=0.14,
               band_yours="#4fc3f7", band_yours_a=0.14,
               excluded="#ff5252", ring="#ffd54f", cmap_lo=0.22),
}

#: Settings key of the dialog's dark-plot switch. Unset, it follows the
#: Pre-Analysis switch.
DARK_SETTING = "calibration_dark_plots"

#: How close (in screen pixels) the pointer must be to a point for a
#: hover read-out or a click to reach it.
_HIT_PX = 8.0


def _coef_label(name: str, coeffs) -> str:
    if len(coeffs) > 1:
        return f"{name}: p0={coeffs[0]:+.3f} V, p1={coeffs[1]:.5f}"
    return name


def _residual_limit(values, sigma: float, extra=()) -> float:
    """Half-height of a residual panel: the points that count, with room.

    Only the points the line was fitted to set the scale. A dropped
    settling glitch sits ~30x further out, and letting it set the scale
    flattens everything else onto the zero line -- it is drawn clipped
    at the edge instead.
    """
    vals = np.abs(np.asarray(values, dtype=float))
    vals = vals[np.isfinite(vals)]
    top = float(vals.max()) if vals.size else 0.0
    for e in extra:
        if np.isfinite(e):
            top = max(top, float(e))
    top = max(top, 2.2 * sigma if np.isfinite(sigma) else 0.0, 1e-6)
    return 1.18 * top


class CalibrationDialog(QDialog):
    """Diagnose and override one run's voltage calibration.

    ``physics`` (mass_amu / laser_cm / harmonic / cooler_v) is optional: with
    it the dialog can quote the impact in MHz, without it it falls back to
    volts. Callers that have a Source block or the Pre-Analysis physics inputs
    should always pass it -- the MHz number is the whole point.

    ``peers`` is ``[(label, path), ...]`` of the other runs currently loaded,
    offered as borrow donors.
    """

    def __init__(self, path: str, *, run_label: str = "",
                 cal_order: int = DEFAULT_ORDER,
                 physics: Optional[dict] = None,
                 peers: Optional[Sequence[Tuple[str, str]]] = None,
                 parent=None):
        super().__init__(parent)
        self._path = path
        self._physics = physics or {}
        self._peers = [p for p in (peers or []) if canonical_path(p[1])
                       != canonical_path(path)]
        self._base_order = int(cal_order)
        self._reg = get_registry()

        run_label = run_label or os.path.basename(path)
        self.setWindowTitle(f"Voltage calibration — {run_label}")
        self.resize(1020, 860)

        try:
            self._set_v, self._read_raw = read_cal_points(path)
        except Exception as exc:
            self._set_v = np.array([])
            self._read_raw = np.array([])
            QMessageBox.warning(
                self, "Calibration",
                f"Could not read the calibration table from\n{path}\n\n{exc}")

        self._n = int(len(self._set_v))
        # Hand-picked exclusions. Seeded from the saved spec so reopening the
        # dialog shows what is actually in force, not a blank slate.
        self._excluded: set[int] = set()
        self._result = None
        self._building = True
        self._error = ""
        # Plot state: the point the table has selected (ringed in every
        # panel), what each panel drew where (for hover and click), and
        # the secondary MHz axes to take down before a redraw.
        self._highlight: Optional[int] = None
        self._hit: Dict[object, Tuple[np.ndarray, np.ndarray,
                                      np.ndarray]] = {}
        self._sec_axes: list = []
        self._tip_idx: Optional[int] = None
        self._dark = self._dark_default()

        # The untouched calibration -- what the file gives you with no override
        # at all. Everything the dialog says is measured against this, and it
        # is drawn alongside the corrected one so "did my fix do anything?" is
        # answered by looking rather than by remembering.
        self._base_fit = None
        if self._n:
            try:
                self._base_fit = fit_calibration(
                    self._set_v, self._read_raw, self._base_order, (),
                    DEFAULT_VACC_DIV)
            except CalibrationError:
                pass
        # MHz per volt of calibrated-voltage error at this run's beam
        # energy, for the residual panels' second axis and the hover.
        self._mhz_per_v = self._sensitivity()

        self._build_ui(run_label)
        self._load_saved_spec()
        self._building = False
        self._on_mode_changed()   # settles the enabled states, then recomputes

    # ── UI ───────────────────────────────────────────────────

    def _build_ui(self, run_label: str):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavToolbar)
        from matplotlib.figure import Figure

        style_dialog(self)
        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        hdr = QLabel(
            f"<b>{run_label}</b> &nbsp;·&nbsp; {self._n} calibration points"
            f" &nbsp;·&nbsp; <span style='color:#8a8a90'>{self._path}</span>")
        hdr.setWordWrap(True)
        hdr.setToolTip(
            "The DAC→HV calibration table stored in this run's ASDF: for each\n"
            "voltage the scanner was *told* to produce (Set), the voltage that\n"
            "was actually *measured* (Readback). A polynomial through these\n"
            "pairs is what converts every event's DAC value into a real\n"
            "voltage — and hence into a frequency.")
        outer.addWidget(hdr)

        body = QHBoxLayout()
        body.setSpacing(8)
        outer.addLayout(body, 1)

        # ---- left: plots ----
        # Three panels, not two: the curve, the residuals you *would* have had,
        # and the residuals you have now. The two residual sets need their own
        # y-scales -- a settling glitch puts them ~30x apart, so sharing an axis
        # would flatten the corrected one into the zero line and hide exactly
        # the thing the user is trying to see.
        # constrained_layout, not a fixed left fraction: the old
        # left=0.16 was sized for the two-line y-labels at 7 inches
        # nominal, and a FRACTION does not shrink as the canvas
        # grows -- on a wide dialog it became a broad empty band
        # (2026-09-21).
        self._fig = Figure(figsize=(7.0, 7.4), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._fig)
        # No tooltip on the canvas itself: a paragraph of help popping
        # up over the data wherever the pointer rests hid the points it
        # explained. Hovering a POINT reads that point out instead, and
        # the help moved to the card title.
        self._ax_fit, self._ax_res0, self._ax_res1 = self._fig.subplots(
            3, 1, sharex=True, height_ratios=[2.2, 1.0, 1.0])
        self._axes = (self._ax_fit, self._ax_res0, self._ax_res1)
        _engine = self._fig.get_layout_engine()
        if _engine is not None:
            # rect + zero pads, or the engine adds its own band on top
            # of the space the labels need. Measured: the y-labels and
            # ticks want 97 px whatever the canvas is, so the plot
            # area goes from a flat 81% to 86% at 7 in and 95% at
            # 18 in -- the old fixed 16% was wasting 190 px on a wide
            # dialog. The panels share an x-axis, so they sit flush.
            _engine.set(w_pad=0.0, h_pad=0.0, hspace=0.02, wspace=0.0,
                        rect=(0, 0, 1, 1))
        # One colour bar for the acquisition order, shared by the three
        # panels: "dark points are early" was only ever explained in a
        # tooltip, and the legend's marker showed one arbitrary colour.
        self._cbar = None
        if self._n > 1:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            self._order_sm = ScalarMappable(
                norm=Normalize(0, self._n - 1), cmap=self._order_cmap())
            self._cbar = self._fig.colorbar(
                self._order_sm, ax=list(self._axes), location="right",
                fraction=0.025, pad=0.01, aspect=60)
            self._cbar.set_ticks([0, self._n - 1])
            self._cbar.set_ticklabels(["first", "last"])
            self._cbar.set_label("order taken", fontsize=8, labelpad=-2)
            self._cbar.ax.tick_params(labelsize=7, length=2)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("figure_leave_event",
                                 lambda _e: self._hide_tip())

        self._toolbar = NavToolbar(self._canvas, self)
        self._dark_btn = QToolButton()
        self._dark_btn.setIcon(lucide_icon("sun-moon"))
        self._dark_btn.setCheckable(True)
        self._dark_btn.setAutoRaise(True)
        self._dark_btn.setFixedSize(24, 22)
        self._dark_btn.setChecked(self._dark)
        self._dark_btn.toggled.connect(self._on_dark_toggled)
        self._update_dark_tip()
        card = make_plot_card(
            "Calibration fit", self._canvas, self._toolbar, parent=self,
            subtitle="click a point to leave it out · hover for its numbers",
            extras=(self._dark_btn,))
        title = card.findChild(QLabel, "plotCardTitle")
        if title is not None:
            title.setToolTip(
                "Top: the measured calibration points and the polynomial\n"
                "through them.\n"
                "Middle: how far each point sits from the file's own line;\n"
                "your line is drawn there too once you change something.\n"
                "Bottom: how far each point sits from your line.\n\n"
                "Points are coloured by the order they were taken (the bar\n"
                "on the right), so a supply that was still settling at the\n"
                "start of the sweep shows up as early points off the line.\n"
                "A left-out point is a red cross; one beyond the panel's\n"
                "scale sits on its edge as a triangle.\n\n"
                "Click a point to leave it out of the fit or put it back.\n"
                "Pick a row in the table to ring its point.")
        body.addWidget(card, 3)

        # ---- right: controls ----
        right = QVBoxLayout()
        right.setSpacing(6)
        body.addLayout(right, 2)

        mode_box = QGroupBox("Where the calibration comes from")
        mode_lay = QVBoxLayout(mode_box)
        mode_lay.setSpacing(4)
        self._mode_group = QButtonGroup(self)

        self._rb_fit = QRadioButton("Fit this run's own points")
        self._rb_fit.setToolTip(
            "Fit a polynomial through this run's calibration table.\n"
            "This is the normal case — use it, and drop any bad points below.")
        self._rb_borrow = QRadioButton("Copy another run's calibration")
        self._rb_borrow.setToolTip(
            "Use a different run's coefficients for this one.\n\n"
            "Useful when this run's calibration table is unusable but a run\n"
            "taken close to it in time is fine — the DAC→HV hardware is the\n"
            "same, so its calibration still describes this beam.")
        self._rb_coeffs = QRadioButton("Type the coefficients in")
        self._rb_coeffs.setToolTip(
            "Enter the polynomial by hand, e.g. from an external calibration.")
        for i, rb in enumerate((self._rb_fit, self._rb_borrow,
                                self._rb_coeffs)):
            self._mode_group.addButton(rb, i)
            mode_lay.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
            if i == 0:
                # --- fit controls, indented under the radio ---
                form = QFormLayout()
                form.setContentsMargins(20, 2, 0, 6)
                self._order = QSpinBox()
                self._order.setRange(1, MAX_ORDER)
                self._order.setValue(self._base_order)
                self._order.setToolTip(
                    "Order of the polynomial fitted through the points.\n\n"
                    "1 — a straight line: a gain and an offset. Almost always\n"
                    "    the right answer; the DAC→HV chain is linear.\n"
                    "2 or 3 — only if the residuals show a real curve. A high\n"
                    "    order will happily bend itself through noise.")
                self._order.valueChanged.connect(self._recompute)
                form.addRow("Polynomial order:", self._order)

                self._rule = QComboBox()
                for key, label in _RULES:
                    self._rule.addItem(label, key)
                self._rule.setToolTip(
                    "Which points to leave OUT of the fit.\n\n"
                    "Keep every point — the default. Nothing is dropped.\n\n"
                    "Drop the first N — matches the physical failure: the HV\n"
                    "    supply is still settling at the start of the sweep, so\n"
                    "    the first few readbacks are simply wrong.\n\n"
                    "Drop residual outliers (n·σ) — a statistical cut. Leave\n"
                    "    'iterative' ticked: a single pass computes σ from data\n"
                    "    that still contains the outliers, so a cluster of bad\n"
                    "    points inflates σ enough to hide inside its own cut.\n\n"
                    "Only the points I unticked — your explicit list. The mode\n"
                    "    switches to this automatically the moment you click a\n"
                    "    point, so what was dropped is never ambiguous later.")
                self._rule.currentIndexChanged.connect(self._on_rule_changed)
                form.addRow("Drop points:", self._rule)

                self._drop_first = QSpinBox()
                self._drop_first.setRange(0, 999)
                self._drop_first.setValue(3)
                self._drop_first.setToolTip(
                    "How many points at the START of the calibration sweep to\n"
                    "discard — these are the dark-coloured ones on the plot.")
                self._drop_first.valueChanged.connect(self._recompute)
                form.addRow("How many:", self._drop_first)

                sig_row = QHBoxLayout()
                sig_row.setContentsMargins(0, 0, 0, 0)
                self._sigma = QDoubleSpinBox()
                self._sigma.setRange(0.5, 10.0)
                self._sigma.setSingleStep(0.5)
                self._sigma.setValue(3.0)
                self._sigma.setToolTip(
                    "Drop any point further than this many standard deviations\n"
                    "from the fitted line. 3 is a sane default; 2 reproduces\n"
                    "the cut clstools applies internally.")
                self._sigma.valueChanged.connect(self._recompute)
                sig_row.addWidget(self._sigma)
                self._iterative = QCheckBox("iterative (recommended)")
                self._iterative.setChecked(True)
                self._iterative.setToolTip(
                    "Re-fit after each cut, and judge the scatter robustly.\n\n"
                    "Leave this ON. Without it the cut is computed once, from\n"
                    "data that still contains the bad points — so a run with\n"
                    "three bad points typically catches only the worst one and\n"
                    "leaves the calibration still wrong.")
                self._iterative.toggled.connect(self._recompute)
                sig_row.addWidget(self._iterative)
                sig_holder = QWidget()
                sig_holder.setLayout(sig_row)
                form.addRow("Cut at:", sig_holder)

                self._ignore_intercept = QCheckBox(
                    "Force zero intercept (p₀ = 0)")
                self._ignore_intercept.setToolTip(
                    "Constrain the calibration offset p0 to exactly 0, so the\n"
                    "line passes through the origin and only the gain (p1, …)\n"
                    "is fitted.\n\n"
                    "Use this when the DAC→HV chain is known to have no DC\n"
                    "offset and you do not want a few volts of apparent offset\n"
                    "absorbing real curvature or a settling glitch. The offset\n"
                    "is dropped from the fit, not fitted and then zeroed, so\n"
                    "the slope is the true through-origin gain.")
                self._ignore_intercept.toggled.connect(self._recompute)
                form.addRow("Constrain:", self._ignore_intercept)

                mode_lay.addLayout(form)

            elif i == 1:
                brow = QHBoxLayout()
                brow.setContentsMargins(20, 2, 0, 6)
                self._donor = QComboBox()
                for label, p in self._peers:
                    self._donor.addItem(label, p)
                if not self._peers:
                    self._donor.addItem("(no other runs loaded)", None)
                self._donor.setToolTip(
                    "The run to take the calibration from. Pick one recorded\n"
                    "close in time — the hardware must not have changed in\n"
                    "between. You are warned if its sweep does not cover the\n"
                    "voltage range this run scans.")
                self._donor.currentIndexChanged.connect(self._recompute)
                brow.addWidget(self._donor, 1)
                browse = QPushButton("Browse…")
                browse.setToolTip("Pick any ASDF run on disk as the donor.")
                browse.clicked.connect(self._browse_donor)
                brow.addWidget(browse)
                mode_lay.addLayout(brow)

            else:
                crow = QHBoxLayout()
                crow.setContentsMargins(20, 2, 0, 6)
                self._coeff_edits = []
                for j in range(MAX_ORDER + 1):
                    e = QLineEdit()
                    e.setPlaceholderText(f"p{j}")
                    e.setToolTip(
                        "Calibrated voltage = p0 + p1·DV + p2·DV² + …\n\n"
                        "p0 is an offset in volts.\n"
                        "p1 is a gain — normally very close to 1.\n"
                        "Leave p2 / p3 empty for a straight line.")
                    e.editingFinished.connect(self._recompute)
                    crow.addWidget(e)
                    self._coeff_edits.append(e)
                mode_lay.addLayout(crow)

        self._rb_fit.setChecked(True)
        right.addWidget(mode_box)

        # ---- stats + impact ----
        result_box = QGroupBox("Resulting calibration")
        result_lay = QVBoxLayout(result_box)
        result_lay.setSpacing(5)

        self._stats = QLabel()
        self._stats.setWordWrap(True)
        self._stats.setTextFormat(Qt.TextFormat.RichText)
        self._stats.setToolTip(
            "The polynomial this dialog will apply, and how tightly the\n"
            "calibration points sit on it.\n\n"
            "σ is the typical distance of a point from the line, in volts.\n"
            "Smaller is better — it means the calibration describes the\n"
            "hardware well.")
        result_lay.addWidget(self._stats)

        self._impact = QLabel()
        self._impact.setWordWrap(True)
        self._impact.setTextFormat(Qt.TextFormat.RichText)
        # Scoped to this label by name. A bare property list applies to
        # the label AND everything styled through it -- its tooltip
        # included, which came out dark (Arda, 2026-10-09).
        self._impact.setObjectName("calImpactBox")
        self._impact.setStyleSheet(
            "QLabel#calImpactBox { padding:7px; border:1px solid #4a4a4f;"
            " border-radius:3px; background-color:#2a2a2e; }")
        self._impact.setToolTip(
            "The only number that decides anything.\n\n"
            "Volts of residual mean nothing on their own. This is what your\n"
            "choice does to the measured frequency — set it against the\n"
            "isotope shift you are trying to measure. If it says 'negligible',\n"
            "this run's calibration is not your problem.")
        result_lay.addWidget(self._impact)
        right.addWidget(result_box)

        # ---- point table ----
        self._table = QTableWidget(self._n, 4)
        self._table.setToolTip(
            "Every calibration point. Untick one to leave it out of the fit —\n"
            "the same as clicking it on the plot.\n\n"
            "'Residual' is how far that point sits from the line, in volts.")
        self._table.setAlternatingRowColors(True)
        self._table.setHorizontalHeaderLabels(
            ["Use", "#", "Set [V]", "Residual [V]"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self._checks: List[QCheckBox] = []
        for i in range(self._n):
            cb = QCheckBox()
            cb.setChecked(True)
            cb.toggled.connect(
                lambda _on, idx=i: self._on_check_toggled(idx))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hl.addWidget(cb)
            self._table.setCellWidget(i, 0, holder)
            self._checks.append(cb)
            self._table.setItem(i, 1, QTableWidgetItem(str(i)))
            self._table.setItem(
                i, 2, QTableWidgetItem(f"{self._set_v[i]:.2f}"))
            self._table.setItem(i, 3, QTableWidgetItem(""))
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.currentCellChanged.connect(
            lambda row, _c, _pr, _pc: self._set_highlight(row))
        right.addWidget(self._table, 1)

        # ---- buttons ----
        btns = QDialogButtonBox()
        self._apply_btn = btns.addButton(
            "Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        self._apply_btn.setToolTip(
            "Use this calibration for this run everywhere — the Pre-Analysis\n"
            "spectrum, the Analysis fit, merges and splits. Saved with the\n"
            "project.")
        self._apply_all_btn = btns.addButton(
            "Apply to other runs…", QDialogButtonBox.ButtonRole.ActionRole)
        self._apply_all_btn.setToolTip(
            "Give other loaded runs the same calibration. Useful when a whole\n"
            "shift of runs shares the same settling problem.")
        self._reset_btn = btns.addButton(
            "Reset to file default", QDialogButtonBox.ButtonRole.ResetRole)
        self._reset_btn.setToolTip(
            "Throw away the override and go back to fitting every point in\n"
            "the file, exactly as it was recorded.")
        btns.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._apply_btn.clicked.connect(self._apply)
        self._apply_all_btn.clicked.connect(self._apply_to_others)
        self._reset_btn.clicked.connect(self._reset)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    # ── state <-> UI ─────────────────────────────────────────

    def _load_saved_spec(self):
        """Seed the controls from whatever override is already in force."""
        spec = self._reg.get(self._path)
        if not spec:
            return
        mode = spec.get("mode")
        self._order.setValue(int(spec.get("order", self._base_order)))
        if mode == "borrow":
            self._rb_borrow.setChecked(True)
            idx = self._donor.findData(spec.get("donor"))
            if idx < 0:
                self._donor.addItem(
                    os.path.basename(spec["donor"]), spec["donor"])
                idx = self._donor.count() - 1
            self._donor.setCurrentIndex(idx)
        elif mode == "coeffs":
            self._rb_coeffs.setChecked(True)
            for e, c in zip(self._coeff_edits, spec.get("coeffs_v", [])):
                e.setText(f"{c:.8g}")
        else:
            self._rb_fit.setChecked(True)
            rule = spec.get("reject", "none")
            self._rule.setCurrentIndex(
                max(0, self._rule.findData(rule)))
            self._sigma.setValue(float(spec.get("sigma", 3.0)))
            self._iterative.setChecked(bool(spec.get("iterative", True)))
            self._drop_first.setValue(int(spec.get("drop_first", 0)))
            self._ignore_intercept.setChecked(
                bool(spec.get("ignore_intercept", False)))
            self._excluded = {int(i) for i in (spec.get("excluded") or ())
                              if 0 <= int(i) < self._n}

    def _current_spec(self) -> Optional[dict]:
        if self._rb_borrow.isChecked():
            donor = self._donor.currentData()
            if not donor:
                return None
            return {"mode": "borrow", "donor": donor,
                    "order": self._order.value()}

        if self._rb_coeffs.isChecked():
            coeffs = []
            for e in self._coeff_edits:
                txt = e.text().strip()
                if not txt:
                    break
                try:
                    coeffs.append(float(txt))
                except ValueError:
                    return None
            if len(coeffs) < 2:
                return None
            return {"mode": "coeffs", "coeffs_v": coeffs}

        rule = self._rule.currentData()
        return {
            "mode": "fit",
            "order": self._order.value(),
            "reject": rule,
            "sigma": self._sigma.value(),
            "iterative": self._iterative.isChecked(),
            "drop_first": self._drop_first.value(),
            "ignore_intercept": self._ignore_intercept.isChecked(),
            "excluded": sorted(self._excluded),
        }

    def _on_mode_changed(self):
        # Radio 'toggled' fires while _build_ui is still wiring widgets up,
        # before the later ones exist. The post-build call re-runs this once
        # everything is in place.
        if self._building:
            return
        is_fit = self._rb_fit.isChecked()
        self._order.setEnabled(is_fit or self._rb_borrow.isChecked())
        self._rule.setEnabled(is_fit)
        rule = self._rule.currentData()
        self._drop_first.setEnabled(is_fit and rule == "first_n")
        self._sigma.setEnabled(is_fit and rule == "sigma")
        self._iterative.setEnabled(is_fit and rule == "sigma")
        self._ignore_intercept.setEnabled(is_fit)
        self._donor.setEnabled(self._rb_borrow.isChecked())
        for e in self._coeff_edits:
            e.setEnabled(self._rb_coeffs.isChecked())
        self._table.setEnabled(is_fit)
        if not self._building:
            self._recompute()

    def _on_rule_changed(self):
        rule = self._rule.currentData()
        self._drop_first.setEnabled(rule == "first_n")
        self._sigma.setEnabled(rule == "sigma")
        self._iterative.setEnabled(rule == "sigma")
        if not self._building:
            self._recompute()

    def _materialize_to_manual(self):
        """Freeze the current effective exclusions into an explicit list.

        The moment the user touches a point by hand, the rule becomes their
        list. Anything else is ambiguous -- you cannot half-apply a σ cut and
        half-hand-pick and still say, six months later, which points were
        dropped and why.
        """
        if self._result is not None:
            self._excluded = set(self._result.excluded)
        self._rule.blockSignals(True)
        self._rule.setCurrentIndex(self._rule.findData("manual"))
        self._rule.blockSignals(False)
        self._on_rule_changed()

    def _toggle_point(self, idx: int):
        if not self._rb_fit.isChecked() or not (0 <= idx < self._n):
            return
        if self._rule.currentData() != "manual":
            self._materialize_to_manual()
        if idx in self._excluded:
            self._excluded.discard(idx)
        else:
            self._excluded.add(idx)
        self._recompute()

    def _point_at(self, event) -> Optional[int]:
        """The point under the pointer, or None.

        Looked up from where each panel actually drew its points (a
        clipped one sits on the panel's edge), nearest first. This
        replaced matplotlib pick events, whose indices count within
        each scatter: the top and bottom panels draw the used points
        and the left-out ones as separate scatters, so clicking there
        toggled whichever point had that index in the FULL table.
        """
        ax = getattr(event, "inaxes", None)
        hit = self._hit.get(ax)
        if hit is None or event.x is None or event.y is None:
            return None
        idx, xd, yd = hit
        if not len(idx):
            return None
        disp = ax.transData.transform(np.column_stack([xd, yd]))
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        j = int(np.argmin(d))
        return int(idx[j]) if d[j] <= _HIT_PX * self._px_scale() else None

    def _px_scale(self) -> float:
        try:
            return float(self._canvas.devicePixelRatioF()) or 1.0
        except Exception:                                 # noqa: BLE001
            return 1.0

    def _on_click(self, event):
        # A click that zooms or pans is not a vote on a point.
        if getattr(self._toolbar, "mode", ""):
            return
        if getattr(event, "button", None) != 1:
            return
        idx = self._point_at(event)
        if idx is None:
            return
        self._toggle_point(idx)
        self._table.selectRow(idx)
        item = self._table.item(idx, 1)
        if item is not None:
            self._table.scrollToItem(
                item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _on_motion(self, event):
        idx = self._point_at(event)
        if idx is None:
            self._hide_tip()
            return
        if idx == self._tip_idx:
            return
        self._tip_idx = idx
        from PySide6.QtGui import QCursor
        QToolTip.showText(QCursor.pos(), self._point_text(idx),
                          self._canvas)

    def _hide_tip(self):
        if self._tip_idx is not None:
            self._tip_idx = None
            QToolTip.hideText()

    def _point_text(self, i: int) -> str:
        """What the hover says about one calibration point."""
        fit = self._result.fit if self._result is not None else None
        if fit is not None and i < len(fit.read_v):
            sv, rv = float(fit.set_v[i]), float(fit.read_v[i])
        else:
            sv, rv = float(self._set_v[i]), float("nan")
        lines = [f"Point #{i}  (taken {i + 1} of {self._n})",
                 f"set {sv:.3f} V  ->  readback {rv:.4f} V"]
        k = self._mhz_per_v

        def _r(v):
            s = f"{v:+.4f} V"
            return s + (f"  (~{abs(v) * k:.2f} MHz)" if k else "")

        if self._base_fit is not None and i < len(self._base_fit.residuals_v):
            lines.append("from the file's line: "
                         + _r(float(self._base_fit.residuals_v[i])))
        if fit is not None and i < len(fit.residuals_v):
            lines.append("from your line: " + _r(float(fit.residuals_v[i])))
        used = (self._result is not None
                and i not in set(self._result.excluded))
        lines.append("used in your fit" if used
                     else "LEFT OUT of your fit")
        if self._rb_fit.isChecked():
            lines.append("click to " + ("leave it out" if used
                                        else "put it back"))
        return "\n".join(lines)

    def _set_highlight(self, row: int):
        new = row if 0 <= row < self._n else None
        if new == self._highlight:
            return
        self._highlight = new
        if not self._building:
            self._draw()

    # ── dark / light canvas ──────────────────────────────────

    @staticmethod
    def _dark_default() -> bool:
        try:
            s = _load_settings()
        except Exception:                                 # noqa: BLE001
            return False
        if DARK_SETTING in s:
            return bool(s.get(DARK_SETTING))
        return bool(s.get("pa_dark_plots", False))

    def _update_dark_tip(self):
        self._dark_btn.setToolTip(
            "Dark plot: ON -- click to go back to the white canvas"
            if self._dark else
            "Dark plot: OFF -- click for a black canvas")

    def _on_dark_toggled(self, on: bool):
        self._dark = bool(on)
        self._update_dark_tip()
        try:
            s = _load_settings()
            s[DARK_SETTING] = self._dark
            _save_settings(s)
        except Exception:                                 # noqa: BLE001
            pass
        if self._cbar is not None:
            self._order_sm.set_cmap(self._order_cmap())
            self._cbar.update_normal(self._order_sm)
        self._draw()

    def _theme(self) -> dict:
        return _PLOT_THEMES[bool(self._dark)]

    def _order_cmap(self):
        """viridis, starting higher on the dark canvas."""
        import matplotlib
        from matplotlib.colors import ListedColormap
        lo = _PLOT_THEMES[bool(getattr(self, "_dark", False))]["cmap_lo"]
        base = matplotlib.colormaps["viridis"]
        if lo <= 0:
            return base
        return ListedColormap(base(np.linspace(lo, 1.0, 256)),
                              name="viridis_on_black")

    def _sensitivity(self) -> Optional[float]:
        """|MHz| that one volt of calibrated-voltage error is worth,
        mid-scan, or None without the physics or a calibration."""
        p = self._physics
        base = self._base_fit
        if base is None or not self._n or not all(
                p.get(k) for k in ("mass_amu", "laser_cm", "cooler_v",
                                   "harmonic")):
            return None
        try:
            shifted = list(base.coeffs_v)
            shifted[0] = float(shifted[0]) + 1.0
            _dv, dnu = shift_mhz_over_scan(
                shifted, base.coeffs_v, float(self._set_v.min()),
                float(self._set_v.max()), cooler_v=float(p["cooler_v"]),
                laser_cm=float(p["laser_cm"]),
                mass_amu=float(p["mass_amu"]),
                harmonic=int(p["harmonic"]), samples=3)
            k = abs(float(dnu[1]))
        except Exception:                                 # noqa: BLE001
            return None
        return k if np.isfinite(k) and k > 0 else None

    def _on_check_toggled(self, idx: int):
        if self._building:
            return
        want_used = self._checks[idx].isChecked()
        is_used = self._result is None or idx not in self._result.excluded
        if want_used != is_used:
            self._toggle_point(idx)

    def _browse_donor(self):
        start = os.path.dirname(self._path)
        p, _ = QFileDialog.getOpenFileName(
            self, "Borrow calibration from…", start, "ASDF runs (*.asdf)")
        if not p:
            return
        if self._donor.findData(p) < 0:
            self._donor.addItem(os.path.basename(p), p)
        self._donor.setCurrentIndex(self._donor.findData(p))
        self._rb_borrow.setChecked(True)

    # ── recompute + draw ─────────────────────────────────────

    def _recompute(self):
        if self._building or self._n == 0:
            return
        spec = self._current_spec()
        cal_map = dict(self._reg.to_dict())
        key = canonical_path(self._path)
        if spec is None:
            cal_map.pop(key, None)
        else:
            cal_map[key] = spec

        try:
            self._result = resolve_calibration(
                self._path, cal_map, cal_order=self._order.value())
            self._error = ""
        except CalibrationError as exc:
            self._result = None
            self._error = str(exc)

        self._sync_table()
        self._draw()
        self._update_stats()
        self._apply_btn.setEnabled(self._result is not None)

    def _sync_table(self):
        excl = set(self._result.excluded) if self._result else set()
        resid = (self._result.fit.residuals_v
                 if self._result and self._result.fit is not None else None)
        for i in range(self._n):
            cb = self._checks[i]
            cb.blockSignals(True)
            cb.setChecked(i not in excl)
            cb.blockSignals(False)
            item = self._table.item(i, 3)
            if resid is not None and i < len(resid):
                item.setText(f"{resid[i]:+.4f}")
            else:
                item.setText("—")

    def _draw(self):
        th = self._theme()
        # clear() also drops an axis's secondary (MHz) axes.
        self._sec_axes = []
        for ax in self._axes:
            ax.clear()
        self._hit = {}

        if self._result is None or self._result.fit is None:
            self._ax_fit.text(0.5, 0.5, self._error or "no calibration table",
                              ha="center", va="center", color=th["excluded"],
                              transform=self._ax_fit.transAxes, wrap=True)
            self._style_axes(th)
            self._canvas.draw_idle()
            return

        fit = self._result.fit
        res = self._result
        base = self._base_fit
        used = np.asarray(fit.used_mask, dtype=bool)
        x, y, r = fit.set_v, fit.read_v, fit.residuals_v
        idx_all = np.arange(self._n)
        cmap = self._order_cmap()
        from matplotlib.colors import Normalize
        norm = Normalize(0, max(self._n - 1, 1))
        xs = np.linspace(float(x.min()), float(x.max()), 200)
        unchanged = (base is not None
                     and not res.excluded
                     and np.allclose(res.coeffs_v, base.coeffs_v,
                                     rtol=1e-12, atol=0.0))
        fs_title = 8

        # ── 1. the calibration curves, before and after ──
        ax = self._ax_fit
        ax.scatter(x[used], y[used], c=idx_all[used], cmap=cmap, norm=norm,
                   s=30, linewidths=0, zorder=3, label="calibration points")
        if (~used).any():
            ax.scatter(x[~used], y[~used], marker="x", s=60,
                       c=th["excluded"], zorder=4, label="left out")
        self._hit[ax] = (idx_all, np.asarray(x, float), np.asarray(y, float))
        # The two curves are genuinely close on an absolute scale -- a 1.3 V
        # disagreement across a 400 V sweep is a third of a percent, which is
        # invisible here and worth 50 MHz downstream. So label them with their
        # coefficients: that keeps them distinguishable when they overlap, and
        # the residual panels below are the magnified view.
        if base is not None:
            ax.plot(xs, base.predict_v(xs), "--", color=th["file"], lw=1.3,
                    zorder=2, label=_coef_label("file's own", base.coeffs_v))
        ax.plot(xs, res.predict_v(xs), "-", color=th["yours"], lw=1.6,
                zorder=3, label=_coef_label("yours", res.coeffs_v))
        ax.set_ylabel("Readback [V]", fontsize=9)
        ax.tick_params(labelsize=8, labelbottom=False)
        leg = ax.legend(fontsize=8, loc="upper left")
        # The scatter's legend marker takes the colour map's FIRST colour,
        # which read as "the early points" -- show a middle one.
        try:
            mid = cmap(0.5)
            h = leg.legend_handles[0]
            h.set_array(None)      # else the colour map wins at draw
            h.set_facecolor(mid)
            h.set_edgecolor(mid)
        except Exception:                                 # noqa: BLE001
            pass

        # ── 2. how far each point sits from the FILE's own calibration ──
        ax = self._ax_res0
        ax.axhline(0, color=th["zero"], lw=0.9, zorder=1)
        if base is not None:
            rb = np.asarray(base.residuals_v, dtype=float)
            sb = float(base.sigma_v)
            if sb > 0:
                ax.axhspan(-2 * sb, 2 * sb, color=th["band_file"],
                           alpha=th["band_file_a"], lw=0, zorder=0)
            # Your line, seen from the file's: where your calibration
            # moves the voltage, point by point along the sweep.
            diff = None
            if not unchanged:
                diff = res.predict_v(xs) - base.predict_v(xs)
                ax.plot(xs, diff, "-", color=th["yours"], lw=1.4, zorder=2,
                        label="your line")
            lim = _residual_limit(
                rb[used], sb,
                extra=() if diff is None else (float(np.abs(diff).max()),))
            self._scatter_residuals(ax, x, rb, used, cmap, norm, lim, th)
            ax.set_title(f"σ = {sb:.4f} V", loc="right", fontsize=fs_title,
                         color=th["dim"])
            if diff is not None:
                ax.legend(fontsize=7, loc="lower right")
        else:
            ax.text(0.5, 0.5, "no baseline fit", ha="center", va="center",
                    fontsize=8, color=th["dim"], transform=ax.transAxes)
        ax.set_title("from the file's own line", loc="left",
                     fontsize=fs_title, color=th["dim"])
        ax.set_ylabel("Residual [V]", fontsize=8)
        ax.tick_params(labelsize=8, labelbottom=False)

        # ── 3. how far each point sits from YOUR calibration ──
        ax = self._ax_res1
        ax.axhline(0, color=th["zero"], lw=0.9, zorder=1)
        s = float(fit.sigma_v)
        if s > 0:
            ax.axhspan(-2 * s, 2 * s, color=th["band_yours"],
                       alpha=th["band_yours_a"], lw=0, zorder=0)
        r = np.asarray(r, dtype=float)
        lim = _residual_limit(r[used], s)
        self._scatter_residuals(ax, x, r, used, cmap, norm, lim, th)
        # With no change the two panels are identical, which is confusing
        # to look at unless it is said out loud -- so say it.
        ax.set_title(f"σ = {s:.4f} V"
                     + ("  ·  same as above: nothing changed yet"
                        if unchanged else ""),
                     loc="right", fontsize=fs_title, color=th["dim"])
        ax.set_title("from your line", loc="left", fontsize=fs_title,
                     color=th["yours"])
        ax.set_xlabel("Voltage the scanner was set to [V]", fontsize=9)
        ax.set_ylabel("Residual [V]", fontsize=8)
        ax.tick_params(labelsize=8)

        # ── MHz on the right of the residual panels ──
        k = self._mhz_per_v
        if k:
            for ax in (self._ax_res0, self._ax_res1):
                sec = ax.secondary_yaxis(
                    "right", functions=(lambda v, k=k: v * k,
                                        lambda m, k=k: m / k))
                sec.set_ylabel("≈ MHz", fontsize=7)
                sec.tick_params(labelsize=7)
                self._sec_axes.append(sec)

        # ── the point the table has selected, ringed everywhere ──
        h = self._highlight
        if h is not None and h < self._n:
            for ax in self._axes:
                hit = self._hit.get(ax)
                if hit is None:
                    continue
                where = np.flatnonzero(hit[0] == h)
                if not len(where):
                    continue
                j = int(where[0])
                ax.scatter([hit[1][j]], [hit[2][j]], s=150,
                           facecolors="none", edgecolors=th["ring"],
                           linewidths=1.8, zorder=7)

        self._style_axes(th)
        self._canvas.draw_idle()

    def _scatter_residuals(self, ax, x, r, used, cmap, norm, lim, th):
        """One residual panel's points, scaled to the used ones.

        A point beyond the panel's scale is drawn on its edge as a
        triangle pointing the way it went, so it stays visible and
        clickable without squashing the rest.
        """
        idx_all = np.arange(len(r))
        inside = np.abs(r) <= lim
        y_drawn = np.where(inside, r, np.sign(r) * lim * 0.92)
        u = used & inside
        ax.scatter(x[u], r[u], c=idx_all[u], cmap=cmap, norm=norm, s=26,
                   linewidths=0, zorder=3)
        e = ~used & inside
        if e.any():
            ax.scatter(x[e], r[e], marker="x", s=54, c=th["excluded"],
                       zorder=4)
        for sign, marker in ((1, "^"), (-1, "v")):
            o = ~inside & (np.sign(r) == sign)
            if not o.any():
                continue
            colours = [th["excluded"] if not used[i] else cmap(norm(i))
                       for i in idx_all[o]]
            ax.scatter(x[o], y_drawn[o], marker=marker, s=46, c=colours,
                       linewidths=0, zorder=4)
        ax.set_ylim(-lim, lim)
        self._hit[ax] = (idx_all, np.asarray(x, float), y_drawn)

    def _style_axes(self, th):
        """Paint the figure for the current canvas (light or dark)."""
        self._fig.set_facecolor(th["fig"])
        axes = list(self._axes) + list(self._sec_axes)
        if self._cbar is not None:
            axes.append(self._cbar.ax)
        for ax in axes:
            ax.set_facecolor(th["ax"])
            for spine in ax.spines.values():
                spine.set_color(th["fg"])
            ax.tick_params(colors=th["fg"], which="both")
            ax.xaxis.label.set_color(th["fg"])
            ax.yaxis.label.set_color(th["fg"])
        for ax in self._axes:
            ax.grid(True, color=th["grid"], alpha=0.6 if self._dark else 0.5,
                    lw=0.6)
            leg = ax.get_legend()
            if leg is not None:
                frame = leg.get_frame()
                frame.set_facecolor(th["ax"])
                frame.set_edgecolor(th["dim"])
                for t in leg.get_texts():
                    t.set_color(th["fg"])
        if self._cbar is not None:
            self._cbar.outline.set_edgecolor(th["fg"])
            # Re-set on every draw: update_normal() on a light/dark
            # switch hands the bar back its numeric ticks (0 .. 140).
            self._cbar.set_ticks([0, self._n - 1], labels=["first", "last"])
            self._cbar.ax.yaxis.label.set_color(th["fg"])

    def _update_stats(self):
        if self._result is None:
            self._stats.setText(
                f"<span style='color:#c33'>{self._error}</span>")
            self._impact.setText("")
            return

        res = self._result
        fit = res.fit
        coef = " &nbsp; ".join(
            f"p{i} = {c:.6g}" for i, c in enumerate(res.coeffs_v))
        lines = [
            "<span style='color:#9a9aa0'>calibrated voltage "
            "= p0 + p1·DV + …</span>",
            f"<b>{coef}</b>",
        ]
        if fit is not None:
            if res.mode in ("borrow", "coeffs"):
                # Nothing was fitted to this run's points -- the coefficients
                # came from elsewhere. What the residuals show here is whether
                # they *describe* this run, which is a different question, and
                # saying "n of n points used" would misrepresent it.
                lines.append(
                    f"<b>{fit.n_points} points</b> compared against it "
                    f"&nbsp;·&nbsp; typical scatter σ = {fit.sigma_v:.4f} V "
                    f"&nbsp;·&nbsp; worst {fit.max_abs_v:.4f} V"
                    f"<br><span style='color:#9a9aa0'>Nothing was fitted to "
                    f"this run — the scatter just says whether the borrowed "
                    f"line still describes it.</span>")
            else:
                lines.append(
                    f"<b>{fit.n_used} of {fit.n_points} points used</b> "
                    f"&nbsp;·&nbsp; typical scatter σ = {fit.sigma_v:.4f} V "
                    f"&nbsp;·&nbsp; worst {fit.max_abs_v:.4f} V")
        if res.mode == "borrow":
            lines.append(self._borrow_note())
        self._stats.setText("<br>".join(lines))
        self._impact.setText(self._impact_text())

    def _borrow_note(self) -> str:
        """Warn when the donor's sweep doesn't cover this run's."""
        try:
            d_set, _ = read_cal_points(self._result.donor)
        except Exception:
            return ""
        if not len(d_set) or not self._n:
            return ""
        if (d_set.min() <= self._set_v.min()
                and d_set.max() >= self._set_v.max()):
            return (f"Borrowed from "
                    f"<b>{os.path.basename(self._result.donor)}</b>.")
        return (f"<span style='color:#fb8c00'>⚠ "
                f"{os.path.basename(self._result.donor)} was only calibrated "
                f"over {d_set.min():.0f}–{d_set.max():.0f} V, but this run "
                f"scans {self._set_v.min():.0f}–{self._set_v.max():.0f} V — "
                f"part of the range is extrapolated.</span>")

    def _impact_text(self) -> str:
        """What this choice costs, against the file's untouched calibration.

        The headline of the dialog. A residual in volts is not actionable; a
        centroid shift in MHz, next to the isotope shift being measured, is.
        """
        if self._result is None or not self._n:
            return ""
        base = self._base_fit
        if base is None:
            return ""

        dv_v = calibration_shift_v(
            self._result.coeffs_v, base.coeffs_v,
            float(self._set_v.min()), float(self._set_v.max()))

        if dv_v <= 0.0:
            return ("<b>What this changes</b><br>"
                    "<span style='color:#9a9aa0'>Nothing yet — this is the "
                    "file's own calibration. Exclude a point, or pick a rule, "
                    "to see the effect.</span>")

        head = ("<b>What this changes</b><br>"
                f"moves the voltage by up to <b>{dv_v:.3f} V</b> "
                f"across the sweep")

        p = self._physics
        if not all(p.get(k) for k in
                   ("mass_amu", "laser_cm", "cooler_v", "harmonic")):
            return head + ("<br><span style='color:#9a9aa0'>Set the mass, "
                           "laser and cooler voltage to see this in "
                           "MHz.</span>")

        try:
            _dv, dnu = shift_mhz_over_scan(
                self._result.coeffs_v, base.coeffs_v,
                float(self._set_v.min()), float(self._set_v.max()),
                cooler_v=float(p["cooler_v"]), laser_cm=float(p["laser_cm"]),
                mass_amu=float(p["mass_amu"]),
                harmonic=int(p["harmonic"]))
        except Exception:
            return head

        mid = float(dnu[len(dnu) // 2])
        tilt = float(dnu.max() - dnu.min())
        big = abs(mid) > 1.0 or tilt > 1.0
        colour = "#ffa726" if big else "#9a9aa0"
        return (
            head
            + f"<br><span style='color:{colour}'>which moves your measured "
              f"<b>centroid by {_fmt_mhz(mid)}</b></span>"
            + (f"<br><span style='color:{colour}'>…and <b>tilts</b> the "
               f"frequency axis by {tilt:.1f} MHz across the scan, so peaks "
               f"move by different amounts depending where they sit</span>"
               if tilt > 1.0 else ""))

    # ── actions ──────────────────────────────────────────────

    def _apply(self):
        spec = self._current_spec()
        if self._result is None:
            return
        self._reg.set(self._path, spec)
        self.accept()

    def _apply_to_others(self):
        """Push this calibration onto other runs.

        For a 'fit' spec this copies the *rule* (drop the first 3, say), which
        is what you want when a whole shift of runs shares a settling problem.
        Hand-picked indices are copied too, but they only mean the same thing
        if the tables line up -- so say so rather than pretending otherwise.
        """
        if self._result is None or not self._peers:
            QMessageBox.information(
                self, "Calibration",
                "No other runs are loaded to copy this calibration to.")
            return
        spec = self._current_spec()

        dlg = _ApplyToRunsDialog(self._peers, describe_spec(spec), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        targets = dlg.selected()
        if not targets:
            return

        # One signal for the whole batch: subscribers reload the runs they
        # touch, and firing per-file would reload each of them N times.
        self._reg.set_many({p: dict(spec) for p in targets})
        self._reg.set(self._path, spec)
        QMessageBox.information(
            self, "Calibration",
            f"Applied to {len(targets)} other run(s).")
        self.accept()

    def _reset(self):
        self._reg.clear(self._path)
        self.accept()


class _ApplyToRunsDialog(QDialog):
    """Tick the runs that should inherit this calibration."""

    def __init__(self, peers: Sequence[Tuple[str, str]], summary: str,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply calibration to other runs")
        self.resize(430, 470)
        style_dialog(self)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Apply <b>{summary}</b> to:"))

        self._boxes: List[Tuple[QCheckBox, str]] = []
        for label, path in peers:
            cb = QCheckBox(label)
            cb.setToolTip(path)
            lay.addWidget(cb)
            self._boxes.append((cb, path))
        lay.addStretch(1)

        note = QLabel(
            "<span style='color:#888'>A rule (e.g. “drop the first 3”) "
            "transfers cleanly. Hand-picked point numbers only mean the same "
            "thing if the runs' calibration tables line up.</span>")
        note.setWordWrap(True)
        lay.addWidget(note)

        row = QHBoxLayout()
        all_btn = QPushButton("Select all")
        all_btn.clicked.connect(
            lambda: [cb.setChecked(True) for cb, _ in self._boxes])
        row.addWidget(all_btn)
        none_btn = QPushButton("Select none")
        none_btn.clicked.connect(
            lambda: [cb.setChecked(False) for cb, _ in self._boxes])
        row.addWidget(none_btn)
        row.addStretch(1)
        lay.addLayout(row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def selected(self) -> List[str]:
        return [p for cb, p in self._boxes if cb.isChecked()]


class CalibrationOverviewDialog(QDialog):
    """Triage every loaded run's calibration at once, worst first.

    The app's default is to change nothing and flag instead, which is only
    workable if the flags find you. Opening a per-run diagnostic across a
    campaign of hundreds of runs is not something anyone will do, so the
    triage comes here: sort by what the calibration is actually costing, in
    MHz, and go fix the top of the list.
    """

    def __init__(self, runs: Sequence[Tuple[str, str]], *,
                 cal_order: int = DEFAULT_ORDER,
                 physics: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voltage calibration — overview")
        self.resize(920, 560)
        style_dialog(self)
        self._runs = list(runs)
        self._physics = physics or {}
        self._cal_order = cal_order

        lay = QVBoxLayout(self)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["", "Run", "Points", "σ [V]", "Cost", "Calibration"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch)
        self._table.cellDoubleClicked.connect(self._open_row)
        lay.addWidget(self._table, 1)

        hint = QLabel("<span style='color:#888'>Double-click a run to open "
                      "its calibration diagnostic.</span>")
        lay.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        exp = btns.addButton("Export…", QDialogButtonBox.ButtonRole.ActionRole)
        imp = btns.addButton("Import…", QDialogButtonBox.ButtonRole.ActionRole)
        exp.setToolTip(
            "Write the calibration overrides to a standalone YAML, so the same\n"
            "fixes can be reused in another project without redoing them.")
        imp.setToolTip(
            "Merge calibration overrides from a YAML (or from another\n"
            "project file) into this session.")
        exp.clicked.connect(self._export)
        imp.clicked.connect(self._import)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        lay.addWidget(btns)

        get_registry().calibrations_changed.connect(self._refresh)
        self._refresh()

    # ── import / export ──────────────────────────────────────

    def _export(self):
        """Write the overrides to a standalone YAML.

        The calibration of a run is a property of the file, but the registry
        is saved inside a project. Without this, a second project over the
        same data starts from scratch and quietly re-fits every bad run with
        its uncorrected calibration.
        """
        import yaml

        cal = get_registry().to_dict()
        if not cal:
            QMessageBox.information(
                self, "Export calibrations",
                "There are no calibration overrides to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export calibrations", "calibrations.yaml",
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        if not path.lower().endswith((".yaml", ".yml")):
            path += ".yaml"
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump({"calibrations": cal}, f, default_flow_style=False,
                          sort_keys=False, allow_unicode=True)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Export calibrations",
            f"Exported {len(cal)} calibration override(s) to\n{path}")

    def _import(self):
        """Merge overrides from a YAML (a calibration export, or a project).

        Merged rather than replaced: importing one campaign's fixes must not
        silently drop the ones already in force for another.
        """
        import yaml

        path, _ = QFileDialog.getOpenFileName(
            self, "Import calibrations", "",
            "YAML files (*.yaml *.yml)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        incoming = (raw or {}).get("calibrations") if isinstance(raw, dict) \
            else None
        if not isinstance(incoming, dict) or not incoming:
            QMessageBox.warning(
                self, "Import calibrations",
                f"No 'calibrations:' section found in\n{path}")
            return

        reg = get_registry()
        before = set(reg.to_dict())
        merged = reg.to_dict()
        merged.update(incoming)
        reg.from_dict(merged)

        # Report what actually landed, not what was in the file. `from_dict`
        # silently drops entries `normalize_spec` cannot make sense of (a
        # hand-edited YAML, a spec from a future version), and claiming to have
        # imported an override that was quietly discarded is exactly how a run
        # ends up fitted with the wrong calibration while the user believes it
        # was fixed.
        after = set(reg.to_dict())
        landed = after - before
        rejected = len(incoming) - len(
            [p for p in incoming if canonical_path(p) in after])

        known = {canonical_path(p) for _lbl, p in self._runs}
        hit = sum(1 for p in incoming if canonical_path(p) in known
                  and canonical_path(p) in after)

        msg = [f"Imported {len(landed)} override(s) from {len(incoming)} "
               f"entries in the file.",
               f"{hit} match a run loaded here; the rest are kept and will "
               f"apply when those runs are loaded."]
        if rejected:
            msg.append("")
            msg.append(f"⚠ {rejected} entr{'y was' if rejected == 1 else 'ies were'} "
                       f"not understood and had to be discarded.")
        QMessageBox.information(self, "Import calibrations", "\n".join(msg))

    def _refresh(self):
        cal_map = get_registry().to_dict()
        rows = []
        for label, path in self._runs:
            d = diagnose(path, cal_map, run_number=label,
                         cal_order=self._cal_order)
            rows.append((d, self._cost_mhz(d, cal_map)))

        # Worst first: that is the entire job of this table.
        rows.sort(key=lambda t: (-(t[1] if t[1] is not None else t[0].impact_v),
                                 t[0].run_number))

        self._table.setRowCount(len(rows))
        n_flagged = 0
        for r, (d, cost) in enumerate(rows):
            flag = "⚠" if d.flagged else ("✓" if not d.error else "!")
            if d.flagged:
                n_flagged += 1
            item = QTableWidgetItem(flag)
            item.setData(Qt.ItemDataRole.UserRole, d.path)
            self._table.setItem(r, 0, item)
            self._table.setItem(r, 1, QTableWidgetItem(d.run_number))
            # All points are in use unless an override says otherwise; the
            # suspects are what a fix *would* drop, not what has been dropped.
            pts = str(d.n_points)
            if d.suspect and not d.has_override:
                pts += f"  ({len(d.suspect)} suspect)"
            self._table.setItem(r, 2, QTableWidgetItem(pts))
            self._table.setItem(r, 3, QTableWidgetItem(
                "—" if np.isnan(d.sigma_v) else f"{d.sigma_v:.4f}"))
            if d.error:
                cost_txt = "error"
            elif cost is not None:
                cost_txt = _fmt_mhz(cost) if d.suspect else "—"
            else:
                cost_txt = (f"{d.impact_v:.3f} V" if d.suspect else "—")
            self._table.setItem(r, 4, QTableWidgetItem(cost_txt))
            self._table.setItem(r, 5, QTableWidgetItem(
                d.error or d.spec_summary))
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch)

        if n_flagged:
            self._summary.setText(
                f"<b style='color:#fb8c00'>{n_flagged} of {len(rows)} run(s) "
                f"have calibration outliers worth acting on.</b> Nothing has "
                f"been changed — open a run to decide.")
        else:
            self._summary.setText(
                f"<b>All {len(rows)} run(s) look fine.</b> No calibration "
                f"outlier would move a centroid by more than "
                f"{FLAG_MIN_IMPACT_V:.2f} V.")

    def _cost_mhz(self, d, cal_map) -> Optional[float]:
        """The flagged outliers' cost in MHz, if the physics is known.

        Beam energy and laser setpoint are read from **this** run's header
        rather than taken from the shared ``physics`` dict: they are per-run
        values, and a table that quoted every row's cost using the first run's
        cooler voltage would be confidently wrong for all the others. Mass and
        harmonic do come from the caller -- those belong to the analysis, not
        the file.
        """
        p = self._physics
        if (not d.suspect or d.error
                or not all(p.get(k) for k in ("mass_amu", "harmonic"))):
            return None
        cooler_v, laser_cm = read_beam_header(d.path)
        if not (cooler_v and laser_cm):
            return None
        try:
            set_v, read_raw = read_cal_points(d.path)
            cur = resolve_calibration(d.path, cal_map,
                                      cal_order=self._cal_order)
            trimmed = fit_calibration(set_v, read_raw, self._cal_order,
                                      d.suspect, DEFAULT_VACC_DIV)
            _dv, dnu = shift_mhz_over_scan(
                cur.coeffs_v, trimmed.coeffs_v,
                float(set_v.min()), float(set_v.max()),
                cooler_v=cooler_v, laser_cm=laser_cm,
                mass_amu=float(p["mass_amu"]), harmonic=int(p["harmonic"]))
            return float(dnu[len(dnu) // 2])
        except Exception:
            return None

    def _open_row(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        label = self._table.item(row, 1).text()
        # This run's own beam energy and laser, not the shared dict's.
        cooler_v, laser_cm = read_beam_header(path)
        phys = dict(self._physics)
        if cooler_v:
            phys["cooler_v"] = cooler_v
        if laser_cm:
            phys["laser_cm"] = laser_cm
        CalibrationDialog(
            path, run_label=label, cal_order=self._cal_order,
            physics=phys, peers=self._runs, parent=self).exec()
        self._refresh()

    def closeEvent(self, event):
        """Drop the registry subscription.

        Qt keeps this dialog alive through its parent after ``exec()`` returns,
        so a live connection would make every later calibration change re-run
        a full ``diagnose`` sweep over every run -- in a dead window nobody is
        looking at, once per stale dialog ever opened.
        """
        try:
            get_registry().calibrations_changed.disconnect(self._refresh)
        except (RuntimeError, TypeError):
            pass          # already gone, or never connected
        super().closeEvent(event)
