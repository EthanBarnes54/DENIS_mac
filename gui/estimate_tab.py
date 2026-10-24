"""Estimate tab GUI: configure runs, launch estimations, view spectra.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Builds the Estimate tab of the DENIS app. Provides the parameter
entry widgets (element/transition, beam, laser, scan, statistics),
the per-isotope/isomer configuration panels with Schmidt-moment
auto-fill, the run options, the Plots view that renders simulated
hyperfine spectra and a peak table on a matplotlib canvas, the
Voltage Range Estimator (a live setpoint map that writes the chosen
setpoint and sweep back into Laser and Scan), and the log.
Serialises all inputs to/from the project config dict used by the
estimation pipeline.

Depends on: gui.shared_widgets (spinbox factories, log highlighter,
plot editor, palettes), gui.analysis.helpers (scrollable info
dialog), cls_estimations.mass_lookup (element/mass tables),
cls_estimations.plotting (palettes, spectrum rendering),
cls_estimations.schmidt (shell-model magnetic moment), and, for the
estimator, cls_estimations.doppler / hfs_model / config_parser plus
cls_estimate's A/B and peak-voltage helpers (imported lazily); built
on PySide6 and matplotlib.
"""

import os
import re
import sys
import warnings

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QRadioButton, QButtonGroup, QPushButton,
    QPlainTextEdit, QScrollArea, QSplitter, QFileDialog, QMessageBox,
    QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget,
    QAbstractButton, QAbstractSpinBox, QApplication, QMenu, QWidgetAction,
)
from PySide6.QtCore import (
    Qt, Signal, QLocale, QTimer, QUrl, QCoreApplication, QEvent,
)
from PySide6.QtGui import QDesktopServices, QFont, QValidator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from gui.shared_widgets import (
    ScientificSpinBox, LogHighlighter, PlotEditorDialog,
    ORBITAL_NAMES, _make_double, _make_int, _make_sci,
    LEGEND_LOCATIONS, LINESTYLE_MAP, lucide_icon,
    _SpinBoxCommand, scroll_zoom,
)
from cls_estimations.mass_lookup import (
    ELEMENT_Z, Z_TO_ELEMENT, load_mass_table, get_mass,
)
from cls_estimations.plotting import PALETTES
from cls_estimations.constants import CM_TO_HZ
from cls_estimations.doppler import (
    generate_voltage_spectrum, laser_frequency_for_resonance,
    transition_frequency_MHz,
)
from cls_estimations.hfs_model import compute_hfs_lines

# Default baseline Optimise leaves beyond the outermost selected peaks (V),
# so each line's wings and some background are scanned; the gear changes it.
OPTIMISE_MARGIN_V = 100.0

# Monospace for the log and the estimator readout: a stylesheet, because
# the zoom stamps fonts straight onto widgets and would undo a setFont.
# Menlo first on macOS: a missing first family made Qt log a font-alias
# warning at every start (2026-10-24).
_MONO_QSS = ("QPlainTextEdit { font-family: "
             + ("'Menlo', " if sys.platform == "darwin" else "")
             + "'Consolas', 'Courier New', 'Liberation Mono', monospace; }")

# A figure drawn before its pane has a size (while a tab lays out) makes
# matplotlib warn on the terminal; the next draw fits, so say nothing.
warnings.filterwarnings("ignore", message="constrained_layout not applied",
                        category=UserWarning)


# code review 2026-06-02, estimate-peak-table-na-sort: sort numeric Peak
# List columns by the value in UserRole rather than display text, so
# "N/A" rows do not interleave with real peaks (sorted by -inf sentinel).
class _NumericTableItem(QTableWidgetItem):
    """Table item that sorts on its numeric UserRole value, not its text."""

    def __lt__(self, other):
        a = self.data(Qt.ItemDataRole.UserRole)
        b = other.data(Qt.ItemDataRole.UserRole)
        if a is None or b is None:
            return super().__lt__(other)
        return a < b


# ══════════════════════════════════════════════════════════════════
#  Schmidt Valence Widget
# ══════════════════════════════════════════════════════════════════
class SchmidtWidget(QWidget):
    """Collapsible Schmidt valence moment configuration.

    Emits ``moment_changed(float)`` whenever the inputs change AND
    the widget is enabled, so the parent IsotopePanel / IsomerPanel
    can auto-fill the ``μ`` field. Without this auto-fill, ticking
    "Use Schmidt moment" while leaving μ at 0 made the reference
    isotope's μ-scaling step divide by zero in the estimation
    pipeline.
    """

    moment_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.enable_check = QCheckBox("Use Schmidt moment")
        self.enable_check.setToolTip(
            "Calculate magnetic moment from the nuclear shell model "
            "instead of using the experimental value. The μ field "
            "is auto-filled with the unquenched Schmidt value "
            "(g_s_factor=1) on tick / orbital change so the "
            "estimation pipeline always sees a non-zero μ; the "
            "actual scaling uses this widget's spec + the run's "
            "configured g_s_factor.")
        layout.addWidget(self.enable_check)

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(15, 0, 0, 0)

        # Radio: single vs two-particle
        self.single_radio = QRadioButton("Single particle")
        self.two_radio = QRadioButton("Two particles")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.single_radio)
        self.mode_group.addButton(self.two_radio)
        self.single_radio.setChecked(True)

        radio_row = QHBoxLayout()
        radio_row.addWidget(self.single_radio)
        radio_row.addWidget(self.two_radio)
        body_layout.addLayout(radio_row)

        # Single particle fields
        self.single_widget = QWidget()
        sf = QFormLayout(self.single_widget)
        sf.setContentsMargins(0, 0, 0, 0)
        self.sp_type = QComboBox()
        self.sp_type.addItems(["proton", "neutron"])
        self.sp_type.setToolTip("Nucleon type: proton or neutron")
        sf.addRow("Type:", self.sp_type)
        self.sp_orbital = QComboBox()
        self.sp_orbital.addItems(ORBITAL_NAMES)
        self.sp_orbital.setToolTip("Nuclear shell model orbital (e.g., 1f7/2)")
        sf.addRow("Orbital:", self.sp_orbital)
        body_layout.addWidget(self.single_widget)

        # Two particle fields
        self.two_widget = QWidget()
        tf = QFormLayout(self.two_widget)
        tf.setContentsMargins(0, 0, 0, 0)
        self.tp_type1 = QComboBox()
        self.tp_type1.addItems(["proton", "neutron"])
        self.tp_type1.setToolTip("Type of the first unpaired nucleon")
        tf.addRow("Type 1:", self.tp_type1)
        self.tp_orbital1 = QComboBox()
        self.tp_orbital1.addItems(ORBITAL_NAMES)
        self.tp_orbital1.setToolTip("Orbital of the first unpaired nucleon")
        tf.addRow("Orbital 1:", self.tp_orbital1)
        self.tp_type2 = QComboBox()
        self.tp_type2.addItems(["proton", "neutron"])
        self.tp_type2.setToolTip("Type of the second unpaired nucleon")
        tf.addRow("Type 2:", self.tp_type2)
        self.tp_orbital2 = QComboBox()
        self.tp_orbital2.addItems(ORBITAL_NAMES)
        self.tp_orbital2.setToolTip("Orbital of the second unpaired nucleon")
        tf.addRow("Orbital 2:", self.tp_orbital2)
        self.tp_I = _make_double(0, 0, 20, 1, 0.5,
                                 tooltip="Total nuclear spin for the two-particle coupling")
        tf.addRow("Total I:", self.tp_I)
        body_layout.addWidget(self.two_widget)

        layout.addWidget(self.body)

        # Connect signals
        self.enable_check.toggled.connect(self.body.setVisible)
        self.single_radio.toggled.connect(self._on_mode_changed)
        # Auto-fill the parent's μ field whenever the user changes
        # the spec OR ticks "Use Schmidt moment". Skipped silently
        # when the toggle is off so the parent's manually-entered
        # μ value isn't clobbered.
        self.enable_check.toggled.connect(self._emit_moment)
        self.single_radio.toggled.connect(self._emit_moment)
        self.two_radio.toggled.connect(self._emit_moment)
        for combo in (self.sp_type, self.sp_orbital,
                       self.tp_type1, self.tp_orbital1,
                       self.tp_type2, self.tp_orbital2):
            combo.currentIndexChanged.connect(self._emit_moment)
        self.tp_I.valueChanged.connect(self._emit_moment)
        self.body.setVisible(False)
        self._on_mode_changed()

    def _on_mode_changed(self):
        single = self.single_radio.isChecked()
        self.single_widget.setVisible(single)
        self.two_widget.setVisible(not single)

    def _emit_moment(self):
        """Compute the Schmidt moment from the current spec and emit
        ``moment_changed`` so the parent panel can populate its μ
        field. No-op when the widget is disabled OR a config restore
        is in progress (``from_dict`` sets ``_restoring`` so the
        cascade of programmatic radio/combo updates doesn't trample
        the saved μ value before the restore finishes)."""
        if getattr(self, "_restoring", False):
            return
        if not self.enable_check.isChecked():
            return
        spec = self.to_dict()
        if spec is None:
            return
        try:
            from cls_estimations.schmidt import calculate_schmidt_moment
            mu, _desc = calculate_schmidt_moment(spec, g_s_factor=1.0)
        except Exception:
            # Bad spec (e.g. two-particle with I not yet set);
            # leave the μ field untouched until the user finishes.
            return
        if not np.isfinite(mu):
            return
        self.moment_changed.emit(float(mu))

    def to_dict(self):
        if not self.enable_check.isChecked():
            return None
        if self.single_radio.isChecked():
            return {
                "type": self.sp_type.currentText(),
                "orbital": self.sp_orbital.currentText(),
            }
        else:
            return {
                "type1": self.tp_type1.currentText(),
                "orbital1": self.tp_orbital1.currentText(),
                "type2": self.tp_type2.currentText(),
                "orbital2": self.tp_orbital2.currentText(),
                "I": self.tp_I.value(),
            }

    def from_dict(self, d):
        # Restore mode: suppress moment_changed until every input
        # has been applied. Otherwise the cascade of intermediate
        # programmatic updates would emit several wrong-μ values
        # and trample the saved value before the load completes.
        self._restoring = True
        try:
            if d is None:
                self.enable_check.setChecked(False)
                return
            self.enable_check.setChecked(True)
            if "type" in d and d["type"] is not None:
                # Single particle
                self.single_radio.setChecked(True)
                idx = self.sp_type.findText(d["type"])
                if idx >= 0:
                    self.sp_type.setCurrentIndex(idx)
                idx = self.sp_orbital.findText(d.get("orbital", ""))
                if idx >= 0:
                    self.sp_orbital.setCurrentIndex(idx)
            elif "type1" in d and d["type1"] is not None:
                # Two particle
                self.two_radio.setChecked(True)
                for attr, key in [("tp_type1", "type1"),
                                   ("tp_type2", "type2")]:
                    idx = getattr(self, attr).findText(d.get(key, ""))
                    if idx >= 0:
                        getattr(self, attr).setCurrentIndex(idx)
                for attr, key in [("tp_orbital1", "orbital1"),
                                   ("tp_orbital2", "orbital2")]:
                    idx = getattr(self, attr).findText(d.get(key, ""))
                    if idx >= 0:
                        getattr(self, attr).setCurrentIndex(idx)
                if "I" in d and d["I"] is not None:
                    self.tp_I.setValue(float(d["I"]))
        finally:
            self._restoring = False
        # Don't fire _emit_moment here -- the saved YAML's ``mu``
        # field was already restored by the parent panel before
        # this method ran. Auto-fill only when the user interacts.


def _Set_Spin_Undoable(spin, value, owner):
    """Set ``spin`` to ``value`` and record it on the global undo stack
    so Ctrl-Z reverts to the previous value. Skips the push (direct
    ``setValue``) when the value is unchanged, when ``owner`` is in a
    config-restore window (``owner._bypass_undo``), or when no undo
    stack is reachable from ``owner.window()`` (headless / test mode)."""
    old = float(spin.value())
    new = float(value)
    if old == new:
        return
    undo_stack = getattr(owner.window(), "_undo_stack", None)
    if getattr(owner, "_bypass_undo", False) or undo_stack is None:
        spin.setValue(new)
        return
    # Set the value first, then push the command: _SpinBoxCommand's
    # initial redo() is a no-op, so the value must already be set
    # before pushing for the undo stack to capture old -> new.
    spin.blockSignals(True)
    spin.setValue(new)
    spin.blockSignals(False)
    undo_stack.push(_SpinBoxCommand(spin, old, new))


# ══════════════════════════════════════════════════════════════════
#  Isomer Panel
# ══════════════════════════════════════════════════════════════════
class IsomerPanel(QWidget):
    """Isomer configuration sub-panel within an IsotopePanel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectrum_only = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 5, 0, 5)

        # Basic fields
        form = QFormLayout()
        self.iso_I = _make_double(0, 0, 20, 1, 0.5,
                                  tooltip="Nuclear spin of the isomeric state (half-integer allowed)")
        form.addRow("I:", self.iso_I)
        self.iso_mu = _make_double(0, -10, 10, 5, 0.1,
                                   tooltip="Magnetic moment in nuclear magnetons (overridden if Schmidt is used)")
        form.addRow("\u03bc (\u03bc\u2099):", self.iso_mu)
        self.iso_Q = _make_double(0, -5, 5, 5, 0.1,
                                  tooltip="Electric quadrupole moment in barns")
        form.addRow("Q (b):", self.iso_Q)
        self.iso_shift = _make_double(0, -1e6, 1e6, 2, 1.0,
                                      tooltip="Isomer shift in MHz, defined as (isomer centroid − reference centroid). "
                                              "Positive = isomer sits at higher frequency than the reference.")
        form.addRow("Isotope shift (MHz):", self.iso_shift)
        self.iso_peaks = _make_int(1, 1, 100,
                                   tooltip="Number of strongest HFS peaks to include in timing")
        form.addRow("Peaks to measure:", self.iso_peaks)
        self.iso_plot_with_gs = QCheckBox("Plot with ground state")
        self.iso_plot_with_gs.setChecked(True)
        self.iso_plot_with_gs.setToolTip(
            "If checked, plot the isomer spectrum overlaid with the ground state")
        form.addRow(self.iso_plot_with_gs)
        layout.addLayout(form)

        # Yield method (timing mode only)
        self.yield_group_box = QGroupBox("Yield method")
        yg = QVBoxLayout(self.yield_group_box)

        self.yield_ratio_radio = QRadioButton("Production ratio")
        self.yield_ratio_radio.setToolTip(
            "Fraction of total yield going to the ground state (0-1). Isomer gets the remainder.")
        self.yield_indep_radio = QRadioButton("Independent yield")
        self.yield_indep_radio.setToolTip(
            "Absolute isomer yield in ions/s, independent of ground state yield")
        self.yield_spin_radio = QRadioButton("Spin distribution \u03c3")
        self.yield_spin_radio.setToolTip(
            "Auto-calculate g.s./isomer ratio from spin distribution statistics")
        self.yield_mode_group = QButtonGroup(self)
        self.yield_mode_group.addButton(self.yield_ratio_radio)
        self.yield_mode_group.addButton(self.yield_indep_radio)
        self.yield_mode_group.addButton(self.yield_spin_radio)
        self.yield_ratio_radio.setChecked(True)

        # Ratio row
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self.yield_ratio_radio)
        self.yield_ratio_spin = _make_double(
            0.5, 0.0, 1.0, 3, 0.05,
            tooltip="Fraction going to ground state (0-1). E.g., 0.8 means 80% g.s., 20% isomer.")
        ratio_row.addWidget(self.yield_ratio_spin)
        yg.addLayout(ratio_row)

        # Independent yield row
        indep_row = QHBoxLayout()
        indep_row.addWidget(self.yield_indep_radio)
        self.yield_indep_spin = _make_sci(
            0.0, 0.0, 1e15,
            tooltip="Independent isomer yield in ions/s")
        indep_row.addWidget(self.yield_indep_spin)
        yg.addLayout(indep_row)

        # Spin distribution row
        spin_row = QHBoxLayout()
        spin_row.addWidget(self.yield_spin_radio)
        self.yield_spin_spin = _make_double(
            7.0, 0.1, 100, 1, 0.5,
            tooltip="Spin distribution width \u03c3 (typical: 5-10 for fission)")
        spin_row.addWidget(self.yield_spin_spin)
        yg.addLayout(spin_row)

        layout.addWidget(self.yield_group_box)

        # Connect yield radio to enable/disable spinboxes
        self.yield_ratio_radio.toggled.connect(
            lambda c: self.yield_ratio_spin.setEnabled(c))
        self.yield_indep_radio.toggled.connect(
            lambda c: self.yield_indep_spin.setEnabled(c))
        self.yield_spin_radio.toggled.connect(
            lambda c: self.yield_spin_spin.setEnabled(c))
        self.yield_indep_spin.setEnabled(False)
        self.yield_spin_spin.setEnabled(False)

        # HFS overrides
        self.hfs_check = QCheckBox("Override A/B constants")
        self.hfs_check.setToolTip(
            "Provide explicit HFS A/B constants instead of scaling from the reference isotope")
        layout.addWidget(self.hfs_check)

        self.hfs_widget = QWidget()
        hf = QFormLayout(self.hfs_widget)
        hf.setContentsMargins(15, 0, 0, 0)
        self.hfs_Al = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Magnetic dipole HFS constant A for lower level (MHz)")
        hf.addRow("A_lower (MHz):", self.hfs_Al)
        self.hfs_Bl = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Electric quadrupole HFS constant B for lower level (MHz)")
        hf.addRow("B_lower (MHz):", self.hfs_Bl)
        self.hfs_Au = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Magnetic dipole HFS constant A for upper level (MHz)")
        hf.addRow("A_upper (MHz):", self.hfs_Au)
        self.hfs_Bu = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Electric quadrupole HFS constant B for upper level (MHz)")
        hf.addRow("B_upper (MHz):", self.hfs_Bu)
        layout.addWidget(self.hfs_widget)
        self.hfs_check.toggled.connect(self.hfs_widget.setVisible)
        self.hfs_widget.setVisible(False)

        # Schmidt valence: auto-fill μ on enable / spec change so the
        # estimation pipeline always sees a non-zero reference μ when
        # the Schmidt model is selected. Routed through
        # _apply_schmidt_moment so the change is recorded on the global
        # undo stack and Ctrl-Z reverts to the pre-Schmidt value.
        self.schmidt = SchmidtWidget()
        self.schmidt.moment_changed.connect(self._apply_schmidt_moment)
        layout.addWidget(self.schmidt)

    def set_spectrum_only(self, spectrum_only):
        self._spectrum_only = spectrum_only
        self.yield_group_box.setVisible(not spectrum_only)
        self.iso_peaks.setEnabled(not spectrum_only)

    def to_dict(self):
        d = {
            "I": self.iso_I.value(),
            "mu": self.iso_mu.value(),
            "Q": self.iso_Q.value(),
            "isotope_shift_MHz": self.iso_shift.value(),
        }
        if not self._spectrum_only:
            d["peaks_to_measure"] = self.iso_peaks.value()
        d["plot_with_gs"] = self.iso_plot_with_gs.isChecked()

        # Yield method (only in timing mode)
        if not self._spectrum_only:
            if self.yield_ratio_radio.isChecked():
                d["production_ratio"] = self.yield_ratio_spin.value()
            elif self.yield_indep_radio.isChecked():
                d["yield_ions_per_sec"] = self.yield_indep_spin.value()
            elif self.yield_spin_radio.isChecked():
                d["spin_distribution_sigma"] = self.yield_spin_spin.value()

        # HFS overrides
        if self.hfs_check.isChecked():
            d["A_lower_MHz"] = self.hfs_Al.value()
            d["B_lower_MHz"] = self.hfs_Bl.value()
            d["A_upper_MHz"] = self.hfs_Au.value()
            d["B_upper_MHz"] = self.hfs_Bu.value()

        # Schmidt
        sv = self.schmidt.to_dict()
        if sv is not None:
            d["schmidt_valence"] = sv

        return d

    def from_dict(self, d):
        if d is None:
            return
        self.iso_I.setValue(float(d.get("I", 0)))
        self.iso_mu.setValue(float(d.get("mu", 0)))
        self.iso_Q.setValue(float(d.get("Q", 0)))
        self.iso_shift.setValue(float(d.get("isotope_shift_MHz", 0)))
        if "peaks_to_measure" in d and d["peaks_to_measure"] is not None:
            self.iso_peaks.setValue(int(d["peaks_to_measure"]))
        self.iso_plot_with_gs.setChecked(d.get("plot_with_gs", True))

        # Yield method
        if "production_ratio" in d and d["production_ratio"] is not None:
            self.yield_ratio_radio.setChecked(True)
            self.yield_ratio_spin.setValue(float(d["production_ratio"]))
        elif "yield_ions_per_sec" in d and d["yield_ions_per_sec"] is not None:
            self.yield_indep_radio.setChecked(True)
            self.yield_indep_spin.setValue(float(d["yield_ions_per_sec"]))
        elif "spin_distribution_sigma" in d and d["spin_distribution_sigma"] is not None:
            self.yield_spin_radio.setChecked(True)
            self.yield_spin_spin.setValue(float(d["spin_distribution_sigma"]))

        # HFS
        has_hfs = any(k in d for k in ("A_lower_MHz", "A_upper_MHz",
                                        "B_lower_MHz", "B_upper_MHz"))
        self.hfs_check.setChecked(has_hfs)
        if has_hfs:
            self.hfs_Al.setValue(float(d.get("A_lower_MHz", 0)))
            self.hfs_Bl.setValue(float(d.get("B_lower_MHz", 0)))
            self.hfs_Au.setValue(float(d.get("A_upper_MHz", 0)))
            self.hfs_Bu.setValue(float(d.get("B_upper_MHz", 0)))

        # Schmidt
        self.schmidt.from_dict(d.get("schmidt_valence"))
        # If the saved YAML pre-dates the auto-fill change and stored
        # μ=0 alongside an enabled Schmidt spec, populate μ now so
        # the estimation pipeline doesn't divide by zero. Custom
        # non-zero saved values are respected.
        # Replay one auto-fill through the no-undo path so old YAMLs
        # don't dirty the undo stack on load.
        self._bypass_undo = True
        try:
            if (self.schmidt.enable_check.isChecked()
                    and self.iso_mu.value() == 0):
                self.schmidt._emit_moment()
        finally:
            self._bypass_undo = False

    def _apply_schmidt_moment(self, value):
        """Apply a Schmidt-computed μ to ``iso_mu``, undoably."""
        _Set_Spin_Undoable(self.iso_mu, value, self)


# ══════════════════════════════════════════════════════════════════
#  Isotope Panel
# ══════════════════════════════════════════════════════════════════
# Per-position accent colors for the isotope blocks (left stripe +
# title tint) so entries are visually distinguishable at a glance.
ISOTOPE_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b4", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


class IsotopePanel(QGroupBox):
    """Configuration panel for a single isotope."""
    removed = Signal(object)
    reference_toggled = Signal(object, bool)

    def __init__(self, index, parent=None):
        super().__init__(f"Isotope {index + 1}", parent)
        # objectName-scoped styling so the color stripe / reference
        # border never cascades onto the nested group boxes.
        self.setObjectName("isotopePanel")
        self.index = index
        self._color_index = index
        self._is_reference = False
        self._spectrum_only = False
        self._z = 0
        self._mass_table = None
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Preferred)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(6, 10, 6, 6)

        # Header with drag handle, reference checkbox and remove button
        header = QHBoxLayout()
        from PySide6.QtWidgets import QToolButton
        self.drag_handle = QToolButton()
        self.drag_handle.setText("☰")
        self.drag_handle.setToolTip("Drag to reorder isotopes")
        self.drag_handle.setFixedSize(22, 22)
        self.drag_handle.setStyleSheet(
            "QToolButton { border: none; font-size: 14px; color: #888; }"
            "QToolButton:hover { color: #ccc; }")
        self.drag_handle.installEventFilter(self)
        header.addWidget(self.drag_handle)
        self._drag_start_pos = None
        self.ref_check = QCheckBox("Reference")
        self.ref_check.setToolTip(
            "Mark this isotope as the reference for HFS scaling. "
            "Its A/B constants will be used to scale other isotopes.")
        self.ref_check.toggled.connect(self._on_reference_toggled)
        header.addWidget(self.ref_check)
        header.addStretch()
        # No fixed width: 70 px fit the dark theme but cramps the text
        # under Classic 98's bevels/font. The style's sizeHint knows
        # its own padding; the header stretch keeps the button compact.
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        header.addWidget(self.remove_btn)
        main_layout.addLayout(header)

        # Basic fields
        form = QFormLayout()
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(12)

        # Mass number A + actual atomic mass display
        a_row = QHBoxLayout()
        a_row.setSpacing(8)
        self.iso_A = _make_int(0, 1, 300,
                               tooltip="Mass number of the isotope")
        a_row.addWidget(self.iso_A)
        self.iso_mass_label = QLabel("—")
        self.iso_mass_label.setToolTip(
            "Atomic mass (amu) used in calculations, "
            "looked up from the IUPAC/AME table for the current Z and A.")
        self.iso_mass_label.setStyleSheet(
            "color: #aaa; font-style: italic;")
        # Cosmetic read-out: never let its text width inflate the
        # column's layout minimum (it did — the amu string alone
        # widened every panel by >100 px vs the empty "—" state, so
        # the measured column band depended on whether a mass had
        # been looked up yet).
        self.iso_mass_label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                          QSizePolicy.Policy.Preferred)
        a_row.addWidget(self.iso_mass_label, 1)
        self.iso_A.valueChanged.connect(self._update_mass_display)
        form.addRow("A:", a_row)
        self.iso_label = QLineEdit()
        self.iso_label.setToolTip("Display label (e.g., '59Co'). Must match anchor label if used.")
        form.addRow("Label:", self.iso_label)
        self.iso_I = _make_double(0, 0, 20, 1, 0.5,
                                  tooltip="Nuclear spin (half-integer allowed, 0 for even-even nuclei)")
        form.addRow("I:", self.iso_I)
        self.iso_mu = _make_double(0, -10, 10, 5, 0.1,
                                   tooltip="Magnetic moment in nuclear magnetons. Set 0 if using Schmidt.")
        form.addRow("\u03bc (\u03bc\u2099):", self.iso_mu)
        self.iso_Q = _make_double(0, -5, 5, 5, 0.1,
                                  tooltip="Electric quadrupole moment in barns. 0 for I=0 or I=1/2.")
        form.addRow("Q (b):", self.iso_Q)
        self.iso_shift = _make_double(0, -1e6, 1e6, 2, 1.0,
                                      tooltip="Isotope shift in MHz, defined as (this isotope's centroid − reference centroid). "
                                              "Positive = this isotope sits at higher frequency than the reference. "
                                              "Leave at 0 for the reference itself.")
        form.addRow("Isotope shift (MHz):", self.iso_shift)
        main_layout.addLayout(form)

        # Rates section (hidden in spectrum-only mode)
        self.rates_group = QGroupBox("Rates && Timing")
        rg = QFormLayout(self.rates_group)
        rg.setContentsMargins(10, 16, 10, 8)
        rg.setVerticalSpacing(8)
        rg.setHorizontalSpacing(12)
        self.iso_yield = _make_sci(
            0, 0, 1e15,
            tooltip="Ion production rate at the interaction region (ions/s)")
        rg.addRow("Yield (ions/s):", self.iso_yield)
        self.iso_eff = _make_double(
            0.001, 0, 1, 6, 0.0001,
            tooltip="Fraction of ions producing detected photons (e.g., 0.001 = 1 per 1000)")
        rg.addRow("Efficiency:", self.iso_eff)
        self.iso_peaks = _make_int(
            1, 1, 100,
            tooltip="Number of strongest HFS peaks to include in timing estimate")
        rg.addRow("Peaks to measure:", self.iso_peaks)

        # Uniform timing override
        self.uniform_check = QCheckBox("Override uniform timing")
        self.uniform_check.setToolTip(
            "Override the global uniform timing setting for this isotope")
        rg.addRow(self.uniform_check)
        self.uniform_combo = QComboBox()
        self.uniform_combo.addItems(["Off", "On"])
        self.uniform_combo.setEnabled(False)
        self.uniform_check.toggled.connect(self.uniform_combo.setEnabled)
        rg.addRow("  Uniform timing:", self.uniform_combo)

        main_layout.addWidget(self.rates_group)

        # Background section (hidden in spectrum-only mode)
        self.bg_group = QGroupBox("Background")
        bg_form = QFormLayout(self.bg_group)
        bg_form.setContentsMargins(10, 16, 10, 8)
        bg_form.setVerticalSpacing(8)
        bg_form.setHorizontalSpacing(12)

        self.bg_simple_radio = QRadioButton("Simple rate (Hz)")
        self.bg_simple_radio.setToolTip("Single background count rate in Hz")
        self.bg_tof_radio = QRadioButton("TOF-gated")
        self.bg_tof_radio.setToolTip(
            "Time-of-flight gated background: effective rate = continuous * (gate / dwell)")
        self.bg_mode_group = QButtonGroup(self)
        self.bg_mode_group.addButton(self.bg_simple_radio)
        self.bg_mode_group.addButton(self.bg_tof_radio)
        self.bg_simple_radio.setChecked(True)

        # Simple background
        self.bg_rate = _make_double(
            10, 0, 1e6, 4, 1.0,
            tooltip="Background count rate (counts/s per voltage step)")
        bg_form.addRow(self.bg_simple_radio, self.bg_rate)

        # TOF-gated background. Cont and Gate used to share the radio's
        # row; that row's layout minimum (radio + two labelled spins,
        # and the Cont spin sizes itself for "100000000.0") was the
        # widest thing on the panel and clipped at the column's default
        # width -- same disease as the Laser anchor row. Two lines,
        # with Gate on an indented form row like "  Uniform timing:".
        tof_row = QHBoxLayout()
        tof_row.setSpacing(8)
        self.bg_cont = _make_double(
            4000, 0, 1e8, 1, 100,
            tooltip="Continuous background rate before TOF gating (Hz)")
        tof_row.addWidget(QLabel("Cont:"))
        tof_row.addWidget(self.bg_cont)
        bg_form.addRow(self.bg_tof_radio, tof_row)
        self.bg_gate = _make_double(
            2.5, 0.01, 1000, 2, 0.1,
            tooltip="Time-of-flight gate width (\u03bcs)")
        bg_form.addRow("  Gate (\u03bcs):", self.bg_gate)

        main_layout.addWidget(self.bg_group)

        # Connect bg radio buttons
        self.bg_simple_radio.toggled.connect(
            lambda c: self.bg_rate.setEnabled(c))
        self.bg_tof_radio.toggled.connect(
            lambda c: (self.bg_cont.setEnabled(c), self.bg_gate.setEnabled(c)))
        self.bg_cont.setEnabled(False)
        self.bg_gate.setEnabled(False)

        # HFS overrides
        self.hfs_check = QCheckBox("Override A/B constants")
        self.hfs_check.setToolTip(
            "Provide explicit HFS A/B constants instead of scaling from the reference")
        main_layout.addWidget(self.hfs_check)

        self.hfs_widget = QWidget()
        hf = QFormLayout(self.hfs_widget)
        hf.setContentsMargins(15, 0, 0, 0)
        self.hfs_Al = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Magnetic dipole HFS constant A, lower level (MHz)")
        hf.addRow("A_lower (MHz):", self.hfs_Al)
        self.hfs_Bl = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Electric quadrupole HFS constant B, lower level (MHz)")
        hf.addRow("B_lower (MHz):", self.hfs_Bl)
        self.hfs_Au = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Magnetic dipole HFS constant A, upper level (MHz)")
        hf.addRow("A_upper (MHz):", self.hfs_Au)
        self.hfs_Bu = _make_double(0, -1e6, 1e6, 4, 1.0,
                                   tooltip="Electric quadrupole HFS constant B, upper level (MHz)")
        hf.addRow("B_upper (MHz):", self.hfs_Bu)
        main_layout.addWidget(self.hfs_widget)
        self.hfs_check.toggled.connect(self.hfs_widget.setVisible)
        self.hfs_widget.setVisible(False)

        # Schmidt valence: ticking the box or changing the spec fills
        # the μ field with the unquenched Schmidt value so downstream
        # HFS scaling never divides by zero. The change is recorded on
        # the global undo stack so Ctrl-Z reverts it.
        self.schmidt = SchmidtWidget()
        self.schmidt.moment_changed.connect(self._apply_schmidt_moment)
        main_layout.addWidget(self.schmidt)

        # Isomer section
        self.isomer_check = QCheckBox("Has isomer")
        self.isomer_check.setToolTip("Enable isomeric state for this isotope")
        main_layout.addWidget(self.isomer_check)

        self.isomer_panel = IsomerPanel()
        main_layout.addWidget(self.isomer_panel)
        self.isomer_check.toggled.connect(self.isomer_panel.setVisible)
        self.isomer_panel.setVisible(False)

        # The input values are short; without a cap the fields stretch
        # to whatever width the column has and the panel is mostly air.
        # The MINIMUM matters just as much: Qt sizes a spinbox for its
        # widest representable text, so a 0..1e8 range demands ~184 px
        # and inflates the whole column's layout minimum. Typed values
        # are far shorter than the range bound; 70 px shows them all.
        # (findChildren takes one type per call in PySide6.)
        for cls in (QDoubleSpinBox, QSpinBox, QLineEdit):
            for w in self.findChildren(cls):
                w.setMaximumWidth(160)
                w.setMinimumWidth(70)
        for f in self.findChildren(QFormLayout):
            f.setVerticalSpacing(4)
            f.setHorizontalSpacing(8)

        self._apply_panel_style()

    def _apply_panel_style(self):
        """Compose the per-position color stripe with the reference
        border (both used to fight over setStyleSheet)."""
        color = ISOTOPE_COLORS[self._color_index % len(ISOTOPE_COLORS)]
        if self._is_reference:
            border = "2px solid #42a5f5"
        else:
            border = "1px solid #4a4a4f"
        self.setStyleSheet(
            f"QGroupBox#isotopePanel {{ border: {border};"
            f" border-left: 4px solid {color}; }}"
            f"QGroupBox#isotopePanel::title {{ color: {color}; }}")

    def set_color_index(self, i):
        if i != self._color_index:
            self._color_index = i
            self._apply_panel_style()

    def set_spectrum_only(self, spectrum_only):
        self._spectrum_only = spectrum_only
        self.rates_group.setVisible(not spectrum_only)
        self.bg_group.setVisible(not spectrum_only)
        self.isomer_panel.set_spectrum_only(spectrum_only)

    def set_z_and_table(self, z, mass_table):
        self._z = int(z) if z else 0
        self._mass_table = mass_table
        self._update_mass_display()

    def _update_mass_display(self):
        if not self._mass_table or not self._z:
            self.iso_mass_label.setText("—")
            return
        try:
            mass = get_mass(self._mass_table, self._z, self.iso_A.value())
            self.iso_mass_label.setText(f"{mass:.6f} amu")
        except KeyError:
            self.iso_mass_label.setText("(not in table)")

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent, QMimeData, QPoint
        from PySide6.QtGui import QDrag
        if obj is self.drag_handle:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_start_pos = event.globalPosition().toPoint()
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if (self._drag_start_pos is not None and
                        (event.globalPosition().toPoint() -
                         self._drag_start_pos).manhattanLength() > 10):
                    drag = QDrag(self)
                    mime = QMimeData()
                    mime.setText(f"isotope_{id(self)}")
                    mime.setData("application/x-isotope-panel",
                                 str(id(self)).encode())
                    drag.setMimeData(mime)
                    pm = self.grab().scaled(
                        160, 100, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    drag.setPixmap(pm)
                    drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
                    self._drag_start_pos = None
                    drag.exec(Qt.DropAction.MoveAction)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_start_pos = None
        return super().eventFilter(obj, event)

    def _on_reference_toggled(self, checked):
        self.reference_toggled.emit(self, checked)
        self._is_reference = bool(checked)
        if checked:
            self.hfs_check.setChecked(True)
            self.hfs_check.setEnabled(False)
        else:
            self.hfs_check.setEnabled(True)
        self._apply_panel_style()

    def set_reference(self, is_ref):
        """Set reference state programmatically without emitting signal."""
        self.ref_check.blockSignals(True)
        self.ref_check.setChecked(is_ref)
        self.ref_check.blockSignals(False)
        self._is_reference = bool(is_ref)
        if is_ref:
            self.hfs_check.setChecked(True)
            self.hfs_check.setEnabled(False)
        else:
            self.hfs_check.setEnabled(True)
        self._apply_panel_style()

    def set_index(self, index):
        self.index = index
        self.setTitle(f"Isotope {index + 1}")
        self.set_color_index(index)

    def _apply_schmidt_moment(self, value):
        """Apply a Schmidt-computed μ to ``iso_mu``, undoably."""
        _Set_Spin_Undoable(self.iso_mu, value, self)

    def to_dict(self):
        d = {
            "A": self.iso_A.value(),
            "label": self.iso_label.text(),
            "I": self.iso_I.value(),
            "mu": self.iso_mu.value(),
            "Q": self.iso_Q.value(),
            "isotope_shift_MHz": self.iso_shift.value(),
        }

        # Rates (only in timing mode)
        if not self._spectrum_only:
            d["yield_ions_per_sec"] = self.iso_yield.value()
            d["spectroscopic_efficiency"] = self.iso_eff.value()
            d["peaks_to_measure"] = self.iso_peaks.value()
            if self.uniform_check.isChecked():
                d["uniform_timing"] = self.uniform_combo.currentText() == "On"

            # Background
            if self.bg_simple_radio.isChecked():
                d["background_rate_Hz"] = self.bg_rate.value()
            else:
                d["background"] = {
                    "continuous_rate_Hz": self.bg_cont.value(),
                    "tof_gate_us": self.bg_gate.value(),
                }

        # HFS overrides
        if self.hfs_check.isChecked():
            d["A_lower_MHz"] = self.hfs_Al.value()
            d["B_lower_MHz"] = self.hfs_Bl.value()
            d["A_upper_MHz"] = self.hfs_Au.value()
            d["B_upper_MHz"] = self.hfs_Bu.value()

        # Schmidt
        sv = self.schmidt.to_dict()
        if sv is not None:
            d["schmidt_valence"] = sv

        # Isomer
        if self.isomer_check.isChecked():
            d["isomer"] = self.isomer_panel.to_dict()

        return d

    def from_dict(self, d):
        self.iso_A.setValue(int(d.get("A", 0)))
        self.iso_label.setText(str(d.get("label", "")))
        self.iso_I.setValue(float(d.get("I", 0)))
        self.iso_mu.setValue(float(d.get("mu", 0)))
        self.iso_Q.setValue(float(d.get("Q", 0)))
        self.iso_shift.setValue(float(d.get("isotope_shift_MHz", 0)))

        # Rates
        if "yield_ions_per_sec" in d and d["yield_ions_per_sec"] is not None:
            self.iso_yield.setValue(float(d["yield_ions_per_sec"]))
        if "spectroscopic_efficiency" in d and d["spectroscopic_efficiency"] is not None:
            self.iso_eff.setValue(float(d["spectroscopic_efficiency"]))
        if "peaks_to_measure" in d and d["peaks_to_measure"] is not None:
            self.iso_peaks.setValue(int(d["peaks_to_measure"]))
        if "uniform_timing" in d and d["uniform_timing"] is not None:
            self.uniform_check.setChecked(True)
            self.uniform_combo.setCurrentText(
                "On" if d["uniform_timing"] else "Off")

        # Background
        bg_block = d.get("background")
        if bg_block is not None:
            self.bg_tof_radio.setChecked(True)
            self.bg_cont.setValue(float(bg_block.get("continuous_rate_Hz", 0)))
            self.bg_gate.setValue(float(bg_block.get("tof_gate_us", 0)))
        elif "background_rate_Hz" in d and d["background_rate_Hz"] is not None:
            self.bg_simple_radio.setChecked(True)
            self.bg_rate.setValue(float(d["background_rate_Hz"]))

        # HFS
        has_hfs = any(d.get(k) is not None
                      for k in ("A_lower_MHz", "A_upper_MHz",
                                "B_lower_MHz", "B_upper_MHz"))
        self.hfs_check.setChecked(has_hfs)
        if has_hfs:
            self.hfs_Al.setValue(float(d.get("A_lower_MHz", 0)))
            self.hfs_Bl.setValue(float(d.get("B_lower_MHz", 0)))
            self.hfs_Au.setValue(float(d.get("A_upper_MHz", 0)))
            self.hfs_Bu.setValue(float(d.get("B_upper_MHz", 0)))

        # Schmidt
        self.schmidt.from_dict(d.get("schmidt_valence"))
        # If the saved YAML pre-dates the auto-fill change and stored
        # μ=0 alongside an enabled Schmidt spec, populate μ now so
        # the estimation pipeline doesn't divide by zero. Custom
        # non-zero saved values are respected. The bypass flag
        # routes the auto-fill through a no-undo path so loading a
        # project doesn't leave a "Change value" entry on the undo
        # stack the user didn't perform.
        self._bypass_undo = True
        try:
            if (self.schmidt.enable_check.isChecked()
                    and self.iso_mu.value() == 0):
                self.schmidt._emit_moment()
        finally:
            self._bypass_undo = False

        # Isomer
        if "isomer" in d and d["isomer"] is not None:
            self.isomer_check.setChecked(True)
            self.isomer_panel.from_dict(d["isomer"])
        else:
            self.isomer_check.setChecked(False)


# ══════════════════════════════════════════════════════════════════
#  Isotope drag-drop container
# ══════════════════════════════════════════════════════════════════
class _IsotopeDropContainer(QWidget):
    """Holds IsotopePanel widgets and accepts drag-drop reordering."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self._owner = owner
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-isotope-panel"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-isotope-panel"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat("application/x-isotope-panel"):
            return
        try:
            block_id = int(event.mimeData().data(
                "application/x-isotope-panel").data().decode())
        except Exception:
            return
        dragged = next((p for p in self._owner.panels
                        if id(p) == block_id), None)
        if dragged is None:
            return
        # Determine target index based on y position
        layout = self._owner.panel_layout
        pos_y = event.position().y()
        target_idx = layout.count()
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w and pos_y < w.y() + w.height() / 2:
                target_idx = i
                break
        self._owner.reorder_panel(dragged, target_idx)
        event.acceptProposedAction()


# ══════════════════════════════════════════════════════════════════
#  Isotope List Widget
# ══════════════════════════════════════════════════════════════════
class IsotopeListWidget(QWidget):
    """Container for dynamic list of IsotopePanel widgets."""

    # Any change to the list: add, remove, clear, reorder, or a new
    # reference (the voltage estimator rebuilds on it).
    order_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels = []
        self._reference_panel = None
        self._spectrum_only = False
        self._z = 0
        try:
            self._mass_table = load_mass_table()
        except Exception:
            self._mass_table = {}

        outer = QVBoxLayout(self)

        # Buttons
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+ Add Isotope")
        self.add_btn.setToolTip("Add a new isotope to the list")
        self.add_btn.clicked.connect(lambda: self.add_isotope())
        btn_row.addWidget(self.add_btn)
        self.count_label = QLabel("0 isotopes")
        btn_row.addWidget(self.count_label)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        # Scroll area for panels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.panel_container = _IsotopeDropContainer(self)
        self.panel_layout = QVBoxLayout(self.panel_container)
        self.panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.panel_container)
        outer.addWidget(scroll)
        self._scroll = scroll

    def preferred_width(self):
        """Column width at which no isotope-panel row clips.

        The scroll area has its horizontal scrollbar off, so a panel
        wider than the viewport is simply cut at the right edge (the
        Remove button and the TOF Gate spin were the casualties).
        Measure a worst-case probe panel instead of hardcoding a band:
        the layout minimum depends on font/DPI, which is exactly why
        the fixed 320-460 px band clipped on this machine.
        """
        from PySide6.QtWidgets import QStyle
        probe = IsotopePanel(1)
        try:
            # Widest realistic state: timing mode (default) with the
            # isomer sub-panel expanded. Schmidt/AB bodies stay
            # collapsed -- their rows are narrow, and ticking Schmidt
            # on a probe would auto-fill mu through the undo stack.
            probe.isomer_check.setChecked(True)
            need = probe.minimumSizeHint().width()
        finally:
            probe.deleteLater()
        sbw = self._scroll.style().pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarExtent, None, self._scroll)
        m = self.layout().contentsMargins()
        pm = self.panel_layout.contentsMargins()
        # The Run Options group under the scroll area must not clip either.
        return max(need + pm.left() + pm.right()
                   + sbw + 2 * self._scroll.frameWidth()
                   + m.left() + m.right(), self.minimumSizeHint().width())

    def set_z(self, z):
        self._z = int(z) if z else 0
        for p in self.panels:
            p.set_z_and_table(self._z, self._mass_table)

    def add_isotope(self, data=None):
        panel = IsotopePanel(len(self.panels))
        panel.removed.connect(self._on_remove)
        panel.reference_toggled.connect(self._on_reference_toggled)
        self.panels.append(panel)
        self.panel_layout.addWidget(panel)
        panel.set_spectrum_only(self._spectrum_only)
        panel.set_z_and_table(self._z, self._mass_table)
        self._update_count()
        if data:
            panel.from_dict(data)
        return panel

    def reorder_panel(self, dragged, target_idx):
        """Move a dragged panel to a new visual index, syncing self.panels."""
        if dragged not in self.panels:
            return
        old_idx = self.panels.index(dragged)
        if old_idx == target_idx or old_idx == target_idx - 1:
            # No-op moves (dropping in current slot or just after self)
            return
        # Remove first, then insert at clamped target
        self.panels.pop(old_idx)
        if target_idx > old_idx:
            target_idx -= 1
        target_idx = max(0, min(target_idx, len(self.panels)))
        self.panels.insert(target_idx, dragged)
        # Resync the layout to match self.panels order
        for p in self.panels:
            self.panel_layout.removeWidget(p)
        for p in self.panels:
            self.panel_layout.addWidget(p)
        for i, p in enumerate(self.panels):
            p.set_index(i)
        self.order_changed.emit()

    def _on_remove(self, panel):
        if panel in self.panels:
            if self._reference_panel is panel:
                self._reference_panel = None
            self.panels.remove(panel)
            self.panel_layout.removeWidget(panel)
            panel.deleteLater()
            # Renumber
            for i, p in enumerate(self.panels):
                p.set_index(i)
            self._update_count()

    def _on_reference_toggled(self, panel, checked):
        if checked:
            if self._reference_panel is not None and self._reference_panel is not panel:
                self._reference_panel.set_reference(False)
            self._reference_panel = panel
        else:
            if self._reference_panel is panel:
                self._reference_panel = None
        self.order_changed.emit()

    def get_reference_panel(self):
        """Return the panel currently marked as reference, or None."""
        return self._reference_panel

    def set_reference_from_config(self, ref_dict):
        """Match reference A to a panel, mark it, and populate HFS values."""
        if not ref_dict:
            return False
        ref_A = ref_dict.get("A")
        if ref_A is None:
            return False
        for panel in self.panels:
            if panel.iso_A.value() == ref_A:
                panel.set_reference(True)
                self._reference_panel = panel
                panel.hfs_Al.setValue(float(ref_dict.get("A_lower_MHz", 0)))
                panel.hfs_Bl.setValue(float(ref_dict.get("B_lower_MHz", 0)))
                panel.hfs_Au.setValue(float(ref_dict.get("A_upper_MHz", 0)))
                panel.hfs_Bu.setValue(float(ref_dict.get("B_upper_MHz", 0)))
                return True
        return False

    def clear_all(self):
        for p in list(self.panels):
            self.panel_layout.removeWidget(p)
            p.deleteLater()
        self.panels.clear()
        self._reference_panel = None
        self._update_count()

    def _update_count(self):
        n = len(self.panels)
        self.count_label.setText(f"{n} isotope{'s' if n != 1 else ''}")
        self.order_changed.emit()

    def set_spectrum_only(self, spectrum_only):
        self._spectrum_only = spectrum_only
        for p in self.panels:
            p.set_spectrum_only(spectrum_only)


# ══════════════════════════════════════════════════════════════════
#  Global Parameters Widget
# ══════════════════════════════════════════════════════════════════
class GlobalParamsWidget(QWidget):
    """Left pane: all global configuration parameters."""
    spectrum_only_changed = Signal(bool)
    z_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        # Room for the group-box borders: with the default margins the
        # 1 px right border lands on the viewport's last pixel (or
        # under the vertical scrollbar) and is clipped away.
        layout.setContentsMargins(4, 4, 9, 4)

        # ── Element & Transition ─────────────────────────────────
        grp = QGroupBox("Element && Transition")
        form = QFormLayout(grp)

        elem_row = QHBoxLayout()
        elem_row.setSpacing(6)
        self.element_edit = QLineEdit()
        self.element_edit.setToolTip("Chemical element symbol (e.g., 'Co', 'Sn')")
        self.element_edit.setMaximumWidth(52)
        elem_row.addWidget(QLabel("Element:"))
        elem_row.addWidget(self.element_edit)
        self.z_spin = _make_int(0, 1, 118,
                                tooltip="Atomic number (proton number)")
        # Explicit cap: without it the spin stretches and, in a narrow
        # column, the trailing stretch let the row's tail clip instead.
        self.z_spin.setMaximumWidth(64)
        elem_row.addWidget(QLabel("Z:"))
        elem_row.addWidget(self.z_spin)
        elem_row.addStretch()
        form.addRow(elem_row)

        self.lower_level = _make_double(
            0, 0, 200000, 6, 1.0,
            tooltip="Lower energy level of the transition (cm\u207b\u00b9)")
        form.addRow("Lower level (cm\u207b\u00b9):", self.lower_level)
        self.upper_level = _make_double(
            0, 0, 200000, 6, 1.0,
            tooltip="Upper energy level of the transition (cm\u207b\u00b9)")
        form.addRow("Upper level (cm\u207b\u00b9):", self.upper_level)
        self.j_lower = _make_double(
            0, 0, 20, 1, 0.5,
            tooltip="Total angular momentum J of the lower level (half-integer allowed)")
        form.addRow("J_lower:", self.j_lower)
        self.j_upper = _make_double(
            0, 0, 20, 1, 0.5,
            tooltip="Total angular momentum J of the upper level (half-integer allowed)")
        form.addRow("J_upper:", self.j_upper)
        layout.addWidget(grp)

        # Element <-> Z sync
        self.element_edit.textChanged.connect(self._on_element_changed)
        self.z_spin.valueChanged.connect(self._on_z_changed)

        # ── Beam ─────────────────────────────────────────────────
        grp = QGroupBox("Beam")
        form = QFormLayout(grp)
        self.voltage_kV = _make_double(
            30.0, 0.001, 200, 3, 0.1,
            tooltip="Ion beam acceleration voltage (kV), typically 30-60 kV")
        form.addRow("Voltage (kV):", self.voltage_kV)
        self.charge_state = _make_int(
            1, 1, 10,
            tooltip="Ion charge state q (number of electrons removed)")
        form.addRow("Charge state:", self.charge_state)
        self.geometry = QComboBox()
        self.geometry.addItems(["anti-collinear", "collinear"])
        self.geometry.setToolTip(
            "Laser-ion beam geometry: anti-collinear (head-on) or collinear (co-propagating)")
        form.addRow("Geometry:", self.geometry)
        layout.addWidget(grp)

        # ── Laser ────────────────────────────────────────────────
        grp = QGroupBox("Laser")
        lv = QVBoxLayout(grp)
        form = QFormLayout()
        self.harmonic = _make_int(
            1, 1, 10,
            tooltip="Laser harmonic multiplier (1=fundamental, 2=SHG, 4=FHG). "
                    "Effective wavenumber = setpoint x harmonic.")
        form.addRow("Harmonic:", self.harmonic)
        lv.addLayout(form)

        # Setpoint radio
        self.laser_setpoint_radio = QRadioButton("Setpoint (cm\u207b\u00b9)")
        self.laser_setpoint_radio.setToolTip("Specify the fundamental laser wavenumber directly")
        self.laser_anchor_radio = QRadioButton("Anchor to isotope")
        self.laser_anchor_radio.setToolTip(
            "Auto-calculate laser frequency from an isotope's centroid position")
        self.laser_mode_group = QButtonGroup(self)
        self.laser_mode_group.addButton(self.laser_setpoint_radio)
        self.laser_mode_group.addButton(self.laser_anchor_radio)
        self.laser_setpoint_radio.setChecked(True)

        # Setpoint row. Trailing stretch keeps the spinbox next to its
        # radio label \u2014 without it the extra row width opened a gap
        # between the two.
        sp_row = QHBoxLayout()
        sp_row.addWidget(self.laser_setpoint_radio)
        self.setpoint_cm = _make_double(
            0, 0, 200000, 6, 1.0,
            tooltip="Fundamental laser wavenumber (cm\u207b\u00b9)")
        sp_row.addWidget(self.setpoint_cm)
        sp_row.addStretch()
        lv.addLayout(sp_row)

        # Anchor rows. Split over two lines: radio + isotope label on
        # the first, dV + state indented below — one line held the
        # column's widest minimum and clipped the right border (the
        # state combo could fall off the edge entirely).
        anc_row = QHBoxLayout()
        anc_row.addWidget(self.laser_anchor_radio)
        self.anchor_isotope = QLineEdit()
        self.anchor_isotope.setToolTip(
            "Label of isotope whose centroid defines the laser frequency "
            "(must match an isotope label)")
        self.anchor_isotope.setPlaceholderText("e.g. 59Co")
        self.anchor_isotope.setMaximumWidth(80)
        anc_row.addWidget(self.anchor_isotope)
        anc_row.addStretch()
        lv.addLayout(anc_row)

        anc_row2 = QHBoxLayout()
        anc_row2.addSpacing(22)
        anc_row2.addWidget(QLabel("dV (V):"))
        self.anchor_dV = _make_double(
            0, -1e4, 1e4, 2, 1.0,
            tooltip="Voltage offset of the anchor isotope's centroid "
                    "from beam voltage (V)")
        self.anchor_dV.setMaximumWidth(72)
        anc_row2.addWidget(self.anchor_dV)
        self.anchor_state = QComboBox()
        self.anchor_state.addItems(["gs", "isomer"])
        self.anchor_state.setToolTip("Which state of the anchor isotope to use")
        self.anchor_state.setMaximumWidth(76)
        anc_row2.addWidget(self.anchor_state)
        anc_row2.addStretch()
        lv.addLayout(anc_row2)

        layout.addWidget(grp)

        # Connect laser mode radio
        self.laser_setpoint_radio.toggled.connect(self._on_laser_mode)
        self._on_laser_mode()

        # ── Linewidth ────────────────────────────────────────────
        grp = QGroupBox("Linewidth")
        form = QFormLayout(grp)
        self.fwhm_MHz = _make_double(
            100, 0.1, 10000, 1, 10,
            tooltip="Full width at half maximum of the spectral line (MHz). "
                    "Includes Doppler broadening + laser linewidth.")
        form.addRow("FWHM (MHz):", self.fwhm_MHz)
        layout.addWidget(grp)

        # ── Scan ─────────────────────────────────────────────────
        grp = QGroupBox("Scan")
        form = QFormLayout(grp)
        self.scan_range = _make_double(
            100, 0.1, 1e5, 1, 10,
            tooltip="Total voltage scan range (V)")
        form.addRow("Voltage range (V):", self.scan_range)
        self.scan_step = _make_double(
            1.0, 0.001, 1000, 3, 0.1,
            tooltip="Voltage step size per DAC step (V)")
        form.addRow("Step size (V):", self.scan_step)
        self.scan_dwell = _make_double(
            200, 0.1, 1e6, 1, 10,
            tooltip="Time spent at each voltage step (ms)")
        form.addRow("Dwell time (ms):", self.scan_dwell)
        layout.addWidget(grp)

        # ── Statistics ───────────────────────────────────────────
        self.stats_group = QGroupBox("Statistics (Timing Mode)")
        self.stats_group.setCheckable(True)
        self.stats_group.setChecked(True)
        self.stats_group.setToolTip(
            "Enable to calculate timing estimates. "
            "Uncheck for spectrum-only mode (no timing calculations).")
        form = QFormLayout(self.stats_group)
        self.sigma = _make_double(
            3.0, 0.1, 20, 1, 0.5,
            tooltip="Required statistical significance (number of \u03c3, typically 3)")
        form.addRow("Required \u03c3:", self.sigma)
        self.uniform_timing = QCheckBox("Uniform timing (global default)")
        self.uniform_timing.setToolTip(
            "If enabled, all peaks are measured for the same time "
            "(based on strongest peak assuming full yield)")
        form.addRow(self.uniform_timing)
        layout.addWidget(self.stats_group)

        # Connect statistics toggle
        self.stats_group.toggled.connect(self._on_stats_toggled)

        # ── Advanced ─────────────────────────────────────────────
        grp = QGroupBox("Advanced")
        form = QFormLayout(grp)
        self.g_s_factor = _make_double(
            1.0, 0, 1, 2, 0.05,
            tooltip="Quenching factor for the nucleon spin g-factor in Schmidt calculations "
                    "(1.0 = free nucleon, typical: 0.6-0.8)")
        form.addRow("g_s factor:", self.g_s_factor)
        layout.addWidget(grp)

        layout.addStretch()

        # Compact pass: the stored values are short, so cap every field
        # (the element/anchor edits keep their tighter caps) and densify
        # the forms — the column narrows without any clipping. 130 px
        # still fits the widest value ("12345.678900" + spin buttons).
        for cls in (QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox):
            for w in content.findChildren(cls):
                if w.maximumWidth() > 130:
                    w.setMaximumWidth(130)
        for f in content.findChildren(QFormLayout):
            f.setVerticalSpacing(4)
            f.setHorizontalSpacing(8)
        layout.setSpacing(6)

        scroll.setWidget(content)
        self._scroll = scroll
        self._content = content

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def preferred_width(self):
        """Column width at which the group borders are fully visible.

        The scroll content cannot shrink below its layout minimum (the
        labels don't elide), so any narrower column pushes the 1 px
        right border past the viewport. Measure instead of hardcoding:
        the minimum depends on font/DPI, which is exactly why fixed
        pixel bands kept clipping on other machines.
        """
        from PySide6.QtWidgets import QStyle
        sbw = self._scroll.style().pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarExtent, None, self._scroll)
        return (self._content.minimumSizeHint().width()
                + sbw + 2 * self._scroll.frameWidth())

    def _on_element_changed(self, text):
        z = ELEMENT_Z.get(text.strip())
        if z is not None:
            self.z_spin.blockSignals(True)
            self.z_spin.setValue(z)
            self.z_spin.blockSignals(False)
            self.z_changed.emit(z)

    def _on_z_changed(self, value):
        sym = Z_TO_ELEMENT.get(value, "")
        self.element_edit.blockSignals(True)
        self.element_edit.setText(sym)
        self.element_edit.blockSignals(False)
        self.z_changed.emit(value)

    def _on_laser_mode(self):
        is_setpoint = self.laser_setpoint_radio.isChecked()
        self.setpoint_cm.setEnabled(is_setpoint)
        self.anchor_isotope.setEnabled(not is_setpoint)
        self.anchor_dV.setEnabled(not is_setpoint)
        self.anchor_state.setEnabled(not is_setpoint)

    def _on_stats_toggled(self, checked):
        spectrum_only = not checked
        self.scan_step.setEnabled(checked)
        self.scan_dwell.setEnabled(checked)
        self.spectrum_only_changed.emit(spectrum_only)

    def is_spectrum_only(self):
        return not self.stats_group.isChecked()

    def to_dict(self):
        d = {
            "element": self.element_edit.text().strip(),
            "Z": self.z_spin.value(),
            "transition": {
                "lower_level_cm": self.lower_level.value(),
                "upper_level_cm": self.upper_level.value(),
                "J_lower": self.j_lower.value(),
                "J_upper": self.j_upper.value(),
            },
            "beam": {
                "voltage_kV": self.voltage_kV.value(),
                "charge_state": self.charge_state.value(),
                "geometry": self.geometry.currentText(),
            },
            "linewidth": {
                "fwhm_MHz": self.fwhm_MHz.value(),
            },
            "scan": {
                "voltage_range_V": self.scan_range.value(),
            },
        }

        # Laser
        laser = {"harmonic": self.harmonic.value()}
        if self.laser_setpoint_radio.isChecked():
            laser["setpoint_cm"] = self.setpoint_cm.value()
        else:
            laser["anchor"] = {
                "isotope": self.anchor_isotope.text().strip(),
                "dV": self.anchor_dV.value(),
                "state": self.anchor_state.currentText(),
            }
        d["laser"] = laser

        # Scan - step and dwell only in timing mode
        if self.stats_group.isChecked():
            d["scan"]["step_size_V"] = self.scan_step.value()
            d["scan"]["dwell_time_ms"] = self.scan_dwell.value()

        # Statistics
        if self.stats_group.isChecked():
            stats = {"required_sigma": self.sigma.value()}
            if self.uniform_timing.isChecked():
                stats["uniform_timing"] = True
            d["statistics"] = stats

        # g_s_factor
        if self.g_s_factor.value() != 1.0:
            d["g_s_factor"] = self.g_s_factor.value()

        return d

    def from_dict(self, d):
        self.element_edit.blockSignals(True)
        self.z_spin.blockSignals(True)
        self.element_edit.setText(str(d.get("element", "")))
        self.z_spin.setValue(int(d.get("Z", 0)))
        self.element_edit.blockSignals(False)
        self.z_spin.blockSignals(False)
        # Signals were blocked above to keep the element/Z spinboxes from
        # echoing each other; emit z_changed manually so the isotope
        # list (which drives the per-panel mass display) picks up the
        # restored Z. Without this, isotopes loaded from YAML show
        # "(not in table)" because the list still holds Z=0.
        self.z_changed.emit(self.z_spin.value())

        t = d.get("transition", {})
        self.lower_level.setValue(float(t.get("lower_level_cm", 0)))
        self.upper_level.setValue(float(t.get("upper_level_cm", 0)))
        self.j_lower.setValue(float(t.get("J_lower", 0)))
        self.j_upper.setValue(float(t.get("J_upper", 0)))

        b = d.get("beam", {})
        self.voltage_kV.setValue(float(b.get("voltage_kV", 30)))
        self.charge_state.setValue(int(b.get("charge_state", 1)))
        idx = self.geometry.findText(b.get("geometry", "anti-collinear"))
        if idx >= 0:
            self.geometry.setCurrentIndex(idx)

        la = d.get("laser", {})
        self.harmonic.setValue(int(la.get("harmonic", 1)))
        if "anchor" in la:
            self.laser_anchor_radio.setChecked(True)
            anc = la["anchor"]
            self.anchor_isotope.setText(str(anc.get("isotope", "")))
            self.anchor_dV.setValue(float(anc.get("dV", 0)))
            idx = self.anchor_state.findText(anc.get("state", "gs"))
            if idx >= 0:
                self.anchor_state.setCurrentIndex(idx)
        else:
            self.laser_setpoint_radio.setChecked(True)
            self.setpoint_cm.setValue(float(la.get("setpoint_cm", 0)))

        lw = d.get("linewidth", {})
        self.fwhm_MHz.setValue(float(lw.get("fwhm_MHz", 100)))

        sc = d.get("scan", {})
        self.scan_range.setValue(float(sc.get("voltage_range_V", 100)))
        self.scan_step.setValue(float(sc.get("step_size_V", 1.0)))
        self.scan_dwell.setValue(float(sc.get("dwell_time_ms", 200)))

        stats = d.get("statistics")
        if stats is not None:
            self.stats_group.setChecked(True)
            self.sigma.setValue(float(stats.get("required_sigma", 3.0)))
            self.uniform_timing.setChecked(
                bool(stats.get("uniform_timing", False)))
        else:
            self.stats_group.setChecked(False)

        self.g_s_factor.setValue(float(d.get("g_s_factor", 1.0)))


# ══════════════════════════════════════════════════════════════════
#  Parameters Tab (combines global + isotope list)
# ══════════════════════════════════════════════════════════════════
class ParametersTab(QWidget):
    """Three-column working surface: Element/Transition, Isotope list
    with Run Options under it, and Plots / Voltage Range Estimator / Log.
    The user can configure, kick off, and watch a run on a single surface
    without tab-switching. All three columns are independently resizable
    via the splitter handles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Column 1: Element / Transition (global params). Narrow — the
        # input values are short and the fields are capped at 130 px.
        # The band is measured, not hardcoded: the column must hold the
        # content's layout minimum (font/DPI dependent) or the group
        # borders clip on the right.
        self.global_params = GlobalParamsWidget()
        col1_need = self.global_params.preferred_width()
        self.global_params.setMinimumWidth(col1_need)
        self.global_params.setMaximumWidth(col1_need + 44)
        splitter.addWidget(self.global_params)

        # Column 2: Isotope list, with RunTab's Run Options group fixed
        # under its scroll area. Same measured-band treatment as column 1
        # (a hardcoded 320-460 band clipped the TOF-gate row and the
        # Remove button); the per-panel fields carry their own 160 px
        # limit too.
        self.isotope_list = IsotopeListWidget()
        self.run_tab = RunTab()
        self.isotope_list.layout().addWidget(self.run_tab._top_host)
        col2_need = self.isotope_list.preferred_width()
        self.isotope_list.setMinimumWidth(col2_need)
        self.isotope_list.setMaximumWidth(col2_need + 80)
        splitter.addWidget(self.isotope_list)

        # Column 3: plots, the voltage range estimator and the
        # (shortened) log, so the freed width from columns 1-2 goes to
        # the plot display.
        self.run_tab.setMinimumWidth(380)
        self.plots_panel = PlotsTab()
        self.Voltage_Estimator = Voltage_Range_Estimator(
            self.global_params, self.isotope_list)
        self.run_tab.set_plots_panel(self.plots_panel, self.Voltage_Estimator)
        splitter.addWidget(self.run_tab)

        # Column 3 (plots) soaks up extra width; columns 1-2 are
        # pinned by their max widths.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([col1_need + 4, col2_need + 4, 950])
        layout.addWidget(splitter)
        self._main_splitter = splitter

        # Connect spectrum-only toggle
        self.global_params.spectrum_only_changed.connect(
            self.isotope_list.set_spectrum_only)
        # Propagate Z so each isotope panel can show its actual mass
        self.global_params.z_changed.connect(self.isotope_list.set_z)
        self.isotope_list.set_z(self.global_params.z_spin.value())

    def showEvent(self, event):
        super().showEvent(event)
        # Defer one tick so width() reflects the final shown geometry.
        QTimer.singleShot(0, self._fit_main_splitter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_main_splitter()

    def _fit_main_splitter(self):
        """Keep the 3-column splitter within the tab width so the right-hand
        Run/Log column is never pushed off the right edge.

        The splitter is seeded with setSizes() at construction, before the
        window is maximized, so the sizes land against the pre-maximized width
        and can overflow the viewport until a manual restore+maximize. Re-fit
        only when the columns actually overflow (column 1 fixed; columns 2 and
        3 share the remainder 2:1 to match the stretch factors), so a manual
        column drag -- which doesn't change the tab width -- is preserved.
        """
        sp = getattr(self, "_main_splitter", None)
        if sp is None:
            return
        avail = self.width()
        if avail <= 100:
            return
        sizes = sp.sizes()
        if len(sizes) != 3:
            return
        if sum(sizes) <= avail + 2:
            return  # already fits within the viewport
        gp = self.global_params
        il = self.isotope_list
        col1 = min(max(sizes[0] or gp.minimumWidth(),
                       gp.minimumWidth()), gp.maximumWidth())
        col2 = min(max(sizes[1] or il.minimumWidth(),
                       il.minimumWidth()), il.maximumWidth())
        col3 = max(420, avail - col1 - col2)
        sp.setSizes([col1, col2, col3])


# ══════════════════════════════════════════════════════════════════
#  Run & Log Tab
# ══════════════════════════════════════════════════════════════════
class RunTab(QWidget):
    """Column 3 of the Estimate surface: the plots panel and the voltage
    range estimator (inserted via :meth:`set_plots_panel`) over the log,
    in a vertical splitter so the user decides how much room each gets.
    Also builds the Run Options group (``_top_host``), which
    ParametersTab hosts under the isotope list."""
    run_requested = Signal()
    save_requested = Signal()
    load_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Options + Run button at their natural (content) height. Built
        # here so run_btn, output_dir & co. stay RunTab handles; the
        # group itself sits under the isotope list (ParametersTab).
        self._top_host = QWidget()
        top_layout = QVBoxLayout(self._top_host)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Run options
        opts = QGroupBox("Run Options")
        opts_form = QFormLayout(opts)

        # Output dir. Default now lives under the unified root output
        # tree at ``<output_root>/estimates``; saved configs keep
        # their own path verbatim so existing projects are unaffected.
        from gui.shared_widgets import get_estimates_dir
        dir_row = QHBoxLayout()
        self.output_dir = QLineEdit(get_estimates_dir())
        # Show a path from its start when set from code (load, browse): Qt
        # left it scrolled to the tail, e.g. "utput/estimates".
        self.output_dir.setCursorPosition(0)
        self.output_dir.textChanged.connect(
            lambda _: self.output_dir.hasFocus()
            or self.output_dir.setCursorPosition(0))
        self.output_dir.setToolTip(
            "Estimate-tab output directory (defaults to "
            "<output_root>/estimates; the root lives in Settings ▸ "
            "Output directory).")
        dir_row.addWidget(self.output_dir)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        opts_form.addRow("Output directory:", dir_row)

        # Palette
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.palette_combo.setToolTip("Color palette for spectrum plots")
        opts_form.addRow("Color palette:", self.palette_combo)

        # Generate plots
        self.gen_plots = QCheckBox("Generate plots")
        self.gen_plots.setChecked(True)
        self.gen_plots.setToolTip("Generate HFS spectra and overview plots")
        opts_form.addRow(self.gen_plots)

        # Run / Save / Load: one row INSIDE the options group, equal
        # widths spanning the group (the natural-width pair with a
        # trailing stretch matched neither each other nor the edge,
        # 2026-10-24). Short labels fit column 2; the tooltips say more.
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setToolTip(
            "Run the estimation. From then on, every change re-runs it and "
            "updates the plots, the log and the estimator.")
        self.Save_Btn = QPushButton("Save", enabled=False)
        self.Save_Btn.setToolTip(
            "Save the latest run (plots, peak table, log) to the output "
            "directory as a cls_<timestamp> folder.")
        # Load a previous run for viewing — no re-run, no new output
        # folder.
        self.load_btn = QPushButton("Load...")
        self.load_btn.setToolTip(
            "Load a previous estimation run's folder (cls_<timestamp> "
            "under the output directory): its plots, peak table and log "
            "are restored for viewing without re-running.")
        for btn, sig in ((self.run_btn, self.run_requested),
                         (self.Save_Btn, self.save_requested),
                         (self.load_btn, self.load_requested)):
            btn.clicked.connect(sig.emit)
            btn_row.addWidget(btn, 1)
        opts_form.addRow(btn_row)

        top_layout.addWidget(opts)

        # Splitter: plots (stretch), the estimator, then the log.
        self._vsplit = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self._vsplit, 1)

        # Log output (bottom pane; the plots panel lands above it)
        self._log_host = QWidget()
        log_layout = QVBoxLayout(self._log_host)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Log Output:"))
        log_header.addStretch()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(lambda: self.log_text.clear())
        log_header.addWidget(self.clear_btn)
        log_layout.addLayout(log_header)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(_MONO_QSS)
        self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._highlighter = LogHighlighter(self.log_text.document())
        log_layout.addWidget(self.log_text)
        self._log_host.setMinimumHeight(80)
        self._vsplit.addWidget(self._log_host)

    def set_plots_panel(self, panel, estimator):
        """Stack the plots panel and the voltage estimator above the log.
        The three fill the column 5 : 5 : 2 and keep that ratio as the
        window resizes (2026-10-24); each handle still drags any pane down
        to its own minimum, as before."""
        self._vsplit.insertWidget(0, panel)
        self._vsplit.insertWidget(1, estimator)
        for i, stretch in enumerate((5, 5, 2)):
            self._vsplit.setStretchFactor(i, stretch)
        # More than any column is tall: Qt scales sizes to fit by weight.
        self._vsplit.setSizes([5000, 5000, 2000])

    def _browse_dir(self):
        from gui.shared_widgets import get_last_dir, remember_last_dir
        start = (self.output_dir.text().strip()
                 or get_last_dir("data", "save"))
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start)
        if d:
            self.output_dir.setText(d)
            remember_last_dir("data", "save", d)

    def append_log(self, text):
        self.log_text.appendPlainText(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def get_options(self):
        return {
            "output_dir": self.output_dir.text(),
            "palette": self.palette_combo.currentText(),
            "no_plot": not self.gen_plots.isChecked(),
        }


# ══════════════════════════════════════════════════════════════════
#  Plots Tab
# ══════════════════════════════════════════════════════════════════
class PlotsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Plot selector + open folder button
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("View:"))
        self.plot_selector = QComboBox()
        self.plot_selector.addItems(
            ["Individual Spectra", "Combined Overview", "Peak List"])
        self.plot_selector.currentIndexChanged.connect(self._on_selection)
        ctrl_row.addWidget(self.plot_selector)
        ctrl_row.addStretch()
        self.open_pdf_btn = QPushButton("Open PDF")
        self.open_pdf_btn.setToolTip("Open the selected plot as a PDF file")
        self.open_pdf_btn.clicked.connect(self._open_current_pdf)
        self.open_pdf_btn.setEnabled(False)
        ctrl_row.addWidget(self.open_pdf_btn)
        self.edit_plot_btn = QPushButton("Edit Plot")
        self.edit_plot_btn.setToolTip("Open plot editor for titles, labels, "
                                      "lines, legend, and export")
        self.edit_plot_btn.clicked.connect(self._open_editor)
        self.edit_plot_btn.setEnabled(False)
        ctrl_row.addWidget(self.edit_plot_btn)
        from PySide6.QtWidgets import QToolButton
        self.expand_btn = QToolButton()
        self.expand_btn.setText("⛶")
        self.expand_btn.setCheckable(True)
        self.expand_btn.setToolTip(
            "Expand the plots over the whole Estimate tab; click again "
            "to bring the parameter columns back.")
        ctrl_row.addWidget(self.expand_btn)
        layout.addLayout(ctrl_row)

        # Stacked widget: page 0 = plot canvas, page 1 = peak table
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Page 0: Matplotlib canvas with scroll area
        plot_page = QWidget()
        pl = QVBoxLayout(plot_page)
        pl.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas._plot_editor_opener = self._open_editor
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        pl.addWidget(self.toolbar)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        pl.addWidget(self.scroll)
        # matplotlib accepts every wheel event as a scroll_event, which
        # nothing here listens to, so the wheel or trackpad over a plot
        # never scrolled it (2026-10-22). eventFilter hands it on.
        self.canvas.installEventFilter(self)
        self._stack.addWidget(plot_page)

        # Page 1: Peak list table
        peak_page = QWidget()
        pk_layout = QVBoxLayout(peak_page)
        pk_layout.setContentsMargins(0, 0, 0, 0)
        self._peak_table = QTableWidget()
        self._peak_table.setColumnCount(7)
        self._peak_table.setHorizontalHeaderLabels(
            ["Isotope", "State", "Shift (MHz)", "Offset (MHz)",
             "dV (V)", "V_acc (V)", "Intensity"])
        self._peak_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._peak_table.setSortingEnabled(True)
        self._peak_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._peak_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        pk_layout.addWidget(self._peak_table)
        self._stack.addWidget(peak_page)

        self._plot_results = None
        self._all_peaks = []
        self._palette = "default"
        self._pdf_paths = {}
        self._editor_dialog = None

        # Redraw at the new width when the hosting column is resized
        # (debounced — a splitter drag fires dozens of resize events).
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self.refresh_current)

        # Show placeholder
        self._show_placeholder()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start()

    def eventFilter(self, obj, event):
        if obj is self.canvas and event.type() == QEvent.Type.Wheel:
            QCoreApplication.sendEvent(self.scroll.viewport(), event)
            return True
        return super().eventFilter(obj, event)

    def _Fit_Canvas(self):
        """Size the canvas widget to the figure. Inches convert to widget
        pixels at the *logical* dpi: matplotlib's Qt canvas sets
        ``figure.dpi`` to 100 times the screen's pixel ratio, so the raw
        dpi doubled every plot on a Retina screen and grew them at
        125-150 % scaling on Windows (2026-10-22)."""
        dpi = self.figure.get_dpi() / self.canvas.device_pixel_ratio
        self.canvas.setFixedSize(int(self.figure.get_figwidth() * dpi),
                                 int(self.figure.get_figheight() * dpi))

    def _fig_width_inches(self):
        """Figure width that fills the scroll viewport (no horizontal
        scrolling in the embedded column; grows in expand mode). Logical
        dpi, as in _Fit_Canvas."""
        dpi = self.figure.get_dpi() / self.canvas.device_pixel_ratio
        vw = self.scroll.viewport().width()
        if vw < 50:  # not laid out yet
            vw = 900
        return max((vw - 4) / dpi, 5.0)

    def refresh_current(self):
        """Redraw the current view (e.g. after the panel was resized
        by the expand toggle)."""
        self._on_selection()

    def _show_placeholder(self):
        self.figure.clear()
        # Reset figure size for placeholder
        self.figure.set_size_inches(self._fig_width_inches(), 6)
        self._Fit_Canvas()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Run a calculation to see plots here",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
        self.edit_plot_btn.setEnabled(False)

    def display_results(self, output_dir, config_name, file_tag, palette):
        """Load plot data and render natively on the canvas."""
        from cls_estimations.plotting import set_palette
        self._palette = palette
        set_palette(palette)

        prefix = f"{config_name}_" if config_name else ""
        self._pdf_paths = {
            0: os.path.join(output_dir, f"{prefix}hfs_spectra{file_tag}.pdf"),
            1: os.path.join(output_dir, f"{prefix}overview{file_tag}.pdf"),
        }
        self.open_pdf_btn.setEnabled(
            any(os.path.exists(p) for p in self._pdf_paths.values()))

        # Plot arrays are not reconstructed here: run() does not export
        # plot data to disk, so the MainWindow stores the in-memory
        # results via set_plot_data() after the run. This call refreshes
        # the view from whatever is currently held (results or PDFs).
        self._on_selection()

    def set_plot_data(self, all_plot_results):
        """Store plot data from the worker thread for native rendering."""
        self._plot_results = all_plot_results
        self._on_selection()

    def set_peak_data(self, all_peaks):
        """Store peak data and populate the peak table."""
        self._all_peaks = all_peaks or []
        self._populate_peak_table()

    def _populate_peak_table(self):
        """Fill the peak table from stored peak data."""
        self._peak_table.setSortingEnabled(False)
        self._peak_table.setRowCount(len(self._all_peaks))
        for row, pk in enumerate(self._all_peaks):
            self._peak_table.setItem(
                row, 0, QTableWidgetItem(str(pk.get("label", ""))))
            self._peak_table.setItem(
                row, 1, QTableWidgetItem(str(pk.get("state", ""))))
            # Numeric columns: use a sortable item
            for col, key, fmt in [
                (2, "iso_shift_MHz", "{:+.3f}"),
                (3, "offset_MHz", "{:+.1f}"),
                (4, "dV", "{:+.3f}"),
                (5, "V_acc", "{:.3f}"),
                (6, "intensity", "{:.4f}"),
            ]:
                val = pk.get(key)
                item = _NumericTableItem()
                if val is not None:
                    item.setText(fmt.format(val))
                    item.setData(Qt.ItemDataRole.UserRole, float(val))
                else:
                    item.setText("N/A")
                    item.setData(Qt.ItemDataRole.UserRole, float('-inf'))
                self._peak_table.setItem(row, col, item)
        self._peak_table.setSortingEnabled(True)

    def _draw_individual(self):
        """Draw individual spectra natively on the canvas figure."""
        from cls_estimations.plotting import _get_color, \
            plot_isotope_spectrum

        results = self._plot_results
        if not results:
            self._show_placeholder()
            return

        subplot_entries = []
        for res in results:
            subplot_entries.append(("gs", res))
            if "isomer_label" in res and not res.get("plot_with_gs", True):
                subplot_entries.append(("isomer_only", res))

        n_plots = len(subplot_entries)
        self.figure.clear()
        h = 3.5 * n_plots
        self.figure.set_size_inches(self._fig_width_inches(), max(h, 4))

        axes = self.figure.subplots(n_plots, 1, squeeze=False)
        axes = axes.flatten()

        color_idx = 0
        plot_idx = 0
        for entry_type, res in subplot_entries:
            ax = axes[plot_idx]
            c_gs = _get_color(color_idx)

            if entry_type == "gs":
                plot_isotope_spectrum(
                    ax, res["dV_array"], res["intensity_array"],
                    res["label"], res.get("measured_peak_dVs"),
                    color=c_gs,
                )
                if "isomer_label" in res and res.get("plot_with_gs", True):
                    c_iso = _get_color(color_idx + 1)
                    plot_isotope_spectrum(
                        ax, res["isomer_dV"], res["isomer_intensity"],
                        res["isomer_label"], res.get("isomer_measured_dVs"),
                        color=c_iso, alpha=0.5,
                    )
            else:
                c_iso = _get_color(color_idx + 1)
                plot_isotope_spectrum(
                    ax, res["isomer_dV"], res["isomer_intensity"],
                    res["isomer_label"], res.get("isomer_measured_dVs"),
                    color=c_iso,
                )

            ax.set_xlabel("Voltage offset $\\Delta V$ (V)")
            ax.legend()
            plot_idx += 1
            color_idx += 2

        self.figure.tight_layout()
        self._Fit_Canvas()
        self.canvas.draw()
        self.edit_plot_btn.setEnabled(True)
        if (self._editor_dialog is not None
                and self._editor_dialog.isVisible()):
            self._editor_dialog.refresh()

    def _draw_overview(self):
        """Draw combined overview natively on the canvas figure."""
        from cls_estimations.plotting import _get_color

        results = self._plot_results
        if not results:
            self._show_placeholder()
            return

        self.figure.clear()
        self.figure.set_size_inches(self._fig_width_inches(), 6)
        ax = self.figure.add_subplot(111)
        color_idx = 0

        for res in results:
            c = _get_color(color_idx)
            ax.plot(res["dV_array"], res["intensity_array"],
                    color=c, label=res["label"], linewidth=1.0)
            if res.get("measured_peak_dVs"):
                for v in res["measured_peak_dVs"]:
                    ax.axvline(v, color=c, linestyle="--",
                               alpha=0.35, linewidth=0.7)

            if "isomer_label" in res:
                c2 = _get_color(color_idx + 1)
                ax.plot(res["isomer_dV"], res["isomer_intensity"],
                        color=c2, linestyle="-", alpha=0.5,
                        label=res["isomer_label"], linewidth=1.0)
                if res.get("isomer_measured_dVs"):
                    for v in res["isomer_measured_dVs"]:
                        ax.axvline(v, color=c2, linestyle="--",
                                   alpha=0.25, linewidth=0.7)

            color_idx += 2

        ax.set_xlabel("Voltage offset $\\Delta V$ (V)")
        ax.set_ylabel("Normalised intensity")
        ax.legend()
        self.figure.tight_layout()
        self._Fit_Canvas()
        self.canvas.draw()
        self.edit_plot_btn.setEnabled(True)
        if (self._editor_dialog is not None
                and self._editor_dialog.isVisible()):
            self._editor_dialog.refresh()

    def _render_pdf(self, path):
        """Render every page of ``path`` INTO the matplotlib figure
        (stacked, axes hidden) so the loaded plot gets the standard
        canvas controls — pan, zoom, save — from the toolbar. Returns
        True when at least one page was rendered."""
        try:
            from PySide6.QtPdf import QPdfDocument
        except ImportError:
            return False
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QImage
        # Unparented + local on purpose: the document holds an OS file
        # handle that is only released when the object is destroyed
        # (close() does NOT release it), and an open handle keeps the
        # whole run folder undeletable on Windows. Destroying it on
        # function exit costs one ~20 kB re-read per redraw.
        doc = QPdfDocument()
        if doc.load(path) != QPdfDocument.Error.None_:
            return False
        vw = self.scroll.viewport().width()
        if vw < 50:  # not laid out yet
            vw = 900
        vw = max(vw - 4, 400)
        # Rasterize at 2x the display width so toolbar zoom has
        # resolution headroom before pixelation (cap for memory).
        render_w = min(vw * 2, 2600)
        pages = []
        for i in range(doc.pageCount()):
            size = doc.pagePointSize(i)
            if size.width() <= 0:
                continue
            scale = render_w / size.width()
            img = doc.render(i, QSize(int(size.width() * scale),
                                      int(size.height() * scale)))
            if img.isNull():
                continue
            img = img.convertToFormat(QImage.Format.Format_RGBA8888)
            h, w = img.height(), img.width()
            buf = np.frombuffer(img.constBits(), dtype=np.uint8,
                                count=h * img.bytesPerLine())
            arr = (buf.reshape(h, img.bytesPerLine())[:, :w * 4]
                   .reshape(h, w, 4).copy())
            pages.append(arr)
        if not pages:
            return False

        fig_w = vw * self.canvas.device_pixel_ratio / self.figure.get_dpi()
        # Each page keeps its own aspect ratio at full column width.
        heights = [fig_w * (a.shape[0] / a.shape[1]) for a in pages]
        self.figure.clear()
        self.figure.set_size_inches(fig_w, max(sum(heights), 2.0))
        axes = self.figure.subplots(
            len(pages), 1, squeeze=False,
            gridspec_kw={"height_ratios": heights, "hspace": 0.01,
                         "left": 0, "right": 1, "top": 1, "bottom": 0})
        for ax, arr in zip(axes.flatten(), pages):
            ax.imshow(arr, aspect="auto")
            ax.set_axis_off()
        self._Fit_Canvas()
        self.canvas.draw()
        return True

    def _on_selection(self):
        idx = self.plot_selector.currentIndex()
        if idx == 2:
            # Peak List view
            self._stack.setCurrentIndex(1)
            self.edit_plot_btn.setEnabled(False)
            return

        self._stack.setCurrentIndex(0)
        if self._plot_results:
            if idx == 0:
                self._draw_individual()
            else:
                self._draw_overview()
        elif self._pdf_paths:
            # No in-memory arrays (run loaded from a folder that
            # pre-dates estimate_results.npz): render the saved PDF
            # onto the canvas (toolbar pan/zoom/save keep working),
            # falling back to a text hint only if it cannot be read.
            self.edit_plot_btn.setEnabled(False)
            path = self._pdf_paths.get(idx, "")
            if path and os.path.exists(path) and self._render_pdf(path):
                return
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            if path and os.path.exists(path):
                ax.text(0.5, 0.5,
                        f"Plot saved at:\n{path}\n\n"
                        "Click 'Open PDF' to view",
                        ha="center", va="center", fontsize=11, color="gray",
                        transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No plot data available",
                        ha="center", va="center", fontsize=11, color="gray",
                        transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.canvas.draw()
        else:
            # Refit to the column after a resize or the expand toggle.
            self._show_placeholder()

    def _open_current_pdf(self):
        idx = self.plot_selector.currentIndex()
        path = self._pdf_paths.get(idx, "")
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_editor(self):
        if (self._editor_dialog is not None
                and self._editor_dialog.isVisible()):
            self._editor_dialog.raise_()
            self._editor_dialog.activateWindow()
            return
        self._editor_dialog = PlotEditorDialog(
            self.figure, self.canvas, parent=self)
        self._editor_dialog.show()


# ══════════════════════════════════════════════════════════════════
#  Voltage Range Estimator
# ══════════════════════════════════════════════════════════════════
class Voltage_Range_Estimator(QGroupBox):
    """Setpoint map: where each hyperfine component resonates (dV from
    the beam voltage) as the laser setpoint moves, which components fall
    inside the +/-dV sweep, and the spectrum at the setpoint.

    Every number comes from the pipeline's own functions, fed the dict
    Run feeds it (``Config_Source``, i.e. MainWindow._build_estimate_dict),
    so the map agrees with Run. A finished edit (Enter or
    focus-out, drag release, 400 ms after the last arrow key) writes
    Laser > Setpoint or Scan > Voltage range = 2*dV, undoably. It builds
    only while ``Live`` (after the first Run); until then it shows a
    placeholder. Any parameter change emits ``Config_Changed``, which
    MainWindow uses to re-run the pipeline live. Optimise fits the
    setpoint and sweep to the peaks chosen under Peaks. The map
    zooms with the wheel and pans when dragged off the bar; that view
    survives commits and refreshes until a double-click refits it.
    """
    Config_Changed = Signal()

    def __init__(self, global_params, isotope_list, parent=None):
        super().__init__("Voltage Range Estimator", parent)
        self.Config_Source = None
        self._Global_Params = global_params
        self._Isotope_List = isotope_list
        self._Snapshot = None
        self._Cfg = None            # parsed config of the last build
        self._Hfs_Cache = {}        # compute_hfs_lines results, by input
        self._Mass_Table = isotope_list._mass_table   # load_mass_table()
        self._Dirty = True          # build on first show
        self._Dragging = False
        self._Pan = None            # (x, y, xlim, ylim) at the press
        self._Zoom = None           # user's (xlim, ylim) on the map
        self._S = 0.0               # setpoint on the map (cm-1)

        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.canvas.setToolTip(
            "Drag the red bar to move the setpoint, or click the map and use "
            "the arrow keys (Shift for larger steps). Scroll to zoom, drag "
            "elsewhere to pan, double-click to refit.")
        for name, slot in (("button_press_event", self._On_Press),
                           ("motion_notify_event", self._On_Motion),
                           ("button_release_event", self._On_Release),
                           ("scroll_event", self._On_Scroll),
                           ("key_press_event", self._On_Key)):
            self.canvas.mpl_connect(name, slot)

        self.Show_Combo = QComboBox()
        # The peaks Optimise must cover: the pipeline's own choice (each
        # state's strongest "Peaks to measure", the ones the run time
        # counts) until components are ticked in the menu.
        self._Picked = None         # None, or {(state label, index)}
        self.Peaks_Btn = QPushButton("To measure", enabled=False)
        self.Peaks_Btn.setMenu(QMenu(self.Peaks_Btn,
                                     aboutToShow=self._Fill_Peaks_Menu))
        self.Peaks_Btn.setToolTip(
            "The peaks the sweep must cover: each state's strongest "
            "\"Peaks to measure\" (the ones the run time counts), all "
            "components, or the components you tick per state.")
        self.Optimise_Btn = QPushButton("Optimise", enabled=False,
                                        clicked=self._Optimise)
        # The gear, in a popover rather than more form rows: the baseline
        # Optimise leaves beyond the outer peaks, and a readout offset that
        # moves what is drawn here (not the axes) to match a DAQ readout
        # shift (2026-10-24). Neither touches the physics or the Plots and
        # Log.
        self.Margin_Box = _make_double(OPTIMISE_MARGIN_V, 0, 10000, 1, 10)
        self.Offset_Box = _make_double(0, -10000, 10000, 2, 1)
        for box in (self.Margin_Box, self.Offset_Box):
            box.setSuffix(" V")
            box.setKeyboardTracking(False)
            box._Own_Undo = True    # a preference, not a parameter
        self.Margin_Box.valueChanged.connect(self._Optimise_Tip)
        self.Margin_Box.valueChanged.connect(self._Draw_View)   # ring fade
        self.Offset_Box.valueChanged.connect(self._Draw_View)
        self.Offset_Box.setToolTip(
            "A known DAQ readout shift: the peaks, rings, spectrum and sweep "
            "band move by it (negative moves them to lower dV), and the "
            "readout's dV and scan range include it. The axes stay put; the "
            "hyperfine model, the Plots and the Log are unchanged.")
        margin = QWidget()
        margin_form = QFormLayout(margin)
        margin_form.setContentsMargins(10, 6, 10, 6)
        margin_form.addRow("Margin beyond the outer peaks (\u00b1):",
                           self.Margin_Box)
        margin_form.addRow("Readout offset:", self.Offset_Box)
        margin_menu = QMenu(self)
        act = QWidgetAction(margin_menu)
        act.setDefaultWidget(margin)
        margin_menu.addAction(act)
        self.Margin_Btn = QPushButton(lucide_icon("settings"), "",
                                      toolTip="Optimise margin and readout "
                                              "offset")
        self.Margin_Btn.setFixedWidth(self.Margin_Btn.sizeHint().height())
        self.Margin_Btn.clicked.connect(lambda: margin_menu.exec(
            self.Margin_Btn.mapToGlobal(self.Margin_Btn.rect().bottomLeft())))
        self._Optimise_Tip()
        self.Setpoint_Box = _make_double(0, 0, 200000, 6, 0.001)
        self.Dv_Box = _make_double(50, 0.05, 50000, 2, 10)
        self.Window_Box = _make_double(0, 0, 100000, 1, 50)
        self.Window_Box.setSpecialValueText("auto")
        for box in (self.Setpoint_Box, self.Dv_Box, self.Window_Box):
            box.setKeyboardTracking(False)
            box._Own_Undo = True    # commits push their own undo step
        self.Readout = QPlainTextEdit(
            readOnly=True, styleSheet=_MONO_QSS,
            lineWrapMode=QPlainTextEdit.LineWrapMode.NoWrap)

        form = QFormLayout()
        form.addRow("Show:", self.Show_Combo)
        peaks_row = QHBoxLayout()
        peaks_row.addWidget(self.Peaks_Btn, 1)
        peaks_row.addWidget(self.Optimise_Btn)
        peaks_row.addWidget(self.Margin_Btn)
        form.addRow("Peaks:", peaks_row)
        form.addRow("Setpoint (cm\u207b\u00b9):", self.Setpoint_Box)
        form.addRow("dV (\u00b1V):", self.Dv_Box)
        form.addRow("Window (\u00b1V):", self.Window_Box)
        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(self.Readout, 1)
        row = QHBoxLayout(self)
        row.addWidget(self.canvas, 1)
        row.addLayout(side)

        self.Show_Combo.currentIndexChanged.connect(self._Reset_View)
        self.Setpoint_Box.editingFinished.connect(self._On_Setpoint_Edited)
        self.Dv_Box.editingFinished.connect(self._Commit_Dv)
        self.Window_Box.valueChanged.connect(self._Reset_View)
        self._Refresh_Timer = QTimer(self, singleShot=True, interval=250,
                                     timeout=self._Refresh)
        self._Key_Timer = QTimer(self, singleShot=True, interval=400,
                                 timeout=self._Commit_Setpoint)
        self._Hook(global_params)
        isotope_list.order_changed.connect(self._On_Isotopes_Changed)
        self.Set_Live(False)

    # ── Refresh from the parameters ─────────────────────────────

    def _Hook(self, root):
        """Refresh after any finished edit under ``root``, once per root.
        A spin box also refreshes on a step (arrows, wheel) or a change from
        code: with editingFinished alone the map ignored those until the box
        lost focus (2026-10-24). Its text is "modified" only while typed, so
        a half-typed number still waits for Enter or focus-out."""
        if getattr(root, "_Estimator_Hooked", False):
            return
        root._Estimator_Hooked = True
        for w in root.findChildren(QLineEdit):
            w.editingFinished.connect(self.Schedule_Refresh)
        for w in root.findChildren(QAbstractSpinBox):
            w.editingFinished.connect(self.Schedule_Refresh)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(
                    lambda *_, w=w: w.lineEdit().isModified()
                    or self.Schedule_Refresh())
        for w in root.findChildren(QComboBox):
            w.currentIndexChanged.connect(self.Schedule_Refresh)
        for w in root.findChildren(QAbstractButton):
            if w.isCheckable():
                w.toggled.connect(self.Schedule_Refresh)

    def _On_Isotopes_Changed(self):
        for panel in self._Isotope_List.panels:
            self._Hook(panel)
        self.Schedule_Refresh()

    def Schedule_Refresh(self, *_):
        """Rebuild from the parameters 250 ms after the last trigger,
        only if they changed, and not while the pane is hidden."""
        self._Refresh_Timer.start()

    def Set_Live(self, on):
        """Build from the parameters (after a Run), or show the "press
        Run" placeholder (at start, and after a Load or a config load)."""
        self.Live, self._Dirty = on, True
        if on:
            self.Schedule_Refresh()
        else:
            self._Show_Placeholder(
                "Press Run to show the voltage range estimator.")

    def showEvent(self, event):
        super().showEvent(event)
        if self._Dirty:
            self.Schedule_Refresh()

    def _Refresh(self):
        """Emit Config_Changed when the parameters changed, before the
        visibility check: an undo while the plots are expanded must still
        reach the live run. Build only while live and shown."""
        if self.Config_Source is None:
            return
        d = self.Config_Source()
        d.pop("last_run", None)
        if d != self._Snapshot:
            self._Snapshot, self._Dirty = d, True
            self.Config_Changed.emit()
        if not (self._Dirty and self.Live and self.isVisible()):
            return
        self._Dirty = False
        if "satlas2" in sys.modules or "reference" not in d:
            self._Rebuild()
        else:
            # The first real build imports satlas2 (about 2 s, once):
            # paint a note first, then build on the next tick.
            self._Show_Placeholder("Preparing hyperfine model...")
            QTimer.singleShot(0, self._Rebuild)

    def _Rebuild(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._Build_Model(self._Snapshot)
        except Exception as exc:  # noqa: BLE001 -- any incomplete config
            self._Cfg = None
            self._Show_Placeholder(
                "Needs an element, a transition, at least one isotope and "
                f"a reference isotope.\n\n{type(exc).__name__}: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        combo, keep = self.Show_Combo, self.Show_Combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(["All"] + [st["label"] for st in self._States])
        combo.setCurrentIndex(max(0, combo.findText(keep)))
        combo.blockSignals(False)
        self._S = self._Cfg.laser.setpoint_cm
        for box, value in ((self.Setpoint_Box, self._S),
                           (self.Dv_Box, self._Cfg.scan.voltage_range_V / 2)):
            if not box.hasFocus():
                box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(False)
        self._Draw_View()

    def _Build_Model(self, d):
        """Parse ``d`` as Run does and collect each state's components."""
        import tempfile
        import yaml
        # The pipeline's single source of the A/B scaling and the peak
        # voltages; copying them would fork the physics.
        from cls_estimate import _get_hfs_params
        from cls_estimations.config_parser import load_config
        # load_config takes only a path. Close the descriptor before
        # loading: Windows can't reopen a file that is still open.
        fd, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as fh:
                yaml.safe_dump(d, fh)
            cfg = load_config(path)
        finally:
            os.unlink(path)
        tr = cfg.transition
        nu0 = transition_frequency_MHz(tr.lower_level_cm, tr.upper_level_cm)
        self._Cfg, self._V0 = cfg, cfg.beam.voltage_kV * 1e3  # V0 as in run()
        self._States = []
        for i, iso in enumerate(cfg.isotopes):
            # Ground state solid, isomer dashed, labelled as run() does.
            for isomer in [None] + [iso.isomer] * (iso.isomer is not None):
                A_l, B_l, A_u, B_u = _get_hfs_params(
                    iso, cfg.reference, is_isomer=isomer is not None,
                    isomer_cfg=isomer, g_s_factor=cfg.g_s_factor)[:4]
                state = isomer or iso
                key = (state.I, tr.J_lower, tr.J_upper, A_l, B_l, A_u, B_u)
                if key not in self._Hfs_Cache:
                    self._Hfs_Cache[key] = compute_hfs_lines(*key)
                self._States.append(dict(
                    label=(iso.label if iso.isomer is None
                           else f"{iso.label} (g.s.)" if isomer is None
                           else f"{iso.label} (isomer, I={isomer.I})"),
                    ls="--" if isomer else "-",
                    mass=get_mass(self._Mass_Table, cfg.Z, iso.A),
                    color=ISOTOPE_COLORS[i % len(ISOTOPE_COLORS)],
                    # centroid = nu0 + isotope (or isomer) shift, as in run()
                    centroid=nu0 + state.isotope_shift_MHz,
                    offsets=self._Hfs_Cache[key][0],
                    intens=self._Hfs_Cache[key][1],
                    n_meas=state.peaks_to_measure,
                    I=state.I, shift=state.isotope_shift_MHz,
                    hfs=(A_l, A_u, B_l, B_u)))
        if cfg.laser.setpoint_cm is None:
            # Mirrors the "Resolve anchor" block of cls_estimate.run().
            anchor = next((iso for iso in cfg.isotopes
                           if iso.label == cfg.laser.anchor_isotope), None)
            state = anchor and (anchor.isomer if cfg.laser.anchor_state
                                == "isomer" else anchor)
            if state is None:
                raise ValueError(f"Anchor '{cfg.laser.anchor_isotope}' "
                                 f"({cfg.laser.anchor_state}) not found.")
            cfg.laser.setpoint_cm = self._Setpoint_For(
                nu0 + state.isotope_shift_MHz, self._V0 - cfg.laser.anchor_dV,
                get_mass(self._Mass_Table, cfg.Z, anchor.A))

    # ── Physics, through the pipeline's functions ───────────────

    def _Nu_Laser(self, s):
        """nu_laser_MHz at setpoint ``s``, as in cls_estimate.run()."""
        return s * self._Cfg.laser.harmonic * CM_TO_HZ / 1e6

    def _Setpoint_For(self, nu_rest, v_acc, mass):
        """Setpoint that puts ``nu_rest`` in resonance at ``v_acc``: the
        anchor conversion of cls_estimate.run()."""
        cfg = self._Cfg
        nu = float(laser_frequency_for_resonance(
            nu_rest, v_acc, mass, cfg.beam.charge_state, cfg.beam.geometry))
        return nu * 1e6 / (cfg.laser.harmonic * CM_TO_HZ)

    def _Peak_Dvs(self, st, s):
        """The pipeline's dV of each component, plus the readout offset:
        dV as the DAQ shows it. The offset is added here and nowhere in the
        physics."""
        from cls_estimate import _compute_peak_voltages
        cfg, off = self._Cfg, self.Offset_Box.value()
        return [None if d is None else d + off for d in _compute_peak_voltages(
            st["offsets"], st["centroid"], self._Nu_Laser(s), st["mass"],
            cfg.beam.charge_state, self._V0, cfg.beam.geometry)]

    def _Shown(self):
        i = self.Show_Combo.currentIndex()
        return self._States if i <= 0 else self._States[i - 1:i]

    def Model_Source(self):
        """``(parsed config, states)`` of the last build while live, for
        Pre-Analysis's "Add Model" menu; None before the first Run."""
        if self.Live and self._Cfg is not None:
            return self._Cfg, self._States
        return None

    def _Weight(self, d):
        """How safely the sweep covers a crossing at DAQ dV ``d``: 1 in its
        core, falling linearly to 0 across the margin at either edge, 0
        outside or unresolved. The rings and the readout's coverage use it:
        the margin regions are riskier to scan over."""
        if d is None:
            return 0.0
        x = self.Dv_Box.value() - abs(d - self.Offset_Box.value())
        m = self.Margin_Box.value()
        return 0.0 if x < 0 else 1.0 if m <= 0 or x >= m else x / m

    def _Side_Widgets(self):
        return (self.Show_Combo, self.Peaks_Btn, self.Optimise_Btn,
                self.Setpoint_Box, self.Dv_Box, self.Window_Box)

    def _Selected(self, st):
        """Indices of ``st``'s components Optimise must cover. By default
        the strongest "Peaks to measure", in the order the pipeline's
        estimate_isotope_timing picks them; no count means all."""
        if self._Picked is None:
            i, n = st["intens"], st["n_meas"]
            return set(sorted(range(len(i)), key=lambda k: i[k],
                              reverse=True)[:len(i) if n is None else n])
        return {k for label, k in self._Picked if label == st["label"]}

    # ── Peak selection and Optimise ─────────────────────────────

    def _Fill_Peaks_Menu(self):
        """The pipeline's choice, all components, or components ticked per
        shown state, strongest first."""
        menu = self.Peaks_Btn.menu()
        for sub in menu.findChildren(QMenu):
            sub.deleteLater()
        menu.clear()
        default = menu.addAction("Peaks to measure", lambda: self._Pick(None))
        default.setCheckable(True)
        default.setChecked(self._Picked is None)
        menu.addAction("All components", lambda: self._Pick(
            {(st["label"], k) for st in self._States
             for k in range(len(st["offsets"]))}))
        menu.addSeparator()
        for st in self._Shown():
            # Parented explicitly: PySide deleted a submenu made by
            # addMenu(title) once its action wrapper was collected.
            sub = QMenu(st["label"], menu)
            menu.addMenu(sub)
            sub.installEventFilter(self)    # stays open while ticking
            sel = self._Selected(st)
            for k in sorted(range(len(st["intens"])),
                            key=lambda k: -st["intens"][k]):
                act = sub.addAction(f"{st['offsets'][k]:+.1f} MHz   "
                                    f"{st['intens'][k]:.3f}")
                act.setCheckable(True)
                act.setChecked(k in sel)
                act.toggled.connect(
                    lambda on, st=st, k=k: self._Toggle_Peak(st, k, on))

    def eventFilter(self, obj, event):
        """Tick a Peaks submenu entry without closing the menu."""
        if (isinstance(obj, QMenu)
                and event.type() == QEvent.Type.MouseButtonRelease):
            act = obj.activeAction()
            if act is not None and act.isCheckable():
                act.trigger()
                return True
        return super().eventFilter(obj, event)

    def _Pick(self, picked):
        self._Picked = picked
        self._Draw_View()

    def _Toggle_Peak(self, st, k, on):
        if self._Picked is None:    # start from the pipeline's choice
            self._Picked = {(s["label"], j) for s in self._States
                            for j in self._Selected(s)}
        (self._Picked.add if on else self._Picked.discard)((st["label"], k))
        self._Draw_View()

    def _Optimise_Tip(self, *_):
        self.Optimise_Btn.setToolTip(
            "Set Scan > Voltage range to the DAQ range that covers the "
            "selected peaks of the shown states at this setpoint, with "
            f"{self.Margin_Box.value():g} V to spare on each side (the gear "
            "changes it). The readout lists the range's start and stop.")

    def _Peak_Range(self, s):
        """[start, stop] in DAQ volts (readout offset applied) covering the
        selected peaks of the shown states at setpoint ``s`` plus the
        margin, start down and stop up to whole steps; None if none of
        them resonates. These are the values to enter as a DAQ scanning
        range."""
        d = [v for st in self._Shown() for k, v in
             enumerate(self._Peak_Dvs(st, s)) if k in self._Selected(st)
             and v is not None]
        if not d:
            return None
        step, m = self._Global_Params.scan_step.value(), self.Margin_Box.value()
        return (step * np.floor((min(d) - m) / step),
                step * np.ceil((max(d) + m) / step))

    def _Optimise(self):
        """Scan > Voltage range = the width of the selected peaks' DAQ
        range at the current setpoint (_Peak_Range); one undo step. The
        setpoint is left alone: the user places it, and the readout gives
        the range's start and stop to scan (2026-10-24)."""
        rng = self._Peak_Range(self._S) if self._Map is not None else None
        if rng is None:
            return
        gp = self._Global_Params
        width = min(gp.scan_range.maximum(), rng[1] - rng[0])
        _Set_Spin_Undoable(gp.scan_range, float(width), self)
        self.Dv_Box.setValue(width / 2)
        self._Draw_View()

    # ── Drawing ─────────────────────────────────────────────────

    def _Show_Placeholder(self, text):
        """Grey out the side controls too: before a Run they showed an
        empty Show list and a 0.000000 setpoint (2026-10-24)."""
        self._Map = None
        for w in self._Side_Widgets():
            w.setEnabled(False)
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center",
                         color="gray", wrap=True)
        self.canvas.draw_idle()
        self.Readout.clear()

    def _Draw_View(self, *_):
        """Full redraw. The x-limits span the setpoints where the shown
        components enter and leave +/-Window, plus 5 %, and always the
        setpoint. Window "auto" fits the components at the setpoint with
        a 10 % margin and is never narrower than +/-2*dV."""
        if self._Cfg is None:
            return
        shown, s, dv = self._Shown(), self._S, self.Dv_Box.value()
        win = self.Window_Box.value()
        if not win:
            # From the model's dV, so the readout offset never moves the
            # axes, only what is drawn on them.
            off = self.Offset_Box.value()
            dvs = [abs(v - off) for st in shown
                   for v in self._Peak_Dvs(st, s) if v is not None]
            win = max(1.1 * max(dvs, default=0.0), 2 * dv)
        xs = [s] + [self._Setpoint_For(st["centroid"] + o, self._V0 + w,
                                       st["mass"])
                    for st in shown for o in st["offsets"]
                    for w in (-win, win)]
        pad = 0.05 * (max(xs) - min(xs)) or 1e-3
        fit = (min(xs) - pad, max(xs) + pad)
        # The FWHM in volts, through the pipeline's conversion, so _Draw_At
        # samples finely enough to resolve every peak.
        f = self._Cfg.linewidth_fwhm_MHz / 2
        fws = [abs(a - b) for a, b in (self._Peak_Dvs(dict(
            st, offsets=[st["offsets"][0] - f, st["offsets"][0] + f]), s)
            for st in shown) if a is not None and b is not None]
        self._Fwhm_V = min(fws, default=0.0)
        # The lines also cover a zoomed-out or panned view. They are near
        # straight, so 60 points stay accurate (to ~0.03 V) at any zoom.
        self._Span = (min(fit[0], self._Zoom[0][0]) if self._Zoom else fit[0],
                      max(fit[1], self._Zoom[0][1]) if self._Zoom else fit[1])
        grid = np.linspace(*self._Span, 60)
        self.figure.clear()
        self._Map, spec = self.figure.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": (3, 1)})
        self._Centres = []
        for st in shown:
            lines = [[np.nan if v is None else v
                      for v in self._Peak_Dvs(st, x)] for x in grid]
            sel = self._Selected(st)
            for k, ln in enumerate(self._Map.plot(
                    grid, lines, color=st["color"], ls=st["ls"], lw=1)):
                # Unselected lines stay traceable but read as background.
                ln.set(**({"lw": 1.8} if k in sel else {"alpha": 0.1}))
            x_c = self._Setpoint_For(st["centroid"], self._V0, st["mass"])
            self._Centres.append(x_c)
            # Unlabelled: the rotated "centre" labels cluttered the map
            # (2026-10-24); the marker's colour names the state.
            self._Map.axvline(x_c, color=st["color"], ls=":", lw=0.8)
        off = self.Offset_Box.value()       # the sweep moves with the peaks
        for ax in (self._Map, spec):
            ax.axhspan(off - dv, off + dv, color="gold", alpha=0.2)
        self._Map.set_xlim(self._Zoom[0] if self._Zoom else fit)
        # Setpoint ticks are long: few of them, and no offset text, which
        # landed on the axis label.
        self._Map.locator_params(axis="x", nbins=4)
        self._Map.ticklabel_format(axis="x", useOffset=False)
        self._Map.set_ylim(self._Zoom[1] if self._Zoom else (-win, win))
        self._Map.set_xlabel("Setpoint (cm$^{-1}$, laser = "
                             f"x{self._Cfg.laser.harmonic})")
        self._Map.set_ylabel("dV (V)")
        spec.set_xlim(0, 1.05)
        # On top: below, it collided with the map's wide setpoint label.
        spec.set_xlabel("Relative\nintensity", fontsize="small")
        spec.xaxis.set_label_position("top")
        self._Bar = self._Map.axvline(s, color="red", lw=1.5)
        self._Rings = self._Map.scatter([], [], s=81, facecolors="none",
                                        edgecolors="red", linewidths=1.5,
                                        zorder=3)
        self._Spectra = [spec.plot([], [], color=st["color"], ls=st["ls"])[0]
                         for st in shown]
        n = sum(len(self._Selected(st)) for st in shown)
        self.Peaks_Btn.setText(
            f"To measure ({n})" if self._Picked is None else
            f"All ({n})" if n == sum(len(st["offsets"]) for st in shown)
            else f"Selected ({n})")
        for w in self._Side_Widgets():
            w.setEnabled(w is not self.Optimise_Btn or n > 0)
        self._Draw_At(s)

    def _Draw_At(self, s):
        """Move the bar to setpoint ``s``; rings, spectrum and readout
        follow live (no satlas2 call, no rescale). Red rings mark only the
        selected peaks' crossings inside the sweep, which moves with the
        readout offset like the peaks. The spectrum is sampled over the visible dV
        range and every component, at a tenth of the FWHM: on a 600-step
        grid of the visible range alone, the peak heights jumped as the
        setpoint moved (2026-10-24)."""
        cfg, self._S, dv = self._Cfg, s, self.Dv_Box.value()
        self._Bar.set_xdata([s, s])
        rows, inside, off = [], [], self.Offset_Box.value()
        lo, hi, fw = *self._Map.get_ylim(), self._Fwhm_V
        for st, line in zip(self._Shown(), self._Spectra):
            dvs, sel = self._Peak_Dvs(st, s), self._Selected(st)
            inside += [d for k, d in enumerate(dvs)
                       if k in sel and d is not None and abs(d - off) <= dv]
            # The spectrum is computed in the pipeline's dV and drawn
            # shifted by the readout offset, like everything else here.
            pk = [d - off for d in dvs if d is not None]
            rng = (min([lo - off] + pk) - 3 * fw, max([hi - off] + pk) + 3 * fw)
            v, y = generate_voltage_spectrum(
                st["offsets"], st["intens"], st["centroid"],
                self._Nu_Laser(s), st["mass"], cfg.beam.charge_state,
                self._V0, cfg.beam.geometry, cfg.linewidth_fwhm_MHz, rng,
                v_steps=int(min(20000, max(
                    600, 10 * (rng[1] - rng[0]) / (fw or np.inf)))))
            line.set_data(y, v + off)
            rows += [(st["label"], o, amp, d, k in sel) for k, (o, amp, d)
                     in enumerate(zip(st["offsets"], st["intens"], dvs))]
        # Fainter towards the scan edge: crossings in the margin are riskier
        # to measure (2026-10-24).
        self._Rings.set_offsets(np.column_stack(
            ([s] * len(inside), inside)) if inside else np.empty((0, 2)))
        self._Rings.set_edgecolors(
            [(1, 0, 0, max(0.1, self._Weight(d))) for d in inside])
        self._Write_Readout(s, rows)
        self.canvas.draw_idle()

    def _Write_Readout(self, s, rows):
        cfg, dv, off = self._Cfg, self.Dv_Box.value(), self.Offset_Box.value()
        h, w = cfg.laser.harmonic, max([5] + [len(r[0]) for r in rows])
        rng = self._Peak_Range(s)
        nm = [f"{1e7 / cm:.3f} nm" if cm else "-" for cm in (s, s * h)]
        text = [f"Setpoint {s:.6f} cm-1  {nm[0]}",
                f"x{h:<7} {s * h:.6f} cm-1  {nm[1]}",
                f"Laser    {self._Nu_Laser(s) / 1e6:.6f} THz",
                f"V0       {self._V0:.1f} V  {cfg.beam.geometry}",
                f"Sweep    \u00b1{dv:.2f} V (scan range {2 * dv:.1f} V)",
                # Short: the readout's widest line sets its width.
                *([f"Offset   {off:+.2f} V (DAQ readout)"] if off else []),
                # The DAQ scanning range for the selected peaks.
                f"Peaks    [{rng[0]:.1f}, {rng[1]:.1f}] V to scan" if rng
                else "Peaks    no selected peak resonates",
                "", f"  {'State':<{w}}{'MHz':>9}{'dV (V)':>10}{'Int':>8}"]
        n_in = n_sel = cover = 0
        for label, o, amp, d, sel in sorted(
                rows, key=lambda r: np.inf if r[3] is None else r[3]):
            ok, w_ = d is not None and abs(d - off) <= dv, self._Weight(d)
            n_in, n_sel, cover = n_in + ok, n_sel + sel, cover + sel * w_
            d_txt = "N/A" if d is None else f"{d:+.3f}"
            mark = "\u2022" if sel else " "    # the peaks Optimise covers
            text.append(f"{mark} {label:<{w}}{o:>+9.1f}"
                        f"{d_txt:>10}{amp:>8.4f}  "
                        f"{'OUT' if not ok else 'in' if w_ >= 1 else 'edge'}")
        # Selected peaks count by their weight: an "edge" one, in the
        # margin, only partly.
        text.append(f"{n_in}/{len(rows)} components in the sweep, selected "
                    f"coverage {cover:.2f}/{n_sel}")
        self.Readout.setPlainText("\n".join(text))
        # Wide enough for the longest line (frame, margins, scrollbar).
        fm = self.Readout.fontMetrics()
        self.Readout.setMinimumWidth(
            32 + max(fm.horizontalAdvance(t) for t in text))

    # ── Interaction and write-back ──────────────────────────────

    def _On_Press(self, event):
        if (self._Map is None or event.inaxes is not self._Map
                or event.button != 1):
            return
        if event.dblclick:
            self._Reset_View()
            return
        # Within 6 px of the bar: drag it; of a centre marker: snap to it;
        # elsewhere: pan. matplotlib reports event positions in physical
        # pixels, hence the pixel ratio.
        tol = 6 * self.canvas.device_pixel_ratio

        def near(x):
            return abs(self._Map.transData.transform((x, 0))[0]
                       - event.x) <= tol
        if near(self._S):
            self._Dragging = True
            return
        for x_c in self._Centres:
            if near(x_c):
                self._Draw_At(x_c)
                self._Commit_Setpoint()
                return
        self._Pan = (event.x, event.y, self._Map.get_xlim(),
                     self._Map.get_ylim())

    def _On_Motion(self, event):
        if self._Dragging and event.inaxes is self._Map:
            self._Draw_At(event.xdata)
        elif self._Pan:
            px, py, (x0, x1), (y0, y1) = self._Pan
            bb = self._Map.bbox
            dx = (event.x - px) * (x1 - x0) / bb.width
            dy = (event.y - py) * (y1 - y0) / bb.height
            self._Map.set_xlim(x0 - dx, x1 - dx)
            self._Map.set_ylim(y0 - dy, y1 - dy)
            self.canvas.draw_idle()

    def _On_Release(self, _event):
        if self._Dragging:
            self._Dragging = False
            self._Commit_Setpoint()
        elif self._Pan:
            moved = self._Map.get_xlim() != self._Pan[2]
            moved |= self._Map.get_ylim() != self._Pan[3]
            self._Pan = None
            if moved:
                self._Keep_View()

    def _On_Scroll(self, event):
        if event.inaxes is self._Map:
            scroll_zoom(event)
            self._Keep_View()

    def _Keep_View(self):
        """Keep the user's zoom or pan through later redraws; redraw the
        lines when the view has left the span they cover."""
        x0, x1 = self._Map.get_xlim()
        self._Zoom = ((x0, x1), self._Map.get_ylim())
        if x0 < self._Span[0] or x1 > self._Span[1]:
            self._Draw_View()
        else:
            self._Draw_At(self._S)

    def _Reset_View(self, *_):
        """Back to the fitted view (double-click, or a Window or Show
        change)."""
        self._Zoom = None
        self._Draw_View()

    def _On_Key(self, event):
        step = {"left": -1e-3, "right": 1e-3,
                "shift+left": -1e-2, "shift+right": 1e-2}.get(event.key)
        if step and self._Map is not None:
            self._Draw_At(self._S + step)
            self._Key_Timer.start()     # one commit, one undo step, per burst

    def _On_Setpoint_Edited(self):
        if self.Setpoint_Box.value() != round(self._S, 6):
            self._S = self.Setpoint_Box.value()
            self._Commit_Setpoint()

    def _Commit_Setpoint(self):
        """Write the setpoint to Laser > Setpoint, undoably. In anchor
        mode this ticks the Setpoint radio; the anchor fields keep their
        values, and undo restores the value only."""
        gp = self._Global_Params
        self._S = round(self._S, 6)
        if gp.laser_anchor_radio.isChecked():
            gp.laser_setpoint_radio.setChecked(True)
        _Set_Spin_Undoable(gp.setpoint_cm, self._S, self)
        self.Setpoint_Box.setValue(self._S)
        self._Draw_View()

    def _Commit_Dv(self):
        """Write Scan > Voltage range = 2*dV, undoably."""
        _Set_Spin_Undoable(self._Global_Params.scan_range,
                           2 * self.Dv_Box.value(), self)
        self._Draw_View()


# ══════════════════════════════════════════════════════════════════
#  Estimate Tab (wrapper combining Parameters, Run & Log, Plots)
# ══════════════════════════════════════════════════════════════════
class EstimateTab(QWidget):
    """Top-level Estimate tab — one surface, no sub-tabs.

    Three resizable columns (see :class:`ParametersTab`): element /
    transition parameters, the isotope list with Run Options under it,
    and a right column stacking the plots panel, the Voltage Range
    Estimator and a shortened log. The plots can be expanded over the
    whole tab with the ⛶ toggle.

    Stable handles for callers: ``run_tab`` (the RunTab in column 3),
    ``plots_tab`` (the plots panel hosted inside it) and
    ``Voltage_Estimator``.
    """
    run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.params_tab = ParametersTab()
        layout.addWidget(self.params_tab, 1)
        # Expose the RunTab (hosted in column 3 of ParametersTab) as a
        # top-level ``run_tab`` attribute so callers can reference
        # ``estimate_tab.run_tab.*`` directly; same for the plots
        # panel, which kept its historical ``plots_tab`` name.
        self.run_tab = self.params_tab.run_tab
        self.run_tab.run_requested.connect(self.run_requested.emit)
        self.plots_tab = self.params_tab.plots_panel
        self.plots_tab.expand_btn.toggled.connect(self.set_plots_expanded)
        self.Voltage_Estimator = self.params_tab.Voltage_Estimator

        # Bottom bar
        bottom = QHBoxLayout()
        info_btn = QPushButton(lucide_icon("circle-help"), "Info")
        info_btn.setFixedWidth(75)
        info_btn.clicked.connect(self._show_info)
        bottom.addWidget(info_btn)
        bottom.addStretch()
        layout.addLayout(bottom)

    def ui_layout(self):
        """Adjustable geometry of this tab (see gui.ui_layout)."""
        from gui.ui_layout import collect
        return collect(
            columns=getattr(self.params_tab, "_main_splitter", None),
            plots_log=getattr(self.run_tab, "_vsplit", None))

    def apply_ui_layout(self, d):
        from gui.ui_layout import restore
        restore(d,
                columns=getattr(self.params_tab, "_main_splitter", None),
                plots_log=getattr(self.run_tab, "_vsplit", None))

    def set_plots_expanded(self, expanded):
        """⛶ toggle: hide (or restore) everything except the plots
        panel so the plot display covers the whole tab."""
        pt = self.params_tab
        pt.global_params.setVisible(not expanded)
        pt.isotope_list.setVisible(not expanded)
        self.run_tab._top_host.setVisible(not expanded)
        self.Voltage_Estimator.setVisible(not expanded)
        self.run_tab._log_host.setVisible(not expanded)
        # Redraw at the new width once the layout has settled.
        QTimer.singleShot(0, self.plots_tab.refresh_current)

    def _show_info(self):
        text = (
            "<h3>Run Time Estimation</h3>"
            "Predict CLS measurement times and simulated HFS spectra for "
            "one or more isotopes. The tab is a single surface with "
            "three resizable columns: element / transition on the left, "
            "the isotope list with the run options under it in the "
            "middle, and plots + voltage range estimator + log on the "
            "right. Drag the dividers to resize; the "
            "⛶ button expands the plots over the whole tab.<br><br>"

            "<h4>Element &amp; Transition</h4>"
            "\u2022 <b>Element / Z</b> \u2013 Chemical symbol or proton "
            "number (auto-synced).<br>"
            "\u2022 <b>Lower / Upper level</b> \u2013 Energy levels of the "
            "atomic transition in cm\u207b\u00b9.<br>"
            "\u2022 <b>J_lower / J_upper</b> \u2013 Total angular momentum "
            "quantum numbers of the lower and upper states "
            "(half-integer allowed).<br><br>"

            "<h4>Beam</h4>"
            "\u2022 <b>Voltage</b> \u2013 Ion beam acceleration voltage "
            "(kV), typically 30\u201360 kV.<br>"
            "\u2022 <b>Charge state</b> \u2013 Ion charge q.<br>"
            "\u2022 <b>Geometry</b> \u2013 Anti-collinear (head-on) or "
            "collinear (co-propagating) laser-ion geometry.<br><br>"

            "<h4>Laser</h4>"
            "\u2022 <b>Harmonic</b> \u2013 Laser harmonic multiplier "
            "(1 = fundamental, 2 = SHG, 4 = FHG).<br>"
            "\u2022 <b>Setpoint</b> \u2013 Fundamental laser wavenumber "
            "(cm\u207b\u00b9) entered directly.<br>"
            "\u2022 <b>Anchor to isotope</b> \u2013 Auto-calculate the laser "
            "frequency from a named isotope\u2019s centroid and a voltage "
            "offset \u0394V.<br><br>"

            "<h4>Linewidth &amp; Scan</h4>"
            "\u2022 <b>FWHM</b> \u2013 Total spectral linewidth in MHz "
            "(Doppler + laser).<br>"
            "\u2022 <b>Voltage range</b> \u2013 Total DAC scan range (V).<br>"
            "\u2022 <b>Step size</b> \u2013 Voltage per scan step (V).<br>"
            "\u2022 <b>Dwell time</b> \u2013 Integration time per step "
            "(ms).<br><br>"

            "<h4>Statistics (Timing Mode)</h4>"
            "Enable this section to compute measurement time estimates. "
            "Uncheck for spectrum-only mode (no timing).<br>"
            "\u2022 <b>Required \u03c3</b> \u2013 Statistical significance "
            "threshold (typically 3\u03c3).<br>"
            "\u2022 <b>Uniform timing</b> \u2013 All peaks measured for the "
            "same duration (based on the strongest peak).<br><br>"

            "<h4>Advanced</h4>"
            "\u2022 <b>g_s factor</b> \u2013 Quenching factor for the "
            "nucleon spin g-factor in Schmidt moment calculations "
            "(1.0 = free nucleon, typical 0.6\u20130.8).<br><br>"

            "<h4>Isotopes (middle panel)</h4>"
            "\u2022 Add isotopes with <b>+ Add Isotope</b> and configure "
            "mass number, label, nuclear spin (I), magnetic moment "
            "(\u03bc), quadrupole moment (Q), and isotope shift.<br>"
            "\u2022 <b>Reference</b> \u2013 Mark one isotope as the "
            "reference; its A/B constants scale to other isotopes.<br>"
            "\u2022 <b>Rates &amp; Timing</b> \u2013 Yield (ions/s), "
            "detection efficiency, and number of peaks to measure.<br>"
            "\u2022 <b>Background</b> \u2013 Background count rate (Hz), set as "
            "a <b>Simple rate</b> or <b>TOF-gated</b>; used in the timing "
            "estimate.<br>"
            "\u2022 <b>Isomer</b> \u2013 Add an isomeric state with its own "
            "I, \u03bc, Q, shift, and yield method (production ratio, "
            "independent yield, or spin distribution).<br>"
            "\u2022 <b>Schmidt moment</b> \u2013 Calculate \u03bc from the "
            "nuclear shell model instead of using an experimental value. "
            "Configure the unpaired nucleon type, orbital, and "
            "coupling.<br><br>"

            "<h4>Run Options &amp; Log</h4>"
            "\u2022 Under the isotope list: set the output directory and "
            "colour palette, and toggle <b>Generate plots</b>.<br>"
            "\u2022 <b>Run</b> starts the calculation; the plots, the log "
            "and the voltage range estimator appear once it finishes. From "
            "then on every change (including undo and the estimator) re-runs "
            "it and updates them live; pressing <b>Run</b> again re-runs at "
            "once.<br>"
            "\u2022 Live runs are temporary. <b>Save</b> copies the latest "
            "run into the output directory as a <i>cls_&lt;timestamp&gt;</i> "
            "folder, with every input it used; <b>Load...</b> opens a saved "
            "run for viewing and restores those inputs, so <b>Run</b> "
            "repeats it.<br><br>"

            "<h4>Plots</h4>"
            "\u2022 <b>Individual Spectra</b> \u2013 Simulated HFS spectrum "
            "for each isotope (and isomers).<br>"
            "\u2022 <b>Combined Overview</b> \u2013 All isotopes overlaid on "
            "a single spectrum plot.<br>"
            "\u2022 <b>Peak List</b> \u2013 Sortable table of all peaks with "
            "isotope, state, shift, offset, voltage, and intensity.<br>"
            "\u2022 Use <b>Edit Plot</b> to customise titles, labels, "
            "colours, and line styles, then export as PNG, PDF, or SVG."
            "<br><br>"

            "<h4>Voltage Range Estimator</h4>"
            "A live map of where every hyperfine component resonates "
            "(\u0394V from the beam voltage, one line per component; "
            "isomers dashed) as the laser setpoint moves, the spectrum at "
            "the current setpoint, and a readout of which components fall "
            "inside the \u00b1dV sweep. It uses the same physics as "
            "<b>Run</b>.<br>"
            "\u2022 Drag the red bar, or click the map and use the arrow "
            "keys (0.001 cm⁻¹ per press, 0.01 with Shift); click a "
            "<i>centre</i> marker to "
            "snap to it. Red rings mark where the selected peaks (see "
            "<b>Peaks</b>) cross the bar inside \u00b1dV; they fade across "
            "the margin towards the sweep's edges, where scanning is "
            "riskier, and the readout calls those crossings <i>edge</i> and "
            "counts them by weight in its coverage line.<br>"
            "\u2022 The mouse wheel zooms the map around the cursor, "
            "dragging away from the bar pans it, and a double-click "
            "refits it.<br>"
            "\u2022 Releasing the bar, or pressing Enter in <b>Setpoint</b>, "
            "writes Laser \u25b8 Setpoint (anchor mode switches to "
            "Setpoint). Pressing Enter in <b>dV</b> writes Scan \u25b8 "
            "Voltage range = 2 \u00d7 dV. Both can be undone.<br>"
            "\u2022 <b>Peaks</b> picks the components the sweep must "
            "cover: by default each state's strongest <i>Peaks to "
            "measure</i> (the ones the run time counts), all components, "
            "or any you tick per state. Selected components are drawn bold "
            "and marked \u2022 in the readout, whose <i>Peaks</i> line "
            "gives the DAQ range [start, stop] that covers them at the "
            "current setpoint, with 100 V to spare on each side (the gear "
            "beside <b>Optimise</b> changes it), start and stop rounded "
            "outward to whole steps. <b>Optimise</b> sets Scan \u25b8 "
            "Voltage range to that range's width (one undo step); the "
            "setpoint is yours to place. With <b>Show</b> on one state it "
            "covers that state alone.<br>"
            "\u2022 The gear also holds a <b>Readout offset</b>: a signed "
            "shift (V) for a known DAQ readout shift. The lines, rings, "
            "spectrum and sweep band move by it, and the readout's dV and "
            "scan range include it, so they can be entered as DAQ scanning "
            "ranges directly. The axes stay put, and the hyperfine model, "
            "the Plots and the Log are unchanged.<br>"
            "\u2022 <b>Window</b> only sets the plotted voltage range "
            "(auto fits the components); <b>Show</b> picks one state or "
            "all.<br>"
            "\u2022 It appears after the first <b>Run</b> and follows every "
            "change from then on."
        )
        from gui.analysis.helpers import _show_scrollable_info
        _show_scrollable_info(self, "Estimate Tab \u2013 Information", text)
