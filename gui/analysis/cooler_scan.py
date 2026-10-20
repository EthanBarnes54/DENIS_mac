"""Walk a calibration project's fit across a grid of cooler offsets.

The Yb calibration needs every run fitted many times over, each
time pretending the cooler sat at a slightly different voltage. Rather
than a second fitting path, this drives the project's OWN fit -- the
one the Run Fit button uses -- with ``_config_overrides["source"]``
set to the offset under test. Whatever the project is configured to do
(binning, gates, model, fitter, splits, merges, scan filters, voltage
calibrations) is what the scan does, and it cannot drift from the real
analysis the way a second copy of the config gathering would.

One offset at a time, sequentially: each fit already runs its own runs
in parallel across cores, and the results have to be attributed to the
offset that produced them.

File output is suppressed for the duration -- an M-point scan over N
runs would otherwise write N x M reports, plot sets and CSVs that
nobody asked for. The fitted values live in memory; only the
calibration's own summary is ever written.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QObject, QTimer, Signal

from cls_estimations.cooler_calibration import ScanPoint

#: Output-block switches turned off while scanning.
_QUIET_OUTPUT = {
    "report": False, "params_csv": False, "metadata_csv": False,
    "fit_plots": False, "tof_plots": False,
}


def extract_parameter(params_df_dict, name, source_name=None):
    """``(value, sigma)`` of parameter *name*, or ``(None, None)``.

    Mirrors cls_estimations.isotope_shift.extract_centroid, which is
    the same lookup for one particular parameter; hyperfine constants
    need the general form.
    """
    params = params_df_dict.get("Parameter", []) or []
    values = params_df_dict.get("Value", []) or []
    errors = (params_df_dict.get("Stderr")
              or params_df_dict.get("Error") or [])
    sources = params_df_dict.get("Source", [None] * len(params))
    for i, p in enumerate(params):
        if str(p) != name:
            continue
        if source_name is not None and i < len(sources):
            if sources[i] is not None and str(sources[i]) != source_name:
                continue
        try:
            val = float(values[i])
        except (TypeError, ValueError, IndexError):
            return None, None
        try:
            err = float(errors[i])
        except (TypeError, ValueError, IndexError):
            err = float("nan")
        return val, err
    return None, None


def _row(params_df_dict, name, source_name=None):
    """Index of parameter *name* (of *source_name*) in a params table."""
    params = params_df_dict.get("Parameter", []) or []
    sources = params_df_dict.get("Source", [None] * len(params))
    for i, p in enumerate(params):
        if str(p) != name:
            continue
        if source_name is not None and i < len(sources):
            if sources[i] is not None and str(sources[i]) != source_name:
                continue
        return i
    return None


def _num(seq, i):
    try:
        v = float(seq[i])
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return v


def _fitted_offset(project):
    """The cooler offset the project's blocks were fitted at (where a
    continuation walk starts), 0 when it cannot be read."""
    getter = getattr(project, "_get_blocks_by_type", None)
    if getter is None:
        return 0.0
    try:
        from gui.analysis.blocks import SourceBlock
        blocks = getter(SourceBlock)
        if blocks:
            return float(blocks[0].get_source_config().get(
                "cooler_offset_v", 0.0) or 0.0)
    except Exception:                                    # noqa: BLE001
        pass
    return 0.0


def fit_point_problem(result, name, value, sigma):
    """Why one scan point cannot be trusted, or "".

    Each of these can produce numbers the calibration would otherwise
    believe: a parameter pinned at its bound with a
    tiny but non-zero error (so a near-infinite weight), a fit with no
    error bar, an optimiser that gave up but was reported a success.
    """
    if value is None or not math.isfinite(value):
        return "value is not finite"
    pdf = result.get("params_df", {}) or {}
    i = _row(pdf, name, result.get("source_name"))
    if i is not None:
        for col, side in (("Minimum", "lower"), ("Maximum", "upper")):
            b = _num(pdf.get(col, []) or [], i)
            if math.isfinite(b) and abs(value - b) <= 1e-6 * max(1.0, abs(b)):
                return f"at its {side} bound ({b:g})"
    if sigma is None or not math.isfinite(sigma) or sigma <= 0:
        # Also what a failed Hessian leaves: every Stderr None.
        return "no error bar"
    msg = str(result.get("fit_message", "") or "")
    if result.get("converged") is False or "abort" in msg.lower():
        return f"the optimiser did not converge ({msg or 'no message'})"
    return ""


class CoolerScan(QObject):
    """Runs (project, offset) jobs one after another.

    ``start`` returns immediately; the work happens on the projects'
    own fit workers. Listen to :attr:`progress`, :attr:`finished` and
    :attr:`failed`.
    """

    progress = Signal(int, int, str)     # done, total, what is running
    finished = Signal(list)              # [ScanPoint]
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: list[tuple] = []
        self._points: list[ScanPoint] = []
        self._done = 0
        self._total = 0
        self._current = None
        self._parameter = "Al"
        self._stop = False
        self._pending = False
        self._connected: set = set()
        # Seeding history: value_key -> {dV: (value, sigma)}, from
        # accepted fits only (see _seed_map).
        self._history: dict = {}

    # ── control ──
    def start(self, projects, offsets, *, parameter="Al"):
        """*projects* is ``[(project, isotope)]``, *offsets* a list of
        cooler offsets in volts.

        Each project walks its offsets outward from the one its blocks
        were fitted at, every fit seeded from its neighbours' answers --
        the systematic scan's continuation. Unseeded, a fit 30-50 V out
        starts hundreds of MHz from the peaks and lands on a bound or in
        a wrong minimum, which can throw the calibration off by
        kilovolts; seeded, it lands where it should.
        """
        from cls_estimations.systematics import order_outward
        if self.busy:
            raise RuntimeError("a scan is already running")
        offsets = [float(v) for v in offsets]
        if len(offsets) < 2:
            raise ValueError("a scan needs at least two offsets")
        if not projects:
            raise ValueError("no calibration projects selected")
        self._jobs = [(p, iso, dv) for p, iso in projects
                      for dv in order_outward(offsets,
                                              _fitted_offset(p))]
        self._history = {}
        self._points, self._done, self._stop = [], 0, False
        self._total = len(self._jobs)
        self._parameter = parameter
        for project, _iso in projects:
            if project not in self._connected:
                project.results_ready.connect(self._on_results)
                _sig = getattr(project, "fit_cancelled", None)
                if _sig is not None:
                    _sig.connect(self._on_cancelled)
                self._connected.add(project)
        # Never finish -- or fail -- inside start(): a caller that
        # connects to the signals on the line after would miss them,
        # and a project that cannot fit does fail immediately.
        self._pending = True
        QTimer.singleShot(0, self._begin)

    def _begin(self):
        self._pending = False
        self._next()

    def stop(self):
        """Finish the fit in flight, then stop."""
        self._stop = True

    @property
    def busy(self):
        return self._current is not None or self._pending

    # ── internals ──
    def _next(self):
        if self._stop or not self._jobs:
            self._finish()
            return
        project, isotope, dv = self._jobs.pop(0)
        self._current = (project, isotope, dv)
        self.progress.emit(
            self._done, self._total,
            f"{project.project_name} @ {dv:+.2f} V")
        project._config_overrides = {
            "source": {"cooler_offset_v": dv},
            "output": dict(_QUIET_OUTPUT),
        }
        seeds = self._seed_map(project, dv)
        if seeds:
            project._config_overrides["model_configs_map"] = seeds
        # No modal box in the middle of a many-fit scan (project.py).
        project._scan_driven = True
        try:
            project._on_fit_requested()
        except Exception as exc:                     # noqa: BLE001
            self._abort(f"{project.project_name}: {exc}")
            return
        # _on_fit_requested returns early -- without starting a worker
        # and without emitting anything -- when the project is not
        # fit-ready (no Source block, bad mass, a validation error it
        # already reported). Without this the scan would wait forever
        # for a signal that is never coming.
        worker = getattr(project, "_fit_worker", None)
        if worker is None or not worker.isRunning():
            self._abort(
                f"{project.project_name} did not start a fit at "
                f"{dv:+.2f} V. Check the project runs on its own "
                f"first.")

    def _on_results(self, project_name, results, _output_config):
        if not self.busy:
            return
        project, isotope, dv = self._current
        if project_name != project.project_name:
            return
        self._collect(project, isotope, dv, results)
        self._done += 1
        self._current = None
        project._config_overrides = {}
        self._next()

    def _on_cancelled(self, project_name):
        """The fit in flight was stopped (the Fitter's Stop): it sends no
        results, so end the scan here instead of waiting for them."""
        if self._current is None or \
                project_name != self._current[0].project_name:
            return
        self._stop = True
        self._on_results(project_name, [], {})

    def _collect(self, project, isotope, dv, results):
        for r in results or []:
            if not r.get("success"):
                continue
            val, sig = extract_parameter(
                r.get("params_df", {}) or {}, self._parameter,
                source_name=r.get("source_name"))
            if val is None:
                continue
            why = fit_point_problem(r, self._parameter, val, sig)
            self._points.append(ScanPoint(
                run=str(r.get("run_number", "") or "?"),
                isotope=isotope, dv=float(dv), value=float(val),
                sigma=float(sig) if sig == sig else float("nan"),
                flag=why))
            if not why:
                self._remember(project, r, dv)

    def _remember(self, project, result, dv):
        """An accepted fit's numbers seed its neighbours."""
        from gui.analysis.systematic_scan import value_key
        pdf = result.get("params_df", {}) or {}
        names = pdf.get("Parameter") or []
        values = pdf.get("Value") or []
        errors = pdf.get("Stderr") or [float("nan")] * len(names)
        models = pdf.get("Model") or [""] * len(names)
        sources = pdf.get("Source") or [None] * len(names)
        for i, pname in enumerate(names):
            v = _num(values, i)
            if not math.isfinite(v):
                continue
            src = (str(sources[i]) if i < len(sources) and sources[i]
                   else str(result.get("source_name", "")))
            key = value_key(project.project_name, src,
                            str(models[i]) if i < len(models) else "",
                            str(pname))
            self._history.setdefault(key, {})[float(dv)] = (
                v, _num(errors, i))

    def _seed_map(self, project, dv):
        """``{file_path: model_configs}`` seeded from the neighbouring
        offsets' accepted fits; {} before any (the blocks as they are)."""
        from cls_estimations.systematics import seed_from_history
        from gui.analysis.systematic_scan import (
            checked_entries_of, model_configs_of, seed_model_configs,
            split_key)
        if not self._history:
            return {}
        entries = checked_entries_of(project)
        base = model_configs_of(project)
        if not entries or not base:
            return {}
        out = {}
        for path, source in entries:
            seeds = {}
            for key, by_dv in self._history.items():
                proj, src, model, pname = split_key(key)
                if proj != project.project_name or src != source:
                    continue
                v = seed_from_history(
                    dv, [(d, val, s) for d, (val, s)
                         in sorted(by_dv.items())])
                if v is not None and math.isfinite(v):
                    seeds[(model, pname)] = float(v)
            if seeds:
                out[path] = seed_model_configs(base, seeds)
        return out

    def _abort(self, message):
        self._cleanup()
        self.failed.emit(message)

    def _finish(self):
        points = list(self._points)
        self._cleanup()
        self.progress.emit(self._done, self._total, "done")
        self.finished.emit(points)

    def _cleanup(self):
        self._pending = False
        if self._current is not None:
            self._current[0]._config_overrides = {}
        for project in self._connected:
            project._scan_driven = False
            try:
                project.results_ready.disconnect(self._on_results)
            except (RuntimeError, TypeError):
                pass
            try:
                project.fit_cancelled.disconnect(self._on_cancelled)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._connected.clear()
        self._current = None
        self._jobs = []
