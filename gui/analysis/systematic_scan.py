"""Repeat the analysis across a range of assumed cooler offsets.

The Yb calibration ends with an interval, not a number: anywhere
between the two isotopes' zero crossings is defensible. The systematic
uncertainty on an extracted parameter is how far that parameter moves
when the whole analysis is repeated across that interval. Doing it by
hand means editing the Source block, fitting, finding the iteration,
copying a number out, ten times over, for every project.

This drives it. Like ``cooler_scan.py`` it never fits anything itself:
it sets ``project._config_overrides`` and calls the very method the
Run Fit button calls, so the scan cannot drift from the real analysis.

**Where each correction lives**, which is why a step is a chain and
not a single fit:

* the scan-voltage calibration and the cooler-voltage offset both act
  in the LAB frame, on the voltages, before anything is converted --
  inside ``pipeline.prepare_run_data``;
* the GP reference correction acts in the REST frame, as an MHz shift
  applied after the Doppler conversion.

Moving the assumed cooler offset therefore moves the rest-frame axis
(of order 10 MHz per volt), which moves the reference centroids the GP
is trained on. A GP trained at the calibrated offset says nothing
about the frame at another one, so each step refits the reference,
retrains the GP, recomputes the per-file corrections, and only then
fits the samples. The isotope shifts follow from those.

One step at a time, and one job at a time within a step: each fit
already parallelises its own runs across cores, and every number has
to be attributable to the offset that produced it.

Seeding is the other half of making this work -- see
``cls_estimations.systematics.seed_from_history``. A centroid moves
hundreds of MHz across a 20 V scan, far outside a line width, so a
step that starts from the baseline fit at the far end of the range
simply does not converge. Steps are walked outward from the baseline
and each one starts from its neighbour's answer.
"""
from __future__ import annotations

import copy
import csv
import math
import os
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QTimer, Signal

from cls_estimations.systematics import (
    order_outward, seed_from_history, shift_bounds,
)

#: How a step's starting values are chosen.
SEED_CONTINUATION = "Continuation (neighbouring offset)"
SEED_BASELINE = "Baseline iteration"
SEED_BLOCKS = "Project blocks as they are"
SEED_MODES = (SEED_CONTINUATION, SEED_BASELINE, SEED_BLOCKS)

#: Output switches turned off for scan steps unless the user asks for
#: everything. A 10-point scan over 5 projects writes 50 iterations;
#: the reports and the CSVs are what gets read when a step misbehaves,
#: the plot sets are what fills the disk.
LIGHT_OUTPUT = {
    "fit_plots": False, "tof_plots": False, "walk_plot": False,
    "correl_plot": False, "chisq_map": False, "conf_bands": False,
}

#: Column holding the per-run reduced chi-square in ``metadata.csv``.
REDCHI_COLUMN = "Reduced Chi-sq"


def value_key(project, source, model, parameter):
    """Flat key for one number. Flat because these get saved: tuple
    keys do not survive a round trip through the session file."""
    return f"{project}|{source}|{model}|{parameter}"


def split_key(key):
    parts = str(key).split("|")
    parts += [""] * (4 - len(parts))
    return tuple(parts[:4])


# ── reading a baseline iteration ────────────────────────────────────

@dataclass
class Baseline:
    """One iteration of one project, as the scan needs it.

    ``params`` is keyed by satlas2 source name rather than run number:
    that is what both ``parameters.csv`` and a live fit result carry,
    and it is the only identity that survives merged spectra and
    virtual splits.
    """
    iter_dir: str = ""
    config: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)    # source -> {(model, param): row}
    redchi: dict = field(default_factory=dict)    # run -> float
    run_of_source: dict = field(default_factory=dict)

    @property
    def sources(self):
        return sorted(self.params)

    @property
    def runs(self):
        return sorted(self.run_of_source.values())

    def value(self, source, model, parameter):
        row = self.params.get(source, {}).get((model, parameter))
        return None if row is None else row.get("value")

    def sigma(self, source, model, parameter):
        row = self.params.get(source, {}).get((model, parameter))
        return float("nan") if row is None else row.get("sigma", float("nan"))

    def parameters(self):
        """Every (model, parameter) pair the iteration fitted."""
        out = set()
        for pmap in self.params.values():
            out |= set(pmap)
        return sorted(out)


def _to_float(text, default=float("nan")):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return default
    return v


def read_baseline(iter_dir):
    """Load one iteration's config, fitted parameters and chi-squares.

    Missing pieces are not fatal: an iteration written with the CSV
    outputs switched off still has its config, and a scan can still be
    set up from it -- it just has nothing to seed from.
    """
    b = Baseline(iter_dir=iter_dir or "")
    if not iter_dir or not os.path.isdir(iter_dir):
        return b

    cfg_path = os.path.join(iter_dir, "config_snapshot.yaml")
    if os.path.isfile(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as fh:
                b.config = yaml.safe_load(fh) or {}
        except Exception:                                # noqa: BLE001
            b.config = {}

    par_path = os.path.join(iter_dir, "parameters.csv")
    if os.path.isfile(par_path):
        try:
            with open(par_path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    src = (row.get("Source") or "").strip()
                    model = (row.get("Model") or "").strip()
                    par = (row.get("Parameter") or "").strip()
                    if not par:
                        continue
                    run = str(row.get("run_number", "") or "").strip()
                    if src and run:
                        b.run_of_source.setdefault(src, run)
                    b.params.setdefault(src, {})[(model, par)] = {
                        "value": _to_float(row.get("Value")),
                        "sigma": _to_float(row.get("Error")
                                           or row.get("Stderr")),
                        "vary": str(row.get("Vary", "True")).strip()
                                not in ("False", "0", "false"),
                        "min": _to_float(row.get("Min") or row.get("Minimum"),
                                         float("-inf")),
                        "max": _to_float(row.get("Max") or row.get("Maximum"),
                                         float("inf")),
                    }
        except Exception:                                # noqa: BLE001
            pass

    meta_path = os.path.join(iter_dir, "metadata.csv")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    run = str(row.get("run_number", "") or "").strip()
                    if run:
                        b.redchi[run] = _to_float(row.get(REDCHI_COLUMN))
        except Exception:                                # noqa: BLE001
            pass
    return b


def iterations_of(project_dir):
    """Iteration directory names under one project, newest last."""
    if not os.path.isdir(project_dir):
        return []
    return sorted(d for d in os.listdir(project_dir)
                  if os.path.isdir(os.path.join(project_dir, d)))


#: Keys whose difference says nothing about how a fit was done.
_COSMETIC = {"order", "name", "collapsed", "geometry", "block_name"}


def config_differences(current, snapshot, path="", out=None, depth=0):
    """Human-readable differences between two project configs.

    The scan fits with the project as it stands now, not as the
    baseline iteration had it -- rebuilding blocks under the user
    would be worse than telling them. So the tab shows this list and
    lets them decide.
    """
    if out is None:
        out = []
    if depth > 6 or len(out) > 40:
        return out
    if isinstance(current, dict) and isinstance(snapshot, dict):
        for key in sorted(set(current) | set(snapshot)):
            if key in _COSMETIC:
                continue
            here = f"{path}.{key}" if path else str(key)
            if key not in current:
                out.append(f"{here}: missing now (was {snapshot[key]!r})")
            elif key not in snapshot:
                out.append(f"{here}: new ({current[key]!r})")
            else:
                config_differences(current[key], snapshot[key], here,
                                   out, depth + 1)
        return out
    if isinstance(current, list) and isinstance(snapshot, list):
        if len(current) != len(snapshot):
            out.append(f"{path}: {len(snapshot)} -> {len(current)} entries")
            return out
        for i, (a, b) in enumerate(zip(current, snapshot)):
            config_differences(a, b, f"{path}[{i}]", out, depth + 1)
        return out
    if current != snapshot:
        out.append(f"{path}: {snapshot!r} -> {current!r}")
    return out


# ── seeding ─────────────────────────────────────────────────────────

def seed_model_configs(model_configs, seeds):
    """A copy of *model_configs* started from *seeds*.

    ``seeds`` is ``{(model_name, parameter): value}``, keyed by the
    satlas2 model name -- the one that appears in a fit result, which
    is the sanitized block name, not the block name itself. A seed may
    also be a dict carrying ``value`` and, when the user has set them
    by hand for one step, ``min`` and ``max``.

    Bounds travel with a value that would otherwise land outside them
    (see systematics.shift_bounds). Expressions and vary flags are left
    exactly as the project has them: a seed says where to start, not
    what is free.
    """
    from gui.analysis.naming import safe_model_name

    out = copy.deepcopy(list(model_configs or []))
    if not seeds:
        return out
    for cfg in out:
        mname = safe_model_name(cfg.get("name", ""))
        for pname, p in (cfg.get("params") or {}).items():
            v, lo_ov, hi_ov = _seed_parts(seeds.get((mname, pname)))
            if v is None:
                continue
            lo, hi = shift_bounds(p.get("value", v), p.get("min"),
                                  p.get("max"), v)
            p["value"] = float(v)
            p["min"] = lo_ov if lo_ov is not None else lo
            p["max"] = hi_ov if hi_ov is not None else hi
        for label, amp in (cfg.get("peak_amplitudes") or {}).items():
            v, _lo, _hi = _seed_parts(seeds.get((mname, f"Amp{label}")))
            if v is None:
                continue
            amp["value"] = float(v)
    return out


def _seed_parts(seed):
    """``(value, min, max)`` from a seed that may be a bare number."""
    if seed is None:
        return None, None, None
    if isinstance(seed, dict):
        value = seed.get("value")
        lo, hi = seed.get("min"), seed.get("max")
    else:
        value, lo, hi = seed, None, None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None, None, None
    if not math.isfinite(value):
        return None, None, None
    return value, lo, hi


def model_configs_of(project):
    """The model configs the project would fit with right now."""
    getter = getattr(project, "_get_blocks_by_type", None)
    if getter is None:
        return []
    try:
        from gui.analysis.blocks import ModelBlock
        return [m.get_model_config() for m in getter(ModelBlock)]
    except Exception:                                    # noqa: BLE001
        return []


def checked_entries_of(project):
    """``[(file_path, source_name)]`` for the runs the project would
    fit, named exactly as the fitter will name them."""
    getter = getattr(project, "_get_blocks_by_type", None)
    if getter is None:
        return []
    try:
        from gui.analysis.blocks import SourceBlock
        from gui.analysis.naming import (
            source_name_for_descriptor, source_name_for_merged,
            source_name_for_path)
        blocks = getter(SourceBlock)
        if not blocks:
            return []
        out = []
        for e in blocks[0].get_checked_files():
            path = e["path"]
            if e.get("is_merged") and "merged_data" in e:
                name = source_name_for_merged(e["merged_data"])
            elif e.get("is_split"):
                name = source_name_for_descriptor({
                    "kind": "virtual_split",
                    "source_id": e["source_id"]})
            else:
                name = source_name_for_path(path)
            out.append((path, name))
        return out
    except Exception:                                    # noqa: BLE001
        return []


# ── the scan ────────────────────────────────────────────────────────

@dataclass
class ScanTarget:
    """One project taking part, and the iteration it starts from."""
    project: object
    role: str = "sample"          # "reference" | "sample"
    baseline: Baseline = field(default_factory=Baseline)

    @property
    def name(self):
        return getattr(self.project, "project_name", "?")

    @property
    def is_reference(self):
        return self.role == "reference"


def new_record(dv):
    return {"dv": float(dv), "status": "pending", "message": "",
            "values": {}, "redchi": {}, "shifts": [], "labels": {},
            "iterations": {}, "failed_projects": []}


class SystematicScan(QObject):
    """Walks the offsets, one chained step at a time.

    ``start`` returns immediately; the work happens on the projects'
    own fit workers and the GP's own thread. Listen to
    :attr:`progress`, :attr:`step_done`, :attr:`finished`,
    :attr:`failed` and :attr:`log`.

    A step that fails does not stop the scan -- the point of the scan
    is to find out which offsets are hard, and the user re-runs those
    with better starting values afterwards. Only something that makes
    every further step meaningless (a project that will not start a
    fit at all) aborts.
    """

    progress = Signal(int, int, str)     # done, total, what is running
    step_done = Signal(float, dict)      # dv, record
    step_failed = Signal(float, str)
    finished = Signal(list)              # [record], ordered by offset
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._targets = []
        self._steps = []
        self._jobs = []
        self._records = {}
        self._history = {}
        self._current_dv = None
        self._current_target = None
        self._label = "sys_001"
        self._seed_mode = SEED_CONTINUATION
        self._gp = None
        self._is_tab = None
        self._light = True
        self._stop = False
        self._pending = False
        self._total = 0
        self._done = 0
        self._connected = set()
        self._gp_connected = None
        self._seed_overrides = {}
        self._fitter_overrides = {}

    # -- control ----------------------------------------------------
    def start(self, targets, offsets, *, label="sys_001",
              seed_mode=SEED_CONTINUATION, baseline_dv=0.0, gp=None,
              is_tab=None, light_output=True, records=None,
              seed_overrides=None, fitter_overrides=None):
        """Fit every target at every offset.

        ``records`` carries results already collected (a resumed or
        partially re-run scan); those offsets keep their numbers and
        seed the ones being fitted now.
        """
        if self.busy:
            raise RuntimeError("a scan is already running")
        targets = list(targets or [])
        offsets = [float(v) for v in (offsets or [])]
        if not targets:
            raise ValueError("no projects selected")
        if not offsets:
            raise ValueError("no offsets to scan")

        if gp is not None:
            # The GP is retrained at every offset from the reference
            # projects ticked on its panel. One the scan does not refit
            # keeps its NOMINAL results, so the "per-offset" GP would be
            # the same at every offset and the band meaningless (a
            # sample-only scan reported 127 MHz for 0.5) -- review 2.
            names = getattr(gp, "reference_project_names", None)
            trained_on = set(names()) if callable(names) else set()
            refit = {t.name for t in targets if t.is_reference}
            stale = sorted(trained_on - refit)
            if stale:
                raise ValueError(
                    "The GP is trained on " + ", ".join(stale)
                    + ", which this scan does not refit. Add "
                    + ("it" if len(stale) == 1 else "them")
                    + " to the scan as reference"
                    + ("" if len(stale) == 1 else "s")
                    + ", or turn the GP leg off.")

        self._targets = targets
        self._label = str(label or "sys_001")
        self._snapshot = self._take_snapshot(targets, gp, is_tab)
        self._seed_mode = seed_mode
        self._gp = gp
        self._is_tab = is_tab
        self._light = bool(light_output)
        # Set by hand for one step, after watching it go wrong:
        # {offset: {key: {value/min/max}}} and {offset: fit method}.
        self._seed_overrides = dict(seed_overrides or {})
        self._fitter_overrides = dict(fitter_overrides or {})
        self._stop = False
        self._records = dict(records or {})
        self._history = {}
        for rec in self._records.values():
            self._remember(rec)
        self._steps = order_outward(offsets, baseline_dv)
        self._total = len(self._steps)
        self._done = 0
        self._jobs = []

        for t in targets:
            p = t.project
            if p not in self._connected:
                p.results_ready.connect(self._on_results)
                _sig = getattr(p, "fit_cancelled", None)
                if _sig is not None:
                    _sig.connect(self._on_cancelled)
                self._connected.add(p)
        if gp is not None and self._gp_connected is not gp:
            gp.gp_fit_done.connect(self._on_gp_done)
            gp.gp_fit_failed.connect(self._on_gp_failed)
            self._gp_connected = gp

        # Never emit inside start(): a caller connecting on the next
        # line would miss it, and an unfittable project fails at once.
        self._pending = True
        QTimer.singleShot(0, self._begin)

    def _begin(self):
        self._pending = False
        self._next_step()

    def stop(self):
        """Finish the job in flight, then stop."""
        self._stop = True

    @property
    def busy(self):
        return (self._pending or self._current_dv is not None
                or bool(self._jobs))

    @property
    def records(self):
        return dict(self._records)

    # -- the walk ---------------------------------------------------
    def _next_step(self):
        if self._stop or not self._steps:
            self._finish()
            return
        dv = self._steps.pop(0)
        self._current_dv = dv
        self._forget(dv)
        self._records[dv] = new_record(dv)
        refs = [t for t in self._targets if t.is_reference]
        self._jobs = (
            [("fit", t) for t in refs]
            + ([("gp", None), ("corr", None)] if self._gp is not None else [])
            # With the GP leg on, the first reference fit used the
            # PREVIOUS offset's corrections (its band came out at -10 /
            # 0 / +20 MHz where 0 was right). The GP is trained on it
            # un-corrected, so that is fine for the GP; the values kept
            # come from this refit, corrected by this offset's GP.
            + ([("fit", t) for t in refs] if self._gp is not None else [])
            + [("fit", t) for t in self._targets if not t.is_reference]
            + ([("shifts", None)] if self._is_tab is not None else [])
            + [("endstep", None)]
        )
        self._pump()

    def _pump(self):
        if not self._jobs:
            self._end_step()
            return
        kind, payload = self._jobs.pop(0)
        dv = self._current_dv
        if kind == "fit":
            self._start_fit(payload, dv)
        elif kind == "gp":
            self._start_gp(dv)
        elif kind == "corr":
            self._recompute_corrections(dv)
        elif kind == "shifts":
            self._collect_shifts(dv)
        else:
            self._end_step()

    def _later(self):
        QTimer.singleShot(0, self._pump)

    def _end_step(self):
        dv = self._current_dv
        if dv is None:
            return
        rec = self._records.get(dv) or new_record(dv)
        if rec["status"] == "pending":
            if not rec["values"]:
                rec["status"] = "failed"
                if not rec["message"]:
                    rec["message"] = "no parameters came back"
            elif rec.get("failed_projects") or rec.get("failed_runs"):
                # Some projects fitted and some did not. What did
                # arrive is real and stays in the band; the step says
                # which projects are missing from it.
                rec["status"] = "partial"
            else:
                rec["status"] = "ok"
        self._remember(rec)
        self._done += 1
        self._current_dv = None
        self._current_target = None
        self._clear_overrides()
        if rec["status"] == "failed":
            self.step_failed.emit(dv, rec["message"])
        else:
            self.step_done.emit(dv, rec)
        self.progress.emit(self._done, self._total,
                           f"{dv:+.2f} V {rec['status']}")
        QTimer.singleShot(0, self._next_step)

    def _fail_step(self, message):
        dv = self._current_dv
        if dv is None:
            return
        rec = self._records.setdefault(dv, new_record(dv))
        rec["status"] = "failed"
        rec["message"] = str(message)
        self.log.emit(f"{dv:+.2f} V: {message}")
        self._jobs = []
        self._end_step()

    def _abort(self, message):
        self._cleanup()
        self.failed.emit(str(message))

    def _finish(self):
        out = [self._records[dv] for dv in sorted(self._records)]
        self._cleanup()
        self.progress.emit(self._done, self._total, "done")
        self.finished.emit(out)

    @staticmethod
    def _take_snapshot(targets, gp, is_tab):
        """Everything a scan changes that is not its own output."""
        snap = {"projects": {}, "gp": None, "is_tab": None}
        for t in targets:
            p = t.project
            snap["projects"][id(p)] = (p, {
                k: getattr(p, k) for k in (
                    "_last_results", "_last_iter_dir",
                    "_last_output_config") if hasattr(p, k)})
        if gp is not None and hasattr(gp, "snapshot_state"):
            snap["gp"] = gp.snapshot_state()
        if is_tab is not None:
            snap["is_tab"] = {"had": bool(getattr(
                is_tab, "_last_shift_data", None))}
        return snap

    def _restore_snapshot(self):
        """Put the analysis back as it was before the scan. Left at the
        farthest offset, the next normal fit of a sample was corrected
        by that offset's GP (90 MHz for 100; at Ge scale +198 MHz), and
        a save stored that GP as the session's (review 2, 2026-10-08)."""
        snap = getattr(self, "_snapshot", None)
        self._snapshot = None
        if not snap:
            return
        for p, attrs in snap["projects"].values():
            for k, v in attrs.items():
                try:
                    setattr(p, k, v)
                except (AttributeError, RuntimeError):
                    pass
        if snap["gp"] is not None and self._gp is not None:
            try:
                self._gp.restore_state(snap["gp"])
            except Exception as exc:                      # noqa: BLE001
                self.log.emit(f"could not restore the GP: {exc}")
        if snap["is_tab"] is not None and self._is_tab is not None:
            # Shifts before the scan: show the nominal ones again, not
            # the last offset's. None before it: show none -- an empty
            # tab used to keep the last offset's shifts, ready for Send
            # to Results and Export (review 3, 2026-10-08).
            try:
                if snap["is_tab"].get("had"):
                    self._is_tab.shifts_for_scan()
                elif hasattr(self._is_tab, "clear_shifts"):
                    self._is_tab.clear_shifts()
                else:
                    self._is_tab._last_shift_data = []
            except Exception as exc:                      # noqa: BLE001
                self.log.emit(f"could not refresh the shifts: {exc}")

    def _cleanup(self):
        self._pending = False
        self._clear_overrides()
        self._restore_snapshot()
        for p in self._connected:
            try:
                p.results_ready.disconnect(self._on_results)
            except (RuntimeError, TypeError):
                pass
            try:
                p.fit_cancelled.disconnect(self._on_cancelled)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._connected.clear()
        if self._gp_connected is not None:
            for sig, slot in ((self._gp_connected.gp_fit_done,
                               self._on_gp_done),
                              (self._gp_connected.gp_fit_failed,
                               self._on_gp_failed)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            self._gp_connected = None
        self._current_dv = None
        self._current_target = None
        self._jobs = []
        self._steps = []

    def _clear_overrides(self):
        """A leftover override would silently move every later fit of
        that project -- clear them even on an abort."""
        for t in self._targets:
            try:
                t.project._config_overrides = {}
                t.project._scan_driven = False
            except (AttributeError, RuntimeError):
                pass

    # -- jobs -------------------------------------------------------
    def _start_fit(self, target, dv):
        project = target.project
        self._current_target = target
        self.progress.emit(self._done, self._total,
                           f"{target.name} @ {dv:+.2f} V")
        output = {"iter_mode": "Manual", "iter_label": self._label}
        if self._light:
            output.update(LIGHT_OUTPUT)
        overrides = {
            "source": {"cooler_offset_v": float(dv)},
            "output": output,
        }
        method = self._fitter_overrides.get(float(dv))
        if method:
            overrides["fitter"] = {"method": method}
        seeds = self._seed_map(target, dv)
        if seeds:
            overrides["model_configs_map"] = seeds
        project._config_overrides = overrides
        # Nothing in the fit path may open a modal box from here on:
        # it would stop the scan dead, waiting for a click.
        project._scan_driven = True
        try:
            project._on_fit_requested()
        except Exception as exc:                          # noqa: BLE001
            self._abort(f"{target.name}: {exc}")
            return
        # _on_fit_requested returns early -- without starting a worker
        # and without emitting anything -- when the project is not
        # fit-ready. Without this the scan waits for a signal that is
        # never coming.
        worker = getattr(project, "_fit_worker", None)
        if worker is None or not worker.isRunning():
            self._abort(
                f"{target.name} did not start a fit at {dv:+.2f} V. "
                f"Check the project runs on its own first.")

    def _on_cancelled(self, project_name):
        """The fit in flight was stopped (the Fitter's Stop): it sends no
        results, so stop the scan here instead of waiting for them."""
        if (self._current_target is None
                or project_name != self._current_target.name):
            return
        self._stop = True
        # The rest of this offset's chain (GP, other projects) would
        # build on a fit that never finished: drop it, and say why.
        self._jobs = []
        rec = self._records.get(self._current_dv)
        if rec is not None and not rec.get("message"):
            rec["message"] = "stopped: a fit was cancelled"
        self._on_results(project_name, [], {})

    def _on_results(self, project_name, results, _output_config):
        if self._current_dv is None or self._current_target is None:
            return
        target = self._current_target
        if project_name != target.name:
            return
        dv = self._current_dv
        self._collect(target, dv, results)
        self._current_target = None
        target.project._config_overrides = {}
        rec = self._records.get(dv)
        if (target.is_reference and rec is not None
                and rec.get("status") == "failed"):
            # The GP is trained on this project. Carrying on would fit
            # every sample against a stale frame and quietly report a
            # number that means nothing.
            self._jobs = []
        self._later()

    def _start_gp(self, dv):
        self.progress.emit(self._done, self._total,
                           f"GP @ {dv:+.2f} V")
        try:
            started = self._gp.start_gp_fit_for_scan()
        except Exception as exc:                          # noqa: BLE001
            self._fail_step(f"GP refit: {exc}")
            return
        if not started:
            self._fail_step(
                "the GP did not refit; the reference centroids for "
                "this offset are not usable")

    def _on_gp_done(self):
        if self._current_dv is None:
            return
        self._later()

    def _on_gp_failed(self, message):
        if self._current_dv is None:
            return
        self._fail_step(f"GP refit failed: {message}")

    def _recompute_corrections(self, dv):
        try:
            self._gp.recompute_corrections()
        except Exception as exc:                          # noqa: BLE001
            self._fail_step(f"GP corrections: {exc}")
            return
        self._later()

    #: Project name the isotope shifts are banded under. They are
    #: computed from every project at once, so they belong to none of
    #: them.
    SHIFT_PROJECT = "Isotope shifts"

    def _collect_shifts(self, dv):
        rec = self._records.setdefault(dv, new_record(dv))
        missing = list(rec.get("failed_projects") or []) + list(
            rec.get("failed_runs") or [])
        if missing:
            # A shift averaged over whichever runs survived is over a
            # different population than the other offsets' and reads
            # as an offset dependence (a false 25 MHz half-width for two
            # flat runs, one failing once); with a failed project it
            # paired another offset's results (review 3, 2026-10-08).
            self.log.emit(
                f"{dv:+.2f} V: isotope shifts not taken -- "
                f"{', '.join(map(str, missing))} did not fit at this "
                f"offset")
            rec["shifts"] = []
            self._later()
            return
        try:
            rows = self._is_tab.shifts_for_scan() or []
        except Exception as exc:                          # noqa: BLE001
            self.log.emit(f"{dv:+.2f} V: isotope shifts: {exc}")
            rows = []
        rec["shifts"] = list(rows)
        # Band them like anything else. The shift is the number that
        # gets published, and its dependence on the assumed offset is
        # the whole reason for the scan -- the reference and the
        # sample move together, so it is far smaller than either
        # centroid's, and that cancellation is exactly what has to be
        # measured rather than assumed.
        for row in rows:
            label = str(row.get("label", "?"))
            for source_key, pname, sigma_key in (
                    ("delta_nu", "delta_nu", "sigma_delta_nu_stat"),
                    ("centroid", "centroid_corrected", "sigma_fit")):
                if source_key not in row:
                    continue
                try:
                    value = float(row[source_key])
                except (TypeError, ValueError):
                    continue
                try:
                    sigma = float(row.get(sigma_key, float("nan")))
                except (TypeError, ValueError):
                    sigma = float("nan")
                rec["values"][value_key(self.SHIFT_PROJECT, label,
                                        "IS", pname)] = {
                    "value": value, "sigma": sigma}
            rec["labels"][f"{self.SHIFT_PROJECT}|{label}"] = label
        self._later()

    # -- collecting -------------------------------------------------
    def _collect(self, target, dv, results):
        rec = self._records.setdefault(dv, new_record(dv))
        n_ok = 0
        for r in results or []:
            source = str(r.get("source_name") or "")
            run = str(r.get("run_number", "") or "?")
            if not r.get("success"):
                rec["labels"][f"{target.name}|{source or run}"] = run
                continue
            n_ok += 1
            if source:
                rec["labels"][f"{target.name}|{source}"] = run
            quality = r.get("fit_quality") or {}
            redchi = quality.get("redchi")
            if redchi is not None:
                rec["redchi"][f"{target.name}|{source or run}"] = \
                    float(redchi)
            self._collect_params(target, rec, source or run,
                                 r.get("params_df") or {})
        failed = [str(r.get("run_number", "") or "?")
                  for r in (results or []) if not r.get("success")]
        if failed and n_ok:
            # Some runs of this project fitted, some did not. The ones
            # that did are real, but an average over a different set of
            # runs than the other offsets is not comparable with them
            # (two flat runs at 0 and 100 MHz, one failing at one offset,
            # made a 25 MHz "band") -- review 2.
            runs = rec.setdefault("failed_runs", [])
            runs.extend(f"{target.name}/{r}" for r in failed
                        if f"{target.name}/{r}" not in runs)
            rec["message"] = ("failed run(s): " + ", ".join(runs))
        if not n_ok:
            rec.setdefault("failed_projects", []).append(target.name)
            rec["message"] = f"{target.name}: no run fitted"
            if target.is_reference:
                # The GP is trained on this project: without it every
                # sample at this offset would be corrected into a
                # frame that never arrived.
                rec["status"] = "failed"
        it = getattr(target.project, "_last_iter_dir", "")
        if it:
            rec["iterations"][target.name] = os.path.basename(
                os.path.normpath(it))

    @staticmethod
    def _collect_params(target, rec, fallback_source, params):
        names = params.get("Parameter") or []
        values = params.get("Value") or []
        errors = (params.get("Stderr") or params.get("Error")
                  or [float("nan")] * len(names))
        models = params.get("Model") or [""] * len(names)
        sources = params.get("Source") or [fallback_source] * len(names)
        mins = params.get("Minimum") or []
        maxs = params.get("Maximum") or []
        for i, pname in enumerate(names):
            try:
                value = float(values[i])
            except (TypeError, ValueError, IndexError):
                continue
            try:
                sigma = float(errors[i])
            except (TypeError, ValueError, IndexError):
                sigma = float("nan")
            at_bound = False
            for seq in (mins, maxs):
                try:
                    b = float(seq[i])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(b) and abs(value - b) <= 1e-6 * max(
                        1.0, abs(b)):
                    at_bound = True
            src = str(sources[i]) if i < len(sources) else ""
            if not src:
                src = str(fallback_source)
            key = value_key(target.name, src,
                            str(models[i]) if i < len(models) else "",
                            str(pname))
            rec["values"][key] = {"value": value, "sigma": sigma}
            if at_bound:
                rec["values"][key]["at_bound"] = True

    def _remember(self, rec):
        """Feed a step's numbers into the seeding history.

        Keyed by offset, not appended: re-running a step replaces what
        it contributed. Otherwise a step that went somewhere silly
        would seed its own re-run with the answer being thrown away.
        """
        dv = float(rec.get("dv", 0.0))
        if rec.get("status") == "failed":
            return
        for key, v in (rec.get("values") or {}).items():
            # A value pinned at its bound is where the fit gave up, not
            # where the next step should start (review 3, 2026-10-08).
            if v.get("at_bound"):
                continue
            try:
                value = float(v["value"])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                sigma = float(v.get("sigma", float("nan")))
            except (TypeError, ValueError):
                sigma = float("nan")
            self._history.setdefault(key, {})[dv] = (value, sigma)

    def _forget(self, dv):
        """Drop everything a step contributed, before re-running it."""
        for by_dv in self._history.values():
            by_dv.pop(float(dv), None)

    def _history_for(self, key):
        return [(dv, v, s) for dv, (v, s)
                in sorted((self._history.get(key) or {}).items())]

    # -- seeding ----------------------------------------------------
    def _seed_map(self, target, dv):
        """``{file_path: model_configs}`` for one project at one offset."""
        if self._seed_mode == SEED_BLOCKS:
            return {}
        entries = checked_entries_of(target.project)
        base_models = model_configs_of(target.project)
        if not entries or not base_models:
            return {}
        out = {}
        for path, source in entries:
            seeds = self._seeds_for(target, source, dv)
            if seeds:
                out[path] = seed_model_configs(base_models, seeds)
        return out

    def _seeds_for(self, target, source, dv):
        """``{(model, parameter): value}`` to start one run from."""
        seeds = {}
        known = target.baseline.params.get(source, {})
        for (model, pname), row in known.items():
            value = None
            if self._seed_mode == SEED_CONTINUATION:
                value = seed_from_history(
                    dv, self._history_for(
                        value_key(target.name, source, model, pname)))
            if value is None:
                value = row.get("value")
            if value is not None and math.isfinite(value):
                seeds[(model, pname)] = float(value)
        # What the user set by hand for this step wins over both the
        # neighbour and the baseline -- it is why they set it.
        for key, ov in (self._seed_overrides.get(float(dv))
                        or {}).items():
            proj, src, model, pname = split_key(key)
            if proj != target.name or src not in ("", source):
                continue
            base = dict(ov)
            if "value" not in base:
                have = seeds.get((model, pname))
                if have is None:
                    continue
                base["value"] = have
            seeds[(model, pname)] = base
        return seeds
