"""A stand-in for fitting._fit_single_run that only sleeps.

Kept in its own module, with no Qt or DENIS imports, so the process
pool (Windows spawn) can pickle it by reference and start its children
quickly. Like the real fit it reports in on the progress queue as it
starts (0.0) -- the parent times a run from that moment -- and again
half-way, so a test can wait until a fit is really running.
"""
import os
import time


def slow_fit(run_file_path, *args, progress_queue=None, **kwargs):
    """Sleep for the seconds encoded in the name: run_<n>_<secs>s.asdf."""
    secs = float(os.path.basename(run_file_path).split("_")[2][:-6])
    if progress_queue is not None:
        progress_queue.put_nowait((run_file_path, 0.0))
    time.sleep(min(secs, 0.2))
    if progress_queue is not None:
        progress_queue.put_nowait((run_file_path, 0.5))
    time.sleep(max(0.0, secs - 0.2))
    return {"success": True, "run_file": run_file_path,
            "run_number": os.path.basename(run_file_path).split("_")[1],
            "pid": os.getpid()}


def slow_autofit(sources_data, model_configs, perturbed_values, vary_keys,
                 *args, **kwargs):
    """A stand-in for auto_fitter._run_one_fit: the centre start
    (perturbed_values[0] == 10.0, always start 0) sleeps the seconds in
    sources_data[0]["stuck"], every other start those in ["quick"]."""
    centre = float(perturbed_values[0]) == 10.0
    time.sleep(float(sources_data[0]["stuck" if centre else "quick"]))
    return {"success": True, "chi2": 1.0, "redchi": 1.0, "objective": 1.0,
            "fitted_vals": list(perturbed_values), "x_smooth": [],
            "y_smooth": [], "components": {}, "per_source": []}
