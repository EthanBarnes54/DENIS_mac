"""Test package for the DENIS collinear-laser-spectroscopy app.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Marks the ``tests`` directory as a Python package so the unittest and
pytest suites under it can be discovered and imported with dotted module
paths (e.g. ``tests.test_binning_yerr``).

Depends on: standard library and third-party packages only.
"""

# Tests never touch the real preferences: every runner that imports this
# package (pytest, python -m unittest) works on a private copy of
# settings/settings.yaml. Several tests wrote into the real file before
# 2026-10-08 (last_data_save_dir pointing into %TEMP%, a rewrite on every
# MainWindow construction). A copy, not an empty file, so tests keep
# seeing the developer's theme and zoom as before.
import os as _os
import shutil as _shutil
import tempfile as _tempfile

if not _os.environ.get("DENIS_SETTINGS_PATH"):
    _real = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), "settings", "settings.yaml")
    _private = _os.path.join(_tempfile.mkdtemp(prefix="denis_test_settings_"),
                             "settings.yaml")
    if _os.path.isfile(_real):
        _shutil.copy2(_real, _private)
    _os.environ["DENIS_SETTINGS_PATH"] = _private
