"""pytest session guard: the real settings file must come out of a test
run byte for byte as it went in (tests/__init__.py redirects every test
to a private copy; this catches anything that bypasses it).

Also deletes every top-level widget a test module leaves behind. Without
it a chunk of the suite ended with ~106,000 live widgets (60 Pre-Analysis
tabs, 25 Results tabs, plot editors with running timers ...), whose event
traffic starved later tests: a scan test's zero-delay timer fired 14.6 s
late under CPU load and the scan "timed out" (review 2, 2026-10-08)."""
import os

import pytest

import tests  # noqa: F401  (sets DENIS_SETTINGS_PATH before any test)

_REAL = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "settings", "settings.yaml")


def _read(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


_START = _read(_REAL)


@pytest.fixture(autouse=True, scope="module")
def _delete_leftover_widgets():
    """After each test MODULE (tests in a module or class may share
    widgets), delete what it left on screen. deleteLater only -- close()
    would run closeEvent, and a window asking to save would block."""
    yield
    if os.environ.get("DENIS_TEST_KEEP_WIDGETS"):
        return                       # debugging: leave them alive
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    from PySide6.QtCore import QThread
    for w in QApplication.topLevelWidgets():
        try:
            # Deleting a widget whose QThread child still runs kills the
            # process (an ASDF viewer still loading its file did, in a
            # -k selection of the suite). Stop them first; a thread that
            # will not stop keeps its widget alive instead.
            alive = False
            for t in w.findChildren(QThread):
                if t.isRunning():
                    for stop in ("stop", "cancel", "requestInterruption"):
                        if hasattr(t, stop):
                            getattr(t, stop)()
                    t.quit()
                    alive = alive or not t.wait(5000)
            if not alive:
                w.deleteLater()
        except RuntimeError:          # already gone on the C++ side
            pass
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    # Run the queued calls now, not in the next module's first test. A
    # slot of a half-destroyed widget can raise "already deleted" here
    # (CoolerCalibrationTab.refresh_projects on projects_changed while
    # its AnalysisTab is torn down) -- only ever at teardown.
    try:
        app.processEvents()
    except RuntimeError:
        pass


def pytest_sessionfinish(session, exitstatus):
    if _read(_REAL) != _START:
        session.exitstatus = 1
        print("\nERROR: the test run changed the REAL settings file "
              f"{_REAL}; a test bypassed DENIS_SETTINGS_PATH.")
