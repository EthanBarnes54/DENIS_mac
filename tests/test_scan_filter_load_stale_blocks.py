"""A Source block that is gone must not break the next scan-filter load.

Every Load (Load All, Load Tab, a reopened session) ends in
ScanFilterRegistry.from_dict, which emits filters_changed. A Source block
used to subscribe with ``filters_changed.connect(self.block_changed.emit)``;
that callable is not a slot of the block, so Qt kept the connection after
the block was destroyed and every later emission raised "Signal source has
been deleted" once per dead block (eight per load after a session of eight
projects, in the 2026-11-12 session log).

Run from the project root:

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_scan_filter_load_stale_blocks.py

Depends on: gui.analysis.blocks (SourceBlock), gui.scan_filter.
"""

import sys
import unittest

from tests.test_import_only_checked import _ensure_app


class Scan_Filter_Load_After_Blocks_Die_Tests(unittest.TestCase):

    def setUp(self):
        _ensure_app()

    def test_A_Dead_Source_Block_Leaves_No_Error_On_The_Next_Load(self):
        import shiboken6
        from gui.analysis.blocks import SourceBlock
        from gui.scan_filter import ScanFilterRegistry, get_registry, set_registry
        saved = get_registry()
        set_registry(ScanFilterRegistry())
        raised = []
        hook, sys.excepthook = sys.excepthook, (
            lambda t, v, tb: raised.append(str(v)))
        try:
            registry = get_registry()
            for _ in range(3):
                block = SourceBlock("S")
                shiboken6.delete(block)
            registry.from_dict(registry.to_dict())
        finally:
            sys.excepthook = hook
            set_registry(saved)
        self.assertEqual(raised, [])

    def test_B_A_Live_Source_Block_Still_Hears_The_Filters(self):
        from gui.analysis.blocks import SourceBlock
        from gui.scan_filter import ScanFilterRegistry, get_registry, set_registry
        saved = get_registry()
        set_registry(ScanFilterRegistry())
        try:
            block = SourceBlock("S")
            heard = []
            block.block_changed.connect(lambda: heard.append(1))
            registry = get_registry()
            registry.from_dict(registry.to_dict())
        finally:
            set_registry(saved)
        self.assertEqual(heard, [1])


if __name__ == "__main__":
    unittest.main()
