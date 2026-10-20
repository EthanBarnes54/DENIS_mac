"""Two bugs reported for checking (2026-09-27), both confirmed and fixed.

1. Pre-Analysis frequency merges were shifted by the transition TWICE.
   The merge backend stores x relative to the transition
   (Compute_WL(ref=ref_freq), ref_freq = the transition); the display
   still assumed the absolute axis of PA's pre-2026-05-14 merge and
   subtracted the transition again, so a merged spectrum was drawn
   about one transition frequency (~1e9 MHz) below its runs. Pre-May
   merges, stored absolute, still display.

2. clstools' PMT gate only removes the channels 1-4 left unticked, so
   events on any other channel -- DC -- always passed. DENIS's own
   in-memory paths filtered them, so the two disagreed whenever a
   run held DC events.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_merge_axis_and_dc_gate.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

C_CM = 2.99792458e10          # cm/s
E_LO, E_UP, HARM = 0.0, 30000.0, 2
TRANSITION_MHZ = (E_UP - E_LO) * C_CM / 1e6


class MergedFrequencyAxisTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from gui.preanalysis_tab import PreAnalysisTab
        cls.tab = PreAnalysisTab()
        cls.tab._e_lower.setValue(E_LO)
        cls.tab._e_upper.setValue(E_UP)
        cls.tab._harmonic.setValue(HARM)
        cls.tab._z_spin.setValue(50)
        cls.tab._a_spin.setValue(116)
        cls.detuning = np.linspace(-800.0, 600.0, 57)
        cls.y = np.exp(-0.5 * (cls.detuning / 120.0) ** 2) * 50 + 3

    @classmethod
    def tearDownClass(cls):
        cls.tab.deleteLater()

    def _drawn(self, stored_x, axis):
        tab = self.tab
        for fe in list(tab._file_entries):
            tab._file_entries.remove(fe)
            fe.deleteLater()
        tab._add_merged_entry("merged_test", stored_x, self.y, "frequency",
                              {})
        tab._xaxis_combo.setCurrentText(axis)
        tab._replot()
        line = next(ln for ln in tab._ax.get_lines()
                    if ln.get_label() == "merged_test")
        return np.asarray(line.get_xdata(), float)

    def test_the_frequency_axis_shows_the_stored_detuning(self):
        np.testing.assert_allclose(
            self._drawn(self.detuning, "Frequency"), self.detuning)

    def test_the_wavenumber_axis_is_the_same_detuning_in_cm(self):
        np.testing.assert_allclose(
            self._drawn(self.detuning, "Wavenumber"),
            self.detuning * 1e6 / C_CM)

    def test_a_pre_may_absolute_merge_lands_in_the_same_place(self):
        legacy = self.detuning + TRANSITION_MHZ
        np.testing.assert_allclose(
            self._drawn(legacy, "Frequency"), self.detuning, atol=1e-3)

    def test_the_voltage_axis_agrees_for_both_kinds(self):
        a = self._drawn(self.detuning, "Voltage")
        b = self._drawn(self.detuning + TRANSITION_MHZ, "Voltage")
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertTrue(np.all(np.isfinite(a)))


def _events():
    """Channels 3 and 4 plus DC (5), five events each."""
    rows = []
    for ch in (3, 4, 5):
        for k in range(5):
            rows.append([1.7e9 + k, 100.0 + k, k, ch, 40.0 + k, 2.99])
    return pd.DataFrame(rows, columns=["TS", "DV", "Bunch", "TDC", "TOF",
                                       "Vrfq"])


class StrictPmtGateTests(unittest.TestCase):
    """Against clstools' real gate code."""

    def _data(self):
        import dask.dataframe as dd
        from clstools.DataFrame import CLSDataFrame
        data = CLSDataFrame()
        data.Run = dd.from_pandas(_events(), npartitions=1)
        return data

    def _tof_counts(self, data):
        return int(data.ToF_binned["counts"].sum())

    def test_clstools_alone_lets_dc_through(self):
        """The bug itself: unticked DC still counted."""
        data = self._data()
        data.Compute_ToF(V_gate=None, PMT_gate=[3, 4])
        self.assertEqual(self._tof_counts(data), 15)

    def test_the_strict_gate_keeps_only_ticked_channels(self):
        from gui.analysis.binning import strict_pmt_gate
        data = self._data()
        with strict_pmt_gate(data, [3, 4]):
            data.Compute_ToF(V_gate=None, PMT_gate=[3, 4])
        self.assertEqual(self._tof_counts(data), 10)

    def test_ticking_dc_counts_it(self):
        from gui.analysis.binning import strict_pmt_gate
        data = self._data()
        with strict_pmt_gate(data, [3, 4, 5]):
            data.Compute_ToF(V_gate=None, PMT_gate=[3, 4, 5])
        self.assertEqual(self._tof_counts(data), 15)

    def test_the_run_is_restored(self):
        from gui.analysis.binning import strict_pmt_gate
        data = self._data()
        run = data.Run
        with strict_pmt_gate(data, [3]):
            self.assertIsNot(data.Run, run)
        self.assertIs(data.Run, run)

    def test_no_gate_changes_nothing(self):
        from gui.analysis.binning import strict_pmt_gate
        data = self._data()
        run = data.Run
        with strict_pmt_gate(data, None):
            self.assertIs(data.Run, run)

    def test_every_clstools_gate_call_is_wrapped(self):
        import inspect
        from gui.analysis import binning, merge
        src = inspect.getsource(binning) + inspect.getsource(merge)
        calls = src.count("PMT_gate=pmt_gate")
        self.assertEqual(calls, 4)          # 2x Compute_Bins, Raw, ToF
        self.assertEqual(src.count("with strict_pmt_gate(data, pmt_gate)"),
                         3)


if __name__ == "__main__":
    unittest.main()
