"""Missing Files dialog: find the data files a session refers to.

Date:    2026-10-05
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Shown by ``MainWindow._resolve_missing_files`` when a loaded session
names data files that are not on disk. One table, one row per missing
file, and three ways to fill it:

* On opening, the session file's own folder is searched (recursively).
  A session moved together with its ``Data`` folder resolves here with
  no clicks at all.
* **Search Folder...** searches any folder, with or without its
  subfolders. Same-named runs in different places are told apart by
  how much of the old path's tail they share (``Data/13/run_7885``).
* **Locate...** asks for the selected files one by one. Each file
  located by hand is turned into a folder move (old folder -> new
  folder) and applied to every other missing file that exists at the
  moved location, so one pick usually resolves the whole session.

Picking a file with a different name asks first (a slip of the mouse
once swapped run_7885 for run_7860). Cancelling a file prompt returns
here with everything found so far kept; nothing is dropped from the
session until the user presses **Load**, and the files still missing
then are left out of this load only.

Depends on: gui.missing_files (index_folder, match_in_index,
derive_prefix, apply_prefix).
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDialog, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressDialog,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from gui.missing_files import (
    apply_prefix, derive_prefix, index_folder, match_in_index,
)
from gui.shared_widgets import maybe_convert_path

#: Directory cap for the automatic search of the session's own folder --
#: a session saved straight onto the Desktop must not stall the load.
AUTO_SEARCH_MAX_DIRS = 3000

_COL_FILE, _COL_STATUS, _COL_FOUND = range(3)


class MissingFilesDialog(QDialog):
    """Resolve missing data files; ``remap()`` holds the answer."""

    def __init__(self, missing, yaml_path, parent=None, auto_search=True):
        super().__init__(parent)
        self.setWindowTitle("Missing Files")
        self._missing = list(missing)
        self._yaml_path = yaml_path
        self._remap = {}          # original path -> located path
        self._ambiguous = {}      # original path -> tied candidates
        self._last_dir = os.path.dirname(os.path.abspath(yaml_path)) \
            if yaml_path else ""

        lay = QVBoxLayout(self)
        self._intro = QLabel(
            f"{len(self._missing)} data file(s) named in "
            f"<b>{os.path.basename(yaml_path or '')}</b> are not on disk. "
            "Search a folder for them, or locate them by hand: each file "
            "you locate also finds every other file that moved with it.")
        self._intro.setWordWrap(True)
        lay.addWidget(self._intro)

        self._table = QTableWidget(len(self._missing), 3)
        self._table.setHorizontalHeaderLabels(["File", "Status", "Found at"])
        self._table.verticalHeader().setVisible(False)
        # One line per file: long paths elide in the middle (the full
        # path is in the tooltip) instead of doubling the row height.
        self._table.setWordWrap(False)
        self._table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(_COL_FILE,
                                 QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_STATUS,
                                 QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_FOUND, QHeaderView.ResizeMode.Stretch)
        for row, orig in enumerate(self._missing):
            item = QTableWidgetItem(os.path.basename(
                orig.replace("\\", "/")))
            item.setToolTip(orig)
            self._table.setItem(row, _COL_FILE, item)
            self._table.setItem(row, _COL_STATUS, QTableWidgetItem())
            self._table.setItem(row, _COL_FOUND, QTableWidgetItem())
        self._table.itemSelectionChanged.connect(self._update_buttons)
        lay.addWidget(self._table, 1)

        row = QHBoxLayout()
        self._search_btn = QPushButton("Search Folder...")
        self._search_btn.setToolTip(
            "Look for the missing files in a folder you choose.")
        self._search_btn.clicked.connect(self._search_folder_clicked)
        self._recursive = QCheckBox("Include subfolders")
        self._recursive.setChecked(True)
        self._recursive.setToolTip(
            "Search every folder below the chosen one, not just the "
            "folder itself.")
        self._locate_btn = QPushButton("Locate...")
        self._locate_btn.setToolTip(
            "Pick the selected files by hand, one by one (all files "
            "still missing when nothing is selected). Every file you pick "
            "also resolves the others that moved with it.")
        self._locate_btn.clicked.connect(self._locate_clicked)
        self._forget_btn = QPushButton("Forget")
        self._forget_btn.setToolTip(
            "Mark the selected files as missing again.")
        self._forget_btn.clicked.connect(self._forget_clicked)
        row.addWidget(self._search_btn)
        row.addWidget(self._recursive)
        row.addStretch(1)
        row.addWidget(self._locate_btn)
        row.addWidget(self._forget_btn)
        lay.addLayout(row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self._load_btn = QPushButton("Load")
        self._load_btn.setDefault(True)
        self._load_btn.clicked.connect(self.accept)
        self._cancel_btn = QPushButton("Cancel Load")
        self._cancel_btn.clicked.connect(self.reject)
        bottom.addWidget(self._load_btn)
        bottom.addWidget(self._cancel_btn)
        lay.addLayout(bottom)

        self.resize(1000, 460)
        self._refresh()
        if auto_search and self._last_dir:
            self.search(self._last_dir, recursive=True,
                        max_dirs=AUTO_SEARCH_MAX_DIRS, automatic=True)

    # ── Result ──────────────────────────────────────────────────────

    def remap(self):
        """``{original path: located path}`` for every file found."""
        return dict(self._remap)

    def unresolved(self):
        return [p for p in self._missing if p not in self._remap]

    # ── Searching ───────────────────────────────────────────────────

    def search(self, folder, recursive=True, max_dirs=20000,
               automatic=False):
        """Search ``folder`` for the files still missing. Returns the
        number newly found."""
        todo = self.unresolved()
        if not todo:
            return 0
        progress = None
        if not automatic:
            progress = QProgressDialog(
                f"Searching {folder} ...", "Stop", 0, 0, self)
            progress.setWindowTitle("Search Folder")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(400)

        def _tick(n_dirs):
            if progress is None:
                return False
            progress.setLabelText(
                f"Searching {folder} ...\n{n_dirs} folders looked at")
            QApplication.processEvents()
            return progress.wasCanceled()

        index, complete = index_folder(folder, todo, recursive=recursive,
                                       max_dirs=max_dirs, progress=_tick)
        if progress is not None:
            progress.close()
        found, ambiguous = match_in_index(todo, index)
        self._remap.update(found)
        for orig in found:
            self._ambiguous.pop(orig, None)
        self._ambiguous.update(ambiguous)
        if found:
            self._last_dir = os.path.dirname(next(iter(found.values())))
        where = os.path.basename(os.path.normpath(folder)) or folder
        if automatic:
            msg = (f"Found {len(found)} file(s) in the session's own folder "
                   f"({where})." if found else "")
        else:
            msg = f"Found {len(found)} of {len(todo)} file(s) in {where}."
        if ambiguous:
            msg += (f" {len(ambiguous)} name(s) match several files equally "
                    "well -- pick those with Locate.")
        if not complete:
            msg += " The search stopped before the end of the folder tree."
        self._set_status(msg)
        self._refresh()
        return len(found)

    def _search_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Search for the missing files in", self._last_dir)
        if folder:
            self.search(folder, recursive=self._recursive.isChecked())

    # ── Locating by hand ────────────────────────────────────────────

    def _selected_originals(self):
        rows = sorted({i.row() for i in self._table.selectedItems()})
        return [self._missing[r] for r in rows]

    def _locate_clicked(self):
        chosen = self._selected_originals() or self.unresolved()
        self.locate(chosen)

    def locate(self, originals):
        """Ask for each of ``originals`` in turn (skipping any resolved
        meanwhile). Cancelling a prompt stops asking; nothing found so
        far is lost."""
        # Files already found are asked again only when asked for
        # explicitly; ones a folder move resolves mid-loop are skipped.
        redo = {o for o in originals if o in self._remap}
        for orig in originals:
            if orig in self._remap and orig not in redo:
                continue
            name = os.path.basename(orig.replace("\\", "/"))
            ext = os.path.splitext(name)[1].lower()
            flt = (f"Same type (*{ext});;ASDF/VASDF (*.asdf *.vasdf);;"
                   "All files (*)") if ext in (".asdf", ".vasdf") \
                else "All files (*)"
            start = self._start_dir_for(orig)
            new_path, _ = QFileDialog.getOpenFileName(
                self, f"Locate: {name}", start, flt)
            if not new_path:
                self._set_status("Stopped locating; files found so far "
                                 "are kept.")
                break
            if os.path.basename(new_path).lower() != name.lower():
                ans = QMessageBox.question(
                    self, "Different file name",
                    f"You picked <b>{os.path.basename(new_path)}</b> for "
                    f"<b>{name}</b>.<br><br>Use it anyway? Only do this "
                    "if the run was renamed.",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if ans != QMessageBox.StandardButton.Yes:
                    continue
            self._remap[orig] = os.path.normpath(new_path)
            self._ambiguous.pop(orig, None)
            self._last_dir = os.path.dirname(new_path)
            extra = self._apply_move(orig, new_path)
            self._set_status(
                f"Located {name}."
                + (f" The same folder move found {extra} more file(s)."
                   if extra else ""))
            self._refresh()

    def _apply_move(self, orig, new_path):
        """Resolve every other missing file that moved like ``orig``."""
        move = derive_prefix(orig, new_path)
        if move is None:
            return 0
        found = apply_prefix(self.unresolved(), *move)
        self._remap.update(found)
        for o in found:
            self._ambiguous.pop(o, None)
        return len(found)

    def _start_dir_for(self, orig):
        if orig in self._ambiguous:
            return os.path.dirname(self._ambiguous[orig][0])
        if self._last_dir and os.path.isdir(self._last_dir):
            return self._last_dir
        d = os.path.dirname(maybe_convert_path(orig))
        return d if os.path.isdir(d) else ""

    def _forget_clicked(self):
        for orig in self._selected_originals():
            self._remap.pop(orig, None)
        self._refresh()

    # ── Display ─────────────────────────────────────────────────────

    def _set_status(self, text):
        self._status.setText(text)

    def _refresh(self):
        for row, orig in enumerate(self._missing):
            st = self._table.item(row, _COL_STATUS)
            fd = self._table.item(row, _COL_FOUND)
            if orig in self._remap:
                st.setText("Found")
                st.setForeground(QColor("#43a047"))
                fd.setText(self._remap[orig])
                fd.setToolTip(self._remap[orig])
            elif orig in self._ambiguous:
                n = len(self._ambiguous[orig])
                st.setText(f"{n} candidates")
                st.setForeground(QColor("#fb8c00"))
                fd.setText("; ".join(self._ambiguous[orig][:3]))
                fd.setToolTip("\n".join(self._ambiguous[orig]))
            else:
                st.setText("Missing")
                st.setForeground(QColor("#e53935"))
                fd.setText("")
                fd.setToolTip(orig)
        n_left = len(self.unresolved())
        n_found = len(self._missing) - n_left
        if n_left == 0:
            self._load_btn.setText("Load")
            self._load_btn.setToolTip("Every file was found.")
        else:
            self._load_btn.setText(
                f"Load without {n_left} missing file(s)")
            self._load_btn.setToolTip(
                "Load with the files found. The ones still missing are "
                "left out of this load only; the session file keeps them "
                "-- unless you save the session again afterwards.")
        self._intro.setToolTip(f"{n_found} of {len(self._missing)} found")
        self._update_buttons()

    def _update_buttons(self):
        any_left = bool(self.unresolved())
        sel = self._selected_originals()
        self._search_btn.setEnabled(any_left)
        self._locate_btn.setEnabled(any_left or bool(sel))
        self._forget_btn.setEnabled(any(o in self._remap for o in sel))
