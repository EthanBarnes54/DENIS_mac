"""Missing-file detection and remapping for session YAML loads.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

When a session YAML refers to ASDF / VASDF data files that no longer
exist on disk (renamed folder, moved Nextcloud share, switched OS),
the loader prompts the user to either locate each missing file or
skip them. This module provides the pure helpers that walk the YAML
structure, scan for missing paths, remap them, and prune skipped
entries; the dialog lives in ``gui.missing_files_dialog`` and the
load-time flow in ``main_window``.

Finding files (2026-10-05): ``index_folder`` + ``match_in_index`` look
the missing names up in a folder tree (recursively or not), breaking
ties between same-named runs by how much of the old path's tail they
share; ``derive_prefix`` + ``apply_prefix`` turn one hand-located file
into a folder move that resolves every other file that moved with it.

Path locations covered:
    preanalysis.files[].path
    analysis.projects[].blocks[type=Source].files[].path

Synthetic paths (``merged://...``) are ignored -- they're not on disk.
``maybe_convert_path`` is applied before the existence check so a
saved POSIX path round-trips on Windows and vice versa.

Depends on: gui.shared_widgets (maybe_convert_path).
"""

import os

from gui.shared_widgets import maybe_convert_path


def _iter_pa_files_lists(pa):
    """Yield each Pre-Analysis ``files`` list in ``pa``.

    Pre-Analysis is multi-project: the live save format is
    ``preanalysis.projects[].files[]``. Legacy single-project saves
    keep ``preanalysis.files[]`` directly under ``preanalysis``;
    PreAnalysisContainer._restore_from_dict still loads those, so we
    have to scan them too.
    """
    if isinstance(pa.get("projects"), list):
        for proj in pa["projects"]:
            if isinstance(proj, dict) and isinstance(
                    proj.get("files"), list):
                yield proj["files"]
    if isinstance(pa.get("files"), list):
        # Legacy flat layout (PA-1-only YAMLs predating the container).
        yield pa["files"]


def _iter_path_holders(yaml_dict):
    """Yield ``(holder_dict, key)`` pairs for every file-path slot in
    the YAML. The caller can read or rewrite ``holder[key]`` in place.

    Synthetic ``merged://`` paths and non-Source analysis blocks are
    skipped here so callers don't need to filter again.
    """
    pa = yaml_dict.get("preanalysis") or {}
    for files_list in _iter_pa_files_lists(pa):
        for fcfg in files_list:
            if isinstance(fcfg, dict) and isinstance(
                    fcfg.get("path"), str):
                if not fcfg["path"].startswith("merged://"):
                    yield fcfg, "path"

    an = yaml_dict.get("analysis") or {}
    for proj in an.get("projects") or []:
        for block in proj.get("blocks") or []:
            if block.get("type") != "Source":
                continue
            for fcfg in block.get("files") or []:
                if isinstance(fcfg, dict) and isinstance(
                        fcfg.get("path"), str):
                    if not fcfg["path"].startswith("merged://"):
                        yield fcfg, "path"


def scan_missing_paths(yaml_dict):
    """Return the unique missing file paths referenced in ``yaml_dict``.

    Order of first appearance is preserved so the locate-files dialog
    walks them top-down (Pre-Analysis first, then Analysis projects in
    order). Paths are returned as the original strings stored in the
    YAML -- ``maybe_convert_path`` is only applied for the existence
    check, not the return value, so the caller's remap dict stays
    keyed on what's actually written to disk.
    """
    seen = set()
    out = []
    for holder, key in _iter_path_holders(yaml_dict):
        original = holder[key]
        if original in seen:
            continue
        if not os.path.isfile(maybe_convert_path(original)):
            seen.add(original)
            out.append(original)
    return out


def remap_paths_inplace(yaml_dict, remap):
    """Rewrite path slots in-place using ``remap`` (old -> new).

    Returns the number of slots replaced. Slots whose current value
    isn't a key in ``remap`` are left alone, which is what we want
    when ``remap`` only covers a subset (the user located 3 of 5).

    Also rewrites the path-*keyed* top-level registries (``scan_filters``,
    ``calibrations``), which ``_iter_path_holders`` cannot reach because
    their paths are dict keys rather than values. Without this, relocating a
    moved project silently drops every scan filter and voltage calibration
    attached to it -- the run would load fine and quietly compute its
    voltages from the file default, which is precisely the silent-wrong-number
    failure the calibration feature exists to eliminate.
    """
    if not remap:
        return 0
    n = 0
    for holder, key in _iter_path_holders(yaml_dict):
        if holder[key] in remap:
            holder[key] = remap[holder[key]]
            n += 1
    n += _remap_registry_keys_inplace(yaml_dict, remap)
    n += _remap_other_strings_inplace(yaml_dict, remap)
    return n


def _remap_other_strings_inplace(yaml_dict, remap):
    """Rewrite every other string value or dict key that names a remapped
    file. Returns slots replaced.

    The file-entry slots and the registries above are not the only places
    a session stores a run's path: merged entries keep ``source_files`` and
    ``source_info`` paths, and the GP reference correction keys its saved
    per-file corrections by path. Those are not needed to *load* a run,
    so ``scan_missing_paths`` does not report them -- but left on the old
    path after a move, the saved corrections silently stop matching the
    relocated runs. A key or value equal to a remapped path (in any
    spelling) is replaced; a canonical (lower-case) spelling stays
    canonical.
    """
    from gui.calibration import canonical_path

    if not remap:
        return 0
    canon = {canonical_path(old): new for old, new in remap.items()}
    count = [0]

    def _new(text):
        if not isinstance(text, str) or len(text) > 1024 or not (
                "/" in text or "\\" in text):
            return None
        if text in remap:
            return remap[text]
        c = canonical_path(text)
        if c not in canon:
            return None
        return canonical_path(canon[c]) if text == c else canon[c]

    def _walk(node):
        if isinstance(node, dict):
            for key in list(node.keys()):
                value = node[key]
                new_key = _new(key)
                if new_key is not None and new_key != key \
                        and new_key not in node:
                    del node[key]
                    node[new_key] = value
                    key = new_key
                    count[0] += 1
                new_val = _new(value)
                if new_val is not None and new_val != value:
                    node[key] = new_val
                    count[0] += 1
                else:
                    _walk(value)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                new_val = _new(value)
                if new_val is not None and new_val != value:
                    node[i] = new_val
                    count[0] += 1
                else:
                    _walk(value)

    _walk(yaml_dict)
    return count[0]


# ── Finding moved files ──────────────────────────────────────────────

#: Directories never worth descending into when searching for runs.
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
              "$recycle.bin", "system volume information"}


def _parts(path):
    """Path components, separator- and case-insensitive."""
    return [p for p in path.replace("\\", "/").lower().split("/") if p]


def suffix_score(original, candidate):
    """How many trailing path components ``candidate`` shares with
    ``original`` (the file name counts as one). Used to pick, among
    several files with the same name, the one sitting in the same
    relative place -- ``Data/13/run_7885.asdf`` over a stray copy."""
    a, b = _parts(original), _parts(candidate)
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def index_folder(root, names, recursive=True, max_dirs=20000,
                 progress=None):
    """Find files called any of ``names`` under ``root``.

    Returns ``(index, complete)``: ``index`` maps a lower-case file name
    to every full path found with that name, ``complete`` is False when
    the walk stopped early -- after ``max_dirs`` directories, or because
    ``progress(n_dirs)`` returned True (the user pressed Cancel).
    Hidden and tool directories (``.git``, ``.venv``...) are skipped.
    """
    wanted = {os.path.basename(n).lower() for n in names}
    index = {}
    if not root or not os.path.isdir(root):
        return index, True
    n_dirs = 0
    for dirpath, dirnames, filenames in os.walk(root):
        n_dirs += 1
        for f in filenames:
            if f.lower() in wanted:
                index.setdefault(f.lower(), []).append(
                    os.path.join(dirpath, f))
        if not recursive:
            break
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d.lower() not in _SKIP_DIRS)
        if n_dirs >= max_dirs:
            return index, False
        if progress is not None and n_dirs % 50 == 0 and progress(n_dirs):
            return index, False
    return index, True


def match_in_index(missing, index):
    """Look ``missing`` paths up in an ``index_folder`` result.

    Returns ``(found, ambiguous)``: ``found`` maps an original path to
    its unique match -- the only file of that name, or the one sharing
    the longest tail with the original; ``ambiguous`` maps an original
    path to its tied candidates, which the user has to choose between.
    """
    found, ambiguous = {}, {}
    for orig in missing:
        cands = index.get(os.path.basename(
            orig.replace("\\", "/")).lower(), [])
        if len(cands) == 1:
            found[orig] = cands[0]
        elif len(cands) > 1:
            scored = sorted(((suffix_score(orig, c), c) for c in cands),
                            reverse=True)
            if scored[0][0] > scored[1][0]:
                found[orig] = scored[0][1]
            else:
                ambiguous[orig] = [c for sc, c in scored
                                   if sc == scored[0][0]]
    return found, ambiguous


def derive_prefix(original, located):
    """The folder move implied by locating ``original`` at ``located``.

    Returns ``(old_prefix, new_prefix)`` -- everything before the path
    tail the two share -- or None when the file names differ (no move
    can be inferred from a different file) or nothing but the name is
    shared and the old folder is a filesystem root.
    """
    a = original.replace("\\", "/").split("/")
    b = located.replace("\\", "/").split("/")
    if a[-1].lower() != b[-1].lower():
        return None
    n = suffix_score(original, located)
    old, new = "/".join(a[:len(a) - n]), "/".join(b[:len(b) - n])
    if not old or not new:
        return None
    return old + "/", new + "/"


def apply_prefix(missing, old_prefix, new_prefix, exists=os.path.isfile):
    """Map every path in ``missing`` that lies under ``old_prefix`` to the
    same place under ``new_prefix`` -- only where that file exists."""
    out = {}
    old_n = old_prefix.replace("\\", "/").lower()
    for orig in missing:
        norm = orig.replace("\\", "/")
        if norm.lower().startswith(old_n):
            cand = os.path.normpath(new_prefix + norm[len(old_prefix):])
            if exists(cand):
                out[orig] = cand
    return out


def _remap_registry_keys_inplace(yaml_dict, remap):
    """Rewrite the path-keyed registry sections. Returns slots replaced.

    ``remap`` is keyed by the raw strings stored in the YAML's *file-entry*
    slots (e.g. ``C:/Users/x/Desktop/run_7735.asdf``), but the registries key
    themselves through ``canonical_path`` (``os.path.normcase`` + ``abspath``),
    which on Windows lowercases and flips the separators
    (``c:\\users\\x\\desktop\\run_7735.asdf``). A direct ``remap.get(path)``
    therefore never matches, and the section would be silently left pointing at
    the dead path -- the run then loads fine and quietly computes its voltages
    from the uncorrected calibration. So match canonically, on both sides.
    """
    from gui.calibration import canonical_path

    if not remap:
        return 0
    # Canonical view of the remap, so a registry key in any spelling resolves.
    canon = {canonical_path(old): new for old, new in remap.items()}

    def _lookup(path):
        if not isinstance(path, str):
            return None
        if path in remap:
            return remap[path]
        return canon.get(canonical_path(path))

    n = 0
    # ``calibration_acks`` is a plain list of paths, not a dict -- a relocated
    # run whose ack was left behind would start blinking its warning again on
    # every project load, which trains the user to ignore the warning.
    acks = yaml_dict.get("calibration_acks")
    if isinstance(acks, list):
        rebuilt_acks = []
        for p in acks:
            new_p = _lookup(p)
            if new_p is None:
                rebuilt_acks.append(p)
            else:
                rebuilt_acks.append(new_p)
                n += 1
        yaml_dict["calibration_acks"] = rebuilt_acks

    for section in ("scan_filters", "calibrations"):
        reg = yaml_dict.get(section)
        if not isinstance(reg, dict):
            continue
        rebuilt = {}
        for path, payload in reg.items():
            new_path = _lookup(path)
            if new_path is None:
                new_path = path
            else:
                n += 1
            # A borrow spec points at a *donor* run, which may itself have
            # moved. Miss this and the calibration resolves to a dead path and
            # fails the fit outright.
            if isinstance(payload, dict):
                new_donor = _lookup(payload.get("donor"))
                if new_donor is not None:
                    payload = dict(payload)
                    payload["donor"] = new_donor
                    n += 1
            rebuilt[new_path] = payload
        yaml_dict[section] = rebuilt
    return n


def prune_missing_paths_inplace(yaml_dict, missing):
    """Drop file entries whose ``path`` is in the ``missing`` set.

    Used after the locate flow to remove the entries the user opted
    to skip (either via the top-level Skip button or by cancelling
    mid-locate). Doesn't touch the on-disk YAML; only the in-memory
    dict the loader is about to consume.
    """
    if not missing:
        return

    def _filter(files_list):
        return [
            f for f in files_list
            if not (isinstance(f, dict) and f.get("path") in missing)
        ]

    pa = yaml_dict.get("preanalysis") or {}
    if isinstance(pa.get("projects"), list):
        for proj in pa["projects"]:
            if isinstance(proj, dict) and isinstance(
                    proj.get("files"), list):
                proj["files"] = _filter(proj["files"])
    if isinstance(pa.get("files"), list):
        pa["files"] = _filter(pa["files"])

    an = yaml_dict.get("analysis") or {}
    for proj in an.get("projects") or []:
        for block in proj.get("blocks") or []:
            if block.get("type") != "Source":
                continue
            if isinstance(block.get("files"), list):
                block["files"] = _filter(block["files"])
