#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click patcher for train.py

Patches:
  A) Add per-iteration guard in _bg_mask_prune to avoid double execution.
  B) Add bg prune stats helpers:
     - _bg_prune_reset_stats_if_needed: prevent shape mismatch crashes
     - _bg_prune_slice_stats: keep evidence aligned when pruning points
  C) Replace local init blocks for bg_prune_seen/bg_prune_hit with helper call.
  D) Slice stats before gaussians.prune_points(prune_mask) inside _bg_mask_prune.

Creates a timestamped backup beside the file before modifying.
Idempotent: re-running should not duplicate inserts.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from datetime import datetime


HELPER_BLOCK = r'''
# -------------------- BG prune stats helpers (auto-patched) --------------------
def _bg_prune_reset_stats_if_needed(gaussians):
    """
    Keep gaussians.bg_prune_seen/bg_prune_hit aligned with current point count.
    If missing or shape-mismatched, re-initialize to zeros to avoid crashes.
    """
    try:
        n = int(gaussians.get_xyz.shape[0])
        dev = gaussians.get_xyz.device
        for name in ("bg_prune_seen", "bg_prune_hit"):
            t = getattr(gaussians, name, None)
            if (not torch.is_tensor(t)) or (t.numel() == 0) or (int(t.shape[0]) != n):
                setattr(gaussians, name, torch.zeros((n, 1), device=dev))
    except Exception:
        # never crash training due to bookkeeping
        pass


def _bg_prune_slice_stats(gaussians, keep_mask: torch.Tensor):
    """
    Slice bg_prune_seen/bg_prune_hit with keep_mask (bool [N]) before pruning points.
    This preserves accumulated evidence for remaining points.
    """
    try:
        if keep_mask is None:
            return
        keep = keep_mask.detach().bool()
        for name in ("bg_prune_seen", "bg_prune_hit"):
            t = getattr(gaussians, name, None)
            if torch.is_tensor(t) and (t.numel() > 0) and (int(t.shape[0]) == int(keep.shape[0])):
                setattr(gaussians, name, t[keep].contiguous())
    except Exception:
        pass
# -------------------- END BG prune stats helpers --------------------
'''


def backup_file(p: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = p.with_suffix(p.suffix + f".bak_{ts}")
    bak.write_bytes(p.read_bytes())
    return bak


def insert_helpers(text: str) -> tuple[str, bool]:
    if "_bg_prune_reset_stats_if_needed" in text and "_bg_prune_slice_stats" in text:
        return text, False

    anchor = "# -------------------- BG mask pruning (aligned with tools/check_masks.py) --------------------"
    idx = text.find(anchor)
    if idx == -1:
        # fallback: insert after imports (after first blank line following imports)
        m = re.search(r"\n\n", text)
        if not m:
            return text, False
        insert_at = m.end()
        return text[:insert_at] + HELPER_BLOCK + "\n" + text[insert_at:], True

    insert_at = idx + len(anchor)
    return text[:insert_at] + HELPER_BLOCK + text[insert_at:], True


def insert_iteration_guard(text: str) -> tuple[str, bool]:
    # Insert guard above the line defining dbg in _bg_mask_prune
    guard_snip = "gaussians._bg_prune_last_iter"
    if guard_snip in text:
        return text, False

    target_line = '    dbg = bool(int(getattr(opt, "bg_prune_debug", 0)) == 1)'
    pos = text.find(target_line)
    if pos == -1:
        return text, False

    guard_block = (
        "    # One-shot per iteration guard (avoid double execution in same iter)\n"
        "    last_iter = getattr(gaussians, \"_bg_prune_last_iter\", -1)\n"
        "    if last_iter == iteration:\n"
        "        return\n"
        "    gaussians._bg_prune_last_iter = iteration\n"
        "    _bg_prune_reset_stats_if_needed(gaussians)\n\n"
    )
    return text[:pos] + guard_block + text[pos:], True


def replace_seen_hit_init(text: str) -> tuple[str, bool]:
    # Replace the manual init blocks inside _bg_mask_prune
    if "Accumulate evidence (seen: visible; hit: candidate)\n            _bg_prune_reset_stats_if_needed(gaussians)" in text:
        return text, False

    pattern = re.compile(
        r"(            # Accumulate evidence \(seen: visible; hit: candidate\)\n)"
        r"(            if \(not hasattr\(gaussians, \"bg_prune_seen\"\)\) or \(gaussians\.bg_prune_seen\.numel\(\) == 0\):\n"
        r"                gaussians\.bg_prune_seen = torch\.zeros\(\(gaussians\.get_xyz\.shape\[0\], 1\), device=\"cuda\"\)\n"
        r"            if \(not hasattr\(gaussians, \"bg_prune_hit\"\)\) or \(gaussians\.bg_prune_hit\.numel\(\) == 0\):\n"
        r"                gaussians\.bg_prune_hit = torch\.zeros\(\(gaussians\.get_xyz\.shape\[0\], 1\), device=\"cuda\"\)\n)",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return text, False

    repl = m.group(1) + "            _bg_prune_reset_stats_if_needed(gaussians)\n"
    return text[:m.start()] + repl + text[m.end():], True


def insert_slice_before_prune(text: str) -> tuple[str, bool]:
    # Only patch the specific prune inside _bg_mask_prune block
    if "_bg_prune_slice_stats(gaussians" in text:
        return text, False

    # Find the first occurrence of the exact line with indentation
    needle = "                gaussians.prune_points(prune_mask)\n"
    pos = text.find(needle)
    if pos == -1:
        return text, False

    insert = (
        "                # Keep bg-prune evidence aligned with remaining points\n"
        "                _bg_prune_slice_stats(gaussians, (~prune_mask))\n"
    )
    return text[:pos] + insert + text[pos:], True


def main() -> int:
    train_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train.py")
    if not train_path.exists():
        print(f"[ERROR] train.py not found: {train_path}")
        return 2

    original = train_path.read_text(encoding="utf-8", errors="replace")
    patched = original

    changed_any = False

    patched, changed = insert_helpers(patched)
    print(f"[PATCH] helpers inserted: {changed}")
    changed_any |= changed

    patched, changed = insert_iteration_guard(patched)
    print(f"[PATCH] iteration guard inserted: {changed}")
    changed_any |= changed

    patched, changed = replace_seen_hit_init(patched)
    print(f"[PATCH] seen/hit init replaced: {changed}")
    changed_any |= changed

    patched, changed = insert_slice_before_prune(patched)
    print(f"[PATCH] slice-before-prune inserted: {changed}")
    changed_any |= changed

    if not changed_any:
        print("[OK] No changes needed (already patched or patterns not found).")
        return 0

    bak = backup_file(train_path)
    train_path.write_text(patched, encoding="utf-8")
    print(f"[OK] Patched: {train_path}")
    print(f"[OK] Backup:  {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())