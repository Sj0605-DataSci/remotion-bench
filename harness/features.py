"""
Real, non-ground-truth-peeking visual signal for the baseline agent:
first/last-frame color histograms per shot, used for greedy nearest-
neighbor chaining. This is a legitimate, classic shot-continuity heuristic
(same family of idea as SKALD's learned coherence score, just unlearned) --
not a stand-in for the ground truth, an actual (weak) editing heuristic.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

BINS = 16


def _extract_frame(video_path: Path, offset_from_end: float | None, offset_from_start: float | None, out_png: Path):
    if offset_from_end is not None:
        args = ["-sseof", f"-{offset_from_end}", "-i", str(video_path)]
    else:
        args = ["-ss", str(offset_from_start or 0), "-i", str(video_path)]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args, "-frames:v", "1", str(out_png)],
        check=True,
    )


def histogram(png_path: Path) -> np.ndarray:
    img = Image.open(png_path).convert("RGB").resize((64, 64))
    arr = np.asarray(img)
    hist = np.concatenate(
        [np.histogram(arr[:, :, c], bins=BINS, range=(0, 255))[0] for c in range(3)]
    ).astype(np.float64)
    return hist / (hist.sum() + 1e-9)


def averaged_histogram(video_path: Path, at_end: bool, tmp: Path, n_frames: int = 3) -> np.ndarray:
    """v1 fix (mined weakness: v0's single boundary frame is noisy -- motion
    blur / a mid-action frame can dominate the match). Averages histograms
    over a small window near the boundary instead of one frame."""
    hists = []
    offsets = [0.02, 0.12, 0.25] if not at_end else [0.02, 0.12, 0.25]
    for i, off in enumerate(offsets[:n_frames]):
        png = tmp / f"{video_path.stem}_{'end' if at_end else 'start'}_{i}.png"
        if at_end:
            _extract_frame(video_path, offset_from_end=off, offset_from_start=None, out_png=png)
        else:
            _extract_frame(video_path, offset_from_end=None, offset_from_start=off, out_png=png)
        if png.exists() and png.stat().st_size > 0:
            hists.append(histogram(png))
    if not hists:
        return np.ones(BINS * 3) / (BINS * 3)
    return np.mean(hists, axis=0)


def chi_square_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2 / (a + b + 1e-9)))


def build_shot_features(shots_dir: Path, shot_files: list, averaged: bool = False) -> dict:
    """Returns {file: {"first": hist, "last": hist}}.
    averaged=True uses the v1 multi-frame-window fix instead of v0's
    single boundary frame."""
    feats = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in shot_files:
            video_path = shots_dir / f
            if averaged:
                feats[f] = {
                    "first": averaged_histogram(video_path, at_end=False, tmp=tmp),
                    "last": averaged_histogram(video_path, at_end=True, tmp=tmp),
                }
            else:
                first_png, last_png = tmp / f"{f}.first.png", tmp / f"{f}.last.png"
                _extract_frame(video_path, offset_from_end=None, offset_from_start=0, out_png=first_png)
                _extract_frame(video_path, offset_from_end=0.15, offset_from_start=None, out_png=last_png)
                feats[f] = {"first": histogram(first_png), "last": histogram(last_png)}
    return feats
