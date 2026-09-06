"""
Standard per-run artifact folder, produced for every eval run so a human
can inspect what actually happened without re-running anything:

  runs/<task_id>/<run_id>/
    instruction.txt        - the task instruction given to the agent
    input_frames/           - one representative frame per input asset
    output.mp4              - the agent's actual rendered result (copied)
    output_frames/          - a few sampled frames from that output
    expected/
      reference.mp4 + reference_frames/   (generation tasks: the real video
                                            these assets were meant to become)
      ground_truth.json                    (editing tasks: the checkable
                                            spec -- timestamps, labels, etc)
    trajectory.json         - full tool-call log with latency
    scores.json             - the four-axis score for this run
"""
import json
import shutil
import subprocess
from pathlib import Path


def _extract_frame(video_path: Path, out_path: Path, at_s: float = 0.3):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at_s), "-i", str(video_path),
         "-frames:v", "1", "-vf", "scale=320:-1", str(out_path)],
        check=False,  # asset might be shorter than at_s -- don't crash the whole bundle over a thumbnail
    )


def make_run_bundle(task: dict, assets_dir: Path, trajectory: dict, scores: dict,
                     rendered_path: Path, runs_root: Path, run_id: str,
                     reference_video: Path | None = None) -> Path:
    bundle = runs_root / task["task_id"] / run_id
    (bundle / "input_frames").mkdir(parents=True, exist_ok=True)
    (bundle / "output_frames").mkdir(parents=True, exist_ok=True)
    (bundle / "expected").mkdir(parents=True, exist_ok=True)

    (bundle / "instruction.txt").write_text(task["instruction"])

    for asset in task["assets"]:
        asset_path = assets_dir / asset
        if asset_path.exists() and asset_path.suffix in (".mp4", ".mov", ".webm"):
            _extract_frame(asset_path, bundle / "input_frames" / f"{asset}.jpg")

    if rendered_path and rendered_path.exists():
        shutil.copy(rendered_path, bundle / "output.mp4")
        for i, t in enumerate([0.3, 0.5, 0.8]):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(rendered_path)],
                capture_output=True, text=True,
            )
            try:
                dur = float(probe.stdout.strip())
            except ValueError:
                dur = 1.0
            _extract_frame(rendered_path, bundle / "output_frames" / f"frame_{i}.jpg", at_s=dur * t)

    if reference_video and reference_video.exists():
        shutil.copy(reference_video, bundle / "expected" / "reference.mp4")
        (bundle / "expected" / "reference_frames").mkdir(exist_ok=True)
        for i, t in enumerate([0.3, 0.5, 0.8]):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(reference_video)],
                capture_output=True, text=True,
            )
            try:
                dur = float(probe.stdout.strip())
            except ValueError:
                dur = 1.0
            _extract_frame(reference_video, bundle / "expected" / "reference_frames" / f"frame_{i}.jpg", at_s=dur * t)
    else:
        (bundle / "expected" / "ground_truth.json").write_text(json.dumps(task["ground_truth"], indent=2))

    (bundle / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    (bundle / "scores.json").write_text(json.dumps(scores, indent=2))

    return bundle
