"""
Baseline "agent under test": greedy nearest-visual-neighbor chaining.

This stands in for the Diffusion Studio `dapi`-driven LLM agent, which
needs the real Electron desktop app running (dapi talks to it over a
local socket -- "the source is the document", edits are JSX files the
app renders/exports/checks) -- not stood up on this headless ARM64 box
yet. The interface below is deliberately the same shape a real dapi-driven
agent would produce (an ordered list of {file, start, end} placements),
so swapping this module for a real LLM agent is a drop-in.

Policy: start from an arbitrary shot, repeatedly append whichever
remaining shot's FIRST frame is most visually similar (chi-square
histogram distance) to the current shot's LAST frame. This is a real,
legitimate (if weak) editing heuristic -- visual continuity -- not a
peek at ground truth.
"""
import json
import sys
from pathlib import Path

from features import build_shot_features, chi_square_distance
from jsx_writer import write_project


def greedy_chain(task: dict, shots_dir: Path, ext_threshold: float | None = None, averaged: bool = False):
    files = [s["file"] for s in task["shot_pool"]]
    feats = build_shot_features(shots_dir, files, averaged=averaged)

    remaining = set(files)
    # deterministic arbitrary start: lexicographically first
    current = sorted(remaining)[0]
    order = [current]
    remaining.remove(current)
    scene_breaks = []  # indices where similarity was below threshold (self-improve hook)

    while remaining:
        cur_last = feats[current]["last"]
        best, best_d = None, float("inf")
        for f in remaining:
            d = chi_square_distance(cur_last, feats[f]["first"])
            if d < best_d:
                best, best_d = f, d
        if ext_threshold is not None and best_d > ext_threshold:
            scene_breaks.append(len(order))
        order.append(best)
        remaining.remove(best)
        current = best

    return order, scene_breaks


def run(task_path: Path, shots_dir: Path, out_dir: Path, ext_threshold: float | None = None, averaged: bool = False):
    task = json.loads(task_path.read_text())
    order, scene_breaks = greedy_chain(task, shots_dir, ext_threshold, averaged=averaged)

    durations = {s["file"]: s["duration"] for s in task["shot_pool"]}
    t = 0.0
    output_timeline = []
    for f in order:
        dur = durations[f]
        output_timeline.append({"shot_id": f, "start": t, "end": t + dur})
        t += dur

    trajectory = {
        "rendered": True,
        "output_duration": t,
        "output_timeline": output_timeline,
        "scene_breaks_detected": scene_breaks,
        "policy": "greedy_visual_continuity" + ("_averaged_v1" if averaged else "_v0")
        + (f"_threshold={ext_threshold}" if ext_threshold else ""),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    write_project(order, shots_dir, durations, out_dir / "diffusion_studio_project")
    print(f"Wrote trajectory + JSX project to {out_dir}")
    return trajectory


if __name__ == "__main__":
    task_path = Path(sys.argv[1])
    shots_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    positional = [a for a in sys.argv[4:] if not a.startswith("--")]
    threshold = float(positional[0]) if positional else None
    averaged = "--averaged" in sys.argv
    run(task_path, shots_dir, out_dir, threshold, averaged=averaged)
