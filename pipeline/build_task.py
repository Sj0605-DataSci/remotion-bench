"""
Builds one "reassembly" eval task from a MovieCuts video_id:
  1. Reads annotated_clips CSV rows for the video, chains shot_left/shot_right
     into a deduped, time-ordered shot list (ground truth).
  2. Extracts each shot to its own mp4 via ffmpeg.
  3. Pulls distractor shots from OTHER movies in the CSV (same-ish duration)
     so selection tasks have real decoys.
  4. Writes ground_truth.json (real order + timestamps + cut-type labels)
     and task.json (shuffled shot pool + instruction, no answer key).

This is step 1-3 of the pipeline discussed with the user:
  pick movie_id -> sort by shot_left_start -> extract shots -> raw material.
"""
import csv
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

MOVIECUTS_DATA = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/MovieCuts/data")
CUT_TYPES = json.load(open(MOVIECUTS_DATA / "cut-type-train.json"))["cut_types"]
CUT_LABELS = json.load(open(MOVIECUTS_DATA / "cut-type-train.json"))["annotations"]


def load_rows_for_video(video_id: str, split: str = "train"):
    rows = []
    with open(MOVIECUTS_DATA / f"annotated_clips_{split}.csv") as f:
        for row in csv.DictReader(f):
            if row["video_id"] == video_id:
                rows.append(row)
    rows.sort(key=lambda r: float(r["shot_left_start"]))
    return rows


def chain_shots(rows):
    """Dedup shot_left/shot_right pairs across consecutive rows into one
    ordered shot list, keeping the clip_id (and its cut-type label) that
    marks the JUNCTION after each shot."""
    shots = []
    seen = set()
    for i, row in enumerate(rows):
        for side in ("left", "right"):
            start, end = float(row[f"shot_{side}_start"]), float(row[f"shot_{side}_end"])
            key = (round(start, 2), round(end, 2))
            if key not in seen:
                seen.add(key)
                shots.append({"start": start, "end": end})
    shots.sort(key=lambda s: s["start"])

    # attach the cut-type label that governs the junction AFTER each shot
    for i, row in enumerate(rows):
        junction_after = float(row["shot_left_end"])
        clip_id = row["clip_id"]
        labels = CUT_LABELS.get(clip_id, {}).get("labels")
        cut_types = [CUT_TYPES[j] for j, v in enumerate(labels)] if labels and 1 in labels else []
        cut_types = [CUT_TYPES[j] for j, v in enumerate(labels) if v == 1] if labels else []
        for s in shots:
            if abs(s["end"] - junction_after) < 0.05:
                s["cut_type_after"] = cut_types
                s["clip_id"] = clip_id
    return shots


def extract_shots(video_path: Path, shots: list, out_dir: Path):
    """Extracts each shot under an opaque, order-uninformative id (a short
    hash of its clip_id/timestamps) rather than a chronological index --
    shot_000.mp4, shot_001.mp4, ... would let an agent "solve" the task by
    just sorting filenames instead of actually watching the footage."""
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    for s in shots:
        dur = s["end"] - s["start"]
        opaque_id = hashlib.sha1(f"{s['start']}_{s['end']}".encode()).hexdigest()[:8]
        out_path = out_dir / f"clip_{opaque_id}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(s["start"]), "-i", str(video_path),
                "-t", str(dur),
                "-c:v", "libx264", "-c:a", "aac", "-avoid_negative_ts", "make_zero",
                str(out_path),
            ],
            check=True,
        )
        s["file"] = out_path.name


def sample_distractors(exclude_video_id: str, n: int, split: str = "train", seed: int = 0):
    """Pull n shot spans from OTHER videos as decoys. We only need
    timestamps here; extraction of decoy pixels happens lazily by the
    caller if a task actually needs distractor VIDEO files."""
    rng = random.Random(seed)
    pool = []
    with open(MOVIECUTS_DATA / f"annotated_clips_{split}.csv") as f:
        for row in csv.DictReader(f):
            if row["video_id"] != exclude_video_id:
                pool.append(row)
    picks = rng.sample(pool, min(n, len(pool)))
    return [
        {
            "video_id": r["video_id"],
            "start": float(r["shot_left_start"]),
            "end": float(r["shot_left_end"]),
            "movie_id": r["movie_id"],
        }
        for r in picks
    ]


def build(video_id: str, raw_video: Path, out_dir: Path, split: str = "train", n_distractors: int = 8):
    rows = load_rows_for_video(video_id, split)
    if not rows:
        raise SystemExit(f"No annotated rows found for video_id={video_id} in split={split}")
    shots = chain_shots(rows)
    extract_shots(raw_video, shots, out_dir / "shots")

    ground_truth = {
        "video_id": video_id,
        "movie_id": rows[0]["movie_id"],
        "n_shots": len(shots),
        "shots_in_order": shots,  # ground-truth order + timestamps + cut_type_after
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    shuffled = list(range(len(shots)))
    random.Random(42).shuffle(shuffled)
    distractors = sample_distractors(video_id, n_distractors, split)

    task = {
        "video_id": video_id,
        "instruction": (
            "You are given a pool of raw video shots (files shot_XXX.mp4) taken "
            "from the same scene, presented in NO particular order. Some additional "
            "'distractor' shots from unrelated footage may also be in the pool. "
            "Assemble the real shots into a single coherent edited sequence: choose "
            "which shots belong, their order, and where you would place cuts. "
            "Do not use any distractor shots."
        ),
        "shot_pool": [
            {"file": shots[i]["file"], "duration": round(shots[i]["end"] - shots[i]["start"], 2)}
            for i in shuffled
        ],
        "distractor_pool": distractors,
    }
    (out_dir / "task.json").write_text(json.dumps(task, indent=2))
    print(f"Built task: {len(shots)} real shots, {len(distractors)} distractors -> {out_dir}")


if __name__ == "__main__":
    import sys

    video_id = sys.argv[1] if len(sys.argv) > 1 else "1cHRBd6l2UM"
    raw_video = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/pilot/raw/source.mp4"
    )
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        "/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/pilot"
    )
    build(video_id, raw_video, out_dir)
