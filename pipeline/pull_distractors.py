"""
Closes the gap flagged in the round-1 dataset: edit_cut_assembly_001's
distractor pool referenced raw YouTube video_ids, not playable clips.
Downloads each distractor source once, extracts the exact annotated span,
saves under an opaque filename (same anti-leakage rule as build_task.py).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DISTRACTOR_DIR = Path(
    "/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/distractors"
)


def download(video_id: str, out_path: Path):
    if out_path.exists():
        return True
    fmt_candidates = ["134+140", "18", "best[height<=480]"]
    for fmt in fmt_candidates:
        proc = subprocess.run(
            ["python3", "-m", "yt_dlp", "--js-runtimes", "node", "-f", fmt,
             "--merge-output-format", "mp4", "-o", str(out_path),
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=120,
        )
        if out_path.exists():
            return True
    print(f"  FAILED to download {video_id}: {proc.stderr[-300:]}")
    return False


def extract_clip(source: Path, start: float, end: float, out_path: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(source),
         "-t", str(end - start), "-c:v", "libx264", "-c:a", "aac",
         "-avoid_negative_ts", "make_zero", str(out_path)],
        check=True,
    )


def main(task_path: Path):
    task = json.loads(task_path.read_text())
    distractor_pool = task["distractor_pool"]
    DISTRACTOR_DIR.mkdir(parents=True, exist_ok=True)

    resolved = []
    for d in distractor_pool:
        vid = d["video_id"]
        raw_path = DISTRACTOR_DIR / f"{vid}.mp4"
        print(f"pulling distractor {vid} ({d['movie_id']}) ...")
        ok = download(vid, raw_path)
        if not ok:
            resolved.append({**d, "file": None, "status": "download_failed"})
            continue
        opaque = hashlib.sha1(f"distractor_{vid}_{d['start']}_{d['end']}".encode()).hexdigest()[:8]
        clip_path = DISTRACTOR_DIR / f"clip_{opaque}.mp4"
        try:
            extract_clip(raw_path, d["start"], d["end"], clip_path)
            resolved.append({**d, "file": clip_path.name, "status": "ok"})
        except subprocess.CalledProcessError as e:
            resolved.append({**d, "file": None, "status": f"extract_failed: {e}"})

    out_path = DISTRACTOR_DIR / "resolved_distractors.json"
    out_path.write_text(json.dumps(resolved, indent=2))
    ok_count = sum(1 for r in resolved if r["status"] == "ok")
    print(f"\n{ok_count}/{len(resolved)} distractors resolved to real clips -> {out_path}")
    return resolved


if __name__ == "__main__":
    main(Path(sys.argv[1]))
