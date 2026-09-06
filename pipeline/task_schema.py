"""
Task dataset schema for end-to-end instructed editing tasks -- the format
the user asked for: instruction + assets + checkable ground truth, executed
by an agent through real tool calls, not a shuffle-and-reconstruct proxy.

{
  "task_id": str,
  "task_type": "subtitle_insertion" | "cut_assembly" | "trim_highlight" | "broll_insertion",
  "instruction": str,               # natural-language instruction, as a user would give it
  "assets": [str],                  # filenames under assets/
  "expected_tool_sequence": [str],  # tool NAMES a competent agent must call, in a valid
                                     # partial order (used for tool-call-correctness scoring,
                                     # not an exact-match script)
  "ground_truth": {...},            # task-type-specific, see below
}

subtitle_insertion ground_truth:
  { "segments": [{"start": float, "end": float, "text": str}, ...] }   # from real ASR
  Scoring axes (see scoring/task_scorer.py):
    - timing coverage: does every ground-truth speech interval have an agent
      subtitle overlapping it?
    - no dead air: agent didn't caption silence / off-transcript time
    - count sanity: agent segment count in the right ballpark (not 1 giant
      subtitle, not one per word)

cut_assembly ground_truth: (reuses pipeline/build_task.py's MovieCuts output)
  { "correct_next_shot": str, "cut_type": [str], "distractor_shots": [str] }
"""
import json
from pathlib import Path


def build_subtitle_task(task_dir: Path, asset_file: str, whisper_segments: list, window_end: float):
    task_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "task_id": "subtitle_insertion_pilot_001",
        "task_type": "subtitle_insertion",
        "instruction": (
            f"Add accurate, well-timed subtitles to {asset_file}. Subtitles should be "
            "synced closely to the spoken dialogue -- no dead air captioned, no speech "
            "left uncaptioned, and don't merge unrelated lines into one subtitle."
        ),
        "assets": [asset_file],
        "expected_tool_sequence": ["probe", "transcribe", "add_to_timeline", "add_subtitle", "render"],
        "ground_truth": {
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in whisper_segments if s["start"] < window_end
            ],
            "duration": window_end,
            "note": "ground truth is faster-whisper tiny.en ASR on the same clip -- a "
                     "reasonable proxy for timing/coverage scoring, not a human-verified "
                     "transcript. Fine for scoring sync/coverage, not wording accuracy.",
        },
    }
    (task_dir / "task.json").write_text(json.dumps(task, indent=2))
    return task
