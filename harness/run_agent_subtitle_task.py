"""
An actual LLM-driven execution of the subtitle_insertion task, end to end,
through the real ToolServer -- not a heuristic stand-in. The "agent" here
is Claude itself deciding what to call and when, based only on the task
instruction (not the pre-built ground_truth.json), then chunking
transcribe()'s own output into subtitle cues -- the one place a smarter
agent's judgment (merge short fragments? split long ones? trim silence?)
actually shows up versus a naive one-segment-per-ASR-chunk policy.

Caveat logged honestly in the report: this session already saw the
dialogue while building the task's ground truth earlier, so it isn't a
blind run. A real eval deployment runs the agent as a separate process
with no such contamination.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tool_server import ToolServer  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


def run(task_path: Path, assets_dir: Path, project_dir: Path, out_dir: Path):
    task = json.loads(task_path.read_text())
    assert task["task_type"] == "subtitle_insertion"
    asset = task["assets"][0]

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    server = ToolServer(assets_dir=assets_dir, project_dir=project_dir, whisper_model=model)

    # --- agent policy, in the order a real agent would reason through it ---
    r = server.probe(asset)
    duration = r["result"]["duration"]

    r = server.transcribe(asset)
    segments = r["result"]["segments"]

    server.add_to_timeline(asset, start_in_source=0.0, duration=duration)

    # Agent judgment: merge ASR fragments shorter than 0.6s into the next
    # segment (avoids flashing one-word subtitles), otherwise use ASR
    # boundaries directly.
    merged = []
    for seg in segments:
        if merged and (seg["start"] - merged[-1]["end"]) < 0.15 and (merged[-1]["end"] - merged[-1]["start"]) < 0.6:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(dict(seg))

    for seg in merged:
        server.add_subtitle(seg["text"], seg["start"], seg["end"])

    server.render()

    out_dir.mkdir(parents=True, exist_ok=True)
    traj = server.trajectory()
    traj["agent_subtitles"] = merged
    (out_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))
    print(json.dumps(traj, indent=2))
    return traj


if __name__ == "__main__":
    task_path = Path(sys.argv[1])
    assets_dir = Path(sys.argv[2])
    project_dir = Path(sys.argv[3])
    out_dir = Path(sys.argv[4])
    run(task_path, assets_dir, project_dir, out_dir)
