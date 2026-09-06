"""
Scores one end-to-end task trajectory on the four axes the eval actually
needs (per the correction: reassembly-only metrics couldn't tell whether
an agent given a task + assets can produce the requested video):

  1. tool_call_correctness - right tools, right order, right preconditions
  2. latency               - per-call and total wall-clock, real numbers
  3. video_quality         - structural checks on the real rendered file
                             (DOVER wiring is the documented next step --
                             needs torch, not installed on this box yet)
  4. editing_capacity      - did the OUTPUT satisfy the INSTRUCTION,
                             checked against task-type-specific ground
                             truth (subtitle timing coverage here)
"""
import json
import subprocess
from pathlib import Path


def score_tool_call_correctness(task: dict, trajectory: dict) -> dict:
    expected = task["expected_tool_sequence"]
    called = [c["name"] for c in trajectory["calls"]]
    called_set = set(called)

    # every expected tool was used at least once
    coverage = sum(1 for e in expected if e in called_set) / len(expected)

    # partial-order correctness: transcribe before any add_subtitle,
    # add_to_timeline before any add_subtitle, render last
    order_violations = []
    if "add_subtitle" in called_set and "transcribe" in called_set:
        if called.index("transcribe") > called.index("add_subtitle"):
            order_violations.append("add_subtitle called before transcribe")
    if "add_subtitle" in called_set and "add_to_timeline" in called_set:
        if called.index("add_to_timeline") > called.index("add_subtitle"):
            order_violations.append("add_subtitle called before add_to_timeline")
    if "render" in called_set and called[-1] != "render":
        order_violations.append("render was not the final call")

    return {
        "expected_tool_coverage": round(coverage, 3),
        "n_failed_calls": trajectory["n_failed_calls"],
        "order_violations": order_violations,
        "correct": coverage == 1.0 and not order_violations and trajectory["n_failed_calls"] == 0,
    }


def score_latency(trajectory: dict) -> dict:
    by_tool = {}
    for c in trajectory["calls"]:
        by_tool.setdefault(c["name"], []).append(c["latency_s"])
    return {
        "total_latency_s": trajectory["total_latency_s"],
        "by_tool_s": {k: round(sum(v), 4) for k, v in by_tool.items()},
        "render_latency_s": by_tool.get("render", [0])[0] if "render" in by_tool else None,
    }


def score_video_quality(rendered_path: Path, expected_duration: float) -> dict:
    """Structural checks that don't need a model -- DOVER (already cloned,
    scoring/README) is the documented next step for a real perceptual
    quality score once torch is installed."""
    if not rendered_path.exists():
        return {"rendered": False}
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,duration",
         "-of", "json", str(rendered_path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    duration = float(info.get("duration", 0))
    # cheap black-frame proxy: mean luma of a handful of sampled frames
    blackdetect = subprocess.run(
        ["ffmpeg", "-i", str(rendered_path), "-vf", "blackdetect=d=0.3:pic_th=0.98",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    black_frames_detected = "black_start" in blackdetect.stderr
    return {
        "rendered": True,
        "resolution": f"{info['width']}x{info['height']}",
        "duration_s": duration,
        "duration_matches_target": abs(duration - expected_duration) < 0.5,
        "black_frames_detected": black_frames_detected,
        "note": "structural checks only -- DOVER perceptual score is the next step (needs torch)",
    }


def score_editing_capacity_subtitles(ground_truth_segments: list, agent_segments: list) -> dict:
    """Timing coverage: for each real speech interval, is there an agent
    subtitle overlapping it? And the inverse: did the agent caption dead
    air (an agent subtitle with no real speech under it)?"""
    def overlaps(a, b):
        return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))

    covered = 0
    for gt in ground_truth_segments:
        gt_dur = gt["end"] - gt["start"]
        covered_dur = sum(overlaps(gt, a) for a in agent_segments)
        if gt_dur > 0 and covered_dur / gt_dur > 0.5:
            covered += 1
    coverage_rate = covered / len(ground_truth_segments) if ground_truth_segments else 0.0

    dead_air_subs = 0
    for a in agent_segments:
        a_dur = a["end"] - a["start"]
        real_speech_dur = sum(overlaps(a, gt) for gt in ground_truth_segments)
        if a_dur > 0 and real_speech_dur / a_dur < 0.5:
            dead_air_subs += 1
    dead_air_rate = dead_air_subs / len(agent_segments) if agent_segments else 0.0

    return {
        "speech_coverage_rate": round(coverage_rate, 3),
        "dead_air_rate": round(dead_air_rate, 3),
        "n_ground_truth_segments": len(ground_truth_segments),
        "n_agent_segments": len(agent_segments),
    }


def score_all(task_path: Path, trajectory_path: Path, rendered_path: Path) -> dict:
    task = json.loads(task_path.read_text())
    trajectory = json.loads(trajectory_path.read_text())
    return {
        "tool_call_correctness": score_tool_call_correctness(task, trajectory),
        "latency": score_latency(trajectory),
        "video_quality": score_video_quality(rendered_path, task["ground_truth"]["duration"]),
        "editing_capacity": score_editing_capacity_subtitles(
            task["ground_truth"]["segments"], trajectory["agent_subtitles"]
        ),
    }


if __name__ == "__main__":
    import sys

    task_path, traj_path, rendered_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    print(json.dumps(score_all(task_path, traj_path, rendered_path), indent=2))
