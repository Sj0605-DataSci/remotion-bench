"""
Dispatches editing-capacity scoring by task subtype -- the one axis that
genuinely differs per task type. tool_call_correctness, latency, and
video_quality (task_scorer.py) stay subtype-agnostic and are reused as-is.

Structural subtypes check the REAL rendered output or the agent's real
declared state, not just self-reported tool args, wherever that's cheap
to do (audio ducking is verified via ffmpeg astats on the actual render;
reframe is verified by re-running the same YuNet detector on the actual
cropped output).
"""
import json
import subprocess
from pathlib import Path

import cv2


def _lcs_ratio(gt_order: list, pred_order: list) -> float:
    gt = [f for f in gt_order if f in pred_order]
    pred = [f for f in pred_order if f in gt_order]
    if not gt:
        return 0.0
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if gt[i - 1] == pred[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / len(gt)


def score_shot_ordering(task: dict, traj: dict) -> dict:
    gt_order = [s["file"] for s in task["ground_truth"]["shots_in_order"]]
    pred_order = [t["file"] for t in traj["final_state"]["timeline"]]
    return {"sequence_order_accuracy": round(_lcs_ratio(gt_order, pred_order), 3),
            "n_agent_placed": len(pred_order), "n_ground_truth": len(gt_order)}


def score_cut_assembly(task: dict, traj: dict) -> dict:
    pred_order = [t["file"] for t in traj["final_state"]["timeline"]]
    correct = task["ground_truth"]["correct_next_shot"]
    picked_correct = correct in pred_order
    used_distractor = any(f in pred_order for f in task["assets"][2:] if f != correct)
    return {"picked_correct_shot": picked_correct, "used_a_distractor": used_distractor}


def score_caption(task: dict, traj: dict) -> dict:
    gt_segments = task["ground_truth"]["segments"]
    agent_segments = traj["final_state"]["captions"]

    def overlap(a, b):
        return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))

    covered = sum(1 for gt in gt_segments if gt["end"] > gt["start"] and
                  sum(overlap(gt, a) for a in agent_segments) / (gt["end"] - gt["start"]) > 0.5)
    coverage = covered / len(gt_segments) if gt_segments else 0.0
    dead_air = sum(1 for a in agent_segments if a["end"] > a["start"] and
                   sum(overlap(a, gt) for gt in gt_segments) / (a["end"] - a["start"]) < 0.5)
    dead_air_rate = dead_air / len(agent_segments) if agent_segments else 0.0
    return {"speech_coverage_rate": round(coverage, 3), "dead_air_rate": round(dead_air_rate, 3),
            "n_ground_truth": len(gt_segments), "n_agent": len(agent_segments)}


def score_audio_mix(task: dict, rendered_path: Path) -> dict:
    """Real check on the real rendered audio, not self-reported args:
    mean volume during real speech windows vs. during silence."""
    if not rendered_path.exists():
        return {"rendered": False}
    windows = task["ground_truth"]["speech_windows"]

    def mean_volume(start, end):
        out = subprocess.run(
            ["ffmpeg", "-i", str(rendered_path), "-ss", str(start), "-to", str(end),
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        for line in out.stderr.splitlines():
            if "mean_volume" in line:
                return float(line.split(":")[1].strip().replace(" dB", ""))
        return None

    speech_vols = [v for w in windows if (v := mean_volume(*w)) is not None]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(rendered_path)],
                            capture_output=True, text=True)
    total_dur = float(probe.stdout.strip() or 0)
    silence_windows = []
    prev_end = 0.0
    for s, e in sorted(windows):
        if s - prev_end > 1.0:
            silence_windows.append((prev_end, s))
        prev_end = e
    silence_vols = [v for w in silence_windows if (v := mean_volume(*w)) is not None]

    avg_speech = sum(speech_vols) / len(speech_vols) if speech_vols else None
    avg_silence = sum(silence_vols) / len(silence_vols) if silence_vols else None
    ducked = (avg_silence is not None and avg_speech is not None and (avg_silence - avg_speech) >= 3.0)
    return {"avg_volume_during_speech_db": avg_speech, "avg_volume_during_silence_db": avg_silence,
            "ducking_detected": ducked}


def score_text_overlay(task: dict, traj: dict) -> dict:
    gt = task["ground_truth"]
    overlays = traj["final_state"]["text_overlays"]
    match = next((o for o in overlays if gt["text"].lower() in o["text"].lower()), None)
    if not match:
        return {"text_present": False}
    timing_ok = abs(match["start"] - gt["start"]) < 1.0 and abs(match["end"] - gt["end"]) < 1.0
    position_ok = match["position"] == gt["position"]
    return {"text_present": True, "timing_ok": timing_ok, "position_ok": position_ok}


def score_reframe(task: dict, rendered_path: Path, model_path: Path) -> dict:
    """Re-runs the SAME face detector on the actual rendered (cropped)
    output at the source's sampled timestamps -- a face detectable in the
    source but missing in the output at the same t = subject cropped out."""
    if not rendered_path.exists():
        return {"rendered": False}
    src_track = task["ground_truth"]["source_face_track"]
    cap = cv2.VideoCapture(str(rendered_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = cv2.FaceDetectorYN_create(str(model_path), "", (w, h))

    kept, lost, total_with_face = 0, 0, 0
    for f in src_track:
        if f["bbox_xywh"] is None:
            continue
        total_with_face += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(f["t"] * fps))
        ok, frame = cap.read()
        if not ok:
            lost += 1
            continue
        n, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            kept += 1
        else:
            lost += 1
    cap.release()
    return {"subject_retention_rate": round(kept / total_with_face, 3) if total_with_face else None,
            "frames_checked": total_with_face, "frames_lost_subject": lost}


def score_editing_capacity(task: dict, traj: dict, rendered_path: Path, model_path: Path | None = None) -> dict:
    subtype = task["subtype"]
    if subtype == "shot_ordering":
        return score_shot_ordering(task, traj)
    if subtype == "cut_assembly":
        return score_cut_assembly(task, traj)
    if subtype == "caption":
        return score_caption(task, traj)
    if subtype == "audio_mix":
        return score_audio_mix(task, rendered_path)
    if subtype == "text_overlay":
        return score_text_overlay(task, traj)
    if subtype == "reframe":
        return score_reframe(task, rendered_path, model_path)
    return {"note": f"subtype {subtype!r} scored by multimodal_judge, see self_improve/judge.py"}
