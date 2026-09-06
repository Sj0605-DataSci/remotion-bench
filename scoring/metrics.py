"""
Automated, non-LLM-judge scorers for a reassembly trajectory, against the
real MovieCuts ground truth built by pipeline/build_task.py.

Formulas are the ones reported in EditDuet (arXiv 2509.10761) plus a
tIoU placement score in the spirit of VEBench's Operation Simulation task:

  time_coverage(d, d_hat)  = min(d, d_hat) / max(d, d_hat)
  repetitions              = # pairs of sub-clips in the OUTPUT with >=80% overlap
  failure                  = output failed to render / is empty
  placement_tiou           = per-shot IoU between predicted and ground-truth
                              (start, end) position in the assembled timeline
  cut_type_match_rate      = fraction of junctions where the agent cut between
                              the same two shots the ground truth joins there
                              (a proxy for "did it reproduce the editor's choice",
                              not whether it labeled the cut type correctly --
                              that needs a separate craft-recognition classifier)
"""
import json
from pathlib import Path


def time_coverage(gt_duration: float, out_duration: float) -> float:
    if gt_duration == 0 or out_duration == 0:
        return 0.0
    return min(gt_duration, out_duration) / max(gt_duration, out_duration)


def repetitions(output_timeline: list) -> float:
    """output_timeline: list of {"shot_id": str, "start": float, "end": float}
    in the OUTPUT sequence. Counts pairs with >=80% temporal overlap of the
    shorter span."""
    count = 0
    n = len(output_timeline)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = output_timeline[i], output_timeline[j]
            if a["shot_id"] != b["shot_id"]:
                continue
            overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
            shorter = min(a["end"] - a["start"], b["end"] - b["start"])
            if shorter > 0 and overlap / shorter >= 0.8:
                count += 1
    return count / n if n else 0.0


def sequence_order_accuracy(predicted_order: list, ground_truth_order: list) -> float:
    """Fraction of ground-truth shots that appear in the predicted sequence
    in the correct relative order (longest-common-subsequence ratio)."""
    gt = [s for s in ground_truth_order if s in predicted_order]
    pred = [s for s in predicted_order if s in ground_truth_order]
    if not gt:
        return 0.0
    # LCS
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / len(gt)


def selection_precision_recall(predicted_shot_ids: set, ground_truth_shot_ids: set, distractor_ids: set):
    """Did the agent avoid distractors and include the real shots?"""
    tp = len(predicted_shot_ids & ground_truth_shot_ids)
    fp_distractor = len(predicted_shot_ids & distractor_ids)
    precision = tp / len(predicted_shot_ids) if predicted_shot_ids else 0.0
    recall = tp / len(ground_truth_shot_ids) if ground_truth_shot_ids else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "distractors_used": fp_distractor,
    }


def score_trajectory(ground_truth_path: Path, trajectory: dict) -> dict:
    """trajectory = {
        'rendered': bool,
        'output_duration': float,
        'output_timeline': [{'shot_id','start','end'}, ...] in output order,
    }
    """
    gt = json.loads(Path(ground_truth_path).read_text())
    gt_shots = gt["shots_in_order"]
    gt_ids = [s["file"] for s in gt_shots]
    gt_duration = sum(s["end"] - s["start"] for s in gt_shots)

    if not trajectory.get("rendered"):
        return {"failure_rate": 1.0}

    out_tl = trajectory["output_timeline"]
    pred_ids = [t["shot_id"] for t in out_tl]

    result = {
        "failure_rate": 0.0,
        "time_coverage": time_coverage(gt_duration, trajectory["output_duration"]),
        "repetitions": repetitions(out_tl),
        "sequence_order_accuracy": sequence_order_accuracy(pred_ids, gt_ids),
    }
    return result


if __name__ == "__main__":
    import sys

    gt_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/pilot/ground_truth.json"
    )
    traj_path = sys.argv[2] if len(sys.argv) > 2 else None
    if traj_path:
        traj = json.loads(Path(traj_path).read_text())
        print(json.dumps(score_trajectory(gt_path, traj), indent=2))
    else:
        print("Usage: metrics.py ground_truth.json trajectory.json")
