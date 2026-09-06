"""
Weakness Mining stage (Self-Harness's terminology): for every ground-truth
junction, check whether the agent's output actually placed those two shots
adjacent in the same order, and bucket hit/miss by the real MovieCuts
cut-type label at that junction. Tests a concrete hypothesis: does a
pure visual-continuity heuristic fail specifically at junctions that are
inherently visually discontinuous by construction (smash-cut, reaction-cut,
speaker-change) vs succeed at ones that are continuous by construction
(match-cut, cutting-on-action)?
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def mine(ground_truth_path: Path, trajectory_path: Path) -> dict:
    gt = json.loads(ground_truth_path.read_text())
    traj = json.loads(trajectory_path.read_text())

    gt_shots = gt["shots_in_order"]
    pred_order = [t["shot_id"] for t in traj["output_timeline"]]
    pred_pos = {f: i for i, f in enumerate(pred_order)}

    hits = defaultdict(int)
    misses = defaultdict(int)
    miss_examples = defaultdict(list)

    for i in range(len(gt_shots) - 1):
        a, b = gt_shots[i], gt_shots[i + 1]
        cut_types = a.get("cut_type_after") or ["<unlabeled>"]
        if a["file"] not in pred_pos or b["file"] not in pred_pos:
            for ct in cut_types:
                misses[ct] += 1
            continue
        adjacent_and_ordered = pred_pos[b["file"]] == pred_pos[a["file"]] + 1
        for ct in cut_types:
            if adjacent_and_ordered:
                hits[ct] += 1
            else:
                misses[ct] += 1
                if len(miss_examples[ct]) < 2:
                    miss_examples[ct].append({"a": a["file"], "b": b["file"]})

    report = {}
    for ct in sorted(set(hits) | set(misses)):
        total = hits[ct] + misses[ct]
        report[ct] = {
            "n_junctions": total,
            "hit_rate": round(hits[ct] / total, 3) if total else None,
            "examples_missed": miss_examples[ct],
        }
    return dict(sorted(report.items(), key=lambda kv: (kv[1]["hit_rate"] is None, kv[1]["hit_rate"])))


if __name__ == "__main__":
    gt_path, traj_path = Path(sys.argv[1]), Path(sys.argv[2])
    report = mine(gt_path, traj_path)
    print(json.dumps(report, indent=2))
