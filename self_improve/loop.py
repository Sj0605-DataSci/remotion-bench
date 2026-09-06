"""
Self-Harness-style loop (Weakness Mining -> Proposal -> Validation),
orchestrating the pieces built in harness/ and scoring/ against one
task built by pipeline/build_task.py.

This is intentionally a thin orchestrator, not a new mechanism: every
step below calls a module that was independently run and verified in
the pilot (agent_greedy, metrics, mine_weaknesses). Wiring them into one
loop is what turns "we can score a run" into "we can hillclimb."

Usage:
  python3 loop.py <pilot_dir> <policy_a> <policy_b>
    policy in {"v0", "v1_averaged"}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "harness"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scoring"))

from agent_greedy import run as run_agent  # noqa: E402
from metrics import score_trajectory  # noqa: E402
from mine_weaknesses import mine  # noqa: E402


def iterate(pilot_dir: Path, policy: str, run_name: str):
    task_path = pilot_dir / "task.json"
    shots_dir = pilot_dir / "shots"
    gt_path = pilot_dir / "ground_truth.json"
    out_dir = pilot_dir / run_name

    averaged = policy == "v1_averaged"
    traj = run_agent(task_path, shots_dir, out_dir, ext_threshold=None, averaged=averaged)
    scores = score_trajectory(gt_path, {**traj})
    weaknesses = mine(gt_path, out_dir / "trajectory.json")
    return scores, weaknesses


def main():
    pilot_dir = Path(sys.argv[1])
    baseline_policy = sys.argv[2] if len(sys.argv) > 2 else "v0"
    candidate_policy = sys.argv[3] if len(sys.argv) > 3 else "v1_averaged"

    print(f"=== Baseline: {baseline_policy} ===")
    base_scores, base_weak = iterate(pilot_dir, baseline_policy, "loop_baseline")
    print(json.dumps(base_scores, indent=2))
    worst = sorted(base_weak.items(), key=lambda kv: (kv[1]["hit_rate"] is None, kv[1]["hit_rate"]))[:3]
    print("Weakest cut-type buckets:", [w[0] for w in worst])

    print(f"\n=== Candidate: {candidate_policy} ===")
    cand_scores, cand_weak = iterate(pilot_dir, candidate_policy, "loop_candidate")
    print(json.dumps(cand_scores, indent=2))

    print("\n=== Validation gate ===")
    key = "sequence_order_accuracy"
    delta = cand_scores[key] - base_scores[key]
    verdict = "PROMOTE" if delta > 0 else "REJECT"
    print(f"{key}: {base_scores[key]:.3f} -> {cand_scores[key]:.3f}  (delta={delta:+.3f})  -> {verdict}")
    print(
        "\nNOTE: validated on the SAME task the weakness was mined from. "
        "Before trusting this in the real harness, rerun on a held-out "
        "movie_id never used for mining -- this loop does not yet do that "
        "split automatically."
    )


if __name__ == "__main__":
    main()
