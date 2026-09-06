"""Runs gemma4:e4b (no KB, baseline capability read) across the 5 tasks not
yet exercised, so "what can it do end to end" is measured, not guessed."""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "harness"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scoring"))
from agent_runner import run_agent  # noqa: E402
from run_bundle import make_run_bundle  # noqa: E402
from multi_scorer import score_editing_capacity  # noqa: E402
from judge import judge_against_reference  # noqa: E402

DS = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/dataset_v1")
RA = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/remotion-agent")
PILOT = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/pilot")
RUNS_ROOT = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/runs")
MODEL = "gemma4:e4b"

TASKS = json.loads if False else None


def load(task_id):
    for line in (DS / "dataset.jsonl").read_text().splitlines():
        t = json.loads(line)
        if t["task_id"] == task_id:
            return t
    raise KeyError(task_id)


def stage(task, src_dir: Path):
    for f in task["assets"]:
        src = src_dir / f
        if src.exists():
            dst = RA / "public" / f
            if not dst.exists():
                shutil.copy(src, dst)


def run(task_id, src_dir, max_turns=20, judge_ref=None):
    task = load(task_id)
    stage(task, src_dir)
    # tasks share one Remotion project_dir/out.mp4 across sequential runs --
    # if a task fails to render, the scorer must not silently read a PRIOR
    # task's leftover output (confirmed bug: reframe scored cut_assembly's
    # stale render this way in an earlier pass).
    stale = RA / "out.mp4"
    if stale.exists():
        stale.unlink()
    t0 = time.time()
    server, turns = run_agent(task, RA / "public", RA, model=MODEL, max_turns=max_turns)
    wall_s = time.time() - t0
    traj = server.trajectory()
    traj["llm_turns"] = turns
    traj["model"] = MODEL
    traj["wall_clock_s"] = round(wall_s, 1)

    rendered_path = RA / "out.mp4"
    if not traj["rendered"]:
        scores = {"rendered": False, "reason": "agent never called render() successfully"}
    elif task["scoring_method"] == "structural":
        model_path = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/models/face_detection_yunet.onnx")
        scores = score_editing_capacity(task, traj, rendered_path, model_path)
    else:
        if traj["rendered"] and judge_ref and judge_ref.exists():
            scores = judge_against_reference(task, rendered_path, judge_ref)
        else:
            scores = {"judged": False, "reason": "not rendered or no reference available"}

    make_run_bundle(task=task, assets_dir=RA / "public", trajectory=traj, scores=scores,
                     rendered_path=rendered_path if traj["rendered"] else None,
                     runs_root=RUNS_ROOT, run_id=f"capability_{MODEL}",
                     reference_video=judge_ref)
    return {"task_id": task_id, "rendered": traj["rendered"], "n_calls": traj["n_calls"],
            "n_failed_calls": traj["n_failed_calls"], "wall_clock_s": traj["wall_clock_s"],
            "n_turns": len(turns), "scores": scores}


if __name__ == "__main__":
    results = []
    results.append(run("edit_cut_assembly_001", PILOT / "shots", max_turns=25))
    results.append(run("edit_reframe_001", DS / "editing_reframe"))
    results.append(run("edit_transition_001", DS / "editing_transition",
                        judge_ref=DS / "editing_transition" / "reference_transition_example.mp4"))
    results.append(run("gen_scene_assembly_001", DS / "generation_scene_assembly",
                        judge_ref=DS / "generation_scene_assembly" / "reference.mp4", max_turns=35))
    results.append(run("edit_shot_ordering_001", PILOT / "shots", max_turns=90))

    print("\n\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
