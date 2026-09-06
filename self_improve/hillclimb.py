"""
The self-improving loop, run for real: baseline agent run (no KB) -> score
-> an LLM (qwen3.8:27b, text) mines a lesson from the trajectory+scores ->
lesson embedded and written to the KB -> SAME tasks rerun with KB-retrieved
lessons injected into the agent's system prompt -> rescore -> compare.

This is Self-Harness's Weakness Mining -> Proposal -> Validation loop,
with the "proposal" concretely being a KB-retrieved ICL exemplar rather
than a code/tool change -- the mechanism discussed early in this project,
now wired to a real dataset, a real tool-calling agent, and a real judge.
"""
import json
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "harness"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scoring"))
from agent_runner import run_agent  # noqa: E402
from run_bundle import make_run_bundle  # noqa: E402
from multi_scorer import score_editing_capacity  # noqa: E402
from kb import LessonKB  # noqa: E402

OLLAMA = "http://localhost:11434/api/chat"
MINER_MODEL = "qwen3.8:27b"

DATASET_DIR = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/dataset_v1")
RUNS_ROOT = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/runs")
KB_PATH = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/lessons_kb.jsonl")
RA = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/remotion-agent")


def load_task(task_id: str) -> dict:
    for line in (DATASET_DIR / "dataset.jsonl").read_text().splitlines():
        t = json.loads(line)
        if t["task_id"] == task_id:
            return t
    raise KeyError(task_id)


def stage_assets(task: dict, subdir_name: str) -> Path:
    """Copies a task's assets into Remotion's public/ dir under the model can reach them."""
    src_dir = DATASET_DIR / subdir_name
    for f in task["assets"]:
        src = src_dir / f
        if src.exists():
            shutil.copy(src, RA / "public" / f)
    return RA / "public"


def mine_lesson(task: dict, trajectory: dict, scores: dict) -> dict | None:
    """Returns {"lesson": str, "scope": "universal"|"task_specific"} or None.

    Confirmed bug this classification fixes: a mined lesson about tool-call
    SEQUENCING (e.g. "always call add_to_timeline before render") scored
    lowest of six lessons under pure embedding similarity against a reframe
    task's instruction (0.547, dead last) -- it shares little vocabulary
    with "9:16 crop" even though it's a universal process rule that applies
    to every task. Similarity retrieval is right for TOPICAL lessons
    (specific to an editing technique) but wrong for PROCESS lessons
    (about tool-call mechanics) -- so lessons are classified at mine time
    and universal ones are always injected regardless of similarity.
    """
    prompt = (
        f"A video editing agent was given this task: \"{task['instruction']}\"\n\n"
        f"It made these tool calls: {json.dumps([c['name'] + '(' + json.dumps(c['args']) + ')' for c in trajectory['calls']])}\n\n"
        f"It scored: {json.dumps(scores)}\n\n"
        "If the score reveals a real mistake (wrong tool order, missed a required step, "
        "wrong argument choice, skipped something the instruction asked for), respond with "
        "JSON: {\"lesson\": \"one concise, general, reusable, imperative sentence, not tied to "
        "this specific clip\", \"scope\": \"universal\" if the lesson is about tool-call "
        "mechanics/sequencing that applies to ANY task (e.g. always populate the timeline "
        "before rendering), or \"task_specific\" if it's about a particular editing technique "
        "(captions, audio ducking, transitions, etc). "
        "If the score is already good and there's nothing worth fixing, respond with exactly: NONE."
    )
    resp = requests.post(OLLAMA, json={"model": MINER_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}, timeout=120)
    text = resp.json().get("message", {}).get("content", "").strip()
    if text.upper().startswith("NONE") or len(text) < 5:
        return None
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        if "lesson" in parsed:
            return {"lesson": parsed["lesson"], "scope": parsed.get("scope", "task_specific")}
    except (ValueError, json.JSONDecodeError):
        pass
    return {"lesson": text, "scope": "task_specific"}
    return text


def run_one(task_id: str, model: str, kb: LessonKB, run_label: str, use_kb: bool, whisper_model, subdir_name: str):
    task = load_task(task_id)
    assets_dir = stage_assets(task, subdir_name)
    lessons = kb.retrieve(task["instruction"], k=3) if use_kb else []

    project_dir = RA
    server, turns = run_agent(task, assets_dir, project_dir, model=model, kb_lessons=lessons, whisper_model=whisper_model)
    trajectory = server.trajectory()
    trajectory["llm_turns"] = turns
    trajectory["model"] = model
    trajectory["kb_lessons_used"] = lessons

    rendered_path = project_dir / "out.mp4"
    scores = score_editing_capacity(task, trajectory, rendered_path)

    bundle = make_run_bundle(task=task, assets_dir=assets_dir, trajectory=trajectory, scores=scores,
                              rendered_path=rendered_path if trajectory["rendered"] else None,
                              runs_root=RUNS_ROOT, run_id=run_label)
    return task, trajectory, scores, bundle


def hillclimb(task_configs: list[tuple[str, str]], model: str = "gemma4:e4b"):
    """task_configs: [(task_id, asset_subdir_name), ...]"""
    kb = LessonKB(KB_PATH)
    results = []

    print("=== BASELINE (no KB) ===")
    baseline = {}
    for task_id, subdir in task_configs:
        task, traj, scores, bundle = run_one(task_id, model, kb, f"baseline_{model}", use_kb=False,
                                              whisper_model=WHISPER, subdir_name=subdir)
        baseline[task_id] = (task, traj, scores)
        print(f"[{task_id}] scores={scores}")
        mined = mine_lesson(task, traj, scores)
        if mined:
            kb.add(mined["lesson"], task_id, task["subtype"], scope=mined["scope"])
            print(f"  MINED LESSON ({mined['scope']}): {mined['lesson']}")
        else:
            print("  no lesson mined (score already clean)")

    print("\n=== CANDIDATE (with KB) ===")
    for task_id, subdir in task_configs:
        task, traj, scores, bundle = run_one(task_id, model, kb, f"candidate_{model}", use_kb=True,
                                              whisper_model=WHISPER, subdir_name=subdir)
        base_scores = baseline[task_id][2]
        print(f"[{task_id}] baseline={base_scores}")
        print(f"[{task_id}] candidate={scores} (lessons used: {len(traj['kb_lessons_used'])})")
        results.append({"task_id": task_id, "baseline": base_scores, "candidate": scores,
                         "lessons_used": traj["kb_lessons_used"]})

    return results


if __name__ == "__main__":
    from faster_whisper import WhisperModel
    WHISPER = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    configs = [
        ("edit_text_overlay_001", "editing_text_overlay"),
        ("edit_audio_mix_001", "editing_audio_mix"),
        ("edit_caption_001", "../task_subtitle/assets"),
    ]
    out = hillclimb(configs)
    print("\n=== SUMMARY ===")
    print(json.dumps(out, indent=2))
