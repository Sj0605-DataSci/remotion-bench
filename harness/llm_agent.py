"""
A genuinely separate LLM agent (Ollama, local -- no shared context with the
orchestrating session) executing the subtitle_insertion task through real
tool calls against ToolServer. This replaces the earlier self-play run,
which was contaminated: this session had already built the task's ground
truth before "acting as the agent." An Ollama call has zero access to that
context -- it only ever sees the system prompt + task instruction assembled
fresh here, so it's blind regardless of which clip is used.

nemotron-mini:4.2B did not reliably emit tool_calls against Ollama's
generic template in smoke tests (empty content, no tool_calls field across
two prompt variants) -- llama3.1:8b did, reliably. Using that as the
executing agent; nemotron-mini is worth retrying with a different
template/runtime later, not written off definitively from two tries.
"""
import json
import time
from pathlib import Path

import requests

from tool_server import ToolServer

OLLAMA = "http://localhost:11434/api/chat"

TOOLS = [
    {"type": "function", "function": {
        "name": "probe", "description": "Get the duration in seconds of a video asset. Call this first.",
        "parameters": {"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "transcribe", "description": "Run speech-to-text on a video asset, returns timed segments.",
        "parameters": {"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "add_to_timeline", "description": "Place an asset onto the edit timeline.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"}, "start_in_source": {"type": "number"}, "duration": {"type": "number"}},
            "required": ["file", "start_in_source", "duration"]}}},
    {"type": "function", "function": {
        "name": "add_subtitle", "description": "Add one subtitle cue. Call once per caption, after add_to_timeline.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"}},
            "required": ["text", "start", "end"]}}},
    {"type": "function", "function": {
        "name": "render", "description": "Render the final video. Call this LAST, once, after all subtitles are added.",
        "parameters": {"type": "object", "properties": {}}}},
]


def run_llm_agent(task: dict, assets_dir: Path, project_dir: Path, model: str = "llama3.1:8b",
                   kb_lessons: list[str] | None = None, max_turns: int = 25, whisper_model=None):
    server = ToolServer(assets_dir=assets_dir, project_dir=project_dir, whisper_model=whisper_model)

    system = (
        "You are a video editing agent. You accomplish tasks ONLY by calling the "
        "provided tools -- you cannot see or hear the video yourself, only tool "
        "results. Work step by step: probe the asset, transcribe it, place it on "
        "the timeline, add one subtitle per spoken line using the transcript's own "
        "timings, then render exactly once as your final action."
    )
    if kb_lessons:
        system += "\n\nLessons from prior runs, apply them where relevant:\n" + "\n".join(f"- {l}" for l in kb_lessons)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{task['instruction']}\nAsset file: {task['assets'][0]}"},
    ]

    turns = []
    for turn in range(max_turns):
        t0 = time.time()
        resp = requests.post(OLLAMA, json={"model": model, "messages": messages, "tools": TOOLS, "stream": False})
        llm_latency = time.time() - t0
        msg = resp.json().get("message", {})
        turns.append({"turn": turn, "llm_latency_s": round(llm_latency, 3),
                      "content": msg.get("content", ""), "tool_calls": msg.get("tool_calls", [])})
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": msg.get("tool_calls")})

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break  # model stopped calling tools -- either done or stuck; trace shows which

        rendered_this_turn = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            fn = getattr(server, name, None)
            if fn is None:
                result = {"ok": False, "error": f"unknown tool {name}"}
            else:
                result = fn(**args)
            messages.append({"role": "tool", "content": json.dumps(result)})
            if name == "render":
                rendered_this_turn = True
        if rendered_this_turn:
            break

    return server, turns


if __name__ == "__main__":
    import sys
    from faster_whisper import WhisperModel

    task_path = Path(sys.argv[1])
    assets_dir = Path(sys.argv[2])
    project_dir = Path(sys.argv[3])
    out_dir = Path(sys.argv[4])
    model = sys.argv[5] if len(sys.argv) > 5 else "llama3.1:8b"

    task = json.loads(task_path.read_text())
    whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    server, turns = run_llm_agent(task, assets_dir, project_dir, model=model, whisper_model=whisper)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj = server.trajectory()
    traj["agent_subtitles"] = server.subtitles
    traj["llm_turns"] = turns
    traj["model"] = model
    (out_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))
    print(f"turns={len(turns)} rendered={traj['rendered']} n_calls={traj['n_calls']} "
          f"failed={traj['n_failed_calls']} total_latency={traj['total_latency_s']:.1f}s")
