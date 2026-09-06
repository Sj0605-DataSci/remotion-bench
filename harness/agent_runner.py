"""
Generalized LLM-driven agent runner: loads ANY task from dataset_v1's
taxonomy, exposes the FULL unified tool set (not a per-task-filtered
subset -- picking the right tools out of the whole toolbox is itself
part of what's being measured), drives it via a local Ollama model with
real tool-calling, and produces the standard run bundle.

Model choice is explicit and logged: gemma4:e4b, qwen3.8:27b, and
llama3.1:8b all reliably emit tool_calls against this Ollama install
(verified); nemotron-mini does not (verified, its template supports tool
tags but the 4.2B Q4 model doesn't reliably produce valid ones) -- not
used as an executing agent. gemma4:e4b is the default: native tool
calling + vision, smaller/faster than qwen3.8:27b, and exposes a
"thinking" trace useful for weakness mining.
"""
import base64
import inspect
import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from tool_server import ToolServer, ToolCallLog

OLLAMA = "http://localhost:11434/api/chat"


def _extract_review_frame_b64(video_path: Path, at_fraction: float = 0.4) -> str | None:
    """Real self-verification frame, not a stand-in -- the agent sees its
    OWN actual rendered output, the same way EditDuet's Critic agent reviews
    the Editor's timeline before accepting it."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        dur = float(probe.stdout.strip())
    except ValueError:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "review.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(dur * at_fraction), "-i", str(video_path),
             "-frames:v", "1", "-vf", "scale=480:-1", str(out)],
            check=False,
        )
        return base64.b64encode(out.read_bytes()).decode() if out.exists() else None

ALL_TOOLS = [
    {"type": "function", "function": {
        "name": "list_assets", "description": "Get durations for EVERY available asset in one call. Prefer this over probing files one at a time, especially when there are many assets.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "look_at_frame", "description": "See an actual video frame at a given timestamp, so you can judge content (what's in the shot, whether framing/composition looks right, which of several clips fits) instead of guessing from filenames alone. Use this whenever the task requires a judgment call about what's actually shown.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"}, "at_seconds": {"type": "number"}}, "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "probe", "description": "Get the duration in seconds of a video asset. Call this first for any asset you'll place on the timeline.",
        "parameters": {"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "transcribe", "description": "Run speech-to-text on a video asset, returns timed segments. Only needed if the task involves captions/subtitles.",
        "parameters": {"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "add_to_timeline", "description": "Place an asset onto the edit timeline, in order.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"}, "start_in_source": {"type": "number"}, "duration": {"type": "number"},
            "track": {"type": "string", "description": "e.g. 'A-roll', 'B-roll'"}},
            "required": ["file", "start_in_source", "duration"]}}},
    {"type": "function", "function": {
        "name": "remove_from_timeline", "description": "Remove a clip from the timeline by its index.",
        "parameters": {"type": "object", "properties": {"index": {"type": "integer"}}, "required": ["index"]}}},
    {"type": "function", "function": {
        "name": "set_transition", "description": "Apply a transition between timeline clip at `index` and the next one. type must be one of fade|wipe|slide|clockWipe|flip|none.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"}, "type": {"type": "string"}, "duration": {"type": "number"}},
            "required": ["index", "type"]}}},
    {"type": "function", "function": {
        "name": "add_captions", "description": "Add subtitle/caption cues. captions is a list of {text, startMs, endMs} objects (the real @remotion/captions Caption shape). Call after add_to_timeline.",
        "parameters": {"type": "object", "properties": {
            "captions": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"}, "startMs": {"type": "number"}, "endMs": {"type": "number"}}}}},
            "required": ["captions"]}}},
    {"type": "function", "function": {
        "name": "add_audio_track", "description": "Add an audio asset (e.g. a music bed) as a background track.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"}, "volume": {"type": "number", "description": "0-1, default 1.0"}},
            "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "set_audio_ducking", "description": "Lower an audio track's volume during specific time windows (e.g. duck music under dialogue). Call after add_audio_track for that file.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string"}, "duck_to_volume": {"type": "number"},
            "during_windows": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}},
            "required": ["file", "duck_to_volume", "during_windows"]}}},
    {"type": "function", "function": {
        "name": "add_text_overlay", "description": "Add a title/lower-third text overlay. position must be one of center|top|lower_third|bottom.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"}, "position": {"type": "string"}},
            "required": ["text", "start", "end", "position"]}}},
    {"type": "function", "function": {
        "name": "set_reframe_crop", "description": "Crop the composition to a different aspect ratio (e.g. 9:16 for mobile), keeping the subject in frame via a focus point.",
        "parameters": {"type": "object", "properties": {
            "target_aspect": {"type": "string"}, "focus_x": {"type": "number"}, "focus_y": {"type": "number"}},
            "required": ["target_aspect"]}}},
    {"type": "function", "function": {
        "name": "render", "description": "Render the final video. Call this LAST, exactly once, after everything else is set up.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM_PROMPT = (
    "You are a video editing agent. You accomplish tasks by calling the "
    "provided tools. You have real vision -- use look_at_frame whenever the "
    "task requires judging what's actually in a shot (choosing between "
    "candidate clips, checking framing/composition); don't guess from "
    "filenames or durations alone when the task calls for a content "
    "judgment. If there are many assets, call list_assets once instead of "
    "probing each one individually. You have a full toolbox (cuts, "
    "transitions, captions, audio, text overlays, reframe) but most tasks "
    "only need a few of these -- use only what the instruction actually "
    "asks for. Work step by step, then render exactly once as your final "
    "action -- don't stop before rendering once you have enough information "
    "to make a decision."
)


def run_agent(task: dict, assets_dir: Path, project_dir: Path, model: str,
              kb_lessons: list[str] | None = None, max_turns: int = 20, whisper_model=None,
              self_critique: bool = True, max_critique_rounds: int = 2):
    server = ToolServer(assets_dir=assets_dir, project_dir=project_dir, whisper_model=whisper_model)

    system = SYSTEM_PROMPT
    if kb_lessons:
        system += "\n\nLessons from prior runs, apply them where relevant:\n" + "\n".join(f"- {l}" for l in kb_lessons)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{task['instruction']}\nAvailable asset files: {', '.join(task['assets'])}"},
    ]

    turns = []
    critique_rounds = 0
    awaiting_critique_response = False
    for turn in range(max_turns):
        t0 = time.time()
        resp = requests.post(OLLAMA, json={"model": model, "messages": messages, "tools": ALL_TOOLS, "stream": False}, timeout=180)
        llm_latency = time.time() - t0
        msg = resp.json().get("message", {})
        turns.append({"turn": turn, "llm_latency_s": round(llm_latency, 3),
                      "content": msg.get("content", ""), "tool_calls": msg.get("tool_calls", [])})
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": msg.get("tool_calls")})

        tool_calls = msg.get("tool_calls") or []
        was_awaiting_critique, awaiting_critique_response = awaiting_critique_response, False
        if not tool_calls:
            # A "DONE" after a self-critique round is a legitimate stop, not
            # the narrated-instead-of-called failure mode below.
            if was_awaiting_critique and "done" in msg.get("content", "").lower():
                break
            # Observed failure mode (mined from a real run): after a tool
            # error, the model sometimes narrates a "corrected call" as
            # plain text instead of actually emitting tool_calls. Nudge it
            # back toward a real call a couple of times before giving up --
            # this is a harness/loop fix, distinct from a KB/prompt lesson,
            # for exactly the case where a lesson in the system prompt
            # can't help because the failure is in turn-loop control.
            already_nudged = sum(1 for t in turns if t.get("nudged"))
            if msg.get("content") and already_nudged < 2:
                messages.append({"role": "user", "content": (
                    "That was plain text, not a tool call. If you intended to call a "
                    "tool, respond with an actual tool call now, not a description of one."
                )})
                turns[-1]["nudged"] = True
                continue
            break

        rendered_this_turn = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            fn = getattr(server, name, None)
            if fn is None:
                result = {"ok": False, "error": f"unknown tool {name}"}
            else:
                # LLM tool calls routinely hallucinate extra/unexpected kwargs --
                # drop them (and note it) rather than crashing the whole run on
                # what is itself a real tool-call-correctness signal.
                accepted = set(inspect.signature(fn).parameters)
                extra = set(args) - accepted
                clean_args = {k: v for k, v in args.items() if k in accepted}
                try:
                    result = fn(**clean_args)
                except TypeError as e:
                    # missing/mismatched required args -- a real tool-call-
                    # correctness failure, reported back like any other tool
                    # error (and logged into the trajectory the same way,
                    # since it never reached ToolServer._call's own try/except)
                    result = {"ok": False, "error": f"invalid arguments for {name}: {e}"}
                    server.log.append(ToolCallLog(name, args, False, result["error"], 0.0, time.time()))
                if extra:
                    result = {**result, "extra_args_dropped": sorted(extra)}

            if name == "look_at_frame" and result.get("ok") and result["result"].get("frame_path"):
                frame_path = Path(result["result"]["frame_path"])
                b64 = base64.b64encode(frame_path.read_bytes()).decode()
                messages.append({"role": "tool", "content": json.dumps({"ok": True, "note": "frame attached below"})})
                messages.append({"role": "user", "content": f"Frame from {args.get('file')} at {args.get('at_seconds', 0)}s:", "images": [b64]})
            else:
                messages.append({"role": "tool", "content": json.dumps(result)})
            if name == "render":
                rendered_this_turn = True

        if rendered_this_turn:
            render_ok = server.log[-1].ok if server.log else False
            if not render_ok:
                # Confirmed bug: any render() call, success or failure, used
                # to end the loop here -- so a failed render (e.g. "called
                # with an empty timeline") never gave the agent a chance to
                # see its own error and fix it, even though the error was
                # already sitting in its context as a tool result. Let the
                # loop continue normally instead of breaking on failure.
                continue
            if not (self_critique and critique_rounds < max_critique_rounds):
                break
            # Real self-verification, not a stand-in: show the agent its OWN
            # rendered output (EditDuet's Editor/Critic pattern) and let it
            # decide whether to keep going instead of always stopping at the
            # first render() call -- this is what would have caught the
            # reframe miss (rendering with an empty timeline never happens
            # here since render() itself would have failed and render_ok
            # would be False, but a *wrong* but technically successful
            # render -- e.g. forgetting a step that doesn't hard-fail --
            # is exactly what this catches instead).
            frame_b64 = _extract_review_frame_b64(project_dir / "out.mp4")
            if frame_b64 is None:
                break
            critique_rounds += 1
            messages.append({"role": "user", "content": (
                f"Here is a frame from your actual rendered output. Original task: \"{task['instruction']}\"\n"
                "Does this genuinely satisfy the task? If yes, respond with exactly the word DONE and nothing else. "
                "If something is missing or wrong, call more tools to fix it, then render again."
            ), "images": [frame_b64]})
            awaiting_critique_response = True
            continue

    return server, turns


if __name__ == "__main__":
    import sys
    from faster_whisper import WhisperModel

    task_path, assets_dir, project_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    model = sys.argv[5] if len(sys.argv) > 5 else "gemma4:e4b"

    task = json.loads(task_path.read_text())
    whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    server, turns = run_agent(task, assets_dir, project_dir, model=model, whisper_model=whisper)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj = server.trajectory()
    traj["llm_turns"] = turns
    traj["model"] = model
    (out_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))
    print(f"model={model} turns={len(turns)} rendered={traj['rendered']} n_calls={traj['n_calls']} "
          f"failed={traj['n_failed_calls']} total_latency={traj['total_latency_s']:.1f}s")
