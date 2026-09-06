"""
LLM-as-judge for the multimodal_judge-scored subtypes (generation/
scene_assembly, editing/transition). Uses qwen3.8:27b's vision capability
via Ollama -- confirmed locally available, tools+vision+thinking.

Samples a few frames from the agent's rendered output and from the
reference (a real video for generation tasks, or 1-2 real reference
examples for transition), asks for a structured verdict. This is
deliberately a judgment call with a rationale, not a hidden scalar --
Self-Harness/VF-Eval both show a bare score is easy to hallucinate and
hard to act on; the rationale is what feeds weakness mining.
"""
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import requests

OLLAMA = "http://localhost:11434/api/chat"


def _sample_frames_b64(video_path: Path, n: int = 3) -> list[str]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        dur = float(probe.stdout.strip())
    except ValueError:
        dur = 1.0
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = dur * (i + 0.5) / n
            out = Path(tmp) / f"f{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(video_path),
                 "-frames:v", "1", "-vf", "scale=480:-1", str(out)],
                check=False,
            )
            if out.exists():
                frames.append(base64.b64encode(out.read_bytes()).decode())
    return frames


def judge_against_reference(task: dict, rendered_path: Path, reference_path: Path,
                             model: str = "qwen3.8:27b") -> dict:
    out_frames = _sample_frames_b64(rendered_path)
    ref_frames = _sample_frames_b64(reference_path)

    if task["subtype"] == "transition":
        # Confirmed bug from a real run: AutoTransition's reference clips are
        # unrelated stock footage (different people/scenes than the task's
        # actual assets), so asking the judge for content "fidelity" against
        # it just penalizes the agent for not reproducing footage it was
        # never given. The reference is only valid as a STYLE example of
        # what a well-executed transition looks like -- score execution,
        # not content match.
        prompt = (
            f"Task given to a video editing agent: \"{task['instruction']}\"\n\n"
            "The first set of images are sampled frames from a REFERENCE example -- showing "
            "what a well-executed transition effect looks like in general. Its subject matter "
            "is UNRELATED to the agent's task; judge it only as a style example of transition "
            "craft, never penalize the agent for showing different content. The second set are "
            "frames from the AGENT's own rendered output.\n\n"
            "Judge the agent's output on three axes, each 1-4 (1=poor, 4=excellent):\n"
            "- transition_smoothness: does the join between the two clips read as an intentional, "
            "well-executed transition (vs. an abrupt/broken cut)\n"
            "- technique_appropriateness: does the transition type suit this footage\n"
            "- technical_quality: any obvious artifacts, black frames, glitches\n\n"
            "Respond ONLY as JSON: {\"transition_smoothness\": int, \"technique_appropriateness\": int, "
            "\"technical_quality\": int, \"reasoning\": \"one or two sentences on what specifically "
            "was wrong or right\"}"
        )
    else:
        prompt = (
            f"Task given to a video editing agent: \"{task['instruction']}\"\n\n"
            "The first set of images are sampled frames from a REFERENCE video -- what these "
            "assets were actually, professionally cut into. The second set are sampled frames "
            "from the AGENT's own rendered output for the same assets.\n\n"
            "Judge the agent's output against the reference on three axes, each 1-4 "
            "(1=poor, 4=matches reference quality):\n"
            "- coherence: does the agent's sequence read as a plausible, watchable edit\n"
            "- fidelity: how close is it in spirit/content to what the reference actually shows\n"
            "- technical_quality: any obvious artifacts, black frames, jarring cuts\n\n"
            "Respond ONLY as JSON: {\"coherence\": int, \"fidelity\": int, \"technical_quality\": int, "
            "\"reasoning\": \"one or two sentences on what specifically was wrong or right\"}"
        )

    resp = requests.post(OLLAMA, json={
        "model": model,
        "messages": [
            {"role": "user", "content": "REFERENCE frames:", "images": ref_frames},
            {"role": "user", "content": prompt, "images": out_frames},
        ],
        "stream": False,
    }, timeout=180)
    content = resp.json().get("message", {}).get("content", "")
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        verdict = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        verdict = {"parse_error": True, "raw": content}
    verdict["model"] = model
    return verdict


if __name__ == "__main__":
    import sys

    task = json.loads(Path(sys.argv[1]).read_text())
    rendered = Path(sys.argv[2])
    reference = Path(sys.argv[3])
    print(json.dumps(judge_against_reference(task, rendered, reference), indent=2))
