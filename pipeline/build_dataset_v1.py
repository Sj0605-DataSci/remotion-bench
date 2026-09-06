"""
Round-1 dataset builder: produces a unified dataset.jsonl spanning both
outcome categories, using only data sources already validated in this
project (no new external pulls needed for round 1):

  generation/scene_assembly  <- MovieCuts real shots + real reference video
  editing/cut_assembly        <- MovieCuts (reuses pipeline/build_task.py)
  editing/shot_ordering       <- MovieCuts (reuses pipeline/build_task.py)
  editing/caption             <- ASR ground truth (reuses task_schema.py)
  editing/audio_mix           <- synthesized locally, no external dataset

editing/transition, text_overlay, reframe are named in task_taxonomy.py
but not built this round -- they need AutoTransition (not pulled yet) or
are still just self-authored stubs. Listed as explicit gaps, not silently
skipped.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from task_schema import build_subtitle_task  # noqa: E402
from build_more_subtypes import build_text_overlay_task, build_reframe_task  # noqa: E402

DISTRACTOR_DIR = Path(
    "/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/distractors"
)

PILOT = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/pilot")
TASK_SUBTITLE = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/task_subtitle")
DATASET_DIR = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/dataset_v1")


def task_generation_scene_assembly():
    """Real shots + real reference segment from the same MovieCuts pilot
    movie. Scored by multimodal judge (Remotion output vs this reference),
    not structurally -- there's no single 'correct' assembly, only a real
    one that actually happened."""
    out_dir = DATASET_DIR / "generation_scene_assembly"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "reference.mp4"
    # first 8 ground-truth shots span roughly this window -- use it as the
    # reference "what these assets were actually cut into"
    gt = json.loads((PILOT / "ground_truth.json").read_text())
    window_end = gt["shots_in_order"][7]["end"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(PILOT / "raw/source.mp4"),
         "-t", str(window_end), "-c:v", "libx264", "-c:a", "aac", str(ref_path)],
        check=True,
    )
    assets = [s["file"] for s in gt["shots_in_order"][:8]]
    for f in assets:
        src = PILOT / "shots" / f
        dst = out_dir / f
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    return {
        "task_id": "gen_scene_assembly_001",
        "category": "generation",
        "subtype": "scene_assembly",
        "instruction": (
            "You are given 8 raw video shots from the same scene, in no particular "
            "order. Assemble them via Remotion into a single coherent edited video "
            "that plays naturally, choosing order and cut points."
        ),
        "assets": assets,
        "expected_tool_sequence": ["probe", "add_to_timeline", "render"],
        "scoring_method": "multimodal_judge",
        "ground_truth": {"note": "no single correct order -- judged against reference_video for plausibility/coherence, not exact match"},
        "reference_video": "reference.mp4",
    }


def task_cut_assembly_and_ordering():
    """Reuses the already-built MovieCuts pipeline output, wrapped into
    the unified schema as two editing-category tasks."""
    gt = json.loads((PILOT / "ground_truth.json").read_text())
    task_json = json.loads((PILOT / "task.json").read_text())
    shot_ordering = {
        "task_id": "edit_shot_ordering_001",
        "category": "editing",
        "subtype": "shot_ordering",
        "instruction": task_json["instruction"],
        "assets": [s["file"] for s in task_json["shot_pool"]],
        "expected_tool_sequence": ["probe", "add_to_timeline", "render"],
        "scoring_method": "structural",
        "ground_truth": {
            "movie_id": gt["movie_id"], "video_id": gt["video_id"],
            "shots_in_order": [{"file": s["file"], "start": s["start"], "end": s["end"],
                                 "cut_type_after": s.get("cut_type_after", [])} for s in gt["shots_in_order"]],
        },
        "reference_video": None,
    }
    resolved_path = DISTRACTOR_DIR / "resolved_distractors.json"
    resolved = json.loads(resolved_path.read_text()) if resolved_path.exists() else []
    real_distractor_files = [r["file"] for r in resolved if r["status"] == "ok"][:4]
    for f in real_distractor_files:
        src = DISTRACTOR_DIR / f
        dst = Path(str(PILOT / "shots" / f))  # colocate with other playable assets for this task
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    cut_assembly = {
        "task_id": "edit_cut_assembly_001",
        "category": "editing",
        "subtype": "cut_assembly",
        "instruction": (
            "Given this A-roll shot and 4 candidate next-shots (some real, some "
            "distractors from other footage), choose the correct next shot and "
            "place it immediately after with an appropriate cut."
        ),
        "assets": [gt["shots_in_order"][0]["file"], gt["shots_in_order"][1]["file"]] + real_distractor_files,
        "expected_tool_sequence": ["probe", "add_to_timeline", "render"],
        "scoring_method": "structural",
        "ground_truth": {
            "correct_next_shot": gt["shots_in_order"][1]["file"],
            "cut_type": gt["shots_in_order"][0].get("cut_type_after", []),
            "distractors_resolved": len(real_distractor_files),
        },
        "reference_video": None,
    }
    return shot_ordering, cut_assembly


def task_caption():
    task = json.loads((TASK_SUBTITLE / "task.json").read_text())
    task["task_id"] = "edit_caption_001"
    task["category"] = "editing"
    task["subtype"] = "caption"
    task["scoring_method"] = "structural"
    task["reference_video"] = None
    return task


def task_audio_mix():
    """Synthesized locally: real dialogue clip + a synthetic music bed
    (sine-wave, no licensing concerns). Ground truth is a loudness rule,
    not a reference file -- checkable with ffmpeg astats, no ASR/judge
    needed."""
    out_dir = DATASET_DIR / "editing_audio_mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogue_src = TASK_SUBTITLE / "assets/dialogue_clip.mp4"
    dialogue_dst = out_dir / "dialogue_clip.mp4"
    if not dialogue_dst.exists():
        dialogue_dst.write_bytes(dialogue_src.read_bytes())
    music_path = out_dir / "music_bed.wav"
    if not music_path.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=220:duration=25", str(music_path)],
            check=True,
        )
    # actual speech intervals for ducking ground truth come from the caption
    # task's own ASR ground truth
    ct = task_caption()
    speech_windows = [(s["start"], s["end"]) for s in ct["ground_truth"]["segments"]]

    return {
        "task_id": "edit_audio_mix_001",
        "category": "editing",
        "subtype": "audio_mix",
        "instruction": (
            "Mix music_bed.wav under dialogue_clip.mp4 as background music. "
            "Duck the music noticeably during spoken dialogue so it doesn't "
            "compete with speech, and let it play at normal volume during silence."
        ),
        "assets": ["dialogue_clip.mp4", "music_bed.wav"],
        "expected_tool_sequence": ["probe", "add_to_timeline", "add_audio_track", "set_audio_ducking", "render"],
        "scoring_method": "structural",
        "ground_truth": {"speech_windows": speech_windows, "duck_db_min": 6.0},
        "reference_video": None,
    }


def task_transition():
    """AutoTransition (HF: yaojie-shen/AutoTransition, cc-by-4.0) ships as a
    9x4GB split tar of rendered template outputs, no separate before/after
    clips + labels reachable without extracting the whole archive. Pulled
    just 2 sample files cheaply (~2MB) by reading only the start of the
    tar stream -- not the full 39GB. These are reference OUTPUTS with a
    transition already applied, so this task is scored qualitatively
    (multimodal judge: does the agent's transition read as smoothly/
    appropriately as these real examples), not against an exact label --
    a real, honest scope difference from the other structural subtypes."""
    out_dir = DATASET_DIR / "editing_transition"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads((PILOT / "ground_truth.json").read_text())
    shot_a, shot_b = gt["shots_in_order"][2]["file"], gt["shots_in_order"][3]["file"]
    for f in (shot_a, shot_b):
        dst = out_dir / f
        if not dst.exists():
            dst.write_bytes((PILOT / "shots" / f).read_bytes())

    return {
        "task_id": "edit_transition_001",
        "category": "editing",
        "subtype": "transition",
        "instruction": (
            f"Join {shot_a} and {shot_b} with a transition effect appropriate "
            "to the content -- a hard cut, dissolve, or wipe, whichever reads "
            "as most natural for this footage."
        ),
        "assets": [shot_a, shot_b],
        "expected_tool_sequence": ["probe", "add_to_timeline", "set_transition", "render"],
        "scoring_method": "multimodal_judge",
        "ground_truth": {
            "reference_examples": ["reference_transition_example.mp4", "reference_transition_example_2.mp4"],
            "source": "AutoTransition (yaojie-shen/AutoTransition, cc-by-4.0) -- 2 real rendered "
                      "template outputs used as quality reference, not exact-match ground truth",
        },
        "reference_video": None,
    }


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    gen_task = task_generation_scene_assembly()
    shot_ordering, cut_assembly = task_cut_assembly_and_ordering()
    caption = task_caption()
    audio_mix = task_audio_mix()
    text_overlay = build_text_overlay_task()
    reframe = build_reframe_task(TASK_SUBTITLE / "assets/dialogue_clip.mp4")
    transition = task_transition()

    dataset = [gen_task, shot_ordering, cut_assembly, caption, audio_mix, text_overlay, reframe, transition]
    out_path = DATASET_DIR / "dataset.jsonl"
    with open(out_path, "w") as f:
        for t in dataset:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(dataset)} tasks to {out_path}")
    for t in dataset:
        print(f"  [{t['category']}/{t['subtype']}] {t['task_id']} -- scoring={t['scoring_method']}")
    print("\nTaxonomy complete: 8/8 subtypes built.")


if __name__ == "__main__":
    main()
