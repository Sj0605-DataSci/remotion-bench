"""
Task taxonomy for the eval dataset -- two outcome categories, each with
subtypes, each with a real data source and a real way to check the result.
This replaces the single-task-type (subtitles-only) scope from earlier.

OUTCOME CATEGORIES
-------------------
"generation": given raw assets + an instruction describing the scene, the
  agent assembles a video from scratch via Remotion. There is a REAL
  reference video that was supposed to result from those assets (a real
  professional edit) -- scored by a multimodal LLM comparing the agent's
  Remotion output against that reference, not by structural diffing.
  This is the "first outcome" -- can the agent get anywhere close to
  what those assets were actually meant to become.

"editing": given an already-assembled video (or partially-assembled) plus
  an instruction naming ONE discrete operation, the agent performs just
  that operation via tool calls. Scored structurally against a checkable
  ground truth specific to the subtype (no LLM judge needed for most of
  these -- that's the point, they're cheap and hard to game).

SUBTYPES (editing category)
----------------------------
  cut_assembly    - place a shot correctly relative to its neighbors
                    (MovieCuts ground truth, tIoU + cut-type match)
  shot_ordering   - reorder a shuffled shot pool into the real sequence
                    (MovieCuts chained shots, LCS order accuracy)
  caption         - add synced subtitles (ASR ground truth, coverage/dead-air)
  transition      - apply the right transition type between two shots
                    (AutoTransition ground truth -- not yet pulled, planned)
  audio_mix       - duck/level audio tracks correctly (loudness ground truth,
                    synthesized -- no external dataset needed)
  text_overlay    - place a title/lower-third at the right time/position
                    (self-authored, structurally checkable)
  reframe         - adapt aspect ratio while keeping the subject in frame
                    (self-authored, structurally checkable via face/motion bbox)

Every task instance, regardless of category/subtype, carries the same
envelope so scoring/task_scorer.py's four axes (tool-call correctness,
latency, video quality, editing capacity) apply uniformly:

{
  "task_id": str,
  "category": "generation" | "editing",
  "subtype": str,
  "instruction": str,
  "assets": [str],                    # filenames under this task's assets/
  "expected_tool_sequence": [str],
  "scoring_method": "multimodal_judge" | "structural",
  "ground_truth": {...},              # subtype-specific
  "reference_video": str | None,      # only for "generation" tasks
}
"""

CATEGORIES = ("generation", "editing")

SUBTYPES = {
    "generation": ["scene_assembly"],
    "editing": [
        "cut_assembly",
        "shot_ordering",
        "caption",
        "transition",       # data source planned, not yet pulled
        "audio_mix",
        "text_overlay",     # not yet built this round
        "reframe",           # not yet built this round
    ],
}

SCORING_METHOD = {
    "scene_assembly": "multimodal_judge",
    "cut_assembly": "structural",
    "shot_ordering": "structural",
    "caption": "structural",
    "transition": "structural",
    "audio_mix": "structural",
    "text_overlay": "structural",
    "reframe": "structural",
}
