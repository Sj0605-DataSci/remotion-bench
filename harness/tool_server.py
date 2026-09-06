"""
Unified agent-facing tool API spanning the full 8-subtype taxonomy
(pipeline/task_taxonomy.py) -- cut/trim, sequencing, transitions,
captions, audio, text overlay, and reframe, not a subtitle-only script
with everything else bolted on. Every call is logged with wall-clock
latency and validated against real preconditions, independent of whether
the final render is any good (scored separately, after render).

All tools are always exposed to the agent regardless of task type -- a
real agent has to pick the RIGHT ones, and which it calls (and skips) is
itself the tool-call-correctness signal, not something we should hide by
pre-filtering the toolset per task.
"""
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolCallLog:
    name: str
    args: dict
    ok: bool
    error: str | None
    latency_s: float
    t_start: float


class ToolServer:
    def __init__(self, assets_dir: Path, project_dir: Path, whisper_model=None):
        self.assets_dir = Path(assets_dir)
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_model = whisper_model

        self.timeline: list[dict] = []       # [{file, start_in_source, duration, track}]
        self.captions: list[dict] = []       # real @remotion/captions Caption shape
        self.transcripts: dict[str, list[dict]] = {}
        self.transitions: dict[int, dict] = {}   # timeline_index -> {type, duration}
        self.audio_tracks: list[dict] = []   # [{file, volume, duck_below}]
        self.text_overlays: list[dict] = []  # [{text, start, end, position}]
        self.reframe: dict | None = None     # {aspect, focus_x, focus_y}

        self.rendered = False
        self.log: list[ToolCallLog] = []

    def _call(self, name: str, args: dict, fn):
        t0 = time.time()
        try:
            result = fn()
            ok, err = True, None
        except Exception as e:  # noqa: BLE001 - every failure must be captured, not raised
            result, ok, err = None, False, str(e)
        dt = time.time() - t0
        self.log.append(ToolCallLog(name, args, ok, err, dt, t0))
        return {"ok": ok, "error": err, "result": result}

    # ---- Inspection ----------------------------------------------------

    def list_assets(self):
        """Bulk metadata for every asset in this task, in one call -- large
        pools (dozens of shots) shouldn't need one probe() per file just to
        find out durations; that alone burns the turn budget before the
        agent gets to do anything (confirmed in testing on a 73-shot task)."""
        def _run():
            out = {}
            for f in sorted(p.name for p in self.assets_dir.iterdir() if p.suffix in (".mp4", ".mov", ".webm", ".wav", ".mp3")):
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(self.assets_dir / f)],
                    capture_output=True, text=True,
                )
                try:
                    out[f] = round(float(probe.stdout.strip()), 3)
                except ValueError:
                    out[f] = None
            return out
        return self._call("list_assets", {}, _run)

    def look_at_frame(self, file: str, at_seconds: float = 0.0):
        """Real vision tool: extracts a frame and returns its path so the
        harness can inject it as an image into the next model turn. This is
        the fix for a confirmed gap -- gemma4:e4b has real vision capability
        but a tool-calling loop that only ever returns text metadata makes
        any content-judgment task (which shot fits, is the framing right)
        structurally unanswerable no matter how capable the model is."""
        def _run():
            path = self.assets_dir / file
            if not path.exists():
                raise FileNotFoundError(file)
            out_path = self.project_dir / f"_frame_{Path(file).stem}_{at_seconds}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at_seconds), "-i", str(path),
                 "-frames:v", "1", "-vf", "scale=480:-1", str(out_path)],
                check=True,
            )
            return {"frame_path": str(out_path)}
        return self._call("look_at_frame", {"file": file, "at_seconds": at_seconds}, _run)

    def probe(self, file: str):
        def _run():
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(self.assets_dir / file)],
                capture_output=True, text=True, check=True,
            )
            return {"duration": float(out.stdout.strip())}
        return self._call("probe", {"file": file}, _run)

    def transcribe(self, file: str):
        def _run():
            if self.whisper_model is None:
                raise RuntimeError("no whisper model loaded")
            segments, _ = self.whisper_model.transcribe(str(self.assets_dir / file), vad_filter=True)
            segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
            self.transcripts[file] = segs
            return {"n_segments": len(segs), "segments": segs}
        return self._call("transcribe", {"file": file}, _run)

    # ---- Sequencing / cuts ----------------------------------------------

    def add_to_timeline(self, file: str, start_in_source: float, duration: float, track: str = "A-roll"):
        def _run():
            if not (self.assets_dir / file).exists():
                raise FileNotFoundError(file)
            self.timeline.append({"file": file, "start_in_source": float(start_in_source), "duration": float(duration), "track": track})
            return {"timeline_length": len(self.timeline)}
        return self._call("add_to_timeline", {"file": file, "start_in_source": start_in_source, "duration": duration, "track": track}, _run)

    def remove_from_timeline(self, index: int):
        def _run():
            if index < 0 or index >= len(self.timeline):
                raise IndexError(f"no timeline entry at {index}")
            return self.timeline.pop(index)
        return self._call("remove_from_timeline", {"index": index}, _run)

    def set_transition(self, index: int, type: str, duration: float = 0.5):
        """@remotion/transitions -- fade | wipe | slide | clockWipe | flip.
        Applies between timeline[index] and timeline[index+1]."""
        def _run():
            valid = {"fade", "wipe", "slide", "clockWipe", "flip", "none"}
            if type not in valid:
                raise ValueError(f"unknown transition type {type!r}, expected one of {valid}")
            if index < 0 or index >= len(self.timeline) - 1:
                raise IndexError(f"no clip pair at boundary {index}")
            self.transitions[index] = {"type": type, "duration": float(duration)}
            return {"n_transitions": len(self.transitions)}
        return self._call("set_transition", {"index": index, "type": type, "duration": duration}, _run)

    # ---- Captions --------------------------------------------------------

    def add_captions(self, captions: list[dict]):
        """captions: list of the real @remotion/captions Caption shape --
        {text, startMs, endMs}. Converts internally to seconds for scoring."""
        def _run():
            if not self.timeline:
                raise RuntimeError("add_captions called before any add_to_timeline -- nothing on screen yet")
            for c in captions:
                start_ms, end_ms = float(c["startMs"]), float(c["endMs"])
                if end_ms <= start_ms:
                    raise ValueError(f"caption end ({end_ms}) must be after start ({start_ms})")
                self.captions.append({"text": c["text"], "start": start_ms / 1000, "end": end_ms / 1000})
            return {"n_captions": len(self.captions)}
        return self._call("add_captions", {"n": len(captions)}, _run)

    # ---- Audio -------------------------------------------------------------

    def add_audio_track(self, file: str, volume: float = 1.0):
        def _run():
            if not (self.assets_dir / file).exists():
                raise FileNotFoundError(file)
            self.audio_tracks.append({"file": file, "volume": float(volume), "duck_below": None})
            return {"n_audio_tracks": len(self.audio_tracks)}
        return self._call("add_audio_track", {"file": file, "volume": volume}, _run)

    def set_audio_ducking(self, file: str, duck_to_volume: float, during_windows: list[list[float]]):
        def _run():
            track = next((t for t in self.audio_tracks if t["file"] == file), None)
            if track is None:
                raise RuntimeError(f"add_audio_track must be called for {file} before ducking it")
            bad = [w for w in during_windows if not (isinstance(w, (list, tuple)) and len(w) == 2)]
            if bad:
                raise ValueError(
                    f"during_windows must be a list of [start, end] pairs, e.g. [[0.5, 2.5], [5.0, 7.0]] -- "
                    f"got malformed entries: {bad[:2]}"
                )
            windows = [[float(a), float(b)] for a, b in during_windows]
            track["duck_below"] = {"volume": float(duck_to_volume), "windows": windows}
            return {"ducked": True}
        return self._call("set_audio_ducking", {"file": file, "duck_to_volume": duck_to_volume, "n_windows": len(during_windows)}, _run)

    # ---- Text / titles --------------------------------------------------

    def add_text_overlay(self, text: str, start, end, position: str = "center"):
        def _run():
            start_f, end_f = float(start), float(end)
            valid_positions = {"center", "top", "lower_third", "bottom"}
            if position not in valid_positions:
                raise ValueError(f"unknown position {position!r}, expected one of {valid_positions}")
            if end_f <= start_f:
                raise ValueError(f"end ({end_f}) must be after start ({start_f})")
            self.text_overlays.append({"text": text, "start": start_f, "end": end_f, "position": position})
            return {"n_overlays": len(self.text_overlays)}
        return self._call("add_text_overlay", {"text": text[:40], "start": start, "end": end, "position": position}, _run)

    # ---- Reframe ----------------------------------------------------------

    def set_reframe_crop(self, target_aspect: str, focus_x: float = 0.5, focus_y: float = 0.5):
        def _run():
            if target_aspect not in {"9:16", "1:1", "4:5", "16:9"}:
                raise ValueError(f"unsupported aspect {target_aspect!r}")
            if not (0 <= focus_x <= 1 and 0 <= focus_y <= 1):
                raise ValueError("focus_x/focus_y must be normalized 0-1")
            self.reframe = {"aspect": target_aspect, "focus_x": focus_x, "focus_y": focus_y}
            return {"reframe_set": True}
        return self._call("set_reframe_crop", {"target_aspect": target_aspect, "focus_x": focus_x, "focus_y": focus_y}, _run)

    # ---- Render ----------------------------------------------------------

    def render(self, fps: int = 24, width: int = 1280, height: int = 720):
        def _run():
            if not self.timeline:
                raise RuntimeError("render called with an empty timeline")
            self._write_remotion_inputs(fps, width, height)
            out_path = self.project_dir / "out.mp4"
            proc = subprocess.run(
                ["npx", "remotion", "render", "index.jsx", "Reassembly", str(out_path), "--concurrency=1"],
                cwd=self.project_dir, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"remotion render failed: {proc.stderr[-800:]}")
            self.rendered = True
            return {"output": str(out_path)}
        return self._call("render", {"fps": fps, "width": width, "height": height}, _run)

    # ---- Internal ------------------------------------------------------

    def _write_remotion_inputs(self, fps: int, width: int, height: int):
        if self.reframe and self.reframe["aspect"] == "9:16":
            width, height = 720, 1280
        clips = []
        for i, t in enumerate(self.timeline):
            clips.append({
                "file": t["file"], "durationInFrames": round(t["duration"] * fps),
                "startInSource": t["start_in_source"],
                "transitionOut": self.transitions.get(i),
            })
        (self.project_dir / "timeline.json").write_text(json.dumps(clips, indent=2))

        subs = [{"text": c["text"], "startFrame": round(c["start"] * fps), "endFrame": round(c["end"] * fps)}
                for c in self.captions]
        (self.project_dir / "subtitles.json").write_text(json.dumps(subs, indent=2))

        overlays = [{"text": o["text"], "startFrame": round(o["start"] * fps), "endFrame": round(o["end"] * fps),
                     "position": o["position"]} for o in self.text_overlays]
        (self.project_dir / "text_overlays.json").write_text(json.dumps(overlays, indent=2))

        audio = [{"file": a["file"], "volume": a["volume"], "duck_below": a["duck_below"]} for a in self.audio_tracks]
        (self.project_dir / "audio_tracks.json").write_text(json.dumps(audio, indent=2))

        (self.project_dir / "composition_config.json").write_text(json.dumps({
            "width": width, "height": height, "fps": fps, "reframe": self.reframe,
        }, indent=2))

    def trajectory(self) -> dict:
        return {
            "calls": [
                {"name": c.name, "args": c.args, "ok": c.ok, "error": c.error, "latency_s": round(c.latency_s, 4)}
                for c in self.log
            ],
            "total_latency_s": round(sum(c.latency_s for c in self.log), 4),
            "n_calls": len(self.log),
            "n_failed_calls": sum(1 for c in self.log if not c.ok),
            "rendered": self.rendered,
            "final_state": {
                "timeline": self.timeline, "captions": self.captions, "transitions": self.transitions,
                "audio_tracks": self.audio_tracks, "text_overlays": self.text_overlays, "reframe": self.reframe,
            },
        }
