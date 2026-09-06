"""
Completes two more taxonomy subtypes with real, checkable ground truth,
built from footage already on disk (no new pulls needed):

  text_overlay - self-authored, ground truth is exact expected tool args
                 (structurally checkable, same pattern as add_subtitle)
  reframe      - ground truth is REAL per-frame face bboxes from YuNet
                 (cv2.FaceDetectorYN, downloaded ONNX model, verified
                 working on our own footage) run on the source 16:9 clip.
                 Scoring re-runs the same detector on the agent's reframed
                 9:16 output at the same timestamps: if a face was
                 detectable in the source frame but not in the reframed
                 one, that's a "cropped the subject out" failure.
"""
import json
import subprocess
from pathlib import Path

import cv2

TASK_SUBTITLE = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/task_subtitle")
MODEL_PATH = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/models/face_detection_yunet.onnx")
OUT_DIR = Path("/tmp/claude-1000/-home-sanyam-0605-Desktop-sanyam-projects/c62f2cd8-7294-4c66-88c7-89e39e1fa6bb/scratchpad/evals-project/dataset_v1")


def build_text_overlay_task():
    return {
        "task_id": "edit_text_overlay_001",
        "category": "editing",
        "subtype": "text_overlay",
        "instruction": (
            'Add a lower-third title reading "INTERROGATION ROOM" to dialogue_clip.mp4, '
            "visible from 0s to 4s, positioned in the lower third of the frame."
        ),
        "assets": ["dialogue_clip.mp4"],
        "expected_tool_sequence": ["probe", "add_to_timeline", "add_text_overlay", "render"],
        "scoring_method": "structural",
        "ground_truth": {
            "text": "INTERROGATION ROOM",
            "start": 0.0,
            "end": 4.0,
            "position": "lower_third",
        },
        "reference_video": None,
    }


def build_reframe_task(asset_path: Path, sample_every_s: float = 1.0):
    detector = None
    cap = cv2.VideoCapture(str(asset_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = cv2.FaceDetectorYN_create(str(MODEL_PATH), "", (w, h))

    face_frames = []
    step = max(1, int(fps * sample_every_s))
    for frame_idx in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        n, faces = detector.detect(frame)
        t = frame_idx / fps
        if faces is not None and len(faces) > 0:
            x, y, fw, fh = faces[0][:4]
            face_frames.append({"t": round(t, 2), "bbox_xywh": [float(x), float(y), float(fw), float(fh)],
                                 "score": float(faces[0][-1])})
        else:
            face_frames.append({"t": round(t, 2), "bbox_xywh": None, "score": 0.0})
    cap.release()

    n_with_face = sum(1 for f in face_frames if f["bbox_xywh"] is not None)
    return {
        "task_id": "edit_reframe_001",
        "category": "editing",
        "subtype": "reframe",
        "instruction": (
            f"Reframe {asset_path.name} (currently {w}x{h}, 16:9-ish) into a 9:16 vertical "
            "crop suitable for mobile, keeping the speaker's face in frame throughout."
        ),
        "assets": [asset_path.name],
        "expected_tool_sequence": ["probe", "add_to_timeline", "set_reframe_crop", "render"],
        "scoring_method": "structural",
        "ground_truth": {
            "source_resolution": [w, h],
            "target_aspect": "9:16",
            "source_face_track": face_frames,
            "n_frames_with_detectable_face": n_with_face,
            "n_frames_sampled": len(face_frames),
            "note": (
                "real YuNet face detections on the SOURCE clip, sampled every "
                f"{sample_every_s}s. Scoring re-runs the same detector on the "
                "agent's reframed output at matching timestamps -- a face "
                "detectable in source but not in output at the same t = subject "
                "cropped out."
            ),
        },
        "reference_video": None,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asset = TASK_SUBTITLE / "assets/dialogue_clip.mp4"

    text_overlay = build_text_overlay_task()
    reframe = build_reframe_task(asset)

    (OUT_DIR / "editing_reframe").mkdir(exist_ok=True)
    (OUT_DIR / "editing_reframe" / "dialogue_clip.mp4").write_bytes(asset.read_bytes())
    (OUT_DIR / "editing_text_overlay").mkdir(exist_ok=True)
    (OUT_DIR / "editing_text_overlay" / "dialogue_clip.mp4").write_bytes(asset.read_bytes())

    print(json.dumps(text_overlay, indent=2))
    print(f"\nreframe: {reframe['ground_truth']['n_frames_with_detectable_face']}/"
          f"{reframe['ground_truth']['n_frames_sampled']} sampled frames had a detectable face")
    return text_overlay, reframe


if __name__ == "__main__":
    main()
