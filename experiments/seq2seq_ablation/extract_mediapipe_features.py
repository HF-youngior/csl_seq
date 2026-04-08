#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import _load_info_samples

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    mp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


ARM_POSE_INDICES = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
FACE_KEY_INDICES = [1, 33, 61, 78, 152, 199, 263, 291, 308, 17]


def resolve_holistic_cls():
    if mp is None:
        raise RuntimeError(
            f"mediapipe import failed: {_IMPORT_ERROR}\nInstall with: pip install mediapipe"
        )

    # Path 1: classic API (most common)
    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        holistic_mod = getattr(solutions, "holistic", None)
        holistic_cls = getattr(holistic_mod, "Holistic", None) if holistic_mod is not None else None
        if holistic_cls is not None:
            return holistic_cls

    # Path 2: some builds expose solutions under mediapipe.python
    for mod_name in [
        "mediapipe.python.solutions.holistic",
        "mediapipe.solutions.holistic",
    ]:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        holistic_cls = getattr(mod, "Holistic", None)
        if holistic_cls is not None:
            return holistic_cls

    raise RuntimeError(
        "Cannot locate MediaPipe Holistic API.\n"
        f"mediapipe module path: {getattr(mp, '__file__', 'unknown')}\n"
        f"mediapipe version: {getattr(mp, '__version__', 'unknown')}\n"
        "Expected one of: mp.solutions.holistic.Holistic or mediapipe.python.solutions.holistic.Holistic.\n"
        "Please check if a non-official 'mediapipe' package is installed in the current env."
    )


def str2bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract offline MediaPipe features for Seq2Seq ablation.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root folder of color videos.")
    parser.add_argument("--info-path", type=str, required=True, help="VAC split info npy path (e.g. train_info.npy).")
    parser.add_argument("--output-dir", type=str, required=True, help="Output feature dir, saved as <fileid>.npy.")
    parser.add_argument("--frames", type=int, default=32, help="Sampled frames per video.")
    parser.add_argument(
        "--components",
        type=str,
        default="hands",
        choices=["hands", "hands_arm", "hands_arm_face"],
        help="Feature composition.",
    )
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shard count for parallel extraction.")
    parser.add_argument("--shard-id", type=int, default=0, help="Current shard id in [0, num_shards).")
    return parser.parse_args()


def sample_frames(video_path: str, target_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = None

    frames = []
    if total:
        wanted = np.linspace(0, total - 1, target_frames).astype(np.int32).tolist()
        wanted_set = set(wanted)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
    else:
        # Fallback when frame count metadata is unavailable.
        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if raw:
            wanted = np.linspace(0, len(raw) - 1, target_frames).astype(np.int32).tolist()
            frames = [raw[i] for i in wanted]
    cap.release()

    if not frames:
        raise RuntimeError(f"Decoded 0 frames: {video_path}")
    while len(frames) < target_frames:
        frames.append(frames[-1])
    if len(frames) > target_frames:
        frames = frames[:target_frames]
    return frames


def _extract_hands(result) -> np.ndarray:
    feats = []
    for hand_attr in ["left_hand_landmarks", "right_hand_landmarks"]:
        hand = getattr(result, hand_attr)
        if hand is None:
            feats.extend([0.0] * (21 * 3))
            continue
        for lm in hand.landmark:
            feats.extend([lm.x, lm.y, lm.z])
    return np.asarray(feats, dtype=np.float32)


def _extract_arm(result) -> np.ndarray:
    if result.pose_landmarks is None:
        return np.zeros((len(ARM_POSE_INDICES) * 4,), dtype=np.float32)
    feats = []
    pose = result.pose_landmarks.landmark
    for idx in ARM_POSE_INDICES:
        lm = pose[idx]
        feats.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.asarray(feats, dtype=np.float32)


def _extract_face(result) -> np.ndarray:
    if result.face_landmarks is None:
        return np.zeros((len(FACE_KEY_INDICES) * 3,), dtype=np.float32)
    feats = []
    face = result.face_landmarks.landmark
    for idx in FACE_KEY_INDICES:
        lm = face[idx]
        feats.extend([lm.x, lm.y, lm.z])
    return np.asarray(feats, dtype=np.float32)


def get_feature_dim(components: str) -> int:
    dim = 21 * 3 * 2  # two hands
    if components in {"hands_arm", "hands_arm_face"}:
        dim += len(ARM_POSE_INDICES) * 4
    if components == "hands_arm_face":
        dim += len(FACE_KEY_INDICES) * 3
    return dim


def extract_video_feature(holistic, frames: List[np.ndarray], components: str) -> np.ndarray:
    per_frame = []
    for frame_rgb in frames:
        result = holistic.process(frame_rgb)
        feats = [_extract_hands(result)]
        if components in {"hands_arm", "hands_arm_face"}:
            feats.append(_extract_arm(result))
        if components == "hands_arm_face":
            feats.append(_extract_face(result))
        per_frame.append(np.concatenate(feats, axis=0))
    stacked = np.stack(per_frame, axis=0)  # [T, D]
    return stacked.mean(axis=0).astype(np.float32)  # compact and disk-friendly


def main():
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError(f"--num-shards must be > 0, got {args.num_shards}")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"--shard-id must be in [0, {args.num_shards}), got {args.shard_id}"
        )
    holistic_cls = resolve_holistic_cls()

    all_samples = _load_info_samples(args.info_path)
    samples = [s for idx, s in enumerate(all_samples) if (idx % args.num_shards) == args.shard_id]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    os.makedirs(args.output_dir, exist_ok=True)

    expected_dim = get_feature_dim(args.components)
    dtype = np.float16 if args.dtype == "float16" else np.float32

    written = 0
    skipped = 0
    bad = 0
    estimated_bytes = 0

    with holistic_cls(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for sample in tqdm(samples, desc="extract mediapipe"):
            fileid = str(sample["fileid"])
            rel_path = str(sample["folder"]).replace("/", os.sep)
            video_path = os.path.join(args.dataset_root, rel_path)
            out_path = os.path.join(args.output_dir, f"{fileid}.npy")

            if os.path.isfile(out_path) and not args.overwrite:
                skipped += 1
                continue

            try:
                frames = sample_frames(video_path, args.frames)
                feat = extract_video_feature(holistic, frames, args.components)
                if feat.shape[0] != expected_dim:
                    raise RuntimeError(f"feature dim mismatch: got {feat.shape[0]}, expect {expected_dim}")
                feat = feat.astype(dtype, copy=False)
                np.save(out_path, feat)
                written += 1
                estimated_bytes += int(feat.nbytes)
            except Exception:
                bad += 1

    print(
        f"components={args.components}, feature_dim={expected_dim}, dtype={args.dtype}, "
        f"shard={args.shard_id}/{args.num_shards - 1}"
    )
    print(
        f"samples_total_split={len(all_samples)}, samples_this_shard={len(samples)}, "
        f"written={written}, skipped={skipped}, bad={bad}"
    )
    print(f"raw_feature_bytes={estimated_bytes} (~{estimated_bytes / (1024 ** 2):.2f} MB)")
    print("Note: npy header overhead is not included above.")


if __name__ == "__main__":
    main()
