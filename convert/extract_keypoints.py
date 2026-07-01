from __future__ import annotations

import numpy as np


def extract_hand_keypoints(joints: dict[str, np.ndarray], hand: str) -> dict[str, np.ndarray]:
    """Extract thumb/index/middle/wrist positions for one hand."""

    if hand not in {"left", "right"}:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    prefix = hand
    names = {
        "thumb": f"{prefix}ThumbTip",
        "index": f"{prefix}IndexFingerTip",
        "middle": f"{prefix}MiddleFingerTip",
        "wrist": f"{prefix}Hand",
    }
    missing = [name for name in names.values() if name not in joints]
    if missing:
        raise KeyError(f"Missing joints for {hand} hand: {missing}")
    return {key: joints[name][:, :3, 3].astype(np.float32) for key, name in names.items()}


def extract_bimanual_keypoints(joints: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        "left": extract_hand_keypoints(joints, "left"),
        "right": extract_hand_keypoints(joints, "right"),
    }

