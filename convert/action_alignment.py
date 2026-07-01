from __future__ import annotations

import numpy as np

EPS = 1e-6


def _normalize(v: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps), norm.squeeze(-1)


def _repair_rotation(R: np.ndarray) -> np.ndarray:
    repaired = np.empty_like(R, dtype=np.float32)
    for i, mat in enumerate(R):
        u, _, vh = np.linalg.svd(mat)
        fixed = u @ vh
        if np.linalg.det(fixed) < 0:
            u[:, -1] *= -1
            fixed = u @ vh
        repaired[i] = fixed.astype(np.float32)
    return repaired


def retarget_to_gripper(keypoints: dict[str, np.ndarray], hand: str) -> dict[str, np.ndarray]:
    """Retarget EgoDex hand keypoints into a virtual parallel-jaw gripper."""

    if hand not in {"left", "right"}:
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")

    thumb = keypoints["thumb"]
    index = keypoints["index"]
    middle = keypoints["middle"]
    wrist = keypoints["wrist"]

    k_vf = 0.7 * index + 0.3 * middle
    position = 0.5 * (thumb + k_vf)
    width_vec = thumb - k_vf
    z_unsigned, width = _normalize(width_vec)
    z = (1.0 if hand == "right" else -1.0) * z_unsigned

    d = k_vf - wrist
    y_raw = np.cross(z, d)
    y, y_norm = _normalize(y_raw)

    bad = y_norm < EPS
    if np.any(bad):
        fallback = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (len(y), 1))
        y[bad] = fallback[bad]
        y[bad], _ = _normalize(y[bad] - (np.sum(y[bad] * z[bad], axis=-1, keepdims=True) * z[bad]))

    x, _ = _normalize(np.cross(y, z))
    y, _ = _normalize(np.cross(z, x))
    rotation = np.stack([x, y, z], axis=-1).astype(np.float32)
    rotation = _repair_rotation(rotation)

    return {
        "position": position.astype(np.float32),
        "rotation": rotation,
        "width": width.astype(np.float32),
        "valid_mask": (width > EPS) & (y_norm > EPS),
    }


def rotation_6d(rotation: np.ndarray) -> np.ndarray:
    """Return Zhou et al. 6D representation: first two rotation columns."""

    return rotation[:, :, :2].transpose(0, 2, 1).reshape(rotation.shape[0], 6).astype(np.float32)


def gripper_action_10d(gripper: dict[str, np.ndarray]) -> np.ndarray:
    """Pack one hand as 3 position + 6D rotation + 1 width."""

    return np.concatenate(
        [gripper["position"], rotation_6d(gripper["rotation"]), gripper["width"][:, None]],
        axis=-1,
    ).astype(np.float32)

