from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


CANONICAL_DIM = 14

# Unified layout: left xyz(3) rpy(3) gripper(1) + right xyz(3) rpy(3) gripper(1)
LEFT_XYZ_SLICE = slice(0, 3)
LEFT_RPY_SLICE = slice(3, 6)
LEFT_GRIPPER_IDX = 6
RIGHT_XYZ_SLICE = slice(7, 10)
RIGHT_RPY_SLICE = slice(10, 13)
RIGHT_GRIPPER_IDX = 13
XYZ_INDICES = (0, 1, 2, 7, 8, 9)
RPY_INDICES = (3, 4, 5, 10, 11, 12)
GRIPPER_INDICES = (6, 13)
JOINT_INDICES = tuple(range(12))


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix."""
    x_raw = rot6d[..., 0:3]
    y_raw = rot6d[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """(T, 6) rot6d -> (T, 3) extrinsic xyz euler radians."""
    matrix = rot6d_to_matrix(rot6d)
    return Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)


def egodex_to_canonical(raw: np.ndarray) -> np.ndarray:
    """EgoDex 20-dim (xyz+rot6d+width x2) -> canonical 14-dim (xyz+rpy+gripper x2)."""
    out = np.zeros((raw.shape[0], CANONICAL_DIM), dtype=np.float64)
    out[:, LEFT_XYZ_SLICE] = raw[:, 0:3]
    out[:, LEFT_RPY_SLICE] = rot6d_to_euler(raw[:, 3:9])
    out[:, LEFT_GRIPPER_IDX] = raw[:, 9]
    out[:, RIGHT_XYZ_SLICE] = raw[:, 10:13]
    out[:, RIGHT_RPY_SLICE] = rot6d_to_euler(raw[:, 13:19])
    out[:, RIGHT_GRIPPER_IDX] = raw[:, 19]
    return out


def humanoid_state_to_canonical(state: np.ndarray) -> np.ndarray:
    """Humanoid state: keep first 14 dims (12 joint + 2 gripper)."""
    return state[:, :CANONICAL_DIM].astype(np.float64, copy=False)


def humanoid_action_to_canonical(action: np.ndarray) -> np.ndarray:
    """Humanoid action is_f already 14-dim (12 joint + 2 gripper)."""
    return action[:, :CANONICAL_DIM].astype(np.float64, copy=False)
