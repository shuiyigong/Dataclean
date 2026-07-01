from __future__ import annotations

import numpy as np


def _odd_window(window: int, length: int) -> int:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 == 1 else max(1, length - 1)
    return window


def savgol_filter_np(values: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    """Small numpy-only Savitzky-Golay smoother for axis 0."""

    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] <= polyorder + 1:
        return values.astype(np.float32)
    window = _odd_window(window, x.shape[0])
    if window <= polyorder + 1:
        return values.astype(np.float32)
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(offsets, N=polyorder + 1, increasing=True)
    coeff = np.linalg.pinv(A)[0]
    padded = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        segment = padded[t : t + window]
        out[t] = np.tensordot(coeff, segment, axes=(0, 0))
    return out.astype(np.float32)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to quaternions in wxyz order."""

    q = np.empty((R.shape[0], 4), dtype=np.float64)
    for i, m in enumerate(R):
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            q[i] = [(0.25 * s), (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            axis = int(np.argmax(np.diag(m)))
            if axis == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
            elif axis == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    return q


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def gaussian_quat_smooth(rotation: np.ndarray, sigma: float = 2.0, radius: int | None = None) -> np.ndarray:
    """Gaussian weighted quaternion average with hemisphere alignment."""

    if sigma <= 0:
        return rotation.astype(np.float32)
    q = rotmat_to_quat(rotation)
    radius = int(radius if radius is not None else max(1, round(3 * sigma)))
    out = np.empty_like(q)
    for t in range(len(q)):
        lo = max(0, t - radius)
        hi = min(len(q), t + radius + 1)
        neigh = q[lo:hi].copy()
        dots = neigh @ q[t]
        neigh[dots < 0] *= -1
        offsets = np.arange(lo, hi) - t
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        avg = (neigh * weights[:, None]).sum(axis=0)
        out[t] = avg / np.maximum(np.linalg.norm(avg), 1e-12)
    return quat_to_rotmat(out)


def smooth_gripper(gripper: dict[str, np.ndarray], *, position_window: int = 11, width_window: int = 11, polyorder: int = 2, rot_sigma: float = 2.0) -> dict[str, np.ndarray]:
    smoothed = dict(gripper)
    smoothed["position"] = savgol_filter_np(gripper["position"], position_window, polyorder)
    smoothed["width"] = savgol_filter_np(gripper["width"], width_window, polyorder)
    smoothed["rotation"] = gaussian_quat_smooth(gripper["rotation"], sigma=rot_sigma)
    return smoothed

