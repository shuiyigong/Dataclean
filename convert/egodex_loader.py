from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

FPS = 30


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_decode_attr(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _episode_paths(path: str | Path) -> tuple[Path, Path | None]:
    path = Path(path)
    if path.suffix == ".hdf5":
        hdf5_path = path
    elif path.is_dir():
        hdf5_files = sorted(path.glob("*.hdf5"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        if len(hdf5_files) != 1:
            raise ValueError(f"Expected one .hdf5 in {path}, found {len(hdf5_files)}")
        hdf5_path = hdf5_files[0]
    else:
        raise FileNotFoundError(path)
    mp4_path = hdf5_path.with_suffix(".mp4")
    return hdf5_path, mp4_path if mp4_path.exists() else None


def _task_description(attrs: dict[str, Any]) -> str:
    if attrs.get("llm_type") == "reversible":
        direction = str(attrs.get("which_llm_description", "1"))
        key = "llm_description" if direction == "1" else "llm_description2"
        return str(attrs.get(key) or attrs.get("llm_description") or attrs.get("task") or "")
    return str(attrs.get("llm_description") or attrs.get("task") or "")


def load_episode(path: str | Path, *, camera_frame: bool = True) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Load one EgoDex episode.

    HDF5 stores all transforms in ARKit origin/world frame. With
    ``camera_frame=True`` this returns joint transforms as
    ``inv(transforms/camera) @ transforms/<joint>`` while preserving the camera
    extrinsics in ``meta["camera_extrinsics_world"]``.
    """

    hdf5_path, mp4_path = _episode_paths(path)
    with h5py.File(hdf5_path, "r") as root:
        attrs = {key: _decode_attr(value) for key, value in root.attrs.items()}
        camera_world = root["transforms/camera"][:].astype(np.float32)
        camera_inv = np.linalg.inv(camera_world) if camera_frame else None

        joints: dict[str, np.ndarray] = {}
        for name in sorted(root["transforms"].keys()):
            tf = root[f"transforms/{name}"][:].astype(np.float32)
            if name != "camera" and camera_frame:
                tf = np.einsum("tij,tjk->tik", camera_inv, tf).astype(np.float32)
            joints[name] = tf

        confidence: dict[str, np.ndarray] = {}
        if "confidences" in root:
            for name in sorted(root["confidences"].keys()):
                confidence[name] = root[f"confidences/{name}"][:].astype(np.float32)

        meta = {
            "hdf5_path": str(hdf5_path),
            "mp4_path": str(mp4_path) if mp4_path else None,
            "episode_id": hdf5_path.stem,
            "fps": FPS,
            "num_frames": int(camera_world.shape[0]),
            "task_name": attrs.get("task", hdf5_path.parent.name),
            "language_description": _task_description(attrs),
            "attrs": attrs,
            "camera_intrinsics": root["camera/intrinsic"][:].astype(np.float32),
            "camera_extrinsics_world": camera_world,
            "joint_frame": "camera" if camera_frame else "arkit_origin",
        }
    return joints, confidence, meta


def iter_episode_hdf5(root: str | Path) -> list[Path]:
    root = Path(root)
    paths = sorted(root.rglob("*.hdf5"), key=lambda p: (str(p.parent), int(p.stem) if p.stem.isdigit() else p.stem))
    return paths

