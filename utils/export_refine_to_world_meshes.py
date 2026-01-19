"""
Export per-instance refined meshes from view0 (PyTorch3D camera convention) to world coordinates.

This script scans:
  {data_dir}/sam3d/instance_*_refine.glb

Then for each instance_id, it looks up the corresponding camera entry in:
  {data_dir}/images/cameras.json

We use the *view0* camera for that instance, defined as the minimal rank value for that instance_id.

The world transform follows `run_instance_best_views_inference_new.py::refined_mesh_view_to_world`:
  T_world = c2w @ T_flip @ T_p3d_to_r3

Outputs are saved under:
  {data_dir}/world_mesh/instance_{id}_refine_world.glb
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh

LOG = logging.getLogger("export_refine_to_world_meshes")

REFINE_RE = re.compile(r"^instance_(\d+)_refine$")


def _as_4x4(matrix_like) -> np.ndarray:
    mat = np.asarray(matrix_like, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    if mat.ndim == 1 and mat.size == 16:
        return mat.reshape(4, 4)
    if mat.ndim == 1 and mat.size == 12:
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat.reshape(3, 4)
        return out
    raise ValueError(f"Expected 4x4 (or 3x4) matrix, got {mat.shape}")


def _load_refined_mesh_only(path: Path) -> Optional[trimesh.Trimesh]:
    """
    Load `instance_*_refine.glb` and extract ONLY the refined mesh geometry,
    excluding frustum/pointcloud.

    We prefer geometry names containing "mesh_refined" (as produced by your data),
    then fallback to "mesh" while excluding "frustum"/"camera"/"pointcloud".
    """
    try:
        obj = trimesh.load(str(path), force="scene")
    except Exception as exc:
        LOG.warning("Failed to load GLB scene: %s (%s)", path, exc)
        return None

    if isinstance(obj, trimesh.Trimesh):
        return obj

    if not isinstance(obj, trimesh.Scene):
        LOG.warning("Unexpected trimesh.load result for %s: %s", path, type(obj))
        return None

    # Scene graph transforms appear to be identity in your files, but keep this
    # robust by using the geometry directly (node transforms are typically baked
    # in the exported refine.glb already).
    geoms = [(name, g) for name, g in obj.geometry.items() if isinstance(g, trimesh.Trimesh)]
    if not geoms:
        return None

    def _bad(name: str) -> bool:
        n = name.lower()
        return ("frustum" in n) or ("camera" in n) or ("pointcloud" in n) or ("pc" in n)

    def _score(name: str, g: trimesh.Trimesh) -> Tuple[int, int]:
        n = name.lower()
        s = 0
        if "mesh_refined" in n:
            s += 100
        if "mesh" in n:
            s += 10
        if _bad(n):
            s -= 1000
        # tie-breaker: prefer larger face count
        return (s, int(getattr(g, "faces", np.zeros((0, 3))).shape[0]))

    best_name, best_geom = max(geoms, key=lambda ng: _score(ng[0], ng[1]))
    if _bad(best_name):
        LOG.warning("Best geometry looks non-mesh (name=%s) for %s", best_name, path)
        return None
    return best_geom.copy()


def _load_base_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    """
    Load base instance_{id}.glb as a single Trimesh (with transforms baked in).
    """
    try:
        obj = trimesh.load(str(path), force="mesh")
    except Exception as exc:
        LOG.warning("Failed to load base mesh: %s (%s)", path, exc)
        return None
    return obj if isinstance(obj, trimesh.Trimesh) else None


def _bake_texture_to_vertex_colors(
    mesh: trimesh.Trimesh, *, image, uv: np.ndarray
) -> Optional[np.ndarray]:
    """
    Best-effort: sample texture image at per-vertex UVs to produce vertex colors.
    Returns Nx4 uint8 colors or None if baking isn't possible.
    """
    if image is None:
        return None
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        return None
    if uv.shape[0] != len(mesh.vertices):
        # Fallback for per-face-vertex UVs is non-trivial; skip.
        return None
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.asarray(image))
        except Exception:
            return None

    w, h = image.size
    if w <= 0 or h <= 0:
        return None

    # Clamp UV into [0,1], convert to pixel coordinates (v is typically top-down in images).
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = (u * (w - 1)).astype(np.int32)
    y = ((1.0 - v) * (h - 1)).astype(np.int32)

    pix = np.array(image.convert("RGBA"), dtype=np.uint8)
    colors = pix[y, x]
    return colors


def _transfer_visuals_if_possible(
    refined_mesh: trimesh.Trimesh, base_mesh: trimesh.Trimesh, *, bake_texture: bool
) -> None:
    """
    Try to copy visual info (vertex colors / textures) from base mesh to refined mesh.
    """
    if len(base_mesh.vertices) != len(refined_mesh.vertices):
        return

    base_visual = getattr(base_mesh, "visual", None)
    if base_visual is None:
        return

    # Prefer vertex colors if present.
    vc = getattr(base_visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(base_mesh.vertices):
        refined_mesh.visual = base_visual.copy()
        return

    if not bake_texture:
        return

    # Try to bake texture to vertex colors.
    uv = getattr(base_visual, "uv", None)
    mat = getattr(base_visual, "material", None)
    image = getattr(mat, "image", None) if mat is not None else None
    colors = _bake_texture_to_vertex_colors(refined_mesh, image=image, uv=uv)
    if colors is not None and len(colors) == len(refined_mesh.vertices):
        refined_mesh.visual = trimesh.visual.ColorVisuals(
            refined_mesh, vertex_colors=colors
        )


def _load_camera_entries(cameras_json: Path) -> Dict[Tuple[int, int], dict]:
    payload = json.loads(cameras_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected cameras.json format (expected list): {cameras_json}")
    out: Dict[Tuple[int, int], dict] = {}
    for e in payload:
        if not isinstance(e, dict):
            continue
        iid = e.get("instance_id")
        rank = e.get("rank")
        if iid is None or rank is None:
            continue
        try:
            key = (int(iid), int(rank))
        except Exception:
            continue
        out[key] = e
    return out


def _pick_view0_cam(cam_map: Dict[Tuple[int, int], dict], instance_id: int) -> Optional[dict]:
    ranks = [rank for (iid, rank) in cam_map.keys() if iid == int(instance_id)]
    if not ranks:
        return None
    r0 = min(ranks)
    return cam_map.get((int(instance_id), int(r0)))


def _get_T_p3d_to_r3() -> np.ndarray:
    """
    Return T_p3d_to_r3 (4x4) in *column-vector* convention, matching
    `run_instance_best_views_inference_new.py`.

    It intentionally sets:
      T_p3d_to_r3[:3,:3] = R_r3_to_p3d_row
    (NOT transpose), due to row/column convention derivation in that script.
    """
    try:
        from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # type: ignore
            camera_to_pytorch3d_camera,
        )

        R_r3_to_p3d_row = (
            camera_to_pytorch3d_camera(device="cpu")
            .rotation.detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32, copy=False)
        )
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_r3_to_p3d_row
        return T
    except Exception as exc:
        # Fallback (best-effort): commonly used conversion for PyTorch3D vs OpenCV-like camera.
        # If this fallback is wrong for your environment, please install/enable `sam3d_objects`
        # so we can use the exact rotation matrix.
        LOG.warning(
            "Failed to import sam3d_objects to get exact camera rotation; using fallback. Error=%s",
            exc,
        )
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return T


def _refine_mesh_view_to_world(
    mesh_view_p3d: trimesh.Trimesh,
    *,
    cam_entry: dict,
    undo_image_flip_lr: bool,
    T_p3d_to_r3: np.ndarray,
) -> trimesh.Trimesh:
    if "c2w" not in cam_entry:
        raise KeyError("cameras.json entry missing 'c2w'")
    c2w = _as_4x4(cam_entry["c2w"])

    T_flip = np.eye(4, dtype=np.float32)
    if undo_image_flip_lr:
        T_flip[0, 0] = -1.0

    T_world = c2w @ T_flip @ T_p3d_to_r3

    out = mesh_view_p3d.copy()
    out.apply_transform(T_world.astype(np.float64))

    det = float(np.linalg.det(T_world[:3, :3]))
    if det < 0.0:
        out.invert()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert instance_*_refine.glb meshes to world coordinates.")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/jack/下载/data"),
        help="Workspace root (contains images/, sam3d/).",
    )
    p.add_argument(
        "--sam3d_dir",
        type=Path,
        default=None,
        help="Optional override for sam3d directory (default: {data_dir}/sam3d).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: {data_dir}/world_mesh).",
    )
    p.add_argument(
        "--undo_image_flip_lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match run_instance_best_views_inference_new.py default; undo left-right flip if present.",
    )
    p.add_argument(
        "--transfer_visuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try to transfer visual info (vertex colors / texture) from instance_{id}.glb.",
    )
    p.add_argument(
        "--bake_texture_to_vertex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If base mesh has texture, attempt to bake to vertex colors (best-effort).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be exported.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    data_dir: Path = args.data_dir
    sam3d_dir: Path = args.sam3d_dir or (data_dir / "sam3d")
    out_dir: Path = args.out_dir or (data_dir / "world_mesh")
    cameras_json: Path = data_dir / "images" / "cameras.json"

    if not sam3d_dir.exists():
        raise FileNotFoundError(f"sam3d_dir not found: {sam3d_dir}")
    if not cameras_json.exists():
        raise FileNotFoundError(f"cameras.json not found: {cameras_json}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cam_map = _load_camera_entries(cameras_json)
    LOG.info("Loaded %d camera entries from %s", len(cam_map), cameras_json)

    refine_files = sorted(sam3d_dir.glob("instance_*_refine.glb"))
    LOG.info("Found %d refine GLBs under %s", len(refine_files), sam3d_dir)
    if not refine_files:
        return

    T_p3d_to_r3 = _get_T_p3d_to_r3()

    ok = 0
    for p in refine_files:
        m = REFINE_RE.match(p.stem)
        if not m:
            LOG.warning("Skip (unexpected filename): %s", p.name)
            continue
        instance_id = int(m.group(1))

        cam_entry = _pick_view0_cam(cam_map, instance_id)
        if cam_entry is None:
            LOG.warning("[instance %d] No camera entry found, skip", instance_id)
            continue

        mesh_view = _load_refined_mesh_only(p)
        if mesh_view is None:
            LOG.warning("[instance %d] Failed to load mesh, skip", instance_id)
            continue

        # Optionally transfer visuals (colors/texture) from base mesh.
        if args.transfer_visuals:
            base_path = sam3d_dir / f"instance_{instance_id}.glb"
            base_mesh = _load_base_mesh(base_path) if base_path.exists() else None
            if base_mesh is not None:
                _transfer_visuals_if_possible(
                    mesh_view, base_mesh, bake_texture=bool(args.bake_texture_to_vertex)
                )

        try:
            mesh_world = _refine_mesh_view_to_world(
                mesh_view,
                cam_entry=cam_entry,
                undo_image_flip_lr=bool(args.undo_image_flip_lr),
                T_p3d_to_r3=T_p3d_to_r3,
            )
        except Exception as exc:
            LOG.warning("[instance %d] Failed to transform to world: %s", instance_id, exc)
            continue

        out_path = out_dir / f"instance_{instance_id}_refine_world.glb"
        if args.dry_run:
            LOG.info("[dry_run] would export %s -> %s", p.name, out_path)
            ok += 1
            continue

        try:
            mesh_world.export(str(out_path))
            ok += 1
        except Exception as exc:
            LOG.warning("[instance %d] Failed to export world GLB: %s", instance_id, exc)

    LOG.info("Done. Exported %d/%d world meshes into %s", ok, len(refine_files), out_dir)


if __name__ == "__main__":
    main()

