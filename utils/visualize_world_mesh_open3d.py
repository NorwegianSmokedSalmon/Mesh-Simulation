"""
Visualize all world-space meshes under `world_mesh/` using Open3D.

Expected input directory:
  /home/jack/下载/data/world_mesh/*.glb

Run (recommended):
  conda run -n instascene python visualize_world_mesh_open3d.py --dir /home/jack/下载/data/world_mesh
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


LOG = logging.getLogger("visualize_world_mesh_open3d")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize all meshes in world_mesh with Open3D.")
    p.add_argument(
        "--dir",
        type=Path,
        default=Path("/home/jack/下载/data/world_mesh"),
        help="Directory containing world-space .glb meshes.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.glb",
        help="Glob pattern for mesh files (default: *.glb).",
    )
    p.add_argument(
        "--max_meshes",
        type=int,
        default=0,
        help="Limit number of meshes to load (0 = no limit).",
    )
    p.add_argument(
        "--show_axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show world coordinate frame.",
    )
    p.add_argument(
        "--axes_size",
        type=float,
        default=0.5,
        help="Axis frame size.",
    )
    p.add_argument(
        "--compute_normals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute triangle normals (can help shading but costs time).",
    )
    p.add_argument(
        "--textured",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Open3D rendering pipeline to show textures if present (default: enabled).",
    )
    p.add_argument(
        "--lighting_profile",
        type=str,
        default="NO_SHADOWS",
        choices=["NO_SHADOWS", "SOFT_SHADOWS", "MED_SHADOWS", "HARD_SHADOWS", "DARK_SHADOWS"],
        help="Lighting profile for Open3D renderer.",
    )
    p.add_argument(
        "--light_dir",
        type=float,
        nargs=3,
        default=[0.0, 0.0, -1.0],  # 从z轴正方向向下照射
        help="Main light direction (x y z).",
    )
    p.add_argument(
        "--background",
        type=float,
        nargs=4,
        default=[0.02, 0.02, 0.02, 1.0],  # 恢复深色背景
        help="Background RGBA (0..1).",
    )
    return p.parse_args()


def _trimesh_to_o3d(mesh, o3d):
    """
    Convert trimesh.Trimesh to open3d.geometry.TriangleMesh (best-effort for colors).
    """
    tm = o3d.geometry.TriangleMesh()
    tm.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    tm.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))

    # Prefer vertex colors if present.
    vc = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices):
        cols = np.asarray(vc, dtype=np.float32)
        if cols.shape[1] == 4:
            cols = cols[:, :3]
        if cols.max() > 1.0:
            cols = cols / 255.0
        tm.vertex_colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    else:
        # If no vertex colors, paint a neutral gray to make it visible.
        tm.paint_uniform_color([0.7, 0.7, 0.7])

    return tm


def _run_textured_viewer(
    files: List[Path],
    *,
    show_axes: bool,
    axes_size: float,
    lighting_profile: str,
    light_dir: np.ndarray,
    background: np.ndarray,
) -> bool:
    """
    Use Open3D rendering + GUI to display GLB models with textures.
    Returns True if viewer launched, False if unsupported.
    """
    import open3d as o3d  # type: ignore

    if not hasattr(o3d.visualization, "gui") or not hasattr(o3d.visualization, "rendering"):
        return False

    from open3d.visualization import gui, rendering  # type: ignore

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("world_mesh (textured)", 1280, 720)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    # Lighting & background
    try:
        profile = getattr(scene_widget.scene.LightingProfile, lighting_profile)
    except Exception:
        profile = scene_widget.scene.LightingProfile.NO_SHADOWS
    try:
        scene_widget.scene.set_lighting(profile, light_dir.astype(np.float32))
    except Exception:
        pass
    try:
        scene_widget.scene.set_background_color(background.astype(np.float32))
    except Exception:
        try:
            scene_widget.scene.set_background(background.astype(np.float32))
        except Exception:
            pass

    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axes_size))
        scene_widget.scene.add_geometry("axes", axes, rendering.MaterialRecord())

    added = 0
    for i, fp in enumerate(files):
        try:
            model = o3d.io.read_triangle_model(str(fp))
        except Exception as exc:
            LOG.warning("Failed to read_triangle_model for %s: %s", fp.name, exc)
            continue
        try:
            scene_widget.scene.add_model(f"model_{i}", model)
            added += 1
        except Exception as exc:
            LOG.warning("Failed to add_model for %s: %s", fp.name, exc)

    if added == 0:
        LOG.warning("No models added to textured viewer.")
        return False

    # Fit camera to scene bounds.
    try:
        bounds = scene_widget.scene.bounding_box
        scene_widget.setup_camera(60.0, bounds, bounds.get_center())
    except Exception:
        pass

    app.run()
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Open3D not available in current Python. Please run with: "
            "`conda run -n instascene python visualize_world_mesh_open3d.py`"
        ) from exc

    if not args.dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.dir}")

    files = sorted(args.dir.glob(args.pattern))
    if args.max_meshes and args.max_meshes > 0:
        files = files[: int(args.max_meshes)]

    if not files:
        LOG.warning("No files matched %s under %s", args.pattern, args.dir)
        return

    if args.textured:
        launched = _run_textured_viewer(
            files,
            show_axes=bool(args.show_axes),
            axes_size=float(args.axes_size),
            lighting_profile=str(args.lighting_profile),
            light_dir=np.asarray(args.light_dir, dtype=np.float32),
            background=np.asarray(args.background, dtype=np.float32),
        )
        if launched:
            return
        LOG.warning("Textured viewer not supported; falling back to mesh-only viewer.")

    geoms: List[object] = []
    if args.show_axes:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.axes_size)))

    # Load GLBs via trimesh, then convert to Open3D.
    import trimesh

    loaded = 0
    for fp in files:
        try:
            obj = trimesh.load(str(fp), force="scene")
        except Exception as exc:
            LOG.warning("Failed to load %s: %s", fp.name, exc)
            continue

        meshes = []
        if isinstance(obj, trimesh.Trimesh):
            meshes = [obj]
        elif isinstance(obj, trimesh.Scene):
            # Scene graph transforms are usually baked in our exported GLBs, but to be safe:
            meshes = [m for m in obj.dump(concatenate=False) if isinstance(m, trimesh.Trimesh)]
        else:
            continue

        if not meshes:
            continue

        for m in meshes:
            tm = _trimesh_to_o3d(m, o3d)
            if args.compute_normals:
                tm.compute_vertex_normals()
                tm.compute_triangle_normals()
            geoms.append(tm)
        loaded += 1

    LOG.info("Loaded %d/%d GLBs (%d Open3D geometries). Launching viewer...", loaded, len(files), len(geoms))

    # Visualize
    o3d.visualization.draw_geometries(
        geoms,
        window_name="world_mesh (Open3D)",
        width=1280,
        height=720,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()

