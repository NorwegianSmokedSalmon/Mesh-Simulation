"""
可视化对比修复前后的mesh

使用Open3D显示原始mesh和修复后mesh的对比
支持并排显示或叠加显示

使用方法：
    python visualize_comparison.py --original ../world_mesh --fixed ../world_mesh_fixed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh

LOG = logging.getLogger("visualize_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化对比修复前后的mesh")
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("../world_mesh"),
        help="原始mesh目录",
    )
    parser.add_argument(
        "--fixed",
        type=Path,
        default=Path("../world_mesh_fixed"),
        help="修复后mesh目录",
    )
    parser.add_argument(
        "--max_meshes",
        type=int,
        default=10,
        help="显示的最大mesh数量",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overlay",
        choices=["overlay", "side_by_side"],
        help="显示模式：overlay=叠加显示，side_by_side=并排显示",
    )
    return parser.parse_args()


def load_meshes_from_dir(directory: Path, max_meshes: int = 0) -> List[trimesh.Trimesh]:
    """从目录加载所有mesh"""
    meshes = []
    glb_files = sorted(directory.glob("*.glb"))
    
    if max_meshes > 0:
        glb_files = glb_files[:max_meshes]
    
    for glb_file in glb_files:
        try:
            scene = trimesh.load(str(glb_file), force="mesh")
            if isinstance(scene, trimesh.Scene):
                mesh = trimesh.util.concatenate([
                    trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                    for m in scene.geometry.values()
                    if isinstance(m, trimesh.Trimesh)
                ])
            else:
                mesh = scene
            
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0:
                meshes.append(mesh)
                LOG.info(f"加载 {glb_file.name}")
        except Exception as e:
            LOG.warning(f"加载 {glb_file.name} 失败: {e}")
    
    return meshes


def visualize_overlay(original_meshes: List[trimesh.Trimesh], fixed_meshes: List[trimesh.Trimesh]) -> None:
    """叠加显示原始和修复后的mesh"""
    try:
        import open3d as o3d
    except ImportError:
        LOG.error("需要安装open3d: pip install open3d")
        return
    
    geometries = []
    
    # 添加坐标轴
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(axes)
    
    # 原始mesh - 红色半透明
    LOG.info("处理原始mesh（红色）...")
    for mesh in original_meshes:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.paint_uniform_color([1.0, 0.3, 0.3])  # 红色
        o3d_mesh.compute_vertex_normals()
        geometries.append(o3d_mesh)
    
    # 修复后mesh - 绿色半透明
    LOG.info("处理修复后mesh（绿色）...")
    for mesh in fixed_meshes:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.paint_uniform_color([0.3, 1.0, 0.3])  # 绿色
        o3d_mesh.compute_vertex_normals()
        geometries.append(o3d_mesh)
    
    LOG.info("启动可视化窗口...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="对比视图 - 红色:原始 / 绿色:修复后",
        width=1600,
        height=900,
        mesh_show_back_face=True,
    )


def visualize_side_by_side(original_meshes: List[trimesh.Trimesh], fixed_meshes: List[trimesh.Trimesh]) -> None:
    """并排显示原始和修复后的mesh"""
    try:
        import open3d as o3d
    except ImportError:
        LOG.error("需要安装open3d: pip install open3d")
        return
    
    # 计算场景中心和范围
    all_vertices = []
    for mesh in original_meshes:
        all_vertices.append(mesh.vertices)
    all_vertices = np.vstack(all_vertices)
    center = np.mean(all_vertices, axis=0)
    extent = np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
    offset = extent[0] * 1.5  # 水平偏移距离
    
    geometries = []
    
    # 左侧：原始mesh（蓝色）
    LOG.info("处理原始mesh（左侧，蓝色）...")
    for mesh in original_meshes:
        o3d_mesh = o3d.geometry.TriangleMesh()
        vertices = mesh.vertices.copy()
        vertices[:, 0] -= offset  # 向左偏移
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.paint_uniform_color([0.3, 0.5, 1.0])  # 蓝色
        o3d_mesh.compute_vertex_normals()
        geometries.append(o3d_mesh)
    
    # 左侧坐标轴
    axes_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    axes_left.translate([-offset, 0, 0])
    geometries.append(axes_left)
    
    # 右侧：修复后mesh（绿色）
    LOG.info("处理修复后mesh（右侧，绿色）...")
    for mesh in fixed_meshes:
        o3d_mesh = o3d.geometry.TriangleMesh()
        vertices = mesh.vertices.copy()
        vertices[:, 0] += offset  # 向右偏移
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.paint_uniform_color([0.3, 1.0, 0.3])  # 绿色
        o3d_mesh.compute_vertex_normals()
        geometries.append(o3d_mesh)
    
    # 右侧坐标轴
    axes_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    axes_right.translate([offset, 0, 0])
    geometries.append(axes_right)
    
    LOG.info("启动可视化窗口...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="对比视图 - 左:原始(蓝) / 右:修复后(绿)",
        width=1600,
        height=900,
        mesh_show_back_face=True,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    
    # 检查目录
    if not args.original.exists():
        LOG.error(f"原始目录不存在: {args.original}")
        return
    
    if not args.fixed.exists():
        LOG.error(f"修复后目录不存在: {args.fixed}")
        LOG.info("请先运行 sdf_collision_resolver.py 生成修复后的mesh")
        return
    
    # 加载mesh
    LOG.info(f"从 {args.original} 加载原始mesh...")
    original_meshes = load_meshes_from_dir(args.original, args.max_meshes)
    
    LOG.info(f"从 {args.fixed} 加载修复后mesh...")
    fixed_meshes = load_meshes_from_dir(args.fixed, args.max_meshes)
    
    if len(original_meshes) == 0:
        LOG.error("没有加载到原始mesh")
        return
    
    if len(fixed_meshes) == 0:
        LOG.error("没有加载到修复后mesh")
        return
    
    LOG.info(f"原始mesh: {len(original_meshes)} 个")
    LOG.info(f"修复后mesh: {len(fixed_meshes)} 个")
    
    # 可视化
    if args.mode == "overlay":
        visualize_overlay(original_meshes, fixed_meshes)
    else:
        visualize_side_by_side(original_meshes, fixed_meshes)


if __name__ == "__main__":
    main()
