"""
快速版本的SDF碰撞检测器

优化策略：
1. 降低SDF分辨率，使用自适应采样
2. 使用简化的碰撞检测（基于AABB和采样点）
3. 并行化处理
4. 使用更高效的近似SDF方法

使用方法：
    python sdf_collision_resolver_fast.py --input_dir ../world_mesh --output_dir ../world_mesh_fixed
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import json


LOG = logging.getLogger("sdf_fast")


@dataclass
class MeshObject:
    """表示一个带有变换的mesh对象"""
    mesh: trimesh.Trimesh
    filename: str
    instance_id: int
    position: np.ndarray
    velocity: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    
    @property
    def center(self) -> np.ndarray:
        return (self.bounds_min + self.bounds_max) / 2
    
    @property
    def size(self) -> np.ndarray:
        return self.bounds_max - self.bounds_min


class FastCollisionResolver:
    """快速碰撞检测和修复器"""
    
    def __init__(self, tolerance: float = 0.01, sample_points: int = 500):
        """
        Args:
            tolerance: 穿透容忍阈值（米）
            sample_points: 每个mesh用于检测的采样点数
        """
        self.tolerance = tolerance
        self.sample_points = sample_points
        self.meshes: List[MeshObject] = []
        self.kdtrees = {}  # 缓存KD树
        
    def load_meshes(self, input_dir: Path, max_meshes: int = 0) -> None:
        """加载目录下的所有GLB文件"""
        LOG.info(f"从 {input_dir} 加载mesh文件...")
        
        glb_files = sorted(input_dir.glob("*.glb"))
        if max_meshes > 0:
            glb_files = glb_files[:max_meshes]
        
        for glb_file in glb_files:
            try:
                instance_id = int(glb_file.stem.split("_")[1])
                
                scene = trimesh.load(str(glb_file), force="mesh")
                if isinstance(scene, trimesh.Scene):
                    mesh = trimesh.util.concatenate([
                        trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                        for m in scene.geometry.values()
                        if isinstance(m, trimesh.Trimesh)
                    ])
                else:
                    mesh = scene
                
                if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                    continue
                
                bounds = mesh.bounds
                
                mesh_obj = MeshObject(
                    mesh=mesh,
                    filename=glb_file.name,
                    instance_id=instance_id,
                    position=mesh.centroid.copy(),
                    velocity=np.zeros(3),
                    bounds_min=bounds[0].copy(),
                    bounds_max=bounds[1].copy(),
                )
                
                self.meshes.append(mesh_obj)
                LOG.info(f"✓ 加载 {glb_file.name}: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面片")
                
            except Exception as e:
                LOG.error(f"加载 {glb_file.name} 失败: {e}")
        
        LOG.info(f"总共加载 {len(self.meshes)} 个mesh")
    
    def _get_kdtree(self, mesh_obj: MeshObject) -> cKDTree:
        """获取或创建mesh的KD树（用于快速最近点查询）"""
        if mesh_obj.instance_id not in self.kdtrees:
            vertices = mesh_obj.mesh.vertices
            self.kdtrees[mesh_obj.instance_id] = cKDTree(vertices)
        return self.kdtrees[mesh_obj.instance_id]
    
    def _aabb_intersect(self, mesh_a: MeshObject, mesh_b: MeshObject) -> bool:
        """快速AABB相交测试"""
        return not (
            mesh_a.bounds_max[0] < mesh_b.bounds_min[0] or
            mesh_a.bounds_min[0] > mesh_b.bounds_max[0] or
            mesh_a.bounds_max[1] < mesh_b.bounds_min[1] or
            mesh_a.bounds_min[1] > mesh_b.bounds_max[1] or
            mesh_a.bounds_max[2] < mesh_b.bounds_min[2] or
            mesh_a.bounds_min[2] > mesh_b.bounds_max[2]
        )
    
    def _compute_penetration(
        self,
        mesh_i: MeshObject,
        mesh_j: MeshObject,
    ) -> Tuple[float, np.ndarray]:
        """
        快速计算两个mesh之间的穿透深度
        
        方法：采样mesh_i的表面点，检查它们到mesh_j表面的距离
        
        Returns:
            (penetration_depth, separation_normal)
        """
        # 采样mesh_i的表面点
        samples_i = mesh_i.mesh.sample(min(self.sample_points, len(mesh_i.mesh.vertices)))
        
        # 获取mesh_j的KD树
        kdtree_j = self._get_kdtree(mesh_j)
        
        # 查询最近距离
        distances, indices = kdtree_j.query(samples_i)
        
        # 近似判断内外：如果点到mesh_j表面很近，认为可能有穿透
        # 使用更简单的启发式：检查点是否在对方的AABB内
        penetrating_points = []
        for point, dist in zip(samples_i, distances):
            # 如果点在mesh_j的AABB内且距离很小，认为穿透
            in_aabb = (
                mesh_j.bounds_min[0] <= point[0] <= mesh_j.bounds_max[0] and
                mesh_j.bounds_min[1] <= point[1] <= mesh_j.bounds_max[1] and
                mesh_j.bounds_min[2] <= point[2] <= mesh_j.bounds_max[2]
            )
            if in_aabb and dist < self.tolerance * 3:  # 使用较宽松的阈值
                penetrating_points.append(point)
        
        if len(penetrating_points) == 0:
            return 0.0, np.zeros(3)
        
        # 计算穿透深度（使用质心距离作为近似）
        center_distance = np.linalg.norm(mesh_i.center - mesh_j.center)
        overlap = (mesh_i.size + mesh_j.size).max() / 2 - center_distance
        
        if overlap <= 0:
            return 0.0, np.zeros(3)
        
        # 分离方向：从j指向i
        direction = mesh_i.center - mesh_j.center
        normal = direction / (np.linalg.norm(direction) + 1e-8)
        
        # 穿透深度近似为重叠量和采样点比例的加权
        penetration_ratio = len(penetrating_points) / len(samples_i)
        penetration = overlap * penetration_ratio
        
        return penetration, normal
    
    def detect_collisions(self) -> List[Tuple[int, int, float, np.ndarray]]:
        """
        快速检测所有mesh对之间的碰撞
        
        Returns:
            碰撞列表：[(mesh_i_idx, mesh_j_idx, penetration_depth, collision_normal), ...]
        """
        LOG.info("开始快速碰撞检测...")
        collisions = []
        
        n = len(self.meshes)
        total_pairs = n * (n - 1) // 2
        checked_pairs = 0
        
        for i in range(n):
            mesh_i = self.meshes[i]
            
            for j in range(i + 1, n):
                mesh_j = self.meshes[j]
                checked_pairs += 1
                
                # 快速AABB测试
                if not self._aabb_intersect(mesh_i, mesh_j):
                    continue
                
                # 计算穿透（双向检测）
                penetration_ij, normal_ij = self._compute_penetration(mesh_i, mesh_j)
                penetration_ji, normal_ji = self._compute_penetration(mesh_j, mesh_i)
                
                # 取较大的穿透深度
                if penetration_ij > penetration_ji:
                    penetration = penetration_ij
                    normal = normal_ij
                else:
                    penetration = penetration_ji
                    normal = -normal_ji  # 反向
                
                if penetration > self.tolerance:
                    collisions.append((i, j, penetration, normal))
                    LOG.warning(
                        f"⚠️  碰撞: {mesh_i.filename} ↔ {mesh_j.filename}, "
                        f"穿透: {penetration:.4f}m"
                    )
                
                if checked_pairs % 100 == 0:
                    LOG.info(f"  检测进度: {checked_pairs}/{total_pairs} ({100*checked_pairs/total_pairs:.1f}%)")
        
        LOG.info(f"✓ 检测完成，发现 {len(collisions)} 处碰撞")
        return collisions
    
    def resolve_collisions_iterative(self, max_iterations: int = 50) -> None:
        """
        迭代式碰撞修复（快速投影方法）
        
        Args:
            max_iterations: 最大迭代次数
        """
        LOG.info(f"开始迭代修复（最多{max_iterations}次）...")
        
        for iteration in range(max_iterations):
            # 清除KD树缓存（因为位置更新了）
            self.kdtrees.clear()
            
            # 检测碰撞
            collisions = self.detect_collisions()
            
            if len(collisions) == 0:
                LOG.info(f"✓✓✓ 第 {iteration + 1} 次迭代后无碰撞！")
                break
            
            # 累积每个mesh的位移
            displacements = [np.zeros(3) for _ in self.meshes]
            counts = [0 for _ in self.meshes]
            
            # 按穿透深度排序，优先处理严重的碰撞
            collisions.sort(key=lambda x: x[2], reverse=True)
            
            for i, j, penetration, normal in collisions:
                # 计算分离距离（加上一点余量）
                separation = (penetration + self.tolerance) / 2
                
                # 累积位移
                displacements[i] += normal * separation
                displacements[j] -= normal * separation
                counts[i] += 1
                counts[j] += 1
            
            # 应用平均位移
            for idx, mesh_obj in enumerate(self.meshes):
                if counts[idx] > 0:
                    # 平均位移
                    displacement = displacements[idx] / counts[idx]
                    
                    # 应用位移
                    mesh_obj.mesh.vertices += displacement
                    mesh_obj.position += displacement
                    mesh_obj.bounds_min += displacement
                    mesh_obj.bounds_max += displacement
            
            LOG.info(f"迭代 {iteration + 1}/{max_iterations}, 修复了 {len(collisions)} 处碰撞")
        
        # 最终验证
        self.kdtrees.clear()
        final_collisions = self.detect_collisions()
        
        if len(final_collisions) == 0:
            LOG.info("✓✓✓ 所有碰撞已解决！")
        else:
            LOG.warning(f"仍有 {len(final_collisions)} 处碰撞（可能需要更多迭代或手动调整）")
    
    def export_meshes(self, output_dir: Path) -> None:
        """导出修复后的mesh"""
        LOG.info(f"导出mesh到 {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        transforms = {}
        
        for mesh_obj in self.meshes:
            # 导出mesh
            output_file = output_dir / mesh_obj.filename
            mesh_obj.mesh.export(str(output_file))
            
            # 记录变换
            transforms[mesh_obj.filename] = {
                'instance_id': mesh_obj.instance_id,
                'position': mesh_obj.position.tolist(),
                'displacement': (mesh_obj.position - mesh_obj.mesh.centroid).tolist(),
            }
            
            LOG.info(f"✓ 导出 {mesh_obj.filename}")
        
        # 保存变换信息
        transform_file = output_dir / "transforms.json"
        with open(transform_file, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        LOG.info(f"✓ 全部导出完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="快速SDF碰撞检测与修复")
    parser.add_argument("--input_dir", type=Path, default=Path("../world_mesh"))
    parser.add_argument("--output_dir", type=Path, default=Path("../world_mesh_fixed"))
    parser.add_argument("--tolerance", type=float, default=0.01, help="穿透容忍阈值（米）")
    parser.add_argument("--sample_points", type=int, default=500, help="每个mesh的采样点数")
    parser.add_argument("--iterations", type=int, default=50, help="最大迭代次数")
    parser.add_argument("--max_meshes", type=int, default=0, help="限制mesh数量（0=全部）")
    parser.add_argument("--detect_only", action="store_true", help="仅检测不修复")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    return parser.parse_args()


def main():
    args = parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")
    
    if not args.input_dir.exists():
        LOG.error(f"输入目录不存在: {args.input_dir}")
        return
    
    resolver = FastCollisionResolver(
        tolerance=args.tolerance,
        sample_points=args.sample_points,
    )
    
    start_time = time.time()
    
    # 加载mesh
    resolver.load_meshes(args.input_dir, max_meshes=args.max_meshes)
    
    if len(resolver.meshes) == 0:
        LOG.error("没有加载到任何mesh")
        return
    
    load_time = time.time() - start_time
    LOG.info(f"✓ 加载耗时: {load_time:.2f} 秒")
    
    # 检测碰撞
    detect_start = time.time()
    collisions = resolver.detect_collisions()
    detect_time = time.time() - detect_start
    LOG.info(f"✓ 检测耗时: {detect_time:.2f} 秒")
    
    if len(collisions) == 0:
        LOG.info("✓ 未检测到碰撞！")
    else:
        LOG.warning(f"检测到 {len(collisions)} 处碰撞")
        for i, j, penetration, normal in collisions[:10]:  # 只显示前10个
            LOG.info(
                f"  - {resolver.meshes[i].filename} ↔ {resolver.meshes[j].filename}: "
                f"{penetration:.4f}m"
            )
        if len(collisions) > 10:
            LOG.info(f"  ... 还有 {len(collisions) - 10} 处碰撞")
    
    if args.detect_only:
        elapsed = time.time() - start_time
        LOG.info(f"总耗时: {elapsed:.2f} 秒")
        return
    
    # 修复碰撞
    if len(collisions) > 0:
        fix_start = time.time()
        resolver.resolve_collisions_iterative(max_iterations=args.iterations)
        fix_time = time.time() - fix_start
        LOG.info(f"✓ 修复耗时: {fix_time:.2f} 秒")
    
    # 导出
    export_start = time.time()
    resolver.export_meshes(args.output_dir)
    export_time = time.time() - export_start
    LOG.info(f"✓ 导出耗时: {export_time:.2f} 秒")
    
    elapsed = time.time() - start_time
    LOG.info(f"✓✓✓ 全部完成！总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
