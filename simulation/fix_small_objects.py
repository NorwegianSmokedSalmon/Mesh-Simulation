"""
针对小物件（如杯子）的精细化穿模修复

特点：
1. 自适应采样：小物件使用更多采样点
2. 更严格的容忍度
3. 针对性的迭代修复

使用方法：
    python fix_small_objects.py --input_dir ../world_mesh_fixed --output_dir ../world_mesh_final
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import json


LOG = logging.getLogger("fix_small")


class SmallObjectFixer:
    """专门修复小物件之间的穿模"""
    
    def __init__(
        self,
        tolerance: float = 0.005,  # 更严格的容忍度
        size_threshold: float = 0.3,  # 小于此尺寸视为小物件
        dense_samples: int = 2000,  # 小物件采样点数
        normal_samples: int = 500,  # 普通物件采样点数
    ):
        self.tolerance = tolerance
        self.size_threshold = size_threshold
        self.dense_samples = dense_samples
        self.normal_samples = normal_samples
        self.meshes = []
        self.kdtrees = {}
    
    def load_meshes(self, input_dir: Path, max_meshes: int = 0) -> None:
        """加载mesh并分类"""
        LOG.info(f"从 {input_dir} 加载mesh文件...")
        
        from dataclasses import dataclass
        
        @dataclass
        class MeshObject:
            mesh: trimesh.Trimesh
            filename: str
            instance_id: int
            position: np.ndarray
            bounds_min: np.ndarray
            bounds_max: np.ndarray
            is_small: bool
            
            @property
            def center(self):
                return (self.bounds_min + self.bounds_max) / 2
            
            @property
            def size(self):
                return self.bounds_max - self.bounds_min
            
            @property
            def max_size(self):
                return self.size.max()
        
        glb_files = sorted(input_dir.glob("*.glb"))
        if max_meshes > 0:
            glb_files = glb_files[:max_meshes]
        
        small_count = 0
        large_count = 0
        
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
                size = bounds[1] - bounds[0]
                max_size = size.max()
                is_small = max_size < self.size_threshold
                
                mesh_obj = MeshObject(
                    mesh=mesh,
                    filename=glb_file.name,
                    instance_id=instance_id,
                    position=mesh.centroid.copy(),
                    bounds_min=bounds[0].copy(),
                    bounds_max=bounds[1].copy(),
                    is_small=is_small,
                )
                
                self.meshes.append(mesh_obj)
                
                if is_small:
                    small_count += 1
                    LOG.info(f"✓ [小] {glb_file.name}: {len(mesh.vertices)} 顶点, 尺寸={max_size:.3f}m")
                else:
                    large_count += 1
                    LOG.info(f"✓ [大] {glb_file.name}: {len(mesh.vertices)} 顶点, 尺寸={max_size:.3f}m")
                
            except Exception as e:
                LOG.error(f"加载 {glb_file.name} 失败: {e}")
        
        LOG.info(f"总共加载 {len(self.meshes)} 个mesh：{small_count} 个小物件，{large_count} 个大物件")
    
    def _get_kdtree(self, mesh_obj) -> cKDTree:
        """获取或创建KD树"""
        if mesh_obj.instance_id not in self.kdtrees:
            self.kdtrees[mesh_obj.instance_id] = cKDTree(mesh_obj.mesh.vertices)
        return self.kdtrees[mesh_obj.instance_id]
    
    def _aabb_intersect(self, mesh_a, mesh_b) -> bool:
        """AABB相交测试"""
        return not (
            mesh_a.bounds_max[0] < mesh_b.bounds_min[0] or
            mesh_a.bounds_min[0] > mesh_b.bounds_max[0] or
            mesh_a.bounds_max[1] < mesh_b.bounds_min[1] or
            mesh_a.bounds_min[1] > mesh_b.bounds_max[1] or
            mesh_a.bounds_max[2] < mesh_b.bounds_min[2] or
            mesh_a.bounds_min[2] > mesh_b.bounds_max[2]
        )
    
    def _compute_penetration(self, mesh_i, mesh_j) -> Tuple[float, np.ndarray]:
        """
        计算穿透深度，使用自适应采样
        """
        # 根据物件大小选择采样点数
        if mesh_i.is_small:
            n_samples = self.dense_samples
        else:
            n_samples = self.normal_samples
        
        # 采样
        samples_i = mesh_i.mesh.sample(min(n_samples, len(mesh_i.mesh.vertices) * 3))
        
        # KD树查询
        kdtree_j = self._get_kdtree(mesh_j)
        distances, _ = kdtree_j.query(samples_i)
        
        # 检测穿透
        penetrating_mask = np.zeros(len(samples_i), dtype=bool)
        
        for idx, (point, dist) in enumerate(zip(samples_i, distances)):
            # 更精确的AABB测试
            in_aabb = (
                mesh_j.bounds_min[0] - self.tolerance <= point[0] <= mesh_j.bounds_max[0] + self.tolerance and
                mesh_j.bounds_min[1] - self.tolerance <= point[1] <= mesh_j.bounds_max[1] + self.tolerance and
                mesh_j.bounds_min[2] - self.tolerance <= point[2] <= mesh_j.bounds_max[2] + self.tolerance
            )
            
            # 小物件使用更严格的阈值
            threshold = self.tolerance * (2 if mesh_i.is_small else 3)
            
            if in_aabb and dist < threshold:
                penetrating_mask[idx] = True
        
        n_penetrating = penetrating_mask.sum()
        
        if n_penetrating == 0:
            return 0.0, np.zeros(3)
        
        # 计算穿透深度
        penetrating_points = samples_i[penetrating_mask]
        penetrating_distances = distances[penetrating_mask]
        
        # 使用最大穿透深度
        max_penetration = penetrating_distances.max()
        
        # 加上基于重叠比例的权重
        penetration_ratio = n_penetrating / len(samples_i)
        
        # 计算分离方向
        direction = mesh_i.center - mesh_j.center
        if np.linalg.norm(direction) < 1e-8:
            # 如果中心重合，使用穿透点的平均方向
            direction = np.mean(penetrating_points - mesh_j.center, axis=0)
        
        normal = direction / (np.linalg.norm(direction) + 1e-8)
        
        # 综合穿透深度
        penetration = max_penetration * (0.5 + 0.5 * penetration_ratio)
        
        return penetration, normal
    
    def detect_collisions(self, focus_on_small: bool = True) -> List[Tuple[int, int, float, np.ndarray]]:
        """
        检测碰撞，可选择只关注小物件之间的碰撞
        """
        LOG.info("检测碰撞（关注小物件）..." if focus_on_small else "检测所有碰撞...")
        collisions = []
        
        n = len(self.meshes)
        checked = 0
        small_collisions = 0
        
        for i in range(n):
            mesh_i = self.meshes[i]
            
            for j in range(i + 1, n):
                mesh_j = self.meshes[j]
                checked += 1
                
                # 如果只关注小物件，跳过两个都是大物件的情况
                if focus_on_small and not (mesh_i.is_small or mesh_j.is_small):
                    continue
                
                # AABB测试
                if not self._aabb_intersect(mesh_i, mesh_j):
                    continue
                
                # 双向检测
                pen_ij, norm_ij = self._compute_penetration(mesh_i, mesh_j)
                pen_ji, norm_ji = self._compute_penetration(mesh_j, mesh_i)
                
                if pen_ij > pen_ji:
                    penetration, normal = pen_ij, norm_ij
                else:
                    penetration, normal = pen_ji, -norm_ji
                
                if penetration > self.tolerance:
                    collisions.append((i, j, penetration, normal))
                    
                    tag = ""
                    if mesh_i.is_small and mesh_j.is_small:
                        tag = "[小-小]"
                        small_collisions += 1
                    elif mesh_i.is_small or mesh_j.is_small:
                        tag = "[小-大]"
                    
                    LOG.warning(
                        f"⚠️  {tag} 碰撞: {mesh_i.filename} ↔ {mesh_j.filename}, "
                        f"穿透: {penetration:.4f}m"
                    )
                
                if checked % 100 == 0:
                    LOG.info(f"  检测进度: {checked}/{n*(n-1)//2}")
        
        LOG.info(f"✓ 检测完成，发现 {len(collisions)} 处碰撞（其中 {small_collisions} 处涉及小物件）")
        return collisions
    
    def resolve_collisions(self, max_iterations: int = 100) -> None:
        """迭代修复碰撞"""
        LOG.info(f"开始迭代修复（最多{max_iterations}次，专注小物件）...")
        
        for iteration in range(max_iterations):
            self.kdtrees.clear()
            
            # 优先检测小物件碰撞
            collisions = self.detect_collisions(focus_on_small=True)
            
            if len(collisions) == 0:
                LOG.info(f"✓✓✓ 第 {iteration + 1} 次迭代后无碰撞！")
                break
            
            # 累积位移
            displacements = [np.zeros(3) for _ in self.meshes]
            counts = [0 for _ in self.meshes]
            
            # 按穿透深度排序
            collisions.sort(key=lambda x: x[2], reverse=True)
            
            for i, j, penetration, normal in collisions:
                mesh_i = self.meshes[i]
                mesh_j = self.meshes[j]
                
                # 计算分离距离（加余量）
                separation = (penetration + self.tolerance * 1.5) / 2
                
                # 如果都是小物件，使用更大的分离距离
                if mesh_i.is_small and mesh_j.is_small:
                    separation *= 1.2
                
                # 累积位移
                displacements[i] += normal * separation
                displacements[j] -= normal * separation
                counts[i] += 1
                counts[j] += 1
            
            # 应用位移
            max_displacement = 0.0
            for idx, mesh_obj in enumerate(self.meshes):
                if counts[idx] > 0:
                    displacement = displacements[idx] / counts[idx]
                    
                    # 对小物件使用更激进的移动
                    if mesh_obj.is_small:
                        displacement *= 1.1
                    
                    mesh_obj.mesh.vertices += displacement
                    mesh_obj.position += displacement
                    mesh_obj.bounds_min += displacement
                    mesh_obj.bounds_max += displacement
                    
                    max_displacement = max(max_displacement, np.linalg.norm(displacement))
            
            LOG.info(
                f"迭代 {iteration + 1}/{max_iterations}, "
                f"处理 {len(collisions)} 处碰撞, "
                f"最大位移: {max_displacement:.4f}m"
            )
            
            # 如果位移很小，说明收敛了
            if max_displacement < self.tolerance * 0.1:
                LOG.info("位移收敛，提前结束")
                break
        
        # 最终验证
        self.kdtrees.clear()
        final_collisions = self.detect_collisions(focus_on_small=False)
        
        if len(final_collisions) == 0:
            LOG.info("✓✓✓ 所有碰撞已解决！")
        else:
            LOG.warning(f"仍有 {len(final_collisions)} 处碰撞")
    
    def export_meshes(self, output_dir: Path) -> None:
        """导出修复后的mesh"""
        LOG.info(f"导出mesh到 {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        transforms = {}
        
        for mesh_obj in self.meshes:
            output_file = output_dir / mesh_obj.filename
            mesh_obj.mesh.export(str(output_file))
            
            transforms[mesh_obj.filename] = {
                'instance_id': mesh_obj.instance_id,
                'position': mesh_obj.position.tolist(),
                'is_small': mesh_obj.is_small,
            }
            
            LOG.info(f"✓ 导出 {mesh_obj.filename}")
        
        transform_file = output_dir / "transforms_final.json"
        with open(transform_file, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        LOG.info("✓ 全部导出完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修复小物件穿模问题")
    parser.add_argument("--input_dir", type=Path, default=Path("../world_mesh_fixed"))
    parser.add_argument("--output_dir", type=Path, default=Path("../world_mesh_final"))
    parser.add_argument("--tolerance", type=float, default=0.005, help="容忍阈值（米）")
    parser.add_argument("--size_threshold", type=float, default=0.3, help="小物件尺寸阈值（米）")
    parser.add_argument("--dense_samples", type=int, default=2000, help="小物件采样点数")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--detect_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    
    if not args.input_dir.exists():
        LOG.error(f"输入目录不存在: {args.input_dir}")
        return
    
    fixer = SmallObjectFixer(
        tolerance=args.tolerance,
        size_threshold=args.size_threshold,
        dense_samples=args.dense_samples,
    )
    
    start_time = time.time()
    
    fixer.load_meshes(args.input_dir)
    
    if len(fixer.meshes) == 0:
        LOG.error("没有加载到任何mesh")
        return
    
    collisions = fixer.detect_collisions(focus_on_small=True)
    
    if args.detect_only:
        elapsed = time.time() - start_time
        LOG.info(f"检测完成，耗时 {elapsed:.2f} 秒")
        return
    
    if len(collisions) > 0:
        fixer.resolve_collisions(max_iterations=args.iterations)
    
    fixer.export_meshes(args.output_dir)
    
    elapsed = time.time() - start_time
    LOG.info(f"✓✓✓ 全部完成！总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
