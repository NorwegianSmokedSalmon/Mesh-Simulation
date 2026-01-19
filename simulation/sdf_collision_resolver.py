"""
SDF-based Mesh Collision Resolver

使用有向距离场（Signed Distance Field）检测并解决mesh之间的穿模问题。

主要功能：
1. 加载world_mesh目录下的所有GLB mesh文件
2. 为每个mesh构建SDF表示
3. 检测mesh之间的碰撞/穿模
4. 使用物理仿真或优化方法调整mesh位置以消除穿模
5. 导出修复后的mesh

使用方法：
    python sdf_collision_resolver.py --input_dir ../world_mesh --output_dir ../world_mesh_fixed

作者：AI Assistant
日期：2026-01-18
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import json


LOG = logging.getLogger("sdf_collision_resolver")


@dataclass
class MeshObject:
    """表示一个带有变换的mesh对象"""
    mesh: trimesh.Trimesh
    filename: str
    instance_id: int
    position: np.ndarray  # 3D位置
    rotation: np.ndarray  # 3x3旋转矩阵
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    
    @property
    def center(self) -> np.ndarray:
        """返回mesh中心"""
        return (self.bounds_min + self.bounds_max) / 2
    
    @property
    def size(self) -> np.ndarray:
        """返回mesh尺寸"""
        return self.bounds_max - self.bounds_min
    
    def get_transformed_mesh(self) -> trimesh.Trimesh:
        """返回应用了当前变换的mesh"""
        mesh_copy = self.mesh.copy()
        # 应用旋转
        mesh_copy.apply_transform(
            np.vstack([
                np.hstack([self.rotation, [[0], [0], [0]]]),
                [0, 0, 0, 1]
            ])
        )
        # 应用平移
        mesh_copy.vertices += self.position
        return mesh_copy


class SDFCollisionResolver:
    """使用SDF方法解决mesh碰撞的主类"""
    
    def __init__(
        self,
        resolution: int = 64,
        padding: float = 0.1,
        tolerance: float = 0.01,
    ):
        """
        Args:
            resolution: SDF体素分辨率
            padding: 包围盒扩展比例
            tolerance: 穿透容忍阈值（米）
        """
        self.resolution = resolution
        self.padding = padding
        self.tolerance = tolerance
        self.meshes: List[MeshObject] = []
        self.sdf_cache = {}
        
    def load_meshes(self, input_dir: Path, max_meshes: int = 0) -> None:
        """加载目录下的所有GLB文件"""
        LOG.info(f"从 {input_dir} 加载mesh文件...")
        
        glb_files = sorted(input_dir.glob("*.glb"))
        if max_meshes > 0:
            glb_files = glb_files[:max_meshes]
        
        for glb_file in glb_files:
            try:
                # 提取instance ID
                instance_id = int(glb_file.stem.split("_")[1])
                
                # 加载mesh
                scene = trimesh.load(str(glb_file), force="mesh")
                if isinstance(scene, trimesh.Scene):
                    # 合并scene中的所有mesh
                    mesh = trimesh.util.concatenate([
                        trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                        for m in scene.geometry.values()
                        if isinstance(m, trimesh.Trimesh)
                    ])
                else:
                    mesh = scene
                
                if not isinstance(mesh, trimesh.Trimesh):
                    LOG.warning(f"跳过 {glb_file.name}: 不是有效的mesh")
                    continue
                
                # 检查mesh有效性
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    LOG.warning(f"跳过 {glb_file.name}: 空mesh")
                    continue
                
                # 修复mesh（如果需要）
                if not mesh.is_watertight:
                    LOG.debug(f"{glb_file.name} 不是watertight mesh")
                    # mesh.fill_holes()  # 可选：修复孔洞
                
                # 获取包围盒
                bounds = mesh.bounds
                center = mesh.centroid
                
                # 创建MeshObject
                mesh_obj = MeshObject(
                    mesh=mesh,
                    filename=glb_file.name,
                    instance_id=instance_id,
                    position=center,  # 初始位置为质心
                    rotation=np.eye(3),  # 初始无旋转
                    bounds_min=bounds[0],
                    bounds_max=bounds[1],
                )
                
                self.meshes.append(mesh_obj)
                LOG.info(f"✓ 加载 {glb_file.name}: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面片")
                
            except Exception as e:
                LOG.error(f"加载 {glb_file.name} 失败: {e}")
        
        LOG.info(f"总共加载 {len(self.meshes)} 个mesh")
    
    def compute_sdf(self, mesh_obj: MeshObject) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        为mesh计算SDF
        
        Returns:
            sdf_grid: 3D SDF值数组
            grid_min: 网格起始坐标
            grid_max: 网格结束坐标
        """
        if mesh_obj.instance_id in self.sdf_cache:
            return self.sdf_cache[mesh_obj.instance_id]
        
        LOG.info(f"计算 {mesh_obj.filename} 的SDF...")
        
        mesh = mesh_obj.get_transformed_mesh()
        
        # 扩展包围盒
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        padding = size * self.padding
        grid_min = bounds[0] - padding
        grid_max = bounds[1] + padding
        
        # 创建采样网格
        x = np.linspace(grid_min[0], grid_max[0], self.resolution)
        y = np.linspace(grid_min[1], grid_max[1], self.resolution)
        z = np.linspace(grid_min[2], grid_max[2], self.resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        query_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # 计算到表面的距离
        # 方法：使用最近点查询
        closest_points, distances, _ = trimesh.proximity.closest_point(mesh, query_points)
        
        # 确定符号（内部/外部）
        # 使用光线投射判断点是否在mesh内部
        LOG.debug("计算内外符号...")
        signs = np.ones(len(query_points))
        
        # 分批处理以节省内存
        batch_size = 10000
        for i in range(0, len(query_points), batch_size):
            batch = query_points[i:i+batch_size]
            # 简化方法：使用mesh的包含检测
            try:
                contains = mesh.contains(batch)
                signs[i:i+batch_size] = np.where(contains, -1, 1)
            except Exception as e:
                LOG.warning(f"内外判定失败: {e}，假定所有点在外部")
        
        # 计算有向距离
        sdf_values = signs * distances
        sdf_grid = sdf_values.reshape((self.resolution, self.resolution, self.resolution))
        
        # 缓存结果
        self.sdf_cache[mesh_obj.instance_id] = (sdf_grid, grid_min, grid_max)
        
        LOG.info(f"✓ SDF计算完成: 范围 [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
        
        return sdf_grid, grid_min, grid_max
    
    def query_sdf(
        self,
        point: np.ndarray,
        sdf_grid: np.ndarray,
        grid_min: np.ndarray,
        grid_max: np.ndarray,
    ) -> float:
        """
        查询空间中某点的SDF值（使用三线性插值）
        
        Args:
            point: 查询点坐标
            sdf_grid: SDF网格
            grid_min: 网格最小坐标
            grid_max: 网格最大坐标
        
        Returns:
            SDF值（正=外部，负=内部）
        """
        # 归一化到[0, resolution-1]
        normalized = (point - grid_min) / (grid_max - grid_min) * (self.resolution - 1)
        
        # 边界检查
        if np.any(normalized < 0) or np.any(normalized >= self.resolution - 1):
            # 点在网格外部，返回大正值
            return 1000.0
        
        # 三线性插值
        i, j, k = normalized.astype(int)
        di, dj, dk = normalized - np.array([i, j, k])
        
        # 8个角的SDF值
        c000 = sdf_grid[i, j, k]
        c001 = sdf_grid[i, j, k+1]
        c010 = sdf_grid[i, j+1, k]
        c011 = sdf_grid[i, j+1, k+1]
        c100 = sdf_grid[i+1, j, k]
        c101 = sdf_grid[i+1, j, k+1]
        c110 = sdf_grid[i+1, j+1, k]
        c111 = sdf_grid[i+1, j+1, k+1]
        
        # 三线性插值
        c00 = c000 * (1 - di) + c100 * di
        c01 = c001 * (1 - di) + c101 * di
        c10 = c010 * (1 - di) + c110 * di
        c11 = c011 * (1 - di) + c111 * di
        
        c0 = c00 * (1 - dj) + c10 * dj
        c1 = c01 * (1 - dj) + c11 * dj
        
        return c0 * (1 - dk) + c1 * dk
    
    def detect_collisions(self) -> List[Tuple[int, int, float, np.ndarray]]:
        """
        检测所有mesh对之间的碰撞
        
        Returns:
            碰撞列表：[(mesh_i_idx, mesh_j_idx, penetration_depth, collision_normal), ...]
        """
        LOG.info("开始检测碰撞...")
        collisions = []
        
        n = len(self.meshes)
        for i in range(n):
            mesh_i = self.meshes[i]
            
            # 计算mesh_i的SDF
            sdf_i, grid_min_i, grid_max_i = self.compute_sdf(mesh_i)
            
            for j in range(i + 1, n):
                mesh_j = self.meshes[j]
                
                # 快速AABB测试
                if not self._aabb_intersect(mesh_i, mesh_j):
                    continue
                
                # 计算mesh_j的SDF
                sdf_j, grid_min_j, grid_max_j = self.compute_sdf(mesh_j)
                
                # 检测mesh_i的顶点是否在mesh_j内部
                mesh_i_transformed = mesh_i.get_transformed_mesh()
                samples_i = mesh_i_transformed.sample(min(1000, len(mesh_i_transformed.vertices)))
                
                max_penetration = 0.0
                penetration_point = None
                
                for point in samples_i:
                    sdf_value = self.query_sdf(point, sdf_j, grid_min_j, grid_max_j)
                    if sdf_value < -self.tolerance:
                        penetration = -sdf_value
                        if penetration > max_penetration:
                            max_penetration = penetration
                            penetration_point = point
                
                if max_penetration > 0:
                    # 计算碰撞法向量（从j指向i）
                    direction = mesh_i.center - mesh_j.center
                    normal = direction / (np.linalg.norm(direction) + 1e-8)
                    
                    collisions.append((i, j, max_penetration, normal))
                    LOG.warning(
                        f"⚠️  碰撞检测: {mesh_i.filename} ↔ {mesh_j.filename}, "
                        f"穿透深度: {max_penetration:.4f}m"
                    )
        
        LOG.info(f"检测到 {len(collisions)} 处碰撞")
        return collisions
    
    def _aabb_intersect(self, mesh_a: MeshObject, mesh_b: MeshObject) -> bool:
        """检测两个AABB是否相交"""
        return not (
            mesh_a.bounds_max[0] < mesh_b.bounds_min[0] or
            mesh_a.bounds_min[0] > mesh_b.bounds_max[0] or
            mesh_a.bounds_max[1] < mesh_b.bounds_min[1] or
            mesh_a.bounds_min[1] > mesh_b.bounds_max[1] or
            mesh_a.bounds_max[2] < mesh_b.bounds_min[2] or
            mesh_a.bounds_min[2] > mesh_b.bounds_max[2]
        )
    
    def resolve_collisions_physical(
        self,
        iterations: int = 100,
        dt: float = 0.01,
        damping: float = 0.9,
    ) -> None:
        """
        使用物理仿真方法解决碰撞
        
        Args:
            iterations: 仿真迭代次数
            dt: 时间步长
            damping: 速度阻尼系数
        """
        LOG.info(f"开始物理仿真（{iterations}次迭代）...")
        
        # 初始化速度
        velocities = [np.zeros(3) for _ in self.meshes]
        
        for iteration in range(iterations):
            # 清空SDF缓存（因为位置更新了）
            self.sdf_cache.clear()
            
            # 检测碰撞
            collisions = self.detect_collisions()
            
            if len(collisions) == 0:
                LOG.info(f"✓ 第 {iteration + 1} 次迭代后无碰撞，提前结束")
                break
            
            # 计算排斥力
            forces = [np.zeros(3) for _ in self.meshes]
            
            for i, j, penetration, normal in collisions:
                # 弹簧-阻尼力模型
                k_spring = 100.0  # 弹簧系数
                k_damping = 10.0  # 阻尼系数
                
                # 相对速度
                relative_velocity = velocities[i] - velocities[j]
                
                # 排斥力
                force = k_spring * penetration * normal - k_damping * relative_velocity
                
                forces[i] += force
                forces[j] -= force
            
            # 更新速度和位置
            for idx, mesh_obj in enumerate(self.meshes):
                # 简化：假设所有mesh质量相同
                mass = 1.0
                
                # 更新速度
                velocities[idx] += forces[idx] / mass * dt
                velocities[idx] *= damping  # 应用阻尼
                
                # 更新位置
                mesh_obj.position += velocities[idx] * dt
                
                # 更新包围盒
                mesh_obj.bounds_min += velocities[idx] * dt
                mesh_obj.bounds_max += velocities[idx] * dt
            
            if (iteration + 1) % 10 == 0:
                LOG.info(f"迭代 {iteration + 1}/{iterations}, 剩余碰撞: {len(collisions)}")
        
        LOG.info("✓ 物理仿真完成")
    
    def resolve_collisions_projection(self) -> None:
        """
        使用约束投影方法解决碰撞（更简单直接）
        """
        LOG.info("使用投影方法解决碰撞...")
        
        max_iterations = 50
        for iteration in range(max_iterations):
            self.sdf_cache.clear()
            collisions = self.detect_collisions()
            
            if len(collisions) == 0:
                LOG.info(f"✓ 第 {iteration + 1} 次迭代后无碰撞")
                break
            
            # 按穿透深度排序，先处理最严重的
            collisions.sort(key=lambda x: x[2], reverse=True)
            
            for i, j, penetration, normal in collisions:
                # 分别移动两个mesh，各移动一半距离
                separation = (penetration + self.tolerance) / 2
                
                self.meshes[i].position += normal * separation
                self.meshes[j].position -= normal * separation
                
                # 更新包围盒
                self.meshes[i].bounds_min += normal * separation
                self.meshes[i].bounds_max += normal * separation
                self.meshes[j].bounds_min -= normal * separation
                self.meshes[j].bounds_max -= normal * separation
            
            LOG.info(f"迭代 {iteration + 1}/{max_iterations}, 处理了 {len(collisions)} 处碰撞")
        
        LOG.info("✓ 投影方法完成")
    
    def export_meshes(self, output_dir: Path) -> None:
        """导出修复后的mesh"""
        LOG.info(f"导出mesh到 {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出变换信息
        transforms = {}
        
        for mesh_obj in self.meshes:
            # 应用变换
            transformed_mesh = mesh_obj.get_transformed_mesh()
            
            # 保存GLB
            output_file = output_dir / mesh_obj.filename
            transformed_mesh.export(str(output_file))
            
            # 记录变换
            transforms[mesh_obj.filename] = {
                'instance_id': mesh_obj.instance_id,
                'position': mesh_obj.position.tolist(),
                'rotation': mesh_obj.rotation.tolist(),
            }
            
            LOG.info(f"✓ 导出 {mesh_obj.filename}")
        
        # 保存变换信息
        transform_file = output_dir / "transforms.json"
        with open(transform_file, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        LOG.info(f"✓ 全部导出完成，变换信息保存到 {transform_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用SDF检测并解决mesh穿模问题"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("../world_mesh"),
        help="输入mesh目录",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../world_mesh_fixed"),
        help="输出mesh目录",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="SDF网格分辨率（64/128/256）",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="穿透容忍阈值（米）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="projection",
        choices=["physical", "projection"],
        help="碰撞解决方法",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="最大迭代次数",
    )
    parser.add_argument(
        "--max_meshes",
        type=int,
        default=0,
        help="限制加载的mesh数量（0=全部）",
    )
    parser.add_argument(
        "--detect_only",
        action="store_true",
        help="仅检测碰撞，不修复",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    
    # 检查输入目录
    if not args.input_dir.exists():
        LOG.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 创建resolver
    resolver = SDFCollisionResolver(
        resolution=args.resolution,
        tolerance=args.tolerance,
    )
    
    # 加载mesh
    start_time = time.time()
    resolver.load_meshes(args.input_dir, max_meshes=args.max_meshes)
    
    if len(resolver.meshes) == 0:
        LOG.error("没有加载到任何mesh")
        return
    
    # 检测碰撞
    collisions = resolver.detect_collisions()
    
    if len(collisions) == 0:
        LOG.info("✓ 未检测到碰撞！")
    else:
        LOG.warning(f"检测到 {len(collisions)} 处碰撞")
        
        # 打印详细信息
        for i, j, penetration, normal in collisions:
            LOG.info(
                f"  - {resolver.meshes[i].filename} ↔ {resolver.meshes[j].filename}: "
                f"{penetration:.4f}m"
            )
    
    # 如果只是检测，到此结束
    if args.detect_only:
        elapsed = time.time() - start_time
        LOG.info(f"检测完成，耗时 {elapsed:.2f} 秒")
        return
    
    # 解决碰撞
    if len(collisions) > 0:
        if args.method == "physical":
            resolver.resolve_collisions_physical(iterations=args.iterations)
        else:  # projection
            resolver.resolve_collisions_projection()
        
        # 最终验证
        resolver.sdf_cache.clear()
        final_collisions = resolver.detect_collisions()
        
        if len(final_collisions) == 0:
            LOG.info("✓✓✓ 所有碰撞已解决！")
        else:
            LOG.warning(f"仍有 {len(final_collisions)} 处碰撞未完全解决")
    
    # 导出结果
    resolver.export_meshes(args.output_dir)
    
    elapsed = time.time() - start_time
    LOG.info(f"✓ 全部完成！总耗时 {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
