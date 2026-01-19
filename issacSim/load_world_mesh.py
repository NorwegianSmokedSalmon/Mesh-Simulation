"""
Isaac Sim - 直接加载世界坐标系 GLB 文件

功能：
1. 扫描 world_mesh_final/ 目录下的所有 GLB 文件
2. 转换为 USD 格式
3. 自动检测地面高度（最小 z 值）
4. 为所有物体添加刚体和碰撞属性
5. 启动重力仿真

使用方法：
    python load_world_mesh.py --input_dir ../world_mesh_final

作者：AI Assistant
日期：2026-01-18
"""

import os
import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
from isaacsim import SimulationApp

# 启动 Isaac Sim
simulation_app = SimulationApp({"headless": False})

import omni
import omni.usd
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.prims import XFormPrim
from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdLux, Sdf, UsdShade
from omni.isaac.core.utils.extensions import enable_extension
import numpy as np

# 启用资产转换扩展
enable_extension("omni.kit.asset_converter")


async def convert_glb_to_usd(in_file, out_file, load_materials=True):
    """转换 GLB 到 USD，保持 Z-up 坐标系"""
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    
    # 关键：保持 Z-up 坐标系，不做转换
    converter_context.use_meter_as_world_unit = True
    converter_context.baking_scales = False
    
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def convert_asset(input_file, output_file):
    """同步转换接口，并修正坐标系"""
    print(f"[转换] {Path(input_file).name} -> USD...")
    status = asyncio.get_event_loop().run_until_complete(
        convert_glb_to_usd(input_file, output_file, load_materials=True)
    )
    if status:
        # 转换成功后，修正USD文件的坐标系
        if fix_usd_coordinate_system(output_file):
            print(f"  ✓ 转换成功，坐标系已修正为Z-up")
        else:
            print(f"  ⚠ 转换成功但坐标系修正失败")
        return True
    else:
        print(f"  ✗ 转换失败")
        return False


def fix_usd_coordinate_system(usd_file):
    """修正USD文件的坐标系为Z-up，统一单位，并预设碰撞近似"""
    try:
        stage = Usd.Stage.Open(usd_file)
        if not stage:
            print(f"  ⚠ 无法打开 USD 文件: {usd_file}")
            return False
        
        # 1. 设置舞台为Z-up
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        
        # 2. 统一单位为米（只在Stage层级，不在Prim层级）
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        
        # 3. 关键：为所有 Mesh 预设碰撞近似属性
        # 这样 PhysX 加载时就能看到正确的设置
        mesh_count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                try:
                    # 先应用 CollisionAPI schema（这是关键！）
                    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                    
                    # 然后设置 physics:approximation 属性
                    # 使用 convexHull 近似（动态物体必须使用近似碰撞）
                    # 注意：使用字符串值，不是 Token 对象
                    attr = prim.GetAttribute("physics:approximation")
                    if not attr or not attr.IsValid():
                        attr = prim.CreateAttribute(
                            "physics:approximation", 
                            Sdf.ValueTypeNames.Token,
                            custom=False  # 这是 USD 物理标准属性
                        )
                    # 设置为字符串 "convexHull"
                    attr.Set("convexHull")
                    mesh_count += 1
                except Exception as e:
                    print(f"    警告: 无法为 {prim.GetPath()} 设置碰撞近似: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 保存修改
        stage.Save()
        
        # 验证设置是否生效
        up_axis = UsdGeom.GetStageUpAxis(stage)
        
        del stage
        
        if up_axis != UsdGeom.Tokens.z:
            print(f"  ⚠ 坐标系设置失败: {up_axis}-up (期望 Z-up)")
            return False
        
        if mesh_count > 0:
            print(f"  ✓ 已为 {mesh_count} 个网格预设碰撞近似")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ 坐标系修正失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_bounding_box(stage, prim_path):
    """获取物体的包围盒"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    bbox = bbox_cache.ComputeWorldBound(prim)
    
    if bbox:
        bounds = bbox.ComputeAlignedBox()
        min_point = bounds.GetMin()
        max_point = bounds.GetMax()
        
        return {
            'min': [min_point[0], min_point[1], min_point[2]],
            'max': [max_point[0], max_point[1], max_point[2]],
            'center': [
                (max_point[0] + min_point[0]) / 2,
                (max_point[1] + min_point[1]) / 2,
                (max_point[2] + min_point[2]) / 2
            ],
            'size': [
                max_point[0] - min_point[0],
                max_point[1] - min_point[1],
                max_point[2] - min_point[2]
            ]
        }
    return None


def add_physics_to_object(stage, prim_path, use_convex_hull=True):
    """
    为物体添加物理属性
    
    Args:
        stage: USD Stage
        prim_path: Prim 路径
        use_convex_hull: 是否使用凸包碰撞（False 则使用包围盒）
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"  ✗ 无效的 prim: {prim_path}")
        return False
    
    # 1. 添加刚体 API（动态刚体，受重力影响）
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    
    # 显式禁用 kinematic 模式，使物体变为动态（受重力影响）
    rigid_body_api.CreateRigidBodyEnabledAttr().Set(True)
    rigid_body_api.CreateKinematicEnabledAttr().Set(False)  # False = 动态物体
    
    # 2. 添加碰撞 API
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    
    # 3. 设置碰撞近似（凸包或包围盒）
    # 直接使用字符串值而不是 Token 对象
    approx_value = "convexHull" if use_convex_hull else "boundingCube"
    
    # 为所有子网格设置碰撞近似（这是关键！）
    count = 0
    for child_prim in Usd.PrimRange(prim):
        if child_prim.IsA(UsdGeom.Mesh):
            # 应用碰撞 API
            child_collision = UsdPhysics.CollisionAPI.Apply(child_prim)
            
            # 直接设置 physics:approximation 属性
            try:
                attr = child_prim.GetAttribute("physics:approximation")
                if not attr or not attr.IsValid():
                    attr = child_prim.CreateAttribute(
                        "physics:approximation",
                        Sdf.ValueTypeNames.Token
                    )
                attr.Set(approx_value)
                count += 1
            except Exception as e:
                print(f"    警告: 无法为 {child_prim.GetPath()} 设置碰撞近似: {e}")
    
    collision_type = "凸包" if use_convex_hull else "包围盒"
    if count > 0:
        print(f"  ✓ 已添加物理属性: 刚体 + 碰撞({collision_type}, {count}个网格)")
    else:
        print(f"  ⚠️  警告: 没有找到子网格，可能碰撞设置失败")
    return True


def scan_glb_files(input_dir, max_files=0):
    """扫描目录下的所有 GLB 文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 目录不存在 {input_dir}")
        return []
    
    glb_files = sorted(input_path.glob("*.glb"))
    
    if max_files > 0 and len(glb_files) > max_files:
        glb_files = glb_files[:max_files]
        print(f"[扫描] 找到 {len(list(input_path.glob('*.glb')))} 个 GLB 文件，限制加载前 {max_files} 个")
    else:
        print(f"[扫描] 找到 {len(glb_files)} 个 GLB 文件")
    
    return glb_files


def load_world_mesh(input_dir, usd_cache_dir, stage, world, enable_physics=True, max_objects=0):
    """
    加载世界坐标系的 mesh 到 Isaac Sim
    
    流程:
    1. 扫描并转换所有 GLB -> USD
    2. 计算地面高度（所有物体 z_min 的最小值）
    3. 加载所有物体到场景
    4. 添加物理属性（可选）
    
    Args:
        enable_physics: 是否启用物理属性（默认True）
        max_objects: 最大加载物体数量（0=全部）
    """
    
    # 1. 扫描 GLB 文件
    glb_files = scan_glb_files(input_dir, max_files=max_objects)
    if len(glb_files) == 0:
        print("错误: 没有找到 GLB 文件")
        return False
    
    # 2. 创建 USD 缓存目录
    usd_cache_path = Path(usd_cache_dir)
    usd_cache_path.mkdir(parents=True, exist_ok=True)
    
    # 3. 转换所有 GLB -> USD
    print(f"\n{'='*60}")
    print("步骤 1: 转换 GLB -> USD")
    print(f"{'='*60}")
    
    usd_files = []
    for glb_file in glb_files:
        usd_file = usd_cache_path / f"{glb_file.stem}.usd"
        
        # 如果 USD 不存在或 GLB 更新了，则转换
        if not usd_file.exists() or glb_file.stat().st_mtime > usd_file.stat().st_mtime:
            if convert_asset(str(glb_file), str(usd_file)):
                usd_files.append((glb_file.stem, str(usd_file)))
        else:
            print(f"[缓存] {glb_file.name} (使用已有USD)")
            usd_files.append((glb_file.stem, str(usd_file)))
    
    if len(usd_files) == 0:
        print("错误: 没有成功转换任何文件")
        return False
    
    # 4. 使用更简单的方法计算地面高度（不预加载）
    print(f"\n{'='*60}")
    print("步骤 2: 计算地面高度")
    print(f"{'='*60}")
    
    # 直接从 USD 文件读取包围盒，不加载到主场景
    z_min_global = float('inf')
    
    for name, usd_path in usd_files:
        try:
            # 打开 USD 文件但不添加到主场景
            temp_stage = Usd.Stage.Open(usd_path)
            if temp_stage:
                # 遍历所有 mesh
                for prim in temp_stage.Traverse():
                    if prim.IsA(UsdGeom.Mesh):
                        mesh = UsdGeom.Mesh(prim)
                        points_attr = mesh.GetPointsAttr()
                        if points_attr:
                            points = points_attr.Get()
                            if points:
                                for point in points:
                                    if point[2] < z_min_global:
                                        z_min_global = point[2]
                # 关闭临时 stage
                del temp_stage
                print(f"  分析: {name}")
        except Exception as e:
            print(f"  警告: 无法分析 {name}: {e}")
    
    ground_z = z_min_global if z_min_global != float('inf') else 0.0
    print(f"\n  ✓ 地面高度设定为: z = {ground_z:.4f}m")
    
    # 5. 第二遍加载：正式放置物体
    print(f"\n{'='*60}")
    print("步骤 3: 加载物体到场景")
    print(f"{'='*60}")
    
    successful_count = 0
    objects_info = []
    
    for name, usd_path in usd_files:
        print(f"\n[加载] {name}")
        
        # 清理名称，创建合法的 prim 路径
        safe_name = name.replace('-', '_').replace('.', '_')
        prim_path = f"/World/{safe_name}"
        
        try:
            # 加载 USD
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            omni.kit.app.get_app().update()
            
            # 获取加载的prim
            prim = stage.GetPrimAtPath(prim_path)
            
            # 获取包围盒
            bbox = get_bounding_box(stage, prim_path)
            if bbox:
                x_range = bbox['size'][0]
                y_range = bbox['size'][1]
                z_range = bbox['size'][2]
                
                print(f"  位置: [{bbox['center'][0]:.3f}, {bbox['center'][1]:.3f}, {bbox['center'][2]:.3f}]")
                print(f"  尺寸: X={x_range:.3f}, Y={y_range:.3f}, Z={z_range:.3f} m")
                
                # 记录信息
                objects_info.append({
                    'name': name,
                    'prim_path': prim_path,
                    'bbox': bbox
                })
            
            # 添加物理属性（刚体 + 碰撞）- 如果启用的话
            if enable_physics:
                # 注意：使用包围盒碰撞更快，凸包更精确但慢
                # 对于大场景，建议使用包围盒
                use_convex = len(usd_files) < 20  # 少于20个物体才用凸包
                add_physics_to_object(stage, prim_path, use_convex_hull=use_convex)
            
            successful_count += 1
            print(f"  ✓ 成功加载")
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✓ 成功加载 {successful_count}/{len(usd_files)} 个物体")
    print(f"{'='*60}")
    
    # 6. 保存 Stage（让物理属性生效）
    if enable_physics:
        print("\n[保存] 保存物理属性到 USD Stage...")
        try:
            stage.Save()
            print("  ✓ Stage 已保存")
        except Exception as e:
            print(f"  警告: 无法保存 Stage: {e}")
    
    # 7. 保存场景信息
    scene_info = {
        'ground_z': ground_z,
        'object_count': successful_count,
        'objects': objects_info
    }
    
    info_file = Path(input_dir) / "isaac_scene_info.json"
    with open(info_file, 'w') as f:
        json.dump(scene_info, f, indent=2)
    print(f"\n场景信息已保存到: {info_file}")
    
    # 返回成功标志和地面高度
    return True, ground_z


def create_checkerboard_ground(stage, ground_z, grid_size=1.0, grid_count=100):
    """
    创建带网格的地面（使用 Isaac Sim 内置功能）
    
    Args:
        stage: USD Stage
        ground_z: 地面高度 (m)
        grid_size: 格子大小 (m)
        grid_count: 格子数量（每边）
    """
    from pxr import UsdGeom, Gf, UsdShade
    import omni.isaac.core.utils.prims as prims_utils
    
    print(f"\n[地面] 创建网格地面于 z = {ground_z:.4f}m")
    
    try:
        # 尝试使用 Isaac Sim 的默认网格地面
        from omni.isaac.core.utils.stage import add_reference_to_stage
        
        # 创建地面 Xform
        ground_path = "/World/GroundPlane"
        ground_xform = UsdGeom.Xform.Define(stage, ground_path)
        ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, ground_z))
        
        # 使用 Isaac Sim 的网格地面资产
        # 注意：需要确认 Isaac Sim 版本和资产路径
        grid_plane_path = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Environments/Grid/gridroom_curved.usd"
        
        # 尝试添加引用
        try:
            add_reference_to_stage(usd_path=grid_plane_path, prim_path=ground_path + "/GridPlane")
            print(f"  ✓ 使用 Isaac Sim 内置网格地面")
            return True
        except:
            pass
            
    except Exception as e:
        print(f"  ⚠️  无法使用内置网格地面: {e}")
    
    # 回退方案：创建简单的带颜色的平面
    print(f"  使用自定义浅灰色地面")
    
    ground_prim_path = "/World/GroundPlane"
    ground_geom = UsdGeom.Mesh.Define(stage, ground_prim_path)
    
    # 创建大平面
    half_size = (grid_count * grid_size) / 2.0
    vertices = [
        Gf.Vec3f(-half_size, -half_size, ground_z),
        Gf.Vec3f(half_size, -half_size, ground_z),
        Gf.Vec3f(half_size, half_size, ground_z),
        Gf.Vec3f(-half_size, half_size, ground_z),
    ]
    
    ground_geom.GetPointsAttr().Set(vertices)
    ground_geom.GetFaceVertexCountsAttr().Set([4])
    ground_geom.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    
    normals = [Gf.Vec3f(0, 0, 1)] * 4
    ground_geom.GetNormalsAttr().Set(normals)
    ground_geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    
    # 创建浅灰色材质（不会在仿真后消失）
    material_path = "/World/Looks/GroundMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    binding_api = UsdShade.MaterialBindingAPI.Apply(ground_geom.GetPrim())
    binding_api.Bind(material)
    
    # 添加碰撞
    UsdPhysics.CollisionAPI.Apply(ground_geom.GetPrim())
    
    print(f"  ✓ 地面已创建（浅灰色材质）")
    return True


def add_ground_plane(stage, ground_z):
    """
    在指定高度添加地面平面（使用 Isaac Sim 推荐的方式）
    
    Args:
        stage: USD Stage
        ground_z: 地面高度 (m)
    """
    from pxr import UsdGeom, Gf, UsdShade
    from omni.isaac.core.utils.prims import create_prim
    
    print(f"\n[地面] 创建地面平面于 z = {ground_z:.4f}m")
    
    # 使用 Isaac Sim 的 create_prim 创建地面
    ground_prim_path = "/World/GroundPlane"
    
    # 创建一个 Cube 作为地面（拉伸成薄片）
    size = 100.0
    ground_prim = create_prim(
        prim_path=ground_prim_path,
        prim_type="Cube",
        position=np.array([0, 0, ground_z - 0.05]),  # 稍微下移，厚度 0.1m
        scale=np.array([size, size, 0.1]),  # 100m x 100m x 0.1m
        attributes={
            "primvars:displayColor": [(0.7, 0.75, 0.8)],  # 偏蓝灰色
            "primvars:displayOpacity": [1.0],
        }
    )
    
    # 添加碰撞
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    
    # 创建并绑定持久化材质
    material_path = "/World/Looks/GroundMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    
    # 深灰色，确保可见
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.55))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    # 强制绑定材质
    binding_api = UsdShade.MaterialBindingAPI.Apply(ground_prim)
    binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
    
    print(f"  ✓ 地面已创建（深灰色，强制材质绑定）")
    print(f"     高度: z = {ground_z:.4f}m")
    print(f"     大小: {size*2}m x {size*2}m x 0.1m")
    return True


def add_wall_for_object(stage, object_name, direction="y+", wall_height=3.0, wall_thickness=0.1, wall_extend=1.0, color=(1.0, 0.8, 0.0)):
    """
    为指定物体添加墙壁（带碰撞，基于包围盒）
    
    Args:
        stage: USD Stage
        object_name: 物体名称（例如 "instance_19_refine_world"）
        direction: 墙壁方向 ("y+", "y-", "x+", "x-")
        wall_height: 墙壁高度（米）
        wall_thickness: 墙壁厚度（米）
        wall_extend: 墙壁在两侧延伸的距离（米）
        color: 墙壁颜色 RGB（默认黄色）
    """
    from pxr import UsdGeom, Gf, UsdShade
    from omni.isaac.core.utils.prims import create_prim
    
    print(f"\n[墙壁] 为 {object_name} 添加墙壁...")
    
    # 获取物体的 prim
    object_path = f"/World/{object_name}"
    object_prim = stage.GetPrimAtPath(object_path)
    
    if not object_prim or not object_prim.IsValid():
        print(f"  ✗ 找不到物体: {object_name}")
        return False
    
    # 获取物体的包围盒
    bbox = get_bounding_box(stage, object_path)
    if not bbox:
        print(f"  ✗ 无法获取物体包围盒")
        return False
    
    # 计算墙壁位置和尺寸
    obj_min = bbox['min']
    obj_max = bbox['max']
    obj_center = bbox['center']
    obj_size = bbox['size']
    
    # 根据方向计算墙壁参数（基于包围盒）
    if direction == "y+":
        # Y 轴正方向（物体的 +Y 侧）
        wall_width = obj_size[0] + wall_extend * 2  # X 方向宽度
        wall_height_actual = wall_height
        wall_depth = wall_thickness
        
        wall_pos = [
            obj_center[0],  # X: 与物体中心对齐
            obj_max[1] + wall_thickness / 2,  # Y: 紧贴物体 +Y 侧
            obj_min[2] + wall_height / 2  # Z: 从物体底部开始
        ]
        wall_scale = [wall_width, wall_depth, wall_height_actual]
        
    elif direction == "y-":
        wall_width = obj_size[0] + wall_extend * 2
        wall_height_actual = wall_height
        wall_depth = wall_thickness
        
        wall_pos = [
            obj_center[0],
            obj_min[1] - wall_thickness / 2,
            obj_min[2] + wall_height / 2
        ]
        wall_scale = [wall_width, wall_depth, wall_height_actual]
        
    elif direction == "x+":
        wall_width = obj_size[1] + wall_extend * 2  # Y 方向宽度
        wall_height_actual = wall_height
        wall_depth = wall_thickness
        
        wall_pos = [
            obj_max[0] + wall_thickness / 2,
            obj_center[1],
            obj_min[2] + wall_height / 2
        ]
        wall_scale = [wall_depth, wall_width, wall_height_actual]
        
    elif direction == "x-":
        wall_width = obj_size[1] + wall_extend * 2
        wall_height_actual = wall_height
        wall_depth = wall_thickness
        
        wall_pos = [
            obj_min[0] - wall_thickness / 2,
            obj_center[1],
            obj_min[2] + wall_height / 2
        ]
        wall_scale = [wall_depth, wall_width, wall_height_actual]
    else:
        print(f"  ✗ 未知方向: {direction}")
        return False
    
    # 创建墙壁（使用 Cube）
    # 转换方向名称为合法的 USD 路径（不能包含 + 或 -）
    direction_name = direction.replace("+", "pos").replace("-", "neg")
    wall_path = f"/World/Walls/Wall_{object_name}_{direction_name}"
    wall_prim = create_prim(
        prim_path=wall_path,
        prim_type="Cube",
        position=np.array(wall_pos),
        scale=np.array(wall_scale),
        attributes={
            "primvars:displayColor": [color],
        }
    )
    
    # 添加碰撞
    UsdPhysics.CollisionAPI.Apply(wall_prim)
    
    # 创建材质
    material_path = f"/World/Looks/WallMaterial_{object_name}"
    material = UsdShade.Material.Define(stage, material_path)
    
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    # 绑定材质
    binding_api = UsdShade.MaterialBindingAPI.Apply(wall_prim)
    binding_api.Bind(material)
    
    print(f"  ✓ 墙壁已创建")
    print(f"     位置: [{wall_pos[0]:.2f}, {wall_pos[1]:.2f}, {wall_pos[2]:.2f}]")
    print(f"     尺寸: {wall_scale[0]:.2f} x {wall_scale[1]:.2f} x {wall_scale[2]:.2f} m")
    print(f"     颜色: 黄色")
    return True


def add_grid_lines(stage, ground_z, grid_size=1.0, grid_range=50):
    """
    在地面上添加网格线（帮助看清物体位置）
    
    Args:
        stage: USD Stage
        ground_z: 地面高度
        grid_size: 网格大小（米）
        grid_range: 网格范围（米）
    """
    from pxr import UsdGeom, Gf
    
    print(f"\n[网格] 添加地面网格线...")
    
    # 创建网格线组
    grid_xform_path = "/World/GridLines"
    grid_xform = UsdGeom.Xform.Define(stage, grid_xform_path)
    
    line_count = 0
    # 创建 X 方向的线
    for i in range(-grid_range, grid_range + 1):
        if i == 0:
            continue  # 跳过中心线，避免重复
        y = i * grid_size
        
        # 创建线段（负数用 n 表示，避免路径错误）
        line_name = f"LineX_n{abs(i)}" if i < 0 else f"LineX_{i}"
        line_path = f"{grid_xform_path}/{line_name}"
        line_geom = UsdGeom.BasisCurves.Define(stage, line_path)
        
        # 设置点
        points = [
            Gf.Vec3f(-grid_range * grid_size, y, ground_z + 0.001),
            Gf.Vec3f(grid_range * grid_size, y, ground_z + 0.001),
        ]
        line_geom.GetPointsAttr().Set(points)
        line_geom.GetCurveVertexCountsAttr().Set([2])
        line_geom.GetTypeAttr().Set("linear")
        
        # 设置线条颜色为深灰色
        line_geom.CreateDisplayColorAttr([(0.3, 0.3, 0.3)])
        line_geom.CreateWidthsAttr([0.01])
        line_count += 1
    
    # 创建 Y 方向的线
    for i in range(-grid_range, grid_range + 1):
        x = i * grid_size
        
        # 负数用 n 表示
        line_name = f"LineY_n{abs(i)}" if i < 0 else f"LineY_{i}"
        line_path = f"{grid_xform_path}/{line_name}"
        line_geom = UsdGeom.BasisCurves.Define(stage, line_path)
        
        points = [
            Gf.Vec3f(x, -grid_range * grid_size, ground_z + 0.001),
            Gf.Vec3f(x, grid_range * grid_size, ground_z + 0.001),
        ]
        line_geom.GetPointsAttr().Set(points)
        line_geom.GetCurveVertexCountsAttr().Set([2])
        line_geom.GetTypeAttr().Set("linear")
        
        line_geom.CreateDisplayColorAttr([(0.3, 0.3, 0.3)])
        line_geom.CreateWidthsAttr([0.01])
        line_count += 1
    
    print(f"  ✓ 已添加 {line_count} 条网格线（{grid_size}m 间距）")
    return True


def add_lighting(stage):
    """添加光照"""
    print(f"\n{'='*60}")
    print("添加光照")
    print(f"{'='*60}")
    
    # 1. 环境光（圆顶光）
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000)
    print("  ✓ 圆顶光: 强度 1000")
    
    # 2. 定向光（从上方照射，模拟太阳）
    distant_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    distant_light.CreateIntensityAttr(500)
    distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.95))  # 微黄色
    
    # 设置光源方向（从 z 轴正方向向下）
    xform = UsdGeom.Xformable(distant_light.GetPrim())
    xform.ClearXformOpOrder()
    
    # 旋转光源，使其从上往下照
    rotate_op = xform.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3f(90, 0, 0))  # 绕 X 轴旋转 90 度
    
    print("  ✓ 定向光: 强度 500, 从上往下照射")


def main():
    parser = argparse.ArgumentParser(description="加载世界坐标系 GLB 文件到 Isaac Sim")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../world_mesh_final",
        help="输入目录（包含 GLB 文件）"
    )
    parser.add_argument(
        "--usd_cache",
        type=str,
        default="./usd_cache",
        help="USD 缓存目录"
    )
    parser.add_argument(
        "--simulation_time",
        type=int,
        default=5,
        help="仿真时长（秒）- 可以随时按 Ctrl+C 退出"
    )
    parser.add_argument(
        "--no_physics",
        action="store_true",
        help="仅加载场景，不启用物理仿真"
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=10,
        help="最大加载物体数量（用于测试，0=全部）"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Isaac Sim - 世界坐标系 Mesh 加载器")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"USD 缓存: {args.usd_cache}")
    print(f"仿真时长: {args.simulation_time}秒")
    if args.max_objects > 0:
        print(f"限制加载: 前 {args.max_objects} 个物体")
    else:
        print(f"加载模式: 全部物体")
    print("="*60)
    
    try:
        # 初始化物理世界
        print("\n[初始化] 创建物理世界...")
        my_world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene")
        stage = omni.usd.get_context().get_stage()
        
        # 设置舞台坐标系为 Z-up（与 GLB 一致）
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        
        # 显式设置重力方向为 Z 轴负方向（Z-up 坐标系）
        physics_scene = UsdPhysics.Scene.Get(stage, "/World/physicsScene")
        if physics_scene:
            # 设置重力为 (0, 0, -9.81) m/s²
            physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
            print(f"  ✓ 重力设置: (0, 0, -9.81) m/s²")
        
        current_up_axis = UsdGeom.GetStageUpAxis(stage)
        print(f"  ✓ 世界单位: {get_stage_units()} 米/单位")
        print(f"  ✓ 坐标系: {current_up_axis}-up")
        
        # 加载所有 mesh
        enable_physics = not args.no_physics
        if enable_physics:
            print("\n物理仿真: 已启用 ✓")
        else:
            print("\n物理仿真: 已禁用 (仅查看场景)")
        
        result = load_world_mesh(
            args.input_dir, 
            args.usd_cache, 
            stage, 
            my_world, 
            enable_physics=enable_physics,
            max_objects=args.max_objects
        )
        
        if not result or result is False:
            print("\n加载失败，退出")
            return
        
        # 解析返回值
        if isinstance(result, tuple):
            success, ground_z = result
        else:
            success = result
            ground_z = 0.0
        
        # 在检测到的地面高度创建地面平面
        print(f"\n[地面] 添加地面平面于 z = {ground_z:.4f}m")
        
        # 使用 Isaac Sim 的 GroundPlane 类（带网格）
        from omni.isaac.core.objects import GroundPlane
        ground_plane = GroundPlane(
            prim_path="/World/GroundPlane",
            size=100.0,
            color=np.array([0.5, 0.5, 0.5]),  # 灰色
        )
        ground_plane.set_world_pose(position=np.array([0, 0, ground_z]))
        
        # 启用网格显示（帮助看清物体位置）
        try:
            import omni.kit.viewport.utility as vp_utils
            viewport_api = vp_utils.get_active_viewport()
            if viewport_api:
                # 启用网格显示
                viewport_api.legacy_window.set_active_camera("/OmniverseKit_Persp")
                # 启用网格
                import carb
                settings = carb.settings.get_settings()
                settings.set("/persistent/app/viewport/displayOptions", 31)  # 启用网格
                settings.set("/persistent/app/viewport/grid/enabled", True)
                settings.set("/persistent/app/viewport/grid/scale", 1.0)  # 1米网格
                print(f"  ✓ 已启用视口网格显示（1m 网格）")
        except Exception as e:
            print(f"  ⚠️  网格显示设置失败: {e}")
        
        print(f"  ✓ 已添加 GroundPlane（灰色，z={ground_z:.4f}m）")
        
        # 添加网格线（帮助看清物体位置）
        add_grid_lines(stage, ground_z, grid_size=1.0, grid_range=50)
        # 为 instance_19 添加定制墙壁
        print("\n[墙壁] 为 instance_19 添加定制墙壁...")
        from omni.isaac.core.utils.prims import create_prim
        import math
        
        # 墙壁参数（基于手动调整的 Transform）
        wall_position = [-1.49, 1.47718, ground_z + 1.5]  # Z 从地面开始，墙高一半
        wall_scale = [4.0, 0.0005, 1.5]  # 宽度 x 厚度 x 高度
        wall_rotation_z_deg = -16.75  # 绕 Z 轴旋转角度（度）
        wall_color = (0.5, 0.5, 0.99)  # 黄色
        
        # 将旋转角度转换为四元数
        wall_rotation_z_rad = math.radians(wall_rotation_z_deg)
        # 绕 Z 轴旋转的四元数: (w, x, y, z)
        qw = math.cos(wall_rotation_z_rad / 2)
        qx = 0.0
        qy = 0.0
        qz = math.sin(wall_rotation_z_rad / 2)
        
        # 创建墙壁（使用四元数设置旋转）
        wall_path = "/World/Walls/Wall_instance19_custom"
        wall_prim = create_prim(
            prim_path=wall_path,
            prim_type="Cube",
            position=np.array(wall_position),
            orientation=np.array([qw, qx, qy, qz]),  # 四元数旋转
            scale=np.array(wall_scale),
            attributes={
                "primvars:displayColor": [wall_color],
            }
        )
        
        # 添加碰撞（仅 CollisionAPI，无 RigidBodyAPI，因此是静态墙壁，不会移动）
        UsdPhysics.CollisionAPI.Apply(wall_prim)
        
        # 创建材质
        material_path = "/World/Looks/WallMaterial_instance19"
        material = UsdShade.Material.Define(stage, material_path)
        
        shader_path = material_path + "/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*wall_color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # 绑定材质
        binding_api = UsdShade.MaterialBindingAPI.Apply(wall_prim)
        binding_api.Bind(material)
        
        print(f"  ✓ 定制墙壁已创建")
        print(f"     位置: [{wall_position[0]:.2f}, {wall_position[1]:.2f}, {wall_position[2]:.2f}]")
        print(f"     尺寸: {wall_scale[0]:.2f} x {wall_scale[1]:.2f} x {wall_scale[2]:.2f} m")
        print(f"     旋转: {wall_rotation_z_deg:.1f}°")
        print(f"     颜色: 黄色")
        
        # 添加光照
        add_lighting(stage)
        
        # **重要**: 重置世界以初始化物理引擎
        print("\n[初始化] 重置物理引擎...")
        my_world.reset()
        print("  ✓ 物理引擎已初始化")
        
        # 暂停物理仿真，先让用户查看场景
        if enable_physics:
            my_world.pause()
            print("  ✓ 物理仿真已暂停（等待用户启动）")
        
        # 场景加载完成
        print(f"\n{'='*60}")
        print("场景加载完成！")
        print(f"{'='*60}")
        print("📋 操作指南:")
        print("  - 使用鼠标拖拽旋转视角")
        print("  - 滚轮缩放")
        print("  - 中键拖拽平移")
        print(f"{'='*60}\n")
        
        # 交互式启动控制
        if enable_physics:
            print("🎮 物理仿真控制:")
            print("  场景已加载，您可以自由拖动观察")
            print("  [Enter]  - 启动重力仿真")
            print("  [Ctrl+C] - 退出程序")
            print()
            
            # 非阻塞等待，保持渲染更新
            import select
            import sys
            
            print("按 Enter 键启动重力仿真（期间可自由观察场景）...")
            try:
                waiting = True
                while waiting:
                    # 更新渲染，保持界面可交互
                    omni.kit.app.get_app().update()
                    time.sleep(0.016)  # ~60 FPS
                    
                    # 检查是否有输入（非阻塞）
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline()
                        waiting = False
            except KeyboardInterrupt:
                print("\n\n用户取消，退出程序")
                return
            
            # 用户确认，启动仿真
            my_world.play()
            print(f"\n{'='*60}")
            print("🚀 重力仿真已启动！")
            print(f"{'='*60}")
            print("提示: 观察物体在重力作用下的行为")
            print("提示: 按 Ctrl+C 可随时退出仿真\n")
        
        if enable_physics:
            # 有物理仿真
            steps_per_second = 60
            total_steps = args.simulation_time * steps_per_second
            
            try:
                for i in range(total_steps):
                    my_world.step(render=True)
                    
                    if (i + 1) % steps_per_second == 0:
                        elapsed = (i + 1) // steps_per_second
                        print(f"仿真时间: {elapsed}/{args.simulation_time} 秒")
                    
                    time.sleep(0.01)  # 稍微长一点的休眠
            except KeyboardInterrupt:
                print("\n\n用户中断仿真")
        else:
            # 无物理仿真，只是显示场景
            print("场景已加载，保持显示...")
            print("(窗口将保持打开，按 Ctrl+C 退出)\n")
            
            try:
                # 只做最基本的渲染更新，不做物理step
                for i in range(args.simulation_time * 10):  # 每0.1秒更新一次
                    omni.kit.app.get_app().update()
                    time.sleep(0.1)
                    
                    if (i + 1) % 10 == 0:
                        elapsed = (i + 1) // 10
                        if elapsed % 5 == 0:  # 每5秒打印一次
                            print(f"显示时间: {elapsed}/{args.simulation_time} 秒")
            except KeyboardInterrupt:
                print("\n\n用户中断")
        
        print(f"\n{'='*60}")
        print("仿真完成")
        print(f"{'='*60}")
        print("\n按 Enter 键退出...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n退出...")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n关闭 Isaac Sim...")
        simulation_app.close()


if __name__ == "__main__":
    main()
