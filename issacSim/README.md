# Isaac Sim 仿真脚本

## 📋 文件说明

### load_world_mesh.py（新版本，推荐）

**简化版本**，直接加载 `world_mesh_final/` 中的 GLB 文件到 Isaac Sim 进行物理仿真。

**特点**：
- ✅ 自动扫描并转换所有 GLB 文件
- ✅ 自动检测地面高度（最小 z 值）
- ✅ z-up 坐标系，直接导入
- ✅ 自动添加刚体 + 碰撞属性
- ✅ 重力仿真

**不需要**：
- ❌ 位姿文件
- ❌ 旋转文件
- ❌ 手动设置坐标

### initial_version.py（旧版本）

复杂版本，需要位姿、旋转等配置文件。已被新版本替代。

## 🚀 快速开始

### 1. 环境要求

- NVIDIA Isaac Sim 2022.2 或更高版本
- Python 3.7+
- CUDA 兼容的 NVIDIA GPU

### 2. 运行仿真

#### 基本用法

```bash
cd issacSim

# 使用默认参数（加载 ../world_mesh_final，仿真10秒）
python load_world_mesh.py

# 或使用 Isaac Sim Python
~/.local/share/ov/pkg/isaac_sim-*/python.sh load_world_mesh.py
```

#### 自定义参数

```bash
# 指定输入目录
python load_world_mesh.py --input_dir ../world_mesh_final

# 指定仿真时长（秒）
python load_world_mesh.py --simulation_time 30

# 完整参数
python load_world_mesh.py \
    --input_dir ../world_mesh_final \
    --usd_cache ./usd_cache \
    --simulation_time 20
```

## 📊 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `../world_mesh_final` | GLB 文件所在目录 |
| `--usd_cache` | `./usd_cache` | USD 缓存目录 |
| `--simulation_time` | `10` | 仿真时长（秒） |

## 🎬 工作流程

### 步骤 1: 转换资产

脚本自动将所有 GLB 文件转换为 USD 格式：

```
world_mesh_final/
  instance_12_refine_world.glb  →  usd_cache/instance_12_refine_world.usd
  instance_13_refine_world.glb  →  usd_cache/instance_13_refine_world.usd
  ...
```

### 步骤 2: 检测地面

遍历所有物体，找到最小的 z 坐标作为地面高度：

```
z_min(instance_12) = -0.523m
z_min(instance_13) = -0.489m
...
ground_z = min(all z_min) = -0.523m
```

### 步骤 3: 加载物体

将所有物体加载到场景中，保持原有的世界坐标位置。

### 步骤 4: 添加物理属性

为每个物体添加：
- **刚体（Rigid Body）**：使物体受重力影响
- **碰撞体（Collision）**：使物体能够相互碰撞
- **凸包近似（Convex Hull）**：精确的碰撞形状

### 步骤 5: 启动仿真

物理引擎开始计算，物体在重力作用下运动。

## 🔧 坐标系说明

### 输入坐标系（world_mesh_final）

- **Z-up**: Z 轴向上
- **单位**: 米（m）
- **原点**: 世界坐标系原点

### Isaac Sim 坐标系

- **Z-up**: Z 轴向上（与输入一致）✅
- **单位**: 米（m）
- **重力**: -Z 方向（向下）

**好消息**: 由于坐标系一致，可以直接导入，无需转换！

## 📝 输出文件

仿真完成后会生成：

```
world_mesh_final/
  isaac_scene_info.json    # 场景信息（地面高度、物体数量、包围盒等）
```

`isaac_scene_info.json` 内容示例：

```json
{
  "ground_z": -0.523,
  "object_count": 65,
  "objects": [
    {
      "name": "instance_12_refine_world",
      "prim_path": "/World/instance_12_refine_world",
      "bbox": {
        "min": [1.2, 3.4, -0.5],
        "max": [1.5, 3.7, 0.2],
        "center": [1.35, 3.55, -0.15],
        "size": [0.3, 0.3, 0.7]
      }
    },
    ...
  ]
}
```

## 🎮 操作指南

### 视角控制

- **旋转视角**: 鼠标左键拖拽
- **缩放**: 鼠标滚轮
- **平移**: 鼠标中键拖拽
- **聚焦物体**: 选中物体后按 `F` 键

### 仿真控制

- **开始/暂停**: 空格键
- **单步执行**: `.` 键
- **重置**: `Ctrl + R`

## 🐛 故障排除

### 问题 1: 找不到 GLB 文件

**错误信息**:
```
错误: 目录不存在 ../world_mesh_final
```

**解决方案**:
```bash
# 检查目录是否存在
ls ../world_mesh_final

# 或指定正确的路径
python load_world_mesh.py --input_dir /完整/路径/to/world_mesh_final
```

### 问题 2: 转换失败

**错误信息**:
```
✗ 转换失败
```

**解决方案**:
- 检查 GLB 文件是否损坏
- 确认 Isaac Sim 资产转换扩展已启用
- 查看完整错误日志

### 问题 3: 物体飘浮或掉落

**现象**: 物体不在正确的位置

**原因**: 地面高度检测不准确

**解决方案**:
检查 `isaac_scene_info.json` 中的 `ground_z` 值，必要时手动调整代码中的地面位置。

### 问题 4: 碰撞不准确

**现象**: 物体穿透或碰撞异常

**解决方案**:
修改代码中的碰撞近似方式：

```python
# 在 load_world_mesh.py 中找到这一行
add_physics_to_object(stage, prim_path, use_convex_hull=True)

# 改为使用包围盒（更快但不太精确）
add_physics_to_object(stage, prim_path, use_convex_hull=False)
```

## 💡 高级用法

### 调整物理参数

在代码中可以修改物理属性：

```python
# 在 add_physics_to_object 函数后添加
rigid_body = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)

# 设置质量（kg）
mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.GetMassAttr().Set(1.0)

# 设置线性阻尼
rigid_body.GetLinearDampingAttr().Set(0.1)

# 设置角阻尼
rigid_body.GetAngularDampingAttr().Set(0.1)
```

### 添加地面

如果需要一个平面地面：

```python
# 在 main() 函数中取消注释
my_world.scene.add_default_ground_plane()
```

### 保存场景

仿真完成后保存当前状态：

```python
# 在仿真循环后添加
stage.Export("./output_scene.usd")
print("场景已保存到 output_scene.usd")
```

## 📚 相关文档

- [Isaac Sim 官方文档](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [USD 文档](https://graphics.pixar.com/usd/docs/index.html)
- [Physics 教程](https://docs.omniverse.nvidia.com/isaacsim/latest/features/physics_simulation.html)

## 🎯 与穿模修复的整合

完整的工作流程：

```bash
# 1. 快速修复大物件
cd simulation
./run_fast.sh

# 2. 精细修复小物件
./run_fix_small.sh

# 3. 导入 Isaac Sim 仿真
cd ../issacSim
python load_world_mesh.py --input_dir ../world_mesh_final

# 4. 观察物理仿真结果
# 如果物体位置不合理，返回步骤2调整参数
```

---

**作者**: AI Assistant  
**日期**: 2026-01-18  
**版本**: 1.0
