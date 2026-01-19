# Mesh-Simulation

3D场景mesh穿模检测与修复工具集

## 📋 项目概述

本项目提供了一套完整的工具，用于处理3D场景重建中常见的mesh穿模（collision/interpenetration）问题。通过使用**有向距离场（Signed Distance Field, SDF）**技术，可以自动检测并修复场景中多个mesh之间的重叠和穿透。

## 🗂️ 项目结构

```
Mesh-Simulation/
├── README.md                    # 项目主文档（本文件）
├── QUICK_START.md              # 快速开始指南
├── world_mesh/                  # 输入：世界坐标系下的原始mesh（GLB格式）
│   ├── instance_12_refine_world.glb
│   ├── instance_13_refine_world.glb
│   └── ...
├── simulation/                  # 核心：SDF碰撞检测与修复模块
│   ├── sdf_collision_resolver.py         # 标准版（精确但慢）
│   ├── sdf_collision_resolver_fast.py    # 快速版（推荐）⚡
│   ├── fix_small_objects.py              # 小物件精修工具
│   ├── simulation.md                     # 详细原理说明
│   ├── README.md                         # 使用说明
│   ├── requirements.txt                  # Python依赖
│   ├── run_fast.sh                       # 快速修复脚本（推荐）⚡
│   ├── run_fix_small.sh                  # 小物件修复脚本
│   └── visualize_comparison.py           # 对比可视化工具
├── utils/                       # 工具：可视化和导出
│   ├── visualize_world_mesh_open3d.py    # Open3D可视化工具
│   └── export_refine_to_world_meshes.py
└── issacSim/                    # Isaac Sim 物理仿真
    ├── load_world_mesh.py                # 新版导入脚本（推荐）⚡
    ├── initial_version.py                # 旧版脚本
    ├── run_simulation.sh                 # 快速运行仿真
    └── README.md                         # 仿真说明文档
```

## 🚀 完整工作流程

### 步骤 1: 环境准备

```bash
# 进入项目目录
cd Mesh-Simulation

# 安装依赖
pip install -r simulation/requirements.txt
```

### 步骤 2: 修复穿模（3-8分钟）⭐

#### 2.1 快速修复大物件（2-5分钟）

```bash
cd simulation
./run_fast.sh
```

处理大部分穿模问题，输出到 `world_mesh_fixed/`

#### 2.2 精细修复小物件（1-3分钟）

```bash
./run_fix_small.sh
```

专门处理杯子等小物件的穿模，输出到 `world_mesh_final/` ✅

### 步骤 3: 可视化验证

```bash
# 查看最终结果
cd ..
python utils/visualize_world_mesh_open3d.py --dir world_mesh_final

# 对比修复前后
cd simulation
python visualize_comparison.py --original ../world_mesh --fixed ../world_mesh_final
```

### 步骤 4: Isaac Sim 物理仿真（可选）

```bash
cd ../issacSim
./run_simulation.sh
```

或手动运行：

```bash
python load_world_mesh.py --input_dir ../world_mesh_final --simulation_time 10
```

观察物体在重力作用下的物理交互。

---

**快速开始指南**: **[QUICK_START.md](QUICK_START.md)**  
**详细使用说明**: [simulation/README.md](simulation/README.md)

## 💡 核心功能

### ✅ 穿模检测
- 自动检测所有mesh对之间的碰撞
- 计算穿透深度和方向
- 生成详细的碰撞报告

### ✅ 穿模修复
- **投影方法**：快速直接的位置调整
- **物理仿真**：基于力学的真实分离
- 保持场景整体布局的合理性

### ✅ 可视化工具
- Open3D交互式3D查看器
- 修复前后对比显示
- 支持纹理渲染

## 📖 技术原理

### SDF（有向距离场）

SDF为空间中每个点分配一个距离值：
- **正值**：点在物体外部
- **零值**：点在物体表面
- **负值**：点在物体内部（穿透）

通过计算mesh之间的SDF，可以精确检测穿透区域并计算分离方向。

详细原理请参阅：**[simulation/simulation.md](simulation/simulation.md)**

## 🎯 使用场景

- ✅ 3D场景重建后的mesh清理
- ✅ 多物体场景的自动布局优化
- ✅ 物理仿真前的初始化
- ✅ 游戏/VR场景的资产处理
- ✅ 机器人导航地图的预处理

## 📊 性能参考

| Mesh数量 | 分辨率 | 检测时间 | 修复时间 |
|---------|--------|---------|---------|
| 10个    | 64³    | ~30秒   | ~1分钟  |
| 20个    | 64³    | ~2分钟  | ~3分钟  |
| 50个    | 64³    | ~10分钟 | ~15分钟 |

*测试环境：Intel i7 + 16GB RAM*

## 🛠️ 高级用法

### 批量处理

```bash
# 处理所有mesh（无数量限制）
python simulation/sdf_collision_resolver.py \
    --input_dir world_mesh \
    --output_dir world_mesh_fixed \
    --method projection \
    --resolution 128 \
    --iterations 100
```

### 高精度模式

```bash
# 使用更高分辨率和更严格的容忍度
python simulation/sdf_collision_resolver.py \
    --input_dir world_mesh \
    --output_dir world_mesh_fixed \
    --resolution 256 \
    --tolerance 0.001 \
    --iterations 200
```

### 物理仿真模式

```bash
# 使用物理引擎进行更真实的分离
python simulation/sdf_collision_resolver.py \
    --input_dir world_mesh \
    --output_dir world_mesh_fixed \
    --method physical \
    --iterations 500
```

## 📁 输出格式

修复后的目录结构：

```
world_mesh_fixed/
├── instance_12_refine_world.glb    # 修复后的mesh文件
├── instance_13_refine_world.glb
├── ...
└── transforms.json                  # 变换记录（位置、旋转）
```

## 🐛 常见问题

### Q: 修复后仍有穿模？
**A**: 尝试增加迭代次数或提高SDF分辨率：
```bash
--iterations 200 --resolution 128
```

### Q: 计算太慢？
**A**: 降低分辨率或限制mesh数量：
```bash
--resolution 32 --max_meshes 10
```

### Q: mesh位置偏移太大？
**A**: 可能初始穿模太严重，尝试分两步处理：
```bash
# 步骤1：粗略分离
python sdf_collision_resolver.py --resolution 32 --iterations 20 ...
# 步骤2：精细调整
python sdf_collision_resolver.py --resolution 128 --iterations 100 ...
```

## 📚 相关文档

- [simulation/simulation.md](simulation/simulation.md) - SDF原理详解
- [simulation/README.md](simulation/README.md) - 详细使用说明
- [Trimesh文档](https://trimsh.org/)
- [Open3D文档](http://www.open3d.org/docs/)

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可

MIT License

---

**更新日期**: 2026-01-18  
**版本**: 1.0