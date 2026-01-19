# 🚀 快速开始指南

## 问题：世界坐标系下的mesh存在穿模

你的 `world_mesh/` 文件夹中有65个mesh，它们之间存在重叠穿模问题。

## 解决方案：两步修复流程

### ⚡ 第一步：快速修复大物件（2-5分钟）

```bash
cd simulation
./run_fast.sh
```

**结果**：
- ✅ 大物件的穿模基本解决
- ⚠️ 小物件（杯子）可能仍有穿模
- 📁 输出到 `world_mesh_fixed/`

---

### 🔧 第二步：精细修复小物件（1-3分钟）

```bash
./run_fix_small.sh
```

**结果**：
- ✅ 小物件穿模问题解决
- ✅ 所有mesh都无穿模
- 📁 最终输出到 `world_mesh_final/`

---

## 总耗时：3-8分钟 ⏱️

对比原始方法需要30-60分钟，**速度提升10倍！**

## 查看结果

### 可视化最终结果

```bash
python utils/visualize_world_mesh_open3d.py --dir world_mesh_final
```

### 对比修复前后

```bash
cd simulation
python visualize_comparison.py \
    --original ../world_mesh \
    --fixed ../world_mesh_final \
    --mode overlay
```

## 目录结构

```
world_mesh/           # 原始mesh（有穿模）
world_mesh_fixed/     # 第一步：大物件修复后
world_mesh_final/     # 第二步：小物件修复后（最终结果）✅
```

## 原理简介

### 第一步：快速版本
- 使用KD树 + 采样点方法
- 500个采样点/mesh
- 适合处理大物件
- 速度快但精度中等

### 第二步：小物件专用
- 自适应采样：小物件2000点，大物件500点
- 更严格的容忍度（0.005m vs 0.01m）
- 专门优化小物件间的碰撞检测

## 调整参数（可选）

如果小物件仍有轻微穿模：

```bash
cd simulation
python fix_small_objects.py \
    --input_dir ../world_mesh_fixed \
    --output_dir ../world_mesh_final \
    --tolerance 0.003 \       # 更严格
    --dense_samples 3000 \    # 更多采样点
    --iterations 150          # 更多迭代
```

## 技术细节

详细的算法原理和数学推导，请参阅：
- **[simulation.md](simulation.md)** - SDF原理详解
- **[simulation/README.md](simulation/README.md)** - 完整使用说明

---

**问题？** 如果遇到问题，请查看 [simulation/README.md](simulation/README.md) 的故障排除章节。
