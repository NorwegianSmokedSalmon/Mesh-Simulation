#!/bin/bash
# 快速启动 Isaac Sim 仿真

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   Isaac Sim 物理仿真"
echo "========================================"
echo ""

# 检查输入目录
if [ ! -d "../world_mesh_final" ]; then
    echo "❌ 错误：找不到 ../world_mesh_final 目录"
    echo "请先运行穿模修复脚本："
    echo "  cd ../simulation"
    echo "  ./run_fast.sh"
    echo "  ./run_fix_small.sh"
    exit 1
fi

# 统计 GLB 文件数量
glb_count=$(ls -1 ../world_mesh_final/*.glb 2>/dev/null | wc -l)
echo "输入目录: ../world_mesh_final"
echo "GLB 文件数量: $glb_count"
echo ""

if [ $glb_count -eq 0 ]; then
    echo "❌ 错误：没有找到 GLB 文件"
    exit 1
fi

echo "配置："
echo "  - 坐标系: Z-up（无需转换）"
echo "  - 地面高度: 自动检测（最小 z 值）"
echo "  - 物理属性: 刚体 + 凸包碰撞"
echo "  - 重力: (0, 0, -9.81) m/s² [Z轴负方向]"
echo "  - 加载物体: 全部 $glb_count 个"
echo "  - 仿真时长: 100秒（可通过 --simulation_time 修改）"
echo ""
echo "启动 Isaac Sim..."
echo ""

# 尝试找到 Isaac Sim Python
ISAAC_PYTHON=""

# 常见 Isaac Sim 安装位置
POSSIBLE_PATHS=(
    "$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh"
    "$HOME/.local/share/ov/pkg/isaac_sim-2023.1.0/python.sh"
    "$HOME/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh"
    "$HOME/.local/share/ov/pkg/isaac_sim-2022.2.0/python.sh"
    "/isaac-sim/python.sh"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ]; then
        ISAAC_PYTHON="$path"
        echo "✓ 找到 Isaac Sim Python: $ISAAC_PYTHON"
        break
    fi
done

if [ -z "$ISAAC_PYTHON" ]; then
    echo "⚠️  未找到 Isaac Sim Python，尝试使用系统 Python"
    echo "   如果失败，请手动指定 Isaac Sim Python 路径："
    echo "   /path/to/isaac_sim/python.sh load_world_mesh.py"
    ISAAC_PYTHON="python"
fi

echo ""
echo "执行仿真..."
echo ""

# 运行仿真（--max_objects 0 表示加载全部物体）
$ISAAC_PYTHON load_world_mesh.py \
    --input_dir ../world_mesh_final \
    --usd_cache ./usd_cache \
    --simulation_time 1000 \
    --max_objects 0

echo ""
echo "仿真结束"
