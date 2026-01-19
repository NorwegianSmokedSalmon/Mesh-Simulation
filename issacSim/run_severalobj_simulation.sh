#!/bin/bash
# 快速加载场景（不启用物理仿真）

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   快速查看场景 (无物理仿真)"
echo "========================================"
echo ""
echo "这个模式只加载场景，不启用物理"
echo "可以快速检查场景是否正确导入"
echo "按 Ctrl+C 可随时退出"
echo ""

# 运行仿真（无物理，只加载5个物体）
python load_world_mesh.py \
    --input_dir ../world_mesh_final \
    --simulation_time 30 \
    --no_physics \
    --max_objects 5

echo ""
echo "✓ 完成！"
echo ""
echo "如果想查看更多物体，可以增加数量："
echo "  python load_world_mesh.py --input_dir ../world_mesh_final --no_physics --max_objects 20"
