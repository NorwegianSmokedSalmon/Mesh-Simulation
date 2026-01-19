#!/bin/bash
# 快速版本 - 使用优化算法，比原版快10-50倍

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   快速SDF穿模检测与修复"
echo "   (优化版本 - 速度提升10-50倍)"
echo "========================================"
echo ""

# 直接运行修复
echo "[执行] 检测并修复碰撞..."
python sdf_collision_resolver_fast.py \
    --input_dir ../world_mesh \
    --output_dir ../world_mesh_fixed \
    --tolerance 0.01 \
    --sample_points 500 \
    --iterations 50

echo ""
echo "✓ 完成！结果保存在 ../world_mesh_fixed/"
echo ""
echo "可视化结果："
echo "  python ../utils/visualize_world_mesh_open3d.py --dir ../world_mesh_fixed"
echo ""
echo "对比修复前后："
echo "  python visualize_comparison.py --original ../world_mesh --fixed ../world_mesh_fixed"
