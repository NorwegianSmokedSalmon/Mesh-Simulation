#!/bin/bash
# 针对小物件（杯子等）的精细化修复

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   小物件精细化修复"
echo "   (针对杯子等小物体的穿模问题)"
echo "========================================"
echo ""

# 检查输入目录
if [ ! -d "../world_mesh_fixed" ]; then
    echo "❌ 错误：找不到 ../world_mesh_fixed 目录"
    echo "请先运行 ./run_fast.sh 生成初步修复结果"
    exit 1
fi

echo "[执行] 检测并修复小物件碰撞..."
echo ""
echo "配置："
echo "  - 小物件尺寸阈值: 0.3m"
echo "  - 小物件采样点数: 2000 (更密集)"
echo "  - 普通物件采样点数: 500"
echo "  - 容忍度: 0.005m (更严格)"
echo ""

python fix_small_objects.py \
    --input_dir ../world_mesh_fixed \
    --output_dir ../world_mesh_final \
    --tolerance 0.005 \
    --size_threshold 0.3 \
    --dense_samples 2000 \
    --iterations 100

echo ""
echo "✓ 完成！最终结果保存在 ../world_mesh_final/"
echo ""
echo "可视化最终结果："
echo "  python ../utils/visualize_world_mesh_open3d.py --dir ../world_mesh_final"
echo ""
echo "对比三个版本："
echo "  原始: ../world_mesh"
echo "  初步修复: ../world_mesh_fixed"
echo "  精细修复: ../world_mesh_final"
