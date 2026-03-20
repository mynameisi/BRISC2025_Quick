#!/bin/bash
# BRISC2025 实验状态检查脚本
# 可设置为 cron 定时任务

cd ~/BRISC2025_Quick

echo "=========================================="
echo "BRISC2025 实验状态检查"
echo "时间: $(date)"
echo "=========================================="

# 检查 Python 环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在"
    exit 1
fi

source venv/bin/activate

# 打印实验摘要
echo ""
python run_conservative.py --summary

# 检查是否有正在运行的训练进程
if pgrep -f "run_conservative.py" > /dev/null; then
    echo ""
    echo "🔄 训练正在进行中..."
    ps aux | grep run_conservative.py | grep -v grep
else
    echo ""
    echo "⏹️  没有正在运行的训练任务"
fi

# 检查磁盘空间
echo ""
echo "磁盘使用情况:"
df -h . | tail -1

# 检查生成的文件
echo ""
echo "生成的模型文件:"
ls -lh *.pth 2>/dev/null || echo "  暂无模型文件"

echo ""
echo "=========================================="
echo "检查完成: $(date)"
echo "=========================================="
