#!/bin/bash
# BRISC2025 Auto Research - 本地一键运行脚本
# 用法: ./run_auto_research.sh [次数]

cd "$(dirname "$0")"

# 激活虚拟环境
source venv/bin/activate

# 获取运行次数（默认1次）
RUN_COUNT=${1:-1}

echo "🧠 BRISC2025 Auto Research"
echo "========================="
echo "计划运行: $RUN_COUNT 次实验"
echo ""

for i in $(seq 1 $RUN_COUNT); do
    echo "📌 第 $i/$RUN_COUNT 轮实验"
    echo "------------------------"
    
    # 生成实验计划
    echo "1️⃣ 生成实验计划..."
    python auto_research.py
    
    # 执行实验
    echo ""
    echo "2️⃣ 执行实验训练..."
    python auto_research_runner.py
    
    # 发送通知
    echo ""
    echo "3️⃣ 发送通知..."
    python auto_research_notify.py
    
    echo ""
    echo "✅ 第 $i 轮完成"
    echo "========================="
    echo ""
    
    # 如果不是最后一轮，等待一下
    if [ $i -lt $RUN_COUNT ]; then
        echo "⏳ 等待5秒后继续下一轮..."
        sleep 5
        echo ""
    fi
done

echo "🎉 所有实验完成！"
echo ""
echo "查看结果:"
echo "  历史记录: experiments/auto_research/history.jsonl"
echo "  最新通知: experiments/auto_research/latest_notification.txt"
