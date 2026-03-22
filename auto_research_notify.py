"""
BRISC2025 Auto Research - 通知脚本
实验完成后发送通知到Feishu
"""
import json
import os
from pathlib import Path
from datetime import datetime

def load_latest_result():
    """加载最新的实验结果"""
    exp_dir = Path('experiments/auto_research')
    if not exp_dir.exists():
        return None
    
    # 找到所有带result.json的实验
    results = []
    for exp_path in exp_dir.iterdir():
        if exp_path.is_dir():
            result_file = exp_path / 'result.json'
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                    data['exp_id'] = exp_path.name
                    data['timestamp'] = result_file.stat().st_mtime
                    results.append(data)
    
    if not results:
        return None
    
    # 返回最新的
    return max(results, key=lambda x: x['timestamp'])


def format_notification(result):
    """格式化通知内容"""
    exp_id = result.get('exp_id', 'unknown')
    dice = result.get('dice', 0)
    iou = result.get('iou', 0)
    epochs = result.get('epochs', 0)
    config = result.get('config', {})
    
    # 计算排名
    exp_dir = Path('experiments/auto_research')
    all_dices = []
    for exp_path in exp_dir.iterdir():
        if exp_path.is_dir() and exp_path.name != exp_id:
            rf = exp_path / 'result.json'
            if rf.exists():
                with open(rf) as f:
                    all_dices.append(json.load(f).get('dice', 0))
    
    rank = sum(1 for d in all_dices if d > dice) + 1
    total = len(all_dices) + 1
    
    # 生成消息
    msg = f"""🧠 **BRISC2025 Auto Research 实验报告**

📊 **实验 {exp_id}**
• Dice Score: **{dice:.4f}** ({dice*100:.2f}%)
• IoU Score: {iou:.4f} ({iou*100:.2f}%)
• 训练轮次: {epochs}
• 当前排名: {rank}/{total}

⚙️ **配置**
• 学习率: {config.get('learning_rate', 'N/A')}
• 冻结层数: {config.get('freeze_layers', 'N/A')}
• 增强强度: {config.get('augmentation', 'N/A')}
• 图像尺寸: {config.get('image_size', 'N/A')}
• Batch大小: {config.get('batch_size', 'N/A')}

📈 **历史最佳**
• 最高Dice: {max(all_dices + [dice]):.4f}
• 实验总数: {total}

---
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    return msg


def send_feishu_notification(message):
    """发送Feishu通知（如果配置了webhook）"""
    webhook = os.environ.get('FEISHU_WEBHOOK')
    if not webhook:
        print("⚠️ 未配置FEISHU_WEBHOOK环境变量")
        print("通知内容:")
        print(message)
        return False
    
    try:
        import requests
        payload = {
            "msg_type": "text",
            "content": {"text": message}
        }
        resp = requests.post(webhook, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"发送通知失败: {e}")
        return False


def main():
    result = load_latest_result()
    
    if not result:
        print("❌ 未找到实验结果")
        return
    
    msg = format_notification(result)
    print(msg)
    
    # 尝试发送
    send_feishu_notification(msg)
    
    # 同时保存到文件
    notify_file = Path('experiments/auto_research') / 'latest_notification.txt'
    with open(notify_file, 'w') as f:
        f.write(msg)
    
    print(f"\n✅ 通知已保存到: {notify_file}")


if __name__ == '__main__':
    main()
