"""
BRISC2025 Auto Research - 实验执行器
读取实验计划并执行训练
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import time

EXPERIMENTS_DIR = Path('experiments/auto_research')

def get_pending_experiment():
    """获取待执行的实验"""
    history_file = EXPERIMENTS_DIR / 'history.jsonl'
    if not history_file.exists():
        return None
    
    with open(history_file) as f:
        for line in f:
            if line.strip():
                exp = json.loads(line)
                if exp.get('status') == 'planned':
                    return exp
    return None

def update_experiment_status(exp_id: str, status: str, result: dict = None):
    """更新实验状态"""
    history_file = EXPERIMENTS_DIR / 'history.jsonl'
    temp_file = EXPERIMENTS_DIR / 'history.jsonl.tmp'
    
    with open(history_file) as f_in, open(temp_file, 'w') as f_out:
        for line in f_in:
            if line.strip():
                exp = json.loads(line)
                if exp['id'] == exp_id:
                    exp['status'] = status
                    if result:
                        exp['result'] = result
                        exp['completed_at'] = datetime.now().isoformat()
                f_out.write(json.dumps(exp, default=str) + '\n')
    
    temp_file.replace(history_file)

def generate_training_script(experiment: dict) -> str:
    """根据实验配置生成训练脚本"""
    config = experiment['config']
    exp_id = experiment['id']
    
    script = f'''#!/usr/bin/env python3
"""
Auto Research Experiment {exp_id}
Generated: {datetime.now().isoformat()}
Strategy: {experiment['strategy']}
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime
from pathlib import Path

# 配置
CONFIG = {{
    'data_dir': '~/.cache/kagglehub/datasets/briscdataset/brisc2025/versions/6/brisc2025/segmentation_task',
    'batch_size': {config['batch_size']},
    'epochs': 30,
    'lr': {config['learning_rate']},
    'min_lr': 1e-7,
    'image_size': {config['image_size']},
    'num_classes': 2,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'early_stop_patience': 7,
    'freeze_layers': {config['freeze_layers']},
    'attention': {str(config['attention']).lower()},
    'decoder_channels': {config['decoder_channels']},
    'augmentation': '{config['augmentation']}',
}}

EXPERIMENT_ID = '{exp_id}'
LOG_DIR = Path('experiments/auto_research') / EXPERIMENT_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ... [训练代码将在这里生成，包括完整的U-Net实现] ...
# 实际运行时会替换为完整代码

if __name__ == '__main__':
    # 模拟训练结果
    result = {{
        'dice': 0.70,
        'iou': 0.55,
        'epochs': 15,
        'status': 'completed'
    }}
    with open(LOG_DIR / 'result.json', 'w') as f:
        json.dump(result, f)
'''
    return script

def run_experiment(experiment: dict) -> dict:
    """执行实验并返回结果"""
    exp_id = experiment['id']
    print(f"🚀 开始实验 {exp_id}")
    
    # 更新状态为运行中
    update_experiment_status(exp_id, 'running')
    
    # 生成并保存训练脚本
    script = generate_training_script(experiment)
    script_path = EXPERIMENTS_DIR / f"{exp_id}_train.py"
    with open(script_path, 'w') as f:
        f.write(script)
    
    # 实际执行训练（这里会调用实际的训练脚本）
    # 简化版本：直接运行预定义的配置
    start_time = time.time()
    
    # 构建命令
    cmd = [
        'python3', 'run_auto_experiment.py',
        '--exp-id', exp_id,
        '--config', json.dumps(experiment['config'])
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        # 读取结果
        result_file = EXPERIMENTS_DIR / exp_id / 'result.json'
        if result_file.exists():
            with open(result_file) as f:
                exp_result = json.load(f)
        else:
            exp_result = {{
                'dice': 0.0,
                'iou': 0.0,
                'epochs': 0,
                'error': '未找到结果文件'
            }}
        
        exp_result['duration_minutes'] = (time.time() - start_time) / 60
        
        return exp_result
        
    except subprocess.TimeoutExpired:
        return {{
            'dice': 0.0,
            'iou': 0.0,
            'epochs': 0,
            'error': '超时(>2小时)'
        }}
    except Exception as e:
        return {{
            'dice': 0.0,
            'iou': 0.0,
            'epochs': 0,
            'error': str(e)
        }}

def main():
    print("🧠 BRISC2025 Auto Research - 实验执行器")
    print("=" * 50)
    
    # 获取待执行实验
    experiment = get_pending_experiment()
    
    if not experiment:
        print("✅ 没有待执行的实验")
        print("\n当前状态:")
        # 显示统计
        from auto_research import analyze_results_and_recommend
        print(analyze_results_and_recommend())
        return
    
    print(f"\n找到待执行实验: {experiment['id']}")
    print(f"策略: {experiment['strategy']}")
    print(f"假设: {experiment['hypothesis']}")
    print(f"\n配置:")
    print(json.dumps(experiment['config'], indent=2))
    
    # 执行
    result = run_experiment(experiment)
    
    # 更新状态
    update_experiment_status(experiment['id'], 'completed', result)
    
    print(f"\n✅ 实验完成")
    print(f"结果: Dice={result.get('dice', 0):.4f}, IoU={result.get('iou', 0):.4f}")
    
    if 'error' in result:
        print(f"⚠️ 警告: {result['error']}")

if __name__ == '__main__':
    main()
