"""
BRISC2025 Auto Research - 自动化实验系统
核心逻辑：读取实验历史 → 生成新配置 → 运行实验 → 记录结果 → 决策下一步
"""
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

EXPERIMENTS_DIR = Path('experiments/auto_research')
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 搜索空间定义
SEARCH_SPACE = {
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
    'freeze_layers': [0, 2, 4],  # 0=全解冻, 2=解冻layer3+4, 4=冻结全部encoder
    'augmentation': ['weak', 'medium', 'strong'],
    'attention': [False, True],
    'image_size': [256, 384],
    'batch_size': [8, 16],
    'decoder_channels': [64, 128],  # decoder复杂度
}

# 已知有效配置（从之前实验学习）
KNOWN_GOOD_CONFIGS = [
    {
        'name': 'baseline_v1',
        'config': {
            'learning_rate': 0.001,
            'freeze_layers': 0,
            'augmentation': 'medium',
            'attention': False,
            'image_size': 256,
            'batch_size': 16,
            'decoder_channels': 64,
        },
        'result': {'dice': 0.6836, 'iou': 0.5374, 'epochs': 31},
        'strategy': 'vanilla_unet'
    },
    {
        'name': 'resnet34_frozen',
        'config': {
            'learning_rate': 0.0001,
            'freeze_layers': 4,
            'augmentation': 'medium',
            'attention': False,
            'image_size': 256,
            'batch_size': 16,
            'decoder_channels': 64,
        },
        'result': {'dice': 0.663, 'iou': 0.515, 'epochs': 3},
        'strategy': 'frozen_pretrained'
    }
]


def load_experiment_history() -> List[Dict]:
    """加载所有历史实验"""
    history_file = EXPERIMENTS_DIR / 'history.jsonl'
    if not history_file.exists():
        return []
    
    experiments = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    return experiments


def save_experiment(exp: Dict):
    """保存实验记录"""
    history_file = EXPERIMENTS_DIR / 'history.jsonl'
    with open(history_file, 'a') as f:
        f.write(json.dumps(exp, default=str) + '\n')


def generate_experiment_id(config: Dict) -> str:
    """生成实验ID"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def select_strategy(history: List[Dict]) -> str:
    """根据历史选择探索策略"""
    if len(history) < 3:
        return 'exploration'  # 早期：广泛探索
    
    # 分析历史结果
    recent = history[-5:]
    best_dice = max([e['result']['dice'] for e in history if 'result' in e], default=0.66)
    
    # 如果最近5次没有改善，切换到exploitation
    recent_best = max([e['result']['dice'] for e in recent if 'result' in e], default=0)
    
    if recent_best <= best_dice * 1.01:  # 1%以内视为无改善
        return 'exploitation'  # 在最优点附近精细搜索
    
    return 'exploration'


def adjust_search_space_based_on_history(history: List[Dict]) -> Dict:
    """根据历史实验结果自动调整搜索空间"""
    
    # 提取已完成的实验
    completed = [e for e in history if e.get('status') == 'completed' and 'result' in e]
    
    if not completed:
        return SEARCH_SPACE.copy()
    
    # 分析各参数的效果
    adjusted_space = SEARCH_SPACE.copy()
    insights = []
    
    # 1. 分析 freeze_layers
    freeze_results = {}
    for e in completed:
        fl = e['config'].get('freeze_layers', 0)
        dice = e['result']['dice']
        if fl not in freeze_results:
            freeze_results[fl] = []
        freeze_results[fl].append(dice)
    
    # 如果 freeze=0 和 freeze=2 都表现不佳，优先 freeze=4
    if 0 in freeze_results and 2 in freeze_results:
        avg_0 = sum(freeze_results[0]) / len(freeze_results[0])
        avg_2 = sum(freeze_results[2]) / len(freeze_results[2])
        if avg_0 < 0.66 and avg_2 < 0.66 and 4 not in freeze_results:
            # 强制尝试 freeze=4
            adjusted_space['freeze_layers'] = [4]
            insights.append(f"freeze=0({avg_0:.3f})和freeze=2({avg_2:.3f})均低于baseline，优先尝试freeze=4")
    
    # 2. 分析 learning_rate
    lr_results = {}
    for e in completed:
        lr = e['config'].get('learning_rate', 1e-4)
        dice = e['result']['dice']
        if lr not in lr_results:
            lr_results[lr] = []
        lr_results[lr].append(dice)
    
    # 如果 5e-4 效果不好，降低学习率
    if 0.0005 in lr_results:
        avg_5e4 = sum(lr_results[0.0005]) / len(lr_results[0.0005])
        if avg_5e4 < 0.65:
            adjusted_space['learning_rate'] = [1e-5, 5e-5, 1e-4]  # 移除 5e-4
            insights.append(f"lr=5e-4效果不佳(avg={avg_5e4:.3f})，降低学习率范围")
    
    # 3. 分析 augmentation
    aug_results = {}
    for e in completed:
        aug = e['config'].get('augmentation', 'medium')
        dice = e['result']['dice']
        if aug not in aug_results:
            aug_results[aug] = []
        aug_results[aug].append(dice)
    
    if 'strong' in aug_results:
        avg_strong = sum(aug_results['strong']) / len(aug_results['strong'])
        if avg_strong < 0.60:
            adjusted_space['augmentation'] = ['weak', 'medium']  # 移除 strong
            insights.append(f"强增强效果不佳(avg={avg_strong:.3f})，降级到weak/medium")
    
    return adjusted_space, insights


def generate_next_config(strategy: str, history: List[Dict], adjusted_space: Dict = None) -> Dict:
    """生成下一个实验配置"""
    
    # 使用调整后的搜索空间或默认空间
    space = adjusted_space if adjusted_space else SEARCH_SPACE
    
    if strategy == 'exploration':
        # 广泛探索：从调整后的空间随机采样
        config = {
            'learning_rate': random.choice(space['learning_rate']),
            'freeze_layers': random.choice(space['freeze_layers']),
            'augmentation': random.choice(space['augmentation']),
            'attention': random.choice(space['attention']),
            'image_size': random.choice(space['image_size']),
            'batch_size': random.choice(space['batch_size']),
            'decoder_channels': random.choice(space['decoder_channels']),
        }
    else:
        # 精细搜索：基于最佳结果微调
        best_exp = max(history, key=lambda e: e.get('result', {}).get('dice', 0))
        base_config = best_exp['config'].copy()
        
        # 随机微调一个维度
        dim_to_tweak = random.choice(['learning_rate', 'freeze_layers', 'augmentation'])
        
        if dim_to_tweak == 'learning_rate':
            # 在当前值附近微调
            current = base_config['learning_rate']
            options = [lr for lr in SEARCH_SPACE['learning_rate'] 
                      if 0.5 <= lr/current <= 2]
            base_config['learning_rate'] = random.choice(options) if options else current
        elif dim_to_tweak == 'freeze_layers':
            base_config['freeze_layers'] = random.choice(SEARCH_SPACE['freeze_layers'])
        elif dim_to_tweak == 'augmentation':
            base_config['augmentation'] = random.choice(SEARCH_SPACE['augmentation'])
        
        config = base_config
    
    return config


def create_experiment_plan() -> Dict:
    """创建实验计划 - 根据历史自动调整搜索空间"""
    history = load_experiment_history()
    
    # 自动调整搜索空间
    adjusted_space, insights = adjust_search_space_based_on_history(history)
    
    # 选择策略
    strategy = select_strategy(history)
    
    # 生成配置（使用调整后的空间）
    config = generate_next_config(strategy, history, adjusted_space)
    
    # 生成实验ID
    exp_id = generate_experiment_id(config)
    
    # 检查是否重复
    existing_ids = {e['id'] for e in history}
    attempts = 0
    while exp_id in existing_ids and attempts < 10:
        # 微小扰动
        config['seed'] = random.randint(1000, 9999)
        exp_id = generate_experiment_id(config)
        attempts += 1
    
    experiment = {
        'id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy,
        'config': config,
        'status': 'planned',
        'hypothesis': generate_hypothesis(config),
        'adjustment_insights': insights,  # 记录调整洞察
    }
    
    return experiment


def generate_hypothesis(config: Dict) -> str:
    """基于配置生成实验假设"""
    hypotheses = []
    
    if config['freeze_layers'] == 4:
        hypotheses.append("冻结全部encoder可能保护预训练特征")
    elif config['freeze_layers'] == 0:
        hypotheses.append("全量微调可能捕获领域特定特征")
    else:
        hypotheses.append(f"解冻{config['freeze_layers']}层可能平衡泛化和适应")
    
    if config['learning_rate'] <= 5e-5:
        hypotheses.append("低学习率保护预训练权重")
    
    if config['attention']:
        hypotheses.append("Attention机制可能改善肿瘤边界分割")
    
    if config['image_size'] > 256:
        hypotheses.append("更高分辨率可能保留更多细节")
    
    if config['augmentation'] == 'strong':
        hypotheses.append("强增强可能提升泛化能力")
    elif config['augmentation'] == 'weak':
        hypotheses.append("弱增强可能加速收敛")
    
    return "；".join(hypotheses) if hypotheses else "探索性实验"


def analyze_results_and_recommend() -> str:
    """分析结果并给出建议"""
    history = load_experiment_history()
    
    if not history:
        return "尚无实验数据，开始第一轮探索"
    
    # 找出最佳配置
    completed = [e for e in history if e.get('status') == 'completed' and 'result' in e]
    if not completed:
        return "等待实验完成..."
    
    best = max(completed, key=lambda e: e['result']['dice'])
    
    # 分析趋势
    recent = completed[-5:]
    recent_dices = [e['result']['dice'] for e in recent]
    
    trend = "stable"
    if len(recent_dices) >= 3:
        if recent_dices[-1] > recent_dices[0] * 1.02:
            trend = "improving"
        elif recent_dices[-1] < recent_dices[0] * 0.98:
            trend = "declining"
    
    report = f"""
## Auto Research 状态报告

### 当前最佳
- **实验ID**: {best['id']}
- **Dice**: {best['result']['dice']:.4f}
- **IoU**: {best['result'].get('iou', 0):.4f}
- **策略**: {best.get('strategy', 'unknown')}

### 关键发现
- 已完成实验: {len(completed)}
- 趋势: {trend}
- 最佳配置:
```json
{json.dumps(best['config'], indent=2)}
```

### 下一步建议
"""
    
    # 基于最佳配置给出建议
    if best['config'].get('freeze_layers') == 4 and trend == 'improving':
        report += "- 冻结策略有效，建议继续探索不同冻结层数\n"
    
    if best['result']['dice'] < 0.70:
        report += "- 尚未达到目标(0.75+)，继续优化\n"
    elif best['result']['dice'] < 0.75:
        report += "- 接近目标，精细调整\n"
    else:
        report += "- 已达标！考虑鲁棒性验证\n"
    
    return report


if __name__ == '__main__':
    # 生成下一个实验
    experiment = create_experiment_plan()
    
    # 保存
    save_experiment(experiment)
    
    # 显示调整洞察
    insights = experiment.get('adjustment_insights', [])
    
    # 输出
    print(f"""
🧠 BRISC2025 Auto Research
==========================

📊 自动搜索空间调整:
""")
    if insights:
        for insight in insights:
            print(f"  • {insight}")
    else:
        print("  • 使用默认搜索空间")
    
    print(f"""
实验ID: {experiment['id']}
策略: {experiment['strategy']}
假设: {experiment['hypothesis']}

配置:
{json.dumps(experiment['config'], indent=2)}

计划已保存到: {EXPERIMENTS_DIR}/history.jsonl
""")
    
    # 也输出当前状态
    print(analyze_results_and_recommend())
