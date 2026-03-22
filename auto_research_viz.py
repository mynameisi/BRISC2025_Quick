"""
BRISC2025 Auto Research - 可视化与优化分析
生成优化轨迹图、配置对比图等
"""
import json
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

EXPERIMENTS_DIR = Path('experiments/auto_research')
FIGURES_DIR = EXPERIMENTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

def load_all_experiments() -> List[Dict]:
    """加载所有实验数据"""
    history_file = EXPERIMENTS_DIR / 'history.jsonl'
    if not history_file.exists():
        return []
    
    experiments = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    return experiments

def generate_optimization_trajectory():
    """生成优化轨迹图 - 显示Dice随实验次数的变化"""
    experiments = load_all_experiments()
    completed = [e for e in experiments if e.get('status') == 'completed' and 'result' in e]
    
    if len(completed) < 1:
        print("⚠️ 实验数量不足，无法生成优化轨迹")
        return None
    
    # 按时间排序
    completed.sort(key=lambda x: x.get('timestamp', ''))
    
    # 提取数据
    exp_ids = [e['id'][:6] for e in completed]
    dices = [e['result']['dice'] for e in completed]
    ious = [e['result'].get('iou', 0) for e in completed]
    strategies = [e.get('strategy', 'unknown') for e in completed]
    
    # 计算最佳值轨迹
    best_so_far = []
    current_best = 0
    for d in dices:
        current_best = max(current_best, d)
        best_so_far.append(current_best)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BRISC2025 Auto Research - Optimization Trajectory', fontsize=16, fontweight='bold')
    
    # 1. Dice优化轨迹
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if s == 'exploitation' else '#3498db' for s in strategies]
    ax1.scatter(range(len(dices)), dices, c=colors, s=100, alpha=0.7, zorder=3)
    ax1.plot(range(len(dices)), dices, 'o-', alpha=0.3, color='gray', linewidth=1)
    ax1.plot(range(len(best_so_far)), best_so_far, 'r-', linewidth=2, label='Best So Far')
    
    # 标记最佳点
    best_idx = np.argmax(dices)
    ax1.scatter(best_idx, dices[best_idx], s=300, c='red', marker='*', 
               edgecolors='black', linewidths=2, zorder=5, label=f'Best: {dices[best_idx]:.4f}')
    
    ax1.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='Target (0.75)')
    ax1.axhline(y=0.6836, color='orange', linestyle='--', alpha=0.5, label='Baseline (0.6836)')
    ax1.set_xlabel('Experiment Number', fontsize=12)
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_title('Dice Score Optimization', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 添加图例说明颜色
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Exploration'),
        Patch(facecolor='#2ecc71', label='Exploitation'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Best So Far'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # 2. IoU优化轨迹
    ax2 = axes[0, 1]
    ax2.plot(range(len(ious)), ious, 'o-', color='purple', alpha=0.7)
    ax2.axhline(y=0.65, color='g', linestyle='--', alpha=0.5, label='Target (0.65)')
    ax2.axhline(y=0.5374, color='orange', linestyle='--', alpha=0.5, label='Baseline (0.5374)')
    ax2.set_xlabel('Experiment Number', fontsize=12)
    ax2.set_ylabel('IoU Score', fontsize=12)
    ax2.set_title('IoU Score Optimization', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 配置参数热力图
    ax3 = axes[1, 0]
    
    # 准备热力图数据
    param_names = ['freeze_layers', 'learning_rate', 'batch_size', 'image_size']
    param_labels = ['Freeze\nLayers', 'Learning\nRate', 'Batch\nSize', 'Image\nSize']
    
    # 归一化参数值
    heatmap_data = []
    for e in completed:
        config = e['config']
        row = [
            config.get('freeze_layers', 0) / 4,  # 0-4
            np.log10(config.get('learning_rate', 1e-4)) / (-4) + 1,  # 归一化
            config.get('batch_size', 16) / 16,  # 0.5-1
            config.get('image_size', 256) / 384,  # 0.67-1
        ]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax3.imshow(heatmap_data.T, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(completed)))
    ax3.set_xticklabels([f"{i+1}\n({e['id'][:4]})" for i, e in enumerate(completed)], fontsize=8)
    ax3.set_yticks(range(len(param_labels)))
    ax3.set_yticklabels(param_labels)
    ax3.set_title('Configuration Heatmap\n(color intensity = normalized value)', fontsize=14)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # 在每个格子上标注策略
    for i in range(len(completed)):
        strategy = completed[i].get('strategy', 'unknown')[0].upper()
        ax3.text(i, -0.5, strategy, ha='center', va='top', fontsize=7, 
                color='green' if strategy == 'E' else 'blue')
    
    # 4. 策略效果对比
    ax4 = axes[1, 1]
    
    exploration_dices = [e['result']['dice'] for e in completed if e.get('strategy') == 'exploration']
    exploitation_dices = [e['result']['dice'] for e in completed if e.get('strategy') == 'exploitation']
    
    if exploration_dices or exploitation_dices:
        data_to_plot = []
        labels = []
        colors = []
        
        if exploration_dices:
            data_to_plot.append(exploration_dices)
            labels.append(f'Exploration\n(n={len(exploration_dices)})')
            colors.append('#3498db')
        
        if exploitation_dices:
            data_to_plot.append(exploitation_dices)
            labels.append(f'Exploitation\n(n={len(exploitation_dices)})')
            colors.append('#2ecc71')
        
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.axhline(y=0.6836, color='orange', linestyle='--', alpha=0.5, label='Baseline')
        ax4.set_ylabel('Dice Score', fontsize=12)
        ax4.set_title('Strategy Comparison', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = FIGURES_DIR / f'optimization_trajectory_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 优化轨迹图已保存: {save_path}")
    return save_path


def generate_config_performance_comparison():
    """生成配置性能对比图"""
    experiments = load_all_experiments()
    completed = [e for e in experiments if e.get('status') == 'completed' and 'result' in e]
    
    if len(completed) < 2:
        print("⚠️ 实验数量不足，无法生成配置对比")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Configuration vs Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Freeze Layers vs Dice
    ax1 = axes[0, 0]
    freeze_data = {}
    for e in completed:
        fl = e['config'].get('freeze_layers', 0)
        dice = e['result']['dice']
        if fl not in freeze_data:
            freeze_data[fl] = []
        freeze_data[fl].append(dice)
    
    x_pos = sorted(freeze_data.keys())
    means = [np.mean(freeze_data[x]) for x in x_pos]
    stds = [np.std(freeze_data[x]) for x in x_pos]
    
    ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Freeze Layers', fontsize=12)
    ax1.set_ylabel('Average Dice Score', fontsize=12)
    ax1.set_title('Freeze Layers vs Performance', fontsize=14)
    ax1.axhline(y=0.6836, color='orange', linestyle='--', alpha=0.5, label='Baseline')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Learning Rate vs Dice
    ax2 = axes[0, 1]
    lr_data = {}
    for e in completed:
        lr = e['config'].get('learning_rate', 1e-4)
        dice = e['result']['dice']
        if lr not in lr_data:
            lr_data[lr] = []
        lr_data[lr].append(dice)
    
    lrs = sorted(lr_data.keys())
    lr_means = [np.mean(lr_data[lr]) for lr in lrs]
    
    ax2.semilogx(lrs, lr_means, 'o-', linewidth=2, markersize=10, color='green')
    ax2.set_xlabel('Learning Rate', fontsize=12)
    ax2.set_ylabel('Average Dice Score', fontsize=12)
    ax2.set_title('Learning Rate vs Performance', fontsize=14)
    ax2.axhline(y=0.6836, color='orange', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Batch Size vs Dice
    ax3 = axes[1, 0]
    batch_data = {}
    for e in completed:
        bs = e['config'].get('batch_size', 16)
        dice = e['result']['dice']
        if bs not in batch_data:
            batch_data[bs] = []
        batch_data[bs].append(dice)
    
    batches = sorted(batch_data.keys())
    batch_means = [np.mean(batch_data[bs]) for bs in batches]
    
    ax3.bar(batches, batch_means, alpha=0.7, color='coral', edgecolor='darkred')
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Average Dice Score', fontsize=12)
    ax3.set_title('Batch Size vs Performance', fontsize=14)
    ax3.axhline(y=0.6836, color='orange', linestyle='--', alpha=0.5, label='Baseline')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 3D散点图：Dice vs IoU vs Epochs
    ax4 = axes[1, 1]
    
    dices = [e['result']['dice'] for e in completed]
    ious = [e['result'].get('iou', 0) for e in completed]
    epochs = [e['result'].get('epochs', 0) for e in completed]
    
    scatter = ax4.scatter(dices, ious, c=epochs, s=200, alpha=0.6, 
                         cmap='viridis', edgecolors='black', linewidths=1)
    
    # 标记最佳点
    best_idx = np.argmax(dices)
    ax4.scatter(dices[best_idx], ious[best_idx], s=400, c='red', 
               marker='*', edgecolors='black', linewidths=2, 
               label=f'Best: Dice={dices[best_idx]:.4f}')
    
    ax4.set_xlabel('Dice Score', fontsize=12)
    ax4.set_ylabel('IoU Score', fontsize=12)
    ax4.set_title('Dice vs IoU (color = epochs)', fontsize=14)
    ax4.axhline(y=0.5374, color='orange', linestyle='--', alpha=0.5)
    ax4.axvline(x=0.6836, color='orange', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Epochs', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = FIGURES_DIR / f'config_performance_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 配置性能对比图已保存: {save_path}")
    return save_path


def generate_best_config_card():
    """生成最佳配置卡片"""
    experiments = load_all_experiments()
    completed = [e for e in experiments if e.get('status') == 'completed' and 'result' in e]
    
    if not completed:
        print("⚠️ 没有完成的实验")
        return None
    
    best = max(completed, key=lambda e: e['result']['dice'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # 背景色
    fig.patch.set_facecolor('#f8f9fa')
    
    # 标题
    ax.text(0.5, 0.95, '🏆 BRISC2025 Auto Research - Best Configuration', 
            fontsize=20, ha='center', va='top', fontweight='bold', transform=ax.transAxes)
    
    # 实验ID和分数
    ax.text(0.5, 0.85, f'Experiment ID: {best["id"]}', 
            fontsize=14, ha='center', transform=ax.transAxes, color='#666')
    
    dice = best['result']['dice']
    iou = best['result'].get('iou', 0)
    
    # 大分数显示
    ax.text(0.25, 0.70, f'Dice', fontsize=16, ha='center', transform=ax.transAxes, color='#333')
    ax.text(0.25, 0.55, f'{dice:.4f}', fontsize=48, ha='center', transform=ax.transAxes, 
            fontweight='bold', color='#2ecc71' if dice > 0.7 else '#3498db')
    
    ax.text(0.75, 0.70, f'IoU', fontsize=16, ha='center', transform=ax.transAxes, color='#333')
    ax.text(0.75, 0.55, f'{iou:.4f}', fontsize=48, ha='center', transform=ax.transAxes, 
            fontweight='bold', color='#9b59b6')
    
    # 配置详情
    config_text = f"""
Configuration Details:
• Learning Rate: {best['config'].get('learning_rate', 'N/A')}
• Freeze Layers: {best['config'].get('freeze_layers', 'N/A')}
• Augmentation: {best['config'].get('augmentation', 'N/A')}
• Batch Size: {best['config'].get('batch_size', 'N/A')}
• Image Size: {best['config'].get('image_size', 'N/A')}
• Attention: {best['config'].get('attention', 'N/A')}

Training Info:
• Strategy: {best.get('strategy', 'N/A')}
• Epochs: {best['result'].get('epochs', 'N/A')}
• Hypothesis: {best.get('hypothesis', 'N/A')[:60]}...
    """
    
    ax.text(0.5, 0.25, config_text, fontsize=11, ha='center', va='top', 
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ddd'))
    
    # 与baseline对比
    baseline_dice = 0.6836
    improvement = (dice - baseline_dice) / baseline_dice * 100
    
    comparison = f"vs Baseline ({baseline_dice:.4f}): "
    if improvement > 0:
        comparison += f"+{improvement:.2f}% ↑"
        color = '#27ae60'
    else:
        comparison += f"{improvement:.2f}% ↓"
        color = '#e74c3c'
    
    ax.text(0.5, 0.05, comparison, fontsize=14, ha='center', transform=ax.transAxes, 
            fontweight='bold', color=color)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = FIGURES_DIR / f'best_config_card_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    
    print(f"✅ 最佳配置卡片已保存: {save_path}")
    return save_path


def generate_all_visualizations():
    """生成所有可视化图表"""
    print("🎨 生成 Auto Research 可视化图表...")
    print("=" * 50)
    
    paths = []
    
    # 1. 优化轨迹图
    path1 = generate_optimization_trajectory()
    if path1:
        paths.append(path1)
    
    # 2. 配置性能对比图
    path2 = generate_config_performance_comparison()
    if path2:
        paths.append(path2)
    
    # 3. 最佳配置卡片
    path3 = generate_best_config_card()
    if path3:
        paths.append(path3)
    
    print("\n" + "=" * 50)
    print(f"✅ 已生成 {len(paths)} 张图表")
    print(f"保存位置: {FIGURES_DIR}/")
    
    for p in paths:
        print(f"  - {p.name}")
    
    return paths


if __name__ == '__main__':
    generate_all_visualizations()
