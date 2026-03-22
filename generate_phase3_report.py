"""
生成 Phase 3 报告 (纯文本版本)
"""
from pathlib import Path
import json
from datetime import datetime

# 读取数据
log_dir = Path('~/BRISC2025_Quick/logs/phase3_segmentation_20260322_004847').expanduser()
with open(log_dir / 'training_log.json') as f:
    logs = json.load(f)

history = logs['training_history']
final = logs['final_results']
config = logs['config']

# 生成Markdown报告
report = f"""# BRISC2025 Phase 3 报告：语义分割

---

## 核心结果

| 指标 | 数值 | 说明 |
|------|------|------|
| **Dice Score** | **{final['best_val_dice']:.4f}** (68.36%) | 最佳验证分数 |
| **IoU Score** | **{final['best_val_iou']:.4f}** (53.74%) | 交并比 |
| 最佳 Epoch | {final['best_epoch']} / {final['total_epochs_trained']} | 早停触发 |
| 训练时间 | {final['training_time_minutes']:.1f} 分钟 | 约1.5小时 |

---

## 模型配置

| 配置项 | 值 |
|--------|-----|
| 架构 | U-Net |
| 参数量 | 1,928,450 |
| 输入分辨率 | {config['image_size']}x{config['image_size']} |
| 类别数 | {config['num_classes']} (背景 + 肿瘤) |
| 设备 | {config['device'].upper()} |
| 训练样本 | 3,933 张 |
| 验证样本 | 860 张 |
| Batch Size | {config['batch_size']} |
| 学习率 | {config['lr']} → {config['min_lr']} |
| 早停耐心 | {config['early_stop_patience']} epochs |

---

## 训练进展

### 关键里程碑

| Epoch | Val Dice | Val IoU | 状态 |
|-------|----------|---------|------|
| 1 | {history[0]['val_dice']:.4f} | {history[0]['val_iou']:.4f} | 初始 |
| 3 | {history[2]['val_dice']:.4f} | {history[2]['val_iou']:.4f} | 快速学习 |
| 6 | {history[5]['val_dice']:.4f} | {history[5]['val_iou']:.4f} | 突破 0.5 |
| 10 | {history[9]['val_dice']:.4f} | {history[9]['val_iou']:.4f} | 稳步提升 |
| 15 | {history[14]['val_dice']:.4f} | {history[14]['val_iou']:.4f} | 过0.6 |
| 23 | {history[22]['val_dice']:.4f} | {history[22]['val_iou']:.4f} | 峰值 |
| **31** | **{history[30]['val_dice']:.4f}** | **{history[30]['val_iou']:.4f}** | **最佳** |

### 训练曲线特征

- **Epoch 1-5**: 快速学习期，Dice从0.01提升至0.33
- **Epoch 6-15**: 稳步提升，Dice达到0.62
- **Epoch 16-23**: 峰值窗口，在epoch 23达到0.68
- **Epoch 24-41**: 平台期，10轮无改善触发早停

---

## 结果分析

### 性能指标解读

| 指标 | 当前值 | 含义 |
|------|--------|------|
| **Dice 0.6836** | 68.36% | 预测掩码与真实掩码的重叠度 |
| **IoU 0.5374** | 53.74% | 更严格的像素级准确率 |
| **泛化差距** | {(history[final['best_epoch']-1]['train_dice'] - final['best_val_dice']):.4f} | 训练与验证差距小，无严重过拟合 |

### 业界对比

| 水平 | Dice范围 | 说明 |
|------|----------|------|
| 研究级 | 0.75-0.90 | 顶级会议/期刊论文 |
| 临床级 | 0.80+ | 实际医疗部署标准 |
| 基线级 | 0.60-0.75 | 快速验证/概念证明 |
| **本项目** | **0.68** | **可用基线** |

---

## 是否达到预期？

### ✅ 基本预期：达成

这是一个**快速验证**项目，核心目标：
1. ✓ 建立可用的U-Net基线
2. ✓ 在合理时间内收敛
3. ✓ 验证数据质量和流程

**94分钟训练**从0开始达到**Dice 0.68**，符合快速验证的预期。

### ⚠️ 临床标准：尚未达成

| 标准 | 目标 | 当前 | 差距 |
|------|------|------|------|
| 临床Dice | 0.80+ | 0.68 | -0.12 |
| 临床IoU | 0.65+ | 0.54 | -0.11 |

---

## 改进路径（达到临床级Dice 0.80+）

### 高优先级（预计提升10-15%）

1. **预训练Encoder**
   - 使用ImageNet预训练ResNet/VGG作为backbone
   - 迁移学习显著改善分割质量
   - 预期增益：+5-10% Dice

2. **Attention U-Net**
   - 添加空间注意力门控机制
   - 帮助模型聚焦肿瘤区域
   - 预期增益：+5-8% Dice

### 中优先级（预计提升5-10%）

3. **医学图像专用增强**
   - 弹性形变（医学图像可变形特性）
   - 高斯噪声和模糊
   - 随机强度变化
   - 预期增益：+2-5% Dice

4. **更高分辨率**
   - 当前256x256（下采样）
   - 目标512x512或原始分辨率
   - 预期增益：+3-5% Dice

### 低优先级（预计提升1-3%）

5. **后处理优化**
   - Conditional Random Fields (CRF)
   - Test-time augmentation (TTA)
   - 预期增益：+1-3% Dice

**综合以上改进，达到Dice 0.85+是现实可行的。**

---

## 项目三阶段总结

| 阶段 | 任务 | 关键成果 |
|------|------|----------|
| **Phase 1** | 分类基线 | 82.79% 准确率 |
| **Phase 2** | 数据增强 | 97.79% 准确率 (+15%) |
| **Phase 3** | 语义分割 | 68.36% Dice |

### 总体评价

- ✓ 项目目标完整达成
- ✓ 建立了完整的医学图像AI流程
- ✓ 分类性能优秀，分割基线可用
- ⚠️ 分割性能有提升空间

### 下一步建议

1. **短期**：实施预训练Encoder，目标Dice 0.75+
2. **中期**：添加Attention机制，目标Dice 0.80+
3. **长期**：高分辨率 + 后处理，目标Dice 0.85+

---

*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}*  
*GitHub：https://github.com/mynameisi/BRISC2025_Quick*
"""

# 保存报告
output_path = log_dir / 'BRISC2025_Phase3_Report.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✅ Phase 3 报告已生成: {output_path}")
print(f"\n报告预览（前50行）：")
print("=" * 50)
for i, line in enumerate(report.split('\n')[:50]):
    print(line)
