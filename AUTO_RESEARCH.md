# BRISC2025 Auto Research 自动化实验系统

基于 Karpathy Auto Research 方法论的自改进AI实验系统。

## 核心思想

```
定义目标 → AI生成实验 → 自动运行 → 收集结果 → 分析改进 → 循环
```

## 搜索空间

| 维度 | 范围 | 当前最佳发现 |
|------|------|-------------|
| learning_rate | 1e-5 ~ 5e-4 | 待探索 |
| freeze_layers | 0, 2, 4 | 4层冻结效果较好 |
| augmentation | weak/medium/strong | 待探索 |
| attention | True/False | 待探索 |
| image_size | 256/384 | 256 |
| batch_size | 8/16 | 16 |

## 快速开始

### 本地运行

```bash
# 运行1次实验
./run_auto_research.sh

# 运行5次实验
./run_auto_research.sh 5
```

### GitHub Actions 自动运行

已配置 `.github/workflows/auto_research.yml`：
- 每4小时自动运行一次
- 或手动触发 `workflow_dispatch`

## 文件结构

```
experiments/auto_research/
├── history.jsonl          # 所有实验历史
├── latest_notification.txt # 最新通知
├── figures/               # 可视化图表
│   ├── optimization_trajectory_*.png
│   ├── config_performance_*.png
│   └── best_config_card_*.png
├── 882528f9/              # 单个实验目录
│   ├── result.json        # 实验结果
│   └── best_model.pth     # 最佳模型
└── ...
```

## 📊 可视化分析

每次实验完成后，系统会自动生成优化图表：

| 图表类型 | 说明 |
|---------|------|
| `optimization_trajectory_*.png` | 优化轨迹图，显示Dice/IoU随实验次数变化 |
| `config_performance_*.png` | 配置性能对比，分析各参数影响 |
| `best_config_card_*.png` | 最佳配置卡片，突出显示最佳实验 |

图表保存在：`experiments/auto_research/figures/`

## 实验结果解读

### 当前最佳基线
- **Dice**: 0.6836 (vanilla U-Net, 41 epochs)
- **Dice**: 0.6630 (ResNet34 frozen, 3 epochs)

### 目标
- **临床级**: Dice 0.80+
- **当前差距**: 约 0.12

## 策略说明

### Exploration（探索）
- 早期阶段（<3次实验）
- 随机采样搜索空间
- 发现有效配置

### Exploitation（利用）
- 后期阶段
- 基于最佳结果微调
- 精细优化参数

## 通知配置

### Feishu 通知
```bash
export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
python auto_research_notify.py
```

## 手动触发单次实验

```bash
# 生成实验计划
python auto_research.py

# 执行实验
python auto_research_runner.py

# 生成可视化图表
python auto_research_viz.py

# 发送通知
python auto_research_notify.py
```

## 查看历史

```bash
# 查看所有实验
python auto_research.py

# 生成最新可视化
python auto_research_viz.py

# 或查看历史文件
cat experiments/auto_research/history.jsonl
```

## 原理说明

系统会：
1. 根据历史结果选择探索策略
2. 生成新的实验配置
3. 自动运行训练
4. 记录结果
5. 生成可视化图表
6. 分析趋势并推荐下一步

长期运行可自动收敛到最优解。
