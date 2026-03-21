# BRISC2025

BRISC2025 (Brain Tumor Image Segmentation and Classification 2025) - 快速验证项目

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目是针对 [BRISC2025 数据集](https://www.kaggle.com/datasets/briscdataset/brisc2025) 的脑肿瘤分类快速验证实现，采用渐进式优化策略：

- **Phase 1**: 基线验证（已完成 ✅）
- **Phase 2**: 数据增强优化（待进行）
- **Phase 3**: 分割任务（待进行）

## 🎯 Phase 1 成果

| 指标 | 结果 |
|------|------|
| **验证准确率** | **81.20%** |
| 训练准确率 | 83.40% |
| 训练 Epochs | 2 / 5 |
| 数据类型 | 真实数据 (Real Data) |
| 数据增强 | ❌ 无 |

### 模型详情
- **架构**: ResNet50 (预训练)
- **输入尺寸**: 512×512
- **类别数**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **设备**: Apple Silicon (MPS)
- **Batch Size**: 16

## 📁 项目结构

```
BRISC2025_Quick/
├── 📄 README.md                 # 本文件
├── 📄 DATA_GUIDE.md             # 数据下载指南
├── 📄 PHASE1_REPORT.md          # Phase 1 详细报告
├── 🐍 train.py                  # 完整训练脚本
├── 🐍 run_conservative.py       # 保守执行方案 (Phase 1)
├── 🔧 check_status.sh           # 状态检查脚本
├── 📁 utils/                    # 工具模块
│   ├── tracker.py              # 实验跟踪
│   └── monitor.py              # 资源监控
├── 📁 experiments/              # 实验记录
│   └── experiment_log.jsonl    # 实验日志
├── 💾 best_baseline_v1.pth      # Phase 1 最佳模型 (90MB)
├── 💾 best_model.pth            # 备用模型 (90MB)
└── 📁 data/                     # 数据集目录 (需下载)
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── notumor/
    └── val/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── notumor/
```

## 🚀 快速开始

### 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/mynameisi/BRISC2025.git
cd BRISC2025

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install torch torchvision numpy pillow matplotlib tqdm scikit-learn
```

### 数据准备

1. 访问 [Kaggle BRISC2025](https://www.kaggle.com/datasets/briscdataset/brisc2025/data)
2. 下载并解压数据到 `data/` 目录
3. 验证数据完整性：

```bash
python -c "
import os
for split in ['train', 'val']:
    print(f'{split.upper()}:')
    for cls in ['glioma', 'meningioma', 'pituitary', 'notumor']:
        path = os.path.join('data', split, cls)
        n = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
        print(f'  {cls}: {n} images')
"
```

### 运行 Phase 1 验证

```bash
# 快速验证（5 epochs）
python run_conservative.py

# 完整训练（20 epochs）
python train.py
```

### 检查状态

```bash
./check_status.sh
```

## 📊 实验记录

所有实验自动记录在 `experiments/experiment_log.jsonl`：

```json
{
  "timestamp": "2026-03-21T03:28:00",
  "phase": 1,
  "experiment_id": 1,
  "config": "baseline_v1 (real_data)",
  "results": {
    "val_acc": 0.812,
    "epochs_trained": 2,
    "final_train_acc": 0.834,
    "use_augmentation": false
  },
  "status": "success"
}
```

## 🗺️ 路线图

- [x] **Phase 1**: 基线验证 (81.2% 准确率)
- [ ] **Phase 2**: 数据增强 + 超参优化 → 目标 86%+
- [ ] **Phase 3**: 语义分割任务
- [ ] **Phase 4**: 模型轻量化与部署

## 💡 关键设计决策

### 1. 保守执行策略
- Phase 1 仅运行 5 epochs 快速验证
- Early stopping ( patience=3 )
- 资源受限时自动降级

### 2. 渐进式优化
- 从简单基线开始
- 每阶段验证后再进行下一阶段
- 避免过早优化

### 3. Mac 友好
- 支持 MPS (Apple Silicon GPU)
- 降低 batch size 适配内存
- 无 CUDA 依赖

## 📦 模型下载

Phase 1 训练好的模型：

| 文件 | 大小 | 描述 |
|------|------|------|
| `best_baseline_v1.pth` | 90MB | Phase 1 最佳模型 |

> ⚠️ 模型文件使用 Git LFS 管理，克隆时需安装 [Git LFS](https://git-lfs.github.com/)

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- 数据集: [BRISC2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)
- 框架: [PyTorch](https://pytorch.org/)
- 实现: OpenClaw Agent (大白)

---

**最后更新**: 2026-03-21  
**当前阶段**: Phase 1 ✅ (已完成)
