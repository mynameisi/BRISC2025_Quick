# BRISC2025 数据下载指南

## 方法一：Kaggle 网页下载（推荐）

1. 访问 https://www.kaggle.com/datasets/briscdataset/brisc2025/data
2. 点击 "Download" 按钮
3. 解压后将 `Training` 和 `Testing` 文件夹放入以下目录：

```
~/BRISC2025_Quick/data/
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

## 方法二：如果 Kaggle 访问困难

我可以创建一个**轻量级数据生成器**，用公开医学影像数据集（如 BraTS）的子集来验证代码逻辑。

## 数据验证脚本

下载完成后运行以下命令验证数据完整性：

```bash
cd ~/BRISC2025_Quick
source venv/bin/activate
python -c "
import os

base = 'data'
for split in ['train', 'val']:
    print(f'\\n{split.upper()}:')
    for cls in ['glioma', 'meningioma', 'pituitary', 'notumor']:
        path = os.path.join(base, split, cls)
        if os.path.exists(path):
            n = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
            print(f'  {cls}: {n} images')
        else:
            print(f'  {cls}: MISSING!')
"
```

## 预期数据量

根据官方说明：
- 训练集：约 4500-5000 张
- 测试集：约 1000-1500 张
- 图像尺寸：512×512
- 格式：JPG/PNG

## 下一步

数据准备好后告诉我，我会启动 Phase 1 验证（5 epoch 快速测试）。
