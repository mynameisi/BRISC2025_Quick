"""
BRISC2025 Quick Verification
快速验证版本 - 小样本 + 少epoch
作者: OpenClaw Agent (大白)
日期: 2026-03-21
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report

# 配置
CONFIG = {
    'data_dir': './data',  # 数据目录
    'batch_size': 16,      # 降低batch size适配Mac内存
    'epochs': 20,          # 快速验证epoch数
    'lr': 0.001,
    'image_size': 512,
    'num_classes': 4,      # Glioma, Meningioma, Pituitary, Non-tumor
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
}

print(f"使用设备: {CONFIG['device']}")
print(f"PyTorch版本: {torch.__version__}")


# ==================== 1. 数据预处理 ====================

class BRISCDataset(Dataset):
    """BRISC2025 数据集加载器"""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # 类别映射
        self.class_map = {
            'glioma': 0,
            'meningioma': 1,
            'pituitary': 2,
            'notumor': 3
        }
        
        # 加载图像路径和标签
        self.samples = []
        self._load_samples()
        
        print(f"[{mode}] 加载了 {len(self.samples)} 个样本")
    
    def _load_samples(self):
        """加载样本路径"""
        # 假设目录结构: data/train/glioma/*.jpg
        split_dir = os.path.join(self.data_dir, self.mode)
        
        if not os.path.exists(split_dir):
            print(f"警告: {split_dir} 不存在，请检查数据路径")
            return
        
        for class_name, class_idx in self.class_map.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(mode='train'):
    """获取数据增强变换"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


# ==================== 2. 模型定义 ====================

class ResNet2D(nn.Module):
    """ResNet50 用于脑肿瘤分类"""
    
    def __init__(self, num_classes=4):
        super(ResNet2D, self).__init__()
        # 使用预训练ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 修改第一层以适应灰度图(复制到3通道)
        # 保持原有权重，但输入会被复制为3通道
        
        # 修改最后一层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def create_model():
    """创建模型"""
    model = ResNet2D(num_classes=CONFIG['num_classes'])
    return model.to(CONFIG['device'])


# ==================== 3. 训练流程 ====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(dataloader), 100.*correct/total


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100.*correct/total
    return running_loss/len(dataloader), acc, all_preds, all_labels


def train_model():
    """完整训练流程"""
    
    # 创建数据集
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = BRISCDataset(CONFIG['data_dir'], transform=train_transform, mode='train')
    val_dataset = BRISCDataset(CONFIG['data_dir'], transform=val_transform, mode='val')
    
    if len(train_dataset) == 0:
        print("错误: 未找到训练数据。请检查数据目录结构:")
        print("  data/train/glioma/*.jpg")
        print("  data/train/meningioma/*.jpg")
        print("  data/train/pituitary/*.jpg")
        print("  data/train/notumor/*.jpg")
        return
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=2
    )
    
    # 创建模型
    model = create_model()
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 训练历史
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    print(f"\n开始训练 {CONFIG['epochs']} epochs...")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 50)
        
        # 训练和验证
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ 保存最佳模型 (Acc: {best_acc:.2f}%)")
    
    print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")
    
    # 绘制训练曲线
    plot_history(history)
    
    # 输出分类报告
    print("\n分类报告:")
    class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    print(classification_report(labels, preds, target_names=class_names))
    
    return history, best_acc


def plot_history(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n训练曲线已保存到 training_history.png")


# ==================== 4. 数据准备工具 ====================

def prepare_sample_data():
    """
    创建模拟数据用于代码验证（当真实数据不可用时）
    """
    import random
    from PIL import Image
    
    print("创建模拟数据用于代码验证...")
    
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    splits = ['train', 'val']
    
    for split in splits:
        for cls in classes:
            dir_path = os.path.join('data', split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            # 每个类别创建50张训练图，10张验证图
            num_samples = 50 if split == 'train' else 10
            
            for i in range(num_samples):
                # 创建随机灰度图（模拟MRI）
                img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
                # 添加一些结构使其看起来像医学图像
                center_x, center_y = 256, 256
                Y, X = np.ogrid[:512, :512]
                dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                mask = dist_from_center <= np.random.randint(30, 100)
                img[mask] = np.clip(img[mask] + 50, 0, 255)
                
                pil_img = Image.fromarray(img)
                pil_img.save(os.path.join(dir_path, f'{cls}_{i:03d}.jpg'))
    
    print("✓ 模拟数据创建完成")
    print("  训练集: 200张 (每类50张)")
    print("  验证集: 40张 (每类10张)")


# ==================== 主程序 ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BRISC2025 Quick Verification')
    parser.add_argument('--prepare-data', action='store_true', help='创建模拟数据')
    parser.add_argument('--train', action='store_true', help='开始训练')
    parser.add_argument('--data-dir', default='./data', help='数据目录')
    
    args = parser.parse_args()
    
    CONFIG['data_dir'] = args.data_dir
    
    if args.prepare_data:
        prepare_sample_data()
    
    if args.train:
        train_model()
    
    if not args.prepare_data and not args.train:
        print("BRISC2025 快速验证工具")
        print("=" * 50)
        print("\n使用方法:")
        print("1. 创建模拟数据: python train.py --prepare-data")
        print("2. 开始训练:     python train.py --train")
        print("\n或使用真实数据:")
        print("  python train.py --train --data-dir /path/to/brisc2025")
        print("\n数据目录结构:")
        print("  data/")
        print("    ├── train/")
        print("    │   ├── glioma/")
        print("    │   ├── meningioma/")
        print("    │   ├── pituitary/")
        print("    │   └── notumor/")
        print("    └── val/")
        print("        ├── glioma/")
        print("        ├── meningioma/")
        print("        ├── pituitary/")
        print("        └── notumor/")
