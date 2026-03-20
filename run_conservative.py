"""
BRISC2025 保守执行方案 - Phase 1 & 2
渐进式优化，可控资源消耗
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# 添加 utils 到路径
sys.path.insert(0, str(Path(__file__).parent))
from utils.tracker import tracker, monitor

# 保守配置 - Phase 1
CONFIG = {
    'data_dir': './data',
    'batch_size': 16,
    'epochs': 5,           # Phase 1: 仅 5 epoch 快速验证
    'lr': 0.001,
    'image_size': 512,
    'num_classes': 4,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'early_stop_patience': 3,  # 3 epoch 无改善则停止
}

print(f"🚀 BRISC2025 Phase 1 - 保守执行")
print(f"设备: {CONFIG['device']}")
print(f"Epochs: {CONFIG['epochs']} (快速验证模式)")
print("-" * 60)


# ==================== 数据加载 ====================

class BRISCDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.class_map = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        split_dir = os.path.join(self.data_dir, self.mode)
        if not os.path.exists(split_dir):
            return
        
        for class_name, class_idx in self.class_map.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_name), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


# ==================== 模型定义 ====================

class ResNet2D(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet2D, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==================== 训练流程 ====================

def train_model(config_name="baseline", use_augmentation=False):
    """
    执行单次实验
    
    Args:
        config_name: 配置名称，用于跟踪
        use_augmentation: 是否使用数据增强
    """
    
    # 检查预算
    can_run, reason = monitor.check_budget(tracker)
    if not can_run:
        print(f"❌ 资源预算耗尽: {reason}")
        return None
    
    print(f"\n📊 开始实验: {config_name}")
    print(f"配置: augmentation={use_augmentation}")
    
    # 准备数据
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = BRISCDataset(CONFIG['data_dir'], transform=train_transform, mode='train')
    val_dataset = BRISCDataset(CONFIG['data_dir'], transform=val_transform, mode='val')
    
    if len(train_dataset) == 0:
        print("❌ 未找到训练数据！请检查 data/train/ 目录")
        return None
    
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 检查数据量 - 保守策略：样本太少则警告
    if len(train_dataset) < 100:
        print("⚠️ 警告: 训练样本少于100，结果可能不可靠")
        print("建议: 使用真实 BRISC2025 数据集")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=2)
    
    # 创建模型
    model = ResNet2D(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    
    # 训练循环
    best_acc = 0.0
    patience_counter = 0
    history = []
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss/len(val_loader),
            'val_acc': val_acc
        })
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{config_name}.pth')
            print(f"  ✓ 保存最佳模型 (Acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"  ⚠️ Early stopping at epoch {epoch+1}")
                break
    
    # 记录实验
    exp_id = tracker.get_experiment_count() + 1
    status = tracker.log_experiment(
        phase=1,
        exp_id=exp_id,
        config=config_name,
        results={
            'val_acc': best_acc / 100,
            'epochs_trained': len(history),
            'final_train_acc': history[-1]['train_acc'] / 100 if history else 0,
            'use_augmentation': use_augmentation
        }
    )
    
    monitor.experiment_count += 1
    
    print(f"\n实验 {exp_id} 完成 - 状态: {status}")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    
    return best_acc


# ==================== 主程序 ====================

def run_phase1():
    """Phase 1: 基础验证"""
    print("\n" + "="*60)
    print("PHASE 1: 基础验证")
    print("="*60)
    
    # 实验 1: 基线（无增强）
    acc1 = train_model("baseline_v1", use_augmentation=False)
    
    if acc1 is None:
        print("❌ 基线实验失败，请检查数据")
        return
    
    # 如果基线效果好，尝试增强
    if acc1 > 30:  # 30% 以上才继续
        print("\n基线效果良好，继续 Phase 1.2...")
        acc2 = train_model("augmentation_v1", use_augmentation=True)
        
        # 比较结果
        if acc2 and acc2 > acc1:
            print(f"\n✓ 数据增强有效: {acc1:.2f}% → {acc2:.2f}%")
        else:
            print(f"\n~ 数据增强无明显提升")
    else:
        print(f"\n⚠️ 基线准确率 {acc1:.2f}% 偏低")
        print("建议:")
        print("1. 检查数据质量（是否真实 MRI 图像）")
        print("2. 检查类别平衡")
        print("3. 考虑使用预训练权重")
    
    # 打印摘要
    tracker.print_summary()
    monitor.print_resource_usage()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, help='执行阶段')
    parser.add_argument('--summary', action='store_true', help='仅打印摘要')
    
    args = parser.parse_args()
    
    if args.summary:
        tracker.print_summary()
        monitor.print_resource_usage()
    elif args.phase == 1:
        run_phase1()
    else:
        print(" Usage:")
        print("  python run_conservative.py --phase 1    # 运行 Phase 1")
        print("  python run_conservative.py --summary    # 查看实验摘要")
