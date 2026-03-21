"""
BRISC2025 Phase 2 - 数据增强优化 + 延长训练
目标: 从 82.79% 提升到 86-88%
改进:
1. 数据增强 (RandomRotation, HorizontalFlip, ColorJitter)
2. 学习率调度 (CosineAnnealingLR)
3. 延长训练 (20 epochs, patience=5)
4. 更详细的日志和可视化
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
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# 配置 - Phase 2
CONFIG = {
    'data_dir': './data',
    'batch_size': 16,
    'epochs': 20,              # Phase 2: 延长到 20 epochs
    'lr': 0.001,
    'min_lr': 1e-6,            # 学习率调度最小值
    'image_size': 512,
    'num_classes': 3,          # glioma, meningioma, pituitary
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'early_stop_patience': 5,  # Phase 2: 增加 patience 到 5
    'seed': 42,
}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# 创建日志目录
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_LOG_DIR = LOG_DIR / f'phase2_augmented_{RUN_ID}'
RUN_LOG_DIR.mkdir(exist_ok=True)

print(f"🚀 BRISC2025 Phase 2 - 数据增强优化")
print(f"运行ID: {RUN_ID}")
print(f"设备: {CONFIG['device']}")
print(f"Epochs: {CONFIG['epochs']} (Patience: {CONFIG['early_stop_patience']})")
print(f"学习率调度: CosineAnnealingLR")
print("=" * 60)


# ==================== 数据集 ====================

class BRISCDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.class_map = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
        self.class_names = list(self.class_map.keys())
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
    
    def get_class_distribution(self):
        """获取类别分布"""
        dist = defaultdict(int)
        for _, label in self.samples:
            dist[self.class_names[label]] += 1
        return dict(dist)


def get_transforms(mode='train'):
    """Phase 2: 添加数据增强"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            # Phase 2 新增: 数据增强
            transforms.RandomRotation(degrees=15),           # 随机旋转 ±15°
            transforms.RandomHorizontalFlip(p=0.5),          # 水平翻转
            transforms.ColorJitter(                          # 颜色抖动
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])


# ==================== 模型 ====================

class ResNet2D(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet2D, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# ==================== 详细日志记录器 ====================

class DetailedLogger:
    """详细训练日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logs = {
            'config': {},
            'data_info': {},
            'training_history': [],
            'final_results': {},
            'timing': {},
        }
        
        self.start_time = time.time()
        
    def log_config(self, config):
        """记录配置"""
        self.logs['config'] = config.copy()
        self.logs['config']['device_name'] = 'MPS' if 'mps' in str(config['device']) else 'CPU'
        
    def log_data_info(self, train_dataset, val_dataset):
        """记录数据信息"""
        self.logs['data_info'] = {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'train_class_distribution': train_dataset.get_class_distribution(),
            'val_class_distribution': val_dataset.get_class_distribution(),
            'num_classes': len(train_dataset.class_names),
            'class_names': train_dataset.class_names,
        }
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """记录每个epoch的指标"""
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'learning_rate': float(lr),
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
        }
        self.logs['training_history'].append(entry)
        self._save_json()
        
    def log_final_results(self, best_acc, best_epoch, total_epochs, early_stopped):
        """记录最终结果"""
        self.logs['final_results'] = {
            'best_val_acc': float(best_acc),
            'best_epoch': best_epoch,
            'total_epochs_trained': total_epochs,
            'early_stopped': early_stopped,
            'final_train_acc': self.logs['training_history'][-1]['train_acc'] if self.logs['training_history'] else 0,
            'training_time_seconds': time.time() - self.start_time,
        }
        self._save_json()
        
    def _save_json(self):
        """保存日志到JSON"""
        with open(self.log_dir / 'training_log.json', 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def generate_plots(self):
        """生成训练曲线图"""
        history = self.logs['training_history']
        if not history:
            return
        
        epochs = [h['epoch'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_acc = [h['val_acc'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准确率
        axes[0].plot(epochs, train_acc, 'b-o', linewidth=2, label='Training Accuracy')
        axes[0].plot(epochs, val_acc, 'r-s', linewidth=2, label='Validation Accuracy')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Phase 2: Training & Validation Accuracy', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 损失
        axes[1].plot(epochs, train_loss, 'b-o', linewidth=2, label='Training Loss')
        axes[1].plot(epochs, val_loss, 'r-s', linewidth=2, label='Validation Loss')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Phase 2: Training & Validation Loss', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.log_dir / 'training_curves.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练曲线图已保存")


# ==================== 训练流程 ====================

def train_phase2():
    """Phase 2 训练 - 数据增强 + 延长训练"""
    
    logger = DetailedLogger(RUN_LOG_DIR)
    logger.log_config(CONFIG)
    
    print("\n📊 加载数据...")
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = BRISCDataset(CONFIG['data_dir'], transform=train_transform, mode='train')
    val_dataset = BRISCDataset(CONFIG['data_dir'], transform=val_transform, mode='val')
    
    if len(train_dataset) == 0:
        print("❌ 未找到训练数据！请检查 data/ 目录")
        return None
    
    logger.log_data_info(train_dataset, val_dataset)
    
    print(f"✓ 训练样本: {len(train_dataset)}")
    print(f"✓ 验证样本: {len(val_dataset)}")
    print("  类别分布:", train_dataset.get_class_distribution())
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # 模型
    print("\n🔄 初始化模型...")
    model = ResNet2D(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Phase 2: 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG['epochs'], 
        eta_min=CONFIG['min_lr']
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")
    print(f"  数据增强: Rotation(±15°) + HFlip + ColorJitter")
    print(f"  学习率调度: CosineAnnealing (min_lr={CONFIG['min_lr']})")
    
    # 训练
    print("\n🚀 开始 Phase 2 训练...")
    print("=" * 60)
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
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
        avg_train_loss = train_loss / len(train_loader)
        
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
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        
        # 更新学习率
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        # 日志
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            train_acc=train_acc,
            val_loss=avg_val_loss,
            val_acc=val_acc,
            lr=current_lr
        )
        
        # 打印
        lr_info = f"LR: {current_lr:.6f}"
        if new_lr != current_lr:
            lr_info += f" → {new_lr:.6f}"
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} ({epoch_time:.1f}s) [{lr_info}]")
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Early stopping - Phase 2: patience=5
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), RUN_LOG_DIR / 'best_model_phase2.pth')
            print(f"  ✓ 新最佳模型! Acc={best_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  ~ 无改善 ({patience_counter}/{CONFIG['early_stop_patience']})")
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
    
    # 最终结果
    early_stopped = patience_counter >= CONFIG['early_stop_patience']
    logger.log_final_results(best_acc, best_epoch, epoch + 1, early_stopped)
    
    # 保存最终模型
    torch.save(model.state_dict(), RUN_LOG_DIR / 'final_model_phase2.pth')
    
    # 生成图表
    logger.generate_plots()
    
    print("\n" + "=" * 60)
    print("🎉 Phase 2 完成!")
    print(f"   最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"   相比 Phase 1 (82.79%): {'+' if best_acc > 82.79 else ''}{best_acc - 82.79:.2f}%")
    print(f"   日志目录: {RUN_LOG_DIR}")
    
    return best_acc, RUN_LOG_DIR


if __name__ == '__main__':
    result = train_phase2()
    
    if result:
        best_acc, log_dir = result
        print(f"\n✅ Phase 2 训练完成!")
    else:
        print("\n❌ 训练失败")
