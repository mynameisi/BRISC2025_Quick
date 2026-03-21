"""
BRISC2025 Phase 1 - 详细日志版本
重新运行并保存完整的训练过程数据
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

# 配置
CONFIG = {
    'data_dir': './data',
    'batch_size': 16,
    'epochs': 5,
    'lr': 0.001,
    'image_size': 512,
    'num_classes': 4,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'early_stop_patience': 3,
    'seed': 42,
}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# 创建详细日志目录
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_LOG_DIR = LOG_DIR / f'phase1_rerun_{RUN_ID}'
RUN_LOG_DIR.mkdir(exist_ok=True)

print(f"🚀 BRISC2025 Phase 1 - 详细日志版本")
print(f"运行ID: {RUN_ID}")
print(f"设备: {CONFIG['device']}")
print(f"日志目录: {RUN_LOG_DIR}")
print("=" * 60)


# ==================== 数据集 ====================

class BRISCDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.class_map = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}
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
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ==================== 模型 ====================

class ResNet2D(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
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
        
        # 初始化日志结构
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
        self.logs['config']['device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'MPS/CPU'
        
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
        
        # 实时保存
        self._save_json()
        
    def log_batch_metrics(self, epoch, batch_idx, loss, acc):
        """记录batch级指标（可选，数据量大时慎用）"""
        pass  # 暂不实现，避免日志过大
        
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
    
    def save_summary_txt(self):
        """保存文本摘要"""
        summary = []
        summary.append("=" * 60)
        summary.append("BRISC2025 Phase 1 - 训练摘要")
        summary.append("=" * 60)
        summary.append(f"运行ID: {RUN_ID}")
        summary.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # 配置
        summary.append("【配置】")
        for k, v in self.logs['config'].items():
            summary.append(f"  {k}: {v}")
        summary.append("")
        
        # 数据
        summary.append("【数据】")
        di = self.logs['data_info']
        summary.append(f"  训练样本: {di.get('train_samples', 'N/A')}")
        summary.append(f"  验证样本: {di.get('val_samples', 'N/A')}")
        if 'train_class_distribution' in di:
            summary.append("  训练集类别分布:")
            for cls, count in di['train_class_distribution'].items():
                summary.append(f"    {cls}: {count}")
        summary.append("")
        
        # 训练过程
        summary.append("【训练过程】")
        for entry in self.logs['training_history']:
            summary.append(f"Epoch {entry['epoch']}:")
            summary.append(f"  Train Loss: {entry['train_loss']:.4f}, Acc: {entry['train_acc']:.2f}%")
            summary.append(f"  Val   Loss: {entry['val_loss']:.4f}, Acc: {entry['val_acc']:.2f}%")
            summary.append(f"  LR: {entry['learning_rate']:.6f}, Time: {entry['elapsed_time']:.1f}s")
        summary.append("")
        
        # 最终结果
        summary.append("【最终结果】")
        fr = self.logs['final_results']
        summary.append(f"  最佳验证准确率: {fr.get('best_val_acc', 0) * 100:.2f}%")
        summary.append(f"  最佳Epoch: {fr.get('best_epoch', 'N/A')}")
        summary.append(f"  总训练时间: {fr.get('training_time_seconds', 0) / 60:.1f} 分钟")
        summary.append(f"  Early Stopped: {fr.get('early_stopped', False)}")
        summary.append("=" * 60)
        
        with open(self.log_dir / 'summary.txt', 'w') as f:
            f.write('\n'.join(summary))
        
        return '\n'.join(summary)


# ==================== 训练流程 ====================

def train_with_detailed_logging():
    """带详细日志的训练"""
    
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
    print("  类别分布:")
    for cls, count in train_dataset.get_class_distribution().items():
        print(f"    {cls}: {count}")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=0)  # Mac上num_workers=0更稳定
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # 模型
    print("\n🔄 初始化模型...")
    model = ResNet2D(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 训练
    print("\n🚀 开始训练...")
    print("=" * 60)
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        
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
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        
        # 日志
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            train_acc=train_acc,
            val_loss=avg_val_loss,
            val_acc=val_acc,
            lr=optimizer.param_groups[0]['lr']
        )
        
        # 打印
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), RUN_LOG_DIR / 'best_model.pth')
            print(f"  ✓ 新最佳模型! Acc={best_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  ~ 无改善 ({patience_counter}/{CONFIG['early_stop_patience']})")
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                break
    
    # 最终结果
    early_stopped = patience_counter >= CONFIG['early_stop_patience']
    logger.log_final_results(best_acc, best_epoch, epoch + 1, early_stopped)
    
    # 保存最终模型
    torch.save(model.state_dict(), RUN_LOG_DIR / 'final_model.pth')
    
    # 生成并打印摘要
    summary = logger.save_summary_txt()
    
    print("\n" + "=" * 60)
    print(summary)
    print(f"\n📁 详细日志已保存到: {RUN_LOG_DIR}")
    
    return best_acc, RUN_LOG_DIR


if __name__ == '__main__':
    result = train_with_detailed_logging()
    
    if result:
        best_acc, log_dir = result
        print(f"\n✅ Phase 1 完成! 最佳准确率: {best_acc:.2f}%")
        print(f"📊 查看详细日志: {log_dir}/training_log.json")
        print(f"📝 查看文本摘要: {log_dir}/summary.txt")
    else:
        print("\n❌ 训练失败")
