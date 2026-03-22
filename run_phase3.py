"""
BRISC2025 Phase 3 - 语义分割任务
使用 U-Net 架构进行脑肿瘤像素级分割
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# 配置 - Phase 3 (修复版)
CONFIG = {
    'data_dir': '~/.cache/kagglehub/datasets/briscdataset/brisc2025/versions/6/brisc2025/segmentation_task',
    'batch_size': 16,          # 增加batch size因为模型变小了
    'epochs': 50,              # 更多epochs因为训练变快了
    'lr': 0.001,               # 增加学习率
    'min_lr': 1e-6,
    'image_size': 256,         # 降低分辨率以加速
    'num_classes': 2,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'early_stop_patience': 10,
    'seed': 42,
}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# 创建日志目录
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_LOG_DIR = LOG_DIR / f'phase3_segmentation_{RUN_ID}'
RUN_LOG_DIR.mkdir(exist_ok=True)

print(f"🧠 BRISC2025 Phase 3 - 语义分割")
print(f"运行ID: {RUN_ID}")
print(f"设备: {CONFIG['device']}")
print(f"架构: U-Net")
print(f"Epochs: {CONFIG['epochs']}")
print("=" * 60)


# ==================== U-Net 架构 ====================

class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample -> Conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """轻量级 U-Net for 语义分割"""
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 减少通道数以加速训练
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


# ==================== 数据集 ====================

class BRISCSegmentationDataset(Dataset):
    """BRISC2025 分割数据集"""
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = Path(data_dir).expanduser() / mode
        self.transform = transform
        self.mode = mode
        
        self.images_dir = self.data_dir / 'images'
        self.masks_dir = self.data_dir / 'masks'
        
        self.image_files = sorted([f for f in self.images_dir.glob('*.jpg')])
        
        print(f"  {mode}: 找到 {len(self.image_files)} 张图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / (img_path.stem + '.png')
        
        # 加载图像和掩码
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 确保尺寸一致 (先 resize 到统一尺寸)
        image = image.resize((CONFIG['image_size'], CONFIG['image_size']), Image.BILINEAR)
        mask = mask.resize((CONFIG['image_size'], CONFIG['image_size']), Image.NEAREST)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # 掩码单独处理 - 先转numpy再处理
        mask_array = np.array(mask)
        # 二值化: 任何非零值都视为肿瘤 (因为掩码值可能是 1-255)
        mask_binary = (mask_array > 0).astype(np.float32)
        mask = torch.from_numpy(mask_binary)
        
        return image, mask.long()


def get_transforms(mode='train'):
    """数据变换 - 仅应用于图像"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# ==================== 评估指标 ====================

def dice_score(pred, target, smooth=1e-6):
    """计算 Dice Score (F1 Score)"""
    # pred: [B, C, H, W] logits
    # target: [B, H, W] long tensor
    
    # 获取预测类别
    pred_probs = torch.softmax(pred, dim=1)[:, 1]  # 肿瘤类别的概率 [B, H, W]
    pred_binary = (pred_probs > 0.5).float()
    
    # target 转为 float
    target_float = target.float()
    
    # 计算 intersection 和 union
    intersection = (pred_binary * target_float).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_float.sum()
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """计算 IoU (Intersection over Union)"""
    pred_probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (pred_probs > 0.5).float()
    target_float = target.float()
    
    intersection = (pred_binary * target_float).sum()
    union = pred_binary.sum() + target_float.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# ==================== 日志记录器 ====================

class SegmentationLogger:
    """分割任务日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logs = {
            'config': {},
            'training_history': [],
            'final_results': {},
        }
        
        self.start_time = time.time()
        
    def log_config(self, config):
        self.logs['config'] = config.copy()
        
    def log_epoch(self, epoch, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, lr):
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_dice': float(train_dice),
            'val_dice': float(val_dice),
            'train_iou': float(train_iou),
            'val_iou': float(val_iou),
            'learning_rate': float(lr),
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
        }
        self.logs['training_history'].append(entry)
        self._save_json()
        
    def log_final_results(self, best_dice, best_iou, best_epoch, total_epochs):
        self.logs['final_results'] = {
            'best_val_dice': float(best_dice),
            'best_val_iou': float(best_iou),
            'best_epoch': best_epoch,
            'total_epochs_trained': total_epochs,
            'training_time_minutes': (time.time() - self.start_time) / 60,
        }
        self._save_json()
        
    def _save_json(self):
        with open(self.log_dir / 'training_log.json', 'w') as f:
            json.dump(self.logs, f, indent=2)


# ==================== 训练流程 ====================

def train_phase3():
    """Phase 3 训练 - 语义分割"""
    
    logger = SegmentationLogger(RUN_LOG_DIR)
    logger.log_config(CONFIG)
    
    print("\n📊 加载分割数据...")
    train_transform = get_transforms('train')
    val_transform = get_transforms('test')  # 使用test作为验证集
    
    train_dataset = BRISCSegmentationDataset(CONFIG['data_dir'], transform=train_transform, mode='train')
    val_dataset = BRISCSegmentationDataset(CONFIG['data_dir'], transform=val_transform, mode='test')
    
    if len(train_dataset) == 0:
        print("❌ 未找到训练数据！")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # 模型
    print("\n🔄 初始化 U-Net...")
    model = UNet(n_channels=3, n_classes=2).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,}")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    
    # 训练
    print("\n🚀 开始 Phase 3 训练...")
    print("=" * 60)
    
    best_dice = 0.0
    best_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 训练
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        
        for images, masks in train_loader:
            images, masks = images.to(CONFIG['device']), masks.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_score(outputs, masks)
            train_iou += iou_score(outputs, masks)
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(CONFIG['device']), masks.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)
                val_iou += iou_score(outputs, masks)
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        # 更新学习率
        scheduler.step()
        
        # 日志
        logger.log_epoch(epoch + 1, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, current_lr)
        
        # 打印
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}, IoU={train_iou:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
        
        # Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), RUN_LOG_DIR / 'best_unet.pth')
            print(f"  ✓ 新最佳模型! Dice={best_dice:.4f}, IoU={best_iou:.4f}")
        else:
            patience_counter += 1
            print(f"  ~ 无改善 ({patience_counter}/{CONFIG['early_stop_patience']})")
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
    
    # 最终结果
    logger.log_final_results(best_dice, best_iou, best_epoch, epoch + 1)
    
    print("\n" + "=" * 60)
    print("🎉 Phase 3 完成!")
    print(f"   最佳验证 Dice: {best_dice:.4f}")
    print(f"   最佳验证 IoU: {best_iou:.4f}")
    print(f"   最佳 Epoch: {best_epoch}")
    print(f"   日志目录: {RUN_LOG_DIR}")
    
    return best_dice, best_iou, RUN_LOG_DIR


if __name__ == '__main__':
    result = train_phase3()
    
    if result:
        best_dice, best_iou, log_dir = result
        print(f"\n✅ Phase 3 训练完成!")
        print(f"   分割任务 Dice Score: {best_dice:.4f}")
        print(f"   分割任务 IoU Score: {best_iou:.4f}")
    else:
        print("\n❌ 训练失败")
