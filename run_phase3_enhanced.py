"""
BRISC2025 Phase 3 Enhanced - 语义分割优化版
使用预训练 ResNet34 作为 U-Net Encoder
预期 Dice 提升 5-10%
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
import matplotlib.pyplot as plt

# 配置 - Phase 3 Enhanced (预训练Encoder)
CONFIG = {
    'data_dir': '~/.cache/kagglehub/datasets/briscdataset/brisc2025/versions/6/brisc2025/segmentation_task',
    'batch_size': 16,
    'epochs': 50,
    'lr': 0.0001,              # 降低学习率，预训练模型用较小LR
    'min_lr': 1e-7,
    'image_size': 256,
    'num_classes': 2,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'early_stop_patience': 10,
    'seed': 42,
    'encoder': 'resnet34',     # 预训练encoder
    'freeze_encoder_epochs': 5, # 前5轮冻结encoder，只训练decoder
}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# 创建日志目录
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_LOG_DIR = LOG_DIR / f'phase3_enhanced_{RUN_ID}'
RUN_LOG_DIR.mkdir(exist_ok=True)

print(f"🧠 BRISC2025 Phase 3 Enhanced - 语义分割优化版")
print(f"运行ID: {RUN_ID}")
print(f"设备: {CONFIG['device']}")
print(f"架构: U-Net with {CONFIG['encoder']} (pretrained)")
print(f"Epochs: {CONFIG['epochs']}")
print(f"策略: 前{CONFIG['freeze_encoder_epochs']}轮冻结encoder")
print("=" * 60)


# ==================== 预训练 U-Net 架构 ====================

class ResNet34Encoder(nn.Module):
    """使用预训练ResNet34作为Encoder"""
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练ResNet34
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        
        # 提取各层作为encoder stages
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )  # 1/2 resolution, 64 channels
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 1/4, 64 channels
        self.layer2 = resnet.layer2  # 1/8, 128 channels
        self.layer3 = resnet.layer3  # 1/16, 256 channels
        self.layer4 = resnet.layer4  # 1/32, 512 channels
        
        self.channels = [64, 64, 128, 256, 512]  # stem + 4 layers
    
    def forward(self, x):
        features = []
        
        x = self.stem(x)
        features.append(x)  # 1/2, 64
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # 1/4, 64
        
        x = self.layer2(x)
        features.append(x)  # 1/8, 128
        
        x = self.layer3(x)
        features.append(x)  # 1/16, 256
        
        x = self.layer4(x)
        features.append(x)  # 1/32, 512
        
        return features


class DecoderBlock(nn.Module):
    """U-Net Decoder Block"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # 处理尺寸不匹配
        if x.shape != skip.shape:
            diffH = skip.size(2) - x.size(2)
            diffW = skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [diffW // 2, diffW - diffW // 2,
                                      diffH // 2, diffH - diffH // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNetResNet34(nn.Module):
    """U-Net with ResNet34 Encoder (Pretrained)"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Encoder (预训练)
        self.encoder = ResNet34Encoder(pretrained=pretrained)
        
        # Decoder
        # features: [64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)]
        self.decoder4 = DecoderBlock(512, 256, 256)   # 1/16
        self.decoder3 = DecoderBlock(256, 128, 128)   # 1/8
        self.decoder2 = DecoderBlock(128, 64, 64)     # 1/4
        self.decoder1 = DecoderBlock(64, 64, 64)      # 1/2
        
        # Final upsampling and classification
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # to 1/1
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features[0]=1/2,64, [1]=1/4,64, [2]=1/8,128, [3]=1/16,256, [4]=1/32,512
        
        # Decoder (带skip connections)
        d4 = self.decoder4(features[4], features[3])  # 1/16
        d3 = self.decoder3(d4, features[2])           # 1/8
        d2 = self.decoder2(d3, features[1])           # 1/4
        d1 = self.decoder1(d2, features[0])           # 1/2
        
        # Final
        out = self.final_up(d1)  # 1/1
        out = self.final_conv(out)
        
        return out
    
    def freeze_encoder(self):
        """冻结encoder参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("  Encoder 已冻结 (只训练decoder)")
    
    def unfreeze_encoder(self):
        """解冻encoder参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("  Encoder 已解冻 (全部微调)")


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
        
        # 确保尺寸一致
        image = image.resize((CONFIG['image_size'], CONFIG['image_size']), Image.BILINEAR)
        mask = mask.resize((CONFIG['image_size'], CONFIG['image_size']), Image.NEAREST)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # 掩码处理
        mask_array = np.array(mask)
        mask_binary = (mask_array > 0).astype(np.float32)
        mask = torch.from_numpy(mask_binary)
        
        return image, mask.long()


def get_transforms(mode='train'):
    """数据变换"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    """计算 Dice Score"""
    pred_probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (pred_probs > 0.5).float()
    target_float = target.float()
    
    intersection = (pred_binary * target_float).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_float.sum()
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """计算 IoU"""
    pred_probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (pred_probs > 0.5).float()
    target_float = target.float()
    
    intersection = (pred_binary * target_float).sum()
    union = pred_binary.sum() + target_float.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# ==================== 日志记录器 ====================

class SegmentationLogger:
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
    
    def log_epoch(self, epoch, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, lr, phase='full'):
        entry = {
            'epoch': epoch,
            'phase': phase,  # 'frozen' or 'full'
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

def train_phase3_enhanced():
    """Phase 3 Enhanced 训练"""
    
    logger = SegmentationLogger(RUN_LOG_DIR)
    logger.log_config(CONFIG)
    
    print("\n📊 加载分割数据...")
    train_transform = get_transforms('train')
    val_transform = get_transforms('test')
    
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
    print("\n🔄 初始化 U-Net with ResNet34 Encoder...")
    model = UNetResNet34(num_classes=2, pretrained=True).to(CONFIG['device'])
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = total_params - encoder_params
    
    print(f"  总参数: {total_params:,}")
    print(f"  Encoder参数: {encoder_params:,} (预训练)")
    print(f"  Decoder参数: {decoder_params:,} (从头训练)")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    
    # 训练
    print("\n🚀 开始 Phase 3 Enhanced 训练...")
    print("=" * 60)
    
    best_dice = 0.0
    best_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        
        # 阶段性解冻策略
        if epoch == 0:
            model.freeze_encoder()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
            phase = 'frozen'
        elif epoch == CONFIG['freeze_encoder_epochs']:
            model.unfreeze_encoder()
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'] * 0.5)  # 解冻后用更小LR
            phase = 'full'
        else:
            phase = 'frozen' if epoch < CONFIG['freeze_encoder_epochs'] else 'full'
        
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
        logger.log_epoch(epoch + 1, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou, current_lr, phase)
        
        # 打印
        phase_str = "[FROZEN]" if phase == 'frozen' else "[FULL]"
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} {phase_str} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}, IoU={train_iou:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
        
        # Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), RUN_LOG_DIR / 'best_unet_resnet34.pth')
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
    print("🎉 Phase 3 Enhanced 完成!")
    print(f"   最佳验证 Dice: {best_dice:.4f}")
    print(f"   最佳验证 IoU: {best_iou:.4f}")
    print(f"   最佳 Epoch: {best_epoch}")
    print(f"   相比 baseline: Dice {(best_dice - 0.6836):+.4f}")
    print(f"   日志目录: {RUN_LOG_DIR}")
    
    return best_dice, best_iou, RUN_LOG_DIR


if __name__ == '__main__':
    result = train_phase3_enhanced()
    
    if result:
        best_dice, best_iou, log_dir = result
        print(f"\n✅ Phase 3 Enhanced 训练完成!")
        print(f"   分割任务 Dice Score: {best_dice:.4f}")
        print(f"   分割任务 IoU Score: {best_iou:.4f}")
        print(f"\n与 baseline 对比:")
        print(f"   Baseline Dice: 0.6836")
        print(f"   Enhanced Dice: {best_dice:.4f}")
        print(f"   提升: {(best_dice - 0.6836) * 100:+.2f} 个百分点")
    else:
        print("\n❌ 训练失败")
