"""
BRISC2025 Phase 3 Enhanced - 继续训练（解冻encoder）
从上一次保存的模型继续
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import json

# 添加原脚本路径
sys.path.insert(0, str(Path(__file__).parent))

from run_phase3_enhanced import (
    UNetResNet34, BRISCSegmentationDataset, 
    dice_score, iou_score, get_transforms
)

# 配置
LOG_DIR = Path('logs/phase3_enhanced_20260322_070158')
CONFIG = {
    'data_dir': '~/.cache/kagglehub/datasets/briscdataset/brisc2025/versions/6/brisc2025/segmentation_task',
    'batch_size': 16,
    'epochs': 50,
    'lr': 0.00005,  # 解冻后用更小的学习率
    'min_lr': 1e-7,
    'image_size': 256,
    'num_classes': 2,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'early_stop_patience': 10,
}

print("🧠 继续 Phase 3 Enhanced 训练（解冻encoder）")
print(f"加载模型: {LOG_DIR / 'best_unet_resnet34.pth'}")
print("=" * 60)

# 加载数据
train_transform = get_transforms('train')
val_transform = get_transforms('test')

train_dataset = BRISCSegmentationDataset(CONFIG['data_dir'], transform=train_transform, mode='train')
val_dataset = BRISCSegmentationDataset(CONFIG['data_dir'], transform=val_transform, mode='test')

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

# 加载模型并解冻
model = UNetResNet34(num_classes=2, pretrained=False).to(CONFIG['device'])
model.load_state_dict(torch.load(LOG_DIR / 'best_unet_resnet34.pth', weights_only=True))
model.unfreeze_encoder()

# 优化器（全部参数）
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
criterion = torch.nn.CrossEntropyLoss()

# 加载之前的日志
with open(LOG_DIR / 'training_log.json') as f:
    logs = json.load(f)

best_dice = 0.663  # 当前最佳
best_iou = 0.515
best_epoch = 3
patience_counter = 0

print(f"\n继续训练，当前最佳 Dice: {best_dice:.4f}")
print(f"开始微调全部参数...")
print("=" * 60)

for epoch in range(3, CONFIG['epochs']):
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
    
    scheduler.step()
    
    # 打印
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} [FULL] LR={current_lr:.6f}")
    print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}, IoU={train_iou:.4f}")
    print(f"  Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
    
    # 记录
    entry = {
        'epoch': epoch + 1,
        'phase': 'full',
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_dice': train_dice,
        'val_dice': val_dice,
        'train_iou': train_iou,
        'val_iou': val_iou,
        'learning_rate': current_lr,
    }
    logs['training_history'].append(entry)
    
    # Early stopping
    if val_dice > best_dice:
        best_dice = val_dice
        best_iou = val_iou
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), LOG_DIR / 'best_unet_resnet34.pth')
        print(f"  ✓ 新最佳模型! Dice={best_dice:.4f}, IoU={best_iou:.4f}")
    else:
        patience_counter += 1
        print(f"  ~ 无改善 ({patience_counter}/{CONFIG['early_stop_patience']})")
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break

# 保存最终结果
logs['final_results'] = {
    'best_val_dice': best_dice,
    'best_val_iou': best_iou,
    'best_epoch': best_epoch,
    'total_epochs_trained': epoch + 1,
}

with open(LOG_DIR / 'training_log.json', 'w') as f:
    json.dump(logs, f, indent=2)

print("\n" + "=" * 60)
print("🎉 Phase 3 Enhanced 完成!")
print(f"   最佳验证 Dice: {best_dice:.4f}")
print(f"   最佳验证 IoU: {best_iou:.4f}")
print(f"   最佳 Epoch: {best_epoch}")
print(f"   相比 baseline: +{(best_dice - 0.6836):.4f} ({(best_dice - 0.6836)*100:+.2f}%)")
