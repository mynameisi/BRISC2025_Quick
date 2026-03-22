"""
BRISC2025 Auto Research - 实际训练执行器
接收配置，执行训练，保存结果
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', required=True)
    parser.add_argument('--config', required=True, help='JSON config string')
    return parser.parse_args()

# ==================== 模型架构 ====================

class ResNet34Encoder(nn.Module):
    """使用预训练ResNet34作为Encoder"""
    def __init__(self, pretrained=True, freeze_layers=4):
        super().__init__()
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256
        self.layer4 = resnet.layer4  # 1/32, 512
        
        # 根据freeze_layers冻结
        if freeze_layers >= 1:
            for p in self.stem.parameters(): p.requires_grad = False
        if freeze_layers >= 2:
            for p in self.layer1.parameters(): p.requires_grad = False
        if freeze_layers >= 3:
            for p in self.layer2.parameters(): p.requires_grad = False
        if freeze_layers >= 4:
            for p in self.layer3.parameters(): p.requires_grad = False
            for p in self.layer4.parameters(): p.requires_grad = False
    
    def forward(self, x):
        features = []
        x = self.stem(x); features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x); features.append(x)
        x = self.layer2(x); features.append(x)
        x = self.layer3(x); features.append(x)
        x = self.layer4(x); features.append(x)
        return features


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c + skip_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            diffH = skip.size(2) - x.size(2)
            diffW = skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [diffW//2, diffW-diffW//2, diffH//2, diffH-diffH//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_layers=4):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained, freeze_layers)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )
    
    def forward(self, x):
        feats = self.encoder(x)
        d4 = self.decoder4(feats[4], feats[3])
        d3 = self.decoder3(d4, feats[2])
        d2 = self.decoder2(d3, feats[1])
        d1 = self.decoder1(d2, feats[0])
        out = self.final_up(d1)
        return self.final_conv(out)


# ==================== 数据集 ====================

class SegDataset(Dataset):
    def __init__(self, data_dir, mode='train', image_size=256, aug='medium'):
        self.data_dir = Path(data_dir).expanduser() / mode
        self.image_size = image_size
        
        # 根据aug强度设置变换
        if aug == 'strong' and mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.3, 0.3, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif aug == 'weak' and mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:  # medium or test
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(10) if mode == 'train' else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(0.5) if mode == 'train' else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.2, 0.2) if mode == 'train' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.images_dir = self.data_dir / 'images'
        self.masks_dir = self.data_dir / 'masks'
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / (img_path.stem + '.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        image = self.transform(image)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32))
        
        return image, mask.long()


# ==================== 指标 ====================

def dice_score(pred, target, smooth=1e-6):
    pred_probs = torch.softmax(pred, dim=1)[:, 1]
    pred_bin = (pred_probs > 0.5).float()
    target_f = target.float()
    intersection = (pred_bin * target_f).sum()
    return (2. * intersection + smooth) / (pred_bin.sum() + target_f.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred_probs = torch.softmax(pred, dim=1)[:, 1]
    pred_bin = (pred_probs > 0.5).float()
    target_f = target.float()
    intersection = (pred_bin * target_f).sum()
    union = pred_bin.sum() + target_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ==================== 主函数 ====================

def main():
    args = parse_args()
    config = json.loads(args.config)
    
    exp_dir = Path('experiments/auto_research') / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(42)
    
    # 数据
    train_ds = SegDataset(config['data_dir'], 'train', config['image_size'], config['augmentation'])
    val_ds = SegDataset(config['data_dir'], 'test', config['image_size'], 'weak')
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    
    # 模型
    model = UNetResNet34(
        num_classes=2,
        pretrained=True,
        freeze_layers=config['freeze_layers']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
    
    # 训练
    best_dice = 0.0
    best_iou = 0.0
    patience = 0
    
    print(f"🚀 开始实验 {args.exp_id}")
    print(f"配置: freeze={config['freeze_layers']}, lr={config['learning_rate']}, aug={config['augmentation']}")
    
    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Val
        model.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_dice += dice_score(out, masks).item()
                val_iou += iou_score(out, masks).item()
        
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Dice={val_dice:.4f}, IoU={val_iou:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            patience = 0
            torch.save(model.state_dict(), exp_dir / 'best_model.pth')
        else:
            patience += 1
            if patience >= 7:
                print(f"Early stop at epoch {epoch+1}")
                break
    
    # 保存结果
    result = {
        'dice': best_dice,
        'iou': best_iou,
        'epochs': epoch + 1,
        'config': config,
        'completed_at': datetime.now().isoformat()
    }
    
    with open(exp_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ 实验完成: Dice={best_dice:.4f}, IoU={best_iou:.4f}")


if __name__ == '__main__':
    main()
