"""
从 segmentation_task 提取分类数据
根据文件名中的肿瘤类型标签组织数据
"""
import os
import shutil
from pathlib import Path

def extract_classification_data():
    """从 segmentation 数据提取分类数据"""
    
    # 路径
    base_path = Path.home() / '.cache/kagglehub/datasets/briscdataset/brisc2025/versions/6/brisc2025'
    seg_path = base_path / 'segmentation_task'
    output_path = Path('~/BRISC2025_Quick/data').expanduser()
    
    # 类别映射（从文件名解析）
    class_map = {
        'gl': 'glioma',
        'me': 'meningioma', 
        'pi': 'pituitary',
        'nt': 'notumor'
    }
    
    print("🔄 开始提取分类数据...")
    print(f"源路径: {seg_path}")
    print(f"目标路径: {output_path}")
    
    # 创建输出目录
    for split in ['train', 'val']:
        for cls in class_map.values():
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # 处理训练集和测试集
    stats = {'train': {}, 'val': {}}
    
    for split in ['train', 'test']:
        images_dir = seg_path / split / 'images'
        if not images_dir.exists():
            print(f"⚠️ 目录不存在: {images_dir}")
            continue
            
        print(f"\n处理 {split} 集...")
        
        for img_file in images_dir.glob('*.jpg'):
            # 解析文件名: brisc2025_train_00001_gl_ax_t1.jpg
            parts = img_file.stem.split('_')
            if len(parts) >= 5:
                tumor_code = parts[3]  # gl, me, pi, nt
                if tumor_code in class_map:
                    cls_name = class_map[tumor_code]
                    
                    # 测试集作为验证集
                    target_split = 'val' if split == 'test' else 'train'
                    target_dir = output_path / target_split / cls_name
                    
                    # 复制文件
                    shutil.copy2(img_file, target_dir / img_file.name)
                    
                    # 统计
                    stats[target_split][cls_name] = stats[target_split].get(cls_name, 0) + 1
    
    # 打印统计
    print("\n" + "="*60)
    print("✅ 数据提取完成!")
    print("="*60)
    
    total_train = 0
    total_val = 0
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for cls, count in sorted(stats[split].items()):
            print(f"  {cls}: {count} images")
            if split == 'train':
                total_train += count
            else:
                total_val += count
    
    print(f"\n总计: {total_train} 训练 / {total_val} 验证")
    
    return total_train, total_val

if __name__ == '__main__':
    extract_classification_data()
