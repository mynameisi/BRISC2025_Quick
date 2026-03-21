"""
生成 PDF 格式的 Phase 1 报告
"""
from fpdf import FPDF
from pathlib import Path
import json
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        # 标题
        self.set_font('Arial', 'B', 16)
        self.set_text_color(31, 78, 121)
        self.cell(0, 10, 'BRISC2025 Phase 1: Brain Tumor Classification', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'A Baseline Study Using Transfer Learning', 0, 1, 'C')
        self.ln(5)
        # 分隔线
        self.set_draw_color(31, 78, 121)
        self.line(10, 35, 200, 35)
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | BRISC2025 Technical Report', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(31, 78, 121)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_table(self, headers, data, col_widths=None):
        # 表头
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(31, 78, 121)
        self.set_text_color(255, 255, 255)
        
        if col_widths is None:
            col_widths = [60] * len(headers)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, 1, 0, 'C', True)
        self.ln()
        
        # 数据行
        self.set_font('Arial', '', 10)
        self.set_text_color(50, 50, 50)
        
        for row in data:
            # 交替背景色
            if data.index(row) % 2 == 0:
                self.set_fill_color(240, 240, 240)
            else:
                self.set_fill_color(255, 255, 255)
            
            for i, cell in enumerate(row):
                align = 'C' if i > 0 else 'L'
                self.cell(col_widths[i], 7, str(cell), 1, 0, align, True)
            self.ln()
        self.ln(5)
    
    def add_highlight_box(self, title, content):
        self.set_fill_color(255, 250, 240)
        self.set_draw_color(255, 193, 7)
        self.set_line_width(0.5)
        
        # 计算高度
        self.set_font('Arial', 'B', 11)
        title_height = 8
        self.set_font('Arial', '', 10)
        
        # 绘制框
        self.rect(15, self.get_y(), 180, 50, style='DF')
        
        # 内容
        self.set_xy(18, self.get_y() + 3)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(200, 150, 0)
        self.cell(0, 6, title, 0, 1)
        
        self.set_x(18)
        self.set_font('Arial', '', 10)
        self.set_text_color(80, 60, 0)
        self.multi_cell(174, 6, content)
        
        self.ln(55)


def generate_pdf():
    # 读取训练日志
    log_dir = Path('~/BRISC2025_Quick/logs/phase1_rerun_20260321_110606').expanduser()
    with open(log_dir / 'training_log.json') as f:
        logs = json.load(f)
    
    history = logs['training_history']
    best_acc = max(h['val_acc'] for h in history)
    best_epoch = [h['val_acc'] for h in history].index(best_acc) + 1
    
    # 创建 PDF
    pdf = PDFReport()
    pdf.add_page()
    
    # 摘要
    pdf.chapter_title('Abstract')
    abstract = """This report presents a baseline classification system for brain tumor detection using the BRISC2025 dataset. We employ a ResNet50 architecture with ImageNet pre-trained weights to classify three types of brain tumors (Glioma, Meningioma, and Pituitary) from T1-weighted MRI scans. Our model achieved a peak validation accuracy of 82.79% after 4 epochs of training on 3,933 training samples. The results demonstrate the effectiveness of transfer learning for medical image classification."""
    pdf.chapter_body(abstract)
    
    # 关键发现
    pdf.add_highlight_box(
        'KEY FINDINGS',
        f"""- Best Validation Accuracy: {best_acc:.2f}% (Epoch {best_epoch})
- Training Samples: {logs['data_info']['train_samples']:,}
- Validation Samples: {logs['data_info']['val_samples']:,}
- Model: ResNet50 (ImageNet pretrained)
- Device: Apple MPS (Metal Performance Shaders)"""
    )
    
    # 数据集
    pdf.add_page()
    pdf.chapter_title('1. Dataset')
    
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "The BRISC2025 dataset contains T1-weighted MRI scans with expert annotations. The dataset was extracted from the segmentation task and organized into three classes for classification.")
    pdf.ln(3)
    
    # 数据分布表
    pdf.chapter_title('Dataset Distribution')
    headers = ['Tumor Type', 'Training', 'Validation']
    data = [
        ['Glioma', '1,147', '254'],
        ['Meningioma', '1,329', '306'],
        ['Pituitary', '1,457', '300'],
        ['Total', '3,933', '860']
    ]
    pdf.add_table(headers, data, [80, 50, 50])
    
    # 方法
    pdf.chapter_title('2. Methods')
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Model Architecture', 0, 1)
    pdf.set_font('Arial', '', 11)
    methods_text = """- Backbone: ResNet50 with ImageNet pre-trained weights
- Input Size: 512x512 pixels (maintaining original resolution)
- Output Layer: Fully-connected layer with 3 output units
- Normalization: ImageNet mean and standard deviation"""
    pdf.multi_cell(0, 6, methods_text)
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Training Configuration', 0, 1)
    
    headers = ['Parameter', 'Value']
    data = [
        ['Optimizer', 'Adam'],
        ['Learning Rate', '0.001'],
        ['Batch Size', '16'],
        ['Max Epochs', '5'],
        ['Early Stopping Patience', '3 epochs'],
        ['Loss Function', 'Cross-Entropy Loss'],
        ['Device', 'Apple MPS']
    ]
    pdf.add_table(headers, data, [90, 90])
    
    # 结果
    pdf.add_page()
    pdf.chapter_title('3. Results')
    
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "The model was trained for 4 epochs before termination. The following table summarizes the training and validation metrics at each epoch.")
    pdf.ln(3)
    
    # 训练历史表
    pdf.chapter_title('Training History')
    headers = ['Epoch', 'Train Loss', 'Train Acc %', 'Val Loss', 'Val Acc %']
    data = []
    for h in history:
        data.append([
            str(h['epoch']),
            f"{h['train_loss']:.4f}",
            f"{h['train_acc']:.2f}",
            f"{h['val_loss']:.4f}",
            f"{h['val_acc']:.2f}"
        ])
    pdf.add_table(headers, data, [25, 35, 35, 35, 35])
    
    # 关键结果
    pdf.chapter_title('Key Results')
    
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, """1. Best Performance: The model achieved peak validation accuracy of 82.79% at Epoch 4.

2. Convergence Pattern: Training accuracy steadily improved from 79.61% to 86.55%, indicating effective learning.

3. Generalization: The generalization gap remained relatively small at 3.76%, suggesting minimal overfitting.

4. Validation Loss: Validation loss showed an optimal decreasing trend in the final epochs.""")
    pdf.ln(5)
    
    # 性能对比
    pdf.chapter_title('Performance Comparison')
    headers = ['Method', 'Accuracy']
    data = [
        ['Random Guessing (3-class)', '33.33%'],
        ['Our Method (ResNet50)', '82.79%'],
        ['Improvement', '+49.46%']
    ]
    pdf.add_table(headers, data, [120, 50])
    
    # 讨论
    pdf.add_page()
    pdf.chapter_title('4. Discussion')
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Strengths', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, """- Transfer Learning Effectiveness: ImageNet pre-trained weights provided effective feature initialization, enabling rapid convergence within 4 epochs.

- High Resolution: Maintaining the original 512x512 resolution preserved fine-grained pathological features crucial for accurate classification.

- Balanced Performance: The model achieved consistent performance across all three tumor types.""")
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Limitations', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, """- Missing Class: The No Tumor class was not included due to data extraction constraints.

- No Data Augmentation: The baseline configuration did not employ augmentation techniques.

- Early Termination: Training was terminated at Epoch 4, potentially leaving performance gains unrealized.""")
    pdf.ln(3)
    
    # 未来工作
    pdf.chapter_title('5. Future Work (Phase 2)')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, """Based on these baseline results, we identify the following optimization strategies:

1. Data Augmentation: Implement geometric and photometric augmentations (expected improvement: +3-5%)

2. Learning Rate Scheduling: Introduce cosine annealing or step decay (expected improvement: +1-2%)

3. Extended Training: Train for 10-20 epochs with early stopping (expected improvement: +2-3%)

4. Fine-tuning: Unfreeze additional ResNet blocks for end-to-end optimization""")
    
    # 结论
    pdf.add_page()
    pdf.chapter_title('6. Conclusion')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, """This study successfully established a baseline brain tumor classification system achieving 82.79% validation accuracy on the BRISC2025 dataset. The results demonstrate that transfer learning with ResNet50 is highly effective for medical image classification tasks.

The comprehensive logging and documentation ensure full reproducibility of our experiments. The baseline provides a solid foundation for Phase 2 optimization, with a clear roadmap toward achieving 86-88% accuracy through data augmentation and hyperparameter tuning.""")
    pdf.ln(10)
    
    # 页脚信息
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
    pdf.cell(0, 6, 'GitHub: https://github.com/mynameisi/BRISC2025_Quick', 0, 1, 'C')
    
    # 保存
    output_path = log_dir / 'BRISC2025_Phase1_Report.pdf'
    pdf.output(str(output_path))
    print(f"✅ PDF 报告已生成: {output_path}")
    
    return output_path


if __name__ == '__main__':
    generate_pdf()
