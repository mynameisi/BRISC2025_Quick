"""
实验跟踪与监控模块
BRISC2025 Phase 1 - 保守执行方案
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    """实验跟踪器 - 记录所有实验配置和结果"""
    
    def __init__(self, log_dir="experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "experiment_log.jsonl"
        
        # 资源限制
        self.max_experiments = 5  # Phase 1 最多 5 次
        self.max_epochs = 5       # 每次最多 5 epoch（快速验证）
        self.baseline_acc = 0.25  # 基线：随机猜测
        
    def log_experiment(self, phase, exp_id, config, results):
        """记录一次实验"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "experiment_id": exp_id,
            "config": config,
            "results": results,
            "status": self._determine_status(results)
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        return entry["status"]
    
    def _determine_status(self, results):
        """判断实验状态"""
        val_acc = results.get("val_acc", 0)
        
        if val_acc > self.baseline_acc + 0.05:
            return "success"  # 显著优于基线
        elif val_acc > self.baseline_acc:
            return "marginal"  # 略优于基线
        else:
            return "failed"   # 无效
    
    def get_experiment_count(self, phase=None):
        """获取已运行实验数量"""
        if not self.log_file.exists():
            return 0
        
        count = 0
        with open(self.log_file) as f:
            for line in f:
                entry = json.loads(line)
                if phase is None or entry["phase"] == phase:
                    count += 1
        return count
    
    def get_best_experiment(self, phase=None):
        """获取最佳实验"""
        if not self.log_file.exists():
            return None
        
        best = None
        best_acc = 0
        
        with open(self.log_file) as f:
            for line in f:
                entry = json.loads(line)
                if phase and entry["phase"] != phase:
                    continue
                acc = entry["results"].get("val_acc", 0)
                if acc > best_acc:
                    best_acc = acc
                    best = entry
        
        return best
    
    def print_summary(self):
        """打印实验摘要"""
        if not self.log_file.exists():
            print("暂无实验记录")
            return
        
        print("\n" + "="*60)
        print("实验跟踪摘要")
        print("="*60)
        
        experiments = []
        with open(self.log_file) as f:
            for line in f:
                experiments.append(json.loads(line))
        
        print(f"\n总实验数: {len(experiments)}")
        
        # 按 phase 分组
        from collections import defaultdict
        by_phase = defaultdict(list)
        for exp in experiments:
            by_phase[exp["phase"]].append(exp)
        
        for phase, exps in sorted(by_phase.items()):
            print(f"\n【Phase {phase}】")
            for exp in exps:
                status_icon = {"success": "✓", "marginal": "~", "failed": "✗"}
                icon = status_icon.get(exp["status"], "?")
                print(f"  {icon} Exp {exp['experiment_id']}: "
                      f"{exp['config'][:40]}... "
                      f"Acc={exp['results'].get('val_acc', 0):.2%}")
        
        best = self.get_best_experiment()
        if best:
            print(f"\n最佳实验: Phase {best['phase']} - Exp {best['experiment_id']}")
            print(f"  验证准确率: {best['results'].get('val_acc', 0):.2%}")
            print(f"  配置: {best['config']}")
        
        print("="*60)


class ResourceMonitor:
    """资源监控 - 防止过度消耗"""
    
    def __init__(self):
        self.start_time = time.time()
        self.experiment_count = 0
        
    def check_budget(self, tracker):
        """检查资源预算"""
        # 检查实验数量
        if tracker.get_experiment_count() >= tracker.max_experiments:
            return False, "实验数量已达上限"
        
        # 检查运行时间（可选）
        elapsed = time.time() - self.start_time
        if elapsed > 3600:  # 1小时
            return False, "运行时间过长，建议检查"
        
        return True, "OK"
    
    def print_resource_usage(self):
        """打印资源使用情况"""
        elapsed = time.time() - self.start_time
        print(f"\n资源使用:")
        print(f"  运行时间: {elapsed/60:.1f} 分钟")
        print(f"  实验次数: {self.experiment_count}")
        
        # 检查磁盘
        import shutil
        disk = shutil.disk_usage(".")
        print(f"  磁盘剩余: {disk.free/1024/1024/1024:.1f} GB")


# 全局实例
tracker = ExperimentTracker()
monitor = ResourceMonitor()

if __name__ == "__main__":
    # 测试
    tracker.print_summary()
    monitor.print_resource_usage()
