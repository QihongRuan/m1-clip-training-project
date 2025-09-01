#!/usr/bin/env python3
"""
M1-Optimized CLIP Training Monitor
Real-time monitoring for multimodal training progress
"""

import time
import psutil
import subprocess
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class CLIPTrainingMonitor:
    """Real-time CLIP training monitor for M1 MacBook Pro"""
    
    def __init__(self, process_name: str = "m1_clip_training.py", 
                 checkpoint_dir: str = "./clip_checkpoints",
                 update_interval: int = 10):
        self.process_name = process_name
        self.checkpoint_dir = checkpoint_dir
        self.update_interval = update_interval
        self.start_time = time.time()
        self.iteration = 0
        
        # Monitoring data
        self.cpu_history = []
        self.memory_history = []
        self.timestamps = []
        self.loss_history = []
        self.accuracy_history = []
        
        print("🎨 CLIP Training Monitor Initialized")
        print(f"📊 Monitoring: {process_name}")
        print(f"💾 Checkpoint dir: {checkpoint_dir}")
        print(f"🔄 Update interval: {update_interval}s")
    
    def get_training_process(self):
        """Find CLIP training process"""
        try:
            result = subprocess.run(['pgrep', '-f', self.process_name], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                pid = int(result.stdout.strip().split('\n')[0])
                process = psutil.Process(pid)
                
                # Get detailed process stats
                cpu_percent = process.cpu_percent(interval=1)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = process.memory_percent()
                
                return {
                    'pid': pid,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_percent,
                    'status': process.status(),
                    'running': True
                }
        except Exception as e:
            pass
        
        return {
            'pid': None,
            'cpu_percent': 0,
            'memory_mb': 0,
            'memory_percent': 0,
            'status': 'not_found',
            'running': False
        }
    
    def get_system_stats(self):
        """Get overall system performance"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # M1-specific CPU details
        cpu_count_physical = psutil.cpu_count(logical=False)  # Physical cores
        cpu_count_logical = psutil.cpu_count(logical=True)    # Total threads
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'cpu_physical': cpu_count_physical,
            'cpu_logical': cpu_count_logical
        }
    
    def load_training_metrics(self):
        """Load current training metrics from checkpoint"""
        try:
            metrics_file = os.path.join(self.checkpoint_dir, 'training_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            'final_train_loss': 0,
            'final_val_loss': 0,
            'final_train_acc': 0,
            'final_val_acc': 0,
            'best_val_acc': 0,
            'total_epochs': 0,
            'model_parameters': 0
        }
    
    def estimate_progress(self, runtime_minutes: float, process_info: dict):
        """Estimate training progress"""
        if not process_info['running']:
            return 0, 0
        
        # CLIP training typically takes longer than CNN
        # Estimate: 15 epochs * 2-3 minutes per epoch = 30-45 minutes
        estimated_total_minutes = 40
        progress_pct = min((runtime_minutes / estimated_total_minutes) * 100, 100)
        
        eta_minutes = max(0, estimated_total_minutes - runtime_minutes)
        
        return progress_pct, eta_minutes
    
    def create_live_plot(self, save_path: str = None):
        """Create live performance visualization"""
        if len(self.timestamps) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert timestamps to minutes
        time_minutes = [(t - self.start_time) / 60 for t in self.timestamps]
        
        # CPU Usage
        ax1.plot(time_minutes, self.cpu_history, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(time_minutes, self.cpu_history, alpha=0.3, color='blue')
        ax1.set_title('M1 CPU Utilization During CLIP Training', fontweight='bold')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 150)
        
        # Memory Usage
        ax2.plot(time_minutes, self.memory_history, 'r-', linewidth=2, alpha=0.8)
        ax2.fill_between(time_minutes, self.memory_history, alpha=0.3, color='red')
        ax2.axhline(y=8000, color='orange', linestyle='--', alpha=0.7, label='8GB Total RAM')
        ax2.set_title('Memory Usage During Training', fontweight='bold')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory (MB)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Training Progress (placeholder - would show actual loss if available)
        if len(self.loss_history) > 1:
            epochs = range(1, len(self.loss_history) + 1)
            ax3.plot(epochs, self.loss_history, 'g-', linewidth=2, marker='o')
            ax3.set_title('CLIP Contrastive Loss', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Training Loss\n(Waiting for data...)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('CLIP Contrastive Loss', fontweight='bold')
        
        # Accuracy Progress
        if len(self.accuracy_history) > 1:
            epochs = range(1, len(self.accuracy_history) + 1)
            ax4.plot(epochs, self.accuracy_history, 'purple', linewidth=2, marker='s')
            ax4.set_title('Multimodal Accuracy', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
        else:
            ax4.text(0.5, 0.5, 'Validation Accuracy\n(Waiting for data...)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Multimodal Accuracy', fontweight='bold')
        
        plt.suptitle('M1-Optimized CLIP Training - Live Monitor\n'
                    f'Image-Text Contrastive Learning', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def display_status(self, process_info: dict, system_info: dict, 
                      training_metrics: dict, progress_pct: float, eta_minutes: float):
        """Display comprehensive training status"""
        
        # Clear screen
        os.system('clear')
        
        runtime_minutes = (time.time() - self.start_time) / 60
        
        print("🎨" + "=" * 70 + "🎨")
        print("        M1-Optimized CLIP Training Monitor")
        print("    Contrastive Language-Image Pre-training")
        print("🎨" + "=" * 70 + "🎨")
        print()
        
        # Runtime info
        print(f"⏱️  Runtime: {runtime_minutes:.1f} minutes")
        print(f"🔄 Monitor Update #{self.iteration + 1}")
        print(f"📅 {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Training Process Status
        print("🧠 TRAINING PROCESS:")
        if process_info['running']:
            print(f"   ✅ Status: ACTIVE (PID: {process_info['pid']})")
            print(f"   🖥️  Process CPU: {process_info['cpu_percent']:.1f}%")
            print(f"   🧠 Process Memory: {process_info['memory_mb']:.1f} MB")
        else:
            print("   ❌ Status: NOT RUNNING")
        print()
        
        # System Performance
        print("📊 SYSTEM PERFORMANCE (M1 MacBook Pro):")
        print(f"   🔧 Overall CPU: {system_info['cpu_percent']:.1f}%")
        print(f"   💾 Memory Usage: {system_info['memory_used_gb']:.1f}GB / 8GB "
              f"({system_info['memory_percent']:.1f}%)")
        print(f"   💽 Available RAM: {system_info['memory_available_gb']:.1f}GB")
        print(f"   🏗️  CPU Cores: {system_info['cpu_physical']} physical, "
              f"{system_info['cpu_logical']} logical")
        print()
        
        # Training Metrics
        print("🎯 TRAINING METRICS:")
        print(f"   📈 Current Train Loss: {training_metrics.get('final_train_loss', 0):.4f}")
        print(f"   📉 Current Val Loss: {training_metrics.get('final_val_loss', 0):.4f}")
        print(f"   🎪 Current Val Accuracy: {training_metrics.get('final_val_acc', 0):.3f}")
        print(f"   🏆 Best Val Accuracy: {training_metrics.get('best_val_acc', 0):.3f}")
        print(f"   🔢 Completed Epochs: {training_metrics.get('total_epochs', 0)}")
        print(f"   🧮 Model Parameters: {training_metrics.get('model_parameters', 0):,}")
        print()
        
        # Progress Estimation
        if process_info['running']:
            progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
            print(f"⏰ ESTIMATED PROGRESS:")
            print(f"   [{progress_bar}] {progress_pct:.1f}%")
            if eta_minutes > 0:
                print(f"   🎯 ETA: ~{eta_minutes:.1f} minutes remaining")
            else:
                print(f"   🎉 Training should be completing soon!")
        print()
        
        # CLIP-specific info
        print("🎨 CLIP MODEL INFO:")
        print("   • Architecture: Vision Transformer + Text Transformer")
        print("   • Task: Contrastive Language-Image Pre-training")
        print("   • Dataset: Multimodal image-text pairs")
        print("   • Optimization: M1-specific (8 cores, memory efficient)")
        print("   • Learning: Image-text similarity matching")
        print()
        
        print("⌨️  Press Ctrl+C to stop monitoring")
        print("🎨" + "=" * 70 + "🎨")
    
    def run(self):
        """Main monitoring loop"""
        print("\n🚀 Starting CLIP Training Monitor...")
        print("📊 Tracking multimodal learning progress on M1")
        print("=" * 60)
        
        try:
            while True:
                # Get current stats
                process_info = self.get_training_process()
                system_info = self.get_system_stats()
                training_metrics = self.load_training_metrics()
                
                # Store history
                current_time = time.time()
                self.timestamps.append(current_time)
                self.cpu_history.append(system_info['cpu_percent'])
                self.memory_history.append(system_info['memory_used_gb'] * 1024)  # Convert to MB
                
                # Keep only recent history (last 100 points)
                if len(self.timestamps) > 100:
                    self.timestamps = self.timestamps[-100:]
                    self.cpu_history = self.cpu_history[-100:]
                    self.memory_history = self.memory_history[-100:]
                
                # Calculate progress
                runtime_minutes = (current_time - self.start_time) / 60
                progress_pct, eta_minutes = self.estimate_progress(runtime_minutes, process_info)
                
                # Display status
                self.display_status(process_info, system_info, training_metrics, 
                                  progress_pct, eta_minutes)
                
                # Create live plot every 5 iterations
                if self.iteration % 5 == 0 and len(self.timestamps) > 2:
                    plot_path = os.path.join(self.checkpoint_dir, 'live_monitor.png')
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    self.create_live_plot(plot_path)
                
                # Wait for next update
                time.sleep(self.update_interval)
                self.iteration += 1
                
        except KeyboardInterrupt:
            print("\n\n🛑 CLIP Training Monitor Stopped")
            print(f"📊 Monitored for {(time.time() - self.start_time)/60:.1f} minutes")
            
            # Save final monitoring data
            if len(self.timestamps) > 0:
                final_plot = os.path.join(self.checkpoint_dir, 'final_monitor_summary.png')
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                self.create_live_plot(final_plot)
                print(f"💾 Final monitoring plot saved: {final_plot}")

def main():
    """Main execution"""
    print("🎨 M1-Optimized CLIP Training Monitor")
    print("=" * 50)
    
    monitor = CLIPTrainingMonitor(
        process_name="m1_clip_training.py",
        checkpoint_dir="./clip_checkpoints",
        update_interval=8  # Update every 8 seconds
    )
    
    monitor.run()

if __name__ == "__main__":
    main()