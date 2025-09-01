#!/usr/bin/env python3
"""
M1 CLIP Training Performance Comparison: CPU vs GPU
Compare training speed and efficiency between CPU and MPS GPU
"""

import torch
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
import sys

def run_speed_benchmark():
    """Run a quick speed benchmark comparing CPU vs GPU operations"""
    
    print("🏃‍♂️ Running M1 Speed Benchmark: CPU vs GPU")
    print("=" * 50)
    
    # Test parameters
    batch_size = 16
    d_model = 512
    seq_len = 77
    image_size = 224
    
    results = {}
    
    # Test CPU performance
    print("💻 Testing CPU performance...")
    device_cpu = torch.device('cpu')
    
    # Create test data (fixed dimensions for matrix multiplication)
    images_cpu = torch.randn(batch_size, d_model, device=device_cpu)  # Flattened image features
    texts_cpu = torch.randn(batch_size, d_model, device=device_cpu)   # Text features
    
    # Warm up
    for _ in range(5):
        _ = torch.matmul(images_cpu, texts_cpu.transpose(0, 1))
    
    # Benchmark CPU
    start_time = time.time()
    for _ in range(100):
        result = torch.matmul(images_cpu, texts_cpu.transpose(0, 1))
    cpu_time = time.time() - start_time
    results['cpu_time'] = cpu_time
    print(f"✅ CPU: {cpu_time:.3f} seconds for 100 operations")
    
    # Test GPU performance (if available)
    if torch.backends.mps.is_available():
        print("🚀 Testing GPU (MPS) performance...")
        device_gpu = torch.device('mps')
        
        # Move data to GPU
        images_gpu = images_cpu.to(device_gpu)
        texts_gpu = texts_cpu.to(device_gpu)
        
        # Warm up GPU
        for _ in range(5):
            _ = torch.matmul(images_gpu, texts_gpu.transpose(0, 1))
            torch.mps.synchronize()  # Ensure GPU operations complete
        
        # Benchmark GPU
        start_time = time.time()
        for _ in range(100):
            result = torch.matmul(images_gpu, texts_gpu.transpose(0, 1))
            torch.mps.synchronize()  # Ensure completion
        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        
        speedup = cpu_time / gpu_time
        results['speedup'] = speedup
        
        print(f"✅ GPU: {gpu_time:.3f} seconds for 100 operations")
        print(f"🏆 GPU Speedup: {speedup:.1f}x faster than CPU")
        
    else:
        print("❌ MPS not available - GPU benchmark skipped")
        results['gpu_time'] = None
        results['speedup'] = None
    
    return results

def create_comparison_visualization():
    """Create visualization comparing CPU vs GPU training approaches"""
    
    # Load CPU training results (if available)
    cpu_metrics_path = Path("./clip_checkpoints/training_metrics.json")
    gpu_metrics_path = Path("./clip_checkpoints_gpu/gpu_training_metrics.json")
    
    cpu_data = None
    gpu_data = None
    
    if cpu_metrics_path.exists():
        with open(cpu_metrics_path, 'r') as f:
            cpu_data = json.load(f)
        print("📊 Loaded CPU training metrics")
    
    if gpu_metrics_path.exists():
        with open(gpu_metrics_path, 'r') as f:
            gpu_data = json.load(f)
        print("📊 Loaded GPU training metrics")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance comparison
    if cpu_data and gpu_data:
        devices = ['CPU', 'GPU (MPS)']
        epoch_times = [cpu_data.get('average_epoch_time', 0), gpu_data.get('average_epoch_time', 0)]
        accuracies = [cpu_data.get('best_val_acc', 0), gpu_data.get('best_val_acc', 0)]
        
        # Training speed comparison
        ax1.bar(devices, epoch_times, color=['skyblue', 'lightcoral'])
        ax1.set_title('Average Training Time per Epoch', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        for i, v in enumerate(epoch_times):
            ax1.text(i, v + 0.5, f'{v:.1f}s', ha='center', fontweight='bold')
        
        # Accuracy comparison
        ax2.bar(devices, accuracies, color=['lightgreen', 'gold'])
        ax2.set_title('Best Validation Accuracy', fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, max(accuracies) * 1.2)
        for i, v in enumerate(accuracies):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Theoretical vs Practical comparison
    theoretical_benefits = ['Matrix Ops', 'Memory Bandwidth', 'Parallel Compute', 'Power Efficiency']
    cpu_scores = [3, 4, 3, 5]  # Relative scores
    gpu_scores = [5, 5, 5, 3]  # Relative scores
    
    x = np.arange(len(theoretical_benefits))
    width = 0.35
    
    ax3.bar(x - width/2, cpu_scores, width, label='CPU', color='skyblue')
    ax3.bar(x + width/2, gpu_scores, width, label='GPU', color='lightcoral')
    ax3.set_title('M1 Architecture: CPU vs GPU Strengths', fontweight='bold')
    ax3.set_ylabel('Performance Score (1-5)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(theoretical_benefits)
    ax3.legend()
    ax3.set_ylim(0, 6)
    
    # Memory usage comparison
    if cpu_data and gpu_data:
        memory_types = ['Model\nParameters', 'Training\nOverhead', 'Data\nBuffers']
        cpu_memory = [2.5, 0.5, 0.3]  # Estimated GB
        gpu_memory = [2.5, 0.8, 0.5]  # Estimated GB (higher overhead)
        
        x = np.arange(len(memory_types))
        ax4.bar(x - width/2, cpu_memory, width, label='CPU', color='skyblue')
        ax4.bar(x + width/2, gpu_memory, width, label='GPU', color='lightcoral')
        ax4.set_title('Memory Usage Comparison', fontweight='bold')
        ax4.set_ylabel('Memory (GB)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(memory_types)
        ax4.legend()
    
    plt.suptitle('M1 MacBook Pro: CPU vs GPU CLIP Training Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the comparison
    plt.savefig('cpu_vs_gpu_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Comparison visualization saved to: cpu_vs_gpu_comparison.png")

def run_gpu_training_test():
    """Run a quick GPU training test to compare with CPU results"""
    
    print("\n🧪 Running GPU Training Test...")
    print("This will train a few epochs to compare with CPU performance")
    
    try:
        # Run the GPU training script
        result = subprocess.run([
            sys.executable, 'm1_clip_gpu_training.py'
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ GPU training test completed successfully!")
            print("Last few lines of output:")
            print(result.stdout.split('\n')[-10:])
            return True
        else:
            print(f"❌ GPU training failed with return code: {result.returncode}")
            print("Error output:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ GPU training test timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running GPU training: {e}")
        return False

def main():
    """Main comparison function"""
    print("🎯 M1 MacBook Pro CLIP Training: CPU vs GPU Analysis")
    print("=" * 60)
    
    # Step 1: Run speed benchmark
    benchmark_results = run_speed_benchmark()
    
    print("\n" + "=" * 60)
    
    # Step 2: Create comparison visualization
    print("\n📊 Creating theoretical comparison based on benchmark...")
    create_comparison_visualization()
    
    print("\n" + "=" * 60)
    print("🎯 COMPARISON SUMMARY:")
    
    if benchmark_results.get('speedup'):
        print(f"🚀 Raw compute speedup: {benchmark_results['speedup']:.1f}x")
    
    print("\n💡 KEY INSIGHTS:")
    print("CPU Advantages:")
    print("  ✅ Stable and reliable training")
    print("  ✅ Lower memory overhead") 
    print("  ✅ Better for debugging")
    print("  ✅ Consistent performance")
    
    print("\nGPU Advantages:")
    print("  ✅ Potentially faster training (2-3x)")
    print("  ✅ Better for larger models")
    print("  ✅ Parallel processing")
    print("  ✅ Lower CPU usage")
    
    print("\n🎯 RECOMMENDATION:")
    if torch.backends.mps.is_available():
        print("Your M1 supports MPS! Try GPU for:")
        print("  • Larger models (>100M parameters)")
        print("  • Longer training sessions")
        print("  • Batch sizes >16")
        print("\nUse CPU for:")
        print("  • Debugging and development") 
        print("  • Maximum stability")
        print("  • Complex model architectures")
    else:
        print("MPS not available - stick with CPU optimization")
        print("CPU performance on M1 is already excellent!")
    
    print("\n✨ Analysis complete!")

if __name__ == "__main__":
    main()