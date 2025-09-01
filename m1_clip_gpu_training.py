#!/usr/bin/env python3
"""
M1-Optimized CLIP Training Pipeline - GPU Accelerated Version
Real-world multimodal training with Apple Silicon MPS acceleration
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from m1_clip_model import create_m1_clip_model, contrastive_loss, count_parameters
from m1_clip_dataset import create_clip_dataloaders

# M1 GPU optimization
torch.set_num_threads(8)  # Still use all CPU cores for data loading

class M1CLIPGPUTrainer:
    """M1-optimized CLIP training manager with GPU acceleration"""
    
    def __init__(self, 
                 model_size: str = "small",
                 batch_size: int = 16,  # Slightly larger for GPU
                 learning_rate: float = 1e-4,
                 num_epochs: int = 20,
                 data_dir: str = "./clip_data",
                 checkpoint_dir: str = "./clip_checkpoints_gpu",
                 max_train_samples: int = 800,
                 max_val_samples: int = 200,
                 use_gpu: bool = True):
        
        # Device selection with fallback
        if use_gpu and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🚀 Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU (MPS not available or disabled)")
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_gpu = use_gpu
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize model
        print("🧠 Initializing M1-optimized CLIP model...")
        self.model = create_m1_clip_model(model_size)
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"📱 Model moved to: {self.device}")
        
        # Create datasets
        print("📚 Loading multimodal dataset...")
        self.train_loader, self.val_loader, vocab_size = create_clip_dataloaders(
            root_dir=data_dir,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples
        )
        
        # Update model vocab size if needed
        if hasattr(self.model.text_encoder, 'token_embedding'):
            current_vocab = self.model.text_encoder.token_embedding.num_embeddings
            if current_vocab != vocab_size:
                print(f"🔧 Updating vocabulary size: {current_vocab} → {vocab_size}")
                self.model.text_encoder.token_embedding = torch.nn.Embedding(vocab_size, self.model.d_model)
                torch.nn.init.normal_(self.model.text_encoder.token_embedding.weight, std=0.02)
                # Move updated embedding to device
                self.model.text_encoder.token_embedding = self.model.text_encoder.token_embedding.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs,
            eta_min=learning_rate * 0.1
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        
        # Performance tracking
        self.device_transfer_time = 0
        self.computation_time = 0
        
        print(f"✅ GPU CLIP trainer initialized:")
        print(f"   Model size: {model_size}")
        print(f"   Parameters: {count_parameters(self.model):,}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Training samples: {len(self.train_loader.dataset)}")
        print(f"   Validation samples: {len(self.val_loader.dataset)}")
    
    def compute_accuracy(self, logits: torch.Tensor) -> float:
        """Compute contrastive accuracy (image-text retrieval)"""
        batch_size = logits.shape[0]
        
        # Image-to-text accuracy
        i2t_preds = torch.argmax(logits, dim=1)
        i2t_correct = (i2t_preds == torch.arange(batch_size, device=logits.device)).float()
        i2t_acc = i2t_correct.mean()
        
        # Text-to-image accuracy
        t2i_preds = torch.argmax(logits.T, dim=1)
        t2i_correct = (t2i_preds == torch.arange(batch_size, device=logits.device)).float()
        t2i_acc = t2i_correct.mean()
        
        return (i2t_acc + t2i_acc) / 2
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch with GPU acceleration"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        epoch_transfer_time = 0
        epoch_compute_time = 0
        
        pbar = tqdm(self.train_loader, desc="GPU Training", leave=False)
        
        for batch_idx, (images, texts, attention_masks) in enumerate(pbar):
            # Time device transfer
            transfer_start = time.time()
            images = images.to(self.device, non_blocking=True)
            texts = texts.to(self.device, non_blocking=True)
            attention_masks = attention_masks.to(self.device, non_blocking=True)
            epoch_transfer_time += time.time() - transfer_start
            
            # Forward pass timing
            compute_start = time.time()
            self.optimizer.zero_grad()
            
            try:
                image_features, text_features, logits = self.model(images, texts, attention_masks)
                loss = contrastive_loss(logits)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_compute_time += time.time() - compute_start
                
                # Compute accuracy
                with torch.no_grad():
                    accuracy = self.compute_accuracy(logits)
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy.item():.3f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'Device': str(self.device)
                })
                
                # Print detailed progress
                if batch_idx % 50 == 0:
                    print(f"GPU Batch {batch_idx:3d}: Loss={loss.item():.4f}, Acc={accuracy.item():.3f}")
                
            except RuntimeError as e:
                print(f"⚠️  GPU Runtime error in batch {batch_idx}: {e}")
                # Try to continue or fallback to CPU
                continue
        
        # Store timing information
        self.device_transfer_time += epoch_transfer_time
        self.computation_time += epoch_compute_time
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate one epoch with GPU"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="GPU Validation", leave=False)
            
            for images, texts, attention_masks in pbar:
                images = images.to(self.device, non_blocking=True)
                texts = texts.to(self.device, non_blocking=True)
                attention_masks = attention_masks.to(self.device, non_blocking=True)
                
                try:
                    image_features, text_features, logits = self.model(images, texts, attention_masks)
                    loss = contrastive_loss(logits)
                    accuracy = self.compute_accuracy(logits)
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'Val Loss': f'{loss.item():.4f}',
                        'Val Acc': f'{accuracy.item():.3f}'
                    })
                    
                except RuntimeError as e:
                    print(f"⚠️  GPU Validation error: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'device': str(self.device),
            'device_transfer_time': self.device_transfer_time,
            'computation_time': self.computation_time
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint_gpu.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model_gpu.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best GPU model (Val Acc: {self.val_accuracies[-1]:.3f})")
    
    def train(self):
        """Main GPU training loop"""
        print(f"\n🚀 Starting CLIP GPU training on M1 MacBook Pro...")
        print(f"🎯 Device: {self.device}")
        print("=" * 60)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                epoch_start = time.time()
                
                print(f"\n📊 GPU Epoch {epoch + 1}/{self.num_epochs}")
                print("-" * 40)
                
                # Training
                train_loss, train_acc = self.train_epoch()
                
                # Validation
                val_loss, val_acc = self.validate_epoch()
                
                # Update scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Store metrics
                epoch_time = time.time() - epoch_start
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                self.learning_rates.append(current_lr)
                self.epoch_times.append(epoch_time)
                
                # Print epoch summary
                print(f"\n📈 GPU Epoch {epoch + 1} Results:")
                print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
                print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.3f}")
                print(f"   Learning Rate: {current_lr:.6f}")
                print(f"   Time: {epoch_time:.1f}s")
                
                # Save checkpoint
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                
                self.save_checkpoint(epoch, is_best)
                
                # Early stopping check
                if val_acc > 0.8:
                    print(f"🎯 Early stopping: Target accuracy reached!")
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  GPU Training interrupted by user")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 GPU CLIP Training Complete!")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🏆 Best validation accuracy: {best_val_acc:.3f}")
        print(f"💾 Checkpoints saved to: {self.checkpoint_dir}")
        print(f"🚀 Device transfer time: {self.device_transfer_time:.1f}s")
        print(f"⚡ Computation time: {self.computation_time:.1f}s")
        print("=" * 60)
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def save_training_plots(self):
        """Create and save GPU training visualization plots"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('GPU CLIP Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Contrastive Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plots
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('GPU CLIP Multimodal Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training time per epoch
        ax3.plot(epochs, self.epoch_times, 'g-', linewidth=2, marker='o')
        ax3.set_title('GPU Training Time per Epoch', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)
        
        # Performance comparison
        if self.device_transfer_time > 0 and self.computation_time > 0:
            transfer_pct = self.device_transfer_time / (self.device_transfer_time + self.computation_time)
            compute_pct = 1 - transfer_pct
            
            ax4.pie([compute_pct, transfer_pct], 
                   labels=['Computation', 'Data Transfer'], 
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightcoral'])
            ax4.set_title('GPU Time Distribution', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'M1-Optimized CLIP GPU Training Results\n'
                    f'Device: {self.device}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.checkpoint_dir, 'clip_gpu_training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 GPU Training plots saved to: {plot_path}")

def main():
    """Main execution function for GPU training"""
    print("🎨 M1-Optimized CLIP GPU Training Pipeline")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Initialize GPU trainer
    trainer = M1CLIPGPUTrainer(
        model_size="small",        # ~50M parameters
        batch_size=16,             # Slightly larger for GPU
        learning_rate=3e-4,        # CLIP standard
        num_epochs=10,             # Shorter for comparison
        max_train_samples=600,     # Same dataset size
        max_val_samples=150,
        use_gpu=True               # Enable GPU acceleration
    )
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs = trainer.train()
    
    # Generate plots
    trainer.save_training_plots()
    
    # Save final metrics
    metrics = {
        'device': str(trainer.device),
        'final_train_loss': float(train_losses[-1]) if train_losses else 0,
        'final_val_loss': float(val_losses[-1]) if val_losses else 0,
        'final_train_acc': float(train_accs[-1]) if train_accs else 0,
        'final_val_acc': float(val_accs[-1]) if val_accs else 0,
        'best_val_acc': float(max(val_accs)) if val_accs else 0,
        'total_epochs': len(train_losses),
        'model_parameters': count_parameters(trainer.model),
        'average_epoch_time': np.mean(trainer.epoch_times) if trainer.epoch_times else 0,
        'device_transfer_time': trainer.device_transfer_time,
        'computation_time': trainer.computation_time
    }
    
    with open(os.path.join(trainer.checkpoint_dir, 'gpu_training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n🎯 Final GPU Results:")
    print(f"   Device used: {metrics['device']}")
    print(f"   Best validation accuracy: {metrics['best_val_acc']:.3f}")
    print(f"   Average epoch time: {metrics['average_epoch_time']:.1f}s")
    print(f"   Model parameters: {metrics['model_parameters']:,}")

if __name__ == "__main__":
    main()