#!/usr/bin/env python3
"""
M1-Optimized CLIP Training Pipeline
Real-world multimodal contrastive learning for M1 MacBook Pro
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time
import os
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from m1_clip_model import create_m1_clip_model, contrastive_loss, count_parameters
from m1_clip_dataset import create_clip_dataloaders

# M1 optimizations
torch.set_num_threads(8)  # Use all 8 M1 cores
torch.backends.mps.is_available = lambda: False  # CPU for stability

class M1CLIPTrainer:
    """M1-optimized CLIP training manager"""
    
    def __init__(self, 
                 model_size: str = "small",
                 batch_size: int = 16,  # Smaller batch for 8GB RAM
                 learning_rate: float = 1e-4,
                 num_epochs: int = 20,
                 data_dir: str = "./clip_data",
                 checkpoint_dir: str = "./clip_checkpoints",
                 max_train_samples: int = 800,
                 max_val_samples: int = 200):
        
        self.device = torch.device('cpu')  # M1 CPU optimization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize model
        print("🧠 Initializing M1-optimized CLIP model...")
        self.model = create_m1_clip_model(model_size)
        self.model = self.model.to(self.device)
        
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
        
        # JIT optimization for M1
        self.model = torch.jit.script(self.model)
        
        print(f"✅ CLIP trainer initialized:")
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
        
        # Image-to-text accuracy (diagonal elements should be highest in each row)
        i2t_preds = torch.argmax(logits, dim=1)
        i2t_correct = (i2t_preds == torch.arange(batch_size, device=logits.device)).float()
        i2t_acc = i2t_correct.mean()
        
        # Text-to-image accuracy (diagonal elements should be highest in each column)
        t2i_preds = torch.argmax(logits.T, dim=1)
        t2i_correct = (t2i_preds == torch.arange(batch_size, device=logits.device)).float()
        t2i_acc = t2i_correct.mean()
        
        # Average accuracy
        return (i2t_acc + t2i_acc) / 2
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, texts, attention_masks) in enumerate(pbar):
            images = images.to(self.device)
            texts = texts.to(self.device)
            attention_masks = attention_masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                image_features, text_features, logits = self.model(images, texts, attention_masks)
                loss = contrastive_loss(logits)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
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
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Print detailed progress every 50 batches
                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx:3d}: Loss={loss.item():.4f}, Acc={accuracy.item():.3f}")
                
            except RuntimeError as e:
                print(f"⚠️  Runtime error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for images, texts, attention_masks in pbar:
                images = images.to(self.device)
                texts = texts.to(self.device)
                attention_masks = attention_masks.to(self.device)
                
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
                    print(f"⚠️  Validation error: {e}")
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
            'learning_rates': self.learning_rates
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (Val Acc: {self.val_accuracies[-1]:.3f})")
    
    def train(self):
        """Main training loop"""
        print("\n🚀 Starting CLIP training on M1 MacBook Pro...")
        print("=" * 60)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                epoch_start = time.time()
                
                print(f"\n📊 Epoch {epoch + 1}/{self.num_epochs}")
                print("-" * 40)
                
                # Training
                train_loss, train_acc = self.train_epoch()
                
                # Validation
                val_loss, val_acc = self.validate_epoch()
                
                # Update scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Store metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                self.learning_rates.append(current_lr)
                
                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"\n📈 Epoch {epoch + 1} Results:")
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
                if val_acc > 0.8:  # Stop if we achieve 80% accuracy
                    print(f"🎯 Early stopping: Target accuracy reached!")
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 CLIP Training Complete!")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🏆 Best validation accuracy: {best_val_acc:.3f}")
        print(f"💾 Checkpoints saved to: {self.checkpoint_dir}")
        print("=" * 60)
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def save_training_plots(self):
        """Create and save training visualization plots"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('CLIP Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Contrastive Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plots
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('CLIP Multimodal Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Final accuracy comparison
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0
        
        ax4.bar(['Training', 'Validation'], [final_train_acc, final_val_acc], 
                color=['blue', 'red'], alpha=0.7)
        ax4.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        ax4.text(0, final_train_acc + 0.01, f'{final_train_acc:.3f}', 
                ha='center', va='bottom', fontweight='bold')
        ax4.text(1, final_val_acc + 0.01, f'{final_val_acc:.3f}', 
                ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('M1-Optimized CLIP Training Results\nMultimodal Image-Text Learning', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.checkpoint_dir, 'clip_training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plots saved to: {plot_path}")

def main():
    """Main execution function"""
    print("🎨 M1-Optimized CLIP Training Pipeline")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("Device: M1 MacBook Pro CPU")
    
    # Initialize trainer with M1-optimized settings
    trainer = M1CLIPTrainer(
        model_size="small",        # ~50M parameters for 8GB RAM
        batch_size=12,             # Conservative for memory
        learning_rate=3e-4,        # CLIP standard
        num_epochs=15,             # Reasonable for demonstration
        max_train_samples=600,     # Manageable dataset size
        max_val_samples=150
    )
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs = trainer.train()
    
    # Generate plots
    trainer.save_training_plots()
    
    # Save final metrics
    metrics = {
        'final_train_loss': float(train_losses[-1]) if train_losses else 0,
        'final_val_loss': float(val_losses[-1]) if val_losses else 0,
        'final_train_acc': float(train_accs[-1]) if train_accs else 0,
        'final_val_acc': float(val_accs[-1]) if val_accs else 0,
        'best_val_acc': float(max(val_accs)) if val_accs else 0,
        'total_epochs': len(train_losses),
        'model_parameters': count_parameters(trainer.model)
    }
    
    with open(os.path.join(trainer.checkpoint_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n🎯 Final Results:")
    print(f"   Best validation accuracy: {metrics['best_val_acc']:.3f}")
    print(f"   Model parameters: {metrics['model_parameters']:,}")
    print(f"   Total epochs completed: {metrics['total_epochs']}")

if __name__ == "__main__":
    main()