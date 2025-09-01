#!/usr/bin/env python3
"""
Quick test to verify CLIP setup
"""

import torch
import os
import sys
from pathlib import Path

def test_clip_imports():
    """Test if all CLIP modules can be imported"""
    print("🧪 Testing CLIP module imports...")
    
    try:
        from m1_clip_model import create_m1_clip_model, contrastive_loss
        print("✅ m1_clip_model imported successfully")
        
        from m1_clip_dataset import create_clip_dataloaders
        print("✅ m1_clip_dataset imported successfully")
        
        # Quick model test
        model = create_m1_clip_model("tiny")  # Smallest size for quick test
        print(f"✅ Created tiny CLIP model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        texts = torch.randint(0, 1000, (batch_size, 77))
        
        with torch.no_grad():
            image_features, text_features, logits = model(images, texts)
            loss = contrastive_loss(logits)
        
        print(f"✅ Forward pass successful:")
        print(f"   Image features: {image_features.shape}")
        print(f"   Text features: {text_features.shape}")
        print(f"   Logits: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n🔧 Testing dependencies...")
    
    required = ['torch', 'torchvision', 'PIL', 'matplotlib', 'numpy', 'tqdm']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg} available")
        except ImportError:
            missing.append(pkg)
            print(f"❌ {pkg} missing")
    
    return len(missing) == 0

def main():
    print("🎨 M1-Optimized CLIP Setup Test")
    print("=" * 40)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test CLIP modules
    clip_ok = test_clip_imports()
    
    print("\n" + "=" * 40)
    if deps_ok and clip_ok:
        print("🎉 All tests passed! CLIP setup is ready.")
        print("\n🚀 To start training, run:")
        print("   python3 run_clip_training.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        if not deps_ok:
            print("\n💡 Install missing dependencies with:")
            print("   pip install torch torchvision matplotlib numpy tqdm pillow")
    
    print("=" * 40)

if __name__ == "__main__":
    main()