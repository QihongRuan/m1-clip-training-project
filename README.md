# M1-Optimized CLIP Training Project

**Contrastive Language-Image Pre-training for M1 MacBook Pro**

This project implements a complete CLIP (Contrastive Language-Image Pre-training) model optimized specifically for Apple M1 MacBook Pro with real-world multimodal datasets.

## 🎯 Project Overview

CLIP learns to understand images and text together by training on image-text pairs. The model learns to match images with their corresponding descriptions, enabling powerful multimodal understanding capabilities.

### Key Features
- **M1-Optimized Architecture**: Designed for 8GB RAM and M1's 8-core CPU
- **Real-World Dataset**: Multimodal image-text pairs (Flickr8k-style)
- **Memory Efficient**: ~50M parameters for optimal M1 performance
- **Real-Time Monitoring**: Live training progress visualization
- **Complete Pipeline**: From dataset loading to model evaluation

## 🏗️ Architecture

### CLIP Model Components
1. **Vision Encoder**: Vision Transformer (ViT) for image processing
2. **Text Encoder**: Text Transformer for language understanding
3. **Contrastive Learning**: Matches images and text in shared embedding space

### M1 Optimizations
- 8-thread CPU utilization
- Memory-efficient attention mechanisms
- JIT compilation for performance
- Optimized data loading pipelines

## 📁 Project Structure

```
m1-pytorch-training-project/
├── m1_clip_model.py          # Core CLIP model implementation
├── m1_clip_dataset.py        # Multimodal dataset loader
├── m1_clip_training.py       # Training pipeline
├── clip_monitor.py           # Real-time monitoring
├── run_clip_training.py      # Main launcher script
├── clip_checkpoints/         # Model checkpoints and results
├── clip_data/               # Dataset storage
└── logs/                    # Training logs
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision matplotlib numpy tqdm psutil pillow
```

### Training CLIP Model

**Option 1: Automatic Launch (Recommended)**
```bash
python3 run_clip_training.py
```

**Option 2: Manual Training**
```bash
# Terminal 1: Start training
python3 m1_clip_training.py

# Terminal 2: Start monitoring
python3 clip_monitor.py
```

### Training Configuration
- **Model Size**: Small (~50M parameters)
- **Batch Size**: 12 (optimized for 8GB RAM)
- **Dataset**: 600 training + 150 validation samples
- **Duration**: ~40 minutes (15 epochs)
- **Memory Usage**: ~2-3GB peak

## 📊 Expected Results

### Training Metrics
- **Final Accuracy**: 40-60% (image-text retrieval)
- **Memory Efficient**: <3GB RAM usage
- **M1 Optimized**: 90%+ CPU utilization
- **Real-World Data**: Actual multimodal learning

### Output Files
- `clip_checkpoints/best_model.pth` - Best model weights
- `clip_checkpoints/training_metrics.json` - Final metrics
- `clip_checkpoints/clip_training_results.png` - Training plots
- `clip_checkpoints/live_monitor.png` - System monitoring

## 🔬 Technical Details

### Model Architecture
```python
M1CLIP(
    vision_encoder: M1VisionEncoder(
        image_size=224, patch_size=16,
        d_model=512, layers=12, heads=8
    ),
    text_encoder: M1TextEncoder(
        vocab_size=~200, max_length=77,
        d_model=512, layers=12, heads=8
    ),
    projection_dim=512
)
```

### Training Process
1. **Dataset Loading**: Multimodal image-text pairs
2. **Contrastive Learning**: Maximize similarity for correct pairs
3. **Evaluation**: Image-to-text and text-to-image retrieval
4. **Optimization**: AdamW with cosine annealing schedule

### M1 Optimizations
- **CPU Threading**: `torch.set_num_threads(8)`
- **JIT Compilation**: `torch.jit.script(model)`
- **Memory Management**: Gradient clipping and efficient attention
- **Data Loading**: 4 workers with persistent workers

## 📈 Monitoring Features

The real-time monitor displays:
- **System Performance**: CPU and memory usage
- **Training Progress**: Loss curves and accuracy metrics
- **Process Status**: Training state and estimated completion
- **Live Plots**: Real-time visualization updates

## 🎨 Understanding CLIP

### What CLIP Learns
- **Visual Understanding**: Recognizes objects, scenes, activities
- **Language Comprehension**: Processes natural language descriptions
- **Multimodal Alignment**: Connects visual and textual concepts
- **Zero-Shot Capability**: Generalizes to new image-text pairs

### Applications
- Image search with text queries
- Automatic image captioning
- Visual question answering
- Content-based image retrieval

## ⚠️ Memory Considerations

For M1 MacBook Pro (8GB RAM):
- **Training**: ~2-3GB peak memory usage
- **System Available**: ~5GB for other applications
- **Monitoring**: Minimal additional overhead
- **Swap Usage**: Should remain minimal

## 🔧 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch_size in `m1_clip_training.py`
2. **Slow Training**: Check CPU utilization in monitor
3. **Process Not Found**: Ensure training script is running
4. **Dependencies**: Install required packages listed above

### Performance Tuning
- Adjust `batch_size` based on available RAM
- Modify `max_train_samples` for faster experimentation
- Change `num_workers` in dataset loading

## 📚 Educational Value

This project demonstrates:
- **Multimodal AI**: How vision and language models work together
- **Contrastive Learning**: Self-supervised learning techniques
- **M1 Optimization**: Hardware-specific performance tuning
- **Real-World ML**: Complete training pipeline with monitoring

## 🎯 Next Steps

Potential extensions:
- Fine-tune on specific domains (medical, scientific images)
- Implement zero-shot image classification
- Add image generation capabilities
- Scale to larger datasets (COCO, Conceptual Captions)

## 📄 License

Created with Claude Code for educational and research purposes.

---

**🚀 Ready to train your CLIP model on M1? Run `python3 run_clip_training.py` to get started!**