# Task 2: Neural Network Quantization

## Overview

This task implements **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** techniques to compress neural networks while maintaining accuracy. Experiments are conducted on VGG-11 (CIFAR-100) and MobileNetV2 (Food-101) using various quantization strategies including INT8, INT4, FP16, and mixed precision approaches.

---

## Notebook Structure

### **Cell 1-2: Environment Setup**

Mounts Google Drive and imports required PyTorch libraries.

### **Cell 3: Configuration**

Sets batch size and device (GPU/CPU) for training and inference.

### **Cell 4: Dataset Loading**

Loads CIFAR-100 dataset with appropriate transforms for training and testing (resize to 224x224, normalization).

### **Cell 5: Model Preparation**

Loads VGG-11 with ImageNet pre-trained weights and modifies final layer for CIFAR-100 (112 classes - note: should be 100).

### **Cell 6: Directory Navigation**

Changes to Google Drive working directory.

### **Cell 7: Scaling Law Visualization**

Creates plot comparing PTQ vs QAT accuracy across different bit-widths (32, 16, 8, 4 bits). Shows QAT outperforms PTQ at lower bit-widths.

### **Cell 8: FP32 Baseline Training**

Fine-tunes VGG-11 on CIFAR-100 for 10 epochs, saves best model, and establishes baseline accuracy.

### **Cell 9: Install THOP**

Installs `thop` package for computing model FLOPs/MACs.

### **Cell 10: Comprehensive PTQ Analysis (VGG-11)**

Implements extensive PTQ experiments using TorchAO library:

- Tests FP16, BF16, INT8 (dynamic, static, weight-only), INT4 quantization
- Implements custom mixed precision strategies (simple and variance-based)
- Measures accuracy, inference time, speedup, VRAM, disk size, and TBOPs
- Saves all quantized models and results to JSON

### **Cell 11: Outlier-Aware Quantization**

Implements activation outlier detection and clipping-based quantization:

- Collects activation statistics from calibration set
- Generates histograms for activation distributions
- Tests INT8/INT4 quantization with/without percentile-based clipping
- Evaluates impact on accuracy

### **Cell 12: Accuracy vs Size Trade-off Plot**

Visualizes model size vs accuracy trade-off for FP32, uniform INT4, and mixed precision approaches.

### **Cell 13: PTQ on VGG-11 (Alternative)**

Alternative PTQ implementation with additional configurations (FP16, BF16, mixed precision variants).

### **Cell 14: MobileNetV2 Training on Food-101**

Fine-tunes MobileNetV2 with ImageNet weights on Food-101 dataset (101 food categories) for 5 epochs.

### **Cell 15: Food-101 Dataset Reload**

Reloads Food-101 dataset with proper transforms.

### **Cell 16: PTQ on MobileNetV2**

Applies various PTQ methods to MobileNetV2 and evaluates on Food-101 test set.

### **Cell 17: Quantization-Aware Training (QAT)**

Implements QAT using PyTorch's native QAT API:

- INT8 static QAT with fine-tuning
- FP16 and BF16 fine-tuning with mixed precision
- Uses GradScaler for FP16 training
- Fine-tunes for 2 epochs and saves best models

### **Cell 18: System Diagnostics**

Checks GPU, CUDA, PyTorch setup, FFmpeg installation, and troubleshoots CUTLASS initialization issues.

---

## Quantization Methods Explored

| Method               | Type          | Precision                     | Key Feature                     |
| -------------------- | ------------- | ----------------------------- | ------------------------------- |
| **FP16**             | Dtype Casting | 16-bit float                  | Simple speedup                  |
| **BF16**             | Dtype Casting | 16-bit bfloat                 | Better numerical stability      |
| **INT8 Dynamic**     | PTQ           | 8-bit int                     | Dynamic activation quantization |
| **INT8 Static**      | PTQ           | 8-bit int                     | Calibrated quantization         |
| **INT8 Weight-Only** | PTQ           | 8-bit weights, FP activations | Memory compression              |
| **INT4 Weight-Only** | PTQ           | 4-bit weights                 | Maximum compression             |
| **Simple Mixed**     | PTQ           | FP16 + INT4                   | Linear layers INT4, others FP16 |
| **Variance Mixed**   | PTQ           | FP16 + INT4                   | Low-variance layers INT4        |
| **Outlier-Aware**    | PTQ           | INT8/INT4 with clipping       | Handles activation outliers     |
| **INT8 QAT**         | QAT           | 8-bit int                     | Fine-tuned quantization         |

---

## Key Findings

1. **Scaling Law**: QAT consistently outperforms PTQ at lower bit-widths (8-bit, 4-bit)
2. **Accuracy Trade-off**: INT4 shows ~2-3% accuracy drop, INT8 minimal drop (~0.5%)
3. **Compression**: INT4 achieves ~5x disk size reduction, INT8 ~4x reduction
4. **Mixed Precision**: Adaptive strategies can maintain accuracy while reducing size
5. **Outlier Impact**: Activation clipping improves INT4 quantization stability

---

## Expected Outputs

### Models Saved

1. `vgg16_cifar100_fp32.pth` - FP32 baseline VGG-11
2. `mobilenetv2_food101_fp32.pth` - FP32 baseline MobileNetV2
3. `saved_ptq_models_torchao/*.pth` - All PTQ quantized models
4. `saved_qat_models/*.pth` - QAT fine-tuned models
5. `outlier_quant_results/*.pth` - Outlier-aware quantized models

### Analysis Outputs

- `ptq_results_torchao_final.json` - Comprehensive PTQ metrics
- `ptq_results_mobilenet_torchao_final.json` - MobileNetV2 PTQ results
- `qat_finetune_results.json` - QAT fine-tuning results
- `activation_stats.json` - Activation statistics for outlier analysis

### Visualizations

- `scaling_law_plot.png` - PTQ vs QAT accuracy comparison
- `accuracy_vs_size_plot.png` - Size-accuracy trade-off visualization
- `outlier_quant_results/hist_*.png` - Activation distribution histograms

---

## Key Hyperparameters

- **Models**: VGG-11 (CIFAR-100), MobileNetV2 (Food-101)
- **FP32 Training**: 10 epochs (VGG-11), 5 epochs (MobileNetV2)
- **QAT Fine-tuning**: 2 epochs
- **Batch Size**: 64
- **Learning Rate**: 0.001 (FP32), 1e-5 (QAT)
- **Calibration**: 4 batches for INT8 static
- **Outlier Clipping**: 99.9th, 99th, 95th percentiles

---

## Requirements

- PyTorch with CUDA support
- torchvision
- torchao (for quantization APIs)
- thop (for FLOPs calculation)
- numpy
- matplotlib
- CIFAR-100 dataset (auto-downloaded)
- Food-101 dataset (auto-downloaded)

## Notes

- GPU highly recommended for training and PTQ
- INT4 quantization may require CPU execution depending on backend
- TorchAO provides efficient quantization implementations
- Mixed precision training uses GradScaler for numerical stability
- Outlier-aware quantization requires calibration data
