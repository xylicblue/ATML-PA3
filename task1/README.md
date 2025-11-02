# Task 1: Neural Network Pruning

## Overview

This task implements both **unstructured** and **structured pruning** techniques on a VGG-11 model fine-tuned on CIFAR-10. The goal is to reduce model size while maintaining accuracy through systematic pruning and fine-tuning strategies. Additionally, **Grad-CAM visualizations** are used to interpret and compare model attention across pruned variants.

---

## Repository Structure

- **`task1.ipynb`**: Main notebook implementing pruning techniques and experiments
- **`gradcam.ipynb`**: Grad-CAM visualization tool for comparing model interpretability

---

## Notebook Structure: task1.ipynb

### **PART 1: UNSTRUCTURED PRUNING**

#### Cell 1: Memory Management

Clears GPU memory cache for clean execution environment.

#### Cell 2: Initial Setup and Model Loading

Loads CIFAR-10 dataset, initializes VGG-11 model, and modifies final layer for 10-class classification.

#### Cell 3: Unstructured Pruning Visualization

Demonstrates unstructured pruning on first convolutional layer and visualizes weight distributions before/after pruning.

#### Cell 4: Model Fine-Tuning and Baseline Evaluation

Fine-tunes VGG-11 on CIFAR-10 (or loads pre-trained model) and evaluates baseline accuracy.

---

### **PART 2: UNSTRUCTURED PRUNING SENSITIVITY ANALYSIS**

#### Cell 5: Layer-by-Layer Sensitivity Analysis

Tests each convolutional layer at different sparsity levels to identify which layers are most sensitive to pruning. Generates visualization plots and caches results.

---

### **PART 3: FINAL UNSTRUCTURED PRUNING**

#### Cell 6: Apply Layer-Wise Pruning Strategy

Applies optimized pruning strategy based on sensitivity analysis, with different sparsity ratios per layer. Saves final pruned model.

---

### **STRUCTURED PRUNING: PART 1**

#### Cell 7: Structured Pruning Visualization

Demonstrates structured pruning (filter removal) on a convolutional layer and visualizes the effect on weight distributions.

---

### **STRUCTURED PRUNING: PART 2**

#### Cell 8: Structured Pruning Sensitivity Analysis

Analyzes how each layer responds to structured pruning at various sparsity levels. Creates heatmap visualization showing accuracy drops.

---

### **STRUCTURED PRUNING: PART 3**

#### Cell 9: Memory Cleanup

Clears GPU memory before intensive operations.

#### Cell 10: Create Structurally Pruned Model

Builds new VGG-11 architecture with reduced filters based on sensitivity analysis. Reports compression ratio and accuracy.

#### Cell 11: Fine-Tune Pruned Model

Recovers accuracy through fine-tuning and saves final optimized model

---

## Notebook Structure: gradcam.ipynb

### **Cell 1: Google Drive Setup**

Mounts Google Drive for accessing saved model files.

### **Cell 2: Grad-CAM Visualization Pipeline**

Implements complete Grad-CAM visualization comparing three model variants:

- Loads original, unstructured pruned, and structured pruned models
- Applies Grad-CAM to visualize where each model focuses attention
- Generates side-by-side comparison showing original image and heatmaps for all models
- Color-codes predictions (green for correct, red for incorrect)
- Saves high-resolution comparison image

**Purpose**: Analyze whether pruning affects model interpretability and attention patterns on CIFAR-10 test images.

---

## Expected Outputs

### Models Saved

1. `vgg11_cifar10_finetuned.pth` - Baseline fine-tuned model
2. `unstructured_pruned_vgg11.pth` - Model with unstructured pruning applied
3. `vgg11_cifar10_structured_pruned.pth` - Structurally pruned model (before fine-tuning)
4. `vgg11_cifar10_structured_pruned_finetuned.pth` - Final pruned and fine-tuned model

### Cached Results

- `unstructured_sensitivity_results.pth` - Sensitivity analysis data for unstructured pruning

### Visualizations

- Weight distribution histograms (before/after pruning)
- Sensitivity analysis line plots (unstructured)
- Sensitivity heatmap (structured)
- **Grad-CAM comparison images** (`grad_cam_comparison.png`)

---

## Workflow Summary

1. **Setup**: Load and fine-tune VGG-11 on CIFAR-10
2. **Unstructured Pruning**:
   - Analyze layer sensitivity
   - Apply layer-specific pruning based on sensitivity
   - Evaluate pruned model
3. **Structured Pruning**:
   - Analyze filter importance per layer
   - Remove entire filters based on L1 norms
   - Fine-tune to recover accuracy
4. **Comparison**: Compare model sizes and accuracies
5. **Interpretability Analysis**: Use Grad-CAM to visualize attention patterns across model variants

---

## Key Differences: Unstructured vs Structured Pruning

| Aspect                  | Unstructured                   | Structured                   |
| ----------------------- | ------------------------------ | ---------------------------- |
| **Granularity**         | Individual weights             | Entire filters/channels      |
| **Sparsity Pattern**    | Irregular (scattered zeros)    | Regular (removed channels)   |
| **Hardware Efficiency** | Requires sparse tensor support | Works with standard hardware |
| **Compression**         | High parameter reduction       | Moderate reduction + speedup |
| **Implementation**      | Masks on weight matrices       | New smaller architecture     |

---

## Requirements

- PyTorch with CUDA support
- torchvision
- numpy
- matplotlib
- tqdm
- pytorch-grad-cam (for Grad-CAM visualizations)
- CIFAR-10 dataset (auto-downloaded)

## Notes

- GPU is highly recommended (training and evaluation are GPU-accelerated)
- First run will download CIFAR-10 (~170 MB)
- Sensitivity analysis is computationally expensive (cached after first run)
- Adjust sparsity ratios based on your accuracy/compression requirements
