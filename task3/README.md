# Task 3: Knowledge Distillation

## Overview

This task explores various **knowledge distillation** techniques to transfer knowledge from a larger teacher model (VGG-16/VGG-19) to a smaller student model (VGG-11) on CIFAR-100. The goal is to improve student performance by learning from teacher predictions rather than training independently. Additionally, **Grad-CAM visualizations** are used to compare attention patterns across different distillation methods.

---

## Repository Structure

- **`task 3.ipynb`**: Main notebook containing all distillation methods and experiments
- **`logit matching.ipynb`**: Standalone implementation of Logit Matching (LM) distillation technique
- **`gradcam.ipynb`**: Grad-CAM visualization tool for comparing teacher and student model interpretability

---

## Notebook Structure: task 3.ipynb

### **Cell 1: Label Smoothing (LS) & Decoupled KD (DKD)**

Trains student models using Label Smoothing and Decoupled Knowledge Distillation, comparing their performance against logit matching baseline.

### **Cell 2: Decoupled Knowledge Distillation (DKD) - Detailed**

Implements DKD with separate target class knowledge distillation (TCKD) and non-target class knowledge distillation (NCKD) components.

### **Cell 3: Memory Cleanup**

Clears GPU memory cache.

### **Cell 4: Independent Student (SI)**

Trains a VGG-11 student model independently without any teacher guidance to establish baseline performance.

### **Cell 5: Hint-based Distillation (Hints)**

Implements feature-level distillation where student learns intermediate representations from teacher's hidden layers using MSE loss and regression layers.

### **Cell 6: Contrastive Representation Distillation (CRD)**

Uses contrastive learning to align student and teacher feature representations through projection heads and InfoNCE-style loss.

### **Cell 7: KL Divergence Analysis**

Calculates average KL divergence between each student model's output distribution and the teacher's distribution to measure approximation quality.

### **Cell 8: Color Invariance Experiment**

Tests model robustness by:

- Fine-tuning teacher on color-jittered images
- Distilling to student using CRD
- Evaluating performance drop on color-augmented test set

### **Cell 9: VGG-19 Teacher Experiment**

Trains a larger VGG-19 teacher and distills to VGG-11 student to evaluate whether a more capable teacher improves student performance compared to VGG-16.

### **Cell 10: Dependency Management**

Handles NumPy version compatibility.

---

## Notebook Structure: logit matching.ipynb

### **Cell 1: Logit Matching (LM) Implementation**

Complete standalone implementation of logit matching distillation:

- Loads/trains VGG-16 teacher on CIFAR-100
- Trains VGG-11 student using KL divergence between soft predictions
- Combines hard loss (cross-entropy) and soft loss (KD) with temperature scaling
- Saves best performing student model

---

## Notebook Structure: gradcam.ipynb

### **Cell 1: Google Drive Setup**

Mounts Google Drive for accessing saved model files.

### **Cell 2: Grad-CAM Visualization Pipeline**

Implements complete Grad-CAM visualization comparing multiple distillation methods:

- Loads teacher (VGG-16) and various student models (SI, LM, Hints, CRD)
- Applies Grad-CAM to visualize attention patterns for each model
- Generates side-by-side comparison showing original CIFAR-100 image and heatmaps
- Color-codes predictions (green for correct, red for incorrect)
- Saves high-resolution comparison image

**Purpose**: Analyze whether different distillation methods affect student model interpretability and whether students learn similar attention patterns to the teacher.

---

## Distillation Methods Compared

| Method                       | Type           | Key Technique                                 |
| ---------------------------- | -------------- | --------------------------------------------- |
| **Independent Student (SI)** | Baseline       | No teacher guidance                           |
| **Logit Matching (LM)**      | Response-based | KL divergence on output logits                |
| **Label Smoothing (LS)**     | Regularization | Soft targets with uniform noise               |
| **Decoupled KD (DKD)**       | Response-based | Separate target/non-target class distillation |
| **Hint-based (Hints)**       | Feature-based  | Intermediate layer matching with regressor    |
| **Contrastive (CRD)**        | Feature-based  | Contrastive learning on representations       |

---

## Experimental Tasks

### **Task 1: Distillation Method Comparison**

Compare accuracy of different distillation approaches on CIFAR-100 test set.

### **Task 3: Distribution Approximation Quality**

Measure KL divergence between student and teacher output distributions to assess how well students approximate teacher behavior.

### **Task 5: Color Invariance**

Evaluate robustness to color augmentations by comparing performance on regular vs color-jittered test sets.

### **Task 6: Teacher Capacity Impact**

Investigate whether using a larger teacher (VGG-19) instead of VGG-16 leads to better student performance.

---

## Expected Outputs

### Models Saved

1. `best_teacher_vgg16_cifar100.pth` - VGG-16 teacher model
2. `best_teacher_vgg19_cifar100.pth` - VGG-19 teacher model
3. `best_teacher_color_invariant.pth` - Color-robust teacher
4. `best_student_si.pth` - Independent student
5. `best_student_lm.pth` - Logit matching student
6. `best_student_ls.pth` - Label smoothing student
7. `best_student_dkd.pth` - Decoupled KD student
8. `best_student_hints.pth` - Hint-based student
9. `best_student_crd.pth` - Contrastive distillation student
10. `best_student_crd_color_invariant.pth` - Color-robust CRD student
11. `best_student_lm_from_vgg19.pth` - Student distilled from VGG-19

### Analysis Outputs

- Accuracy comparison tables
- KL divergence rankings
- Color robustness performance drops
- Teacher capacity impact analysis
- **Grad-CAM comparison images** (`distillation_grad_cam_comparison.png`)

---

## Key Hyperparameters

- **Teacher**: VGG-16 or VGG-19
- **Student**: VGG-11
- **Dataset**: CIFAR-100 (100 classes)
- **Training Epochs**: 40-200 depending on method
- **Batch Size**: 128
- **Temperature**: 4.0 (for KD)
- **Alpha**: 0.5-1.0 (hard/soft loss balance)
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.001-0.05 with Cosine Annealing

---

## Key Findings

1. **Distillation Effectiveness**: All distillation methods outperform independent student training
2. **Method Comparison**: Different distillation approaches have varying effectiveness
3. **Distribution Quality**: Lower KL divergence indicates better teacher approximation
4. **Robustness**: CRD with color-invariant teacher improves robustness to augmentations
5. **Teacher Capacity**: Larger teachers (VGG-19) can improve student performance

---

## Requirements

- PyTorch with CUDA support
- torchvision
- numpy
- pandas
- tqdm
- pytorch-grad-cam (for Grad-CAM visualizations)
- CIFAR-100 dataset (auto-downloaded)

## Notes

- GPU highly recommended for training
- Teacher models should be trained/loaded before student distillation
- Color invariance experiments require additional training time
- Some experiments use cached models to avoid retraining
