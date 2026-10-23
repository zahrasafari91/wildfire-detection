# Wildfire Detection Using Deep Learning

**Image Classification and Semantic Segmentation for UAV Imagery**

## Overview

This project investigates lightweight computer vision models for detecting and localizing fire in aerial images captured by unmanned aerial vehicles (UAVs). It combines two tasks:

- **Classification:** Predict whether an image contains fire.
- **Segmentation:** Identify fire regions at the pixel level.

The focus is on model accuracy and computational efficiency under limited GPU resources. The models were evaluated experimentally; real-time deployment on UAV hardware remains future work.

## Results

| Task | Metric | Result |
| --- | --- | ---: |
| Classification | Accuracy | **89%** |
| Classification | Reference accuracy | 85.12% |
| Segmentation | Macro accuracy | **99.83%** |
| Segmentation | Mean F1 | 86.29% |
| Segmentation | Mean IoU | 75.96% |
| Segmentation | Mean recall | 88.94% |

Segmentation macro accuracy was close to the 99.90% reference result, although the reference model had a higher F1 score. The project presentation reports a model approximately one-sixth the size of the reference approach.

## Dataset

We used the **FLAME dataset** of UAV aerial imagery, including fire/no-fire images and segmentation masks.

Key challenges included class imbalance, visually similar non-fire regions (such as sunlight and reflections), limited data, and the need to downscale 4K images to **512 × 512** for GPU compatibility.

Dataset reference: [FLAME — IEEE DataPort](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-usingdrones-uavs)

## Methodology

### 1. Preprocessing and augmentation

- Applied **red highlighting** to emphasize fire-related image regions.
- Used **Sobel edge detection** to enhance boundary information.
- Applied rotations, flips, and random crops for data augmentation.
- Used weighted sampling to address classification class imbalance.
- Resized high-resolution images to fit available GPU memory.

### 2. Fire classification

Built an ensemble combining **MobileNetV3 Large** and **EfficientNet-B0**. Features from both backbones were combined and passed through a classification head with global average pooling, dropout, and a sigmoid output for binary fire/no-fire predictions.

**Training:** PyTorch, AdamW, weight decay, dropout (0.5), and early stopping; batch size 50, up to 100 epochs.

| Classification metric | Fire | No fire |
| --- | ---: | ---: |
| Precision | 95% | 82% |
| Recall | 86% | 93% |
| F1 score | 90% | 87% |

Overall classification accuracy: **89%**, compared with **85.12%** for the reference model.

### 3. Fire segmentation

Used **U-Net++** for pixel-level fire localization. Evaluated multiple encoder options, including MobileNetV3, ResNet-34/101, EfficientNet-B0/B6, and **RegNet** (selected encoder). The decoder produces a probability map, which is converted into a binary fire mask using a **0.6 threshold**.

**Training:** PyTorch, Adam, learning-rate scheduling; batch size 4, up to 50 epochs.

| Segmentation metric | Result |
| --- | ---: |
| Macro accuracy | **99.83%** |
| Mean F1 | 86.29% |
| Mean IoU | 75.96% |
| Mean recall | 88.94% |

## Tools and Hardware

- **Language / framework:** Python, PyTorch, CUDA
- **Computer vision:** CNN classification, semantic segmentation, Sobel edge detection, image augmentation
- **Architectures:** MobileNetV3, EfficientNet, U-Net++, ResNet, RegNet
- **Training techniques:** Adam, AdamW, dropout, early stopping, learning-rate scheduling
- **Hardware:** NVIDIA RTX 3080 Ti, AMD Ryzen 9 5900X, 32 GB RAM

## Limitations and Future Work

- Improve generalization with more diverse training data.
- Further investigate overfitting and class imbalance.
- Evaluate transformer-based segmentation alternatives.
- Benchmark real-time inference and deployment on actual UAV hardware.
- Explore environmental context such as wind and vegetation.

## Authors

**Zahra Safari, Mohsen Hosseini, Maryam Kermanshahani**  
University of Calgary

## References

- Shamsoshoara et al. (2021), *Aerial Imagery Pile Burn Detection Using Deep Learning: The FLAME Dataset*, Computer Networks.
- Ghali, Akhloufi, and Mseddi (2022), *Deep Learning and Transformer Approaches for UAV-Based Wildfire Detection and Segmentation*, Sensors.
