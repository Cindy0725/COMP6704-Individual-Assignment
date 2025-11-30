# COMP6704-Individual-Assignment: Adversarial Attack Analysis on ResNet18 and ViT-Tiny

This repository contains the implementation and evaluation of four adversarial attack algorithms (**FGSM**, **PGD**, **MI-FGSM**, **DI-FGSM**) on two different model architectures: **ResNet18** (CNN) and **ViT-Tiny** (Vision Transformer).

The project evaluates these models on the **CIFAR-10** dataset, focusing on:
1.  Computational Efficiency
2.  Convergence Rate
3.  Transferability
4.  Visual Comparison
5.  Attack Success Rate (ASR)

All attack algorithms are implemented from scratch using PyTorch. If you don't want to train the ResNet and ViT by yourself, please download the pre-trained checkpoints from here: https://drive.google.com/drive/folders/1rOL0v-_HKchMkfzWjiYqwPFBAaDKcmgE?usp=drive_link. 

## 📋 Requirements

To run this code, you need Python installed along with the following libraries:

*   `torch`
*   `torchvision`
*   `timm`
*   `pandas`
*   `matplotlib`
*   `numpy`

You can install the dependencies using pip:

```bash
pip install torch torchvision timm pandas matplotlib numpy

