# Comparative Analysis of Deep Learning Optimizers
## Executive Summary
This project explores the impact of optimization algorithms on the convergence speed and accuracy of Convolutional Neural Networks (CNNs). Using the Fashion MNIST dataset, I benchmarked four industry-standard optimizers—Adam, SGD, RMSprop, and Adagrad—to determine the most robust configuration for image classification tasks.

Key Result: The Adam optimizer proved to be the most robust choice for this architecture, achieving ~84.8% accuracy within 10 epochs, while other optimizers required significantly more aggressive hyperparameter tuning to escape local minima.

## Project Objectives
Architecture Design: Construct a scalable CNN capable of feature extraction from grayscale telemetry.

Performance Benchmarking: Evaluate loss landscapes and convergence rates across different optimization strategies.

Hyperparameter Tuning: Analyze the sensitivity of the model to Learning Rate (LR) adjustments to maximize performance.

## Methodology
### Data Pipeline & Preprocessing

Source: Fashion MNIST (via Kaggle/Zalando Research).

Structure: 60,000 Training images / 10,000 Test images (28×28 pixels).

Normalization: Pixel intensity scaled to [0,1] to prevent vanishing/exploding gradients.

Reshaping: Input tensor reshaped to (28,28,1) to satisfy Conv2D dimensional requirements.

Validation Split: Implemented a hold-out validation set (10k images) to monitor overfitting during training.

### Model Architecture (CNN)

I implemented a Sequential model focused on spatial hierarchy extraction:

Input Layer: Conv2D (32 filters, ReLU)

Feature Extraction: Stacked Convolutional layers with Max Pooling to reduce dimensionality and Dropout (0.25) to prevent overfitting.

Classification Head: Dense layer (64 units) -> Softmax Output (10 classes).

### Experimental Design

I conducted two distinct phases of experimentation:

Phase A (Optimizer Sweep): Fixed learning rate (0.001) and epochs (10) to isolate the optimizer's effect.

Phase B (Sensitivity Analysis): Tuning the Learning Rate for the top-performing optimizer (Adam).

## Results & Analysis
Phase A: Optimizer Performance (Fixed LR = 0.001)

| Optimizer | Test Accuracy | Loss | Insight |
| :--- | :--- | :--- | :--- |
| **Adam** | **84.76%** | **0.42** | **Optimal Convergence.** Effectively adapted individual learning rates per parameter. |
| **SGD** | 14.11% | 2.30 | Failed to converge. Likely requires a higher LR or momentum tuning. |
| **RMSprop** | 10.00% | 2.30 | Stuck at random initialization baseline. |
| **Adagrad** | 10.00% | 2.30 | Stuck at random initialization baseline. |

Phase B: Learning Rate Tuning (Adam)

Learning Rate	Accuracy	Observation
0.001	82.54%	Stable. Good balance between speed and stability.
0.01	10.00%	Diverged. Steps were too large, overshooting the global minimum.
0.1	10.00%	Diverged. Immediate gradient explosion/instability.

## Technologies Used
Deep Learning: TensorFlow, Keras

Data Manipulation: NumPy

Visualization: Matplotlib, Seaborn

Environment: Jupyter / Kaggle Kernels

## Future Improvements
To further improve model performance, the following steps are recommended:

Learning Rate Schedulers: Implement ReduceLROnPlateau to dynamically adjust LR during training.

Data Augmentation: Introduce rotation and zooming to improve generalization.

Batch Normalization: Add BN layers to stabilize learning for the SGD and RMSprop optimizers.
