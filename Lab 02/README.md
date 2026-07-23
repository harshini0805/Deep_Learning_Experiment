# Deep Learning Lab 02: Multi-Layer Perceptron (MLP)

Implementation of a **Multi-Layer Perceptron (MLP)** using **TensorFlow/Keras** for multi-class image classification on the **Fashion-MNIST** dataset, including **hyperparameter optimization** using **RandomizedSearchCV**.

## Features

- MLP implementation with TensorFlow/Keras
- Fashion-MNIST preprocessing and normalization
- Baseline model training and evaluation
- Hyperparameter tuning using RandomizedSearchCV
- Performance comparison and visualization

## Dataset

- **Dataset:** Fashion-MNIST
- **Training Samples:** 60,000
- **Testing Samples:** 10,000
- **Classes:** 10 clothing categories
- **Image Size:** 28 × 28 grayscale

## Results

| Model | Test Accuracy |
|--------|--------------:|
| Baseline MLP | **87.81%** |
| Optimized MLP | **87.12%** |

**Key Findings**
- Baseline model slightly outperformed the optimized model on the test set.
- Hyperparameter search improved validation stability but not test accuracy.
- Most misclassifications occurred between visually similar clothing categories.

## Repository

```text
.
├── Deep_Learning_Lab02.ipynb
├── README.md
├── requirements.txt
├── data/
└── images/
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook Deep_Learning_Lab02.ipynb
```

## References

- Goodfellow et al., *Deep Learning*, MIT Press
- Fashion-MNIST Dataset
- TensorFlow/Keras Documentation
- Scikit-learn Documentation
