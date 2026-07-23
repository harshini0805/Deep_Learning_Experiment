# Single Layer Perceptron for Binary Classification

Implementation of a **Single Layer Perceptron from scratch using NumPy** for binary classification of **authentic** and **forged** banknotes using the **UCI Banknote Authentication** dataset.

## Objective

- Implement a Single Layer Perceptron from scratch
- Understand the perceptron learning rule and step activation function
- Visualize the learning process
- Evaluate the model on a real-world dataset
- Compare results with Scikit-learn's implementation

## Repository Structure

```text
.
├── Deep_Learning_Lab01.ipynb
├── README.md
├── requirements.txt
├── data/
└── images/
```

## Dataset

- **Dataset:** UCI Banknote Authentication
- **Samples:** 1,372
- **Features:** Variance, Skewness, Curtosis, Entropy
- **Classes:** Authentic (0), Forged (1)

## Methodology

- Data exploration and EDA
- Train-test split (80:20) and feature scaling
- Perceptron implementation from scratch (NumPy)
- Model evaluation using Accuracy, Precision, Recall and F1-score
- Comparison with Scikit-learn Perceptron

## Results

| Metric | Value |
|--------|------:|
| Accuracy | **98.55%** |
| Precision | 96.83% |
| Recall | **100.00%** |
| F1-score | **98.39%** |

### Key Findings

- Scratch implementation achieved **98.55%** accuracy.
- Performance matched Scikit-learn's Perceptron.
- Recall of **100%** indicates all forged banknotes were correctly identified.
- **Variance** and **Skewness** were the most influential features.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Deep_Learning_Lab01.ipynb
```

## References

- Rosenblatt, F. (1958). *The Perceptron*
- Goodfellow et al. (2016). *Deep Learning*
- UCI Machine Learning Repository – Banknote Authentication Dataset
