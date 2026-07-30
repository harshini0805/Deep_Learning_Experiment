# CS3807 — Experiment 3: CNN Image Classification on CIFAR-10

Implementation of a Convolutional Neural Network for image classification on CIFAR-10, covering the convolution operation, output-size calculation under stride/padding, feature-map visualization, max/average pooling, CNN parameter counting, and full training + evaluation.

**Framework:** PyTorch (the lab manual specifies TensorFlow/Keras; this implementation uses PyTorch instead — the architecture, hyperparameters, and evaluation protocol are functionally equivalent).

## Contents

| File | Description |
|---|---|
| `Experiment_3_CNN_CIFAR10.ipynb` | Full notebook — all 7 tasks, 8 mandatory plots, 5 additional exercises, discussion answers |
| `Experiment_3_Report.tex` / `.pdf` | Typeset lab report (Overleaf-ready) |
| `images/` | Exported plots referenced by the report |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is optional but strongly recommended — CIFAR-10 (50,000 training images, 20 epochs) trains in well under 2 minutes on GPU and considerably longer on CPU.

## Running

```bash
jupyter notebook Experiment_3_CNN_CIFAR10.ipynb
```

Run all cells top to bottom. The data-loading cell downloads CIFAR-10 automatically via `torchvision.datasets.CIFAR10` on first run (~170 MB) and caches it locally in `./data`. If the download is unreachable (e.g. a network-restricted sandbox), the notebook falls back to a small synthetic placeholder dataset of identical shape so the rest of the pipeline can still be verified — this is clearly flagged wherever it happens and should not occur on a normal machine or Google Colab.

## Architecture

```
Input (32×32×3)
 → Conv(16, 3×3, pad 1) → ReLU → MaxPool(2×2)
 → Conv(32, 3×3, pad 1) → ReLU → MaxPool(2×2)
 → Flatten → Dense(128) → ReLU → Dense(10, logits)
```

Trained with Adam, batch size 32, 20 epochs, `CrossEntropyLoss` (which applies softmax internally — no explicit `Softmax` layer is used, to avoid double-applying it).

## Results (real CIFAR-10 run, GPU)

| Metric | Value |
|---|---|
| Trainable parameters | 268,650 |
| Final training accuracy | 88.46% |
| Test accuracy | 67.39% |
| Test precision (macro) | 67.55% |
| Test recall (macro) | 67.39% |
| Test F1-score (macro) | 67.30% |

Weakest class: **cat** (F1 = 48%), largely confused with dog/bird. Strongest classes: **automobile** and **ship** (F1 = 78%). Validation loss bottoms out around epoch 8 and rises thereafter while training loss keeps falling — the network overfits past that point; see the report's Discussion section for mitigation suggestions (early stopping, dropout, data augmentation, weight decay).

## References

1. Goodfellow, Bengio, Courville — *Deep Learning*, MIT Press.
2. Bishop — *Pattern Recognition and Machine Learning*, Springer.
3. Haykin — *Neural Networks and Learning Machines*, Pearson.
4. [PyTorch Documentation](https://pytorch.org/docs)
5. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
