# Experiment 5 — CNN Training, Regularization, Optimization, Tuning, Transfer Learning & Cross-Validation

**Model:** MobileNetV2 (ImageNet-pretrained) · **Dataset:** Oxford-IIIT Pet (37 breeds, 224×224×3) · **Framework:** TensorFlow / Keras 3

A controlled, section-by-section study of the design choices around training a CNN classifier.
A frozen MobileNetV2 backbone is used as a feature extractor, and a small classifier head
(`Dense(128, ReLU) → [BN] → [Dropout] → Dense(37, softmax)`) is trained on top. Each study fixes
the winner of the previous one and changes one factor at a time:

```
Weight init  →  Regularization  →  Optimizer  →  Hyperparameters  →  Fine-tuning  →  5-fold CV  →  Final test
```

The notebook follows the lab manual layout (Sections 4–13, 16; Plots 1–15).

---

## Repository contents

| File | Purpose |
|---|---|
| `Experiment5_MobileNetV2_Pets.ipynb` | The full experiment, run top to bottom |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `plot1_*.png` … `plot15_*.png` | Figures written by the notebook into the working directory |

---

## Setup

**Python:** 3.10–3.12 (developed on 3.12.5).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook Experiment5_MobileNetV2_Pets.ipynb
```

### GPU notes
- **Linux / WSL2 + NVIDIA GPU:** swap `tensorflow` for `tensorflow[and-cuda]` in `requirements.txt` (the line is there, commented).
- **Native Windows:** TensorFlow ≥ 2.11 has no native Windows GPU support, so training runs on CPU
  (≈ 0.45–0.6 s/step at batch 32, ≈ 45–60 s/epoch for the head-only runs). Use WSL2 or Google Colab for GPU.
- **Google Colab:** all dependencies are preinstalled, so `pip install` can be skipped; the default
  data path (`/content/oxford_pets`) already follows Colab's convention.

### Hardware requirements
- **Disk:** ~800 MB for the dataset tarballs + extracted images.
- **RAM:** Section 11 loads the whole train+val pool into memory as `float32`
  (3680 × 224 × 224 × 3 × 4 B ≈ **2.2 GB**) and makes preprocessed per-fold copies on top of that.
  Plan for **≥ 8 GB** free RAM.
- **Runtime (CPU):** several hours end to end; Section 11 (4 configs × 5 folds) is the longest block.

---

## Dataset

Downloaded automatically in Section 3 from the official VGG site (no `tensorflow_datasets`,
which avoids its `protobuf` version conflicts):

- `https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz`
- `https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz`

| Split | Images | Source |
|---|---|---|
| Train | 2944 | `trainval.txt`, 80 % after a seeded shuffle |
| Validation | 736 | `trainval.txt`, remaining 20 % |
| Test | 3669 | `test.txt`, touched only in Section 12 |

To store the data elsewhere, change `DATA_DIR` in the Section 3 cell. (On Windows, the default
`/content/oxford_pets` resolves to `\content\oxford_pets` on the current drive root.)

### Preprocessing — the one thing not to change
`mobilenet_v2.preprocess_input` expects raw pixels in **[0, 255]** and maps them to **[−1, 1]**:

$$
x' = \frac{x}{127.5} - 1
$$

Images are therefore kept as raw `float32` in [0, 255] and **never divided by 255 first**; doing so
double-normalizes the input into roughly [−1, −0.99] and the model collapses.

---

## Notebook walkthrough

| Section | Study | What varies | Plots |
|---|---|---|---|
| 3 | Data loading | — | — |
| 4 | Architecture | Inspect MobileNetV2 (2.26 M params, all frozen) | — |
| 5 | Weight initialization | Zero, Random Normal, Xavier/Glorot, He | 1, 2 |
| 6 | Regularization | None, L2 (λ = 10⁻³), Dropout (0.5), BatchNorm | 3, 4 |
| 7 | Batch normalization | Worked numerical example + with/without BN | 5 |
| 8 | Optimizers | SGD, Momentum (0.9), RMSProp, Adam | 6, 7 |
| 9 | Hyperparameters | LR ∈ {10⁻³, 10⁻⁴}, batch ∈ {16, 32, 64}, dropout ∈ {0, 0.25, 0.5} | 8, 9, 10 |
| 10 | Transfer learning | Case A: frozen base · Case B: unfreeze last 30 layers, LR = 10⁻⁵ | 11, 12 |
| 11 | 5-fold stratified CV | 4 candidate configs on the train+val pool | 13 |
| 12 | Final evaluation | Retrain best config on train+val, evaluate once on test | 14, 15 |
| 13 | Summary table | Collects the numbers above | — |
| 16 | Additional exercise | Scaffold for two extra configs | — |

### Batch-norm worked example (Section 7)
For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$:

$$
\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad
\sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2, \qquad
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad
y_i = \gamma\,\hat{x}_i + \beta
$$

With $x = [2, 4, 6, 8]$: $\mu_\mathcal{B} = 5$, $\sigma_\mathcal{B}^2 = 5$, $\sigma_\mathcal{B} \approx 2.236$,
and $\hat{x} \approx [-1.342,\ -0.447,\ 0.447,\ 1.342]$ (with $\gamma = 1,\ \beta = 0$, $y = \hat{x}$).

---

## Results (recorded run)

Validation numbers are the best epoch on the fixed 736-image validation split.

**Section 5 — initialization** → best: **Xavier** (He was close behind at 90.2 % final-epoch val acc).

**Section 6 — regularization** → best: **BatchNorm** (91.2 % best val acc).

**Section 8 — optimizers** (Xavier + BN head, 6 epochs)

| Optimizer | Final train loss | Best val acc (%) | Epoch of best | Time (s) |
|---|---|---|---|---|
| SGD (lr 10⁻²) | 0.2650 | 88.72 | 6 | 303.7 |
| Momentum | 0.0344 | 90.76 | 5 | 343.9 |
| RMSProp | 0.0109 | 90.22 | 5 | 334.0 |
| **Adam** | 0.0195 | **91.44** | 6 | 311.4 |

**Section 9 — hyperparameters** → LR 10⁻³ (90.2 %) ≫ 10⁻⁴ (82.5 %); batch size and dropout
differences are all within ~1 pp (≈ 7 images on the validation set), i.e. within noise.

**Section 10 — transfer learning:** head-only reached 88.9 % val; fine-tuning the last 30 layers
at LR 10⁻⁵ reached 89.5 % — a small gain, as expected for a dataset this close to ImageNet.

**Section 11 — 5-fold CV** (as recorded; see *Known issues* before interpreting)

| Config | F1 | F2 | F3 | F4 | F5 | Mean | SD |
|---|---|---|---|---|---|---|---|
| C1_Baseline | 50.27 | 40.76 | 49.59 | 56.66 | 61.28 | 51.71 | 7.79 |
| C2_BestReg | 95.52 | 99.73 | 99.18 | 98.23 | 83.83 | 95.30 | 6.61 |
| C3_HighDropout | 98.91 | 96.47 | 97.83 | 95.79 | 97.42 | 97.28 | 1.21 |
| **C4_L2_BN** | 94.57 | 99.05 | 99.86 | 100.00 | 97.96 | **98.29** | 2.23 |

**Section 12 — final test set (C4_L2_BN)**

| Metric | Value |
|---|---|
| Test accuracy | 77.49 % |
| Macro precision / recall / F1 | 0.838 / 0.775 / 0.783 |
| Training time | 678 s (10 epochs, CPU) |
| Parameters | 2,427,237 |
| Most confused pair | Basset Hound → Beagle (30 images) |

---

## Known issues (read before citing the CV or test numbers)

1. **Shared, partially unfrozen backbone leaks across Sections 10 → 11 → 12.**
   Every call to `build_head(base_model, …)` reuses the *same* `base_model` object. Section 10 sets
   `base_model.trainable = True` for the last 30 layers and never re-freezes it. As a result, in
   Section 11 each fold **keeps fine-tuning that shared backbone at LR 10⁻³**, so:
   - fold *k*'s validation images were training images for the backbone in folds 1…*k*−1 and in
     every earlier config (and ~80 % of them were already seen in Section 10);
   - the first config (C1) absorbs the damage of high-LR fine-tuning (≈ 52 %), and later configs
     score 95–100 % because the backbone has memorized the pool.

   The 98.3 % CV mean is therefore **not** a valid generalization estimate. The signature is visible
   in Section 12: training accuracy is 99.5 % from epoch 1, yet test accuracy is 77.5 % — well
   *below* the ~89–91 % the frozen-head models reached on held-out validation data in Sections 6–10.

   **Fix:** build a fresh backbone per model, e.g.
   ```python
   def make_base(trainable_top=0):
       base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
                          weights='imagenet', pooling='avg')
       base.trainable = trainable_top > 0
       for layer in base.layers[:len(base.layers) - trainable_top]:
           layer.trainable = False
       return base
   ```
   and call `build_head(make_base(), …)` in Sections 5–9, 11 and 12 (and `make_base(30)` for
   Case B in Section 10). At minimum, add `base_model.trainable = False` at the top of Section 11.
   Also pass `training=False` when calling the backbone so its BatchNorm statistics stay frozen
   during fine-tuning.

2. **The "CV Accuracy" column in Section 13** for the Baseline / Initialization / Regularization /
   Optimizer / Hyperparameter rows is a single hold-out validation score (max over epochs), not a
   cross-validated one. Taking the max over epochs is also mildly optimistic.

3. **Section 9 differences are within noise.** With 736 validation images, 1 pp ≈ 7 images; a single
   seed cannot separate batch sizes 16/32/64 or dropout 0.25/0.5. Repeat over 3+ seeds if a choice
   matters.

4. `tar.extractall` raises a `DeprecationWarning` on Python 3.12+; pass `filter='data'` to silence it
   and to get safe extraction on Python 3.14.

---

## Reproducibility

- Seeds: `tf.random.set_seed(42)`, `np.random.seed(42)`, split shuffle `RandomState(42)`,
  `StratifiedKFold(random_state=42)`.
- GPU kernels (cuDNN) are not bit-deterministic by default; add
  `tf.config.experimental.enable_op_determinism()` for exact reruns (at some speed cost).
- Exact numbers will differ slightly across TensorFlow versions and CPU vs GPU.

## References

- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
- Parkhi et al., *Cats and Dogs*, CVPR 2012 (Oxford-IIIT Pet dataset).
- Ioffe & Szegedy, *Batch Normalization*, ICML 2015.
- Glorot & Bengio, 2010; He et al., 2015 (initialization).
- Kingma & Ba, *Adam*, ICLR 2015.
