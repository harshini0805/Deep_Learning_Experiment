# Dataset: Banknote Authentication

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

**File:** `data_banknote_authentication.txt` (CSV, no header row)

Data were extracted from 400x400-pixel grayscale images (660 dpi) of genuine and forged
banknote-like specimens. A Wavelet Transform was applied to the images, and statistical
features were computed from the wavelet coefficients.

## Summary

| Property | Value |
|---|---|
| Instances | 1372 |
| Features | 4 (numerical, continuous) |
| Classes | 2 |
| Missing values | None |
| Task | Binary classification |

## Columns (in file order)

| # | Name | Description |
|---|---|---|
| 1 | `variance` | Variance of the wavelet-transformed image |
| 2 | `skewness` | Skewness of the wavelet-transformed image |
| 3 | `curtosis` | Curtosis of the wavelet-transformed image |
| 4 | `entropy` | Entropy of the image |
| 5 | `class` | Target: `0` = authentic banknote, `1` = forged banknote |

## Class distribution

| Class | Count |
|---|---|
| 0 (Authentic) | 762 |
| 1 (Forged) | 610 |

## Citation

Lohweg, V. (2012). *Banknote Authentication* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C55P57
