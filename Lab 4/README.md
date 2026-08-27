# Experiment 4: CNN Comparative Study with Transfer Learning

**CS3807 Deep Learning Laboratory** | Shiv Nadar University Chennai

**GitHub:** [Deep_Learning_Experiment/Lab4](https://github.com/Deep_Learning_Experiment/tree/main/Lab4)

## Quick Start

```bash
git clone https://github.com/Deep_Learning_Experiment.git
cd Deep_Learning_Experiment/Lab4

pip install -r requirements.txt
jupyter notebook Experiment_4_Final.ipynb
```

## What's in Here

- **Experiment_4_Final.ipynb** — Main notebook (all code)
- **experiment4_report_with_plots.pdf** — Lab report with 8 plots
- **requirements.txt** — Dependencies

## Results

| Model | Accuracy | Time (min) |
|-------|----------|-----------|
| LeNet-5 | 58.42% | 2.7 |
| AlexNet | 72.63% | 139 |
| GoogleNet | 74.09% | 46 |
| VGG16 | 88.25% | 761 |
| **ResNet50** | **89.63%** | **420** |
| InceptionV3 | 87.94% | 386 |

**Key Finding:** Transfer learning achieves 30% higher accuracy than training from scratch.

## What You Get

CNN architectures compared  
Transfer learning (frozen base + fine-tuning)  
Hyperparameter sensitivity study  
Confusion matrices & training curves  
Lab report with discussion answers  

## Running It

Notebook runs end-to-end in ~60–90 minutes (GPU recommended).

Already trained a model? Restart and it skips completed ones (checkpointed to CSV).

## Plots Generated

- Training curves (accuracy + loss)
- Confusion matrices
- Accuracy comparison bar chart
- Training time comparison
- Hyperparameter sensitivity (3 parameters)
- Model capacity vs accuracy

All saved as `.eps` files in `images/`
