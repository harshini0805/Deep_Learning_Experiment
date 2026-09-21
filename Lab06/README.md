# Lab 06 – RNN, LSTM and GRU for Sequence Learning

CS3807 Deep Learning Lab, Experiment 6. Covers vanilla RNN, LSTM and GRU on the UCI HAR sensor dataset, CNN–LSTM/GRU video classification (UCF101 subset), and encoder–decoder sequence-to-sequence learning.

## Contents

| Path | Description |
|---|---|
| `Experiment_6_RNN_LSTM_GRU_2.ipynb` | Main notebook (HAR, video and seq2seq experiments) |
| `Experiment_61.tex` / `.pdf` | Lab report (LaTeX source and compiled PDF) |
| `images/` | Figures used in the report (needed to compile the `.tex`) |
| `results/` | Metrics and comparison tables (CSV/JSON) |
| `additional_exercises/` | Code, raw runs and figures for the seven additional exercises |

## Data (not included)

- **UCI HAR Dataset:** download from the UCI Machine Learning Repository and place it in `data/UCI_HAR_Dataset/`.
- **UCF101 subset:** 5 classes (Basketball, Biking, WalkingWithDog, JumpingJack, TennisSwing), 8 videos each, in `data/`.

## Setup and run

```bash
pip install tensorflow numpy scikit-learn matplotlib pandas
jupyter notebook Experiment_6_RNN_LSTM_GRU_2.ipynb
```

If you re-run the video cells, delete `results/eval_video_*.json` first (or use `force=True`), otherwise cached predictions from an earlier run are reused.

## Building the report

Upload `Experiment_61.tex` and the `images/` folder to Overleaf (pdfLaTeX), or run `pdflatex Experiment_61.tex` twice.

## Notes

- `additional_exercises/code/` uses hard-coded workspace paths; edit the data paths before running.
- Exercise results are mean ± s.d. over 3 seeds (42, 43, 44).
