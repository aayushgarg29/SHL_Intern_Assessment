# Grammar Scoring from Spoken English 🎙️

**SHL Hiring Assessment 2026 | Speech processing · NLP · Machine learning**

Predict human-rated grammar scores (0–5) from 45–60-second English speech recordings. This repository contains the complete Kaggle notebook used to build and evaluate a multi-model speech and text ensemble.

> **Data access:** Competition audio and CSV files are not included. You must have access to the SHL Hiring Assessment 2026 dataset on Kaggle. This project is an independent assessment submission, not an official SHL product.

## Overview

| Item | Details |
| --- | --- |
| Training clips | 769 (732 non-zero, 37 score-zero) |
| Test clips | 216 |
| Evaluation | RMSE; Pearson correlation also reported in notebook |
| Environment | Kaggle Notebook, CUDA GPUs recommended |
| Main deliverable | `submission.csv` |

## Pipeline

```text
Audio clips
  ├── Whisper large-v3 transcription + word timestamps
  │     ├── Fluency / disfluency features
  │     └── Qwen2.5-1.5B surprisal + hidden-state embeddings
  └── WavLM / HuBERT / Whisper encoder embeddings
        └── Selected frame-level attention-pooling heads
                   ↓
        Speaker-aware cross-validation
                   ↓
          Ridge / kernel ridge models
                   ↓
        Non-negative / convex blending
                   ↓
               submission.csv
```

## Key design choices

- **Speaker-aware evaluation:** Speaker groups are used to reduce leakage between training and validation folds.
- **Complementary features:** Audio representations capture acoustic cues; transcripts and Qwen capture linguistic patterns.
- **Cached intermediates:** Transcriptions and embeddings can be reused between notebook runs.
- **No paid inference API:** The notebook uses locally executed pretrained models.

## Reported results

The notebook's own report states **0.5230 out-of-fold RMSE on the 732 non-zero training clips**. This is a validation result, **not a Kaggle public/private leaderboard score**. The report also describes weaker performance on the 37 score-zero clips, which were excluded from model fitting. See the notebook's final **Report** section for full results, limitations, and diagnostic plots.

## Run on Kaggle

1. Obtain authorized access to the **SHL Hiring Assessment 2026** competition dataset.
2. Create a Kaggle notebook, import [`shl-intern-assessment.ipynb`](shl-intern-assessment.ipynb), and attach the competition dataset.
3. Enable a CUDA GPU accelerator (the notebook's Whisper transcription uses two CUDA device indices; adjust this if using a different GPU setup). Enable internet or provide cached pretrained model weights where required.
4. Confirm the dataset exists at:

   ```text
   /kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/
   ├── train.csv
   ├── test.csv
   ├── sample_submission.csv
   ├── train/...
   └── test/...
   ```

5. Run notebook cells **in order**. Some dependencies are imported directly, and `faster-whisper` may be installed by the notebook when absent. Use `requirements.txt` as a dependency reference; Kaggle's preinstalled packages and GPU-compatible PyTorch builds may differ.
6. The notebook writes predictions to `/kaggle/working/submission.csv`.

**Important:** The notebook reports 216 test rows versus 204 sample-submission rows. Check the competition's current submission requirements before uploading. The notebook validates column names and test-row order but does not automatically resolve that row-count discrepancy.

## Repository structure

```text
shl-grammar-scoring-2026/
├── shl-intern-assessment.ipynb   # Complete 13-stage modeling workflow and report
├── README.md                    # Project overview and setup
├── requirements.txt             # Main Python dependencies
├── .gitignore                   # Excludes data, caches, models, submissions
└── LICENSE                      # MIT license
```

## Limitations

- Training has relatively few unique speakers, so generalization remains challenging.
- Score-zero clips are not used for model training; this affects full-dataset performance.
- Large pretrained models require substantial GPU memory and download time.
- The final notebook includes an additional Kaggle-specific submission export cell.

## Responsible data use

Do not commit the competition dataset, audio recordings, cached embeddings, model checkpoints, API credentials, or private leaderboard materials. Follow the competition's rules on publishing code and derived artifacts.
