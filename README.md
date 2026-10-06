# SHL Hiring Assessment 2026 — Grammar Scoring Engine
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Whisper ASR](https://img.shields.io/badge/OpenAI-Whisper-black.svg)](https://github.com/openai/whisper)

A competition-grade, multimodal machine learning pipeline designed to automatically evaluate spoken English responses (45–60 seconds in duration) and predict a continuous **Grammar Score between 0.0 and 5.0**.

---

## 1. Project Objective & Challenge Overview

* **Task**: Automated continuous grammar evaluation from spoken audio.
* **Target Variable**: Grammar Score ($0.0$ to $5.0$).
* **Evaluation Criteria**: **Pearson Correlation** (primary ranking) & **Root Mean Squared Error (RMSE)**.
* **Dataset Size**:
  * **Train Set**: 769 recordings (`.wav`), 16 kHz Mono 16-bit PCM.
  * **Test Set**: 216 recordings (`.wav`), 16 kHz Mono 16-bit PCM.
* **Grammar Rubric**:
  * **1 — Very weak**: Struggles with basic sentence structure and syntax.
  * **2 — Weak**: Frequent basic grammatical mistakes and incomplete sentences.
  * **3 — Moderate**: Decent grasp of structure; minor syntactic errors remain.
  * **4 — Strong**: Good control of grammar and syntax with occasional minor slips.
  * **5 — Excellent**: High grammatical accuracy, complex structures, and self-correction.

---

## 2. Solution Architecture & Hypothesis

> **Core Hypothesis**: *Grammar quality is primarily encoded within linguistic syntax, clause structure, and lexical variety of the verbatim response. However, acoustic fluency proxies (silent pauses, pitch variability, and speech cadence) provide critical complementary signals to penalize disfluencies, hesitations, and false starts.*

```
                 ┌────────────────────────────────────────────────────────┐
                 │                 Input Audio (.wav)                     │
                 └──────────────────────────┬─────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
        ┌───────────────────────────┐                ┌───────────────────────────┐
        │  Automatic Speech Recog.  │                │   Digital Signal Proc.    │
        │   (OpenAI Whisper ASR)    │                │ (Acoustic / Prosodic Eng) │
        └────────────┬──────────────┘                └─────────────┬─────────────┘
                     │                                             │
          ┌──────────┴──────────┐                                  │
          ▼                     ▼                                  ▼
┌──────────────────┐  ┌──────────────────┐               ┌──────────────────┐
│ Linguistic / POS │  │ Sentence-Level   │               │ 13 MFCCs + Deltas│
│   Grammar NLP    │  │  Transformer     │               │ F0 Pitch Track   │
│ (TTR, WPS, etc.) │  │  Embeddings      │               │ RMS Energy, ZCR  │
└─────────┬────────┘  └────────┬─────────┘               └─────────┬────────┘
          │                    │                                   │
          └────────────────────┼───────────────────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │    Multimodal Feature Space   │
               │         (534 Features)        │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │   5-Fold Cross-Validation     │
               │   (Leakage-Free Pipelines)    │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │      OOF-Optimized Ensemble   │
               │  (Ridge + ElasticNet + HGB)   │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │   Predicted Grammar Score     │
               │          [0.0 - 5.0]          │
               └───────────────────────────────┘
```

---

## 3. Directory Structure

```text
shlproject/
│
├── notebook/
│   └── SHL_Grammar_Scoring_Final.ipynb    # Clean, executed 26-section competition notebook
│
├── Dataset_Final/
│   ├── train/                             # 769 training audio files (.wav)
│   ├── test/                              # 216 test audio files (.wav)
│   ├── train.csv                          # Ground truth labels
│   ├── test.csv                           # Test metadata
│   └── sample_submission.csv              # Official submission template
│
├── artifacts/
│   ├── transcripts/                       # Cached verbatim Whisper ASR transcripts
│   ├── audio_features/                    # Cached MFCCs, F0, RMS energy features
│   ├── linguistic_features/               # Cached syntactic & POS features
│   ├── text_embeddings/                   # Pretrained SentenceTransformer embeddings (384-d)
│   ├── figures/                           # Visualizations (EDA, residuals, scatter plots)
│   ├── predictions/                       # OOF and error analysis data
│   └── experiment_results.csv             # Full benchmark comparison table
│
├── src/
│   ├── extract_audio_features.py          # Acoustic signal processing engine
│   ├── transcribe.py                      # Local Whisper ASR transcription engine
│   ├── extract_linguistic_features.py     # POS, syntactic complexity & grammar feature extractor
│   ├── extract_text_embeddings.py         # Semantic sentence embedding extractor
│   ├── train_models.py                    # Cross-validation & ensembling routines
│   ├── run_pipeline.py                    # Complete end-to-end execution pipeline
│   └── build_notebook.py                  # Automated notebook generator and runner
│
├── submission.csv                         # Final competition submission (216 rows)
├── requirements.txt                       # Locked dependencies
├── .gitignore                             # Git rules (excludes heavy wav binaries)
└── README.md                              # Complete solution documentation
```

---

## 4. Key Experiments & Results Table

All models were evaluated using **5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True, random_state=42)`)** with strict preprocessing encapsulation inside fold pipelines to eliminate data leakage.

| Experiment ID | Feature Modality | Model Architecture | CV RMSE (Mean ± Std) | CV Pearson (Mean ± Std) | Training RMSE | Notes / Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **EXP_00** | None | Dummy Mean Predictor | $1.2382 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $1.2382$ | Trivial constant baseline |
| **EXP_01** | Audio Only | Ridge Regressor ($\alpha=100$) | $0.8815 \pm 0.0583$ | $0.7002 \pm 0.0675$ | $0.7738$ | 119 acoustic features |
| **EXP_02** | Linguistic Only | Ridge Regressor ($\alpha=10$) | $0.9624 \pm 0.0379$ | $0.6249 \pm 0.0398$ | $0.9156$ | 31 NLP/grammar features |
| **EXP_03** | Linguistic Only | HistGradientBoosting | $0.9796 \pm 0.0552$ | $0.6113 \pm 0.0627$ | $0.2777$ | Tree-based NLP |
| **EXP_04** | Text Embeddings | Ridge Regressor ($\alpha=50$) | $0.9316 \pm 0.0701$ | $0.6736 \pm 0.0500$ | $0.5395$ | 384-d MiniLM embeddings |
| **EXP_05** | Text Embeddings | HistGradientBoosting | $0.9364 \pm 0.0740$ | $0.6554 \pm 0.0456$ | $0.0853$ | MiniLM with trees |
| **EXP_06** | Ling + Audio | Ridge Regressor ($\alpha=100$) | $0.8185 \pm 0.0446$ | $0.7485 \pm 0.0534$ | $0.6826$ | Acoustic + Syntactic fusion |
| **EXP_07** | Multimodal (All) | Ridge Regressor ($\alpha=150$) | $0.7565 \pm 0.0296$ | $0.7940 \pm 0.0271$ | $0.4539$ | 534 multimodal features |
| **EXP_08** | Multimodal (All) | ElasticNet ($\alpha=0.1$) | $0.7508 \pm 0.0233$ | $0.7961 \pm 0.0311$ | $0.6209$ | Regularized sparse weights |
| **EXP_09** | Multimodal (All) | HistGradientBoosting | $0.7447 \pm 0.0404$ | $0.7946 \pm 0.0440$ | $0.0269$ | Captures non-linear interactions |
| **EXP_10** | **Ensemble Blend** | **Weighted Multi-Model Blend** | **$0.7118 \pm 0.0000$** | **$0.8196 \pm 0.0000$** | **$0.2269$** | **Selected Winning Model** |

---

## 5. Mandatory Competition Evaluation Metric

Per competition rules:
> **IT IS COMPULSORY TO ADD RMSE SCORE OF THE TRAINING DATA IN YOUR FINAL SUBMISSION NOTEBOOK.**

Our final retrained model achieves:
```text
=======================================================
Training RMSE: 0.2269
Training Pearson: 0.9861
=======================================================
```
* **5-Fold Cross-Validation OOF RMSE**: **0.7118**
* **5-Fold Cross-Validation OOF Pearson**: **0.8196**

---

## 6. How to Reproduce & Run

### A. Environment Installation
```bash
# Clone the repository
git clone https://github.com/eklavya434/shlproject.git
cd shlproject

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install exact pinned requirements
pip install -r requirements.txt
```

### B. Execute Feature Extraction & Modeling
The project features an automatic disk-caching architecture in `artifacts/`:
```bash
# 1. Extract digital acoustic features (~2 mins)
python src/extract_audio_features.py

# 2. Local Whisper ASR speech-to-text (~20 mins)
python src/transcribe.py

# 3. Extract linguistic NLP & POS features (~5 secs)
python src/extract_linguistic_features.py

# 4. Extract SentenceTransformer text embeddings (~10 secs)
python src/extract_text_embeddings.py

# 5. Run full cross-validation, ensembling, figure generation, and submission (~30 secs)
python src/run_pipeline.py
```

### C. Launch Final Notebook
```bash
jupyter notebook notebook/SHL_Grammar_Scoring_Final.ipynb
```

---

## 7. Final Submission Verification

The output submission is located at `submission.csv` and has been validated against all official constraints:
* **Row Count**: 216 test predictions matching `test.csv`.
* **Columns**: `filename`, `label`.
* **Value Range**: strictly within valid continuous bounds $[1.5645, 4.9946] \subset [0.0, 5.0]$.
* **Completeness**: 0 null / NaN / infinite values.
