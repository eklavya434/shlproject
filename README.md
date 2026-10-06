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
      ┌──────────────┼──────────────┐                              │
      ▼              ▼              ▼                              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐              ┌──────────────────┐
│ Linguistic│  │Word & Char│  │Pretrained │              │ 13 MFCCs + Deltas│
│  Grammar  │  │  TF-IDF   │  │Sentence   │              │ F0 Pitch Track   │
│  Features │  │ (in-fold) │  │Embeddings │              │ RMS Energy, ZCR  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘              └─────────┬────────┘
      │              │              │                              │
      └──────────────┴──────┬───────┴──────────────────────────────┘
                            ▼
            ┌───────────────────────────────┐
            │    Multimodal Feature Space   │
            │         (534 Features)        │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │    Strict Nested 5-Fold CV    │
            │   (Outer 5 x Inner 4 Folds)   │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │   Leakage-Safe Blended Model  │
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
│   └── SHL_Grammar_Scoring_Final.ipynb    # Clean, executed 24-section competition notebook
│
├── Dataset_Final/
│   ├── train/                             # 769 training audio files (.wav)
│   ├── test/                              # 216 test audio files (.wav)
│   ├── train.csv                          # Ground truth labels
│   ├── test.csv                           # Test metadata
│   └── sample_submission.csv              # Official placeholder template
│
├── artifacts/
│   ├── transcripts/                       # Cached verbatim Whisper ASR transcripts
│   ├── audio_features/                    # Cached MFCCs, F0, RMS energy features
│   ├── linguistic_features/               # Cached syntactic & POS features
│   ├── text_embeddings/                   # Pretrained SentenceTransformer embeddings (384-d)
│   ├── figures/                           # Visualizations (EDA, residuals, scatter plots)
│   ├── oof/                               # Out-of-fold prediction files for all models
│   ├── predictions/                       # Top 10 best and worst prediction audits
│   ├── audio_quality_report.csv           # 100% full audio quality audit report
│   ├── experiment_results.csv             # Full benchmark comparison table
│   └── submission_validation.txt          # Submission verification audit log
│
├── src/
│   ├── audit_data_and_audio.py            # Audio data quality & label audit
│   ├── extract_audio_features.py          # Acoustic signal processing engine
│   ├── transcribe.py                      # Local Whisper ASR transcription engine
│   ├── extract_linguistic_features.py     # POS, syntactic complexity & grammar feature extractor
│   ├── extract_text_embeddings.py         # Semantic sentence embedding extractor
│   ├── run_pipeline_v2.py                 # Complete end-to-end V2 execution pipeline
│   └── build_notebook_v2.py               # Automated notebook generator and runner
│
├── submission.csv                         # Final competition submission (216 rows)
├── requirements.txt                       # Locked dependencies
├── .gitignore                             # Git rules (excludes heavy wav binaries)
└── README.md                              # Complete solution documentation
```

---

## 4. Key Experiments & Benchmark Results Table

All models are evaluated using **5-Fold Cross-Validation** with strict in-fold pipeline encapsulation. For the final ensemble, **Nested 5-Fold Cross-Validation (Outer 5 folds $\times$ Inner 4 folds)** is employed to eliminate ensemble weight optimization leakage.

| Experiment ID | Feature Set | Model Architecture | CV RMSE (Mean ± Std) | CV Pearson (Mean ± Std) | Nested CV RMSE | Nested CV Pearson | Training RMSE | Training Pearson | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP_00** | None | Dummy Mean Predictor | $1.2382 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $1.2382$ | $0.0000$ | $1.2382$ | $0.0000$ | Constant mean baseline |
| **EXP_01** | Audio | Ridge Regressor ($\alpha=100$) | $0.8815 \pm 0.0583$ | $0.7002 \pm 0.0675$ | $0.8834$ | $0.7010$ | $0.7813$ | $0.7780$ | 119 acoustic/prosodic features |
| **EXP_02** | Linguistic | Ridge Regressor ($\alpha=10$) | $0.9624 \pm 0.0379$ | $0.6249 \pm 0.0398$ | $0.9632$ | $0.6292$ | $0.9203$ | $0.6691$ | 31 NLP/grammar features |
| **EXP_03** | Linguistic | HistGradientBoosting | $0.9796 \pm 0.0552$ | $0.6113 \pm 0.0627$ | $0.9813$ | $0.6184$ | $0.3397$ | $0.9695$ | Non-linear tree NLP |
| **EXP_04** | Word TF-IDF | Ridge Regressor ($\alpha=10$) | $0.9152 \pm 0.0682$ | $0.6724 \pm 0.0489$ | $0.9173$ | $0.6719$ | $0.0045$ | $1.0000$ | In-fold fitted vocabulary n-grams |
| **EXP_05** | Char TF-IDF | Ridge Regressor ($\alpha=10$) | $0.9854 \pm 0.0712$ | $0.6358 \pm 0.0461$ | $0.9878$ | $0.6347$ | $0.0065$ | $1.0000$ | In-fold character n-grams (3-5) |
| **EXP_06** | Text Embeddings | Ridge Regressor ($\alpha=50$) | $0.9316 \pm 0.0701$ | $0.6736 \pm 0.0500$ | $0.9344$ | $0.6726$ | $0.5769$ | $0.8870$ | 384-d MiniLM vectors |
| **EXP_07** | Ling + Audio | Ridge Regressor ($\alpha=100$) | $0.8185 \pm 0.0446$ | $0.7485 \pm 0.0534$ | $0.8196$ | $0.7499$ | $0.6954$ | $0.8296$ | Acoustic + Syntactic fusion |
| **EXP_08** | Multimodal (All) | Ridge Regressor ($\alpha=150$) | $0.7565 \pm 0.0296$ | $0.7940 \pm 0.0271$ | $0.7571$ | $0.7926$ | $0.4787$ | $0.9250$ | 534 multimodal features |
| **EXP_09** | Multimodal (All) | ElasticNet ($\alpha=0.1$) | $0.7508 \pm 0.0233$ | $0.7961 \pm 0.0311$ | $0.7511$ | $0.7963$ | $0.6406$ | $0.8619$ | Sparse regularized fusion |
| **EXP_10** | Multimodal (All) | HistGradientBoosting | $0.7447 \pm 0.0404$ | $0.7946 \pm 0.0440$ | $0.7458$ | $0.7984$ | $0.0282$ | $0.9998$ | Best individual model |
| **EXP_11** | **Multimodal Ensemble** | **Nested CV Blended Ensemble** | — | — | **$0.7172$** | **$0.8158$** | **$0.2610$** | **$0.9809$** | **Winning Leakage-Safe Model** |

---

## 5. Mandatory Competition Evaluation Metric

Per competition rules:
> **IT IS COMPULSORY TO ADD RMSE SCORE OF THE TRAINING DATA IN YOUR FINAL SUBMISSION NOTEBOOK.**

Our final retrained model achieves:
```text
=======================================================
FINAL TRAINING PERFORMANCE
Training RMSE: 0.2610
Training Pearson: 0.9809
=======================================================
```
* **Strict Nested Cross-Validation RMSE**: **0.7172**
* **Strict Nested Cross-Validation Pearson**: **0.8158**

> **Note on Training Metrics**: Training metrics are calculated on the full training dataset after fitting the final model. They are reported because the competition explicitly requires Training RMSE. They are not used as the primary model-selection criterion; cross-validation provides the more meaningful estimate of generalization.

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

# Install dependencies
pip install -r requirements.txt
```

### B. Run End-to-End Pipeline
```bash
# 1. Full Audio Data Quality Audit
python src/audit_data_and_audio.py

# 2. Extract Acoustic Features (~2 mins)
python src/extract_audio_features.py

# 3. Local Whisper ASR (~20 mins, cached)
python src/transcribe.py

# 4. Extract Linguistic NLP Features (~5 secs)
python src/extract_linguistic_features.py

# 5. Extract Text Embeddings (~10 secs)
python src/extract_text_embeddings.py

# 6. Run Complete V2 Pipeline (Nested CV, TF-IDF, Ensembling, Submissions)
python src/run_pipeline_v2.py
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
* **Value Range**: strictly within valid continuous bounds $[1.3710, 4.9971] \subset [0.0, 5.0]$.
* **Completeness**: 0 null / NaN / infinite values.
* **Verification Status**: **PASS** (documented in `artifacts/submission_validation.txt`).
