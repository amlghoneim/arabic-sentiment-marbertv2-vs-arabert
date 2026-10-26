# MARBERTv2 vs AraBERT for Arabic sentiment classification

Code, per-run results and analysis tables for the manuscript *"MARBERTv2 versus AraBERT for Arabic Sentiment Classification: A Controlled Multi-Seed Evaluation on Review and Social Media Datasets"*.

## What is here
| Path | Content |
|---|---|
| `code/02_transformers_multiseed.py` | Fine-tunes MARBERTv2 and AraBERT (3 seeds) on all datasets |
| `code/03_baselines.py` | TF-IDF baselines (LR, SVM, NB, DT, RF, KNN) and CNN / BiLSTM |
| `code/04_analysis_final.py` | Recomputes metrics from saved predictions, McNemar, bootstrap, Holm, tables and figures |
| `code/05_error_originals_100K.py`, `code/06_duplicate_check.py` | Error-example lookup and train/test overlap check |
| `results/` | Per-run metrics (`all_runs.csv`), summary and significance tables, categorised error examples |

## Protocol
Binary task (Positive = 1, Negative = 0; Neutral/Mixed removed). One fixed stratified 80/10/10 split per dataset (split seed 42). Three training seeds (42, 123, 2024) change only initialisation and data order. Checkpoints are selected on validation macro-F1; the test set is used once.

Transformers: `UBC-NLP/MARBERTv2`, `aubmindlab/bert-base-arabertv02`; max length 256, batch 16, lr 2e-5, weight decay 0.01, linear schedule with 10% warm-up, at most 3 epochs, fp16.
Software: Python 3.13, PyTorch 2.11.0, Transformers 5.20.1, scikit-learn 1.6.1.

## Data (not redistributed)
Download from the original sources and place in `/content` (or edit the paths in the scripts):
- Arabic 100K Reviews: https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews (`ar_reviews_100k.csv`, tab-separated)
- ArSAS (`ArSAS.txt`) and SS2030 (`SS2030.csv`): see the manuscript references.
Licences: Arabic 100K Reviews "Data files © Original Authors"; ArSAS and SS2030 licence "Unknown" on Kaggle.

## Reproducing
Run `02` and `03` (GPU recommended for `02`), then `04_analysis_final.py`. Fp16 training is not bit-wise deterministic (about 0.1 pt on Arabic 100K Reviews).

## Citation
Please cite the manuscript (IJCI, under revision).
