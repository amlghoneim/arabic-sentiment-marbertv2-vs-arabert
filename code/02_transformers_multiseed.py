# =============================================================================
# 02_transformers_multiseed.py  —  MARBERTv2 vs AraBERT, 3 seeds, 80/10/10 split
# Colab: Runtime > GPU. Upload ArSAS.txt, SS2030.csv, ar_reviews_100k.csv to /content (edit paths in CELL C).
# Paste each "# %% CELL" block into its own Colab cell, run in order (first line of each block: pip installs go in CELL 0).
# CELL 0:  !pip -q install transformers datasets accelerate scikit-learn
# Output per run (saved to Drive or /content/results): preds_*.csv, metrics_*.json, history_*.json, env.json, splits_*.csv
# =============================================================================
# %% CELL A — Imports, shared config, optional Google Drive (so results survive a Colab disconnect)
import os, re, json, random, shutil, math, time, platform
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SPLIT_SEED = 42                # ONE fixed split per dataset, shared by every model and every training seed
                               # -> all models are compared on the IDENTICAL test set (needed for McNemar)
TRAIN_SEEDS = [42, 123, 2024]  # seeds that change initialisation / data order only
DATASET_ORDER = ["SS2030", "ArSAS", "Arabic100KReviews"]   # small -> large, so partial results are usable early

# Save results to Google Drive if you can (survives disconnects). Otherwise falls back to /content/results.
RESULTS_DIR = "/content/results"
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:                       # on Colab, NEVER fall back silently to the temporary disk (its files vanish on disconnect)
    for _attempt in range(3):
        try:
            drive.mount("/content/drive", force_remount=(_attempt > 0))
        except Exception as e:
            print(f"Drive mount attempt {_attempt + 1} failed:", e)
        if os.path.isdir("/content/drive/MyDrive"):
            break
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError("Google Drive is NOT mounted, so results would be lost on disconnect. "
                           "Run this cell again and approve the Drive permission pop-up.")
    RESULTS_DIR = "/content/drive/MyDrive/ijci_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Results folder:", RESULTS_DIR)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# %% CELL B — Arabic preprocessing (Section 3.1)
ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # tashkeel + tatweel
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
MULTI_SPACE_RE = re.compile(r"\s+")

def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)
    text = re.sub(r"[^؀-ۿ0-9a-zA-Z\s]", " ", text)
    return MULTI_SPACE_RE.sub(" ", text).strip()

# %% CELL C — Dataset loaders + label mapping (Positive->1, Negative->0, Neutral/Mixed excluded)
def _finish(df):
    df["text"] = df["text"].apply(normalize_arabic)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["text", "binary_label"]]

def load_100k_reviews(path):
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", quoting=3, on_bad_lines="skip", engine="python")
    if df.shape[1] < 2:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    df = df[df["label"].isin(["Positive", "Negative"])].copy()      # Mixed excluded
    df["binary_label"] = (df["label"] == "Positive").astype(int)
    return _finish(df)

def load_arsas(path):
    # IMPORTANT: this file is tab-separated WITHOUT proper quoting. Default quoting lets a stray " swallow following
    # lines (it silently merged tweets and dropped 1,167 of 21,064 rows). QUOTE_NONE reads every physical row.
    import csv
    df = pd.read_csv(path, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE, on_bad_lines="skip", engine="python")
    assert len(df) == 21064, f"ArSAS: expected 21,064 rows (official size), got {len(df)} - parsing problem"
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Tweet_text": "text", "Sentiment_label": "label"})
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    df = df[df["label"].isin(["Positive", "Negative"])].copy()      # Neutral + Mixed excluded
    df["binary_label"] = (df["label"] == "Positive").astype(int)
    return _finish(df)

def load_ss2030(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"sentiment": "binary_label"})
    df["binary_label"] = df["binary_label"].astype(int)
    return _finish(df)

# EDIT THESE PATHS to where you uploaded the files in Colab
DATASETS = {
    "ArSAS": (load_arsas, "/content/ArSAS.txt"),
    "SS2030": (load_ss2030, "/content/SS2030.csv"),
    "Arabic100KReviews": (load_100k_reviews, "/content/ar_reviews_100k.csv"),
}

# %% CELL D — 80/10/10 stratified split (train / validation / TEST) + saving the split ids
def make_split(df, name, split_seed=SPLIT_SEED):
    df = df.reset_index(drop=True).copy()
    df["row_id"] = df.index
    train_val, test = train_test_split(df, test_size=0.10, stratify=df["binary_label"], random_state=split_seed)
    train, val = train_test_split(train_val, test_size=1/9, stratify=train_val["binary_label"], random_state=split_seed)
    ids = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])[["row_id", "split"]]
    ids.to_csv(os.path.join(RESULTS_DIR, f"splits_{name}.csv"), index=False)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# %% CELL E — Metrics computed DIRECTLY from predictions + run saving (same format for every model)
def metrics_from_preds(gold, pred):
    gold = np.asarray(gold); pred = np.asarray(pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
    p_p, r_p, f_p, _ = precision_recall_fscore_support(gold, pred, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(gold, pred)),
            "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
            "precision_pos": float(p_p), "recall_pos": float(r_p), "f1_pos": float(f_p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def run_done(run_name):
    return os.path.exists(os.path.join(RESULTS_DIR, f"metrics_{run_name}.json"))

def save_run(run_name, meta, test_df, pred, prob_pos, val_metrics=None, history=None):
    pd.DataFrame({"row_id": test_df["row_id"].values, "text": test_df["text"].values,
                  "gold_label": test_df["binary_label"].values, "pred_label": np.asarray(pred),
                  "prob_positive": np.asarray(prob_pos)}
                 ).to_csv(os.path.join(RESULTS_DIR, f"preds_{run_name}.csv"), index=False, encoding="utf-8-sig")
    rec = {**meta, "n_test": int(len(test_df)),
           "test": metrics_from_preds(test_df["binary_label"].values, pred), "val": val_metrics}
    with open(os.path.join(RESULTS_DIR, f"metrics_{run_name}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    if history is not None:
        with open(os.path.join(RESULTS_DIR, f"history_{run_name}.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=float)
    print(f"[saved] {run_name}: test acc={rec['test']['accuracy']:.4f} macro-F1={rec['test']['f1_macro']:.4f}")
    return rec

def write_env_json(extra):
    import sklearn
    info = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "device": DEVICE,
            "split_seed": SPLIT_SEED, "train_seeds": TRAIN_SEEDS, **extra}
    try:
        import transformers; info["transformers"] = transformers.__version__
    except Exception:
        pass
    with open(os.path.join(RESULTS_DIR, ENV_NAME), "w") as f:
        json.dump(info, f, indent=2, default=str)

# %% CELL F — Transformer config (all values go into the paper's hyperparameter table)
ENV_NAME = "env_transformers.json"
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset as HFDataset

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3                 # max epochs. Best epoch on 100K was 2 and val-loss rises after epoch 1-2. Change here if needed.
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1         # converted to explicit warm-up steps per dataset (answers R3 #8)
MAX_GRAD_NORM = 1.0
FP16 = torch.cuda.is_available()   # roughly 2x faster on Colab T4; set False to match fp32 exactly
CKPT_DIR = "/content/ckpt"            # local, fast: used for the small datasets
RESUME_DATASETS = {"Arabic100KReviews"}   # these keep checkpoints on Drive and RESUME after a disconnect
RUN_ALL = True

MODELS = {
    "MARBERTv2": "UBC-NLP/MARBERTv2",
    "AraBERT": "aubmindlab/bert-base-arabertv02",
}

# %% CELL G — Train one (model, dataset, seed), pick best epoch on VAL, report on TEST
def tokenize_df(df, tokenizer):
    ds = HFDataset.from_pandas(df[["text", "binary_label"]].reset_index(drop=True))
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH), batched=True)
    ds = ds.rename_column("binary_label", "labels")
    keep = ("input_ids", "attention_mask", "token_type_ids", "labels")
    return ds.remove_columns([c for c in ds.column_names if c not in keep])

def _val_metrics_fn(eval_pred):
    logits, labels = eval_pred
    m = metrics_from_preds(labels, np.argmax(logits, axis=-1))
    return {"accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"]}

def train_and_eval(model_key, model_name, dataset_name, splits, seed):
    run_name = f"{model_key}_{dataset_name}_seed{seed}"
    if run_done(run_name):
        print("[skip, already done]", run_name); return
    t0 = time.time()
    set_seed(seed)
    train_df, val_df, test_df = splits
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tr, va, te = tokenize_df(train_df, tok), tokenize_df(val_df, tok), tokenize_df(test_df, tok)

    steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    resumable = dataset_name in RESUME_DATASETS
    ckpt = os.path.join(RESULTS_DIR if resumable else CKPT_DIR, "ckpt_resume" if resumable else "", run_name)
    args = TrainingArguments(
        output_dir=ckpt, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=64,
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear", warmup_steps=warmup_steps, max_grad_norm=MAX_GRAD_NORM,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        logging_strategy="epoch", fp16=FP16, seed=seed, data_seed=seed, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=_val_metrics_fn)
    last_ckpt = None
    if resumable and os.path.isdir(ckpt):
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(ckpt)
    if last_ckpt:
        print(f"[resume] {run_name} continues from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    val_m = trainer.evaluate(va)                       # best checkpoint on validation
    out = trainer.predict(te)                          # untouched TEST set
    probs = torch.softmax(torch.tensor(out.predictions, dtype=torch.float32), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    evals = [h for h in trainer.state.log_history if "eval_f1_macro" in h]
    best_epoch = max(evals, key=lambda h: h["eval_f1_macro"])["epoch"] if evals else None
    meta = {"model": model_key, "hf_checkpoint": model_name, "dataset": dataset_name, "seed": seed,
            "split_seed": SPLIT_SEED, "n_train": len(train_df), "n_val": len(val_df),
            "epochs_max": EPOCHS, "best_epoch": best_epoch, "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "warmup_steps": warmup_steps,
            "total_steps": total_steps, "scheduler": "linear", "max_length": MAX_LENGTH,
            "fp16": FP16, "grad_accumulation": 1, "minutes": round((time.time() - t0) / 60, 1)}
    save_run(run_name, meta, test_df, preds, probs[:, 1],
             val_metrics={k.replace("eval_", ""): float(v) for k, v in val_m.items() if isinstance(v, (int, float))},
             history=trainer.state.log_history)
    shutil.rmtree(ckpt, ignore_errors=True)
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% CELL H — RUN EVERYTHING (safe to re-run after a disconnect: finished runs are skipped)
def collect_summary():
    rows = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.startswith("metrics_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(RESULTS_DIR, fn), encoding="utf-8"))
            rows.append({"model": r["model"], "dataset": r["dataset"], "seed": r["seed"],
                         "best_epoch": r.get("best_epoch"), **{f"test_{k}": v for k, v in r["test"].items()},
                         "val_f1_macro": (r.get("val") or {}).get("f1_macro")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_all_runs.csv"), index=False)
    return df

if RUN_ALL:
    write_env_json({"models": MODELS, "epochs_max": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO, "max_length": MAX_LENGTH,
                    "fp16": FP16})
    loaded = {n: DATASETS[n][0](DATASETS[n][1]) for n in DATASET_ORDER}
    for n, d in loaded.items():
        print(f"{n}: {len(d)} rows, pos={int(d['binary_label'].sum())}, neg={int((d['binary_label']==0).sum())}")
    splits_all = {n: make_split(d, n) for n, d in loaded.items()}
    for ds_name in DATASET_ORDER:
        for seed in TRAIN_SEEDS:
            for mk, mn in MODELS.items():
                train_and_eval(mk, mn, ds_name, splits_all[ds_name], seed)
        print(collect_summary().groupby(["dataset", "model"])["test_f1_macro"].agg(["mean", "std", "count"]))

# %% CELL I — Download results (only if you did not mount Drive)
# !cd /content && zip -r results.zip results
# from google.colab import files; files.download("/content/results.zip")
