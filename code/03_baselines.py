# =============================================================================
# 03_baselines.py  —  SVM, LR, RF, NB, DT, KNN (TF-IDF) + CNN, BiLSTM, SAME split/test sets as script 02
# Run in a SEPARATE Colab notebook (CPU is enough, GPU makes CNN/BiLSTM faster), upload the same 3 data files.
# IMPORTANT: use the SAME RESULTS_DIR (Google Drive) as script 02 so predictions sit side by side.
# CELL 0:  !pip -q install scikit-learn
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

# %% CELL F — Baselines config (this table goes into the paper: features, architecture, hyper-parameters, tuning)
ENV_NAME = "env_baselines.json"
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

TFIDF_PARAMS = dict(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True)  # fit on TRAIN only
CV_FOLDS = 3                       # tuning uses cross-validation INSIDE the training split only
NN_EPOCHS, NN_BATCH, NN_LR = 5, 64, 1e-3
NN_MAXLEN, NN_EMB, NN_VOCAB, NN_DROPOUT = 128, 128, 50000, 0.5
RUN_ALL = True

# name -> (estimator, param grid or None, is_stochastic)
def classical_models(seed):
    return {
        "LR":  (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}, False),
        "SVM": (LinearSVC(), {"C": [0.1, 1, 10]}, False),
        "NB":  (MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]}, False),
        "DT":  (DecisionTreeClassifier(random_state=seed), None, True),           # defaults (untuned)
        "RF":  (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed), None, True),  # untuned
        "KNN": (KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1), None, False),
    }

# %% CELL G — Classical models on TF-IDF
def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1 / (1 + np.exp(-d))          # monotone score in (0,1) for LinearSVC

def run_classical(model_key, dataset_name, splits, seed):
    run_name = f"{model_key}_{dataset_name}_seed{seed}"
    if run_done(run_name):
        print("[skip]", run_name); return
    t0 = time.time()
    train_df, val_df, test_df = splits
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    Xtr = vec.fit_transform(train_df["text"]); Xva = vec.transform(val_df["text"]); Xte = vec.transform(test_df["text"])
    est, grid, _ = classical_models(seed)[model_key]
    best_params = {}
    if grid:
        gs = GridSearchCV(est, grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        gs.fit(Xtr, train_df["binary_label"]); est = gs.best_estimator_; best_params = gs.best_params_
    else:
        est.fit(Xtr, train_df["binary_label"])
    val_pred = est.predict(Xva); test_pred = est.predict(Xte)
    meta = {"model": model_key, "dataset": dataset_name, "seed": seed, "split_seed": SPLIT_SEED,
            "features": f"TF-IDF {TFIDF_PARAMS}", "tuned": bool(grid), "best_params": best_params,
            "cv_folds": CV_FOLDS if grid else None, "best_epoch": None,
            "n_train": len(train_df), "n_val": len(val_df), "minutes": round((time.time() - t0) / 60, 1)}
    save_run(run_name, meta, test_df, test_pred, _scores(est, Xte),
             val_metrics=metrics_from_preds(val_df["binary_label"].values, val_pred))

# %% CELL H — CNN and BiLSTM (randomly initialised trainable embeddings, no pre-trained vectors)
def build_vocab(texts, max_size=NN_VOCAB, min_freq=2):
    from collections import Counter
    c = Counter(w for t in texts for w in t.split())
    words = [w for w, f in c.most_common(max_size) if f >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}          # 0 = pad, 1 = unk

def encode(texts, vocab, maxlen=NN_MAXLEN):
    arr = np.zeros((len(texts), maxlen), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()][:maxlen]
        arr[i, :len(ids)] = ids
    return torch.tensor(arr)

class TextCNN(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, NN_EMB, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(NN_EMB, 100, k) for k in (3, 4, 5)])
        self.drop = nn.Dropout(NN_DROPOUT); self.fc = nn.Linear(300, 2)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)
        return self.fc(self.drop(h))

class BiLSTM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, NN_EMB, padding_idx=0)
        self.rnn = nn.LSTM(NN_EMB, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(NN_DROPOUT); self.fc = nn.Linear(256, 2)
    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        o, _ = self.rnn(self.emb(x))
        o = o.masked_fill(~mask, -1e9).max(dim=1).values      # masked max-pool over time
        return self.fc(self.drop(o))

def _predict(model, X, bs=256):
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(torch.softmax(model(X[i:i + bs].to(DEVICE)), dim=-1).cpu())
    return torch.cat(outs).numpy()

def run_nn(model_key, dataset_name, splits, seed):
    run_name = f"{model_key}_{dataset_name}_seed{seed}"
    if run_done(run_name):
        print("[skip]", run_name); return
    t0 = time.time(); set_seed(seed)
    train_df, val_df, test_df = splits
    vocab = build_vocab(train_df["text"])
    Xtr, Xva, Xte = (encode(d["text"], vocab) for d in (train_df, val_df, test_df))
    ytr = torch.tensor(train_df["binary_label"].values)
    model = (TextCNN if model_key == "CNN" else BiLSTM)(len(vocab) + 2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=NN_LR); lossf = nn.CrossEntropyLoss()
    best_f1, best_state, best_ep, hist = -1, None, None, []
    for ep in range(1, NN_EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), NN_BATCH):
            idx = perm[i:i + NN_BATCH]
            opt.zero_grad(); loss = lossf(model(Xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE)); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        vm = metrics_from_preds(val_df["binary_label"].values, _predict(model, Xva).argmax(1))
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_f1_macro": vm["f1_macro"]})
        if vm["f1_macro"] > best_f1:
            best_f1, best_ep = vm["f1_macro"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)                         # best epoch chosen on VAL only
    probs = _predict(model, Xte)
    meta = {"model": model_key, "dataset": dataset_name, "seed": seed, "split_seed": SPLIT_SEED,
            "arch": "TextCNN: emb128 (random init) + conv k=3,4,5 x100 + maxpool + dropout0.5" if model_key == "CNN"
                    else "BiLSTM: emb128 (random init) + 1 layer 128 hidden bidirectional + masked maxpool + dropout0.5",
            "epochs_max": NN_EPOCHS, "best_epoch": best_ep, "batch_size": NN_BATCH, "lr": NN_LR, "optimizer": "Adam",
            "max_len": NN_MAXLEN, "vocab_size": len(vocab) + 2, "n_train": len(train_df), "n_val": len(val_df),
            "minutes": round((time.time() - t0) / 60, 1)}
    save_run(run_name, meta, test_df, probs.argmax(1), probs[:, 1],
             val_metrics={"f1_macro": best_f1}, history=hist)

# %% CELL I — RUN EVERYTHING (finished runs are skipped on re-run)
def collect_summary():
    rows = []
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.startswith("metrics_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(RESULTS_DIR, fn), encoding="utf-8"))
            rows.append({"model": r["model"], "dataset": r["dataset"], "seed": r["seed"],
                         **{f"test_{k}": v for k, v in r["test"].items()}})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS_DIR, "summary_baselines.csv"), index=False); return df

if RUN_ALL:
    write_env_json({"tfidf": TFIDF_PARAMS, "cv_folds": CV_FOLDS, "nn_epochs": NN_EPOCHS, "nn_lr": NN_LR})
    loaded = {n: DATASETS[n][0](DATASETS[n][1]) for n in DATASET_ORDER}
    splits_all = {n: make_split(d, n) for n, d in loaded.items()}
    for ds_name in DATASET_ORDER:
        sp = splits_all[ds_name]
        for mk in ["LR", "SVM", "NB", "DT", "RF", "KNN"]:
            stoch = classical_models(42)[mk][2]
            for seed in (TRAIN_SEEDS if stoch else [TRAIN_SEEDS[0]]):
                run_classical(mk, ds_name, sp, seed)
        for mk in ["CNN", "BiLSTM"]:
            for seed in TRAIN_SEEDS:
                run_nn(mk, ds_name, sp, seed)
    print(collect_summary().groupby(["dataset", "model"])["test_f1_macro"].agg(["mean", "std", "count"]))
