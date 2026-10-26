# =============================================================================
# 06_duplicate_check.py - how many TEST texts also appear in TRAIN (retweets / repeated reviews)?
# And: do the scores change when those test items are removed?  (CPU is enough, ~2 minutes)
# Needs: the 3 raw files + the splits_*.csv and preds_*.csv already on Drive. It does NOT train and does NOT change any result file.
# Output: RESULTS_DIR/analysis/duplicate_check.csv  (send it to me) + a printed summary
# =============================================================================
import os, re, csv, glob, numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
try:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_DIR = "/content/drive/MyDrive/ijci_results"
except Exception:
    RESULTS_DIR = "/content/results"
assert os.path.isdir(RESULTS_DIR), RESULTS_DIR
OUT = os.path.join(RESULTS_DIR, "analysis"); os.makedirs(OUT, exist_ok=True)

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


# --- paths of the ORIGINAL raw files (same ones used in 02/03) ---
DATASETS = {
    "ArSAS": (load_arsas, "/content/ArSAS.txt"),
    "SS2030": (load_ss2030, "/content/SS2030.csv"),
    "Arabic100KReviews": (load_100k_reviews, "/content/ar_reviews_100k.csv"),
}
rows = []
for name, (loader, path) in DATASETS.items():
    if not os.path.exists(path):
        print("missing raw file, skipped:", name, path); continue
    df = loader(path); df["row_id"] = df.index
    sp = pd.read_csv(os.path.join(RESULTS_DIR, f"splits_{name}.csv"))
    df = df.merge(sp, on="row_id", how="inner"); assert len(df) == len(sp)
    train_txt = set(df[df.split == "train"]["text"]); val_txt = set(df[df.split == "val"]["text"])
    test = df[df.split == "test"].copy()
    test["dup_train"] = test["text"].isin(train_txt)
    pct = 100 * test["dup_train"].mean()
    lab = df[df.split == "train"].groupby("text")["binary_label"].agg(lambda s: s.mode().iloc[0])
    d = test[test.dup_train]; agree = float(np.mean([lab[t] == l for t, l in zip(d.text, d.binary_label)])) if len(d) else float("nan")
    print(f"{name}: test n={len(test)}, test texts also in train = {pct:.1f}%  (label agreement of those = {agree:.2f})")
    dupmap = dict(zip(test.row_id, test.dup_train))
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, f"preds_*_{name}_seed*.csv"))):
        p = pd.read_csv(f); m = os.path.basename(f)[len("preds_"):-4]
        model = m.split("_")[0]; seed = m.split("_seed")[1]
        p["dup"] = p["row_id"].map(dupmap); keep = p[~p["dup"].astype(bool)]
        rows.append({"dataset": name, "model": model, "seed": int(seed), "n_test": len(p), "n_dup": int(p["dup"].sum()), "pct_dup": round(pct, 2),
                     "f1_all": f1_score(p.gold_label, p.pred_label, average="macro"),
                     "f1_nodup": f1_score(keep.gold_label, keep.pred_label, average="macro"),
                     "acc_all": accuracy_score(p.gold_label, p.pred_label), "acc_nodup": accuracy_score(keep.gold_label, keep.pred_label)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "duplicate_check.csv"), index=False)
if len(res):
    print(res.groupby(["dataset", "model"])[["f1_all", "f1_nodup"]].mean().round(4).to_string())
print("saved:", os.path.join(OUT, "duplicate_check.csv"))
