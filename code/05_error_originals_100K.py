# 05_error_originals_100K.py - run in Colab (CPU). Attaches the ORIGINAL review text to the 30 Arabic100K error examples.
# Needs: ar_reviews_100k.csv (the original Kaggle file) + the analysis folder written by 04.
import os, re, csv, numpy as np, pandas as pd
from google.colab import drive
drive.mount("/content/drive")
RAW_100K = "/content/ar_reviews_100k.csv"        # <- upload the original file to Colab here (left sidebar > Files), or change the path
OUT = "/content/drive/MyDrive/ijci_results/analysis"
_AD = re.compile(r"[ً-ْٰـ]"); _URL = re.compile(r"http\S+|www\.\S+"); _MEN = re.compile(r"@\w+"); _REP = re.compile(r"(.)\1{2,}")
def _norm(t):
    if not isinstance(t, str): return ""
    t = _URL.sub(" ", t); t = _MEN.sub(" ", t); t = _AD.sub("", t)
    t = re.sub(r"[إأآا]", "ا", t); t = re.sub(r"ى", "ي", t); t = re.sub(r"ؤ", "ء", t); t = re.sub(r"ئ", "ء", t); t = re.sub(r"ة", "ه", t)
    t = _REP.sub(r"\1\1", t); t = re.sub(r"[^؀-ۿ0-9a-zA-Z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
def _anonymise(t):                    # remove @mentions, URLs and the RT prefix for printing in the paper
    t = _URL.sub("", t); t = _MEN.sub("@user", t); t = re.sub(r"^\s*RT\s+", "", t)
    return re.sub(r"\s+", " ", t).strip()
def raw_frame(name, path):
    if name == "ArSAS":
        d = pd.read_csv(path, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE, on_bad_lines="skip", engine="python"); d.columns = [c.strip() for c in d.columns]
        d = d.rename(columns={"Tweet_text": "orig", "Sentiment_label": "label"})
        d["label"] = d["label"].astype(str).str.strip().str.capitalize(); d = d[d.label.isin(["Positive", "Negative"])]
    elif name == "SS2030":
        d = pd.read_csv(path, encoding="utf-8-sig"); d.columns = [c.strip().lower() for c in d.columns]; d = d.rename(columns={"text": "orig"})
    else:
        d = pd.read_csv(path, sep="\t", encoding="utf-8-sig", quoting=3, on_bad_lines="skip", engine="python")
        if d.shape[1] < 2: d = pd.read_csv(path, encoding="utf-8-sig")
        d.columns = [c.strip().lower() for c in d.columns]; d = d.rename(columns={d.columns[0]: "label", d.columns[1]: "orig"})
        d["label"] = d["label"].astype(str).str.strip().str.capitalize(); d = d[d.label.isin(["Positive", "Negative"])]
    d["norm"] = d["orig"].apply(_norm); d = d[d["norm"].str.len() > 0].reset_index(drop=True)
    return d

ex = pd.read_csv(os.path.join(OUT, "error_examples_to_label.csv"), encoding="utf-8-sig")
name = "Arabic100KReviews"
d = raw_frame(name, RAW_100K); sub = ex[ex.dataset == name].copy()
sub["original_text"] = [_anonymise(str(d.loc[int(i), "orig"])) for i in sub.row_id]
ok = np.mean([d.loc[int(i), "norm"] == t for i, t in zip(sub.row_id, sub["text(normalised)"])])
print(f"{name}: rows in file after cleaning = {len(d)} (expected 66666); match rate = {ok:.3f} (must be 1.000)")
sub.to_csv(os.path.join(OUT, "error_examples_100K_with_original.csv"), index=False, encoding="utf-8-sig")
print("saved:", os.path.join(OUT, "error_examples_100K_with_original.csv"))
