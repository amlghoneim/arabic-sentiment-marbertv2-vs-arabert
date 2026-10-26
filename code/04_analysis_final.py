# =============================================================================
# 04_analysis.py — turns the saved runs of scripts 02 + 03 into paper-ready tables, tests and figures
# Colab: CPU is enough. CELL 0:  !pip -q install scikit-learn scipy matplotlib
# Reads  : RESULTS_DIR/metrics_*.json, preds_*.csv, history_*.json   (written by 02 and 03)
# Writes : RESULTS_DIR/analysis/  (CSV/MD tables, PNG figures, error samples to label by hand)
# =============================================================================

# %% CELL A — Config
import os, json, glob
import numpy as np
import pandas as pd
from scipy.stats import binomtest

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
OUT = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)

DATASETS = ["Arabic100KReviews", "ArSAS", "SS2030"]
TRANSFORMERS = ["MARBERTv2", "AraBERT"]
REF_SEED = 42                       # seed used for single-run artefacts (confusion matrix, error examples)
N_BOOT = 2000
rng = np.random.default_rng(0)

# %% CELL B — Load every run
rows = []
for fn in sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_*.json"))):
    r = json.load(open(fn, encoding="utf-8"))
    t = r["test"]
    rows.append({"model": r["model"], "dataset": r["dataset"], "seed": r["seed"],
                 "best_epoch": r.get("best_epoch"), "n_test": r.get("n_test"), **t})
runs = pd.DataFrame(rows)
assert len(runs), f"no metrics_*.json found in {RESULTS_DIR}"
runs.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
print(runs.groupby(["dataset", "model"]).size().unstack(0))

# %% CELL C — Main table: mean ± std over seeds (single-run models are marked)
def to_md(df):                      # tiny markdown table writer (no extra dependency)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)

def pm(x, pct=True):
    x = np.asarray(x, dtype=float)
    k = 100 if pct else 1
    return f"{x.mean()*k:.2f}" if len(x) == 1 else f"{x.mean()*k:.2f} ± {x.std(ddof=1)*k:.2f}"

main = []
for ds in DATASETS:
    d = runs[runs.dataset == ds]
    for m, g in d.groupby("model"):
        main.append({"dataset": ds, "model": m, "n_runs": len(g),
                     "Accuracy": pm(g.accuracy), "Precision(macro)": pm(g.precision_macro),
                     "Recall(macro)": pm(g.recall_macro), "F1(macro)": pm(g.f1_macro),
                     "_f1": g.f1_macro.mean()})
main = pd.DataFrame(main).sort_values(["dataset", "_f1"], ascending=[True, False])
main.drop(columns="_f1").to_csv(os.path.join(OUT, "table_main_results.csv"), index=False)
with open(os.path.join(OUT, "table_main_results.md"), "w", encoding="utf-8") as f:
    for ds in DATASETS:
        f.write(f"\n### {ds}  (test set, % ; mean ± sample std over seeds; n_runs=1 -> deterministic model, no std)\n\n")
        f.write(to_md(main[main.dataset == ds].drop(columns=["dataset", "_f1"])) + "\n")
print(open(os.path.join(OUT, "table_main_results.md"), encoding="utf-8").read())

# %% CELL D — Paired significance on the IDENTICAL test set: exact McNemar + paired bootstrap CI of macro-F1 difference
def load_preds(model, ds, seed):
    p = os.path.join(RESULTS_DIR, f"preds_{model}_{ds}_seed{seed}.csv")
    return pd.read_csv(p).set_index("row_id") if os.path.exists(p) else None

def macro_f1_boot(y, p, idx):
    yy, pp = y[idx], p[idx]                                   # idx: (B, n)
    f = []
    for c in (0, 1):
        tp = ((yy == c) & (pp == c)).sum(1); fp = ((yy != c) & (pp == c)).sum(1); fn = ((yy == c) & (pp != c)).sum(1)
        f.append(np.where(2*tp + fp + fn > 0, 2*tp / np.maximum(2*tp + fp + fn, 1), 0.0))
    return (f[0] + f[1]) / 2

def compare(a_model, b_model, ds, seed_a, seed_b):
    A, B = load_preds(a_model, ds, seed_a), load_preds(b_model, ds, seed_b)
    if A is None or B is None:
        return None
    ids = A.index.intersection(B.index)
    y = A.loc[ids, "gold_label"].values; pa = A.loc[ids, "pred_label"].values; pb = B.loc[ids, "pred_label"].values
    ca, cb = pa == y, pb == y
    b01, c10 = int((ca & ~cb).sum()), int((~ca & cb).sum())       # A right/B wrong ; A wrong/B right
    p = 1.0 if b01 + c10 == 0 else binomtest(min(b01, c10), b01 + c10, 0.5).pvalue
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    d = macro_f1_boot(y, pa, idx) - macro_f1_boot(y, pb, idx)
    f1 = lambda q: float(macro_f1_boot(y, q, np.arange(len(y))[None, :])[0])
    return {"dataset": ds, "A": a_model, "B": b_model, "seed_A": seed_a, "seed_B": seed_b, "n_test": len(ids),
            "acc_A": ca.mean(), "acc_B": cb.mean(), "A_right_B_wrong": b01, "A_wrong_B_right": c10,
            "mcnemar_p": p, "dF1_macro": f1(pa) - f1(pb),
            "dF1_CI95_low": float(np.percentile(d, 2.5)), "dF1_CI95_high": float(np.percentile(d, 97.5))}

def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj

res = []
for ds in DATASETS:
    d = runs[runs.dataset == ds]
    # (1) MARBERTv2 vs AraBERT, same seed
    for s in sorted(set(d[d.model == "MARBERTv2"].seed) & set(d[d.model == "AraBERT"].seed)):
        r = compare("MARBERTv2", "AraBERT", ds, s, s)
        if r: res.append(r)
    # (2) MARBERTv2 vs the strongest non-transformer baseline (by mean macro-F1), seed 42 vs its seed 42
    base = d[~d.model.isin(TRANSFORMERS)].groupby("model").f1_macro.mean()
    if len(base):
        best = base.idxmax()
        r = compare("MARBERTv2", best, ds, REF_SEED, REF_SEED)
        if r: res.append(r)
sig = pd.DataFrame(res)
if len(sig):
    sig["mcnemar_p_holm"] = holm(sig.mcnemar_p.values)
    sig.to_csv(os.path.join(OUT, "table_significance.csv"), index=False)
    print(sig.round(4).to_string())
    print("\nNote: p-values are per-seed tests on one fixed test set; Holm correction applied across all rows above.")

# %% CELL E — Confusion matrices + metrics recomputed from the SAME predictions (fixes the Table 4 vs Table 11 inconsistency)
cm = runs[runs.model.isin(TRANSFORMERS)][["dataset", "model", "seed", "tn", "fp", "fn", "tp",
                                         "accuracy", "precision_pos", "recall_pos", "f1_pos",
                                         "precision_macro", "recall_macro", "f1_macro"]].sort_values(["dataset", "model", "seed"])
cm.to_csv(os.path.join(OUT, "table_confusion_transformers.csv"), index=False)
print(cm.round(4).to_string(index=False))

# %% CELL F — Real misclassified examples (to label MANUALLY into error categories, R1 / R3 #11)
ERR_N = 10
samples = []
for ds in DATASETS:
    P = load_preds("MARBERTv2", ds, REF_SEED)
    if P is None: continue
    err = P[P.gold_label != P.pred_label].copy(); err["row_id"] = err.index
    err["conf"] = np.where(err.pred_label == 1, err.prob_positive, 1 - err.prob_positive)
    fp = err[(err.gold_label == 0)]; fn = err[(err.gold_label == 1)]
    pick = [("random false positive (gold neg, pred pos)", fp.sample(min(ERR_N, len(fp)), random_state=0)),
            ("random false negative (gold pos, pred neg)", fn.sample(min(ERR_N, len(fn)), random_state=0)),
            ("most confident error", err.sort_values("conf", ascending=False).head(ERR_N))]
    for kind, df in pick:
        for _, r in df.iterrows():
            samples.append({"dataset": ds, "how_selected": kind, "row_id": r.row_id, "text(normalised)": r.text,
                            "gold": int(r.gold_label), "pred": int(r.pred_label), "confidence": round(float(r.conf), 3),
                            "category (fill by hand)": "", "note": ""})
    print(f"{ds}: {len(err)} errors out of {len(P)} test examples ({len(fp)} FP, {len(fn)} FN)")
pd.DataFrame(samples).to_csv(os.path.join(OUT, "error_examples_to_label.csv"), index=False, encoding="utf-8-sig")

# %% CELL G — Figure: macro-F1 of ALL models on every dataset (replaces old Fig. 3; dot = mean, bar = ±1 std over seeds)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 4.6), sharey=False)
for ax, ds in zip(np.atleast_1d(axes), DATASETS):
    g = runs[runs.dataset == ds].groupby("model").f1_macro.agg(["mean", "std"]).fillna(0).sort_values("mean")
    if g.empty:
        ax.set_visible(False); continue
    colors = ["#c0392b" if m == "MARBERTv2" else "#2c6fbb" if m == "AraBERT" else "#7f8c8d" for m in g.index]
    ax.errorbar(g["mean"] * 100, range(len(g)), xerr=g["std"] * 100, fmt="none", ecolor="#555", elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(g["mean"] * 100, range(len(g)), c=colors, s=55, zorder=2)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index)
    ax.set_xlim(max(0, (g["mean"] - g["std"]).min() * 100 - 2), min(100.5, (g["mean"] + g["std"]).max() * 100 + 1.5))
    ax.set_title(ds); ax.set_xlabel("Test macro-F1 (%)"); ax.grid(axis="x", alpha=0.3)
fig.suptitle("Macro-F1 on the held-out test set (mean ± std over seeds; models with one run have no bar)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure_all_models_f1.png"), dpi=300); plt.close(fig)

# %% CELL H — Figure: train vs validation loss per epoch (replaces old Fig. 4; epochs numbered 1..E consistently)
def load_hist(model, ds, seed):
    p = os.path.join(RESULTS_DIR, f"history_{model}_{ds}_seed{seed}.json")
    if not os.path.exists(p): return None
    h = json.load(open(p, encoding="utf-8"))
    tr, va = {}, {}
    for x in h:                       # keep the FIRST entry per epoch: the final trainer.evaluate() adds a duplicate
        if "epoch" not in x:          # entry (epoch = last) with the BEST checkpoint's loss, which must not overwrite it
            continue
        if "loss" in x and "eval_loss" not in x:
            tr.setdefault(round(x["epoch"]), x["loss"])
        if "eval_loss" in x:
            va.setdefault(round(x["epoch"]), x["eval_loss"])
    return tr, va

fig, axes = plt.subplots(len(TRANSFORMERS), len(DATASETS), figsize=(13, 6), sharex=False, squeeze=False)
for i, m in enumerate(TRANSFORMERS):
    for j, ds in enumerate(DATASETS):
        ax = axes[i][j]; H = load_hist(m, ds, REF_SEED)
        if H:
            tr, va = H
            ax.plot(sorted(tr), [tr[k] for k in sorted(tr)], "-o", label="train loss", color="#2c6fbb")
            ax.plot(sorted(va), [va[k] for k in sorted(va)], "--s", label="validation loss", color="#c0392b")
            ax.set_xticks(sorted(set(tr) | set(va)))
        ax.set_title(f"{m} – {ds}", fontsize=10); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        if i == 0 and j == 0: ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure_loss_curves.png"), dpi=300); plt.close(fig)

# %% CELL I — Epoch picked by validation (answers the "5 epochs vs epoch 1" contradiction with data)
be = runs[runs.model.isin(TRANSFORMERS)].groupby(["dataset", "model"]).best_epoch.apply(lambda s: list(s.astype(int)))
be.to_csv(os.path.join(OUT, "table_best_epochs.csv"))
print(be)
print("\nDone. Files in:", OUT)

# %% CELL J — Attach the ORIGINAL text to the error examples (the normalised text is not suitable to print in the paper)
# EDIT the three raw-file paths (same files you used in 02/03). Row ids = position after label filtering + empty-text removal.
import re, csv
RAW = {"ArSAS": "/content/drive/MyDrive/ijci_raw/ArSAS.txt", "SS2030": "/content/drive/MyDrive/ijci_raw/SS2030.csv", "Arabic100KReviews": "/content/drive/MyDrive/ijci_raw/ar_reviews_100k.csv"}   # EDIT to where you uploaded the raw files
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
outs = []
for name, path in RAW.items():
    if not os.path.exists(path):
        print("missing raw file, skipped:", name, path); continue
    d = raw_frame(name, path); sub = ex[ex.dataset == name].copy()
    sub["original_text"] = [_anonymise(str(d.loc[int(i), "orig"])) for i in sub.row_id]
    ok = np.mean([d.loc[int(i), "norm"] == t for i, t in zip(sub.row_id, sub["text(normalised)"])])
    print(f"{name}: row_id -> raw text match rate = {ok:.3f}  (must be 1.000)")
    outs.append(sub)
if outs:
    pd.concat(outs).to_csv(os.path.join(OUT, "error_examples_with_original.csv"), index=False, encoding="utf-8-sig")
