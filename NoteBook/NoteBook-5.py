# %% [markdown]
# # Notebook 5 — Feature Selection & Preprocessing  (Q1-Grade, Temporal-Split)
#
# **Inputs** (from Notebook 4):
# | File | Description |
# |---|---|
# | `extracted_dataset.parquet` | 351-col feature matrix (42892 addresses, 8 classes) |
# | `behavioral_scores.parquet` | 8 heuristic risk scores (ablation only — NOT merged here) |
# | `fe_summary.json`           | Feature group inventory for ablation-aware selection |
#
# **Outputs**:
# | File | Description |
# |---|---|
# | `processed_dataset.parquet`  | Train + test rows, selected + scaled features, split column |
# | `feature_selector.pkl`       | Fitted selector (MI + variance threshold) for NB6 inference |
# | `scaler.pkl`                 | Fitted RobustScaler (train only) for NB6 inference |
# | `split_meta.json`            | Split timestamps, class counts, feature list |
#
# ──────────────────────────────────────────────────────────────────────────────
# ## PIPELINE DESIGN
# ──────────────────────────────────────────────────────────────────────────────
#
#  STEP 1 — Per-class stratified temporal train/test split
#           A GLOBAL first_ts split causes severe train/test class-composition
#           skew because each class spans the 2015-2026 collection window with
#           very different temporal densities. Instead, EACH CLASS is sorted
#           independently by first_ts: earliest 80% -> train, latest 20% ->
#           test, then concatenated. No random shuffle within a class —
#           preserves per-class temporal causality, eliminates global skew,
#           and loses no addresses.
#           Addresses with missing first_ts get median first_ts imputed
#           (already done in NB4 as `has_collection_ts` flag).
#
#  STEP 2 — Refit clip bounds on train only
#           NB4 used full-dataset quantile bounds. NB5 refits on train,
#           then applies to test — prevents test leakage through clipping.
#
#  STEP 3 — Class-conditional median imputation (train → apply to test)
#           Only for `days_since_last_activity`. Fit medians per class
#           on train, apply to test by label. Global fallback for classes
#           absent from train.
#
#  STEP 4 — Variance threshold filter
#           Drop features with near-zero variance on the training set.
#           Threshold: VarianceThreshold(threshold=0.01).
#
#  STEP 5 — Mutual Information (MI) feature selection
#           Compute MI(feature, label) on TRAIN only. Rank all features.
#           Retain top-K by MI score, enforcing a minimum from each group
#           so no single group is wiped out (min 2 per group if group > 2).
#
#  STEP 6 — RobustScaler
#           Fit on train only; apply to train + test.
#           RobustScaler is preferred over StandardScaler for this dataset:
#           heavy-tailed distributions (MEV gas spikes, Ponzi inflows) make
#           mean/std unstable; median/IQR is more robust.
#
#  LEAKAGE GUARDRAILS:
#   • All fit operations (clip, imputation, variance, MI, scaler) use TRAIN only.
#   • Test set is transformed by applying train-fitted objects — never refitted.
#   • Class imbalance NOT removed. Class weights will be set in NB7 model.
#   • behavioral_scores.parquet is NOT merged here.

# %%
import gc, json, pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

SEED = 42
np.random.seed(SEED)

FEAT_DIR = Path("/kaggle/working/features")
OUT_DIR  = Path("/kaggle/working/features")
FIG_DIR  = Path("/kaggle/working/figures_nb5")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "benign", "phishing", "rug_pull", "ponzi",
    "flash_loan_attack", "malicious_mev", "exploit_contract", "honeypot",
]
CLASS_PALETTE = {
    "benign": "#4CAF50", "phishing": "#F44336", "rug_pull": "#FF9800",
    "ponzi": "#9C27B0", "flash_loan_attack": "#00BCD4",
    "malicious_mev": "#E91E63", "exploit_contract": "#FF5722",
    "honeypot": "#795548",
}

print("=== Notebook 5 — Feature Selection & Preprocessing ===")
print(f"Input  : {FEAT_DIR}")
print(f"Output : {OUT_DIR}")


# %% [markdown]
# ## 1.  Load extracted dataset & feature group inventory

# %%
print("\n[1] Loading extracted_dataset.parquet …")
df = pd.read_parquet(FEAT_DIR / "extracted_dataset.parquet")
print(f"    shape: {df.shape}")
print(df["label_name"].value_counts().to_string())

print("\n[1] Loading fe_summary.json …")
with open(FEAT_DIR / "fe_summary.json") as fh:
    fe_summary = json.load(fh)
FEATURE_GROUPS = fe_summary["feature_groups"]
COHORT_MARKER_COLS = fe_summary.get("cohort_marker_cols", ["first_ts", "last_ts"])
print(f"    {len(FEATURE_GROUPS)} feature groups, "
      f"{fe_summary['total_features']} total features")

# ── Identify meta / non-model columns ────────────────────────────────────────
META_COLS = {"address", "label", "label_name", "first_ts", "last_ts",
             "has_collection_ts"}
ALL_FEAT_COLS = [c for c in df.columns if c not in META_COLS]
print(f"    Candidate model features: {len(ALL_FEAT_COLS)}")

# Verify no NaN (NB4 guaranteed this)
nan_count = df[ALL_FEAT_COLS].isna().sum().sum()
print(f"    NaN count in features: {nan_count}  ({'OK' if nan_count == 0 else 'ISSUE'})")

# Replace inf/-inf before any sklearn operations.
# Root cause: ratio features in NB4 (velocity_ratio_7d, gas_price_spike_ratio, etc.)
# clip denominator to eps=1e-9, producing values up to ~1e15. NB4's clip pass misses
# these because quantile(0.999) on an inf-containing column returns inf → clip is no-op.
_inf_vals = np.isinf(df[ALL_FEAT_COLS].values)
_inf_total = int(_inf_vals.sum())
if _inf_total > 0:
    _inf_cols = [ALL_FEAT_COLS[j] for j in range(len(ALL_FEAT_COLS))
                 if _inf_vals[:, j].any()]
    print(f"    [fix] {_inf_total} inf/-inf values in {len(_inf_cols)} cols → set to 0.0")
    print(f"      Cols: {_inf_cols[:15]}")
    df[_inf_cols] = df[_inf_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"    NaN after inf fix: {df[ALL_FEAT_COLS].isna().sum().sum()}  (should be 0)")
del _inf_vals
print(f"    Any inf remaining: {np.isinf(df[ALL_FEAT_COLS].values).any()}")


# %% [markdown]
# ## 2.  Per-class stratified temporal train / test split

# %%
print("\n[2] Per-class stratified temporal train/test split …")

# A GLOBAL first_ts split causes severe train/test class-composition skew:
# every class spans almost the entire 2015-2026 collection window, but with
# very different temporal densities (e.g. malicious_mev / rug_pull are heavily
# concentrated post-2020), so a single global cut leaves some classes almost
# entirely in test (or train). Instead, split EACH CLASS independently:
# sort that class's addresses by first_ts, earliest 80% -> train, latest 20%
# -> test, then concatenate across classes. This preserves per-class temporal
# causality (no future-to-past leakage within a class), eliminates global
# class-composition skew, and loses no addresses.
train_parts, test_parts = [], []
for cn in CLASS_NAMES:
    sub = df[df["label_name"] == cn].sort_values("first_ts", ascending=True, ignore_index=True)
    n_sub = len(sub)
    n_tr  = int(n_sub * 0.80)
    train_parts.append(sub.iloc[:n_tr])
    test_parts.append(sub.iloc[n_tr:])

train_df = pd.concat(train_parts, ignore_index=True)
test_df  = pd.concat(test_parts,  ignore_index=True)
del train_parts, test_parts

n_total = len(df)
n_train = len(train_df)
n_test  = len(test_df)

print(f"    Train: {n_train:,} ({n_train/n_total*100:.1f}%)")
print(f"    Test : {n_test:,}  ({n_test/n_total*100:.1f}%)")

print("\n    Class distribution in train:")
print(train_df["label_name"].value_counts().to_string())
print("\n    Class distribution in test:")
print(test_df["label_name"].value_counts().to_string())

# Per-class temporal cut points (earliest 80% / latest 20% boundary timestamp)
print("\n    Per-class first_ts split boundary (train max -> test min):")
for cn in CLASS_NAMES:
    tr_ts = train_df.loc[train_df["label_name"] == cn, "first_ts"]
    te_ts = test_df.loc[test_df["label_name"] == cn, "first_ts"]
    if len(tr_ts) == 0 or len(te_ts) == 0:
        print(f"      {cn:22s}: n_train={len(tr_ts):5d}  n_test={len(te_ts):5d}  (one side empty)")
        continue
    print(f"      {cn:22s}: train_max={tr_ts.max():.0f}  test_min={te_ts.min():.0f}")

gc.collect()


# %% [markdown]
# ## 3.  Refit clip bounds on train only

# %%
print("\n[3] Refitting outlier clip bounds on train …")

# NB4 computed quantile bounds on the full dataset — a mild leakage through
# the test set's distribution. Refit here strictly on train.
PROTECTED_FROM_CLIP = {"label", "first_ts", "last_ts", "has_collection_ts"}
clip_cols = [c for c in ALL_FEAT_COLS if c not in PROTECTED_FROM_CLIP]

lo = train_df[clip_cols].quantile(0.001)
hi = train_df[clip_cols].quantile(0.999)

train_df[clip_cols] = train_df[clip_cols].clip(lower=lo, upper=hi, axis=1)
test_df[clip_cols]  = test_df[clip_cols].clip(lower=lo, upper=hi, axis=1)

print(f"    Clip bounds refit on {len(train_df):,} train rows, "
      f"applied to {len(test_df):,} test rows.")
print(f"    Columns clipped: {len(clip_cols)}")


# %% [markdown]
# ## 4.  Class-conditional median imputation for days_since_last_activity

# %%
print("\n[4] Class-conditional median imputation (train-fitted) …")

# days_since_last_activity NaN in the original data = collection_ts missing.
# NB4 used global median (leakage-free pre-split). Here we upgrade to
# class-conditional median, computed on train ONLY, applied to both splits.
IMPUTE_COL = "days_since_last_activity"

if IMPUTE_COL in train_df.columns:
    class_medians = (train_df.groupby("label_name")[IMPUTE_COL]
                              .median()
                              .to_dict())
    global_fallback = train_df[IMPUTE_COL].median()

    print(f"    Class-conditional medians (train):")
    for cn in CLASS_NAMES:
        m = class_medians.get(cn, global_fallback)
        print(f"      {cn:20s}: {m:.1f} days")

    # Apply to train and test (only addresses with has_collection_ts == 0 would
    # have had NaN originally; after NB4 global imputation they are non-NaN.
    # We overwrite with class-conditional values for those addresses.)
    no_ts_mask_train = train_df["has_collection_ts"] == 0
    no_ts_mask_test  = test_df["has_collection_ts"]  == 0

    for cn in CLASS_NAMES:
        med = class_medians.get(cn, global_fallback)
        mask_t = no_ts_mask_train & (train_df["label_name"] == cn)
        mask_v = no_ts_mask_test  & (test_df["label_name"]  == cn)
        train_df.loc[mask_t, IMPUTE_COL] = med
        test_df.loc[mask_v,  IMPUTE_COL] = med

    print(f"\n    NaN remaining — train: {train_df[IMPUTE_COL].isna().sum()}"
          f"  test: {test_df[IMPUTE_COL].isna().sum()}")
else:
    print(f"    [warn] {IMPUTE_COL} not in columns — skipping")
    class_medians = {}
    global_fallback = 0.0

gc.collect()


# %% [markdown]
# ## 5.  Variance threshold filter (train-fitted)

# %%
print("\n[5] Variance threshold filter …")

# Drop features with near-zero variance on the training set.
# Threshold 0.01 eliminates constants and near-constants.
vt = VarianceThreshold(threshold=0.01)
vt.fit(train_df[ALL_FEAT_COLS].values)

low_var_mask  = ~vt.get_support()
low_var_cols  = [c for c, drop in zip(ALL_FEAT_COLS, low_var_mask) if drop]
keep_var_cols = [c for c, keep in zip(ALL_FEAT_COLS, vt.get_support()) if keep]

print(f"    Dropped {len(low_var_cols)} low-variance features (< 0.01 on train)")
if low_var_cols:
    print(f"      Examples: {low_var_cols[:8]}")
print(f"    Remaining: {len(keep_var_cols)} features")

gc.collect()


# %% [markdown]
# ## 6.  Mutual Information feature selection (train only)

# %%
print("\n[6] Mutual Information feature selection …")

# Compute MI(feature, label) on train only — discrete_features=False (all float).
# MI is label-aware but is computed on TRAINING rows only, so no leakage.
X_train_var = train_df[keep_var_cols].values.astype(np.float32)
y_train     = train_df["label"].values.astype(int)

print(f"    Computing MI scores for {len(keep_var_cols)} features on "
      f"{len(train_df):,} train rows …")
mi_scores = mutual_info_classif(
    X_train_var, y_train,
    discrete_features=False,
    n_neighbors=5,
    random_state=SEED
)
del X_train_var; gc.collect()

mi_series = pd.Series(mi_scores, index=keep_var_cols).sort_values(ascending=False)
print(f"    MI score range: {mi_series.min():.4f} → {mi_series.max():.4f}")
print(f"    Top 20 features by MI:\n{mi_series.head(20).to_string()}")

# ── Group-aware top-K selection ───────────────────────────────────────────────
# Target: keep ~200 features, ensuring at least 2 per group (if group has > 2 cols)
TARGET_FEATURES = 200
MIN_PER_GROUP   = 2

# Collect group membership for kept_var_cols
col_to_group = {}
for grp, cols in FEATURE_GROUPS.items():
    for c in cols:
        if c in keep_var_cols:
            col_to_group[c] = grp

# Mandatory: top MIN_PER_GROUP per group from the MI-ranked list
mandatory = set()
for grp in FEATURE_GROUPS:
    grp_ranked = [c for c in mi_series.index
                  if col_to_group.get(c) == grp]
    for c in grp_ranked[:MIN_PER_GROUP]:
        mandatory.add(c)

# Fill up to TARGET_FEATURES with highest-MI features
selected = list(mandatory)
for c in mi_series.index:
    if len(selected) >= TARGET_FEATURES:
        break
    if c not in selected:
        selected.append(c)

# Sort selected by MI score (descending) for readability
selected = sorted(selected, key=lambda c: -mi_series.get(c, 0.0))

print(f"\n    Selected {len(selected)} features "
      f"(target={TARGET_FEATURES}, mandatory_min={MIN_PER_GROUP}/group)")

# Group breakdown of selected features
print("\n    Selected feature counts by group:")
grp_counts = {}
for c in selected:
    g = col_to_group.get(c, "Z_raw_passthrough")
    grp_counts[g] = grp_counts.get(g, 0) + 1
for g in sorted(grp_counts):
    print(f"      {g:35s}: {grp_counts[g]:3d}")

gc.collect()


# %% [markdown]
# ## 7.  RobustScaler (fit on train, apply to train+test)

# %%
print("\n[7] RobustScaler fit on train …")

scaler = RobustScaler(quantile_range=(5.0, 95.0))
scaler.fit(train_df[selected].values.astype(np.float32))

X_train_scaled = scaler.transform(train_df[selected].values.astype(np.float32))
X_test_scaled  = scaler.transform(test_df[selected].values.astype(np.float32))

# Clip scaled values to [-10, 10] to handle the rare extreme outliers that
# survive the 0.1% / 99.9% clip from Step 3 (Huber-style saturation)
X_train_scaled = np.clip(X_train_scaled, -10.0, 10.0)
X_test_scaled  = np.clip(X_test_scaled,  -10.0, 10.0)

print(f"    Train scaled: {X_train_scaled.shape}  "
      f"range [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
print(f"    Test  scaled: {X_test_scaled.shape}  "
      f"range [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")

# NaN check
print(f"    NaN in train_scaled: {np.isnan(X_train_scaled).sum()}")
print(f"    NaN in test_scaled:  {np.isnan(X_test_scaled).sum()}")

gc.collect()


# %% [markdown]
# ## 8.  Assemble processed_dataset.parquet

# %%
print("\n[8] Assembling processed_dataset.parquet …")

# Build combined DataFrame
train_proc = pd.DataFrame(X_train_scaled, columns=selected)
train_proc["address"]    = train_df["address"].values
train_proc["label"]      = train_df["label"].values
train_proc["label_name"] = train_df["label_name"].values
train_proc["split"]      = "train"

test_proc = pd.DataFrame(X_test_scaled, columns=selected)
test_proc["address"]    = test_df["address"].values
test_proc["label"]      = test_df["label"].values
test_proc["label_name"] = test_df["label_name"].values
test_proc["split"]      = "test"

processed_df = pd.concat([train_proc, test_proc], ignore_index=True)
del train_proc, test_proc, X_train_scaled, X_test_scaled; gc.collect()

print(f"    processed_dataset shape: {processed_df.shape}")
print(f"    Split counts: {processed_df['split'].value_counts().to_dict()}")

# Verify no NaN
nan_final = processed_df[selected].isna().sum().sum()
print(f"    NaN count in features: {nan_final}  ({'OK' if nan_final == 0 else 'ISSUE'})")

processed_df.to_parquet(OUT_DIR / "processed_dataset.parquet", index=False)
print(f"    Saved: processed_dataset.parquet")

gc.collect()


# %% [markdown]
# ## 9.  Save artefacts (selector state + scaler + split meta)

# %%
print("\n[9] Saving artefacts …")

# Scaler (RobustScaler fitted on train)
with open(OUT_DIR / "scaler.pkl", "wb") as fh:
    pickle.dump(scaler, fh)
print("    scaler.pkl saved")

# Selector state (not a sklearn object — save as dict with all info needed to
# reproduce the selection in NB6 / inference)
selector_state = {
    "variance_threshold": 0.01,
    "low_variance_dropped": low_var_cols,
    "mi_scores":  {c: float(mi_series[c]) for c in selected},
    "selected_features": selected,
    "target_features":   TARGET_FEATURES,
    "min_per_group":     MIN_PER_GROUP,
}
with open(OUT_DIR / "feature_selector.pkl", "wb") as fh:
    pickle.dump(selector_state, fh)
print("    feature_selector.pkl saved")

# Per-class temporal ranges (for paper / sanity checks)
per_class_ranges = {}
for cn in CLASS_NAMES:
    tr_ts = train_df.loc[train_df["label_name"] == cn, "first_ts"]
    te_ts = test_df.loc[test_df["label_name"] == cn, "first_ts"]
    per_class_ranges[cn] = {
        "train_first_ts_min": float(tr_ts.min()) if len(tr_ts) else None,
        "train_first_ts_max": float(tr_ts.max()) if len(tr_ts) else None,
        "test_first_ts_min":  float(te_ts.min()) if len(te_ts) else None,
        "test_first_ts_max":  float(te_ts.max()) if len(te_ts) else None,
    }

# Split meta (JSON — human-readable, used by NB6/NB7)
split_meta = {
    "n_train": n_train,
    "n_test":  n_test,
    "train_first_ts_min": float(train_df["first_ts"].min()),
    "train_first_ts_max": float(train_df["first_ts"].max()),
    "test_first_ts_min":  float(test_df["first_ts"].min()),
    "test_first_ts_max":  float(test_df["first_ts"].max()),
    "split_strategy":     "per_class_stratified_temporal_80_20",
    "per_class_temporal_ranges": per_class_ranges,
    "selected_features":  selected,
    "n_selected":         len(selected),
    "class_counts_train": train_df["label_name"].value_counts().to_dict(),
    "class_counts_test":  test_df["label_name"].value_counts().to_dict(),
    "scaler_type":        "RobustScaler(quantile_range=(5,95))",
    "class_medians_imputation": {k: float(v) for k, v in class_medians.items()},
    "leakage_notes": [
        "split: per-class stratified temporal 80/20 (each class independently "
        "sorted by first_ts, earliest 80% -> train, latest 20% -> test) — "
        "avoids global class-composition skew without losing data (Step 1)",
        "clip bounds fitted on train only (Step 3)",
        "class-conditional medians fitted on train only (Step 4)",
        "variance threshold fitted on train only (Step 5)",
        "MI scores computed on train only (Step 6)",
        "RobustScaler fitted on train only (Step 7)",
    ],
}
with open(OUT_DIR / "split_meta.json", "w") as fh:
    json.dump(split_meta, fh, indent=2)
print("    split_meta.json saved")

gc.collect()


# %% [markdown]
# ## 10.  Class weight computation for NB7

# %%
print("\n[10] Class weights for imbalanced training …")

# Compute inverse-frequency weights (NOT removing imbalance — weighting only)
label_counts = train_df["label_name"].value_counts()
total_train  = len(train_df)
n_classes    = len(label_counts)

class_weights = {}
for cn in CLASS_NAMES:
    cnt = label_counts.get(cn, 1)
    # sklearn-style balanced weight: n_samples / (n_classes * n_samples_for_class)
    class_weights[cn] = total_train / (n_classes * cnt)

print("    Class weights (balanced, inverse-frequency):")
for cn in CLASS_NAMES:
    cnt = label_counts.get(cn, 0)
    print(f"      {cn:22s}: count={cnt:5d}  weight={class_weights[cn]:.3f}")

# Save to split_meta for NB7
split_meta["class_weights"] = class_weights
split_meta["label_to_idx"]  = {cn: i for i, cn in enumerate(CLASS_NAMES)}
with open(OUT_DIR / "split_meta.json", "w") as fh:
    json.dump(split_meta, fh, indent=2)
print("    split_meta.json updated with class weights")


# %% [markdown]
# ## 11.  Visualisations (Q1-publication quality)

# %%
print("\n[11] Generating visualisations …")
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── VIS 1: MI score distribution (top 50 features) ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
top50 = mi_series[selected].head(50)
colors_mi = [CLASS_PALETTE.get(col_to_group.get(c, "").split("_")[-1], "#607D8B")
             for c in top50.index]
ax.barh(range(len(top50)), top50.values[::-1],
        color=list(reversed(colors_mi)), edgecolor="none")
ax.set_yticks(range(len(top50)))
ax.set_yticklabels(top50.index[::-1], fontsize=7)
ax.set_xlabel("Mutual Information Score")
ax.set_title("Top 50 Selected Features by MI Score (coloured by group prefix)")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis1_mi_top50.png", bbox_inches="tight")
plt.close()
print("  vis1 saved")

# ── VIS 2: MI score by feature group ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
grp_mi_means = {}
for grp, cols in FEATURE_GROUPS.items():
    in_sel = [c for c in cols if c in mi_series.index]
    if in_sel:
        grp_mi_means[grp.replace("_", " ")] = float(mi_series[in_sel].mean())

grp_sorted = sorted(grp_mi_means, key=grp_mi_means.get, reverse=True)
ax.barh(grp_sorted, [grp_mi_means[g] for g in grp_sorted], color="#5C6BC0")
ax.set_xlabel("Mean MI Score")
ax.set_title("Mean MI Score per Feature Group (train)")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis2_mi_by_group.png", bbox_inches="tight")
plt.close()
print("  vis2 saved")

# ── VIS 3: Temporal split timeline ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
for i, cn in enumerate(CLASS_NAMES):
    tr = train_df[train_df["label_name"] == cn]["first_ts"].values
    te = test_df[test_df["label_name"] == cn]["first_ts"].values
    ax.scatter(tr, np.full_like(tr, i, dtype=float), alpha=0.15, s=4,
               color=CLASS_PALETTE.get(cn, "grey"), label=cn if i == 0 else "")
    ax.scatter(te, np.full_like(te, i, dtype=float), alpha=0.3, s=4,
               color="red", marker="x")
ax.axvline(train_df["first_ts"].max(), color="black", linewidth=1.5,
           linestyle="--", label="train/test boundary")
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel("first_ts (Unix seconds)")
ax.set_title("Temporal Split: Train (●) vs Test (✗) by Class")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis3_temporal_split.png", bbox_inches="tight")
plt.close()
print("  vis3 saved")

# ── VIS 4: Class distribution in train vs test ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split_name, split_df) in zip(axes, [("Train", train_df), ("Test", test_df)]):
    vc = split_df["label_name"].value_counts().reindex(CLASS_NAMES).fillna(0)
    bars = ax.barh(CLASS_NAMES, vc.values,
                   color=[CLASS_PALETTE.get(c, "grey") for c in CLASS_NAMES])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(f"Class Distribution — {split_name} ({len(split_df):,} samples)")
    ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis4_class_distribution_splits.png", bbox_inches="tight")
plt.close()
print("  vis4 saved")

# ── VIS 5: Feature selection waterfall (cumulative MI vs feature rank) ────────
fig, ax = plt.subplots(figsize=(10, 4))
cum_mi = mi_series[selected].cumsum() / mi_series[selected].sum()
ax.plot(range(1, len(cum_mi) + 1), cum_mi.values, color="#1976D2", linewidth=1.5)
ax.axvline(TARGET_FEATURES, color="red", linestyle="--", linewidth=1,
           label=f"target={TARGET_FEATURES}")
ax.axhline(0.90, color="green", linestyle=":", linewidth=1, label="90% MI")
ax.axhline(0.95, color="orange", linestyle=":", linewidth=1, label="95% MI")
ax.set_xlabel("Feature rank (by MI score)")
ax.set_ylabel("Cumulative MI (fraction of total)")
ax.set_title("Cumulative MI Coverage vs Feature Count (selected features)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis5_mi_cumulative.png", bbox_inches="tight")
plt.close()
print("  vis5 saved")

# ── VIS 6: Scaled feature distribution for top-10 features by class ──────────
top10_feats = list(mi_series[selected].head(10).index)
nf = len(top10_feats)
ncols = 2; nrows = (nf + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.5))
axes = axes.flatten()

scaled_train_df = pd.DataFrame(
    scaler.transform(train_df[selected].values.astype(np.float32)),
    columns=selected
)
scaled_train_df["label_name"] = train_df["label_name"].values

for i, feat in enumerate(top10_feats):
    ax = axes[i]
    for cn in CLASS_NAMES:
        sub = scaled_train_df[scaled_train_df["label_name"] == cn][feat].dropna()
        if len(sub) < 5: continue
        sub = sub.clip(-5, 5)
        ax.hist(sub, bins=30, density=True, alpha=0.5, label=cn,
                color=CLASS_PALETTE.get(cn, "grey"), histtype="stepfilled")
    ax.set_title(feat, fontsize=8)
    ax.set_xlabel("scaled value")
    if i == 0: ax.legend(fontsize=6, framealpha=0.4)

for j in range(i + 1, len(axes)): axes[j].set_visible(False)
del scaled_train_df; gc.collect()
fig.suptitle("Top-10 MI Features — Scaled Distribution by Class (train)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis6_top10_scaled_dist.png", bbox_inches="tight")
plt.close()
print("  vis6 saved")

# ── VIS 7: Scaler centre + scale for top-30 features ─────────────────────────
top30 = list(mi_series[selected].head(30).index)
centers = scaler.center_[:len(selected)]
scales  = scaler.scale_[:len(selected)]
top30_idx = [selected.index(c) for c in top30 if c in selected]
top30_feats_found = [selected[i] for i in top30_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(range(len(top30_feats_found)),
             [centers[i] for i in top30_idx], color="#FF7043")
axes[0].set_yticks(range(len(top30_feats_found)))
axes[0].set_yticklabels(top30_feats_found, fontsize=7)
axes[0].set_xlabel("Median (RobustScaler centre)")
axes[0].set_title("Scaler centres — Top 30 MI features")

axes[1].barh(range(len(top30_feats_found)),
             [scales[i] for i in top30_idx], color="#42A5F5")
axes[1].set_yticks(range(len(top30_feats_found)))
axes[1].set_yticklabels(top30_feats_found, fontsize=7)
axes[1].set_xlabel("IQR (RobustScaler scale)")
axes[1].set_title("Scaler scales — Top 30 MI features")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis7_scaler_params.png", bbox_inches="tight")
plt.close()
print("  vis7 saved")

# ── VIS 8: Variance of selected features on train vs test ────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
var_train = train_df[selected].var()
var_test  = test_df[selected].var()
ax.scatter(var_train.values, var_test.values, alpha=0.5, s=12, color="#5C6BC0")
_m = max(var_train.max(), var_test.max())
ax.plot([0, _m], [0, _m], "r--", linewidth=0.8, label="y=x")
ax.set_xlabel("Feature variance (train)")
ax.set_ylabel("Feature variance (test)")
ax.set_title("Feature Variance: Train vs Test (no distribution shift → near diagonal)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis8_variance_train_vs_test.png", bbox_inches="tight")
plt.close()
print("  vis8 saved")

# ── VIS 9: Class weight bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
wt_labels = list(class_weights.keys())
wt_vals   = [class_weights[cn] for cn in wt_labels]
bars = ax.bar(wt_labels, wt_vals,
              color=[CLASS_PALETTE.get(cn, "grey") for cn in wt_labels])
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.set_ylabel("Class weight")
ax.set_title("Balanced Class Weights for NB7 Training (inverse frequency)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis9_class_weights.png", bbox_inches="tight")
plt.close()
print("  vis9 saved")

# ── VIS 10: Correlation heatmap of top-20 selected features ──────────────────
top20 = list(mi_series[selected].head(20).index)
top20 = [c for c in top20 if c in train_df.columns]
corr20 = train_df[top20].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr20, dtype=bool))
sns.heatmap(corr20, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation Matrix — Top 20 Selected Features (train)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis10_top20_correlation.png", bbox_inches="tight")
plt.close()
print("  vis10 saved")


# %% [markdown]
# ## 12.  Final summary diagnostics

# %%
print("\n" + "=" * 80)
print("NOTEBOOK 5 — FEATURE SELECTION & PREPROCESSING COMPLETE")
print("=" * 80)

print(f"""
  processed_dataset.parquet  : {processed_df.shape}
  feature_selector.pkl       : {len(selected)} selected features
  scaler.pkl                 : RobustScaler(quantile_range=(5,95))
  split_meta.json            : n_train={n_train}, n_test={n_test}
  Figures                    : {len(list(FIG_DIR.glob('*.png')))} PNGs in {FIG_DIR}
""")

print(f"""
PIPELINE STEPS SUMMARY
─────────────────────────────────────────────────────────────────────────────
  Step 1  Per-class stratified temporal 80/20 split: train={n_train:,}  test={n_test:,}
  Step 2  Clip bounds refit on train
  Step 3  Class-conditional median imputation for days_since_last_activity
  Step 4  VarianceThreshold(0.01): dropped {len(low_var_cols)} cols → {len(keep_var_cols)} remain
  Step 5  MI selection: {len(selected)} / {len(keep_var_cols)} features retained (target={TARGET_FEATURES})
  Step 6  RobustScaler(quantile_range=(5,95)) fit on train

LEAKAGE CHECKLIST (all PASS)
─────────────────────────────────────────────────────────────────────────────
  ✓ Clip bounds: fit on train only
  ✓ Class-conditional medians: fit on train only
  ✓ Variance threshold: fit on train only
  ✓ MI scores: computed on train only
  ✓ RobustScaler: fit on train only
  ✓ behavioral_scores.parquet: NOT merged (kept separate for ablation)
  ✓ Class imbalance: NOT removed (class weights saved for NB7)
  ✓ Temporal ordering: preserved PER CLASS (each class's earliest 80% ->
    train, latest 20% -> test) — eliminates global class-composition skew
    without losing any addresses

SELECTED FEATURE GROUPS
─────────────────────────────────────────────────────────────────────────────""")

for grp in sorted(grp_counts):
    print(f"  {grp:35s}: {grp_counts[grp]:3d} features")

print(f"""
─────────────────────────────────────────────────────────────────────────────
PRECONDITIONS FOR NB6 (GRAPH CONSTRUCTION)
─────────────────────────────────────────────────────────────────────────────
 1. processed_dataset.parquet has columns: address, label, label_name, split,
    + {len(selected)} scaled model features. Do NOT re-scale in NB6.

 2. Use split column to separate train/test graph nodes. Do NOT leak test
    node features into train graph convolutions (use train subgraph for
    message passing; test nodes attend to train neighbors via inductive GNN).

 3. node_type_map.parquet: use for HetGNN edge-relation routing (wallet vs
    contract node types). Available at {FEAT_DIR}/node_type_map.parquet.

 4. edges_engineered.parquet: 34M edge rows with 24 edge-level features.
    NB6 builds the heterogeneous graph from these 5 edge types:
    normal / internal / erc20 / erc721 / erc1155.

 5. Class weights in split_meta.json['class_weights']: pass to NB7 model
    loss function to handle the 20:1 benign:minority imbalance.

 6. feature_selector.pkl + scaler.pkl: save for inference pipeline.
─────────────────────────────────────────────────────────────────────────────
OUTPUT FILES:
  {OUT_DIR}/processed_dataset.parquet
  {OUT_DIR}/feature_selector.pkl
  {OUT_DIR}/scaler.pkl
  {OUT_DIR}/split_meta.json
  {FIG_DIR}/*.png  (10 publication-quality figures)
─────────────────────────────────────────────────────────────────────────────
[done] All outputs ready for Notebook 6 (Graph Construction).
""")
