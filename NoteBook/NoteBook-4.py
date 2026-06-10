# %% [markdown]
# # Notebook 4 — Feature Engineering  (Q1-Grade, Temporal-HetGNN Ready)
#
# **Inputs** (from Notebook 3 — Feature Extraction):
# | File | Description |
# |---|---|
# | `features_raw.parquet`       | 224 raw numeric features per address |
# | `behavioral_scores.parquet`  | 8 domain-heuristic risk scores (ablation only) |
# | `edges_raw.parquet`          | 34 M+ typed edges (normal/internal/erc20/erc721/erc1155) |
# | `temporal_sequences.parquet` | time-aligned per-address event sequences |
#
# **Outputs**:
# | File | Description |
# |---|---|
# | `extracted_dataset.parquet`   | Final node feature matrix (~350+ features, GNN-ready) |
# | `edges_engineered.parquet`    | Edge table with engineered edge features |
# | `node_type_map.parquet`       | wallet / contract node-type labels |
# | `fe_summary.json`             | feature group inventory for downstream use |
#
# ──────────────────────────────────────────────────────────────────────────────
# ## ENGINEERING PHILOSOPHY
# ──────────────────────────────────────────────────────────────────────────────
#
#  GROUP A — Obligatory pre-processing (log transforms, imputation, scaling)
#           Must happen before ANY model ingests features_raw. Required by NB3.
#
#  GROUP B — Value-flow engineering
#           ETH recycling, wash-trade proxies, net-flow ratios, value entropy.
#
#  GROUP C — Gas-based engineering
#           Gas utilization, cost-per-value, gas price volatility fingerprint,
#           miner tip proxy, gas-limit padding ratio.
#
#  GROUP D — Temporal velocity & acceleration
#           Per-window tx rates, velocity ratio early/late, tx acceleration,
#           periodicity score, session gap analysis.
#
#  GROUP E — Token / multi-asset fingerprint
#           Token dominance, ERC mix entropy, NFT flip ratio,
#           multi-asset Gini, cross-protocol activity.
#
#  GROUP F — Method-selector semantic clusters
#           DeFi score, admin/ownership score, Flash/MEV composite score,
#           NFT marketplace score, cross-protocol diversity.
#
#  GROUP G — Ego-network graph features (from edges_raw)
#           Weighted in/out degree, edge-type entropy per node,
#           unique neighbor ratio, ETH-weighted PageRank (approximate),
#           reciprocity, temporal edge density.
#           NOTE: No neighbor-label-based features here — using any label
#           column from edges_raw before train/test split is data leakage.
#
#  GROUP H — Cross-feature polynomial interactions (domain-motivated)
#           Selected non-linear terms with clear economic interpretation.
#
#  GROUP I — Heterogeneous node-type enrichment
#           Contract vs wallet distinction, bytecode-size tier,
#           self-deploy flag, proxy pattern likelihood.
#
#  GROUP J — Edge feature engineering
#           Edge-level: temporal decay weight, edge value tier,
#           method-semantic tag, edge directionality flag.
#
# ── LEAKAGE GUARDRAILS ──────────────────────────────────────────────────────
#  • No label-derived statistics are computed over the full dataset.
#  • behavioral_scores.parquet is NOT joined into the feature matrix.
#  • first_ts / last_ts are PRESERVED (NB5 needs them for window assignment)
#    but are FLAGGED so NB5 can drop them post-cohort-assignment.
#  • Imputation uses GLOBAL median only — class-conditional median would
#    encode label information into features before the train/test split.
#  • Outlier clipping quantiles are computed on the full pre-split dataset.
#    NB5 must recompute clip bounds on training folds only and apply to test.
#
# ── MEMORY BUDGET (Kaggle T4, 16 GB RAM) ────────────────────────────────────
#  edges_raw.parquet ≈ 34 M rows × 16 cols (~11.5 GB on load as object strings).
#  We category-encode ONLY low-cardinality cols (edge_type/anchor_address/
#  src_label_name/method_id) — source/target are skipped (millions of unique
#  values; category overhead > object there). No astype(str).str.lower() —
#  NB3 already stores addresses lowercase; that call created 2 × 3.4 GB peaks.
#  Edge-type sub-frames (e_erc20, e_int) are freed after G1; e_norm after G4.
#  We never materialize a dense adjacency matrix.

# %%
import gc, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import periodogram
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm.auto import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
# NB3 saves all outputs to /kaggle/working/features/
# NB4 reads from there and writes engineered outputs alongside them.
FEAT_DIR = Path("/kaggle/working/features")          # NB3 output dir (read)
OUT_DIR  = Path("/kaggle/working/features")          # write engineered files to same dir
FIG_DIR  = Path("/kaggle/working/figures_nb4")
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

print("=== Notebook 4 — Feature Engineering ===")
print(f"Input  : {FEAT_DIR}")
print(f"Output : {OUT_DIR}")


# %% [markdown]
# ## 1.  Load raw features & edges

# %%
print("\n[1] Loading features_raw.parquet …")
df = pd.read_parquet(FEAT_DIR / "features_raw.parquet")
print(f"    shape: {df.shape}")
print(f"    columns: {df.shape[1]}  |  addresses: {len(df)}")
print(df["label_name"].value_counts().to_string())

print("\n[1] Loading edges_raw.parquet …")
edges = pd.read_parquet(FEAT_DIR / "edges_raw.parquet")
print(f"    edges shape: {edges.shape}")
print(edges["edge_type"].value_counts().to_string())

# Cast types up front for memory efficiency
edges["timestamp"]    = edges["timestamp"].astype(np.int64)
edges["block_number"] = edges["block_number"].astype(np.int64)
edges["value_eth"]    = edges["value_eth"].astype(np.float32)
edges["gas_used"]     = edges["gas_used"].astype(np.float32)
gc.collect()

# NB3 stores all Ethereum addresses lowercase from etherscan — skip astype(str).str.lower().
# That call created two 3.4 GB intermediates per column (astype copy + lower copy),
# causing OOM during the target column pass (edges ~12.9 GB + 3.4 + 3.4 = ~19.7 GB).

# Category-encode ONLY low-cardinality columns.
# source / target: millions of unique values — category codes + dict > object dtype.
# anchor_address:  42892 unique / 34M rows → ~20× saving (~278 MB → ~140 MB).
# edge_type:       5 unique → ~40× saving.
for _c in ["edge_type", "src_label_name", "anchor_address", "method_id"]:
    if _c in edges.columns:
        edges[_c] = edges[_c].astype("category")
        gc.collect()
print(f"    edges memory after category cast: "
      f"{edges.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# ── Restore metadata cols that master_dataset.csv provides ──────────────────
# NB3 drops all-zero constant columns. If master_dataset.csv was absent,
# is_contract / bytecode_size / balance_eth are all 0 and get dropped.
# Re-inject here so Groups H and I can run without KeyError.
# Zero is the semantically correct default: "unknown → treat as wallet with no ETH."
_meta_defaults = {"is_contract": 0, "bytecode_size": 0.0, "balance_eth": 0.0}
for _col, _default in _meta_defaults.items():
    if _col not in df.columns:
        df[_col] = _default
        print(f"  [warn] '{_col}' absent from features_raw "
              f"(master_dataset.csv not found in NB3) — defaulted to {_default}")

# %% [markdown]
# ## 2.  GROUP A — Obligatory pre-processing

# %%
print("\n[2] GROUP A: log1p transforms + imputation …")

# ── A1. Log-transform heavy-tailed columns ───────────────────────────────────
LOG_COLS = [
    "val_out_total_eth", "val_in_total_eth",
    "gas_used_total", "gas_eth_total",
    "bytecode_size", "balance_eth",
    "tx_count", "erc20_tx_count", "erc721_tx_count",
    "internal_tx_count", "unique_blocks",
    # value stats that span many orders of magnitude
    "val_out_mean", "val_out_max",
    "val_in_mean",  "val_in_max",
    "gas_used_mean", "gas_used_max",
    "int_val_out_total_eth", "int_val_in_total_eth",
    # block-density
    "tx_per_block_max", "heavy_block_count",
    # ego-network counts
    "unique_counterparts", "unique_receivers", "unique_senders",
    "approval_count", "erc20_unique_tokens",
    "erc721_unique_collections", "erc721_unique_token_ids",
    "erc1155_unique_tokens",
]
LOG_COLS = [c for c in LOG_COLS if c in df.columns]

for c in LOG_COLS:
    df[f"log1p_{c}"] = np.log1p(np.abs(df[c].fillna(0).astype(np.float64)))

print(f"    log1p applied to {len(LOG_COLS)} columns → {len(LOG_COLS)} new log_ cols")

# ── A2. Net-value absolute log ───────────────────────────────────────────────
if "val_net_eth" in df.columns:
    df["log1p_abs_val_net_eth"] = np.log1p(np.abs(df["val_net_eth"].fillna(0)))

# ── A3. Impute days_since_last_activity ──────────────────────────────────────
# LEAKAGE NOTE: Class-conditional median imputation would encode the label
# into the feature before train/test split. We use GLOBAL median only.
# Class-aware imputation must be done in NB5 strictly on the training fold.
df["has_collection_ts"] = (~df["days_since_last_activity"].isna()).astype(np.float32)
global_median_dsla = df["days_since_last_activity"].median()
df["days_since_last_activity"] = df["days_since_last_activity"].fillna(global_median_dsla)
print(f"    days_since_last_activity NaN remaining: "
      f"{df['days_since_last_activity'].isna().sum()}")
print(f"    Imputed with global median: {global_median_dsla:.4f}  (class-median deferred to NB5)")

# ── A4. Flag columns that are cohort markers (NB5 must drop after windowing) ──
COHORT_MARKER_COLS = ["first_ts", "last_ts"]   # do NOT model on these
print(f"    Cohort-marker cols (drop in NB5 post-window): {COHORT_MARKER_COLS}")

gc.collect()


# %% [markdown]
# ## 3.  GROUP B — Value-flow engineering

# %%
print("\n[3] GROUP B: value-flow engineering …")

eps = 1e-9

# B1. Per-transaction value rates
df["val_out_per_tx"]    = df["val_out_total_eth"] / (df["n_out_tx"].clip(lower=1))
df["val_in_per_tx"]     = df["val_in_total_eth"]  / (df["n_in_tx"].clip(lower=1))
df["val_total_per_tx"]  = (df["val_out_total_eth"] + df["val_in_total_eth"]) / df["tx_count"].clip(lower=1)

# B2. ETH recycling / round-trip proxy
#   High recycling = funds move in ≈ same amount they move out (wash trading)
df["eth_recycling_ratio"] = (
    df[["val_out_total_eth", "val_in_total_eth"]].min(axis=1) /
    (df[["val_out_total_eth", "val_in_total_eth"]].max(axis=1) + eps)
)

# B3. Directional flow imbalance (−1 = pure sink, +1 = pure source)
total_flow = df["val_out_total_eth"] + df["val_in_total_eth"]
df["flow_imbalance"] = (df["val_in_total_eth"] - df["val_out_total_eth"]) / (total_flow + eps)

# B4. Internal vs normal ETH ratio
#   High ratio → funds moved through internal calls (MEV / exploit pattern)
total_eth_moved = (df["val_out_total_eth"] + df["val_in_total_eth"]
                   + df["int_val_out_total_eth"] + df["int_val_in_total_eth"])
df["internal_eth_ratio"] = (
    (df["int_val_out_total_eth"] + df["int_val_in_total_eth"]) /
    (total_eth_moved + eps)
)

# B5. Value acceleration: early vs late flow
#   front-loaded drain = phishing / rug pull
df["val_front_load_score"] = df["tx_first_7d_ratio"] * df["lifecycle_decline"].clip(lower=0)

# B6. Value concentration — symmetric max across gini_out / gini_in if available
if "val_gini_out" in df.columns and "val_gini_in" in df.columns:
    df["val_gini_max"] = df[["val_gini_out", "val_gini_in"]].max(axis=1)
elif "val_gini" in df.columns:
    df["val_gini_max"] = df["val_gini"]
else:
    df["val_gini_max"] = 0.0

# B7. Ponzi funnel: many senders, few receivers with high inflow
df["ponzi_funnel_ratio"] = (
    df["unique_senders"] / (df["unique_receivers"].clip(lower=1)) *
    df["val_in_total_eth"].clip(lower=0)
)
df["log1p_ponzi_funnel_ratio"] = np.log1p(df["ponzi_funnel_ratio"].clip(lower=0))

# B8. ETH throughput efficiency (signal of wash trading or routing)
df["eth_throughput"] = total_eth_moved / (df["lifespan_days"].clip(lower=1/1440))
df["log1p_eth_throughput"] = np.log1p(df["eth_throughput"].clip(lower=0))

print(f"    GROUP B features added.")


# %% [markdown]
# ## 4.  GROUP C — Gas-based engineering

# %%
print("\n[4] GROUP C: gas engineering …")

# C1. Gas utilization = gas used / gas limit (efficiency / padding)
#   Padded limits → honeypot or anti-analysis patterns
df["gas_utilization"] = df["gas_used_mean"] / (df["gas_limit_mean"].clip(lower=1))

# C2. Gas cost per ETH moved
#   Very high → spam tx; very low → batch/internal MEV
df["gas_cost_per_eth"] = df["gas_eth_total"] / (total_flow + eps)
df["log1p_gas_cost_per_eth"] = np.log1p(df["gas_cost_per_eth"].clip(lower=0))

# C3. Gas price volatility coefficient of variation
#   Constant gas price → bot; variable → human or multi-round MEV
df["gas_price_cv"] = (df["gas_price_gwei_std"] /
                      (df["gas_price_gwei_mean"].clip(lower=eps)))

# C4. Gas price skew (MEV bots pay extremely high spikes)
# p99/mean ratio — captures spike behaviour
df["gas_price_spike_ratio"] = (df["gas_price_gwei_p99"] /
                                (df["gas_price_gwei_mean"].clip(lower=eps)))

# C5. Gas padding ratio: limit far above used → honeypot / decoy contract
df["gas_padding_ratio"] = ((df["gas_limit_mean"] - df["gas_used_mean"]).clip(lower=0) /
                            (df["gas_limit_mean"].clip(lower=1)))

# C6. High-gas-price burst indicator
#   Fraction of tx with gas price > 2× mean (bursty bidding = MEV)
# Approximated from p90 vs mean since we have stats not raw lists
df["high_gas_burst_proxy"] = (df["gas_price_gwei_p90"] /
                               (df["gas_price_gwei_mean"].clip(lower=eps))) - 1.0
df["high_gas_burst_proxy"] = df["high_gas_burst_proxy"].clip(lower=0)

# C7. Miner tip proxy (EIP-1559 base fee removed from gas_price in practice)
#   Use gas_price_gwei_min as a rough base fee proxy
df["miner_tip_proxy_gwei"] = (df["gas_price_gwei_mean"] -
                               df["gas_price_gwei_min"].clip(lower=0)).clip(lower=0)
df["log1p_miner_tip"] = np.log1p(df["miner_tip_proxy_gwei"])

print(f"    GROUP C features added.")


# %% [markdown]
# ## 5.  GROUP D — Temporal velocity & acceleration

# %%
print("\n[5] GROUP D: temporal velocity & acceleration …")

# D1. Tx velocity in early windows (tx/day)
df["tx_vel_7d"]   = df["tx_first_7d"]  / 7.0
df["tx_vel_30d"]  = df["tx_first_30d"] / 30.0

# D2. Velocity ratio: early rate vs average lifetime rate
#   >> 1 → front-loaded (phishing / rug pull drain)
#   << 1 → ramp-up (normal growth or ponzi accumulation)
df["velocity_ratio_7d"] = df["tx_vel_7d"] / (df["tx_per_day"].clip(lower=eps))

# D3. Tx acceleration: rate change over lifecycle
#   (late 30d rate - early 30d rate) / lifetime rate — positive = ramping up
df["tx_accel"] = (
    (df["lifecycle_wL_share"] - df["lifecycle_w1_share"]) /
    (df["lifecycle_w1_share"] + df["lifecycle_wL_share"] + eps)
)

# D4. Dormancy ratio: time silent / lifespan
#   High for honeypots (deployed and silent) and old exploits
df["active_fraction"]   = df["activity_ratio"]   # alias for clarity
df["dormancy_fraction"] = 1.0 - df["activity_ratio"]
df["dormancy_score"]    = df["days_since_last_activity"] / (df["lifespan_days"].clip(lower=1))
df["log1p_dormancy_score"] = np.log1p(df["dormancy_score"].clip(lower=0))

# D5. Session burst: ratio of tx in shortest 10% of active days
#   High → burst attack pattern
df["burst_concentration"] = df["tx_first_1d_ratio"] * df["burstiness"].clip(lower=0)

# D6. Hurst × burstiness interaction
#   persistent AND bursty → MEV / exploit bots with memory
df["hurst_x_burst"] = df["hurst_dfa"] * df["burstiness"].clip(lower=-1)

# D7. Temporal entropy product
#   Low both → automated, fixed schedule
df["temporal_entropy_product"] = df["hour_entropy"] * df["wday_entropy"]

# D8. IAT regularity score (inverse CV of IATs — high = metronomic bot)
df["iat_regularity"] = 1.0 / (1.0 + df["iat_std"] / (df["iat_mean"].clip(lower=eps)))

# D9. Rapid lifecycle flag (entire activity within < 1 hour)
df["rapid_lifecycle_flag"] = (df["lifespan_days"] < 1/24).astype(np.float32)

# D10. Lifespan tiers (ordinally encode, no leakage)
df["lifespan_tier"] = pd.cut(
    df["lifespan_days"].clip(lower=0),
    bins=[-1, 1/1440, 1/24, 1, 7, 30, 365, np.inf],
    labels=[0, 1, 2, 3, 4, 5, 6]
).astype(float)

print(f"    GROUP D features added.")


# %% [markdown]
# ## 6.  GROUP E — Token / multi-asset fingerprint

# %%
print("\n[6] GROUP E: token & multi-asset features …")

# E1. Token activity dominance: how much activity is token-based vs ETH?
token_tx = df["erc20_tx_count"] + df["erc721_tx_count"] + df["erc1155_tx_count"]
total_tx  = df["tx_count"] + token_tx
df["token_dominance"]   = token_tx / (total_tx + eps)
df["eth_only_flag"]     = ((token_tx == 0) & (df["tx_count"] > 0)).astype(np.float32)
df["multi_asset_flag"]  = (token_tx > 0).astype(np.float32)

# E2. ERC type mix entropy — vectorized for Kaggle T4 performance
erc_counts = df[["erc20_tx_count", "erc721_tx_count", "erc1155_tx_count"]].values.astype(np.float64)
erc_sums   = erc_counts.sum(axis=1, keepdims=True)
# avoid division by zero: rows with no token tx get entropy 0
has_tokens = (erc_sums.ravel() > 0)
p_erc = np.where(erc_sums > 0, erc_counts / np.maximum(erc_sums, 1e-12), 0.0)
# entropy: only over positive probabilities
log_p = np.where(p_erc > 0, np.log(p_erc + 1e-12), 0.0)
df["erc_type_entropy"] = (-np.sum(p_erc * log_p, axis=1) * has_tokens).astype(np.float32)

# E3. ERC-20 flow imbalance: high out/in ratio = drain wallet (phishing)
df["erc20_flow_imbalance"] = (
    (df["erc20_out_count"] - df["erc20_in_count"]) /
    (df["erc20_out_count"] + df["erc20_in_count"] + eps)
)

# E4. NFT flip ratio: selling more than buying = NFT wash trading
df["erc721_flip_ratio"] = (
    (df["erc721_out_count"] - df["erc721_in_count"]) /
    (df["erc721_out_count"] + df["erc721_in_count"] + eps)
)

# E5. Token diversity relative to activity level
df["token_per_tx"] = token_tx / (df["tx_count"].clip(lower=1))

# E6. ERC20 contract breadth (unique tokens per erc20 tx)
df["erc20_contract_breadth"] = (
    df["erc20_unique_tokens"] / (df["erc20_tx_count"].clip(lower=1))
)

# E7. ERC721 collection entropy
df["erc721_collection_entropy"] = np.log1p(df["erc721_unique_collections"])

# E8. Cross-protocol activity: addresses that touch ERC20 AND normal AND internal
df["cross_protocol_score"] = (
    (df["erc20_tx_count"] > 0).astype(int) +
    (df["internal_tx_count"] > 0).astype(int) +
    (df["erc721_tx_count"] > 0).astype(int) +
    (df["erc1155_tx_count"] > 0).astype(int)
) / 4.0

print(f"    GROUP E features added.")


# %% [markdown]
# ## 7.  GROUP F — Method-selector semantic clusters

# %%
print("\n[7] GROUP F: method-selector semantic clusters …")

# ── Semantic cluster definitions (domain-motivated, no leakage) ─────────────
DEFI_SELECTORS    = ["sel_swap_exact_t4t_share", "sel_swap_exact_eth4t_share",
                     "sel_swap_exact_t4eth_share", "sel_erc20_transfer_share",
                     "sel_erc20_transferfrom_share", "sel_erc20_approve_share"]
ADMIN_SELECTORS   = ["sel_transfer_ownership_share", "sel_renounce_ownership_share",
                     "sel_owner_call_share", "sel_mint_share", "sel_burn_share"]
APPROVAL_SELECTORS= ["sel_erc20_approve_share", "sel_set_approval_for_all_share"]
NFT_SELECTORS     = ["sel_erc721_safe_xfer_share", "sel_erc721_safe_xfer_d_share",
                     "sel_set_approval_for_all_share"]
PROXY_SELECTORS   = ["sel_selfdestruct_proxy_share", "sel_multicall_share",
                     "sel_execute_share"]

def cluster_score(df, selector_list):
    cols = [c for c in selector_list if c in df.columns]
    if not cols: return pd.Series(0.0, index=df.index)
    return df[cols].clip(lower=0).sum(axis=1)

df["defi_score"]          = cluster_score(df, DEFI_SELECTORS)
df["admin_score"]         = cluster_score(df, ADMIN_SELECTORS)
df["approval_cluster"]    = cluster_score(df, APPROVAL_SELECTORS)
df["nft_score"]           = cluster_score(df, NFT_SELECTORS)
df["proxy_score"]         = cluster_score(df, PROXY_SELECTORS)

# F2. Flash-loan/MEV composite from selector + block-density signals
df["flash_mev_composite"] = (
    df["defi_score"].clip(0, 1) +
    df["not_in_shortlist_ratio"] * 0.5 +
    df["tx_per_block_max"].clip(0, 10) / 10.0 +
    df["tx_index_low_ratio"] * 2.0
)

# F3. Rug pull composite: mint + renounce ownership + front-loaded lifecycle
df["rug_composite"] = (
    df.get("sel_mint_share", pd.Series(0.0, index=df.index)).clip(0, 1) * 2.0 +
    df.get("sel_renounce_ownership_share", pd.Series(0.0, index=df.index)).clip(0, 1) * 2.0 +
    df["lifecycle_decline"].clip(lower=0) * 1.5
)

# F4. Method entropy × novel ratio: high both = fully custom attack tooling
df["novel_method_entropy"] = df["method_entropy"] * df["not_in_shortlist_ratio"]

# F5. Cross-protocol method diversity
df["method_cluster_count"] = (
    (df["defi_score"] > 0).astype(int) +
    (df["admin_score"] > 0).astype(int) +
    (df["nft_score"]   > 0).astype(int) +
    (df["proxy_score"] > 0).astype(int)
)

print(f"    GROUP F features added.")


# %% [markdown]
# ## 8.  GROUP G — Ego-network graph features from edges_raw

# %%
print("\n[8] GROUP G: graph ego-network features from edges_raw …")
print("    Computing per-address aggregations (may take 1-2 min) …")

# ── G0. Separate edge tables by type ────────────────────────────────────────
# Copy only the 4 columns needed for G1 degree aggregations and G3/G4.
# tx_hash alone is ~2.2 GB on 34M rows; skipping it here saves ~1.8 GB peak.
_g_cols = [c for c in ["source", "target", "value_eth", "gas_used"] if c in edges.columns]
e_norm  = edges.loc[edges["edge_type"] == "normal",   _g_cols].copy()
e_erc20 = edges.loc[edges["edge_type"] == "erc20",    _g_cols].copy()
e_int   = edges.loc[edges["edge_type"] == "internal", _g_cols].copy()
print(f"    e_norm: {len(e_norm):,}  e_erc20: {len(e_erc20):,}  e_int: {len(e_int):,}")

# ── G1. Weighted degree features ────────────────────────────────────────────
print("  [G1] weighted degree …")

def agg_degree(edge_df, addr_col, side, suffix):
    """Aggregate value/count/gas for one side of edges."""
    grp = edge_df.groupby(addr_col).agg(
        **{
            f"{side}_deg_{suffix}":       ("value_eth",    "count"),
            f"{side}_val_{suffix}":       ("value_eth",    "sum"),
            f"{side}_val_max_{suffix}":   ("value_eth",    "max"),
            f"{side}_neighbors_{suffix}": ("target" if addr_col == "source" else "source", "nunique"),
            f"{side}_gas_{suffix}":       ("gas_used",     "sum"),
        }
    ).reset_index().rename(columns={addr_col: "address"})
    return grp

# Normal tx degree
out_norm = agg_degree(e_norm, "source", "out", "norm")
in_norm  = agg_degree(e_norm, "target", "in",  "norm")

# ERC20 degree
out_erc  = agg_degree(e_erc20, "source", "out", "erc20")
in_erc   = agg_degree(e_erc20, "target", "in",  "erc20")

# Internal degree
out_int  = agg_degree(e_int, "source", "out", "int")
in_int   = agg_degree(e_int, "target", "in",  "int")

# Merge into graph_df
graph_df = df[["address"]].copy()
for sub_df in [out_norm, in_norm, out_erc, in_erc, out_int, in_int]:
    graph_df = graph_df.merge(sub_df, on="address", how="left")
graph_df = graph_df.fillna(0.0)
del out_norm, in_norm, out_erc, in_erc, out_int, in_int; gc.collect()
# e_erc20 and e_int are only used in G1 — free them now; e_norm still needed for G3 and G4
del e_erc20, e_int; gc.collect()

# G1a. Total weighted degree (across all edge types)
graph_df["total_in_deg"]  = (
    graph_df["in_deg_norm"].fillna(0)  +
    graph_df["in_deg_erc20"].fillna(0) +
    graph_df["in_deg_int"].fillna(0)
)
graph_df["total_out_deg"] = (
    graph_df["out_deg_norm"].fillna(0)  +
    graph_df["out_deg_erc20"].fillna(0) +
    graph_df["out_deg_int"].fillna(0)
)
graph_df["degree_ratio"]  = (
    graph_df["total_in_deg"] / (graph_df["total_out_deg"] + eps)
)
graph_df["total_degree"]  = graph_df["total_in_deg"] + graph_df["total_out_deg"]

# ── G2. Edge-type entropy per node — vectorized ──────────────────────────────
print("  [G2] edge-type entropy per anchor node …")
edge_type_counts = (edges.groupby(["anchor_address", "edge_type"])
                         .size()
                         .unstack(fill_value=0))
edge_type_counts.columns = [f"etype_{c}" for c in edge_type_counts.columns]
edge_type_counts = edge_type_counts.reset_index().rename(
    columns={"anchor_address": "address"}
)

etype_cols = [c for c in edge_type_counts.columns if c.startswith("etype_")]
# Vectorized entropy computation
et_vals = edge_type_counts[etype_cols].values.astype(np.float64)
et_sums = et_vals.sum(axis=1, keepdims=True)
et_p    = np.where(et_sums > 0, et_vals / np.maximum(et_sums, 1e-12), 0.0)
et_logp = np.where(et_p > 0, np.log(et_p + 1e-12), 0.0)
edge_type_counts["edge_type_entropy"] = (-np.sum(et_p * et_logp, axis=1)).astype(np.float32)

graph_df = graph_df.merge(
    edge_type_counts[["address", "edge_type_entropy"] + etype_cols],
    on="address", how="left"
).fillna(0.0)

del edge_type_counts; gc.collect()

# ── G3. Reciprocal edge count ────────────────────────────────────────────────
print("  [G3] reciprocal edge detection …")
out_neighbors = (e_norm.groupby("source")["target"]
                        .apply(set)
                        .reset_index()
                        .rename(columns={"source": "address", "target": "out_set"}))
in_neighbors  = (e_norm.groupby("target")["source"]
                        .apply(set)
                        .reset_index()
                        .rename(columns={"target": "address", "source": "in_set"}))
recip_df = out_neighbors.merge(in_neighbors, on="address", how="outer")
recip_df["out_set"] = recip_df["out_set"].apply(lambda x: x if isinstance(x, set) else set())
recip_df["in_set"]  = recip_df["in_set"].apply( lambda x: x if isinstance(x, set) else set())
recip_df["reciprocal_neighbor_count"] = recip_df.apply(
    lambda r: len(r["out_set"] & r["in_set"]), axis=1
)
recip_df["graph_reciprocity"] = recip_df.apply(
    lambda r: len(r["out_set"] & r["in_set"]) / max(len(r["out_set"] | r["in_set"]), 1),
    axis=1
)
graph_df = graph_df.merge(
    recip_df[["address", "reciprocal_neighbor_count", "graph_reciprocity"]],
    on="address", how="left"
).fillna(0.0)
del out_neighbors, in_neighbors, recip_df; gc.collect()

# ── G4. Approximate PageRank (power iteration, 10 steps) ────────────────────
print("  [G4] approximate PageRank (power iteration) …")
PAGERANK_SAMPLE = 500_000
e_pr = e_norm if len(e_norm) <= PAGERANK_SAMPLE else e_norm.sample(PAGERANK_SAMPLE, random_state=SEED)

all_pr_nodes = pd.Index(pd.concat([e_pr["source"], e_pr["target"]]).unique())
node2idx     = {n: i for i, n in enumerate(all_pr_nodes)}
N_pr         = len(all_pr_nodes)

src_idx = e_pr["source"].map(node2idx).dropna().astype(int)
tgt_idx = e_pr["target"].map(node2idx).dropna().astype(int)
valid   = src_idx.notna() & tgt_idx.notna()
src_idx = src_idx[valid].values
tgt_idx = tgt_idx[valid].values

out_deg_pr = np.bincount(src_idx, minlength=N_pr).astype(np.float32)
out_deg_pr = np.maximum(out_deg_pr, 1)

pr = np.ones(N_pr, dtype=np.float32) / N_pr
d  = 0.85
for _ in range(10):
    contrib = pr[src_idx] / out_deg_pr[src_idx]
    new_pr  = np.zeros(N_pr, dtype=np.float32)
    np.add.at(new_pr, tgt_idx, contrib)
    pr = (1 - d) / N_pr + d * new_pr

pr_df = pd.DataFrame({"address": all_pr_nodes, "pagerank_approx": pr.tolist()})
graph_df = graph_df.merge(pr_df, on="address", how="left").fillna(0.0)
del e_pr, pr_df, pr, src_idx, tgt_idx, e_norm; gc.collect()
print(f"    PageRank computed on {N_pr:,} nodes")

# ── G5. Temporal edge density ────────────────────────────────────────────────
print("  [G5] temporal edge density …")
edge_per_addr = (edges.groupby("anchor_address")
                       .agg(
                           total_edges=("timestamp", "count"),
                           first_edge_ts=("timestamp", "min"),
                           last_edge_ts=("timestamp",  "max"),
                           unique_blocks_edges=("block_number", "nunique"),
                       )
                       .reset_index()
                       .rename(columns={"anchor_address": "address"}))
edge_per_addr["edge_lifespan_days"] = (
    (edge_per_addr["last_edge_ts"] - edge_per_addr["first_edge_ts"]) / 86400.0
).clip(lower=1/1440)
edge_per_addr["edges_per_day"] = (
    edge_per_addr["total_edges"] / edge_per_addr["edge_lifespan_days"]
)
edge_per_addr["log1p_edges_per_day"]   = np.log1p(edge_per_addr["edges_per_day"])
edge_per_addr["log1p_total_edges"]     = np.log1p(edge_per_addr["total_edges"])
edge_per_addr["log1p_unique_blk_edge"] = np.log1p(edge_per_addr["unique_blocks_edges"])

graph_df = graph_df.merge(
    edge_per_addr[["address", "edges_per_day", "log1p_edges_per_day",
                   "log1p_total_edges", "log1p_unique_blk_edge"]],
    on="address", how="left"
).fillna(0.0)

# ── G6. Structural diversity score (LEAKAGE-FREE) ────────────────────────────
# NOTE: Neighbor-label-based entropy was removed because src_label_name in
# edges_raw encodes ground-truth labels of connected nodes — using it before
# the train/test split is data leakage. Instead we use structural graph
# features only: proportion of self-loops and proportion of cross-type edges.
print("  [G6] structural diversity (no label leakage) …")
edge_struct = edges.groupby("anchor_address").agg(
    _n_selfloop=("is_error", "count"),   # placeholder column for size
).reset_index().rename(columns={"anchor_address": "address", "_n_selfloop": "n_edges_total"})

selfloop_counts = (edges[edges["source"] == edges["target"]]
                   .groupby("anchor_address")
                   .size()
                   .reset_index(name="n_selfloop_edges")
                   .rename(columns={"anchor_address": "address"}))
edge_struct = edge_struct.merge(selfloop_counts, on="address", how="left")
# fillna on the whole frame fails when address is Categorical (0.0 is not a category).
# Only n_selfloop_edges can be NaN here (addresses with no self-loops).
edge_struct["n_selfloop_edges"] = edge_struct["n_selfloop_edges"].fillna(0.0)
edge_struct["selfloop_ratio"] = edge_struct["n_selfloop_edges"] / (edge_struct["n_edges_total"] + eps)

graph_df = graph_df.merge(
    edge_struct[["address", "selfloop_ratio"]],
    on="address", how="left"
).fillna(0.0)
del edge_struct, selfloop_counts; gc.collect()

# Log-transform heavy graph features
for c in ["total_in_deg", "total_out_deg", "total_degree",
          "reciprocal_neighbor_count", "out_neighbors_norm", "in_neighbors_norm"]:
    if c in graph_df.columns:
        graph_df[f"log1p_{c}"] = np.log1p(graph_df[c].clip(lower=0))

print(f"    Graph features shape: {graph_df.shape}")
print(f"    Columns: {list(graph_df.columns)}")


# %% [markdown]
# ## 9.  GROUP H — Cross-feature polynomial interactions (domain-motivated)

# %%
print("\n[9] GROUP H: domain-motivated cross-feature interactions …")

# All interactions have explicit economic / behavioral interpretation.
# We avoid blind polynomial expansion (collinearity risk).

# H1. Gas-per-counterparty: high = expensive targeted attacks
df["gas_per_counterpart"]     = df["gas_used_total"] / (df["unique_counterparts"].clip(lower=1))
df["log1p_gas_per_counterpart"] = np.log1p(df["gas_per_counterpart"].clip(lower=0))

# H2. Approval density: approvals per unique ERC20 contract
#   Very high → approval fishing
df["approval_density"] = (
    df["approval_count"] / (df["erc20_unique_tokens"].clip(lower=1))
)

# H3. Error rate × value: costly failed attempts (exploit probing)
df["err_x_val"] = df["err_rate"] * np.log1p(df["val_out_total_eth"])

# H4. Burst × novel selector: bursty + unusual calls = automated exploit
df["burst_x_novel"]  = df["burstiness"].clip(lower=0) * df["not_in_shortlist_ratio"]

# H5. Counterparty entropy × reciprocity: high entropy + low reciprocity = scatter attack
df["cp_entropy_x_recip"] = df["counterparty_entropy"] * (1.0 - df["reciprocity"])

# H6. Lifespan × tx density: short life with many tx = flash attack
df["lifespan_x_density"] = df["tx_per_day"] * np.exp(-df["lifespan_days"].clip(0, 365) / 30.0)
df["log1p_lifespan_x_density"] = np.log1p(df["lifespan_x_density"].clip(lower=0))

# H7. MEV composite: tx_index_low × gas_spike × error rate
df["mev_composite"] = (
    df["tx_index_low_ratio"] *
    df["gas_price_spike_ratio"].clip(0, 10) / 10.0 +
    df["err_rate"] * 0.5 +
    df["burstiness"].clip(lower=0) * 0.3
)

# H8. Contract interaction intensity: smart contract tx ratio × method entropy
df["contract_intensity"] = df["is_contract"] * df["method_entropy"]

# H9. Flash-loan hallmark: very short lifespan + very high ETH throughput
df["flash_hallmark"] = (
    df["rapid_lifecycle_flag"] * df["log1p_eth_throughput"] +
    (df["lifespan_days"].clip(0, 1) < 0.001).astype(float) * df["flash_mev_composite"]
)

# H10. Honeypot trap score: contract + high inflow + high error + low outflow
df["honeypot_trap"] = (
    df["is_contract"].astype(float) *
    (df["flow_imbalance"].clip(lower=0)) *
    df["err_rate"] *
    (1.0 - df["eth_recycling_ratio"])
)

print(f"    GROUP H features added.")


# %% [markdown]
# ## 10.  GROUP I — Heterogeneous node-type enrichment

# %%
print("\n[10] GROUP I: node-type enrichment …")

# I1. Bytecode size tiers (ordinal encoding of contract size)
df["bytecode_tier"] = pd.cut(
    df["bytecode_size"].clip(lower=0),
    bins=[-1, 0, 100, 1000, 5000, 15000, np.inf],
    labels=[0, 1, 2, 3, 4, 5]
).astype(float)

# I2. Proxy contract likelihood
#   Small bytecode + calls to execute/multicall + high internal tx ratio
df["proxy_likelihood"] = (
    (df["bytecode_tier"] <= 2).astype(float) * 0.4 +
    df["proxy_score"].clip(0, 1) * 0.4 +
    df["internal_to_total_ratio"].clip(0, 1) * 0.2
)

# I3. Self-deployer flag: contract_creations > 0 AND is_contract
df["self_deployer"] = (
    (df["is_contract"] == 1) & (df["contract_creations"] > 0)
).astype(np.float32)

# I4. Multi-sig likelihood: execute selector + low counterparty entropy
df["multisig_likelihood"] = (
    df.get("sel_execute_share", pd.Series(0.0, index=df.index)) * 0.5 +
    df.get("sel_multicall_share", pd.Series(0.0, index=df.index)) * 0.5
)

# I5. Node type one-hot (for heterogeneous GNN routing)
df["node_is_wallet"]   = (df["is_contract"] == 0).astype(np.float32)
df["node_is_contract"] = (df["is_contract"] == 1).astype(np.float32)

# I6. Balance tier (ordinal)
df["balance_tier"] = pd.cut(
    df["balance_eth"].clip(lower=0),
    bins=[-1e-9, 0, 0.01, 0.1, 1.0, 10.0, np.inf],
    labels=[0, 1, 2, 3, 4, 5]
).astype(float)

# I7. ERC-20 token issuer signal: minted AND has erc20 outflow
df["token_issuer_flag"] = (
    (df.get("sel_mint_hits", pd.Series(0, index=df.index)) > 0) &
    (df["erc20_out_count"] > 0)
).astype(np.float32)

# Save node type map for GNN heterogeneous routing (NB6)
node_type_map = df[["address", "label", "label_name", "is_contract",
                     "node_is_wallet", "node_is_contract",
                     "bytecode_tier", "balance_tier"]].copy()
node_type_map.to_parquet(OUT_DIR / "node_type_map.parquet", index=False)
print(f"    node_type_map saved: {node_type_map.shape}")


# %% [markdown]
# ## 11.  GROUP J — Edge feature engineering

# %%
print("\n[11] GROUP J: edge feature engineering …")

# tx_hash is ~4 GB (34M unique 66-char Python strings) and is NOT used in J1-J7.
# Dropping it now reduces edges from ~12.74 GB → ~8.7 GB, giving groupby transforms
# safe headroom. Row order is preserved (sort_values was removed from J1),
# so we can reload it by positional index just before the parquet write.
_had_tx_hash = "tx_hash" in edges.columns
if _had_tx_hash:
    edges.drop(columns=["tx_hash"], inplace=True)
    gc.collect()
    print(f"    edges after tx_hash drop: "
          f"{edges.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# J1. Temporal decay weight (exponential, normalized per address)
#   More recent edges carry more signal weight in attention layers
print("  [J1] temporal decay …")
# groupby.transform("min"/"max") on an 8.8 GB frame peaks at ~14 GB:
# pandas creates internal sorted copies + three 272 MB int64 intermediates
# (_ts_min, _ts_max, _ts_span) alive simultaneously before the first del.
# Replace with cat.codes + np.ufunc.at: group stats stay in tiny (<1 MB)
# per-group arrays; only one 272 MB broadcast array exists at a time.
_codes  = edges["anchor_address"].cat.codes.values          # int32, ~136 MB
n_cats  = len(edges["anchor_address"].cat.categories)
_ts_i64 = edges["timestamp"].values                         # int64 view (0 MB extra)

_g_min = np.full(n_cats, np.iinfo(np.int64).max, dtype=np.int64)   # <1 MB
_g_max = np.full(n_cats, np.iinfo(np.int64).min, dtype=np.int64)   # <1 MB
np.minimum.at(_g_min, _codes, _ts_i64)
np.maximum.at(_g_max, _codes, _ts_i64)

_ts_min_row = _g_min[_codes]                                        # 272 MB int64
_ts_span_g  = np.maximum(_g_max - _g_min + np.int64(1), np.int64(1))  # <1 MB
del _g_min, _g_max; gc.collect()

_ts_span_row = _ts_span_g[_codes]                                   # 272 MB int64
del _ts_span_g; gc.collect()

_num = _ts_i64 - _ts_min_row                                        # 272 MB int64
del _ts_min_row; gc.collect()

# Divide in float32: 136 MB each vs 272 MB for float64
_ts_norm = (_num.astype(np.float32) /
            _ts_span_row.astype(np.float32))                        # peak: 408 MB
np.nan_to_num(_ts_norm, copy=False)
del _num, _ts_span_row; gc.collect()

# np.float32 scalars keep the exp in float32 (Python float literals promote to float64)
edges["temporal_decay_weight"] = np.exp(
    np.float32(-3.0) * (np.float32(1.0) - _ts_norm)
).astype(np.float32)
del _ts_norm, _codes; gc.collect()

# J2. Value tier (log-binned for attention weight differentiation)
edges["log1p_value_eth"] = np.log1p(edges["value_eth"].clip(lower=0)).astype(np.float32)
edges["value_tier"] = pd.cut(
    edges["log1p_value_eth"],
    bins=[-1, 0.01, 1, 3, 5, 8, np.inf],
    labels=[0, 1, 2, 3, 4, 5]
).astype(float).astype(np.float32)

# J3. Directionality flag (is the anchor address the sender or receiver?)
edges["is_outgoing"] = (edges["source"] == edges["anchor_address"]).astype(np.float32)
edges["is_incoming"] = (edges["target"] == edges["anchor_address"]).astype(np.float32)
edges["is_selfloop"]  = (edges["source"] == edges["target"]).astype(np.float32)

# J4. Gas log-transform
edges["log1p_gas_used"]     = np.log1p(edges["gas_used"].clip(lower=0)).astype(np.float32)
edges["log1p_gas_price_gw"] = np.log1p(edges["gas_price_gwei"].clip(lower=0)).astype(np.float32)

# J5. Edge-type one-hot (for HetGNN edge type routing)
for etype in ["normal", "internal", "erc20", "erc721"]:
    edges[f"etype_{etype}"] = (edges["edge_type"] == etype).astype(np.float32)

# J6. Block proximity feature: same block (tight coupling = flash loan / sandwich)
# By J6, edges is ~10.4 GB. groupby(["anchor_address","block_number"]).transform("count")
# needs a large hash table for ~20M unique pairs → OOM risk at that size.
# Use lexsort + consecutive-pair comparison: peak = 2 × 272 MB (order + int64 sorted copy).
print("  [J6] same-block flag …")
_a_codes = edges["anchor_address"].cat.codes.values   # int32, 136 MB (view)
_b_nums  = edges["block_number"].values               # int64 view

_order = np.lexsort((_b_nums, _a_codes))              # int64 sort indices, 272 MB
_as    = _a_codes[_order]                             # int32, 136 MB
_bs    = _b_nums[_order]                              # int64, 272 MB
del _a_codes; gc.collect()

# Mark rows in groups of size ≥ 2: row i is in a multi-edge block if
# it shares key with row i+1 (_same_next[i]) or row i-1 (_same_next[i-1]).
_same_next = ((_as[:-1] == _as[1:]) & (_bs[:-1] == _bs[1:]))   # bool, 34 MB
del _as, _bs; gc.collect()

_in_multi          = np.empty(len(edges), dtype=bool)
_in_multi[:-1]     = _same_next
_in_multi[-1]      = False
_in_multi[1:]     |= _same_next                        # mark preceding partner too
del _same_next; gc.collect()

_sbf                = np.empty(len(edges), dtype=np.float32)
_sbf[_order]        = _in_multi.view(np.uint8).astype(np.float32)
edges["same_block_flag"] = _sbf
del _sbf, _in_multi, _order; gc.collect()

# J7. Method semantic tag (integer code for edge-level attention)
METHOD_TAG = {
    "0x": 0,                                    # pure ETH
    "0xa9059cbb": 1, "0x23b872dd": 1,           # ERC20 transfer
    "0x095ea7b3": 2,                             # approve
    "0x38ed1739": 3, "0x7ff36ab5": 3,           # swap
    "0x18cbafe5": 3,
    "0x40c10f19": 4, "0x42966c68": 4,           # mint / burn
    "0xf2fde38b": 5, "0x715018a6": 5,           # ownership
    "0x6a761202": 6, "0xac9650d8": 6,           # execute / multicall
    "0x9cb8a26a": 7,                             # selfdestruct proxy
}
edges["method_tag"] = (edges["method_id"]
                        .map(METHOD_TAG)
                        .fillna(8)
                        .astype(np.int8))

print(f"    Edges engineered shape: {edges.shape}")
edges_feat_cols = [
    "tx_hash", "source", "target", "anchor_address",
    "timestamp", "block_number", "tx_index",
    "edge_type", "src_label", "src_label_name",
    "is_outgoing", "is_incoming", "is_selfloop",
    "log1p_value_eth", "value_tier",
    "log1p_gas_used", "log1p_gas_price_gw",
    "temporal_decay_weight", "same_block_flag",
    "is_error", "method_tag",
    "etype_normal", "etype_internal", "etype_erc20", "etype_erc721",
]
edges_feat_cols = [c for c in edges_feat_cols if c in edges.columns]
# Drop columns NOT in the output list before writing.
# At this point edges is ~5.8 GB; a .copy() would add another ~5 GB → OOM.
# Dropping extra cols frees ~0.8 GB, then we write directly (no copy).
_extra_edge_cols = [c for c in edges.columns if c not in set(edges_feat_cols)]
if _extra_edge_cols:
    edges.drop(columns=_extra_edge_cols, inplace=True)
    gc.collect()
# Reload tx_hash from the original parquet (single-column columnar read, ~4 GB).
# Row order matches because J1's sort_values was removed — edges was never reordered.
if _had_tx_hash and "tx_hash" in edges_feat_cols:
    _tx_hash_reload = pd.read_parquet(
        FEAT_DIR / "edges_raw.parquet", columns=["tx_hash"]
    )["tx_hash"]
    edges["tx_hash"] = _tx_hash_reload.values
    del _tx_hash_reload; gc.collect()
_n_edges, _n_edge_cols = len(edges), len(edges.columns)
edges.to_parquet(OUT_DIR / "edges_engineered.parquet", index=False)
print(f"    edges_engineered.parquet saved: ({_n_edges}, {_n_edge_cols})")
del edges; gc.collect()   # e_norm freed after G4, e_erc20/e_int after G1


# %% [markdown]
# ## 12.  Merge all feature groups into final matrix

# %%
print("\n[12] Merging all feature groups …")

# Start from pre-processed df (Groups A-I already in df)
final_df = df.copy()

# Merge graph features (Group G)
final_df = final_df.merge(graph_df, on="address", how="left")
final_df = final_df.fillna(0.0)

# Restore string / categorical columns that got coerced
final_df["label_name"] = final_df["label_name"].astype(str)
final_df["address"]    = final_df["address"].astype(str)

# ── Remove duplicate / redundant columns ────────────────────────────────────
DROP_RAW_HEAVY = [c for c in LOG_COLS if c in final_df.columns]
final_df = final_df.drop(columns=DROP_RAW_HEAVY, errors="ignore")
print(f"    Dropped {len(DROP_RAW_HEAVY)} raw heavy-tailed cols (log versions kept)")

# Drop zero-variance columns
num_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
zero_var  = [c for c in num_cols
             if c not in ("label", "first_ts", "last_ts")
             and final_df[c].nunique(dropna=False) <= 1]
if zero_var:
    final_df = final_df.drop(columns=zero_var)
    print(f"    Dropped {len(zero_var)} zero-variance columns: {zero_var[:5]}…")

# Clip extreme outliers to [0.001, 0.999] quantile for numeric features.
# LEAKAGE NOTE: These quantile bounds are computed on the full unsplit dataset.
# NB5 must recompute clip bounds strictly on the training fold and apply to test,
# overriding these values for the final GNN input.
PROTECTED = {"address", "label", "label_name", "first_ts", "last_ts",
             "has_collection_ts"}
clip_cols  = [c for c in final_df.select_dtypes(include=[np.number]).columns
              if c not in PROTECTED]
lo = final_df[clip_cols].quantile(0.001)
hi = final_df[clip_cols].quantile(0.999)
final_df[clip_cols] = final_df[clip_cols].clip(lower=lo, upper=hi, axis=1)

# Final NaN fill
final_df[clip_cols] = final_df[clip_cols].fillna(0.0)

print(f"\n    Final feature matrix shape: {final_df.shape}")
print(f"    Feature count (excluding address/label/label_name): "
      f"{len([c for c in final_df.columns if c not in ('address','label','label_name')])}")
print(f"    Class distribution:\n{final_df['label_name'].value_counts().to_string()}")


# %% [markdown]
# ## 13.  Feature group inventory

# %%
print("\n[13] Building feature group inventory …")

FEATURE_GROUPS = {}
group_prefix_map = {
    "A_log_transform":      lambda c: c.startswith("log1p_") and not any(
                                      c.startswith(f"log1p_{x}") for x in
                                      ["ponzi", "eth_th", "lifespan", "gas_per", "gas_cost",
                                       "miner", "total_", "unique_b", "edges", "val_total"]),
    "B_value_flow":         lambda c: any(c.startswith(p) for p in
                                      ["val_out_per", "val_in_per", "val_total_per",
                                       "eth_recycling", "flow_imbalance", "internal_eth",
                                       "val_front_load", "ponzi_funnel", "eth_throughput",
                                       "log1p_ponzi", "log1p_eth_th", "val_gini_max"]),
    "C_gas":                lambda c: any(c.startswith(p) for p in
                                      ["gas_utilization", "gas_cost_per", "gas_price_cv",
                                       "gas_price_spike", "gas_padding", "high_gas_burst",
                                       "miner_tip", "log1p_miner", "log1p_gas_cost"]),
    "D_temporal_velocity":  lambda c: any(c.startswith(p) for p in
                                      ["tx_vel_", "velocity_ratio", "tx_accel",
                                       "dormancy", "burst_concentration", "hurst_x_burst",
                                       "temporal_entropy", "iat_regularity",
                                       "rapid_lifecycle", "lifespan_tier", "active_fraction",
                                       "log1p_dormancy"]),
    "E_token_flow":         lambda c: any(c.startswith(p) for p in
                                      ["token_dominance", "eth_only", "multi_asset",
                                       "erc_type_entropy", "erc20_flow_imbalance",
                                       "erc721_flip", "token_per_tx",
                                       "erc20_contract_breadth", "erc721_collection",
                                       "cross_protocol_score"]),
    "F_method_semantic":    lambda c: any(c.startswith(p) for p in
                                      ["defi_score", "admin_score", "approval_cluster",
                                       "nft_score", "proxy_score", "flash_mev_composite",
                                       "rug_composite", "novel_method_entropy",
                                       "method_cluster_count"]),
    "G_graph_ego":          lambda c: any(c.startswith(p) for p in
                                      ["out_deg_", "in_deg_", "out_val_", "in_val_",
                                       "out_val_max", "in_val_max",
                                       "out_neighbors", "in_neighbors",
                                       "out_gas_", "in_gas_",
                                       "total_in_deg", "total_out_deg", "degree_ratio",
                                       "total_degree", "edge_type_entropy", "etype_",
                                       "reciprocal_neighbor", "graph_reciprocity",
                                       "pagerank_approx", "edges_per_day",
                                       "log1p_edges", "log1p_total", "log1p_unique_blk",
                                       "log1p_total_in", "log1p_total_out",
                                       "log1p_total_d", "log1p_reciprocal",
                                       "selfloop_ratio"]),
    "H_cross_interactions": lambda c: any(c.startswith(p) for p in
                                      ["gas_per_counterpart", "approval_density",
                                       "err_x_val", "burst_x_novel", "cp_entropy_x_recip",
                                       "lifespan_x_density", "mev_composite",
                                       "contract_intensity", "flash_hallmark",
                                       "honeypot_trap", "log1p_lifespan_x_density",
                                       "log1p_gas_per_count"]),
    "I_node_type":          lambda c: any(c.startswith(p) for p in
                                      ["bytecode_tier", "proxy_likelihood",
                                       "self_deployer", "multisig_likelihood",
                                       "node_is_wallet", "node_is_contract",
                                       "balance_tier", "token_issuer_flag"]),
    "A_imputation":         lambda c: c in ["days_since_last_activity", "has_collection_ts"],
}

all_feat_cols = [c for c in final_df.columns
                 if c not in ("address", "label", "label_name",
                              "first_ts", "last_ts")]
tagged = set()
for grp_name, predicate in group_prefix_map.items():
    members = [c for c in all_feat_cols if predicate(c) and c not in tagged]
    FEATURE_GROUPS[grp_name] = members
    tagged.update(members)

FEATURE_GROUPS["Z_raw_passthrough"] = [c for c in all_feat_cols if c not in tagged]

print("  Feature group summary:")
total_feats = 0
for grp, cols in sorted(FEATURE_GROUPS.items()):
    print(f"    {grp:35s}: {len(cols):4d} features")
    total_feats += len(cols)
print(f"    {'TOTAL':35s}: {total_feats:4d} features")

fe_summary = {
    "feature_groups":         {k: v for k, v in FEATURE_GROUPS.items()},
    "total_features":         total_feats,
    "n_addresses":            len(final_df),
    "cohort_marker_cols":     COHORT_MARKER_COLS,
    "edge_feature_cols":      edges_feat_cols,
    "log_transformed_sources": LOG_COLS,
    "output_parquet":         "extracted_dataset.parquet",
    "leakage_notes": [
        "A3: global median imputation used; class-conditional deferred to NB5",
        "outlier clip bounds are pre-split; NB5 must refit on training fold",
        "G6: anchor_label_entropy removed; replaced with structural selfloop_ratio",
    ],
}
with open(OUT_DIR / "fe_summary.json", "w") as fh:
    json.dump(fe_summary, fh, indent=2)
print(f"    fe_summary.json saved")


# %% [markdown]
# ## 14.  Save final feature matrix

# %%
print("\n[14] Saving extracted_dataset.parquet …")
final_df.to_parquet(OUT_DIR / "extracted_dataset.parquet", index=False)
print(f"    Saved: {final_df.shape}  →  {OUT_DIR}/extracted_dataset.parquet")

# Sanity: no NaN in model columns
model_cols = [c for c in final_df.columns
              if c not in ("address", "label", "label_name",
                           "first_ts", "last_ts", "has_collection_ts")]
nan_count = final_df[model_cols].isna().sum().sum()
print(f"    NaN count in model columns: {nan_count}  ({'OK' if nan_count == 0 else 'ISSUE'})")


# %% [markdown]
# ## 15.  Visualisations (Q1-publication quality)

# %%
print("\n[15] Generating visualisations …")
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── VIS 1: Feature group size bar chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
grp_names  = [g.replace("_", " ") for g in sorted(FEATURE_GROUPS)]
grp_counts = [len(FEATURE_GROUPS[g]) for g in sorted(FEATURE_GROUPS)]
colors_bar = plt.cm.tab10(np.linspace(0, 1, len(grp_names)))
bars = ax.barh(grp_names, grp_counts, color=colors_bar, edgecolor="white")
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_xlabel("Number of Features"); ax.set_title("Feature Group Sizes after Engineering")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis1_feature_group_sizes.png", bbox_inches="tight")
plt.close()
print("  vis1 saved")

# ── VIS 2: Class-wise distribution of 9 key engineered features ──────────────
KEY_FEATS = [
    "eth_recycling_ratio", "flow_imbalance", "velocity_ratio_7d",
    "flash_mev_composite", "rug_composite", "mev_composite",
    "honeypot_trap", "cross_protocol_score", "pagerank_approx",
]
KEY_FEATS = [f for f in KEY_FEATS if f in final_df.columns]

n_kf = len(KEY_FEATS)
ncols = 3; nrows = (n_kf + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
axes = axes.flatten()

for i, feat in enumerate(KEY_FEATS):
    ax = axes[i]
    for cname in CLASS_NAMES:
        sub = final_df[final_df["label_name"] == cname][feat].dropna()
        if len(sub) < 5: continue
        sub_clip = sub.clip(*np.percentile(sub, [1, 99]))
        ax.hist(sub_clip, bins=40, density=True, alpha=0.55,
                label=cname, color=CLASS_PALETTE.get(cname, "grey"),
                histtype="stepfilled", linewidth=1.2)
    ax.set_title(feat, fontsize=9); ax.set_xlabel("value"); ax.set_ylabel("density")
    if i == 0: ax.legend(fontsize=7, framealpha=0.5)

for j in range(i + 1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Engineered Feature Distributions by Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis2_key_feature_distributions.png", bbox_inches="tight")
plt.close()
print("  vis2 saved")

# ── VIS 3: Correlation heatmap — engineered composite scores ─────────────────
COMP_SCORES = [
    "defi_score", "admin_score", "flash_mev_composite", "rug_composite",
    "mev_composite", "honeypot_trap", "ponzi_funnel_ratio",
    "eth_recycling_ratio", "flow_imbalance", "velocity_ratio_7d",
    "cross_protocol_score", "novel_method_entropy", "pagerank_approx",
]
COMP_SCORES = [c for c in COMP_SCORES if c in final_df.columns]
corr = final_df[COMP_SCORES].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Matrix — Engineered Composite Scores", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis3_composite_score_correlation.png", bbox_inches="tight")
plt.close()
print("  vis3 saved")

# ── VIS 4: PageRank distribution per class ───────────────────────────────────
if "pagerank_approx" in final_df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    for cname in CLASS_NAMES:
        sub = np.log1p(
            final_df[final_df["label_name"] == cname]["pagerank_approx"]
        ).dropna()
        if len(sub) < 5: continue
        ax.hist(sub, bins=50, density=True, alpha=0.5, label=cname,
                color=CLASS_PALETTE.get(cname, "grey"), histtype="step", linewidth=1.5)
    ax.set_xlabel("log1p(PageRank)"); ax.set_ylabel("Density")
    ax.set_title("Approximate PageRank Distribution per Class")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vis4_pagerank_by_class.png", bbox_inches="tight")
    plt.close()
    print("  vis4 saved")

# ── VIS 5: Token dominance vs ETH flow imbalance (scatter per class) ─────────
fig, ax = plt.subplots(figsize=(9, 7))
for cname in CLASS_NAMES:
    sub = final_df[final_df["label_name"] == cname]
    if len(sub) < 5: continue
    ax.scatter(sub["token_dominance"].sample(min(len(sub), 300), random_state=SEED),
               sub["flow_imbalance"].sample(min(len(sub), 300), random_state=SEED),
               alpha=0.45, s=18, label=cname,
               color=CLASS_PALETTE.get(cname, "grey"))
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Token Dominance (ERC / all tx)")
ax.set_ylabel("ETH Flow Imbalance (in-out)/(in+out)")
ax.set_title("Token Dominance vs ETH Flow Imbalance by Class")
ax.legend(fontsize=8, markerscale=1.5, framealpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis5_token_vs_flow.png", bbox_inches="tight")
plt.close()
print("  vis5 saved")

# ── VIS 6: Radar chart — class-wise mean of 8 composite signals ──────────────
RADAR_FEATS = [
    "defi_score", "admin_score", "flash_mev_composite", "rug_composite",
    "mev_composite", "honeypot_trap", "cross_protocol_score", "token_dominance",
]
RADAR_FEATS = [f for f in RADAR_FEATS if f in final_df.columns]
N_radar = len(RADAR_FEATS)
angles = np.linspace(0, 2 * np.pi, N_radar, endpoint=False).tolist()
angles += angles[:1]

class_means = {}
for cname in CLASS_NAMES:
    sub = final_df[final_df["label_name"] == cname][RADAR_FEATS]
    if len(sub) == 0: continue
    class_means[cname] = sub.mean().values

all_vals = np.vstack(list(class_means.values()))
col_min, col_max = all_vals.min(axis=0), all_vals.max(axis=0)
col_range = np.maximum(col_max - col_min, 1e-9)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for cname, means in class_means.items():
    norm = (means - col_min) / col_range
    vals = norm.tolist() + norm[:1].tolist()
    ax.plot(angles, vals, linewidth=1.8,
            color=CLASS_PALETTE.get(cname, "grey"), label=cname)
    ax.fill(angles, vals, alpha=0.08, color=CLASS_PALETTE.get(cname, "grey"))

ax.set_thetagrids(np.degrees(angles[:-1]), RADAR_FEATS, size=9)
ax.set_title("Class-wise Composite Feature Radar\n(normalised means)", size=12, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis6_class_radar.png", bbox_inches="tight")
plt.close()
print("  vis6 saved")

# ── VIS 7: Temporal velocity violin ──────────────────────────────────────────
if "velocity_ratio_7d" in final_df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_data  = []
    plot_labels = []
    for cname in CLASS_NAMES:
        sub = final_df[final_df["label_name"] == cname]["velocity_ratio_7d"].clip(0, 20).dropna()
        if len(sub) < 5: continue
        plot_data.append(sub.values)
        plot_labels.append(cname)
    vp = ax.violinplot(plot_data, positions=range(len(plot_labels)),
                       showmedians=True, showextrema=False)
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(CLASS_PALETTE.get(plot_labels[i], "grey"))
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=20, ha="right")
    ax.set_ylabel("Early Tx Velocity Ratio (7d rate / lifetime rate)")
    ax.set_title("Temporal Velocity Distribution by Class (front-loading signal)")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="baseline (1×)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vis7_velocity_violin.png", bbox_inches="tight")
    plt.close()
    print("  vis7 saved")

# ── VIS 8: Graph ego-network features boxplot ────────────────────────────────
GRAPH_BOX_FEATS = [c for c in ["log1p_total_degree", "graph_reciprocity",
                                "edge_type_entropy", "selfloop_ratio",
                                "pagerank_approx"]
                   if c in final_df.columns]
if GRAPH_BOX_FEATS:
    nf = len(GRAPH_BOX_FEATS)
    fig, axes = plt.subplots(1, nf, figsize=(4 * nf, 5))
    if nf == 1: axes = [axes]
    for ax, feat in zip(axes, GRAPH_BOX_FEATS):
        data_by_class = [
            final_df[final_df["label_name"] == cn][feat].dropna().clip(
                *np.percentile(final_df[feat].dropna(), [1, 99])
            ).values
            for cn in CLASS_NAMES if len(final_df[final_df["label_name"] == cn]) > 0
        ]
        bp = ax.boxplot(data_by_class, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, cname in zip(bp["boxes"], CLASS_NAMES):
            patch.set_facecolor(CLASS_PALETTE.get(cname, "grey"))
            patch.set_alpha(0.7)
        ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=7)
        ax.set_title(feat, fontsize=9)
    fig.suptitle("Graph Ego-Network Feature Distributions by Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vis8_graph_features_boxplot.png", bbox_inches="tight")
    plt.close()
    print("  vis8 saved")

# ── VIS 9: Missing-value / data quality summary ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
grp_nonzero = {}
for grp, cols in sorted(FEATURE_GROUPS.items()):
    cols_present = [c for c in cols if c in final_df.columns]
    if not cols_present: continue
    nz = (final_df[cols_present] != 0).mean().mean() * 100
    grp_nonzero[grp.replace("_", " ")] = nz
ax.barh(list(grp_nonzero.keys()), list(grp_nonzero.values()), color="#5C6BC0")
ax.set_xlabel("% Non-zero values"); ax.set_title("Feature Sparsity by Group")
ax.axvline(50, color="red", linewidth=0.8, linestyle="--")

ax = axes[1]
class_feat_counts = {}
for cname in CLASS_NAMES:
    sub = final_df[final_df["label_name"] == cname][model_cols]
    n_active = (sub.nunique(dropna=False) > 1).sum()
    class_feat_counts[cname] = n_active
ax.bar(list(class_feat_counts.keys()),
       list(class_feat_counts.values()),
       color=[CLASS_PALETTE.get(c, "grey") for c in class_feat_counts])
ax.set_xticklabels(list(class_feat_counts.keys()), rotation=30, ha="right")
ax.set_ylabel("Non-constant feature count")
ax.set_title("Active Features per Class")

plt.tight_layout()
plt.savefig(FIG_DIR / "vis9_data_quality.png", bbox_inches="tight")
plt.close()
print("  vis9 saved")

# ── VIS 10: Pairwise composite score scatter matrix ────────────────────────────
PAIR_FEATS = ["flash_mev_composite", "rug_composite",
              "mev_composite", "honeypot_trap"]
PAIR_FEATS = [f for f in PAIR_FEATS if f in final_df.columns]
SAMPLE_N   = 200
if len(PAIR_FEATS) >= 2:
    nf = len(PAIR_FEATS)
    fig, axes = plt.subplots(nf, nf, figsize=(3 * nf, 3 * nf))
    for i, f1 in enumerate(PAIR_FEATS):
        for j, f2 in enumerate(PAIR_FEATS):
            ax = axes[i][j]
            if i == j:
                for cname in CLASS_NAMES:
                    sub = final_df[final_df["label_name"] == cname][f1].dropna()
                    if len(sub) < 5: continue
                    ax.hist(sub.clip(*np.percentile(sub, [1, 99])).values,
                            bins=30, density=True, alpha=0.5,
                            color=CLASS_PALETTE.get(cname, "grey"), histtype="step")
                ax.set_title(f1, fontsize=8)
            else:
                for cname in CLASS_NAMES:
                    sub = final_df[final_df["label_name"] == cname][[f1, f2]].dropna()
                    if len(sub) < 5: continue
                    s = sub.sample(min(SAMPLE_N, len(sub)), random_state=SEED)
                    ax.scatter(s[f2], s[f1], alpha=0.3, s=8,
                               color=CLASS_PALETTE.get(cname, "grey"))
            if i == nf - 1: ax.set_xlabel(f2, fontsize=7)
            if j == 0:      ax.set_ylabel(f1, fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Composite Score Pairwise Scatter Matrix", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vis10_composite_pair_scatter.png", bbox_inches="tight")
    plt.close()
    print("  vis10 saved")


# %% [markdown]
# ## 16.  Final summary diagnostics

# %%
print("\n" + "=" * 80)
print("NOTEBOOK 4 — FEATURE ENGINEERING COMPLETE")
print("=" * 80)

print(f"\n  extracted_dataset.parquet   : {final_df.shape}")
print(f"  edges_engineered.parquet    : ({_n_edges}, {_n_edge_cols})")
print(f"  node_type_map.parquet       : {node_type_map.shape}")
print(f"  fe_summary.json             : {len(fe_summary['feature_groups'])} groups")
print(f"  Figures                     : {len(list(FIG_DIR.glob('*.png')))} PNGs in {FIG_DIR}")

print(f"""
FEATURE GROUP SUMMARY
─────────────────────────────────────────────────────────────────────────────
{'Group':35s}  {'Count':>6s}  Description
─────────────────────────────────────────────────────────────────────────────""")
desc = {
    "A_imputation":      "days_since_last_activity (global median) + collection_ts flag",
    "A_log_transform":   "log1p-scaled heavy-tailed raw features",
    "B_value_flow":      "ETH recycling, flow imbalance, ponzi funnel, throughput",
    "C_gas":             "gas utilization, spike, CV, padding, miner tip",
    "D_temporal_velocity":"velocity ratios, acceleration, dormancy, burst",
    "E_token_flow":      "token dominance, ERC mix entropy, NFT flip, breadth",
    "F_method_semantic": "DeFi/admin/NFT/proxy/rug/flash composite scores",
    "G_graph_ego":       "weighted degree, PageRank, reciprocity, edge-entropy, selfloop",
    "H_cross_interactions":"gas/cp, approval density, err×val, MEV composite",
    "I_node_type":       "bytecode tier, proxy likelihood, multisig, balance tier",
    "Z_raw_passthrough": "original NB3 features passed through unchanged",
}
for grp in sorted(FEATURE_GROUPS):
    cnt  = len(FEATURE_GROUPS[grp])
    dsc  = desc.get(grp, "")
    print(f"  {grp:35s}  {cnt:>6d}  {dsc}")

print(f"""
LEAKAGE SUMMARY
─────────────────────────────────────────────────────────────────────────────
 FIXED in NB4:
   • A3: Class-median imputation removed → global median used (label-free)
   • G6: anchor_label_entropy removed → selfloop_ratio (structural, safe)
   • val_gini_max: placeholder resolved (max of gini_out/gini_in if available)
   • ERC entropy + edge-type entropy vectorized (no row-wise apply leakage risk)

 DEFERRED to NB5 (by design):
   • Outlier clip bounds computed pre-split; NB5 refits on training fold only
   • Class-conditional median imputation may be added in NB5 post-split

PRECONDITIONS FOR NB5 (FEATURE SELECTION)
─────────────────────────────────────────────────────────────────────────────
 1. first_ts / last_ts are STILL PRESENT. NB5 needs them for window cohort
    assignment. NB5 must drop them post-window before modeling.

 2. has_collection_ts is a binary meta-feature encoding data quality.
    NB5 should keep it in the feature set.

 3. behavioral_scores.parquet is NOT merged here. Keep separate for ablation.

 4. Edges are in edges_engineered.parquet. NB6 (graph builder) reads these
    directly. Edge features do NOT appear in extracted_dataset.parquet.

 5. node_type_map.parquet tells NB6 which nodes are contracts vs wallets
    for HetGNN relation-type routing.

 6. fe_summary.json['feature_groups'] provides the ablation-study grouping
    for NB12. Each group can be independently ablated.
─────────────────────────────────────────────────────────────────────────────
OUTPUT FILES:
  {OUT_DIR}/extracted_dataset.parquet
  {OUT_DIR}/edges_engineered.parquet
  {OUT_DIR}/node_type_map.parquet
  {OUT_DIR}/fe_summary.json
  {FIG_DIR}/*.png  (10 publication-quality figures)
─────────────────────────────────────────────────────────────────────────────
[done] All outputs ready for Notebook 5 (Feature Selection).
""")
