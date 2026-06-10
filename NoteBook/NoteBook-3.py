# %% [markdown]
# # Notebook 3 — Feature Extraction  (v6 — full-history, robust anchors)
#
# Reads raw Etherscan JSON dumps from Notebook 2 and emits five clean tables:
#
# | File | Contents | Used by |
# |---|---|---|
# | `features_raw.parquet`        | ~150+ numeric features per address, NO behavioral scores | NB4 engineering, GNN node init |
# | `behavioral_scores.parquet`   | 8 risk scores — ablation & LLM governance only         | NB7 LLM hook, NB12 ablation |
# | `behavioral_cards.parquet`    | text fingerprint + top-k method IDs                    | NB7 LLM hook |
# | `temporal_sequences.parquet`  | per-address time-aligned event sequences (incl. tx_hash) | TGNN, temporal transformer |
# | `edges_raw.parquet`           | one row per tx edge, keyed by tx_hash                  | NB6 graph builder |
#
# ──────────────────────────────────────────────────────────────────────────────
# ## TIME-TEMPORAL DESIGN PHILOSOPHY (read before reviewing)
# ──────────────────────────────────────────────────────────────────────────────
#
# Labels in this dataset were collected by distinct teams over an extended
# period: some addresses' on-chain histories begin in 2017, others in 2023+.
# A naive calendar split would create cohorts where some attack classes do
# not yet exist (flash loans before Aave V1, large-scale MEV before 2021).
# That is NOT the right way to model temporal structure here.
#
# This notebook produces THREE orthogonal kinds of temporal features:
#
#   1. ADDRESS-RELATIVE features  (calendar-invariant, fair across cohorts)
#        • IAT statistics, burstiness, DFA Hurst, IAT autocorrelation
#        • tx_first_{1,7,30}d — counts within first N days of each address's
#          OWN lifetime, NOT relative to any global reference date
#        • Block-density features (tx_per_block_max, multi_tx_block_ratio)
#          — describe an address's behavior at its own time of activity
#        • Transaction-position features (tx_index stats, tx_index_low_ratio)
#          — describe within-block positioning, calendar-invariant
#
#   2. CYCLICAL features  (calendar-pattern-invariant)
#        • Hour-of-day entropy, weekday entropy
#        • Capture diurnal/weekly patterns ("US business hours only") WITHOUT
#          anchoring to any specific date
#
#   3. ANCHORED TIMESTAMPS  (used ONLY for cohort assignment in NB5)
#        • first_ts, last_ts (Unix timestamps, computed from the UNION of all
#          six raw record streams — normal, internal, failed, erc20, erc721,
#          erc1155 — not just normal tx, giving a more accurate lifetime span)
#        • days_since_last_activity (anchored to per-address NB1 collection_ts,
#          not the runtime clock)
#        • These columns are CONSUMED by NB5's per-class stratified temporal
#          split. NB5 must drop them from the model input matrix after
#          assigning each address to its chronological cohort.
#        • NOTE: features themselves are computed from an address's FULL
#          on-chain history (no observation-window truncation) — every record
#          contributes to counts/value/gas/entropy/edge/sequence features, so
#          there is zero data loss. The class-composition skew that a naive
#          GLOBAL first_ts split would otherwise cause is handled in NB5 via
#          a PER-CLASS stratified 80/20 split (each class independently sorted
#          by first_ts, earliest 80% -> train, latest 20% -> test), which is
#          defensible and loses no addresses or records.
#
# ──────────────────────────────────────────────────────────────────────────────
# ## FIXES IN v6
# ──────────────────────────────────────────────────────────────────────────────
#  1. Removed the v5 point-in-time observation window (90 days / 500 events) —
#     it silently dropped post-window records, which is not reviewer-defensible
#     and loses data. All features are again computed from FULL address history.
#  2. first_ts / last_ts are still recomputed from the union of all six raw
#     streams (kept from v5 — a more robust lifetime anchor than normal-tx-only)
#  3. Removed pit_anchor_ts / pit_cutoff_ts / pit_n_events / pit_capped columns
#     (no longer applicable without a PIT window)
#  4. Class-composition skew from the global temporal split is now fixed in
#     NB5 via a per-class stratified 80/20 temporal split (see NB5 v2)
#
# ──────────────────────────────────────────────────────────────────────────────
# ## FIXES IN v4
# ──────────────────────────────────────────────────────────────────────────────
#  1. RAW_BASE path corrected to /kaggle/input/datasets/research1234567890/raw-dataset
#  2. gas_eth_total: accumulated per-event (not from misaligned length-checked lists)
#  3. Q1 visualizations added (10 publication-quality figures)
#  4. ERC-1155 edges added to edges_raw.parquet (previously missing)
#  5. ERC-721 gas fields read from event records (previously hardcoded to 0)
#  6. failed_transactions loaded and contributes failed_tx_* features
#  7. Print/header references corrected (→ Notebook 4)

# %%
import json, gc, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## 1. Paths, constants, and selector dictionaries

# %%
# Per PDF spec: /kaggle/input/datasets/research1234567890/raw-dataset
# Fallback to legacy path if the canonical one doesn't exist.
_CANDIDATE_BASES = [
    Path("/kaggle/input/datasets/research1234567890/raw-dataset"),
    Path("/kaggle/input/datasets/zerotoinfinitya/raw-data/kaggle/working/dataset"),
]
RAW_BASE = next((p for p in _CANDIDATE_BASES if p.exists()), _CANDIDATE_BASES[0])
print(f"[paths] RAW_BASE = {RAW_BASE}  (exists={RAW_BASE.exists()})")

OUT_DIR = Path("/kaggle/working/features")
FIG_DIR = Path("/kaggle/working/figures_nb3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "benign", "phishing", "rug_pull", "ponzi",
    "flash_loan_attack", "malicious_mev", "exploit_contract", "honeypot",
]
LABEL_MAP = {n: i for i, n in enumerate(CLASS_NAMES)}

# ── Primary selector probes (18 high-signal signatures) ──────────────────────
SELECTORS = {
    "erc20_transfer":       "0xa9059cbb",
    "erc20_transferfrom":   "0x23b872dd",
    "erc20_approve":        "0x095ea7b3",
    "erc721_safe_xfer":     "0x42842e0e",
    "erc721_safe_xfer_d":   "0xb88d4fde",
    "set_approval_for_all": "0xa22cb465",
    "balance_of":           "0x70a08231",
    "mint":                 "0x40c10f19",
    "burn":                 "0x42966c68",
    "transfer_ownership":   "0xf2fde38b",
    "renounce_ownership":   "0x715018a6",
    "owner_call":           "0x8da5cb5b",
    "swap_exact_t4t":       "0x38ed1739",
    "swap_exact_eth4t":     "0x7ff36ab5",
    "swap_exact_t4eth":     "0x18cbafe5",
    "selfdestruct_proxy":   "0x9cb8a26a",
    "multicall":            "0xac9650d8",
    "execute":              "0x6a761202",
}

# ── Extended common-selector set for `not_in_shortlist_ratio` ────────────────
# Source: 4byte.directory top-frequency selectors on Ethereum mainnet
# (snapshot compiled November 2024 — cite this date in paper Methods § Data).
_RAW_COMMON = [
    # Uniswap V2 router & factory
    "0x022c0d9f", "0x89afcb44", "0xd0e30db0", "0x2e1a7d4d",
    "0x791ac947", "0xe8e33700", "0x4a25d94a", "0xbaa2abde",
    "0xded9382a", "0x5b0d5984", "0xbc25cf77", "0x6a627842",
    "0xf305d719", "0x02751cec", "0xaf2979eb", "0x472b43f3",
    # Uniswap V3
    "0x414bf389", "0xc04b8d59", "0xdb3e2198", "0xa4232640",
    "0x13ead562", "0x04e45aaf", "0x5023b4df", "0x46423aa7",
    "0xa34123a7", "0x3c8a7d8d", "0x11ed56c9",
    # Aave V2/V3
    "0xab9c4b5d", "0x69328dec", "0xe8eda9df", "0x573ade81",
    "0xd65dc7a1", "0x617ba037", "0xa415bcad", "0x02c205f0",
    "0x94ba89a2", "0xb16a19de",
    # Compound
    "0xa0712d68", "0xdb006a75", "0xc5ebeaec", "0x852a12e3",
    "0xaae40a2a", "0x4e4d9fea", "0x1249c58b",
    # OpenSea / NFT marketplaces
    "0xab834bab", "0x9a1fc3a7", "0x87201b41", "0xe7acab24",
    "0x32fb901a", "0xfb0f3ee1",
    # Safe / Gnosis multisig
    "0x468721a7", "0x5229073f", "0xa3f4df7e",
    "0x0d582f13", "0xf698da25",
    # ENS
    "0x57f7789e", "0x96e494e8", "0xd5fa2b00", "0x1896f70a",
    # ERC-20 standard view/admin
    "0x18160ddd", "0x06fdde03", "0x95d89b41", "0x313ce567",
    "0xdd62ed3e",
    # ERC-1155 transfers
    "0xf242432a", "0x2eb2c2d6",
    # Proxy / upgrade patterns
    "0x3659cfe6", "0x4f1ef286", "0x8f283970", "0xf851a440",
    "0xcfee7c08", "0x439fab91",
    # Chainlink / oracles
    "0x50d25bcd", "0x668a0f02",
    # Curve
    "0x3df02124", "0xa6417ed6", "0x394747c5", "0xed8e84f3",
    # 1inch aggregator
    "0x2e95b6c8", "0xe449022e", "0x12aa3caf",
]

for s in _RAW_COMMON:
    assert (
        isinstance(s, str) and len(s) == 10 and s.startswith("0x")
        and all(c in "0123456789abcdef" for c in s[2:])
    ), f"malformed selector: {s!r}"

COMMON_SELECTORS = set(SELECTORS.values()) | set(_RAW_COMMON)
_extended_unique = len(set(_RAW_COMMON) - set(SELECTORS.values()))
print(f"[selectors] {len(COMMON_SELECTORS)} unique common selectors loaded "
      f"({len(SELECTORS)} primary + {_extended_unique} extended)")

# Try class name as-is (lowercase) first; if missing, try Title-case (e.g. "Benign")
SUBDIRS = {
    "trans":    "transactions",
    "internal": "internal_transactions",
    "failed":   "failed_transactions",
    "erc20":    "token/erc-20",
    "erc721":   "token/erc-721",
    "erc1155":  "token/erc-1155",
    "addr":     "address",
}

def class_path(class_name, kind):
    p = RAW_BASE / class_name / SUBDIRS[kind]
    if not p.exists():
        # Try Title-case ("benign" → "Benign")
        p2 = RAW_BASE / class_name.capitalize() / SUBDIRS[kind]
        if p2.exists():
            return p2
    return p

# %% [markdown]
# ## 2. Build address universe

# %%
class_index = {}
for cname in CLASS_NAMES:
    d = class_path(cname, "trans")
    if not d.exists():
        print(f"[warn] missing {d}"); class_index[cname] = []
        continue
    pairs = [(p.stem, p.stem.lower()) for p in sorted(d.glob("*.json"))]
    class_index[cname] = pairs
    print(f"  {cname:>22s}: {len(pairs):>6d} addresses")

# %% [markdown]
# ## 3. Load master CSV backbone

# %%
master_candidates = [
    RAW_BASE / "master_dataset.csv",
    Path("/kaggle/input/datasets/research1234567890/raw-dataset/master_dataset.csv"),
    Path("/kaggle/input/datasets/zerotoinfinitya/raw-data/kaggle/working/dataset/master_dataset.csv"),
]
master_df = None
for p in master_candidates:
    if p.exists():
        master_df = pd.read_csv(p, low_memory=False)
        print(f"[master] loaded {p} → {master_df.shape}")
        break
if master_df is None:
    print("[master] WARN: not found — proceeding without backbone")
    master_df = pd.DataFrame(columns=[
        "address", "label", "class_name", "is_contract",
        "bytecode_size", "balance_wei", "collected_at_utc",
    ])

master_df["address"] = master_df["address"].astype(str).str.lower()
master_df = master_df.drop_duplicates(subset=["address"], keep="last")
master_lookup = master_df.set_index("address").to_dict(orient="index")

universe_rows = []
for cname, pairs in class_index.items():
    for raw_stem, lower_stem in pairs:
        universe_rows.append({
            "address":     lower_stem,
            "address_raw": raw_stem,
            "label_name":  cname,
            "label":       LABEL_MAP[cname],
        })
universe = (pd.DataFrame(universe_rows)
              .drop_duplicates(subset=["address"], keep="first")
              .reset_index(drop=True))
print(f"[universe] {len(universe)} addresses | {universe['label_name'].nunique()} classes")
print(universe["label_name"].value_counts().to_string())

# %% [markdown]
# ## 4. Numeric helpers

# %%
def load_json_result(path):
    if not path.exists(): return []
    try:
        with open(path) as fh:
            blob = json.load(fh)
        res = blob.get("result", [])
        return res if isinstance(res, list) else []
    except Exception:
        return []

def load_json_dict(path):
    if not path.exists(): return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}

def safe_stats(arr, prefix, percentiles=(10, 90, 99)):
    a = np.asarray(arr, dtype=np.float64)
    out = {}
    if a.size == 0:
        for k in ["mean", "std", "min", "max", "median", "skew", "kurt"]:
            out[f"{prefix}_{k}"] = 0.0
        for p in percentiles: out[f"{prefix}_p{p}"] = 0.0
        return out
    out[f"{prefix}_mean"]   = float(a.mean())
    out[f"{prefix}_std"]    = float(a.std())
    out[f"{prefix}_min"]    = float(a.min())
    out[f"{prefix}_max"]    = float(a.max())
    out[f"{prefix}_median"] = float(np.median(a))
    for p in percentiles:
        out[f"{prefix}_p{p}"] = float(np.percentile(a, p))
    out[f"{prefix}_skew"] = float(sp_stats.skew(a)     if a.size > 2 and a.std() > 0 else 0.0)
    out[f"{prefix}_kurt"] = float(sp_stats.kurtosis(a) if a.size > 3 and a.std() > 0 else 0.0)
    return out

def shannon_entropy(items):
    if not items: return 0.0
    c = Counter(items); n = sum(c.values())
    if n <= 0: return 0.0
    p = np.fromiter(c.values(), dtype=np.float64) / n
    return float(-np.sum(p * np.log(p + 1e-12)))

def burstiness(iats):
    iats = np.asarray(iats, dtype=np.float64)
    if iats.size < 2: return 0.0
    mu, sd = iats.mean(), iats.std()
    s = sd + mu
    return float((sd - mu) / s) if s > 0 else 0.0

def dfa_hurst(timestamps_sorted, min_n=50):
    """
    Detrended Fluctuation Analysis (DFA) Hurst exponent (Peng et al. 1994).
    H ≈ 0.5 → uncorrelated  H > 0.5 → persistent  H < 0.5 → mean-reverting
    Returns 0.5 when n < min_n or computation fails.
    FIX: tracks good_windows in lockstep with flucts to prevent log-log misalignment.
    """
    if len(timestamps_sorted) < min_n + 1: return 0.5
    iats = np.diff(np.asarray(timestamps_sorted, dtype=np.float64))
    if iats.size < min_n or iats.std() == 0: return 0.5

    profile = np.cumsum(iats - iats.mean())
    N = len(profile)

    n_min, n_max = max(4, N // 50), N // 4
    if n_min >= n_max: return 0.5
    windows = np.unique(
        np.logspace(np.log10(n_min), np.log10(n_max), num=20).astype(int)
    )
    windows = windows[windows >= 4]
    if len(windows) < 4: return 0.5

    flucts, good_windows = [], []
    for w in windows:
        n_segs = N // int(w)
        if n_segs == 0: continue
        segments = profile[:n_segs * int(w)].reshape(n_segs, int(w))
        x = np.arange(int(w), dtype=np.float64)
        rms_segs = []
        for seg in segments:
            coeffs = np.polyfit(x, seg, 1)
            trend  = np.polyval(coeffs, x)
            rms_segs.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_segs:
            flucts.append(float(np.mean(rms_segs)))
            good_windows.append(int(w))

    if len(flucts) < 4: return 0.5
    try:
        slope, _ = np.polyfit(np.log(good_windows), np.log(flucts), 1)
        return float(np.clip(slope, 0.0, 1.5))
    except Exception:
        return 0.5

def gini_coef(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or v.sum() == 0: return 0.0
    v = np.sort(np.abs(v))
    n = v.size; cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)

def acf_lag1(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 or x.std() == 0: return 0.0
    xn = (x - x.mean()) / (x.std() + 1e-12)
    return float(np.mean(xn[:-1] * xn[1:]))

def _ts(rec):
    try:
        return int(rec.get("timeStamp") or 0)
    except Exception:
        return 0

def compute_lifetime_anchors(record_lists):
    """
    Compute (first_ts, last_ts) for an address as the min/max timestamp across
    the UNION of all six raw record streams (normal, internal, failed, erc20,
    erc721, erc1155) -- a more robust lifetime span than normal-tx-only.
    No filtering is applied; all records remain available for feature computation.
    Returns (first_ts, last_ts), both 0 if no timestamped records exist.
    """
    all_ts = [_ts(r) for lst in record_lists.values() for r in lst]
    all_ts = [t for t in all_ts if t > 0]
    if not all_ts:
        return 0, 0
    return min(all_ts), max(all_ts)

def parse_collection_ts(addr_meta, meta):
    for source in [addr_meta, meta]:
        raw = source.get("collected_at_utc") or source.get("collection_timestamp")
        if not raw: continue
        try:
            if isinstance(raw, (int, float)): return float(raw)
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
        except Exception: continue
    return None

# %% [markdown]
# ## 5. Per-address feature extraction
#
# **LOG-TRANSFORM PRECONDITION FOR NB4**
# These features span 10+ orders of magnitude and MUST be log1p-transformed
# in NB4 before GNN node initialization (GATv2 attention is scale-sensitive):
#   val_out_total_eth, val_in_total_eth, abs(val_net_eth),
#   gas_used_total, gas_eth_total, bytecode_size, balance_eth,
#   tx_count, erc20_tx_count, erc721_tx_count, internal_tx_count, unique_blocks
# Tree models (XGBoost, LightGBM) are scale-invariant and do NOT require this.

# %%
def extract_for_address(addr_raw, addr_l, label_name, label):
    """
    Returns (features_dict, scores_dict, card_row, edge_rows, seq_row).
    All parallel arrays (timestamps, method_ids, etc.) are guaranteed time-aligned
    by constructing a single sorted events list and reading from it in lockstep.
    """
    trans     = load_json_result(class_path(label_name, "trans")    / f"{addr_raw}.json")
    interns   = load_json_result(class_path(label_name, "internal") / f"{addr_raw}.json")
    failed_tx = load_json_result(class_path(label_name, "failed")   / f"{addr_raw}.json")
    erc20s    = load_json_result(class_path(label_name, "erc20")    / f"{addr_raw}.json")
    erc721s   = load_json_result(class_path(label_name, "erc721")   / f"{addr_raw}.json")
    erc1155s  = load_json_result(class_path(label_name, "erc1155")  / f"{addr_raw}.json")
    addr_meta = load_json_dict(  class_path(label_name, "addr")     / f"{addr_raw}.json")
    meta      = master_lookup.get(addr_l, {})

    # ── Lifetime anchors (v6) — computed from the union of ALL six raw streams,
    # WITHOUT filtering. Every record remains available for feature computation;
    # these anchors are used only to recompute first_ts/last_ts below. ────────
    _anchor_first_ts, _anchor_last_ts = compute_lifetime_anchors(
        {"trans": trans, "internal": interns, "failed": failed_tx,
         "erc20": erc20s, "erc721": erc721s, "erc1155": erc1155s}
    )

    collection_ts = parse_collection_ts(addr_meta, meta)
    statuses      = addr_meta.get("statuses", {})

    f = {
        "address":    addr_l,
        "label":      int(label),
        "label_name": label_name,
        "is_contract":   int(bool(meta.get("is_contract", False))),
        "bytecode_size": float(meta.get("bytecode_size", 0) or 0),
        "balance_eth":   float(meta.get("balance_eth", 0.0) or 0.0),
        "has_normal_tx":   int(len(trans)    > 0),
        "has_internal_tx": int(len(interns)  > 0),
        "has_erc20_tx":    int(len(erc20s)   > 0),
        "has_erc721_tx":   int(len(erc721s)  > 0),
        "has_erc1155_tx":  int(len(erc1155s) > 0),
        "api_normal_tx_ok": int(statuses.get("normal_tx",   "0") == "1"),
        "api_erc20_ok":     int(statuses.get("erc20_tx",    "0") == "1"),
        "api_internal_ok":  int(statuses.get("internal_tx", "0") == "1"),
    }

    # ═══════════════════════════════════════════════════════════════════════
    # NORMAL TRANSACTIONS — build a single canonical events list, sort once.
    # Guarantees all parallel arrays downstream are time-aligned.
    # ═══════════════════════════════════════════════════════════════════════
    events = []
    for tx in trans:
        ts = int(tx.get("timeStamp") or 0)
        if ts == 0: continue
        inp = tx.get("input") or "0x"
        events.append({
            "ts":        ts,
            "block":     int(tx.get("blockNumber") or 0),
            "tx_index":  int(tx.get("transactionIndex") or 0),
            "from":      (tx.get("from") or "").lower(),
            "to":        (tx.get("to")   or "").lower(),
            "value":     int(tx.get("value")    or 0),
            "gas_used":  int(tx.get("gasUsed")  or 0),
            "gas_price": int(tx.get("gasPrice") or 0),
            "gas_limit": int(tx.get("gas")      or 0),
            "method_id": inp[:10].lower() if len(inp) >= 10 else "0x",
            "is_error":  (tx.get("isError") or "0") == "1",
            "creates":   bool((tx.get("contractAddress") or "").strip()),
            "tx_hash":   (tx.get("hash") or "").lower(),
        })
    events.sort(key=lambda e: (e["ts"], e["block"], e["tx_index"]))
    n_processed = len(events)

    timestamps, blocks, tx_indices = [], [], []
    method_ids, tx_hashes          = [], []
    signed_values, counterparties, dir_seq = [], [], []

    values_out, values_in           = [], []
    gas_used_lst, gas_price_lst     = [], []
    gas_limit_lst                   = []
    gas_eth_total_acc               = 0.0   # accumulated per-event (FIX: avoids list-length mismatch)
    recvs, sends, all_cps           = [], [], []

    err_count = succ_count = zero_count = self_loop = approval_ct = creations = 0
    edge_rows = []

    for e in events:
        timestamps.append(e["ts"])
        blocks.append(e["block"])
        tx_indices.append(e["tx_index"])
        method_ids.append(e["method_id"])
        tx_hashes.append(e["tx_hash"])
        if e["gas_used"]:  gas_used_lst.append(e["gas_used"])
        if e["gas_price"]: gas_price_lst.append(e["gas_price"])
        if e["gas_limit"]: gas_limit_lst.append(e["gas_limit"])

        # Accumulate gas cost per-event regardless of zero-value edge cases
        gas_eth_total_acc += (e["gas_used"] * e["gas_price"]) / 1e18

        if e["is_error"]: err_count  += 1
        else:             succ_count += 1
        if e["value"] == 0: zero_count += 1
        if e["creates"]:    creations  += 1
        if e["method_id"] == SELECTORS["erc20_approve"]: approval_ct += 1

        if e["from"] == addr_l and e["to"] == addr_l:
            self_loop += 1
            signed_values.append(0.0)
            counterparties.append("")
            dir_seq.append("s")
        elif e["from"] == addr_l:
            values_out.append(e["value"])
            signed_values.append(-(e["value"] / 1e18))
            counterparties.append(e["to"] if e["to"] else "")
            dir_seq.append("o")
            if e["to"]: recvs.append(e["to"]); all_cps.append(e["to"])
        elif e["to"] == addr_l:
            values_in.append(e["value"])
            signed_values.append(e["value"] / 1e18)
            counterparties.append(e["from"] if e["from"] else "")
            dir_seq.append("i")
            if e["from"]: sends.append(e["from"]); all_cps.append(e["from"])
        else:
            signed_values.append(0.0)
            counterparties.append("")
            dir_seq.append("s")

        edge_rows.append({
            "tx_hash":        e["tx_hash"],
            "source":         e["from"] if e["from"] else addr_l,
            "target":         e["to"]   if e["to"]   else addr_l,
            "timestamp":      e["ts"],
            "block_number":   e["block"],
            "tx_index":       e["tx_index"],
            "value_eth":      e["value"] / 1e18,
            "gas_used":       e["gas_used"],
            "gas_price_gwei": e["gas_price"] / 1e9,
            "gas_limit":      e["gas_limit"],
            "method_id":      e["method_id"],
            "edge_type":      "normal",
            "is_error":       int(e["is_error"]),
            "src_label":      label,
            "src_label_name": label_name,
            "anchor_address": addr_l,
        })

    # ── Count / value / gas features ─────────────────────────────────────────
    f["tx_count"]           = n_processed
    f["err_count"]          = err_count
    f["succ_count"]         = succ_count
    f["err_rate"]           = err_count   / max(n_processed, 1)
    f["zero_val_count"]     = zero_count
    f["zero_val_ratio"]     = zero_count  / max(n_processed, 1)
    f["self_loop_count"]    = self_loop
    f["contract_creations"] = creations
    f["approval_count"]     = approval_ct
    f["approval_ratio"]     = approval_ct / max(n_processed, 1)
    f["n_out_tx"]           = len(values_out)
    f["n_in_tx"]            = len(values_in)
    f["out_in_tx_ratio"]    = len(values_out) / max(len(values_in), 1)

    vo_eth = [v / 1e18 for v in values_out]
    vi_eth = [v / 1e18 for v in values_in]
    f.update(safe_stats(vo_eth, "val_out"))
    f.update(safe_stats(vi_eth, "val_in"))
    f["val_out_total_eth"]   = float(sum(vo_eth))
    f["val_in_total_eth"]    = float(sum(vi_eth))
    f["val_net_eth"]         = f["val_in_total_eth"] - f["val_out_total_eth"]
    f["val_out_to_in_ratio"] = f["val_out_total_eth"] / max(f["val_in_total_eth"], 1e-9)
    f["val_gini"]            = gini_coef(vo_eth + vi_eth)
    f["signed_val_acf_lag1"] = acf_lag1(signed_values)

    gp_gwei = [g / 1e9 for g in gas_price_lst]
    f.update(safe_stats(gas_used_lst,  "gas_used"))
    f.update(safe_stats(gp_gwei,       "gas_price_gwei"))
    f.update(safe_stats(gas_limit_lst, "gas_limit"))
    f["gas_used_total"] = float(sum(gas_used_lst))
    f["gas_eth_total"]  = gas_eth_total_acc   # FIX: per-event accumulation, always correct

    # ── Method-ID and selector features ──────────────────────────────────────
    mid_counter      = Counter(method_ids)
    all_real_methods = [m for m in method_ids if m not in ("0x", "")]
    n_real           = max(len(all_real_methods), 1)

    f["method_unique"]  = len(mid_counter)
    f["method_entropy"] = shannon_entropy(method_ids)

    top_methods = mid_counter.most_common(3)
    f["method_top1_share"]       = (top_methods[0][1] / max(n_processed, 1)) if top_methods else 0.0
    f["method_top3_share"]       = sum(c for _, c in top_methods) / max(n_processed, 1)
    f["method_top1_is_pure_eth"] = int(top_methods[0][0] in ("0x", "")) if top_methods else 0

    not_in_shortlist = sum(1 for m in all_real_methods if m not in COMMON_SELECTORS)
    rare_novel       = sum(
        1 for m, cnt in mid_counter.items()
        if cnt <= 2 and m not in ("0x", "") and m not in COMMON_SELECTORS
    )
    f["not_in_shortlist_ratio"] = not_in_shortlist / n_real
    f["rare_novel_ratio"]       = rare_novel       / n_real

    for name, sel in SELECTORS.items():
        hits = mid_counter.get(sel, 0)
        f[f"sel_{name}_hits"]  = hits
        f[f"sel_{name}_share"] = hits / max(n_processed, 1)

    # ── Counterparty topology ─────────────────────────────────────────────────
    cp_counter = Counter(all_cps)
    f["unique_counterparts"]    = len(cp_counter)
    f["unique_receivers"]       = len(set(recvs))
    f["unique_senders"]         = len(set(sends))
    union = len(set(recvs) | set(sends))
    inter = len(set(recvs) & set(sends))
    f["reciprocity"]            = inter / max(union, 1)
    f["counterparty_entropy"]   = shannon_entropy(all_cps)
    f["top_counterparty_share"] = (cp_counter.most_common(1)[0][1] / n_processed) if cp_counter else 0.0
    f["counterparty_gini"]      = gini_coef(list(cp_counter.values())) if cp_counter else 0.0
    f["fan_out_ratio"]          = f["unique_receivers"] / max(f["n_out_tx"], 1)

    cp_filtered = [c for c in counterparties if c]
    if cp_filtered:
        run, mx = 1, 1
        for i in range(1, len(cp_filtered)):
            if cp_filtered[i] == cp_filtered[i - 1]:
                run += 1; mx = max(mx, run)
            else:
                run = 1
        f["max_run_same_counterparty"] = mx
    else:
        f["max_run_same_counterparty"] = 0

    if len(dir_seq) > 1:
        flips = sum(1 for i in range(1, len(dir_seq)) if dir_seq[i] != dir_seq[i - 1])
        f["dir_flip_rate"] = flips / (len(dir_seq) - 1)
    else:
        f["dir_flip_rate"] = 0.0

    # ── Block-level temporal density (flash-loan + MEV detection) ─────────────
    if blocks:
        blk_counter = Counter(blocks)
        f["unique_blocks"]        = len(blk_counter)
        f["tx_per_block_mean"]    = n_processed / max(len(blk_counter), 1)
        f["tx_per_block_max"]     = max(blk_counter.values())
        f["multi_tx_block_ratio"] = sum(1 for c in blk_counter.values() if c > 1) / len(blk_counter)
        f["heavy_block_count"]    = sum(1 for c in blk_counter.values() if c >= 3)
        f["block_span"]           = float(max(blocks) - min(blocks))
    else:
        f["unique_blocks"]        = 0
        f["tx_per_block_mean"]    = 0.0
        f["tx_per_block_max"]     = 0
        f["multi_tx_block_ratio"] = 0.0
        f["heavy_block_count"]    = 0
        f["block_span"]           = 0.0

    # ── Transaction-position (MEV positioning) ────────────────────────────────
    if tx_indices:
        f.update(safe_stats(tx_indices, "tx_index"))
        f["tx_index_low_ratio"]  = sum(1 for i in tx_indices if i < 5) / n_processed
        f["tx_index_zero_ratio"] = sum(1 for i in tx_indices if i == 0) / n_processed
    else:
        f.update(safe_stats([], "tx_index"))
        f["tx_index_low_ratio"]  = 0.0
        f["tx_index_zero_ratio"] = 0.0

    # ── Temporal features (all address-relative, calendar-invariant) ──────────
    if timestamps:
        first_ts, last_ts = timestamps[0], timestamps[-1]
        f["first_ts"]      = int(first_ts)
        f["last_ts"]       = int(last_ts)
        f["lifespan_days"] = (last_ts - first_ts) / 86400.0
        f["tx_per_day"]    = n_processed / max(f["lifespan_days"], 1.0)

        iats = np.diff(timestamps)
        f.update(safe_stats(iats, "iat"))
        f["burstiness"]   = burstiness(iats)
        f["hurst_dfa"]    = dfa_hurst(timestamps, min_n=50)
        f["iat_acf_lag1"] = acf_lag1(iats)

        dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
        f["hour_entropy"] = shannon_entropy([d.hour      for d in dts])
        f["wday_entropy"] = shannon_entropy([d.weekday() for d in dts])

        days        = [t // 86400 for t in timestamps]
        day_counter = Counter(days)
        f["unique_active_days"] = len(day_counter)
        f["activity_ratio"]     = len(day_counter) / max(f["lifespan_days"] + 1, 1.0)
        f["top_day_share"]      = max(day_counter.values()) / n_processed
        f["activity_day_gini"]  = gini_coef(list(day_counter.values()))

        f["tx_first_1d"]        = int(sum(1 for t in timestamps if t <= first_ts + 1  * 86400))
        f["tx_first_7d"]        = int(sum(1 for t in timestamps if t <= first_ts + 7  * 86400))
        f["tx_first_30d"]       = int(sum(1 for t in timestamps if t <= first_ts + 30 * 86400))
        f["tx_first_1d_ratio"]  = f["tx_first_1d"]  / n_processed
        f["tx_first_7d_ratio"]  = f["tx_first_7d"]  / n_processed
        f["tx_first_30d_ratio"] = f["tx_first_30d"] / n_processed

        if f["lifespan_days"] > 14:
            w1_end   = first_ts + 7 * 86400
            wL_start = last_ts  - 7 * 86400
            w1 = sum(1 for t in timestamps if t <= w1_end)
            wL = sum(1 for t in timestamps if t >= wL_start)
            f["lifecycle_w1_share"] = w1 / n_processed
            f["lifecycle_wL_share"] = wL / n_processed
            f["lifecycle_decline"]  = (w1 - wL) / max(w1 + wL, 1)
        else:
            f["lifecycle_w1_share"] = 0.0
            f["lifecycle_wL_share"] = 0.0
            f["lifecycle_decline"]  = 0.0

        if collection_ts is not None:
            f["days_since_last_activity"] = max((collection_ts - last_ts) / 86400.0, 0.0)
        else:
            f["days_since_last_activity"] = np.nan

    else:
        for k in ["first_ts", "last_ts"]: f[k] = 0
        for k in ["lifespan_days", "tx_per_day", "burstiness", "iat_acf_lag1",
                  "hour_entropy", "wday_entropy", "activity_ratio", "top_day_share",
                  "activity_day_gini", "tx_first_1d_ratio", "tx_first_7d_ratio",
                  "tx_first_30d_ratio", "lifecycle_w1_share", "lifecycle_wL_share",
                  "lifecycle_decline"]: f[k] = 0.0
        f.update(safe_stats([], "iat"))
        f["hurst_dfa"]           = 0.5
        for k in ["unique_active_days", "tx_first_1d", "tx_first_7d", "tx_first_30d"]: f[k] = 0
        f["days_since_last_activity"] = np.nan

    # ── Lifetime anchors (v6): recompute first_ts/last_ts from the union of ALL
    # six raw streams (previously: normal-tx-only), so the cohort-split anchor
    # reflects an address's TRUE first/last on-chain activity even when its
    # earliest/latest activity was an internal/erc20/erc721/erc1155 transfer.
    # No records are dropped -- this only overrides the anchor timestamps. ───
    if _anchor_first_ts > 0:
        f["first_ts"] = int(_anchor_first_ts)
        f["last_ts"]  = int(_anchor_last_ts)
        if collection_ts is not None:
            f["days_since_last_activity"] = max((collection_ts - f["last_ts"]) / 86400.0, 0.0)

    # ── Internal transactions ─────────────────────────────────────────────────
    int_n = len(interns)
    f["internal_tx_count"]       = int_n
    f["internal_to_total_ratio"] = int_n / max(n_processed + int_n, 1)
    int_vals_out, int_vals_in, int_types = [], [], []

    for tx in interns:
        v   = int(tx.get("value") or 0)
        frm = (tx.get("from") or "").lower()
        to  = (tx.get("to")   or "").lower()
        ts2 = int(tx.get("timeStamp") or 0)
        typ = tx.get("type") or ""
        int_types.append(typ)
        if frm == addr_l:   int_vals_out.append(v / 1e18)
        elif to  == addr_l: int_vals_in.append(v / 1e18)
        if ts2 > 0:
            edge_rows.append({
                "tx_hash":        (tx.get("hash") or "").lower(),
                "source":         frm if frm else addr_l,
                "target":         to  if to  else addr_l,
                "timestamp":      ts2,
                "block_number":   int(tx.get("blockNumber") or 0),
                "tx_index":       0,
                "value_eth":      v / 1e18,
                "gas_used":       int(tx.get("gas") or 0),
                "gas_price_gwei": 0.0,
                "gas_limit":      0,
                "method_id":      "0x",
                "edge_type":      "internal",
                "is_error":       int((tx.get("isError") or "0") == "1"),
                "src_label":      label,
                "src_label_name": label_name,
                "anchor_address": addr_l,
            })

    f["int_val_out_total_eth"] = float(sum(int_vals_out))
    f["int_val_in_total_eth"]  = float(sum(int_vals_in))
    f.update(safe_stats(int_vals_out, "int_val_out"))
    f.update(safe_stats(int_vals_in,  "int_val_in"))
    f["int_type_entropy"]      = shannon_entropy(int_types)
    f["int_has_selfdestruct"]  = int("suicide" in int_types or "selfdestruct" in int_types)

    # ── Failed transactions (FIX: now loaded and used) ─────────────────────────
    # failed_tx folder contains transactions where isError=1, queried separately.
    # We deduplicate against normal tx by tx_hash to avoid double-counting.
    normal_hashes = set(tx_hashes)
    failed_gas_eth = 0.0
    failed_unique  = 0
    failed_vals    = []
    for tx in failed_tx:
        h = (tx.get("hash") or "").lower()
        if h in normal_hashes: continue   # already counted in normal tx
        failed_unique += 1
        gu = int(tx.get("gasUsed") or 0)
        gp = int(tx.get("gasPrice") or 0)
        failed_gas_eth += (gu * gp) / 1e18
        v = int(tx.get("value") or 0)
        if v > 0: failed_vals.append(v / 1e18)

    f["failed_tx_count"]          = failed_unique
    f["failed_tx_gas_eth_wasted"]  = failed_gas_eth
    f["failed_val_total_eth"]      = float(sum(failed_vals))
    f["failed_to_total_ratio"]     = failed_unique / max(n_processed + failed_unique, 1)

    # ── ERC-20 ────────────────────────────────────────────────────────────────
    e20_n = len(erc20s)
    f["erc20_tx_count"] = e20_n
    e20_out = e20_in = 0
    e20_contracts, e20_symbols = [], []

    for t in erc20s:
        ca  = (t.get("contractAddress") or "").lower()
        sym = t.get("tokenSymbol") or ""
        frm = (t.get("from") or "").lower()
        to  = (t.get("to")   or "").lower()
        ts2 = int(t.get("timeStamp") or 0)
        if ca:  e20_contracts.append(ca)
        if sym: e20_symbols.append(sym)
        if frm == addr_l: e20_out += 1
        if to  == addr_l: e20_in  += 1
        if ts2 > 0 and ca:
            decimals = int(t.get("tokenDecimal") or 18)
            edge_rows.append({
                "tx_hash":        (t.get("hash") or "").lower(),
                "source":         frm,
                "target":         to,
                "timestamp":      ts2,
                "block_number":   int(t.get("blockNumber") or 0),
                "tx_index":       int(t.get("transactionIndex") or 0),
                "value_eth":      float(t.get("value") or 0) / (10 ** decimals),
                "gas_used":       int(t.get("gasUsed") or 0),
                "gas_price_gwei": int(t.get("gasPrice") or 0) / 1e9,
                "gas_limit":      int(t.get("gas") or 0),
                "method_id":      ca,
                "edge_type":      "erc20",
                "is_error":       0,
                "src_label":      label,
                "src_label_name": label_name,
                "anchor_address": addr_l,
            })

    f["erc20_unique_tokens"]    = len(set(e20_contracts))
    f["erc20_contract_entropy"] = shannon_entropy(e20_contracts)
    f["erc20_symbol_entropy"]   = shannon_entropy(e20_symbols)
    f["erc20_out_count"]        = e20_out
    f["erc20_in_count"]         = e20_in
    f["erc20_out_in_ratio"]     = e20_out / max(e20_in, 1)
    f["erc20_to_eth_ratio"]     = e20_n   / max(n_processed, 1)

    # ── ERC-721 (NFT) ─────────────────────────────────────────────────────────
    e721_n = len(erc721s)
    f["erc721_tx_count"] = e721_n
    e721_tokens, e721_composite_ids = [], []
    e721_out = e721_in = 0

    for t in erc721s:
        ca  = (t.get("contractAddress") or "").lower()
        tid = t.get("tokenID") or t.get("tokenId") or ""
        frm = (t.get("from") or "").lower()
        to  = (t.get("to")   or "").lower()
        ts2 = int(t.get("timeStamp") or 0)
        if ca: e721_tokens.append(ca)
        if ca and tid: e721_composite_ids.append(f"{ca}:{tid}")
        if frm == addr_l: e721_out += 1
        if to  == addr_l: e721_in  += 1
        if ts2 > 0 and ca:
            edge_rows.append({
                "tx_hash":        (t.get("hash") or "").lower(),
                "source":         frm,
                "target":         to,
                "timestamp":      ts2,
                "block_number":   int(t.get("blockNumber") or 0),
                "tx_index":       int(t.get("transactionIndex") or 0),
                "value_eth":      0.0,
                "gas_used":       int(t.get("gasUsed") or 0),     # FIX: read actual value
                "gas_price_gwei": int(t.get("gasPrice") or 0) / 1e9,  # FIX: read actual value
                "gas_limit":      int(t.get("gas") or 0),          # FIX: read actual value
                "method_id":      ca,
                "edge_type":      "erc721",
                "is_error":       0,
                "src_label":      label,
                "src_label_name": label_name,
                "anchor_address": addr_l,
            })

    f["erc721_unique_collections"] = len(set(e721_tokens))
    f["erc721_unique_token_ids"]   = len(set(e721_composite_ids))
    f["erc721_out_count"]          = e721_out
    f["erc721_in_count"]           = e721_in
    f["erc721_out_in_ratio"]       = e721_out / max(e721_in, 1)

    # ── ERC-1155 (FIX: edges now generated) ───────────────────────────────────
    e1155_n = len(erc1155s)
    f["erc1155_tx_count"] = e1155_n
    e1155_tokens = []
    e1155_out = e1155_in = 0

    for t in erc1155s:
        ca  = (t.get("contractAddress") or "").lower()
        frm = (t.get("from") or "").lower()
        to  = (t.get("to")   or "").lower()
        ts2 = int(t.get("timeStamp") or 0)
        if ca: e1155_tokens.append(ca)
        if frm == addr_l: e1155_out += 1
        if to  == addr_l: e1155_in  += 1
        if ts2 > 0 and ca:
            edge_rows.append({
                "tx_hash":        (t.get("hash") or "").lower(),
                "source":         frm,
                "target":         to,
                "timestamp":      ts2,
                "block_number":   int(t.get("blockNumber") or 0),
                "tx_index":       int(t.get("transactionIndex") or 0),
                "value_eth":      0.0,
                "gas_used":       int(t.get("gasUsed") or 0),
                "gas_price_gwei": int(t.get("gasPrice") or 0) / 1e9,
                "gas_limit":      int(t.get("gas") or 0),
                "method_id":      ca,
                "edge_type":      "erc1155",
                "is_error":       0,
                "src_label":      label,
                "src_label_name": label_name,
                "anchor_address": addr_l,
            })

    f["erc1155_unique_tokens"] = len(set(x for x in e1155_tokens if x))
    f["erc1155_out_count"]     = e1155_out
    f["erc1155_in_count"]      = e1155_in

    # ── Behavioral card (LLM-inferenced governance) ───────────────────────────
    top_sel_name = max(SELECTORS.items(),
                       key=lambda kv: f.get(f"sel_{kv[0]}_share", 0.0),
                       default=("none", "0x"))[0]
    top_sel_share = f.get(f"sel_{top_sel_name}_share", 0.0)
    lc_tag = ("front-loaded" if f["lifecycle_decline"] >  0.4 else
              "back-loaded"  if f["lifecycle_decline"] < -0.4 else "steady")
    role     = "contract" if f["is_contract"] else "wallet"
    ls       = f["lifespan_days"]
    span_str = (f"{ls*24:.1f}h" if ls < 1 else
                f"{ls:.1f}d"    if ls < 30 else
                f"{ls/30:.1f}mo")
    dsla     = f["days_since_last_activity"]
    dsla_str = "unknown" if (dsla is None or (isinstance(dsla, float) and np.isnan(dsla))) else f"{dsla:.1f}d"

    behavioral_card = (
        f"{role}|tx={f['tx_count']}|in/out={f['n_in_tx']}/{f['n_out_tx']}"
        f"|val_net={f['val_net_eth']:+.3f}ETH|err={f['err_rate']:.2f}"
        f"|span={span_str}|dormant={dsla_str}"
        f"|burst={f['burstiness']:+.2f}|hurst_dfa={f['hurst_dfa']:.2f}"
        f"|cp_uniq={f['unique_counterparts']}|cp_H={f['counterparty_entropy']:.2f}"
        f"|method_H={f['method_entropy']:.2f}|top_sel={top_sel_name}({top_sel_share:.2f})"
        f"|novel_sel={f['not_in_shortlist_ratio']:.2f}"
        f"|gas_p99={f['gas_price_gwei_p99']:.0f}gwei"
        f"|tx_per_blk_max={f['tx_per_block_max']}"
        f"|tx_idx_low={f['tx_index_low_ratio']:.2f}"
        f"|approvals={f['approval_count']}|erc20={f['erc20_tx_count']}"
        f"|erc721={f['erc721_tx_count']}|erc1155={f['erc1155_tx_count']}"
        f"|failed={f['failed_tx_count']}|cycle={lc_tag}"
        f"|w7d={f['tx_first_7d_ratio']:.2f}|wL={f['lifecycle_wL_share']:.2f}"
    )

    # ── Temporal sequences (all positionally aligned) ─────────────────────────
    seq_row = {
        "address":           addr_l,
        "label":             label,
        "label_name":        label_name,
        "timestamps_seq":    timestamps,
        "blocks_seq":        blocks,
        "tx_indices_seq":    tx_indices,
        "signed_values_seq": signed_values,
        "method_ids_seq":    method_ids,
        "counterparty_seq":  counterparties,
        "tx_hashes_seq":     tx_hashes,
        "dir_seq":           dir_seq,
    }

    card_row = {
        "address":         addr_l,
        "label":           label,
        "label_name":      label_name,
        "behavioral_card": behavioral_card,
        "top_method_1":    top_methods[0][0] if len(top_methods) > 0 else "0x",
        "top_method_2":    top_methods[1][0] if len(top_methods) > 1 else "0x",
        "top_method_3":    top_methods[2][0] if len(top_methods) > 2 else "0x",
    }

    # ── Behavioral scores (NOT joined into features_raw — designer leakage) ───
    lc_pos = max(0.0, f["lifecycle_decline"])
    scores = {
        "address": addr_l, "label": label, "label_name": label_name,
        "phish_score": (
            2.0 * f["approval_ratio"]
            + min(f.get("sel_set_approval_for_all_share", 0.0) * 4.0, 1.0)
            + min(f["unique_counterparts"] / 50.0, 1.0) * 0.5
            + 0.5 * lc_pos
        ),
        "rug_score": (
            min(f["contract_creations"] / 3.0, 1.0)
            + 1.5 * lc_pos
            + min(f.get("sel_renounce_ownership_share", 0.0) * 5.0, 1.0)
            + min(f.get("sel_mint_share", 0.0) * 3.0, 1.0)
        ),
        "flash_score": (
            (1.0 if 0.0 < f["lifespan_days"] < 1 / 1440 else 0.0)
            + min((f["val_out_total_eth"] + f["val_in_total_eth"]) / 1000.0, 5.0) * 0.4
            + min(f.get("sel_swap_exact_t4t_share", 0.0)
                  + f.get("sel_swap_exact_eth4t_share", 0.0)
                  + f.get("sel_swap_exact_t4eth_share", 0.0), 1.0)
            + min(f["tx_per_block_max"] / 5.0, 1.0)
        ),
        "mev_score": (
            min(f["gas_price_gwei_p99"]   / 200.0, 2.0) * 0.5
            + min(f["gas_price_gwei_mean"] / 100.0, 2.0) * 0.5
            + 2.0 * f["err_rate"]
            + 0.3 * f["burstiness"]
            + 0.5 * f["tx_index_low_ratio"]
        ),
        "honeypot_score": min(
            f["is_contract"] * (
                f["val_in_total_eth"] / max(f["val_out_total_eth"], 1e-9)
            ) * 0.1
            + f["err_rate"] * f["is_contract"],
            10.0,
        ),
        "ponzi_score": (
            (f["unique_senders"] / max(f["unique_receivers"], 1)) * 0.5
            + (f["n_in_tx"] / max(f["n_out_tx"], 1)) * 0.3
            + f["is_contract"] * 0.2
        ),
        "exploit_score": (
            2.0 * f["err_rate"]
            + min(f["self_loop_count"] / max(n_processed, 1) * 5.0, 1.0)
            + f["int_has_selfdestruct"] * 1.0
            + f["is_contract"] * 0.5
            + min(f["failed_tx_count"] / max(n_processed, 1) * 3.0, 1.0)
        ),
        "benign_score": max(
            0.0,
            1.0 - f["err_rate"]
            - min(f["approval_ratio"] * 2.0, 1.0)
            - 0.5 * lc_pos
        ),
    }

    return f, scores, card_row, edge_rows, seq_row

# %% [markdown]
# ## 6. Main extraction loop — O(n) shard-based checkpointing

# %%
SHARD_DIR = OUT_DIR / "shards";      SHARD_DIR.mkdir(exist_ok=True)
EDGE_DIR  = OUT_DIR / "edge_shards"; EDGE_DIR.mkdir(exist_ok=True)
CKPT_EVERY = 2000

def _next_shard_idx(directory, prefix):
    return len(list(directory.glob(f"{prefix}_*.parquet")))

done_set = set()
for p in sorted(SHARD_DIR.glob("feat_*.parquet")):
    try:
        done_set.update(
            pd.read_parquet(p, columns=["address"])["address"]
              .astype(str).str.lower().tolist()
        )
    except Exception:
        pass
print(f"[resume] {len(done_set)} addresses already processed")

todo_df = universe[~universe["address"].isin(done_set)].reset_index(drop=True)
print(f"[loop] todo={len(todo_df)}  done={len(done_set)}")

feat_buf, score_buf, card_buf, seq_buf, edge_buf = [], [], [], [], []
processed = 0

def _flush(feat_buf, score_buf, card_buf, seq_buf, edge_buf):
    idx_f = _next_shard_idx(SHARD_DIR, "feat")
    pd.DataFrame(feat_buf ).to_parquet(SHARD_DIR / f"feat_{idx_f:05d}.parquet",  index=False)
    pd.DataFrame(score_buf).to_parquet(SHARD_DIR / f"score_{idx_f:05d}.parquet", index=False)
    pd.DataFrame(card_buf ).to_parquet(SHARD_DIR / f"card_{idx_f:05d}.parquet",  index=False)
    pd.DataFrame(seq_buf  ).to_parquet(SHARD_DIR / f"seq_{idx_f:05d}.parquet",   index=False)
    if edge_buf:
        idx_e = _next_shard_idx(EDGE_DIR, "edge")
        pd.DataFrame(edge_buf).to_parquet(EDGE_DIR / f"edge_{idx_e:05d}.parquet", index=False)

try:
    for _, row in tqdm(todo_df.iterrows(), total=len(todo_df), desc="extract"):
        addr_raw = row["address_raw"]
        addr_l   = row["address"]
        lname    = row["label_name"]
        lab      = int(row["label"])
        try:
            f, scores, card_row, e_rows, seq_row = extract_for_address(
                addr_raw, addr_l, lname, lab
            )
        except Exception as e:
            f        = {"address": addr_l, "label": lab, "label_name": lname,
                        "extract_error": str(e)[:300]}
            scores   = {"address": addr_l, "label": lab, "label_name": lname}
            card_row = {"address": addr_l, "label": lab, "label_name": lname}
            seq_row  = {"address": addr_l, "label": lab, "label_name": lname,
                        "timestamps_seq": [], "blocks_seq": [], "tx_indices_seq": [],
                        "signed_values_seq": [], "method_ids_seq": [],
                        "counterparty_seq": [], "tx_hashes_seq": [], "dir_seq": []}
            e_rows   = []

        feat_buf.append(f); score_buf.append(scores); card_buf.append(card_row)
        seq_buf.append(seq_row); edge_buf.extend(e_rows)
        processed += 1

        if processed % CKPT_EVERY == 0:
            _flush(feat_buf, score_buf, card_buf, seq_buf, edge_buf)
            feat_buf.clear(); score_buf.clear(); card_buf.clear()
            seq_buf.clear();  edge_buf.clear()
            gc.collect()
            print(f"[ckpt] {processed} rows flushed")

except KeyboardInterrupt:
    print("[loop] interrupted — flushing buffers…")

if feat_buf:
    _flush(feat_buf, score_buf, card_buf, seq_buf, edge_buf)
    print("[ckpt] final flush done")

# %% [markdown]
# ## 7. Finalize — merge shards, validate, save
#
# MEMORY STRATEGY: load → process → save → DEL each table before touching the next.
# seq_df stores list-valued columns (10k+ timestamps per address) and takes 3-5 GB.
# Keeping it alive while concat-ing edge shards causes OOM on 16 GB Kaggle T4.
# Edge shards are merged with a streaming pyarrow writer + 64-bit hash dedup
# so we never hold the full edge frame in RAM.

# %%
import pyarrow as pa
import pyarrow.parquet as pq

def _merge_shards_to_disk(directory, prefix, out_path, dedup_col="address"):
    """Load shards one-by-one, concat, dedup, save, return shape. No large df left in RAM."""
    paths = sorted(directory.glob(f"{prefix}_*.parquet"))
    if not paths:
        print(f"[warn] no shards for prefix={prefix}"); return (0, 0)
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset=[dedup_col], keep="last").reset_index(drop=True)
    shape = df.shape
    print(f"  {prefix}: {len(paths)} shards → {shape[0]} unique rows")
    df.to_parquet(out_path, index=False)
    del df; gc.collect()
    return shape

print("=== Merging shards (one at a time — frees RAM before edge merge) ===")
key_cols = {"address", "label", "label_name"}

# ── features_raw ─────────────────────────────────────────────────────────────
feat_paths = sorted(SHARD_DIR.glob("feat_*.parquet"))
feats_df   = pd.concat([pd.read_parquet(p) for p in feat_paths], ignore_index=True)
feats_df   = feats_df.drop_duplicates(subset=["address"], keep="last").reset_index(drop=True)
feats_df   = feats_df.drop(columns=["extract_error"], errors="ignore")

num_cols = [c for c in feats_df.columns if c not in key_cols]
for c in num_cols:
    if c == "days_since_last_activity":
        feats_df[c] = pd.to_numeric(feats_df[c], errors="coerce")
    else:
        feats_df[c] = pd.to_numeric(feats_df[c], errors="coerce").fillna(0.0).astype(np.float64)

const_cols = [c for c in num_cols
              if c != "days_since_last_activity" and feats_df[c].nunique(dropna=False) <= 1]
if const_cols:
    print(f"Dropping {len(const_cols)} constant cols: "
          f"{const_cols[:8]}{'…' if len(const_cols) > 8 else ''}")
    feats_df = feats_df.drop(columns=const_cols)

feat_shape = feats_df.shape
feats_df.to_parquet(OUT_DIR / "features_raw.parquet", index=False)
print(f"[saved] features_raw.parquet         {feat_shape}")
del feats_df; gc.collect()   # FREE — do not keep alive

# ── behavioral_scores ─────────────────────────────────────────────────────────
score_paths = sorted(SHARD_DIR.glob("score_*.parquet"))
scores_df   = pd.concat([pd.read_parquet(p) for p in score_paths], ignore_index=True)
scores_df   = scores_df.drop_duplicates(subset=["address"], keep="last").reset_index(drop=True)
for c in [x for x in scores_df.columns if x not in key_cols]:
    scores_df[c] = pd.to_numeric(scores_df[c], errors="coerce").fillna(0.0)
score_shape = scores_df.shape
scores_df.to_parquet(OUT_DIR / "behavioral_scores.parquet", index=False)
print(f"[saved] behavioral_scores.parquet    {score_shape}")
del scores_df; gc.collect()

# ── behavioral_cards ──────────────────────────────────────────────────────────
card_paths = sorted(SHARD_DIR.glob("card_*.parquet"))
cards_df   = pd.concat([pd.read_parquet(p) for p in card_paths], ignore_index=True)
cards_df   = cards_df.drop_duplicates(subset=["address"], keep="last").reset_index(drop=True)
card_shape = cards_df.shape
cards_df.to_parquet(OUT_DIR / "behavioral_cards.parquet", index=False)
print(f"[saved] behavioral_cards.parquet     {card_shape}")
del cards_df; gc.collect()

# ── temporal_sequences (largest — list columns, must free before edge merge) ──
seq_paths = sorted(SHARD_DIR.glob("seq_*.parquet"))
seq_df    = pd.concat([pd.read_parquet(p) for p in seq_paths], ignore_index=True)
seq_df    = seq_df.drop_duplicates(subset=["address"], keep="last").reset_index(drop=True)
seq_shape = seq_df.shape

# Capture seq stats BEFORE del (needed for section 8 diagnostics)
if "timestamps_seq" in seq_df.columns:
    seq_lens  = seq_df["timestamps_seq"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    seq_stats = {
        "mean": seq_lens.mean(), "median": seq_lens.median(),
        "p90":  seq_lens.quantile(.9), "max": seq_lens.max(),
    }
else:
    seq_stats = {}

seq_df.to_parquet(OUT_DIR / "temporal_sequences.parquet", index=False)
print(f"[saved] temporal_sequences.parquet   {seq_shape}")
del seq_df, seq_lens; gc.collect()   # CRITICAL — frees 3-5 GB before edge merge

# ── edges_raw — streaming pyarrow writer + 64-bit hash dedup ─────────────────
print("\n=== Merging edge shards (streaming — avoids OOM) ===")
edge_paths = sorted(EDGE_DIR.glob("edge_*.parquet"))
edge_total_in = edge_total_out = 0
edge_type_counts = {}     # for section-8 diagnostics

if edge_paths:
    seen_keys  = set()    # 64-bit hashes: ~8 bytes each, O(n_unique_edges)
    writer     = None
    out_path   = OUT_DIR / "edges_raw.parquet"
    no_hash_ct = 0

    for p in tqdm(edge_paths, desc="merge edges"):
        tbl = pq.read_table(p)
        df  = tbl.to_pandas()
        del tbl; gc.collect()
        edge_total_in += len(df)

        # Fix dtypes
        df["timestamp"]    = pd.to_numeric(df["timestamp"],    errors="coerce").fillna(0).astype(np.int64)
        df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce").fillna(0).astype(np.int64)
        no_hash_ct        += (df["tx_hash"].astype(str) == "").sum()

        # Dedup via 64-bit hash of (tx_hash, edge_type, source, target)
        key_hash = pd.util.hash_pandas_object(
            df[["tx_hash", "edge_type", "source", "target"]], index=False
        )
        mask = ~key_hash.isin(seen_keys)
        seen_keys.update(key_hash[mask].tolist())

        df = df[mask].reset_index(drop=True)
        if len(df) == 0:
            del df; continue

        # Track edge-type counts for diagnostics
        for et, cnt in df["edge_type"].value_counts().items():
            edge_type_counts[et] = edge_type_counts.get(et, 0) + int(cnt)

        tbl_out = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl_out.schema, compression="snappy")
        writer.write_table(tbl_out)
        edge_total_out += len(df)
        del df, tbl_out; gc.collect()

    if writer: writer.close()
    del seen_keys; gc.collect()

    if no_hash_ct:
        print(f"  [audit] {no_hash_ct} edges had empty tx_hash "
              f"({no_hash_ct/max(edge_total_in,1):.2%})")
    print(f"[saved] edges_raw.parquet: {edge_total_in} raw → {edge_total_out} after dedup "
          f"({len(edge_paths)} shards)")
else:
    print("[warn] no edge shards found")

# %% [markdown]
# ## 8. Sanity diagnostics  (reads back from saved parquet — no large frames in RAM)

# %%
# Load only the columns needed for diagnostics — no full feature matrix in RAM
diag_cols = ["label_name", "first_ts", "last_ts", "hurst_dfa",
             "not_in_shortlist_ratio", "tx_per_block_max", "tx_index_low_ratio",
             "days_since_last_activity"]
feat_saved   = pd.read_parquet(OUT_DIR / "features_raw.parquet")
n_feat_rows  = len(feat_saved)
n_feat_cols  = feat_saved.shape[1] - 3   # subtract address/label/label_name
avail_diag   = [c for c in diag_cols if c in feat_saved.columns]
feat_diag    = feat_saved[["label_name"] + [c for c in avail_diag if c != "label_name"]]
del feat_saved; gc.collect()

print("\n=== Class distribution ===")
print(feat_diag["label_name"].value_counts().to_string())
print(f"\n=== Feature count (features_raw): {n_feat_cols} ===")
print("    (behavioral scores excluded — in behavioral_scores.parquet)")

if "days_since_last_activity" in feat_diag.columns:
    nan_frac = feat_diag["days_since_last_activity"].isna().mean()
    print(f"\n=== days_since_last_activity NaN rate: {nan_frac:.1%} ===")

print("\n=== Calendar coverage per class (first_ts/last_ts = true "
      "address-lifetime span, full history, no truncation) ===")
for cname in CLASS_NAMES:
    sub = feat_diag[feat_diag["label_name"] == cname]
    if len(sub) == 0: continue
    if "first_ts" not in sub.columns:
        print(f"  {cname:>22s}: no timestamp data"); continue
    ft = sub["first_ts"].replace(0, np.nan).dropna()
    lt = sub["last_ts"].replace(0, np.nan).dropna()
    if ft.empty:
        print(f"  {cname:>22s}: no timestamp data"); continue
    ft_min = datetime.fromtimestamp(ft.min(), tz=timezone.utc).date()
    lt_max = datetime.fromtimestamp(lt.max(), tz=timezone.utc).date()
    print(f"  {cname:>22s}: {ft_min} → {lt_max}  n={len(sub)}")

print("\n=== Hurst DFA per class ===")
if "hurst_dfa" in feat_diag.columns:
    print(feat_diag.groupby("label_name")["hurst_dfa"]
                   .agg(["mean", "std", "median"]).round(3).to_string())

print("\n=== not_in_shortlist_ratio per class ===")
if "not_in_shortlist_ratio" in feat_diag.columns:
    print(feat_diag.groupby("label_name")["not_in_shortlist_ratio"]
                   .agg(["mean", "median"]).round(3).to_string())

print("\n=== tx_per_block_max per class (flash-loan signal) ===")
if "tx_per_block_max" in feat_diag.columns:
    print(feat_diag.groupby("label_name")["tx_per_block_max"]
                   .agg(["mean", "median", "max"]).round(2).to_string())

print("\n=== tx_index_low_ratio per class (MEV signal) ===")
if "tx_index_low_ratio" in feat_diag.columns:
    print(feat_diag.groupby("label_name")["tx_index_low_ratio"]
                   .agg(["mean", "median"]).round(3).to_string())

del feat_diag; gc.collect()

# Edge-type distribution from streaming counters (no reload needed)
if edge_type_counts:
    print("\n=== Edge type distribution ===")
    for et, cnt in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {et:>10s}: {cnt:>10,}")

# Sequence stats captured before seq_df was deleted
if seq_stats:
    print(f"\n=== Sequence length stats ===")
    print(f"  mean={seq_stats['mean']:.1f}  median={seq_stats['median']:.0f}  "
          f"p90={seq_stats['p90']:.0f}  max={seq_stats['max']}")

# %% [markdown]
# ## 9.  Visualisations (Q1-publication quality)
#
# All vis read from the saved parquets with only the columns they need.
# feats_df and seq_df are gone from RAM; edges are read as 2-column slices.

# %%
print("\n[9] Generating visualisations …")
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})
CLASS_PALETTE = {
    "benign": "#4CAF50", "phishing": "#F44336", "rug_pull": "#FF9800",
    "ponzi": "#9C27B0", "flash_loan_attack": "#00BCD4",
    "malicious_mev": "#E91E63", "exploit_contract": "#FF5722",
    "honeypot": "#795548",
}

# Load only the columns needed across vis 1–9 (~150 MB instead of full 228-col frame)
_vis_cols = [
    "label_name", "tx_count", "lifespan_days", "err_rate",
    "gas_price_gwei_p99", "method_entropy", "hurst_dfa",
    "erc20_tx_count", "erc721_tx_count", "erc1155_tx_count",
    "burstiness", "tx_index_low_ratio", "approval_count",
]
_avail    = set(pq.read_schema(OUT_DIR / "features_raw.parquet").names)
_vis_load = [c for c in _vis_cols if c in _avail]
vdf = pd.read_parquet(OUT_DIR / "features_raw.parquet", columns=_vis_load)

# ── VIS 1: Class distribution bar chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
vc   = vdf["label_name"].value_counts()
bars = ax.bar(vc.index, vc.values,
              color=[CLASS_PALETTE.get(c, "grey") for c in vc.index])
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_xlabel("Attack Class"); ax.set_ylabel("Address Count")
ax.set_title("Dataset Class Distribution (imbalanced — preserved by design)")
plt.xticks(rotation=25, ha="right"); plt.tight_layout()
plt.savefig(FIG_DIR / "vis1_class_distribution.png", bbox_inches="tight")
plt.close(); print("  vis1 saved")

# ── VIS 2: Transaction count violin per class ─────────────────────────────────
if "tx_count" in vdf.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    labels  = [cn for cn in CLASS_NAMES if (vdf["label_name"] == cn).sum() > 0]
    data    = [vdf[vdf["label_name"] == cn]["tx_count"].clip(0, 500).values for cn in labels]
    vp = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=False)
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(CLASS_PALETTE.get(labels[i], "grey")); pc.set_alpha(0.7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("tx_count (clipped at 500)")
    ax.set_title("Transaction Count Distribution by Class"); plt.tight_layout()
    plt.savefig(FIG_DIR / "vis2_tx_count_violin.png", bbox_inches="tight")
    plt.close(); print("  vis2 saved")

# ── VIS 3: Lifespan distribution violin ──────────────────────────────────────
if "lifespan_days" in vdf.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [cn for cn in CLASS_NAMES if (vdf["label_name"] == cn).sum() > 0]
    data   = [np.log1p(vdf[vdf["label_name"] == cn]["lifespan_days"].clip(0).values) for cn in labels]
    vp = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=False)
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(CLASS_PALETTE.get(labels[i], "grey")); pc.set_alpha(0.7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("log1p(lifespan_days)")
    ax.set_title("Address Lifespan Distribution by Class (log-scale)"); plt.tight_layout()
    plt.savefig(FIG_DIR / "vis3_lifespan_violin.png", bbox_inches="tight")
    plt.close(); print("  vis3 saved")

# ── VIS 4: Error rate by class ────────────────────────────────────────────────
if "err_rate" in vdf.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    means = vdf.groupby("label_name")["err_rate"].mean().reindex(CLASS_NAMES).fillna(0)
    stds  = vdf.groupby("label_name")["err_rate"].std().reindex(CLASS_NAMES).fillna(0)
    ax.bar(means.index, means.values, yerr=stds.values, capsize=4,
           color=[CLASS_PALETTE.get(c, "grey") for c in means.index],
           error_kw=dict(elinewidth=1.2))
    ax.set_ylabel("Mean Error Rate"); ax.set_xlabel("Attack Class")
    ax.set_title("Mean Transaction Error Rate by Class (± std)")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(FIG_DIR / "vis4_error_rate_bar.png", bbox_inches="tight")
    plt.close(); print("  vis4 saved")

# ── VIS 5: Gas price p99 per class ────────────────────────────────────────────
if "gas_price_gwei_p99" in vdf.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [cn for cn in CLASS_NAMES if (vdf["label_name"] == cn).sum() > 0]
    data   = [np.log1p(vdf[vdf["label_name"] == cn]["gas_price_gwei_p99"].clip(0).values) for cn in labels]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, cn in zip(bp["boxes"], labels):
        patch.set_facecolor(CLASS_PALETTE.get(cn, "grey")); patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("log1p(gas_price_gwei_p99)")
    ax.set_title("Gas Price P99 Distribution by Class (MEV/Flash-loan signal)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vis5_gas_price_p99_box.png", bbox_inches="tight")
    plt.close(); print("  vis5 saved")

# ── VIS 6: Method entropy and Hurst DFA ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, feat, title in zip(axes,
        ["method_entropy", "hurst_dfa"],
        ["Method Entropy (call diversity)", "Hurst DFA (temporal persistence)"]):
    if feat not in vdf.columns: continue
    labels = [cn for cn in CLASS_NAMES if (vdf["label_name"] == cn).sum() > 0]
    data   = [vdf[vdf["label_name"] == cn][feat].dropna().values for cn in labels]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, cn in zip(bp["boxes"], labels):
        patch.set_facecolor(CLASS_PALETTE.get(cn, "grey")); patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8); ax.set_title(title, fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis6_method_entropy_hurst.png", bbox_inches="tight")
plt.close(); print("  vis6 saved")

# ── VIS 7: ERC activity stacked bar ──────────────────────────────────────────
erc_cols = [c for c in ["erc20_tx_count", "erc721_tx_count", "erc1155_tx_count"] if c in vdf.columns]
if erc_cols:
    erc_means = vdf.groupby("label_name")[erc_cols].mean().reindex(CLASS_NAMES).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom  = np.zeros(len(CLASS_NAMES))
    for col, color in zip(erc_cols, ["#2196F3", "#FF9800", "#9C27B0"]):
        ax.bar(erc_means.index, erc_means[col].values, bottom=bottom,
               label=col.replace("_tx_count", ""), color=color, alpha=0.8)
        bottom += erc_means[col].values
    ax.set_ylabel("Mean Token Tx Count"); ax.set_title("ERC Token Activity Distribution by Class")
    ax.legend(); plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(FIG_DIR / "vis7_erc_activity_stacked.png", bbox_inches="tight")
    plt.close(); print("  vis7 saved")

# ── VIS 8: Burstiness vs tx_index_low_ratio scatter ──────────────────────────
if "burstiness" in vdf.columns and "tx_index_low_ratio" in vdf.columns:
    fig, ax = plt.subplots(figsize=(9, 7))
    for cname in CLASS_NAMES:
        sub = vdf[vdf["label_name"] == cname]
        if len(sub) < 5: continue
        s = sub.sample(min(len(sub), 300), random_state=42)
        ax.scatter(s["burstiness"], s["tx_index_low_ratio"],
                   alpha=0.45, s=18, label=cname, color=CLASS_PALETTE.get(cname, "grey"))
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Burstiness Index (B)"); ax.set_ylabel("Tx-Index-Low Ratio (top-of-block rate)")
    ax.set_title("Burstiness vs Top-of-Block Rate (MEV/Flash signal)")
    ax.legend(fontsize=8, markerscale=1.5); plt.tight_layout()
    plt.savefig(FIG_DIR / "vis8_burstiness_vs_txindex.png", bbox_inches="tight")
    plt.close(); print("  vis8 saved")

# ── VIS 9: Failed tx & approval count per class ───────────────────────────────
vis9_feats = [c for c in ["failed_tx_count", "approval_count"] if c in vdf.columns]
if vis9_feats:
    n_panels = len(vis9_feats)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(7 * n_panels, 5))
    if n_panels == 1: axes = [axes]
    titles = {"failed_tx_count": "Failed Tx Count (exploit probing signal)",
              "approval_count":  "Approval Count (phishing signal)"}
    for ax, feat in zip(axes, vis9_feats):
        labels = [cn for cn in CLASS_NAMES if (vdf["label_name"] == cn).sum() > 0]
        data   = [np.log1p(vdf[vdf["label_name"] == cn][feat].clip(0).values) for cn in labels]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, cn in zip(bp["boxes"], labels):
            patch.set_facecolor(CLASS_PALETTE.get(cn, "grey")); patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("log1p(count)"); ax.set_title(titles.get(feat, feat), fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vis9_failed_and_approvals.png", bbox_inches="tight")
    plt.close(); print("  vis9 saved")

del vdf; gc.collect()

# ── VIS 10: Edge type distribution (uses 2-col slice from saved parquet) ──────
edge_out_path = OUT_DIR / "edges_raw.parquet"
if edge_out_path.exists():
    # Only load the 2 columns needed — keeps RAM ~50-100 MB even for 10M+ rows
    edges_slim = pd.read_parquet(edge_out_path, columns=["edge_type", "src_label_name"])
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

    etype_vc = edges_slim["edge_type"].value_counts()
    pie_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    axes[0].pie(etype_vc.values, labels=etype_vc.index,
                autopct="%1.1f%%", startangle=140,
                colors=pie_colors[:len(etype_vc)])
    axes[0].set_title("Overall Edge Type Distribution")

    edge_by_class = (edges_slim.groupby(["src_label_name", "edge_type"])
                                .size().unstack(fill_value=0)
                                .reindex(CLASS_NAMES, fill_value=0))
    edge_by_class.plot(kind="bar", ax=axes[1], stacked=True,
                       color=pie_colors[:len(edge_by_class.columns)], edgecolor="none")
    axes[1].set_xlabel("Class"); axes[1].set_ylabel("Edge Count")
    axes[1].set_title("Edge Count per Class by Type")
    axes[1].legend(fontsize=8); plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vis10_edge_distribution.png", bbox_inches="tight")
    plt.close(); print("  vis10 saved")
    del edges_slim; gc.collect()

print(f"\n  Figures saved: {len(list(FIG_DIR.glob('*.png')))} PNGs → {FIG_DIR}")

print(f"""
═══════════════════════════════════════════════════════════════════════════════
PRECONDITIONS FOR NB4 (FEATURE ENGINEERING)
═══════════════════════════════════════════════════════════════════════════════
1. LOG-TRANSFORM required before GNN node init (GATv2 attention scale-sensitive):
   apply log1p to: val_out_total_eth, val_in_total_eth, abs(val_net_eth),
   gas_used_total, gas_eth_total, bytecode_size, balance_eth,
   tx_count, erc20_tx_count, erc721_tx_count, internal_tx_count, unique_blocks
   Tree models (XGBoost, LightGBM) are scale-invariant — skip for those.

2. IMPUTE days_since_last_activity (NaN = collection_ts unknown).
   Use GLOBAL median + binary has_collection_ts flag.
   Class-conditional median imputation deferred to NB5 (post train/test split).

3. DROP first_ts and last_ts from the model feature matrix AFTER NB5 uses them
   to assign chronological cohorts. They are not features — they are window
   labels that would let the model memorize calendar time directly.

4. behavioral_scores.parquet must NOT be joined into features_raw before
   model training. Use only for: (a) NB12 ablation, (b) NB7 LLM prompt.

5. edges_raw.parquet now includes erc1155 edge_type rows (new in v4).
   NB6 graph builder should handle edge_type in {"normal","internal","erc20","erc721","erc1155"}.

6. NEW (v6): first_ts/last_ts are recomputed from the union of all six raw
   record streams (was: normal-tx-only), giving a more robust lifetime span
   per address. All features still reflect FULL on-chain history through
   collection time — no observation-window truncation, zero data loss.
   NB5 must use first_ts/last_ts to perform a PER-CLASS stratified 80/20
   temporal split (each class independently sorted by first_ts, earliest
   80% -> train, latest 20% -> test) to avoid global class-composition skew,
   then drop first_ts/last_ts from the model feature matrix.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FILES
═══════════════════════════════════════════════════════════════════════════════
  {OUT_DIR}/features_raw.parquet
  {OUT_DIR}/behavioral_scores.parquet
  {OUT_DIR}/behavioral_cards.parquet
  {OUT_DIR}/temporal_sequences.parquet
  {OUT_DIR}/edges_raw.parquet
  {FIG_DIR}/*.png  (10 publication-quality figures)

[done] All outputs ready for Notebook 4 (Feature Engineering).
""")
