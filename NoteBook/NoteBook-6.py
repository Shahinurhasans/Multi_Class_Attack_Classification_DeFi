# %% [markdown]
# # Notebook 6 — Heterogeneous Graph Construction  (Q1-Grade, Temporal)
#
# **Inputs** (from NB4 + NB5):
# | File | Description |
# |---|---|
# | `processed_dataset.parquet` | 200 scaled node features, labels, split (42892 nodes) |
# | `edges_engineered.parquet`  | 34M directed edges × 24 features across 5 relation types |
# | `node_type_map.parquet`     | Node-type annotations (bytecode tier, proxy, multisig) |
# | `split_meta.json`           | Selected feature list, class weights, split metadata |
#
# **Outputs**:
# | File | Description |
# |---|---|
# | `graph_data.pt`    | PyG HeteroData: node feats/labels/masks + 5 edge relations |
# | `graph_meta.json`  | Graph statistics, edge counts, feature dims for NB7 |
#
# ──────────────────────────────────────────────────────────────────────────────
# ## GRAPH DESIGN
# ──────────────────────────────────────────────────────────────────────────────
#
#  Node type : "addr"  (single type — 42892 labeled Ethereum addresses)
#              is_contract = 0 for all nodes (master_dataset.csv absent in NB3,
#              so bytecode_size = 0 → dropped as zero-variance in NB4).
#              Node feature dim: 200 (RobustScaled, from processed_dataset).
#
#  Edge relation types (5 heterogeneous relations):
#    ("addr", "normal",   "addr")  — standard ETH transfers
#    ("addr", "internal", "addr")  — internal / smart-contract calls
#    ("addr", "erc20",    "addr")  — ERC-20 token transfers
#    ("addr", "erc721",   "addr")  — ERC-721 NFT transfers
#    ("addr", "erc1155",  "addr")  — ERC-1155 multi-token transfers
#
#  Edge scope: anchor_address (src) is always a labeled node. Edges where
#  the counterpart (dst) is NOT in the labeled universe are dropped — these
#  correspond to DEX contracts, bridges, validators etc. that lack features.
#  Edges between labeled nodes carry full structural + temporal signal.
#
#  Edge features: all numeric columns in edges_engineered.parquet except
#  address identifiers, edge_type, label columns, tx_hash, block_number,
#  timestamp. Stored as float16 to halve memory footprint.
#  Includes: temporal_decay_weight, same_block_flag (NB4 Group J).
#
#  Train/test masks: from processed_dataset["split"]. Test nodes ARE present
#  in the graph so NB7 can use inductive neighbor-sampling (test nodes attend
#  to train neighbors via read-only propagation — no label leakage).
#
#  LEAKAGE GUARDRAILS:
#    • src_label_name / dst_label_name / anchor_label_name excluded from
#      edge features (encodes ground-truth class — direct leakage).
#    • Node features are already leakage-free (NB5 guarantee).
#    • Temporal order: temporal_decay_weight in edge_attr encodes recency;
#      NB7 should exploit this for temporal attention, not use raw timestamps.

# %%
import gc, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch Geometric — install if missing (Kaggle GPU usually has it)
try:
    from torch_geometric.data import HeteroData
    import torch_geometric
    HAS_PYG = True
    print(f"PyTorch Geometric {torch_geometric.__version__}: available")
except ImportError:
    try:
        import subprocess, sys
        print("Installing torch-geometric …")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "torch-geometric", "-q"], check=True)
        from torch_geometric.data import HeteroData
        HAS_PYG = True
        print("PyTorch Geometric installed and imported")
    except Exception as _e:
        HAS_PYG = False
        print(f"[warn] PyTorch Geometric unavailable ({_e}) — saving as npz fallback")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

FEAT_DIR = Path("/kaggle/working/features")
OUT_DIR  = Path("/kaggle/working/features")
FIG_DIR  = Path("/kaggle/working/figures_nb6")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EDGE_TYPES  = ["normal", "internal", "erc20", "erc721", "erc1155"]
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

print("=== Notebook 6 — Heterogeneous Graph Construction ===")
print(f"Input  : {FEAT_DIR}")
print(f"Output : {OUT_DIR}")


# %% [markdown]
# ## 1.  Load node data & build address → index mapping

# %%
print("\n[1] Loading node data …")

with open(FEAT_DIR / "split_meta.json") as fh:
    split_meta = json.load(fh)
SELECTED_FEATURES = split_meta["selected_features"]
CLASS_WEIGHTS     = split_meta["class_weights"]

node_df = pd.read_parquet(FEAT_DIR / "processed_dataset.parquet")
node_df = node_df.reset_index(drop=True)
print(f"    node_df shape: {node_df.shape}")
print(f"    split: {node_df['split'].value_counts().to_dict()}")
print(f"    node features: {len(SELECTED_FEATURES)}")

# Address → node index (deterministic: row order in node_df)
addr_to_idx = {addr: int(idx)
               for idx, addr in enumerate(node_df["address"].values)}
N_NODES = len(node_df)
print(f"    N_NODES = {N_NODES}")

# Node feature matrix (float32)
X_node = node_df[SELECTED_FEATURES].values.astype(np.float32)
y_node = node_df["label"].values.astype(np.int64)

# Per-class label vector on node_df for downstream visualizations
node_label_name = node_df["label_name"].values

train_mask = (node_df["split"] == "train").values
test_mask  = (node_df["split"] == "test").values
print(f"    train={train_mask.sum():,}  test={test_mask.sum():,}")

# Load node_type_map (for statistics + NB7 metadata)
node_type_df = pd.read_parquet(FEAT_DIR / "node_type_map.parquet")
node_type_df = node_type_df.reset_index(drop=True)
print(f"\n    node_type_map shape: {node_type_df.shape}")
print(f"    node_type_map columns: {list(node_type_df.columns)}")

gc.collect()


# %% [markdown]
# ## 2.  Load edges & identify feature columns

# %%
print("\n[2] Loading edges_engineered.parquet …")
edges = pd.read_parquet(FEAT_DIR / "edges_engineered.parquet")
print(f"    edges shape: {edges.shape}")
print(f"    columns: {list(edges.columns)}")

# edge_type may be Categorical → string for mask comparisons
if hasattr(edges["edge_type"], "cat"):
    edges["edge_type"] = edges["edge_type"].astype(str)

print(f"\n    edge_type distribution:")
print(edges["edge_type"].value_counts().to_string())

# ── Detect src / dst address columns ─────────────────────────────────────────
_src_candidates = ["anchor_address", "src", "from_address", "from", "source"]
_dst_candidates = ["counterpart_address", "dst", "to_address", "to", "target"]
SRC_COL = next((c for c in _src_candidates if c in edges.columns), None)
DST_COL = next((c for c in _dst_candidates if c in edges.columns), None)
if SRC_COL is None or DST_COL is None:
    raise ValueError(f"Cannot auto-detect src/dst columns from {list(edges.columns)}")
print(f"\n    src column: {SRC_COL}  |  dst column: {DST_COL}")

# ── Edge feature columns ──────────────────────────────────────────────────────
# Exclude: address identifiers, edge_type, metadata, label columns (leakage)
_exclude = {
    SRC_COL, DST_COL, "edge_type",
    "tx_hash", "block_number", "timestamp", "nonce",
    # leakage: label columns encode ground truth
    "src_label", "dst_label", "src_label_name", "dst_label_name",
    "anchor_label", "counterpart_label",
    "anchor_label_name", "counterpart_label_name",
    "label", "label_name",
}
EDGE_FEAT_COLS = [
    c for c in edges.columns
    if c not in _exclude
    and edges[c].dtype.kind in ("f", "i", "u", "b")  # numeric types only
]
print(f"\n    Edge feature columns ({len(EDGE_FEAT_COLS)}): {EDGE_FEAT_COLS}")


# %% [markdown]
# ## 3.  Filter edges to labeled-node universe & map to indices

# %%
print("\n[3] Filtering edges to labeled-universe endpoints …")

# anchor_address (src) is always a labeled node (NB3 extracted per labeled address).
# counterpart_address (dst) may be outside the universe (DEX contracts, bridges, etc.)
_all_addrs = set(addr_to_idx.keys())

# src check (should be 100% in universe)
src_in = edges[SRC_COL].isin(_all_addrs)
n_src_out = (~src_in).sum()
if n_src_out > 0:
    print(f"    [warn] {n_src_out:,} edges have src outside universe (unexpected) — dropping")

# dst filter
dst_in = edges[DST_COL].isin(_all_addrs)
n_dst_out = (~dst_in).sum()
print(f"    Edges with dst outside labeled universe: {n_dst_out:,} "
      f"({n_dst_out / len(edges) * 100:.1f}%)")

keep_mask = src_in & dst_in
n_kept = keep_mask.sum()
print(f"    Edges retained (both endpoints labeled): {n_kept:,} "
      f"({n_kept / len(edges) * 100:.1f}%)")

edges = edges[keep_mask].copy()
del src_in, dst_in, keep_mask; gc.collect()

# Map address strings → int32 node indices
print("\n    Mapping addresses → node indices …")
edges["_src_idx"] = edges[SRC_COL].map(addr_to_idx).astype(np.int32)
edges["_dst_idx"] = edges[DST_COL].map(addr_to_idx).astype(np.int32)

# Sanity: no NaN indices
assert edges["_src_idx"].notna().all() and edges["_dst_idx"].notna().all(), \
    "NaN in node indices after map — check address format"

print(f"    src idx range: [{edges['_src_idx'].min()}, {edges['_src_idx'].max()}]")
print(f"    dst idx range: [{edges['_dst_idx'].min()}, {edges['_dst_idx'].max()}]")

gc.collect()


# %% [markdown]
# ## 4.  Build per-relation-type edge tensors

# %%
print("\n[4] Building per-relation-type edge tensors …")

# Node index → label_name for edge-class analysis
idx_to_label = dict(zip(range(N_NODES), node_label_name))

edge_data = {}   # {etype: {"edge_index": (2,E) int32, "edge_attr": (E,F) float16, "n_edges": int}}

total_edges_built = 0
for etype in EDGE_TYPES:
    mask = (edges["edge_type"] == etype).values
    n_e  = mask.sum()
    if n_e == 0:
        print(f"    {etype:12s}: 0 edges — skipping")
        continue

    sub_src  = edges["_src_idx"].values[mask]   # int32
    sub_dst  = edges["_dst_idx"].values[mask]   # int32
    sub_feat = edges[EDGE_FEAT_COLS].values[mask].astype(np.float32)

    # Clean any inf/-inf in edge features before float16 cast
    _bad = ~np.isfinite(sub_feat)
    if _bad.any():
        sub_feat[_bad] = 0.0

    sub_feat_f16 = sub_feat.astype(np.float16)
    del sub_feat; gc.collect()

    edge_index = np.stack([sub_src, sub_dst], axis=0)   # (2, E) int32

    edge_data[etype] = {
        "edge_index": edge_index,    # shape (2, E)
        "edge_attr":  sub_feat_f16,  # shape (E, F)
        "n_edges":    int(n_e),
    }

    mem_mb = (edge_index.nbytes + sub_feat_f16.nbytes) / 1e6
    print(f"    {etype:12s}: {n_e:8,} edges | "
          f"feat {sub_feat_f16.shape[1]}D | mem={mem_mb:.0f} MB")

    total_edges_built += n_e
    del sub_src, sub_dst, sub_feat_f16, edge_index; gc.collect()

N_EDGE_FEAT   = len(EDGE_FEAT_COLS)
BUILT_ETYPES  = list(edge_data.keys())
print(f"\n    Total labeled-universe edges: {total_edges_built:,}")
print(f"    Edge feature dim: {N_EDGE_FEAT}")
print(f"    Relation types with edges: {BUILT_ETYPES}")

# Free the filtered edges dataframe — no longer needed
del edges; gc.collect()


# %% [markdown]
# ## 5.  Build PyG HeteroData (or numpy fallback)

# %%
print("\n[5] Building graph object …")

if HAS_PYG:
    data = HeteroData()

    # ── Node tensors ──────────────────────────────────────────────────────────
    data["addr"].x          = torch.from_numpy(X_node)            # (N, 200) float32
    data["addr"].y          = torch.from_numpy(y_node)            # (N,) int64
    data["addr"].train_mask = torch.from_numpy(train_mask)        # (N,) bool
    data["addr"].test_mask  = torch.from_numpy(test_mask)         # (N,) bool
    data["addr"].num_nodes  = N_NODES

    # ── Edge tensors ──────────────────────────────────────────────────────────
    for etype, ed in edge_data.items():
        rel = ("addr", etype, "addr")
        data[rel].edge_index = torch.from_numpy(
            ed["edge_index"].astype(np.int64))                    # (2, E) int64
        data[rel].edge_attr  = torch.from_numpy(
            ed["edge_attr"])                                       # (E, F) float16

    # ── Validate ──────────────────────────────────────────────────────────────
    print(f"    HeteroData summary:")
    print(f"      Node types : {data.node_types}")
    print(f"      Edge types : {len(data.edge_types)}")
    print(f"      data['addr'].x.shape  : {data['addr'].x.shape}")
    print(f"      data['addr'].y.shape  : {data['addr'].y.shape}")
    print(f"      train/test mask       : {data['addr'].train_mask.sum().item():,} / "
          f"{data['addr'].test_mask.sum().item():,}")
    print()
    for et in data.edge_types:
        n_e = data[et].edge_index.shape[1]
        f_e = data[et].edge_attr.shape[1]
        print(f"      {str(et):45s}: {n_e:9,} edges  {f_e}D attr")

    max_idx = max(data[et].edge_index.max().item() for et in data.edge_types)
    assert max_idx < N_NODES, f"Edge index {max_idx} >= N_NODES {N_NODES}"
    print(f"\n    All edge indices in [0, {N_NODES - 1}] ✓")

    torch.save(data, OUT_DIR / "graph_data.pt")
    print(f"    Saved: graph_data.pt")
    GRAPH_FORMAT = "HeteroData"

else:
    # ── Numpy fallback ────────────────────────────────────────────────────────
    save_dict = {
        "node_x":       X_node,
        "node_y":       y_node,
        "train_mask":   train_mask,
        "test_mask":    test_mask,
        "edge_feat_cols_json": np.array(json.dumps(EDGE_FEAT_COLS)),
    }
    for etype, ed in edge_data.items():
        save_dict[f"edge_index_{etype}"] = ed["edge_index"]
        save_dict[f"edge_attr_{etype}"]  = ed["edge_attr"]
    np.savez_compressed(OUT_DIR / "graph_data.npz", **save_dict)
    print("    Saved: graph_data.npz (PyG unavailable — NB7 will load with np.load)")
    GRAPH_FORMAT = "npz"

gc.collect()


# %% [markdown]
# ## 6.  Compute graph statistics & save graph_meta.json

# %%
print("\n[6] Computing graph statistics …")

# Per-edge-type degree stats
degree_stats = {}
total_in_deg  = np.zeros(N_NODES, dtype=np.int64)
total_out_deg = np.zeros(N_NODES, dtype=np.int64)

for etype, ed in edge_data.items():
    src_arr = ed["edge_index"][0].astype(np.int64)
    dst_arr = ed["edge_index"][1].astype(np.int64)
    in_d    = np.bincount(dst_arr, minlength=N_NODES)
    out_d   = np.bincount(src_arr, minlength=N_NODES)
    total_in_deg  += in_d
    total_out_deg += out_d

    # Reciprocity: fraction of edges (u,v) where (v,u) also exists
    pairs_fwd  = set(zip(src_arr[:100000].tolist(), dst_arr[:100000].tolist()))
    pairs_rev  = set(zip(dst_arr[:100000].tolist(), src_arr[:100000].tolist()))
    reciprocal = len(pairs_fwd & pairs_rev) / max(len(pairs_fwd), 1)

    degree_stats[etype] = {
        "n_edges":       int(ed["n_edges"]),
        "in_deg_mean":   float(in_d.mean()),
        "in_deg_max":    int(in_d.max()),
        "in_deg_median": float(np.median(in_d)),
        "out_deg_mean":  float(out_d.mean()),
        "out_deg_max":   int(out_d.max()),
        "out_deg_median":float(np.median(out_d)),
        "reciprocity_sample": float(reciprocal),
    }
    print(f"    {etype:12s}: in_deg_mean={in_d.mean():.2f}  "
          f"out_deg_mean={out_d.mean():.2f}  "
          f"reciprocity≈{reciprocal:.3f}")

# Class-level degree stats
print("\n    Class-level total degree (train only):")
class_deg_stats = {}
for cn in CLASS_NAMES:
    mask_c = (node_label_name == cn) & train_mask
    if mask_c.sum() == 0:
        continue
    in_d_c  = total_in_deg[mask_c]
    out_d_c = total_out_deg[mask_c]
    class_deg_stats[cn] = {
        "n": int(mask_c.sum()),
        "mean_in":    float(in_d_c.mean()),
        "mean_out":   float(out_d_c.mean()),
        "median_in":  float(np.median(in_d_c)),
        "median_out": float(np.median(out_d_c)),
    }
    print(f"      {cn:22s}: n={mask_c.sum():5d}  "
          f"in={in_d_c.mean():.1f}  out={out_d_c.mean():.1f}")

graph_meta = {
    "n_nodes":           N_NODES,
    "n_node_features":   len(SELECTED_FEATURES),
    "n_edge_features":   N_EDGE_FEAT,
    "n_classes":         len(CLASS_NAMES),
    "class_names":       CLASS_NAMES,
    "node_type":         "addr",
    "edge_types":        BUILT_ETYPES,
    "edge_feat_cols":    EDGE_FEAT_COLS,
    "selected_features": SELECTED_FEATURES,
    "edge_counts":       {et: degree_stats[et]["n_edges"] for et in degree_stats},
    "total_edges":       total_edges_built,
    "degree_stats":      degree_stats,
    "class_degree_stats":class_deg_stats,
    "train_nodes":       int(train_mask.sum()),
    "test_nodes":        int(test_mask.sum()),
    "class_counts":      node_df["label_name"].value_counts().to_dict(),
    "class_weights":     CLASS_WEIGHTS,
    "pyg_available":     HAS_PYG,
    "graph_format":      GRAPH_FORMAT,
    "src_col":           SRC_COL,
    "dst_col":           DST_COL,
}
with open(OUT_DIR / "graph_meta.json", "w") as fh:
    json.dump(graph_meta, fh, indent=2, cls=NumpyEncoder)
print("\n    graph_meta.json saved")
gc.collect()


# %% [markdown]
# ## 7.  Visualisations (10 Q1-publication figures)

# %%
print("\n[7] Generating visualisations …")
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── VIS 1: Graph overview — node/edge counts ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: class distribution
vc = node_df["label_name"].value_counts().reindex(CLASS_NAMES).fillna(0)
axes[0].barh(CLASS_NAMES, vc.values,
             color=[CLASS_PALETTE.get(c, "grey") for c in CLASS_NAMES])
axes[0].set_title("Node Count by Class", fontsize=10)
axes[0].set_xlabel("Count")

# Panel 2: edge count by type
et_names = list(degree_stats.keys())
et_counts = [degree_stats[et]["n_edges"] for et in et_names]
colors_et = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
bars = axes[1].bar(et_names, et_counts,
                   color=colors_et[:len(et_names)], edgecolor="none")
axes[1].bar_label(bars, fmt="%d", padding=3, fontsize=8)
axes[1].set_title("Labeled-Universe Edge Count by Type", fontsize=10)
axes[1].set_ylabel("Count")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

# Panel 3: avg total degree by class
cn_list = [c for c in CLASS_NAMES if c in class_deg_stats]
mean_in  = [class_deg_stats[c]["mean_in"]  for c in cn_list]
mean_out = [class_deg_stats[c]["mean_out"] for c in cn_list]
x = np.arange(len(cn_list))
w = 0.4
axes[2].bar(x - w/2, mean_in,  width=w, label="In-degree",  color="#42A5F5")
axes[2].bar(x + w/2, mean_out, width=w, label="Out-degree", color="#EF5350")
axes[2].set_xticks(x)
axes[2].set_xticklabels(cn_list, rotation=30, ha="right", fontsize=8)
axes[2].set_title("Mean Total Degree by Class (train)", fontsize=10)
axes[2].set_ylabel("Mean degree")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "vis1_graph_overview.png", bbox_inches="tight")
plt.close()
print("  vis1 saved")


# ── VIS 2: Degree distribution CDF (log scale) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, deg_arr, title in [
    (axes[0], total_in_deg,  "In-degree CDF (all relations)"),
    (axes[1], total_out_deg, "Out-degree CDF (all relations)"),
]:
    deg_nz = deg_arr[deg_arr > 0]
    if len(deg_nz) == 0:
        ax.set_title(title); continue
    sorted_d = np.sort(deg_nz)
    cdf = np.arange(1, len(sorted_d) + 1) / len(deg_arr)
    ax.plot(sorted_d, cdf, color="#1565C0", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Degree (log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(title, fontsize=10)
    ax.axvline(np.median(deg_nz), color="red", linestyle="--",
               linewidth=1, label=f"median={np.median(deg_nz):.0f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "vis2_degree_cdf.png", bbox_inches="tight")
plt.close()
print("  vis2 saved")


# ── VIS 3: Class-conditional degree heatmap ───────────────────────────────────
# Build (n_class × n_etype) matrix of mean out-degree per relation type per class
per_class_etype_deg = np.zeros((len(CLASS_NAMES), len(BUILT_ETYPES)))

for j, etype in enumerate(BUILT_ETYPES):
    ed = edge_data[etype]
    src_arr = ed["edge_index"][0]
    out_d_e = np.bincount(src_arr.astype(np.int64), minlength=N_NODES)
    for i, cn in enumerate(CLASS_NAMES):
        mask_c = node_label_name == cn
        if mask_c.sum() > 0:
            per_class_etype_deg[i, j] = out_d_e[mask_c].mean()

fig, ax = plt.subplots(figsize=(10, 6))
df_heat = pd.DataFrame(per_class_etype_deg, index=CLASS_NAMES, columns=BUILT_ETYPES)
sns.heatmap(np.log1p(df_heat.values), annot=df_heat.values.round(1),
            fmt=".1f", cmap="YlOrRd", xticklabels=BUILT_ETYPES,
            yticklabels=CLASS_NAMES, ax=ax, annot_kws={"size": 8},
            cbar_kws={"label": "log1p(mean out-degree)"})
ax.set_title("Mean Out-Degree per Class per Relation Type (log1p scale)", fontsize=11)
ax.set_xlabel("Edge relation type")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis3_class_etype_degree.png", bbox_inches="tight")
plt.close()
print("  vis3 saved")


# ── VIS 4: Edge type composition per source class ─────────────────────────────
# For each class: what fraction of its outgoing edges are each type?
# Build (n_class × n_etype) edge count matrix
class_etype_cnt = np.zeros((len(CLASS_NAMES), len(BUILT_ETYPES)), dtype=np.int64)
for j, etype in enumerate(BUILT_ETYPES):
    ed = edge_data[etype]
    for i, cn in enumerate(CLASS_NAMES):
        mask_c = node_label_name == cn
        src_in_class = np.isin(ed["edge_index"][0], np.where(mask_c)[0])
        class_etype_cnt[i, j] = int(src_in_class.sum())

fig, ax = plt.subplots(figsize=(12, 5))
class_etype_frac = class_etype_cnt / class_etype_cnt.sum(axis=1, keepdims=True).clip(min=1)
bot = np.zeros(len(CLASS_NAMES))
for j, etype in enumerate(BUILT_ETYPES):
    ax.bar(CLASS_NAMES, class_etype_frac[:, j], bottom=bot,
           color=colors_et[j], label=etype, edgecolor="none")
    bot += class_etype_frac[:, j]
ax.set_ylabel("Fraction of outgoing edges")
ax.set_title("Edge Type Composition per Class (outgoing edges)")
ax.legend(fontsize=8, loc="upper right")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "vis4_edge_type_composition.png", bbox_inches="tight")
plt.close()
print("  vis4 saved")


# ── VIS 5: Violin plot of total degree by class (log scale) ───────────────────
total_deg = total_in_deg + total_out_deg

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (deg_arr, title) in zip(axes, [
    (total_in_deg,  "In-degree by Class"),
    (total_out_deg, "Out-degree by Class"),
]):
    class_degs = []
    cls_labels = []
    for cn in CLASS_NAMES:
        mask_c = node_label_name == cn
        d = deg_arr[mask_c]
        d_nz = d[d > 0]
        if len(d_nz) < 3: continue
        class_degs.append(np.log1p(d_nz))
        cls_labels.append(cn)
    if not class_degs: continue
    parts = ax.violinplot(class_degs, showmedians=True)
    for i, (pc, cn) in enumerate(zip(parts["bodies"], cls_labels)):
        pc.set_facecolor(CLASS_PALETTE.get(cn, "grey"))
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(cls_labels) + 1))
    ax.set_xticklabels(cls_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("log1p(degree)")
    ax.set_title(title)
plt.suptitle("Degree Distribution by Class (non-zero nodes, log1p scale)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis5_degree_violin.png", bbox_inches="tight")
plt.close()
print("  vis5 saved")


# ── VIS 6: Temporal decay weight distribution ─────────────────────────────────
tdw_col_idx = (EDGE_FEAT_COLS.index("temporal_decay_weight")
               if "temporal_decay_weight" in EDGE_FEAT_COLS else None)

fig, axes = plt.subplots(1, len(BUILT_ETYPES), figsize=(16, 3), sharey=True)
for ax, etype in zip(axes, BUILT_ETYPES):
    ed = edge_data[etype]
    if tdw_col_idx is not None and ed["edge_attr"].shape[1] > tdw_col_idx:
        tdw = ed["edge_attr"][:, tdw_col_idx].astype(np.float32)
        ax.hist(tdw, bins=40, color="#1976D2", edgecolor="none", density=True, alpha=0.8)
        ax.axvline(float(np.median(tdw)), color="red", linewidth=1, linestyle="--",
                   label=f"med={np.median(tdw):.2f}")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title(etype, fontsize=9)
    ax.set_xlabel("temporal_decay_weight")
axes[0].set_ylabel("Density")
plt.suptitle("Temporal Decay Weight Distribution per Relation Type", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis6_temporal_decay_dist.png", bbox_inches="tight")
plt.close()
print("  vis6 saved")


# ── VIS 7: Same-block-flag rate by class (flash loan / MEV signal) ────────────
sbf_col_idx = (EDGE_FEAT_COLS.index("same_block_flag")
               if "same_block_flag" in EDGE_FEAT_COLS else None)

fig, ax = plt.subplots(figsize=(11, 4))
if sbf_col_idx is not None:
    class_sbf_rate = {}
    for cn in CLASS_NAMES:
        mask_c = node_label_name == cn
        node_ids_c = set(np.where(mask_c)[0].tolist())
        sbf_vals = []
        for etype in BUILT_ETYPES:
            ed = edge_data[etype]
            if ed["edge_attr"].shape[1] <= sbf_col_idx:
                continue
            src_arr = ed["edge_index"][0]
            src_in = np.isin(src_arr, list(node_ids_c))
            sbf = ed["edge_attr"][src_in, sbf_col_idx].astype(np.float32)
            sbf_vals.append(sbf)
        if sbf_vals:
            all_sbf = np.concatenate(sbf_vals)
            class_sbf_rate[cn] = float(all_sbf.mean())
        else:
            class_sbf_rate[cn] = 0.0

    cn_sbf  = list(class_sbf_rate.keys())
    rates   = [class_sbf_rate[c] for c in cn_sbf]
    bar_c   = [CLASS_PALETTE.get(c, "grey") for c in cn_sbf]
    bars    = ax.bar(cn_sbf, rates, color=bar_c, edgecolor="none")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Same-block flag rate")
    ax.set_title("Same-Block-Flag Rate per Source Class\n"
                 "(high rate → flash loan / MEV — multiple txs in one block)")
    plt.xticks(rotation=30, ha="right")
else:
    ax.text(0.5, 0.5, "same_block_flag column not found in edge features",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis7_same_block_flag.png", bbox_inches="tight")
plt.close()
print("  vis7 saved")


# ── VIS 8: Train vs test subgraph comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: node degree for train vs test nodes
for ax, (split_name, mask_s, col) in zip(axes, [
    ("Train", train_mask, "#1976D2"),
    ("Test",  test_mask,  "#E53935"),
]):
    deg = total_in_deg[mask_s] + total_out_deg[mask_s]
    deg_nz = deg[deg > 0]
    if len(deg_nz) == 0: continue
    ax.hist(np.log1p(deg_nz), bins=40, color=col, edgecolor="none",
            alpha=0.8, density=True)
    ax.axvline(np.log1p(np.median(deg_nz)), color="black", linewidth=1.5,
               linestyle="--",
               label=f"median={np.median(deg_nz):.0f}")
    ax.set_xlabel("log1p(total degree)")
    ax.set_ylabel("Density")
    ax.set_title(f"{split_name} Node Degree Distribution")
    ax.legend(fontsize=9)

plt.suptitle("Train vs Test Node Degree (in labeled-universe edges)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis8_train_test_degree.png", bbox_inches="tight")
plt.close()
print("  vis8 saved")


# ── VIS 9: Reciprocity & sparsity per relation type ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Reciprocity (from degree_stats)
recip_rates = [degree_stats.get(et, {}).get("reciprocity_sample", 0)
               for et in BUILT_ETYPES]
bars = axes[0].bar(BUILT_ETYPES, recip_rates,
                   color=colors_et[:len(BUILT_ETYPES)], edgecolor="none")
axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
axes[0].set_title("Reciprocity (sample, first 100K edges)", fontsize=10)
axes[0].set_ylabel("Fraction bidirectional")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")

# Sparsity: edges per node pair possible in labeled universe
n_possible_pairs = N_NODES * (N_NODES - 1)
sparsity_vals = [degree_stats.get(et, {}).get("n_edges", 0) / n_possible_pairs
                 for et in BUILT_ETYPES]
bars2 = axes[1].bar(BUILT_ETYPES, sparsity_vals,
                    color=colors_et[:len(BUILT_ETYPES)], edgecolor="none")
axes[1].bar_label(bars2, fmt="%.2e", padding=3, fontsize=8)
axes[1].set_title("Graph Density (edges / possible pairs)", fontsize=10)
axes[1].set_ylabel("Density (log scale)")
axes[1].set_yscale("log")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

plt.suptitle("Graph Structural Properties per Relation Type", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis9_reciprocity_sparsity.png", bbox_inches="tight")
plt.close()
print("  vis9 saved")


# ── VIS 10: Cross-class edge flow (who transacts with whom?) ───────────────────
# Build (n_class × n_class) edge count matrix aggregated over all relation types
cn_to_idx_vis = {cn: i for i, cn in enumerate(CLASS_NAMES)}
cross_class = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

for etype in BUILT_ETYPES:
    ed = edge_data[etype]
    src_arr = ed["edge_index"][0]
    dst_arr = ed["edge_index"][1]
    src_cn_idx = np.array([cn_to_idx_vis.get(node_label_name[s], -1) for s in src_arr])
    dst_cn_idx = np.array([cn_to_idx_vis.get(node_label_name[d], -1) for d in dst_arr])
    valid = (src_cn_idx >= 0) & (dst_cn_idx >= 0)
    for s_i, d_i in zip(src_cn_idx[valid], dst_cn_idx[valid]):
        cross_class[s_i, d_i] += 1

fig, ax = plt.subplots(figsize=(11, 9))
cross_log = np.log1p(cross_class).astype(float)
# Annotate with actual counts (formatted)
annot = np.array([[f"{v:,.0f}" if v < 1e6 else f"{v/1e6:.1f}M"
                   for v in row] for row in cross_class])
sns.heatmap(cross_log, annot=annot, fmt="", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5, cbar_kws={"label": "log1p(edge count)"},
            annot_kws={"size": 7})
ax.set_xlabel("Destination class")
ax.set_ylabel("Source class")
ax.set_title("Cross-Class Edge Flow (all 5 relation types combined, log1p scale)",
             fontsize=11)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "vis10_cross_class_flow.png", bbox_inches="tight")
plt.close()
print("  vis10 saved")

print(f"\n  Figures saved: {len(list(FIG_DIR.glob('*.png')))} PNGs → {FIG_DIR}")


# %% [markdown]
# ## 8.  Final summary

# %%
print("\n" + "=" * 80)
print("NOTEBOOK 6 — HETEROGENEOUS GRAPH CONSTRUCTION COMPLETE")
print("=" * 80)

print(f"""
  graph_data.{("pt" if HAS_PYG else "npz"):3s}     : {GRAPH_FORMAT} format
  graph_meta.json    : {len(BUILT_ETYPES)} relation types, {total_edges_built:,} total edges
  Figures            : {len(list(FIG_DIR.glob('*.png')))} PNGs in {FIG_DIR}
""")

print(f"""
GRAPH STATISTICS
─────────────────────────────────────────────────────────────────────────────
  Nodes              : {N_NODES:,}  (all labeled Ethereum addresses)
  Node feature dim   : {len(SELECTED_FEATURES)}  (RobustScaled, from NB5)
  Classes            : {len(CLASS_NAMES)}
  Train / Test masks : {int(train_mask.sum()):,} / {int(test_mask.sum()):,}

  Edge relation types and labeled-universe edge counts:""")

for et in BUILT_ETYPES:
    n_e = degree_stats.get(et, {}).get("n_edges", 0)
    print(f"    {et:12s}: {n_e:10,} edges")

print(f"""
  Total labeled-universe edges  : {total_edges_built:,}
  Edge feature dim              : {N_EDGE_FEAT}
  Edge feature columns          : {EDGE_FEAT_COLS}

DESIGN NOTES
─────────────────────────────────────────────────────────────────────────────
  • Single node type "addr" — is_contract=0 for all (master_dataset.csv absent
    in NB3; NB4 dropped bytecode_size as zero-variance). Heterogeneity comes
    entirely from the 5 edge relation types.
  • Edges outside labeled universe dropped (DEX contracts, validators etc.
    lack node features). Ego-network degree features in node_x already
    encode the full neighborhood including out-of-universe counterparts.
  • Edge attrs stored float16 — NB7 HetConv attention should upcast to
    float32 before computing attention coefficients.
  • temporal_decay_weight in edge_attr: use as edge weight in temporal GATv2
    to emphasise recent interactions.

PRECONDITIONS FOR NB7 (TEMPORAL HETGNN + INCREMENTAL LEARNING)
─────────────────────────────────────────────────────────────────────────────
 1. Load graph_data.pt with torch.load(); use data["addr"].x, .y,
    .train_mask, .test_mask for node features, labels, and split masks.

 2. Use HeteroConv / HGTConv from PyG with relation types:
    ("addr", "normal",   "addr"),
    ("addr", "internal", "addr"),
    ("addr", "erc20",    "addr"),
    ("addr", "erc721",   "addr"),
    ("addr", "erc1155",  "addr")

 3. Use class_weights from graph_meta.json["class_weights"] in the
    CrossEntropyLoss (or FocalLoss) to handle 20:1 benign/minority ratio.

 4. Apply neighbor sampling (PyG NeighborLoader) for mini-batch training —
    full-graph GATv2 on 34M edges does not fit in 16 GB GPU memory.

 5. temporal_decay_weight = edge_attr[:, {EDGE_FEAT_COLS.index('temporal_decay_weight') if 'temporal_decay_weight' in EDGE_FEAT_COLS else 'N/A'}]
    Use as edge weight multiplier in temporal attention.

 6. same_block_flag = edge_attr[:, {EDGE_FEAT_COLS.index('same_block_flag') if 'same_block_flag' in EDGE_FEAT_COLS else 'N/A'}]
    Use as a hard relation subtype signal: same_block_flag=1 edges indicate
    atomic flash-loan / MEV bundles — crucial for class {CLASS_NAMES.index('flash_loan_attack')} and {CLASS_NAMES.index('malicious_mev')}.

 7. Incremental learning: time-ordered addresses — sort by node_df["first_ts"].
    Add newest ~10% of train nodes as a "new batch" to simulate concept drift.
    Freeze lower GNN layers; fine-tune upper + classifier head.

 8. LLM governance: use Gemma-2B or Mistral-7B (free) via HuggingFace
    inference API. Feed: behavioral_scores.parquet features + GNN embedding
    + class prediction → LLM adjudication for low-confidence predictions.
─────────────────────────────────────────────────────────────────────────────
OUTPUT FILES:
  {OUT_DIR}/graph_data.{"pt" if HAS_PYG else "npz"}
  {OUT_DIR}/graph_meta.json
  {FIG_DIR}/*.png  (10 publication-quality figures)
─────────────────────────────────────────────────────────────────────────────
[done] All outputs ready for Notebook 7 (Temporal HetGNN + Incremental Learning).
""")
