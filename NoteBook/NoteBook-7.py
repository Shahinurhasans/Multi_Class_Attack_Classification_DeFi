# %% [markdown]
# # Notebook 7 — Temporal HetGNN + Incremental Learning + LLM Governance  (Q1-Grade)
#
# **Architecture**: 2-layer HeteroConv(GATv2Conv) over 5 Ethereum relation types.
# temporal_decay_weight is part of 16D edge_attr fed into GATv2 attention.
#
# **Inputs**:
# | File | Description |
# |---|---|
# | `graph_data.pt`             | HeteroData: 42892 nodes, 5 rels, 24.25M edges, 16D edge feat |
# | `graph_meta.json`           | Edge feature cols, class weights, feature list |
# | `split_meta.json`           | Train/test masks, class weights |
# | `behavioral_scores.parquet` | 8 heuristic scores (LLM prompt context only) |
#
# **Outputs**:
# | File | Description |
# |---|---|
# | `model_best.pth`             | Best model checkpoint (Phase 1 static) |
# | `model_incremental.pth`      | Post-EWC incremental model (Phase 2) |
# | `test_predictions.parquet`   | Per-node: true label, pred, confidence, embedding, llm_flag |
# | `training_history.json`      | Loss/F1 curves for both phases |
# | `model_config.json`          | Hyperparameters, architecture config |
#
# **Memory note**: Full-graph GATv2 on 14.79M edges needs ~15 GB for x_i/x_j intermediates
# (14.79M × 256 × 4 bytes each). NeighborLoader with [10,5] sampling keeps each batch
# to ~20K edges, reducing peak attention memory to ~40 MB.

# %%
import gc, json, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                              confusion_matrix, roc_auc_score)

# ── Install pyg-lib (or torch-sparse) BEFORE importing torch_geometric ────────
# HeteroData NeighborLoader needs pyg-lib/torch-sparse for C++ neighbor sampling.
# PyG reads WITH_PYG_LIB at import time, so this MUST run before the pyg import.
import subprocess as _sp, sys as _sys
_tv  = torch.__version__.split("+")[0]           # e.g. "2.4.0"
_cv  = ("cu" + torch.version.cuda.replace(".", "")[:3]
        if (torch.cuda.is_available() and torch.version.cuda) else "cpu")
_url = f"https://data.pyg.org/whl/torch-{_tv}+{_cv}.html"
print(f"[deps] Installing pyg-lib  torch={_tv}  cuda={_cv} …")
_r = _sp.run([_sys.executable, "-m", "pip", "install", "pyg-lib",
              "-f", _url, "-q"], capture_output=True, text=True)
if _r.returncode != 0:
    print(f"[deps] pyg-lib not available for this torch/cuda — trying torch-sparse …")
    _r2 = _sp.run([_sys.executable, "-m", "pip", "install", "torch-sparse",
                   "-f", _url, "-q"], capture_output=True, text=True)
    if _r2.returncode == 0:
        print("[deps] torch-sparse installed OK")
    else:
        print(f"[deps] WARN: both pyg-lib and torch-sparse failed.\n"
              f"       torch-sparse stderr: {_r2.stderr[-400:]}")
else:
    print("[deps] pyg-lib installed OK")
del _r, _url, _tv, _cv

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, GATv2Conv
    from torch_geometric.loader import NeighborLoader
    import torch_geometric
    print(f"[deps] torch_geometric {torch_geometric.__version__}  "
          f"WITH_PYG_LIB={torch_geometric.typing.WITH_PYG_LIB}  "
          f"WITH_TORCH_SPARSE={torch_geometric.typing.WITH_TORCH_SPARSE}")
    HAS_PYG = True
except ImportError:
    raise RuntimeError("PyTorch Geometric not found — run NB6 first (it installs PyG)")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

FEAT_DIR = Path("/kaggle/working/features")
OUT_DIR  = Path("/kaggle/working/features")
FIG_DIR  = Path("/kaggle/working/figures_nb7")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

CLASS_NAMES = [
    "benign", "phishing", "rug_pull", "ponzi",
    "flash_loan_attack", "malicious_mev", "exploit_contract", "honeypot",
]
N_CLASSES = len(CLASS_NAMES)
CLASS_PALETTE = {
    "benign": "#4CAF50", "phishing": "#F44336", "rug_pull": "#FF9800",
    "ponzi": "#9C27B0", "flash_loan_attack": "#00BCD4",
    "malicious_mev": "#E91E63", "exploit_contract": "#FF5722",
    "honeypot": "#795548",
}
print("=== Notebook 7 — Temporal HetGNN + Incremental Learning + LLM Governance ===")


# %% [markdown]
# ## 1.  Load graph & metadata

# %%
print("\n[1] Loading graph_data.pt …")
data = torch.load(FEAT_DIR / "graph_data.pt", weights_only=False)
print(f"    Node types : {data.node_types}")
print(f"    Edge types : {len(data.edge_types)}")
print(f"    Nodes      : {data['addr'].x.shape}")
print(f"    y dtype    : {data['addr'].y.dtype}")
for et in data.edge_types:
    print(f"    {str(et):45s}: {data[et].edge_index.shape[1]:,} edges  "
          f"{data[et].edge_attr.shape[1]}D attr")

with open(FEAT_DIR / "graph_meta.json")  as fh: graph_meta  = json.load(fh)
with open(FEAT_DIR / "split_meta.json") as fh: split_meta  = json.load(fh)

N_NODES     = data["addr"].x.shape[0]
N_FEAT      = data["addr"].x.shape[1]
N_EDGE_FEAT = data[data.edge_types[0]].edge_attr.shape[1]
RELATIONS   = data.edge_types                         # list of (src, rel, dst) tuples

train_mask = data["addr"].train_mask.numpy()          # bool (N,)
test_mask  = data["addr"].test_mask.numpy()           # bool (N,)
y_all      = data["addr"].y.numpy()                   # int64 (N,)

# Class weights tensor (for loss)
cw_dict = split_meta["class_weights"]
class_weight_vec = torch.tensor(
    [cw_dict[cn] for cn in CLASS_NAMES], dtype=torch.float32, device=DEVICE
)
print(f"\n    N_NODES={N_NODES}  N_FEAT={N_FEAT}  N_EDGE_FEAT={N_EDGE_FEAT}")
print(f"    train={train_mask.sum():,}  test={test_mask.sum():,}")

# Load behavioral scores for LLM governance
beh_df = pd.read_parquet(FEAT_DIR / "behavioral_scores.parquet")
beh_df = beh_df.set_index("address") if "address" in beh_df.columns else beh_df
# Load node addresses for lookup
proc_df = pd.read_parquet(FEAT_DIR / "processed_dataset.parquet",
                          columns=["address", "label", "label_name", "split"])
proc_df = proc_df.reset_index(drop=True)
print(f"    behavioral_scores: {beh_df.shape}  (for LLM prompt context)")

gc.collect()


# %% [markdown]
# ## 2.  Temporal incremental-split of train set

# %%
print("\n[2] Building temporal incremental splits …")

# Sort train nodes by first_ts to define historical vs new-batch phases
# first_ts was dropped from processed_dataset.parquet in NB5 (final 204-col file
# has no room for it). Read from extracted_dataset.parquet (351-col NB4 output).
first_ts = pd.read_parquet(FEAT_DIR / "extracted_dataset.parquet",
                            columns=["first_ts"]).values.flatten()

train_idx = np.where(train_mask)[0]
train_ts  = first_ts[train_idx]
train_order = train_idx[np.argsort(train_ts)]    # sorted oldest → newest

n_hist  = int(len(train_order) * 0.80)           # Phase 1: oldest 80%
hist_idx = train_order[:n_hist]
new_idx  = train_order[n_hist:]

historical_mask = torch.zeros(N_NODES, dtype=torch.bool)
historical_mask[hist_idx] = True
new_batch_mask  = torch.zeros(N_NODES, dtype=torch.bool)
new_batch_mask[new_idx] = True

# Inner validation split (temporal): within historical, newest 20% is val
n_inner_train = int(n_hist * 0.80)
inner_train_idx = hist_idx[:n_inner_train]
inner_val_idx   = hist_idx[n_inner_train:]

inner_train_mask = torch.zeros(N_NODES, dtype=torch.bool)
inner_train_mask[inner_train_idx] = True
inner_val_mask   = torch.zeros(N_NODES, dtype=torch.bool)
inner_val_mask[inner_val_idx] = True

print(f"    historical  : {n_hist:,} nodes (Phase 1 training)")
print(f"    new_batch   : {len(new_idx):,} nodes (Phase 2 incremental)")
print(f"    inner_train : {n_inner_train:,}  |  inner_val: {len(inner_val_idx):,}")

# Class distribution in each split
for name, mask_np in [("historical", hist_idx), ("new_batch", new_idx)]:
    labels = y_all[mask_np]
    dist   = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(N_CLASSES) if (labels == i).sum() > 0}
    print(f"    {name} classes: { {k: v for k, v in sorted(dist.items(), key=lambda x:-x[1])[:4]} } …")

gc.collect()


# %% [markdown]
# ## 3.  Model definition

# %%
class TemporalHetGNN(nn.Module):
    """
    2-layer Heterogeneous Graph Attention Network (GATv2) over 5 Ethereum
    relation types. Edge features (16D, including temporal_decay_weight at idx 2
    and same_block_flag at idx 14) are incorporated into attention computation
    via the GATv2Conv edge_dim parameter — no separate temporal gating needed.
    """
    def __init__(self, in_ch, hidden_ch, out_ch, edge_dim,
                 n_heads, dropout, relations):
        super().__init__()
        assert hidden_ch % n_heads == 0, "hidden_ch must be divisible by n_heads"
        head_dim = hidden_ch // n_heads

        self.conv1 = HeteroConv({
            rel: GATv2Conv(in_ch, head_dim, heads=n_heads,
                          edge_dim=edge_dim, dropout=dropout,
                          concat=True, add_self_loops=False)
            for rel in relations
        }, aggr="sum")

        self.conv2 = HeteroConv({
            rel: GATv2Conv(hidden_ch, head_dim, heads=n_heads,
                          edge_dim=edge_dim, dropout=dropout,
                          concat=True, add_self_loops=False)
            for rel in relations
        }, aggr="sum")

        self.bn1      = nn.BatchNorm1d(hidden_ch)
        self.bn2      = nn.BatchNorm1d(hidden_ch)
        self.drop     = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_ch, out_ch)
        self.hidden_ch  = hidden_ch

    def _edge_dicts(self, batch):
        ei, ea = {}, {}
        for rel in RELATIONS:
            try:
                es = batch[rel]
                if hasattr(es, "edge_index") and es.edge_index.shape[1] > 0:
                    ei[rel] = es.edge_index
                    ea[rel] = es.edge_attr.float()
            except (KeyError, AttributeError):
                pass
        return ei, ea

    def encode(self, x_dict, ei_dict, ea_dict):
        h = self.conv1(x_dict, ei_dict, edge_attr_dict=ea_dict)
        h = {"addr": self.drop(F.elu(self.bn1(h["addr"])))}
        h = self.conv2(h, ei_dict, edge_attr_dict=ea_dict)
        h = {"addr": self.drop(F.elu(self.bn2(h["addr"])))}
        return h["addr"]                              # (N_batch, hidden_ch)

    def forward(self, x_dict, ei_dict, ea_dict):
        emb = self.encode(x_dict, ei_dict, ea_dict)
        return self.classifier(emb)                   # (N_batch, N_CLASSES)


# %% [markdown]
# ## 4.  Focal Loss + EWC

# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # class weight vector (N_CLASSES,)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class EWC:
    """Elastic Weight Consolidation — prevents catastrophic forgetting in Phase 2."""
    def __init__(self, model, data, phase1_mask, criterion, device,
                 n_ewc_batches=10):
        self.star_params = {n: p.detach().clone()
                            for n, p in model.named_parameters() if p.requires_grad}
        self.fisher      = {n: torch.zeros_like(p)
                            for n, p in model.named_parameters() if p.requires_grad}
        self._compute_fisher(model, data, phase1_mask, criterion, device, n_ewc_batches)

    def _compute_fisher(self, model, data, mask, criterion, device, n_batches):
        loader = _make_loader(data, mask, batch_size=256, n_neighbors=[5, 3], shuffle=True)
        model.train()
        collected = 0
        for batch in loader:
            if collected >= n_batches:
                break
            batch = batch.to(device)
            n_seed  = batch["addr"].batch_size
            ei, ea  = model._edge_dicts(batch)
            x_dict  = {"addr": batch["addr"].x.float()}
            out     = model(x_dict, ei, ea)[:n_seed]
            true    = batch["addr"].y[:n_seed]
            loss    = criterion(out, true)
            model.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.detach().pow(2) / n_batches
            collected += 1
        model.zero_grad()
        print(f"    EWC Fisher computed over {collected} batches")

    def penalty(self, model):
        return sum(
            (self.fisher[n] * (p - self.star_params[n]).pow(2)).sum()
            for n, p in model.named_parameters()
            if p.requires_grad and n in self.fisher
        )


# %% [markdown]
# ## 5.  Loader factory + train/eval utilities

# %%
def _make_loader(data, mask, batch_size, n_neighbors, shuffle=True):
    return NeighborLoader(
        data,
        num_neighbors={rel: n_neighbors for rel in RELATIONS},
        batch_size=batch_size,
        input_nodes=("addr", mask),
        shuffle=shuffle,
        num_workers=0,
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        batch     = batch.to(device)
        n_seed    = batch["addr"].batch_size
        x_dict    = {"addr": batch["addr"].x.float()}
        ei, ea    = model._edge_dicts(batch)
        out       = model(x_dict, ei, ea)[:n_seed]
        true      = batch["addr"].y[:n_seed]
        loss      = criterion(out, true)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss    += loss.item() * n_seed
        total_correct += (out.argmax(1) == true).sum().item()
        total_n       += n_seed
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


@torch.no_grad()
def eval_mask(model, data, mask_bool, device, batch_size=256):
    """Evaluate on nodes selected by mask_bool. Returns (macro_f1, loss)."""
    model.eval()
    mask_t  = torch.from_numpy(mask_bool) if isinstance(mask_bool, np.ndarray) else mask_bool
    loader  = _make_loader(data, mask_t, batch_size=batch_size,
                           n_neighbors=[20, 10], shuffle=False)
    criterion_eval = FocalLoss(class_weight_vec.to(device))

    all_preds, all_true = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch  = batch.to(device)
        n_seed = batch["addr"].batch_size
        x_dict = {"addr": batch["addr"].x.float()}
        ei, ea = model._edge_dicts(batch)
        out    = model(x_dict, ei, ea)[:n_seed]
        true   = batch["addr"].y[:n_seed]
        loss   = criterion_eval(out, true)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_true.append(true.cpu().numpy())
        total_loss += loss.item() * n_seed
        total_n    += n_seed

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_true)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return macro_f1, total_loss / max(total_n, 1)


@torch.no_grad()
def get_all_predictions(model, data, device, batch_size=256):
    """Returns (logits, embeddings) for ALL N_NODES, in node-index order."""
    model.eval()
    all_logits = torch.zeros(N_NODES, N_CLASSES)
    all_embeds = torch.zeros(N_NODES, model.hidden_ch)
    loader = _make_loader(data, torch.ones(N_NODES, dtype=torch.bool),
                          batch_size=batch_size, n_neighbors=[20, 10], shuffle=False)
    for batch in loader:
        batch   = batch.to(device)
        n_seed  = batch["addr"].batch_size
        n_id    = batch["addr"].n_id[:n_seed]
        x_dict  = {"addr": batch["addr"].x.float()}
        ei, ea  = model._edge_dicts(batch)
        emb     = model.encode(x_dict, ei, ea)[:n_seed]
        logits  = model.classifier(emb)
        all_logits[n_id] = logits.cpu()
        all_embeds[n_id] = emb.cpu()
    return all_logits, all_embeds


# %% [markdown]
# ## 6.  Hyperparameter search (Optuna)

# %%
print("\n[6] Hyperparameter search with Optuna …")

N_TRIALS       = 10
N_EPOCHS_TRIAL = 20
PATIENCE_TRIAL = 5

def objective(trial):
    hp = {
        "lr"          : trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "hidden_ch"   : trial.suggest_categorical("hidden_ch", [128, 256]),
        "n_heads"     : trial.suggest_categorical("n_heads", [2, 4]),
        "dropout"     : trial.suggest_float("dropout", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
    }
    model = TemporalHetGNN(
        in_ch=N_FEAT, hidden_ch=hp["hidden_ch"], out_ch=N_CLASSES,
        edge_dim=N_EDGE_FEAT, n_heads=hp["n_heads"],
        dropout=hp["dropout"], relations=RELATIONS,
    ).to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                             weight_decay=hp["weight_decay"])
    crit = FocalLoss(class_weight_vec)
    train_loader = _make_loader(data, inner_train_mask, batch_size=512,
                                n_neighbors=[10, 5], shuffle=True)

    best_val_f1, pat = 0.0, 0
    for epoch in range(N_EPOCHS_TRIAL):
        train_epoch(model, train_loader, opt, crit, DEVICE)
        val_f1, _ = eval_mask(model, data, inner_val_mask.numpy(), DEVICE)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE_TRIAL:
                break
    del model; gc.collect(); torch.cuda.empty_cache()
    return best_val_f1

pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
study  = optuna.create_study(direction="maximize", pruner=pruner,
                              sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

BEST_HP = study.best_params
print(f"\n    Best trial  : val macro-F1 = {study.best_value:.4f}")
print(f"    Best params : {BEST_HP}")


# %% [markdown]
# ## 7.  Phase 1 — Static training on historical nodes

# %%
print("\n[7] Phase 1: static training on historical nodes …")

N_EPOCHS_P1 = 100
PATIENCE_P1 = 12

model = TemporalHetGNN(
    in_ch=N_FEAT, hidden_ch=BEST_HP["hidden_ch"], out_ch=N_CLASSES,
    edge_dim=N_EDGE_FEAT, n_heads=BEST_HP["n_heads"],
    dropout=BEST_HP["dropout"], relations=RELATIONS,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),
                              lr=BEST_HP["lr"],
                              weight_decay=BEST_HP["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_P1)
criterion = FocalLoss(class_weight_vec)

train_loader_p1 = _make_loader(data, historical_mask, batch_size=512,
                                n_neighbors=[10, 5], shuffle=True)

hist_p1 = {"train_loss": [], "train_acc": [], "val_f1": [], "val_loss": []}
best_val_f1_p1, best_state, pat = 0.0, None, 0
t0 = time.time()

for epoch in range(1, N_EPOCHS_P1 + 1):
    tr_loss, tr_acc = train_epoch(model, train_loader_p1, optimizer, criterion, DEVICE)
    val_f1, val_loss = eval_mask(model, data, inner_val_mask.numpy(), DEVICE)
    scheduler.step()

    hist_p1["train_loss"].append(tr_loss)
    hist_p1["train_acc"].append(tr_acc)
    hist_p1["val_f1"].append(val_f1)
    hist_p1["val_loss"].append(val_loss)

    if val_f1 > best_val_f1_p1:
        best_val_f1_p1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pat = 0
    else:
        pat += 1

    if epoch % 10 == 0 or epoch <= 5:
        elapsed = time.time() - t0
        print(f"    ep{epoch:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"val_f1={val_f1:.4f}  pat={pat}  t={elapsed:.0f}s")

    if pat >= PATIENCE_P1:
        print(f"    Early stop at epoch {epoch}")
        break

# Restore best checkpoint
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
torch.save(best_state, OUT_DIR / "model_best.pth")
print(f"\n    Phase 1 done. Best val macro-F1 = {best_val_f1_p1:.4f}")
print(f"    model_best.pth saved")
gc.collect(); torch.cuda.empty_cache()


# %% [markdown]
# ## 8.  Phase 2 — Incremental learning with EWC

# %%
print("\n[8] Phase 2: incremental learning (EWC) …")

N_EPOCHS_P2  = 40
PATIENCE_P2  = 8
EWC_LAMBDA   = 400.0

# Compute Fisher information on historical data before fine-tuning
ewc = EWC(model, data, historical_mask, criterion, DEVICE, n_ewc_batches=15)

train_loader_p2 = _make_loader(data, new_batch_mask, batch_size=256,
                                n_neighbors=[10, 5], shuffle=True)
optimizer_p2 = torch.optim.Adam(model.parameters(),
                                  lr=BEST_HP["lr"] * 0.3,
                                  weight_decay=BEST_HP["weight_decay"])

hist_p2 = {"train_loss": [], "val_f1": []}
best_val_f1_p2, best_state_p2, pat2 = 0.0, None, 0

for epoch in range(1, N_EPOCHS_P2 + 1):
    # Phase 2 training with EWC penalty
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in train_loader_p2:
        batch  = batch.to(DEVICE)
        n_seed = batch["addr"].batch_size
        x_dict = {"addr": batch["addr"].x.float()}
        ei, ea = model._edge_dicts(batch)
        out    = model(x_dict, ei, ea)[:n_seed]
        true   = batch["addr"].y[:n_seed]
        task_loss = criterion(out, true)
        ewc_loss  = EWC_LAMBDA * ewc.penalty(model)
        loss      = task_loss + ewc_loss
        optimizer_p2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer_p2.step()
        total_loss += task_loss.item() * n_seed
        total_n    += n_seed

    tr_loss = total_loss / max(total_n, 1)
    val_f1, _ = eval_mask(model, data, inner_val_mask.numpy(), DEVICE)
    hist_p2["train_loss"].append(tr_loss)
    hist_p2["val_f1"].append(val_f1)

    if val_f1 > best_val_f1_p2:
        best_val_f1_p2 = val_f1
        best_state_p2  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pat2 = 0
    else:
        pat2 += 1
        if pat2 >= PATIENCE_P2:
            print(f"    Early stop at epoch {epoch}")
            break

    if epoch % 5 == 0 or epoch <= 3:
        print(f"    ep{epoch:3d}  tr_loss={tr_loss:.4f}  val_f1={val_f1:.4f}  pat={pat2}")

# Compare Phase 1 and Phase 2 validation F1
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_p2.items()})
torch.save(best_state_p2, OUT_DIR / "model_incremental.pth")
print(f"\n    Phase 2 done. Best val macro-F1 = {best_val_f1_p2:.4f}  "
      f"(Phase1={best_val_f1_p1:.4f}  Δ={best_val_f1_p2-best_val_f1_p1:+.4f})")
print(f"    model_incremental.pth saved")
gc.collect(); torch.cuda.empty_cache()


# %% [markdown]
# ## 9.  Test evaluation

# %%
print("\n[9] Test evaluation …")

all_logits, all_embeds = get_all_predictions(model, data, DEVICE)
all_probs  = torch.softmax(all_logits, dim=-1).numpy()
all_preds  = all_probs.argmax(axis=1)
max_probs  = all_probs.max(axis=1)

# Test-set metrics
test_idx    = np.where(test_mask)[0]
test_true   = y_all[test_idx]
test_pred   = all_preds[test_idx]
test_probs  = all_probs[test_idx]

macro_f1   = f1_score(test_true, test_pred, average="macro",    zero_division=0)
weighted_f1= f1_score(test_true, test_pred, average="weighted", zero_division=0)
prec, rec, f1_per, _ = precision_recall_fscore_support(
    test_true, test_pred, labels=range(N_CLASSES), zero_division=0)

print(f"\n    Test Macro F1   : {macro_f1:.4f}")
print(f"    Test Weighted F1: {weighted_f1:.4f}")
print(f"\n    Per-class results:")
print(f"    {'Class':22s}  {'N':>5s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
print(f"    {'-'*55}")
for i, cn in enumerate(CLASS_NAMES):
    n = int((test_true == i).sum())
    print(f"    {cn:22s}  {n:5d}  {prec[i]:.4f}  {rec[i]:.4f}  {f1_per[i]:.4f}")

# AUROC (macro OvR, only classes present in test set)
try:
    present_classes = np.unique(test_true)
    auc = roc_auc_score(test_true, test_probs[:, present_classes],
                        multi_class="ovr", average="macro",
                        labels=present_classes)
    print(f"\n    Macro AUROC (OvR, present classes): {auc:.4f}")
except Exception as e:
    auc = 0.0
    print(f"    [warn] AUROC skipped: {e}")

# Also evaluate Phase 1 model for incremental comparison
model_p1 = TemporalHetGNN(
    in_ch=N_FEAT, hidden_ch=BEST_HP["hidden_ch"], out_ch=N_CLASSES,
    edge_dim=N_EDGE_FEAT, n_heads=BEST_HP["n_heads"],
    dropout=BEST_HP["dropout"], relations=RELATIONS,
).to(DEVICE)
model_p1.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

logits_p1, _ = get_all_predictions(model_p1, data, DEVICE)
pred_p1 = torch.softmax(logits_p1, dim=-1).numpy().argmax(axis=1)
f1_p1_per = f1_score(test_true, pred_p1[test_idx], labels=range(N_CLASSES),
                      average=None, zero_division=0)
del model_p1; gc.collect(); torch.cuda.empty_cache()

print(f"\n    Incremental gain per class (Phase2 - Phase1 F1):")
for i, cn in enumerate(CLASS_NAMES):
    n = int((test_true == i).sum())
    delta = f1_per[i] - f1_p1_per[i]
    if n > 0:
        print(f"      {cn:22s}: n={n:5d}  Δ={delta:+.4f}  (P1={f1_p1_per[i]:.4f} → P2={f1_per[i]:.4f})")


# %% [markdown]
# ## 10.  LLM Governance (TinyLlama-1.1B, low-confidence adjudication)

# %%
print("\n[10] LLM Governance …")

LLM_CONF_THRESHOLD = 0.65   # govern predictions below this confidence

# Low-confidence test nodes
low_conf_mask = (max_probs < LLM_CONF_THRESHOLD) & test_mask
low_conf_idx  = np.where(low_conf_mask)[0]
print(f"    Low-confidence test nodes (< {LLM_CONF_THRESHOLD:.0%}): {len(low_conf_idx):,}")

# Load TinyLlama (free, 1.1B, fits in GPU alongside GNN weights)
llm_available = False
llm_adjudications = {}     # {node_idx: {"final_class": int, "response": str}}

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("    Loading TinyLlama-1.1B-Chat-v1.0 …")
    _llm_name  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    _llm_tok   = AutoTokenizer.from_pretrained(_llm_name)
    _llm_model = AutoModelForCausalLM.from_pretrained(
        _llm_name, torch_dtype=torch.float16, device_map="auto"
    )
    _llm_model.eval()
    llm_available = True
    print(f"    TinyLlama loaded on {next(_llm_model.parameters()).device}")
except Exception as _llm_err:
    print(f"    [warn] LLM load failed ({_llm_err}) — using rule-based fallback")

def _get_beh_features(address):
    """Return dict of behavioral features for LLM prompt."""
    defaults = {c: 0.0 for c in beh_df.columns if beh_df[c].dtype.kind == "f"}
    try:
        row = beh_df.loc[address]
        return {c: float(row[c]) if c in beh_df.columns else 0.0
                for c in ["velocity_ratio_7d", "gas_price_spike_ratio",
                          "ponzi_funnel_ratio", "days_since_last_activity"]}
    except (KeyError, Exception):
        return defaults

def _llm_adjudicate(address, pred_cls_idx, alt_cls_idx, confidence, beh):
    """Call TinyLlama; returns (final_class_idx, response_text)."""
    pred_cn = CLASS_NAMES[pred_cls_idx]
    alt_cn  = CLASS_NAMES[alt_cls_idx]
    prompt  = (
        f"<|system|>You are a blockchain security expert.</s>"
        f"<|user|>Address analysis:\n"
        f"- Tx velocity ratio (7d vs baseline): {beh.get('velocity_ratio_7d', 0):.2f}x\n"
        f"- Gas price spike ratio: {beh.get('gas_price_spike_ratio', 0):.2f}\n"
        f"- Ponzi funnel ratio: {beh.get('ponzi_funnel_ratio', 0):.2f}\n"
        f"- Days since last activity: {beh.get('days_since_last_activity', 0):.0f}\n"
        f"GNN prediction: {pred_cn} ({confidence:.0%} confidence)\n"
        f"Alternative: {alt_cn}\n"
        f"Reply ONLY with A (keep {pred_cn}) or B (prefer {alt_cn}).</s>"
        f"<|assistant|>"
    )
    inputs = _llm_tok(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out_ids = _llm_model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            pad_token_id=_llm_tok.eos_token_id
        )
    response = _llm_tok.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    final_cls = alt_cls_idx if response.upper().startswith("B") else pred_cls_idx
    return final_cls, response

def _rule_adjudicate(pred_cls_idx, alt_cls_idx, beh):
    """Simple rule-based fallback when LLM is unavailable."""
    # If gas spike > 50 and alt is flash_loan_attack or malicious_mev → prefer alt
    fl_idx  = CLASS_NAMES.index("flash_loan_attack")
    mev_idx = CLASS_NAMES.index("malicious_mev")
    if (beh.get("gas_price_spike_ratio", 0) > 50 and
            alt_cls_idx in (fl_idx, mev_idx)):
        return alt_cls_idx, "rule:high_gas_spike→attack"
    return pred_cls_idx, "rule:kept_primary"

# Govern low-confidence test predictions
MAX_LLM_NODES = 200  # cap to avoid very long runtime
govern_indices = low_conf_idx[:MAX_LLM_NODES]

n_overridden = 0
for node_idx in govern_indices:
    address   = proc_df.iloc[node_idx]["address"]
    probs_row = all_probs[node_idx]
    top2      = np.argsort(probs_row)[::-1][:2]
    pred_ci, alt_ci = int(top2[0]), int(top2[1])
    conf      = float(probs_row[pred_ci])
    beh       = _get_beh_features(address)

    if llm_available:
        final_ci, resp = _llm_adjudicate(address, pred_ci, alt_ci, conf, beh)
    else:
        final_ci, resp = _rule_adjudicate(pred_ci, alt_ci, beh)

    llm_adjudications[int(node_idx)] = {
        "original_pred": pred_ci, "final_pred": final_ci,
        "confidence": conf, "response": resp,
    }
    if final_ci != pred_ci:
        n_overridden += 1

# Apply LLM overrides to predictions
final_preds = all_preds.copy()
for nidx, adj in llm_adjudications.items():
    final_preds[nidx] = adj["final_pred"]

final_macro_f1 = f1_score(test_true, final_preds[test_idx],
                           average="macro", zero_division=0)

print(f"    Governed {len(govern_indices)} low-confidence nodes")
print(f"    LLM/rule overridden: {n_overridden} ({n_overridden/max(len(govern_indices),1):.1%})")
print(f"    Pre-governance macro F1 : {macro_f1:.4f}")
print(f"    Post-governance macro F1: {final_macro_f1:.4f}  "
      f"(Δ={final_macro_f1 - macro_f1:+.4f})")

if llm_available:
    del _llm_model; gc.collect(); torch.cuda.empty_cache()


# %% [markdown]
# ## 11.  Save outputs

# %%
print("\n[11] Saving outputs …")

# test_predictions.parquet
llm_flag = np.zeros(N_NODES, dtype=bool)
for nidx in llm_adjudications:
    llm_flag[nidx] = True

pred_df = pd.DataFrame({
    "address":           proc_df["address"].values,
    "true_label":        y_all,
    "true_label_name":   [CLASS_NAMES[i] for i in y_all],
    "pred_label":        final_preds,
    "pred_label_name":   [CLASS_NAMES[i] for i in final_preds],
    "confidence":        max_probs,
    "llm_adjudicated":   llm_flag,
    "split":             proc_df["split"].values,
})
# Append softmax probabilities
for i, cn in enumerate(CLASS_NAMES):
    pred_df[f"prob_{cn}"] = all_probs[:, i]
# Append embeddings (float16 to save space)
emb_np = all_embeds.numpy().astype(np.float16)
for dim in range(emb_np.shape[1]):
    pred_df[f"emb_{dim}"] = emb_np[:, dim]

pred_df.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
print(f"    test_predictions.parquet: {pred_df.shape}")

# training_history.json
training_hist = {
    "phase1": hist_p1,
    "phase2": hist_p2,
    "best_hp": BEST_HP,
    "test_metrics": {
        "macro_f1_before_governance":   float(macro_f1),
        "macro_f1_after_governance":    float(final_macro_f1),
        "weighted_f1":                  float(weighted_f1),
        "auroc_macro":                  float(auc),
        "per_class_f1":                 {CLASS_NAMES[i]: float(f1_per[i])
                                          for i in range(N_CLASSES)},
        "per_class_f1_phase1":          {CLASS_NAMES[i]: float(f1_p1_per[i])
                                          for i in range(N_CLASSES)},
        "n_llm_governed":               len(govern_indices),
        "n_llm_overridden":             n_overridden,
    },
}
with open(OUT_DIR / "training_history.json", "w") as fh:
    json.dump(training_hist, fh, indent=2)
print("    training_history.json saved")

# model_config.json
model_cfg = {
    "architecture":    "TemporalHetGNN",
    "n_layers":        2,
    "in_channels":     N_FEAT,
    "hidden_channels": BEST_HP["hidden_ch"],
    "out_channels":    N_CLASSES,
    "edge_dim":        N_EDGE_FEAT,
    "n_heads":         BEST_HP["n_heads"],
    "dropout":         BEST_HP["dropout"],
    "lr":              BEST_HP["lr"],
    "weight_decay":    BEST_HP["weight_decay"],
    "ewc_lambda":      EWC_LAMBDA,
    "focal_gamma":     2.0,
    "n_epochs_phase1": N_EPOCHS_P1,
    "n_epochs_phase2": N_EPOCHS_P2,
    "neighbor_sample_train":  [10, 5],
    "neighbor_sample_eval":   [20, 10],
    "phase1_val_macro_f1":    float(best_val_f1_p1),
    "phase2_val_macro_f1":    float(best_val_f1_p2),
    "test_macro_f1":          float(macro_f1),
    "llm_governance_model":   "TinyLlama-1.1B-Chat" if llm_available else "rule-based",
    "llm_threshold":          LLM_CONF_THRESHOLD,
    "edge_feat_cols":         graph_meta["edge_feat_cols"],
    "selected_node_features": graph_meta["selected_features"],
}
with open(OUT_DIR / "model_config.json", "w") as fh:
    json.dump(model_cfg, fh, indent=2)
print("    model_config.json saved")
gc.collect()


# %% [markdown]
# ## 12.  Visualisations (10 Q1-publication figures)

# %%
print("\n[12] Generating visualisations …")
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── VIS 1: Training curves (Phase 1 + Phase 2) ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
ep1 = range(1, len(hist_p1["train_loss"]) + 1)
ep2 = range(1, len(hist_p2["train_loss"]) + 1)

axes[0].plot(ep1, hist_p1["train_loss"], label="P1 train", color="#1976D2")
axes[0].plot(ep2, hist_p2["train_loss"], label="P2 train", color="#E53935", linestyle="--")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal Loss")
axes[0].set_title("Training Loss"); axes[0].legend(fontsize=8)

axes[1].plot(ep1, hist_p1["val_f1"], label="P1 val F1", color="#1976D2")
axes[1].plot(ep2, hist_p2["val_f1"],  label="P2 val F1", color="#E53935", linestyle="--")
axes[1].axhline(best_val_f1_p1, color="#1976D2", linestyle=":", alpha=0.5)
axes[1].axhline(best_val_f1_p2, color="#E53935", linestyle=":", alpha=0.5)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1 (val)")
axes[1].set_title("Validation Macro F1"); axes[1].legend(fontsize=8)

axes[2].plot(ep1, hist_p1["train_acc"], label="P1 train acc", color="#1976D2")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Train Accuracy")
axes[2].set_title("Train Accuracy (Phase 1)"); axes[2].legend(fontsize=8)

plt.suptitle("Training History — Phase 1 (Static) + Phase 2 (Incremental/EWC)", fontsize=11)
plt.tight_layout(); plt.savefig(FIG_DIR / "vis1_training_curves.png", bbox_inches="tight")
plt.close(); print("  vis1 saved")


# ── VIS 2: Confusion matrix (test, post-governance) ──────────────────────────
cm = confusion_matrix(test_true, final_preds[test_idx], labels=range(N_CLASSES))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar_kws={"label": "Row-normalised fraction"},
            annot_kws={"size": 8})
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title(f"Confusion Matrix (test, post-governance) — Macro F1={final_macro_f1:.4f}")
plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig(FIG_DIR / "vis2_confusion_matrix.png", bbox_inches="tight")
plt.close(); print("  vis2 saved")


# ── VIS 3: Per-class Precision / Recall / F1 ─────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x  = np.arange(N_CLASSES); w = 0.25
ax.bar(x - w, prec,   width=w, label="Precision", color="#1976D2")
ax.bar(x,     rec,    width=w, label="Recall",    color="#E53935")
ax.bar(x + w, f1_per, width=w, label="F1",        color="#388E3C")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax.set_ylabel("Score"); ax.set_title("Per-Class Precision / Recall / F1 (test set, Phase 2)")
ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig(FIG_DIR / "vis3_per_class_metrics.png", bbox_inches="tight")
plt.close(); print("  vis3 saved")


# ── VIS 4: Phase 1 vs Phase 2 per-class F1 comparison ────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(N_CLASSES); w = 0.35
ax.bar(x - w/2, f1_p1_per, width=w, label="Phase 1 (static)", color="#90CAF9")
ax.bar(x + w/2, f1_per,    width=w, label="Phase 2 (EWC)",    color="#1565C0")
for i, (p1, p2) in enumerate(zip(f1_p1_per, f1_per)):
    delta = p2 - p1
    ax.annotate(f"{delta:+.2f}", xy=(i + w/2, p2 + 0.01),
                ha="center", fontsize=7, color="green" if delta > 0 else "red")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax.set_ylabel("F1 Score"); ax.set_title("Phase 1 vs Phase 2 F1 per Class (Incremental Learning Gain)")
ax.legend(fontsize=9); ax.set_ylim(0, 1.1)
plt.tight_layout(); plt.savefig(FIG_DIR / "vis4_incremental_gain.png", bbox_inches="tight")
plt.close(); print("  vis4 saved")


# ── VIS 5: ROC curves per class (OvR, test set) ──────────────────────────────
from sklearn.metrics import roc_curve, auc as sk_auc
fig, ax = plt.subplots(figsize=(9, 7))
for i, cn in enumerate(CLASS_NAMES):
    n_pos = int((test_true == i).sum())
    if n_pos == 0: continue
    fpr, tpr, _ = roc_curve((test_true == i).astype(int), test_probs[:, i])
    roc_auc_i   = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{cn} (AUC={roc_auc_i:.3f})",
            color=CLASS_PALETTE.get(cn, "grey"), linewidth=1.5)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves per Class — One-vs-Rest (test set)")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(FIG_DIR / "vis5_roc_curves.png", bbox_inches="tight")
plt.close(); print("  vis5 saved")


# ── VIS 6: Confidence distribution (correct vs incorrect) ─────────────────────
test_conf   = max_probs[test_idx]
test_correct = (final_preds[test_idx] == test_true)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(test_conf[test_correct],  bins=30, alpha=0.7, label="Correct",
        color="#388E3C", density=True)
ax.hist(test_conf[~test_correct], bins=30, alpha=0.7, label="Incorrect",
        color="#E53935", density=True)
ax.axvline(LLM_CONF_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
           label=f"LLM threshold ({LLM_CONF_THRESHOLD:.0%})")
ax.set_xlabel("Max softmax confidence"); ax.set_ylabel("Density")
ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions (test)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(FIG_DIR / "vis6_confidence_dist.png", bbox_inches="tight")
plt.close(); print("  vis6 saved")


# ── VIS 7: Hyperparameter importance (Optuna) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
try:
    importance = optuna.importance.get_param_importances(study)
    names = list(importance.keys())
    vals  = list(importance.values())
    ax.barh(names, vals, color="#5C6BC0")
    ax.set_xlabel("Importance (Optuna FAnova)")
    ax.set_title("Hyperparameter Importance")
except Exception:
    ax.text(0.5, 0.5, "Optuna importance not available\n(< 3 completed trials)",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Hyperparameter Importance")
plt.tight_layout(); plt.savefig(FIG_DIR / "vis7_hp_importance.png", bbox_inches="tight")
plt.close(); print("  vis7 saved")


# ── VIS 8: t-SNE of test node embeddings coloured by true class ───────────────
print("  Computing t-SNE (test embeddings) …")
test_emb = all_embeds[test_idx].numpy().astype(np.float32)
# Use at most 2000 test nodes for speed
n_tsne = min(2000, len(test_idx))
rng = np.random.default_rng(SEED)
tsne_sel = rng.choice(len(test_idx), n_tsne, replace=False)
tsne_emb  = test_emb[tsne_sel]
tsne_true = test_true[tsne_sel]

tsne = TSNE(n_components=2, perplexity=30, n_iter=500,
            random_state=SEED, n_jobs=1)
z = tsne.fit_transform(tsne_emb)

fig, ax = plt.subplots(figsize=(10, 8))
for i, cn in enumerate(CLASS_NAMES):
    mask_i = tsne_true == i
    if mask_i.sum() == 0: continue
    ax.scatter(z[mask_i, 0], z[mask_i, 1], s=8, alpha=0.6,
               label=f"{cn} ({mask_i.sum()})", color=CLASS_PALETTE.get(cn, "grey"))
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title(f"t-SNE of GNN Embeddings — Test Nodes (n={n_tsne})")
ax.legend(fontsize=7, markerscale=2, loc="upper right")
plt.tight_layout(); plt.savefig(FIG_DIR / "vis8_tsne_embeddings.png", bbox_inches="tight")
plt.close(); print("  vis8 saved")


# ── VIS 9: LLM governance summary ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel 1: pre/post governance F1 per class
f1_post_gov = f1_score(test_true, final_preds[test_idx],
                        labels=range(N_CLASSES), average=None, zero_division=0)
x = np.arange(N_CLASSES); w = 0.35
axes[0].bar(x - w/2, f1_per,      width=w, label="Pre-governance",  color="#90CAF9")
axes[0].bar(x + w/2, f1_post_gov, width=w, label="Post-governance", color="#1565C0")
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=7)
axes[0].set_ylabel("F1 Score"); axes[0].set_title("Pre vs Post LLM Governance F1")
axes[0].legend(fontsize=8)

# Panel 2: confidence histogram for governed nodes
if llm_adjudications:
    gov_confs     = [adj["confidence"] for adj in llm_adjudications.values()]
    gov_overridden= [adj["original_pred"] != adj["final_pred"]
                     for adj in llm_adjudications.values()]
    axes[1].hist([c for c, o in zip(gov_confs, gov_overridden) if o],
                  bins=20, color="#E53935", alpha=0.7, label="Overridden", density=True)
    axes[1].hist([c for c, o in zip(gov_confs, gov_overridden) if not o],
                  bins=20, color="#388E3C", alpha=0.7, label="Kept", density=True)
    axes[1].set_xlabel("Model confidence"); axes[1].set_ylabel("Density")
    axes[1].set_title(f"Governed Nodes: Overridden ({n_overridden}) vs Kept")
    axes[1].legend(fontsize=8)
else:
    axes[1].text(0.5, 0.5, "No governed nodes", ha="center", va="center",
                 transform=axes[1].transAxes)

plt.suptitle("LLM Governance Layer — Impact on Test Predictions", fontsize=11)
plt.tight_layout(); plt.savefig(FIG_DIR / "vis9_llm_governance.png", bbox_inches="tight")
plt.close(); print("  vis9 saved")


# ── VIS 10: Optuna trial history ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

trial_vals = [t.value for t in study.trials if t.value is not None]
axes[0].plot(range(1, len(trial_vals) + 1), trial_vals, "o-", color="#1976D2")
axes[0].axhline(study.best_value, color="red", linestyle="--",
                linewidth=1, label=f"best={study.best_value:.4f}")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Val Macro F1")
axes[0].set_title("Optuna Trial History"); axes[0].legend(fontsize=8)

# Best trial HP values
hp_names = list(BEST_HP.keys())
hp_vals  = [BEST_HP[k] for k in hp_names]
axes[1].barh(hp_names, [float(v) if isinstance(v, (int, float)) else 0
                         for v in hp_vals], color="#5C6BC0")
axes[1].set_title("Best Hyperparameters")
for i, (n, v) in enumerate(zip(hp_names, hp_vals)):
    axes[1].text(0.02, i, f"{v:.2e}" if isinstance(v, float) else str(v),
                 va="center", fontsize=8, color="white", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG_DIR / "vis10_optuna_trials.png", bbox_inches="tight")
plt.close(); print("  vis10 saved")

print(f"\n  Figures saved: {len(list(FIG_DIR.glob('*.png')))} PNGs → {FIG_DIR}")


# %% [markdown]
# ## 13.  Final summary

# %%
print("\n" + "=" * 80)
print("NOTEBOOK 7 — TEMPORAL HETGNN + INCREMENTAL LEARNING COMPLETE")
print("=" * 80)

print(f"""
  model_best.pth          : Phase 1 static checkpoint
  model_incremental.pth   : Phase 2 EWC checkpoint
  test_predictions.parquet: {pred_df.shape}
  training_history.json   : Phase 1 + Phase 2 curves
  model_config.json       : Hyperparameters + arch config
  Figures                 : {len(list(FIG_DIR.glob('*.png')))} PNGs in {FIG_DIR}

ARCHITECTURE SUMMARY
─────────────────────────────────────────────────────────────────────────────
  TemporalHetGNN:
    Input dim       : {N_FEAT} (RobustScaled node features from NB5)
    Hidden dim      : {BEST_HP["hidden_ch"]} (best Optuna)
    Heads           : {BEST_HP["n_heads"]}  head_dim={BEST_HP["hidden_ch"]//BEST_HP["n_heads"]}
    Edge feature dim: {N_EDGE_FEAT} (temporal_decay_weight@idx2, same_block_flag@idx14)
    Relation types  : {len(RELATIONS)}  {[rel[1] for rel in RELATIONS]}
    Parameters      : ~{sum(p.numel() for p in model.parameters()):,}
    Training loss   : FocalLoss(gamma=2.0) + class weights

RESULTS SUMMARY
─────────────────────────────────────────────────────────────────────────────
  Phase 1 best val macro-F1 : {best_val_f1_p1:.4f}
  Phase 2 best val macro-F1 : {best_val_f1_p2:.4f}  (Δ={best_val_f1_p2-best_val_f1_p1:+.4f})
  Test macro-F1 (pre-gov)   : {macro_f1:.4f}
  Test macro-F1 (post-gov)  : {final_macro_f1:.4f}
  Test weighted F1          : {weighted_f1:.4f}
  Test AUROC (macro OvR)    : {auc:.4f}
  LLM governed / overridden : {len(govern_indices)} / {n_overridden}

Per-class test F1:""")
for i, cn in enumerate(CLASS_NAMES):
    n = int((test_true == i).sum())
    delta = f1_per[i] - f1_p1_per[i]
    print(f"  {cn:22s}: n={n:5d}  F1={f1_per[i]:.4f}  (Δ_incr={delta:+.4f})")

print(f"""
PRECONDITIONS FOR NB8 (ADVERSARIAL DATASET)
─────────────────────────────────────────────────────────────────────────────
 1. test_predictions.parquet has: address, true_label, pred_label, confidence,
    llm_adjudicated, emb_0 … emb_{BEST_HP["hidden_ch"]-1} (GNN embeddings for NB8 attack).

 2. model_best.pth + model_incremental.pth: load with TemporalHetGNN(config).
    Config is in model_config.json.

 3. Adversarial perturbation targets: nodes where confidence > 0.90 and
    true_label in {{flash_loan_attack, malicious_mev, rug_pull}} — high-confidence
    attack predictions are the most useful targets for adversarial robustness testing.

 4. graph_data.pt is used directly in NB8 for graph-structure adversarial attacks
    (edge injection / feature perturbation). Load with torch.load(..., weights_only=False).

 5. class_weights in split_meta.json: use for NB10/NB11 evaluation weighting.
─────────────────────────────────────────────────────────────────────────────
OUTPUT FILES:
  {OUT_DIR}/model_best.pth
  {OUT_DIR}/model_incremental.pth
  {OUT_DIR}/test_predictions.parquet
  {OUT_DIR}/training_history.json
  {OUT_DIR}/model_config.json
  {FIG_DIR}/*.png  (10 publication-quality figures)
─────────────────────────────────────────────────────────────────────────────
[done] All outputs ready for Notebook 8 (Adversarial Dataset Creation).
""")
