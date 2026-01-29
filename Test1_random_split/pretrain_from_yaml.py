import argparse
import os
import sys

import torch
from sklearn.model_selection import KFold


# path setup (your repo layout assumption: this script is runnable from repo)
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGT.load_data import load_data
from drGT.myutils import get_all_edges_and_labels
from drGT.sampler import BalancedSampler
from drGT import drGT  # drGT.train is used

# yaml loader
try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


def load_yaml_params(config_path: str, data: str, method: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if data not in cfg:
        raise KeyError(f"Dataset '{data}' not found in {config_path}. Keys: {list(cfg.keys())}")

    if method not in cfg[data]:
        raise KeyError(
            f"Method '{method}' not found under '{data}' in {config_path}. "
            f"Available methods: {list(cfg[data].keys())}"
        )

    params = cfg[data][method]
    if not isinstance(params, dict):
        raise ValueError(f"Config entry {data}->{method} must be a dict, got {type(params)}")
    return params


def apply_defaults(params: dict) -> dict:
    """Fill only missing keys with sane defaults."""
    defaults = {
        "optimizer": "Adam",
        "scheduler": None,
        "norm_type": "GraphNorm",
        "n_layers": 2,
        "final_mlp_layers": 2,
        "attention_dropout": 0.0,
        "dropout1": 0.1,
        "dropout2": 0.2,
        "dropout3": 0.5,
        "activation": "relu",
        "weight_decay": 0.0,
        "lr": 1e-4,
        "epochs": 500,
        "heads": 4,
        "hidden1": 256,
        "hidden2": 64,
        "hidden3": 128,
    }
    out = dict(defaults)
    out.update(params or {})
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/test1_params.yaml",
                        help="YAML config path (default: configs/test1_params.yaml)")
    parser.add_argument("--data", type=str, choices=["gdsc1", "gdsc2", "ctrp", "nci"], default="ctrp")
    parser.add_argument("--method", type=str, choices=["GAT", "GATv2", "Transformer"], default="GAT")
    parser.add_argument("--is_zero_pad", action="store_true", default=True)

    # pretrain split + save
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Output checkpoint path. Default: pretrained_{data}_{method}.pt")

    # optional CLI overrides (if you pass them, they overwrite YAML)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--hidden1", type=int, default=None)
    parser.add_argument("--hidden2", type=int, default=None)
    parser.add_argument("--hidden3", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--final_mlp_layers", type=int, default=None)
    parser.add_argument("--attention_dropout", type=float, default=None)
    parser.add_argument("--dropout1", type=float, default=None)
    parser.add_argument("--dropout2", type=float, default=None)
    parser.add_argument("--dropout3", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--norm_type", type=str, default=None)

    args = parser.parse_args()

    # 1) load YAML params for (data, method)
    yaml_params = load_yaml_params(args.config, args.data, args.method)
    params = apply_defaults(yaml_params)

    # 2) apply CLI overrides if provided
    for k in [
        "epochs", "lr", "weight_decay", "heads", "hidden1", "hidden2", "hidden3",
        "n_layers", "final_mlp_layers", "attention_dropout",
        "dropout1", "dropout2", "dropout3",
        "activation", "optimizer", "scheduler", "norm_type",
    ]:
        v = getattr(args, k)
        if v is not None:
            params[k] = v

    # normalize "None" scheduler string
    if isinstance(params.get("scheduler"), str) and params["scheduler"].lower() == "none":
        params["scheduler"] = None

    # 3) load data
    (
        drugAct,
        null_mask,
        S_d,
        S_c,
        S_g,
        _,
        _,
        _,
        A_cg,
        A_dg,
    ) = load_data(args.data, is_zero_pad=args.is_zero_pad)

    # 4) set dimension-dependent fields (must match dataset)
    params.update(
        {
            "n_drug": S_d.shape[0],
            "n_cell": S_c.shape[0],
            "n_gene": S_g.shape[0],
            "gnn_layer": args.method,
        }
    )

    # 5) build edges/labels + split
    all_edges, all_labels = get_all_edges_and_labels(drugAct, null_mask)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # fold0 only
    train_idx, test_idx = next(kf.split(all_edges))

    sampler = BalancedSampler(
        drugAct,
        all_edges,
        all_labels,
        train_idx,
        test_idx,
        null_mask,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
    )

    # Train the model once (use train/val within sampler to select the best epoch)
    model, train_attn, val_attn, val_true, val_prob, best_metrics = drGT.train(
        sampler,
        params=params,
        device=device,
        verbose=True,
        is_save=False,   # Set to False if not using internal save in drGT.train
    )

    # Recommended: Save state_dict + params together (for reproducibility)
    out_path = args.out or f"pretrained_{args.data}_{args.method}_fold0.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "data": args.data,
            "method": args.method,
            "fold": 0,
            "best_metrics": best_metrics,
        },
        out_path,
    )
    print(f"Saved pretrained checkpoint: {out_path}")

if __name__ == "__main__":
    main()
