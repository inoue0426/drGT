import argparse
import os
import sys
from pathlib import Path

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add path to drGT package (compatible with both Singularity/local environments)
if os.path.exists("/workspace/drGT"):
    sys.path.append("/workspace")
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGT import drGT
from drGT.load_data import load_data
from drGT.metrics import compute_metrics_stats
from drGT.sampler import NewSampler
from drGT.utility import filter_target

# Parse command-line options for method, data, mode, parallelism, and all hyperparameters
parser = argparse.ArgumentParser(description="Run drGT with custom hyperparameters (no tuning)")

parser.add_argument("--method", type=str, choices=["GAT", "GATv2", "Transformer"], default="Transformer")
parser.add_argument("--data", type=str, choices=["gdsc1", "gdsc2", "ctrp", "nci"], default="nci")
parser.add_argument("--target_dim", type=int, choices=[0, 1], default=0)
parser.add_argument("--n_jobs", type=int, default=3, help="Number of parallel jobs for target processing")

# Model hyperparameters (with Transformer defaults as specified)
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--attention_dropout", type=float, default=0.2)
parser.add_argument("--dropout1", type=float, default=0.2)
parser.add_argument("--dropout2", type=float, default=0.2)
parser.add_argument("--dropout3", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=600)
parser.add_argument("--final_mlp_layers", type=int, default=2)
parser.add_argument("--heads", type=int, default=3)
parser.add_argument("--hidden1", type=int, default=421)
parser.add_argument("--hidden2", type=int, default=120)
parser.add_argument("--hidden3", type=int, default=47)
parser.add_argument("--is_zero_pad", type=bool, default=True)
parser.add_argument("--lr", type=float, default=0.00010970329600928436)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--norm_type", type=str, default="GraphNorm", choices=["GraphNorm", "BatchNorm", "LayerNorm"])
parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"])
parser.add_argument("--scheduler", type=str, default="Cosine", choices=["None", "Cosine"])
parser.add_argument("--weight_decay", type=float, default=0.001773507293085185)
parser.add_argument("--T_max", type=int, default=183)

args = parser.parse_args()

method = args.method
data = args.data
target_dim = args.target_dim
n_jobs = args.n_jobs

# Setup the parameters dictionary with command-line args (defaults included)
params = {
    "dropout1": args.dropout1,
    "dropout2": args.dropout2,
    "dropout3": args.dropout3,
    "hidden1": args.hidden1,
    "hidden2": args.hidden2,
    "hidden3": args.hidden3,
    "epochs": args.epochs,
    "heads": args.heads,
    "activation": args.activation,
    "optimizer": args.optimizer,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "scheduler": args.scheduler,
    "norm_type": args.norm_type,
    "n_layers": args.n_layers,
    "gnn_layer": method,
    "final_mlp_layers": args.final_mlp_layers,
    "attention_dropout": args.attention_dropout,
    "T_max": args.T_max,
}

def drGT_new(
    res,
    null_mask,
    target_dim,
    target_index,
    S_d,
    S_c,
    S_g,
    A_cg,
    A_dg,
    params,
    device,
):
    sampler = NewSampler(
        res,
        null_mask,
        target_dim,
        target_index,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
    )

    # Perform training for a single target
    (_, _, _, best_val_labels, best_val_prob, best_metrics, _, _, _) = drGT.train(
        sampler, params=params, device=device, verbose=False
    )
    return best_val_labels, best_val_prob

if __name__ == "__main__":
    # Load data according to provided hyperparameters
    (
        res,
        null_mask,
        S_d,
        S_c,
        S_g,
        drug_feature,
        gene_norm_gene,
        gene_norm_cell,
        A_cg,
        A_dg,
    ) = load_data(data, is_zero_pad=args.is_zero_pad)

    params["n_drug"] = S_d.shape[0]
    params["n_cell"] = S_c.shape[0]
    params["n_gene"] = S_g.shape[0]

    # Identify valid/invalid targets
    samples = res.shape[target_dim]
    passed_targets = []
    skipped_targets = []
    for target_index in range(samples):
        label_vec = res.iloc[target_index] if target_dim == 0 else res.iloc[:, target_index]
        passed, reason, pos, neg, total = filter_target(label_vec)
        if passed:
            passed_targets.append(target_index)
        else:
            skipped_targets.append((target_index, reason, pos, neg, total))

    print(f"\n🚫 Skipped Targets: {len(skipped_targets)}")
    for idx, reason, pos, neg, total in skipped_targets:
        print(
            f"Target {idx}: skipped because {reason} (total={total}, pos={pos}, neg={neg})"
        )

    # Run training for each valid target in parallel using joblib
    def run_single_target(target_index):
        true_data, predict_data = drGT_new(
            res=res,
            null_mask=null_mask.values,
            target_dim=target_dim,
            target_index=target_index,
            S_d=S_d,
            S_c=S_c,
            S_g=S_g,
            A_cg=A_cg,
            A_dg=A_dg,
            params=params,
            device=device,
        )
        return true_data, predict_data

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_target)(target_index)
        for target_index in tqdm(passed_targets)
    )

    # Collect predictions and ground truth for metrics calculation
    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()
    for true_data, predict_data in results:
        true_datas = pd.concat([true_datas, pd.DataFrame(true_data).T], ignore_index=True)
        predict_datas = pd.concat([predict_datas, pd.DataFrame(predict_data).T], ignore_index=True)

    # Calculate evaluation metrics
    metrics_result = compute_metrics_stats(
        trial=None,  # No Optuna, so trial=None
        true=true_datas,
        pred=predict_datas,
        target_metrics=["AUROC", "AUPR", "F1", "ACC"],
    )

    print("Metrics:")
    print(metrics_result)

