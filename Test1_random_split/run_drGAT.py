import argparse
import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

# Add parent directory of the current directory to the system path
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGT import drGT
from drGT.load_data import load_data
from drGT.metrics import compute_metrics_stats
from drGT.myutils import get_all_edges_and_labels
from drGT.sampler import BalancedSampler

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", type=str, choices=["GAT", "GATv2", "Transformer"], default="Transformer",
    help="Type of GNN layer"
)
parser.add_argument(
    "--data", type=str, choices=["gdsc1", "gdsc2", "ctrp", "nci"], default="nci",
    help="Dataset to use"
)
# Set default values for each parameter
parser.add_argument("--activation", type=str, default="relu", help="Activation function (default: relu)")
parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout rate for attention (default: 0.1)")
parser.add_argument("--dropout1", type=float, default=0.1, help="Dropout rate for 1st layer (default: 0.1)")
parser.add_argument("--dropout2", type=float, default=0.2, help="Dropout rate for 2nd layer (default: 0.2)")
parser.add_argument("--dropout3", type=float, default=0.5, help="Dropout rate for 3rd layer (default: 0.5)")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs (default: 1000)")
parser.add_argument("--final_mlp_layers", type=int, default=2, help="Number of layers in the final MLP (default: 2)")
parser.add_argument("--heads", type=int, default=8, help="Number of heads (default: 8)")
parser.add_argument("--hidden1", type=int, default=289, help="Number of hidden units in 1st layer (default: 289)")
parser.add_argument("--hidden2", type=int, default=215, help="Number of hidden units in 2nd layer (default: 215)")
parser.add_argument("--hidden3", type=int, default=66, help="Number of hidden units in 3rd layer (default: 66)")
parser.add_argument(
    "--is_zero_pad", action="store_true", default=True,
    help="Specify if zero padding is used (default: True)"
)
parser.add_argument("--lr", type=float, default=0.00011219837311484963, help="Learning rate (default: 0.00011219837311484963)")
parser.add_argument("--n_layers", type=int, default=2, help="Number of GNN layers (default: 2)")
parser.add_argument("--norm_type", type=str, default="GraphNorm", choices=["GraphNorm", "BatchNorm", "LayerNorm"], help="Normalization type (default: GraphNorm)")
parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"], help="Optimizer (default: Adam)")
parser.add_argument("--weight_decay", type=float, default=0.0023400993222972882, help="Weight decay (default: 0.0023400993222972882)")
parser.add_argument("--scheduler", type=str, default="None", choices=["None", "Cosine"], help="Scheduler (default: None)")
args = parser.parse_args()

method = args.method
data = args.data

if __name__ == "__main__":
    # Load data
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
    ) = load_data(data, is_zero_pad=args.is_zero_pad)

    # Define parameters for the model
    params = {
        "n_drug": S_d.shape[0],
        "n_cell": S_c.shape[0],
        "n_gene": S_g.shape[0],
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
        "scheduler": args.scheduler if args.scheduler != "None" else None,
        "norm_type": args.norm_type,
        "n_layers": args.n_layers,
        "gnn_layer": method,
        "final_mlp_layers": args.final_mlp_layers,
        "attention_dropout": args.attention_dropout,
    }

    # Get all edges and labels
    all_edges, all_labels = get_all_edges_and_labels(drugAct, null_mask)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()

    # Train and evaluate with 5-fold cross-validation
    for train_idx, test_idx in tqdm(kf.split(all_edges)):
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

        (
            _,
            _,
            _,
            true_data,
            predict_data,
            _,
            _,
            _,
            _,
        ) = drGT.train(sampler, params=params, device=device, verbose=False)

        true_datas = pd.concat(
            [true_datas, pd.DataFrame(true_data).T], ignore_index=True
        )
        predict_datas = pd.concat(
            [predict_datas, pd.DataFrame(predict_data).T], ignore_index=True
        )

    # Calculate evaluation metrics
    metrics_result = compute_metrics_stats(
        trial=None,
        true=true_datas,
        pred=predict_datas,
        target_metrics=["AUROC", "AUPR", "F1", "ACC"],
    )
    print("5-fold evaluation results:")
    for k, v in zip(["AUROC", "AUPR", "F1", "ACC"], metrics_result["target_values"]):
        print(f"{k}: {v:.4f}")

