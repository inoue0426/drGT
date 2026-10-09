# 🧬 drGT

Official implementation of **drGT: Interpretable Drug Response Prediction with Attention-Guided Gene Attribution on a Drug-Cell-Gene Heterogeneous Graph**, published in [BMC Bioinformatics (2026)](https://doi.org/10.1186/s12859-026-06417-z).
[![BMC Bioinformatics](https://img.shields.io/badge/BMC%20Bioinformatics-2026-0073B1)](https://doi.org/10.1186/s12859-026-06417-z)
[![arXiv](https://img.shields.io/badge/arXiv-2405.08979-b31b1b.svg)](https://arxiv.org/abs/2405.08979)
[![Formatting](https://github.com/inoue0426/drGT/actions/workflows/python-format.yml/badge.svg?branch=main)](https://github.com/inoue0426/drGT/actions/workflows/python-format.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-%231674b1)](https://github.com/PyCQA/isort)
[![uv](https://img.shields.io/badge/uv-astral-6F2CAC)](https://github.com/astral-sh/uv)

![](Figs/Fig1.png)

`drGT` utilizes attention-based GNNs (e.g., GAT, GATv2, Transformer) to model a heterogeneous graph of drugs, cells, and genes. It predicts drug sensitivity and uncovers gene-level contributions via attention mechanisms.

---

## 🚀 Quick Start

> Requires: Python 3.10 or 3.11
> Uses [`uv`](https://github.com/astral-sh/uv) for lightweight execution

1. Clone the repository:

```bash
git clone https://github.com/inoue0426/drGT.git
cd drGT
```

2. Run a short training demo (CPU or GPU):

```bash
uv run --script run_drGT.py --task test1 --data nci --method GATv2
```

> ✅ If `uv` is not installed:
> ```bash
> pip install uv
> ```

This demo trains for **3 epochs per fold** using a **5-fold random split** of observed drug-cell pairs. It checks the training workflow; it does not run pretrained inference or reproduce the paper's full-training results. For inference without training, use the pretrained notebook below. For leave-cell-out or leave-drug-out evaluation, use [`Test2_leave_X_out/run_drGAT.py`](Test2_leave_X_out/run_drGAT.py); the demo runner only supports `test1`.

3. Example output from the short demo (values vary between runs):

```
Using device: cpu
Best model found at epoch 2
ACC           : 0.511 (±0.009)
Precision     : 0.215 (±0.296)
Recall        : 0.221 (±0.438)
F1            : 0.170 (±0.289)
AUROC         : 0.535 (±0.024)
AUPR          : 0.532 (±0.029)
```

---

## 🧠 Using Pretrained Models

To evaluate without retraining:

```python
from drGT import drGT
from drGT.metrics import evaluate_predictions

probs, true_labels, attention = drGT.predict('best_model_nci.pt', sampler, params)
evaluate_predictions(true_labels, probs)
```

> ℹ️ If you only want to generate predictions (inference) without reporting performance metrics, you can use the same pretrained model and skip evaluation.

✅ Ensure that `params` match the pretrained model's configuration (e.g., GNN layer, hidden sizes, etc.).

📓 For a full example, see [`predict_with_pretrained_model.ipynb`](predict_with_pretrained_model.ipynb).

You can use this pretrained model for existing drugs and cell lines. See [`cell_drug_availability_across_datasets.ipynb`](cell_drug_availability_across_datasets.ipynb) to check which cell lines and drugs are available across datasets.

---

## ⚙️ Interactive Use with Jupyter

To analyze results or explore predictions interactively:

```bash
# Create and activate a project environment
uv venv --python 3.10
source .venv/bin/activate

# Install libraries
uv pip install -e . jupyter

# Register Jupyter kernel
python -m ipykernel install --user --name=drGT --display-name "Python (drGT)"

# Launch notebook
jupyter notebook
```

---

## 🔄 Retraining

To retrain **drGT** from scratch, please use the scripts provided in the
[`Test1_random_split`](./Test1_random_split/) and
[`Test2_leave_X_out`](./Test2_leave_X_out/) directories (e.g., `run_drGAT.py`).
These scripts allow you to retrain the model under various experimental setups and
data-splitting strategies.

If you wish to use your own dataset, please prepare the following:
- A **drug response matrix** (drugs × cell lines)
- A **gene expression matrix** (cell lines × genes)
- **Drug chemical structures** as **SMILES** strings (column 1: Drug, column 2: SMILES)

You can modify the `drGT/load_data` module to support your custom file formats
or alternative data sources as needed.

---

## 🔄 When is Retraining Required?

You do **NOT** need retraining if:
- You use the same dataset and split as the pretrained model
- Model architecture and hyperparameters are unchanged

Retraining **IS REQUIRED** if:
- You use a different dataset or add new drugs/cell lines
- You change data splitting strategy (e.g., leave-drug-out)
- You modify model architecture or hyperparameters
- You aim to interpret attention scores on your own data

---

## ⚡️ GPU Acceleration

All experiments were conducted on **Linux with NVIDIA A100**.
`drGT` benefits significantly from GPU acceleration via PyTorch and PyTorch Geometric.

> ✅ Ensure you install a **CUDA-compatible PyTorch version**
> (e.g., `torch==2.x` with `CUDA 11.8` for A100)

👉 [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

---

## 📁 Directory Overview

```
drGT/                 # Core model implementation
configs/              # YAML configs for experiments
Test1_random_split/   # Scripts for random masking experiments
Test2_leave_X_out/    # Scripts for leave-X-out experiments
preprocess/           # Data preprocessing notebooks
data/                 # Preprocessed datasets
```

---

## ❓ Questions or Issues

Please feel free to:

- Open a GitHub Issue and mention **@inoue0426**
- Or email **inoue019@umn.edu**

We're happy to help and collaborate!

---

## 📖 Citation

If you use drGT in your research, please cite the published article in [BMC Bioinformatics](https://doi.org/10.1186/s12859-026-06417-z).

```bibtex
@article{inoue2026drgt,
  title={drGT: Interpretable Drug Response Prediction with Attention-Guided Gene Attribution on a Drug-Cell-Gene Heterogeneous Graph},
  author={Inoue, Yoshitaka and Lee, Hunmin and Fu, Tianfan and Kuang, Rui and Luna, Augustin},
  journal={BMC Bioinformatics},
  volume={27},
  number={1},
  pages={204},
  year={2026},
  doi={10.1186/s12859-026-06417-z},
  url={https://doi.org/10.1186/s12859-026-06417-z}
}
```

