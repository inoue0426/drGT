# 🧬 drGT

Official implementation of **drGT: Attention-Guided Gene Assessment for Drug Response in Drug-Cell-Gene Heterogeneous Network**
[![arXiv](https://img.shields.io/badge/arXiv-2405.08979-b31b1b.svg)](https://arxiv.org/abs/2405.08979)

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

2. Run the prediction script directly (CPU or GPU):

```bash
./run_drGT.py --task test2 --data nci --method GATv2 --cell_or_drug cell
```

> ✅ If `uv` is not installed:
> ```bash
> pip install uv
> ```

3. Example output:

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

probs, true_labels, attention = drGT.predict('best_model.pt', sampler, params)
evaluate_predictions(true_labels, probs)
```

✅ Ensure that `params` match the pretrained model's configuration (e.g., GNN layer, hidden sizes, etc.).

📓 For a full example, see [`predict_with_pretrained_model.ipynb`](predict_with_pretrained_model.ipynb).

---

## ⚙️ Interactive Use with Jupyter

To analyze results or explore predictions interactively:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install libraries
uv pip install -e .

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

```bibtex
@article{inoue2024drgat,
  title={drGT: Attention-Guided Gene Assessment of Drug Response Utilizing a Drug-Cell-Gene Heterogeneous Network},
  author={Inoue, Yoshitaka and Lee, Hunmin and Fu, Tianfan and Luna, Augustin},
  journal={arXiv preprint arXiv:2405.08979},
  year={2024}
}
```
