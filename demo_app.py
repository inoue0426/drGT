"""Local browser demo for the bundled NCI pretrained checkpoint."""

from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from drGT import drGT
from drGT.load_data import load_data
from drGT.myutils import get_all_edges_and_labels, get_model_params
from drGT.sampler import BalancedSampler


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "best_model_nci.pt"


@lru_cache(maxsize=1)
def prepare():
    """Load the same NCI data and model settings used by the pretrained notebook."""
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT}")
    params = get_model_params(
        "test1", "nci", "Transformer", config_dir=str(ROOT / "configs")
    ).copy()
    (
        responses, missing, drug_features, cell_features, gene_features,
        _, _, _, cell_gene, drug_gene,
    ) = load_data("nci", is_zero_pad=params["is_zero_pad"])
    params.update(
        n_drug=drug_features.shape[0],
        n_cell=cell_features.shape[0],
        n_gene=gene_features.shape[0],
        gnn_layer="Transformer",
    )
    edges, labels = get_all_edges_and_labels(responses, missing)
    # Only observed, binary-labelled pairs are offered. This also avoids
    # presenting an invented ground-truth label for missing response data.
    by_cell = {}
    pair_to_index = {}
    cells = [str(value) for value in responses.index]
    drugs = [str(value) for value in responses.columns]
    for index, (cell_index, drug_index) in enumerate(edges):
        cell, drug = cells[cell_index], drugs[drug_index]
        by_cell.setdefault(cell, []).append(drug)
        pair_to_index[(cell, drug)] = index
    if not by_cell:
        raise ValueError("No observed drug response pairs found in the NCI data")
    for choices in by_cell.values():
        choices.sort()
    return (
        params, responses, missing, drug_features, cell_features, gene_features,
        cell_gene, drug_gene, edges, labels, by_cell, pair_to_index,
    )


def available_drugs(cell):
    choices = prepare()[-2].get(cell, [])
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def predict(cell, drug):
    (
        params, responses, missing, drug_features, cell_features, gene_features,
        cell_gene, drug_gene, edges, labels, _, pair_to_index,
    ) = prepare()
    selected = pair_to_index.get((cell, drug))
    if selected is None:
        raise gr.Error("Select an available cell line and drug pair.")

    # Match the pretrained notebook's sampler interface. Exclude the selected
    # response from the graph for this request to avoid direct label leakage.
    train_indices = np.arange(len(edges))
    train_indices = train_indices[train_indices != selected]
    sampler = BalancedSampler(
        responses, edges, labels, train_indices, np.array([selected]),
        missing, drug_features, cell_features, gene_features, cell_gene, drug_gene,
    )
    probabilities, _, _ = drGT.predict(str(CHECKPOINT), sampler, params)
    probability = float(np.asarray(probabilities).reshape(-1)[0])
    return round(probability, 4)


def build_demo():
    by_cell = prepare()[-2]
    first_cell = next(iter(by_cell))
    with gr.Blocks(title="drGT · NCI drug response demo") as demo:
        gr.Markdown(
            "# drGT · Drug response demo\n"
            "Select an NCI cell line and a drug with an observed response. "
            "The bundled pretrained Transformer checkpoint returns a predicted "
            "probability of **sensitivity (label 1)**."
        )
        with gr.Row():
            cell = gr.Dropdown(
                choices=list(by_cell), value=first_cell,
                label="Cell line", filterable=True,
            )
            drug = gr.Dropdown(
                choices=by_cell[first_cell], value=by_cell[first_cell][0],
                label="Drug", filterable=True,
            )
        result = gr.Number(label="Predicted sensitivity probability", precision=4)
        cell.change(available_drugs, inputs=cell, outputs=drug)
        gr.Button("Predict", variant="primary").click(
            predict, inputs=[cell, drug], outputs=result, concurrency_limit=1
        )
        gr.Markdown(
            "**Research demonstration only.** The selected response is removed "
            "from the graph assembled for this request. The original training "
            "split of the bundled checkpoint has not been verified, so this "
            "display does not establish held-out accuracy or independent validation. "
            "Inputs are limited to the NCI dataset; other drugs or cell lines "
            "require new data preparation and retraining."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(share=False)
