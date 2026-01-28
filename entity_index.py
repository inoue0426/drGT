from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import re

import json
from pathlib import Path

# ============================================================
# Normalization utilities
# ============================================================

def normalize_cell_line(name: str) -> str:
    """
    Normalize cell line names across datasets.

    Rules:
      - strip whitespace
      - uppercase
      - convert '_' and '/' to '-'
      - keep only A-Z, 0-9, '-'
      - collapse repeated '-'
    """
    s = str(name).strip().upper()
    s = s.replace("_", "-").replace("/", "-")
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s)
    return s


def normalize_drug(name: str) -> str:
    """
    Normalize drug names across datasets.

    Rules:
      - strip whitespace
      - uppercase
      - remove special characters
      - collapse whitespace
    """
    s = str(name).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


# load_data signature (dependency injection for testability)
LoadDataFn = Callable[..., Tuple]


# ============================================================
# Cell line index
# ============================================================

@dataclass(frozen=True)
class CellLineIndex:
    datasets: Tuple[str, ...]
    global_index: Dict[str, Dict[str, str]]          # norm -> {dataset: raw}
    entity_map_by_ds: Dict[str, Dict[str, str]]      # dataset -> {norm: raw}

    def all(self) -> List[str]:
        """All canonical (normalized) cell line names."""
        return sorted(self.global_index.keys())

    def where(self, cell_line: str) -> Dict[str, str]:
        """Which datasets contain this cell line (with dataset-specific spelling)."""
        key = normalize_cell_line(cell_line)
        return dict(self.global_index.get(key, {}))

    def has(self, cell_line: str, dataset: str) -> bool:
        """Check existence in a dataset."""
        key = normalize_cell_line(cell_line)
        return key in self.entity_map_by_ds.get(dataset, {})

    def dataset_name(self, cell_line: str, dataset: str) -> Optional[str]:
        """Return dataset-specific spelling."""
        key = normalize_cell_line(cell_line)
        return self.entity_map_by_ds.get(dataset, {}).get(key)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(
                {
                    "datasets": self.datasets,
                    "global_index": self.global_index,
                    "entity_map_by_ds": self.entity_map_by_ds,
                },
                f,
                indent=2,
            )

    @staticmethod
    def from_json(path: str | Path) -> "CellLineIndex":
        with Path(path).open() as f:
            obj = json.load(f)
        return CellLineIndex(
            datasets=tuple(obj["datasets"]),
            global_index=obj["global_index"],
            entity_map_by_ds=obj["entity_map_by_ds"],
        )


def build_cell_line_index(
    datasets: Iterable[str],
    load_data: LoadDataFn,
    *,
    is_zero_pad: bool,
    cache_path: str | Path | None = None,
    verbose: bool = False,
) -> CellLineIndex:

    if cache_path is not None and Path(cache_path).exists():
        if verbose:
            print(f"[CellLineIndex] loading cache: {cache_path}")
        return CellLineIndex.from_json(cache_path)

    entity_map_by_ds: Dict[str, Dict[str, str]] = {}
    global_index: Dict[str, Dict[str, str]] = {}

    for ds in datasets:
        if verbose:
            print(f"[CellLineIndex] loading {ds} ...")

        drugAct, *_ = load_data(ds, is_zero_pad=is_zero_pad, verbose=False)
        raw_names = [str(x) for x in drugAct.index.tolist()]

        ds_map: Dict[str, str] = {}
        for raw in raw_names:
            norm = normalize_cell_line(raw)
            ds_map.setdefault(norm, raw)

        entity_map_by_ds[ds] = ds_map
        for norm, raw in ds_map.items():
            global_index.setdefault(norm, {})[ds] = raw

    # --- existing build logic ---
    idx = CellLineIndex(datasets, global_index, entity_map_by_ds)

    if cache_path is not None:
        if verbose:
            print(f"[CellLineIndex] saving cache: {cache_path}")
        idx.to_json(cache_path)

    return idx


# ============================================================
# Drug index
# ============================================================

@dataclass(frozen=True)
class DrugIndex:
    datasets: Tuple[str, ...]
    global_index: Dict[str, Dict[str, str]]          # norm -> {dataset: raw}
    entity_map_by_ds: Dict[str, Dict[str, str]]      # dataset -> {norm: raw}

    def all(self) -> List[str]:
        """All canonical (normalized) drug names."""
        return sorted(self.global_index.keys())

    def where(self, drug: str) -> Dict[str, str]:
        """Which datasets contain this drug (with dataset-specific spelling)."""
        key = normalize_drug(drug)
        return dict(self.global_index.get(key, {}))

    def has(self, drug: str, dataset: str) -> bool:
        """Check existence in a dataset."""
        key = normalize_drug(drug)
        return key in self.entity_map_by_ds.get(dataset, {})

    def dataset_name(self, drug: str, dataset: str) -> Optional[str]:
        """Return dataset-specific spelling."""
        key = normalize_drug(drug)
        return self.entity_map_by_ds.get(dataset, {}).get(key)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(
                {
                    "datasets": self.datasets,
                    "global_index": self.global_index,
                    "entity_map_by_ds": self.entity_map_by_ds,
                },
                f,
                indent=2,
            )

    @staticmethod
    def from_json(path: str | Path) -> "DrugIndex":
        with Path(path).open() as f:
            obj = json.load(f)
        return DrugIndex(
            datasets=tuple(obj["datasets"]),
            global_index=obj["global_index"],
            entity_map_by_ds=obj["entity_map_by_ds"],
        )

def build_drug_index(
    datasets: Iterable[str],
    load_data: LoadDataFn,
    *,
    is_zero_pad: bool,
    cache_path: str | Path | None = None,
    verbose: bool = False,
) -> DrugIndex:

    if cache_path is not None and Path(cache_path).exists():
        if verbose:
            print(f"[DrugIndex] loading cache: {cache_path}")
        return DrugIndex.from_json(cache_path)

    entity_map_by_ds: Dict[str, Dict[str, str]] = {}
    global_index: Dict[str, Dict[str, str]] = {}

    for ds in datasets:
        if verbose:
            print(f"[DrugIndex] loading {ds} ...")

        drugAct, *_ = load_data(ds, is_zero_pad=is_zero_pad, verbose=False)
        raw_names = [str(x) for x in drugAct.columns.tolist()]

        ds_map: Dict[str, str] = {}
        for raw in raw_names:
            norm = normalize_drug(raw)
            ds_map.setdefault(norm, raw)

        entity_map_by_ds[ds] = ds_map
        for norm, raw in ds_map.items():
            global_index.setdefault(norm, {})[ds] = raw

    # --- existing build logic ---
    idx = DrugIndex(datasets, global_index, entity_map_by_ds)

    if cache_path is not None:
        if verbose:
            print(f"[DrugIndex] saving cache: {cache_path}")
        idx.to_json(cache_path)

    return idx