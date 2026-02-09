"""
Utilities for dataset pipelines shared by API handlers.
"""

from typing import Any, Dict, Tuple


def random_select_dataset(ds, n: int, seed: int, requested_split: str = None):
    """Select N random rows using np.random with seed; restore RNG state after.
    Returns same type as input: DatasetDict or single Dataset.
    """
    import numpy as np

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        if hasattr(ds, "keys"):
            out = {}
            for split_name, d in ds.items():
                total = d.num_rows
                k = min(n, total)
                indices = np.random.choice(total, size=k, replace=False)
                out[split_name] = d.select(indices.tolist())
            import datasets

            return datasets.DatasetDict(out)
        else:
            split_name = (requested_split and str(requested_split).strip()) or "dataset"
            d = ds
            total = d.num_rows
            k = min(n, total)
            indices = np.random.choice(total, size=k, replace=False)
            return d.select(indices.tolist())
    finally:
        np.random.set_state(state)


def dataset_to_info(ds, requested_split: str = None) -> dict:
    """Build serializable dataset_info: columns, num_rows per split, sample rows.
    Handles both DatasetDict (multiple splits) and single Dataset (when split= is used).
    """
    # When split= is passed, load_dataset returns a single Dataset, not DatasetDict
    if hasattr(ds, "keys"):
        # DatasetDict
        splits = list(ds.keys())
        datasets_by_split = {s: ds[s] for s in splits}
    else:
        # Single Dataset
        split_name = (requested_split and str(requested_split).strip()) or "dataset"
        splits = [split_name]
        datasets_by_split = {split_name: ds}

    num_rows = {s: d.num_rows for s, d in datasets_by_split.items()}
    columns = []
    sample_rows = {}
    sample_size = 50
    for split in splits:
        d = datasets_by_split[split]
        if not columns and hasattr(d, "column_names"):
            columns = list(d.column_names)
        try:
            n = min(sample_size, d.num_rows)
            if n > 0:
                batch = d.select(range(n))
                rows = []
                for i in range(n):
                    row = {col: safe_value(batch[col][i]) for col in (batch.column_names or [])}
                    rows.append(row)
                sample_rows[split] = rows
        except Exception:  # noqa: BLE001
            sample_rows[split] = []
    return {
        "splits": splits,
        "num_rows": num_rows,
        "columns": columns,
        "sample_rows": sample_rows,
    }


def safe_value(v):
    """Convert value to JSON-serializable form."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): safe_value(x) for k, x in v.items()}
    return str(v)


def load_pipeline_dataset(pipeline: Dict[str, Any]) -> Tuple[Any, str]:
    """Load HF dataset from pipeline's path and options. Returns (ds, requested_split)."""
    import datasets

    path = (pipeline.get("hf_dataset_path") or "").strip()
    if not path:
        raise ValueError("Pipeline has no dataset path. Load a dataset first.")
    opts = pipeline.get("hf_load_options") or {}
    config_name = (opts.get("config") or "").strip() or None
    split = opts.get("split")
    random_n = opts.get("random_n")
    seed = opts.get("seed")
    load_kwargs = {"path": path}
    if config_name:
        load_kwargs["name"] = config_name
    if split:
        load_kwargs["split"] = split
    ds = datasets.load_dataset(**load_kwargs)
    if random_n is not None and random_n > 0 and seed is not None:
        ds = random_select_dataset(ds, n=random_n, seed=seed, requested_split=split)
    return ds, split


def get_process_function(code: str):
    """Execute user code and return a callable named 'process' (example -> dict)."""
    if not (code or code.strip()):
        return None
    namespace = {
        "json": __import__("json"),
        "re": __import__("re"),
    }
    exec(code.strip(), namespace)
    fn = namespace.get("process")
    if not callable(fn):
        raise ValueError(
            "Code must define a function named 'process' that takes one argument (example dict) and returns a dict."
        )
    return fn

