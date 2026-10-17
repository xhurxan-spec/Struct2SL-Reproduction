"""Data loading and gene-pair feature construction for Struct2SL-Reproduction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FEATURE_DIMENSIONS = {
    "structure": 128,
    "sequence": 1024,
    "ppi": 128,
}

PAIR_COLUMNS = ("n1.name", "n2.name")


def load_pickle(path):
    """Load a feature dictionary from a trusted Python pickle file.

    Pickle files can execute arbitrary code during loading; only load files
    from trusted sources.
    """
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a dictionary in {path}; got {type(data).__name__}"
        )

    return data


def load_structure_features(structure_dir):
    """Load one numeric structure-feature vector per gene."""
    structure_dir = Path(structure_dir)
    features = {}

    for path in sorted(structure_dir.glob("*.txt")):
        with path.open("r", encoding="utf-8") as handle:
            values = np.fromstring(handle.readline(), sep=" ", dtype=np.float32)

        if values.size == 0:
            raise ValueError(f"Empty structure feature vector: {path}")

        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite structure feature values: {path}")

        features[path.stem] = values

    if not features:
        raise FileNotFoundError(
            f"No structural feature .txt files found in {structure_dir}"
        )

    return features


def load_feature_sources(structure_dir, sequence_file, ppi_file):
    """Load and validate structural, sequence, and PPI feature dictionaries."""
    structure = load_structure_features(structure_dir)
    sequence_raw = load_pickle(sequence_file)
    ppi_raw = load_pickle(ppi_file)

    sequence = {
        gene: np.asarray(value, dtype=np.float32).reshape(-1)
        for gene, value in sequence_raw.items()
    }
    ppi = {
        gene: np.asarray(value, dtype=np.float32).reshape(-1)
        for gene, value in ppi_raw.items()
    }

    feature_sources = {
        "structure": structure,
        "sequence": sequence,
        "ppi": ppi,
    }

    for source_name, source in feature_sources.items():
        expected_dim = EXPECTED_FEATURE_DIMENSIONS[source_name]

        for gene, vector in source.items():
            if vector.size != expected_dim:
                raise ValueError(
                    f"{source_name}: gene {gene!r} has {vector.size} "
                    f"features; expected {expected_dim}"
                )

            if not np.isfinite(vector).all():
                raise ValueError(
                    f"{source_name}: gene {gene!r} has non-finite values"
                )

    return feature_sources


def load_pair_table(path):
    """Load an SL/non-SL pair table using the original column names."""
    df = pd.read_csv(path)

    missing = set(PAIR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    pairs = []
    for row_index, row in df.iterrows():
        identifiers = []
        for column in PAIR_COLUMNS:
            value = row[column]
            if pd.isna(value):
                raise ValueError(
                    f"{path} row {row_index}: {column} contains a "
                    "missing gene identifier"
                )

            identifier = str(value)
            if not identifier.strip():
                raise ValueError(
                    f"{path} row {row_index}: {column} contains an "
                    "empty or whitespace-only gene identifier"
                )

            identifiers.append(identifier)

        pairs.append(tuple(identifiers))

    return pairs


def build_pair_features(pair, feature_sources):
    """Concatenate the three feature vectors for both genes."""
    gene1, gene2 = pair
    parts = []

    for gene in (gene1, gene2):
        for source_name in ("structure", "sequence", "ppi"):
            source = feature_sources[source_name]

            if gene not in source:
                raise KeyError(
                    f"Gene {gene!r} is missing {source_name} features"
                )

            parts.append(source[gene])

    vector = np.concatenate(parts).astype(np.float32, copy=False)

    if vector.size != 2560:
        raise ValueError(
            f"Expected 2560 pair features, received {vector.size}"
        )

    if not np.isfinite(vector).all():
        raise ValueError(f"Pair {pair} has non-finite feature values")

    return vector


def build_labeled_dataset(
    feature_sources,
    positive_pairs,
    negative_pairs,
    *,
    require_both_genes=True,
):
    """
    Build a dataset from supplied positive and negative pairs.

    This function does not generate additional negative pairs and does not
    standardize features. Those decisions belong to the experiment protocol.
    """
    valid_genes = set.intersection(
        *(set(source) for source in feature_sources.values())
    )

    features = []
    labels = []
    excluded_positive = 0
    excluded_negative = 0

    for pairs, label in ((positive_pairs, 1), (negative_pairs, 0)):
        for pair in pairs:
            gene1, gene2 = pair

            if require_both_genes and (
                gene1 not in valid_genes or gene2 not in valid_genes
            ):
                if label == 1:
                    excluded_positive += 1
                else:
                    excluded_negative += 1
                continue

            features.append(build_pair_features(pair, feature_sources))
            labels.append(label)

    if not features:
        raise ValueError("No usable gene pairs remain after feature filtering")

    X = np.stack(features).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)

    report = {
        "feature_shape": list(X.shape),
        "label_shape": list(y.shape),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
        "excluded_positive_pairs": excluded_positive,
        "excluded_negative_pairs": excluded_negative,
        "shared_gene_count": len(valid_genes),
    }

    return X, y, report
