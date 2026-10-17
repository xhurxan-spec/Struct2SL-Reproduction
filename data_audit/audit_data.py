"""Audit Struct2SL feature coverage and synthetic-lethality pairs."""

import argparse
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAIR_COLUMNS = ("n1.name", "n2.name")


def load_pickle(path):
    """Load a feature dictionary from a trusted Python pickle file.

    Pickle files can execute arbitrary code during loading; only load files
    from trusted sources.
    """
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def numeric_array(value):
    return np.asarray(value, dtype=np.float32).reshape(-1)


def audit_pair_relationships(positive_pairs, negative_pairs, max_examples=10):
    """Summarize unordered duplicate, reversed, and cross-label pairs."""

    def summarize(pairs):
        counts = Counter(tuple(sorted(pair)) for pair in pairs)
        orientations = defaultdict(set)
        for pair in pairs:
            canonical = tuple(sorted(pair))
            orientations[canonical].add(tuple(pair))

        duplicate_groups = sorted(
            (pair, count)
            for pair, count in counts.items()
            if count > 1
        )
        reversed_groups = sorted(
            (pair, sorted(orientation_set))
            for pair, orientation_set in orientations.items()
            if len(orientation_set) > 1 and pair[0] != pair[1]
        )

        return {
            "rows": len(pairs),
            "unordered_unique_pairs": len(counts),
            "duplicate_unordered_pair_groups": len(duplicate_groups),
            "duplicate_unordered_pair_rows": sum(
                count - 1 for _, count in duplicate_groups
            ),
            "duplicate_unordered_pair_examples": [
                {"pair": list(pair), "count": count}
                for pair, count in duplicate_groups[:max_examples]
            ],
            "reversed_pair_groups": len(reversed_groups),
            "reversed_pair_examples": [
                {
                    "pair": list(pair),
                    "orientations": [
                        list(orientation) for orientation in orientations
                    ],
                }
                for pair, orientations in reversed_groups[:max_examples]
            ],
            "_counts": counts,
        }

    positive = summarize(positive_pairs)
    negative = summarize(negative_pairs)
    positive_counts = positive.pop("_counts")
    negative_counts = negative.pop("_counts")
    overlap = sorted(set(positive_counts) & set(negative_counts))

    return {
        "positive": positive,
        "negative": negative,
        "positive_negative_overlap_groups": len(overlap),
        "positive_negative_overlap_examples": [
            {
                "pair": list(pair),
                "positive_count": positive_counts[pair],
                "negative_count": negative_counts[pair],
            }
            for pair in overlap[:max_examples]
        ],
    }


def run_audit(data_root, output_path, *, overwrite=False):
    """Run the audit and write its JSON report."""
    data_root = Path(data_root).expanduser().resolve()
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing audit report: "
            f"{output_path}. Choose a new output path or pass --overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STRUCT2SL DATA AUDIT")
    print("=" * 65)

    sequence = load_pickle(data_root / "sequence_features.npz")
    ppi_raw = load_pickle(data_root / "ppi_features.npz")
    ppi = {gene: numeric_array(value) for gene, value in ppi_raw.items()}

    structure = {}
    structure_dir = data_root / "struct_features"
    for path in structure_dir.glob("*.txt"):
        with path.open("r", encoding="utf-8") as handle:
            structure[path.stem] = np.fromstring(
                handle.readline(), sep=" ", dtype=np.float32
            )

    feature_dicts = {
        "structure": structure,
        "sequence": {
            gene: numeric_array(value) for gene, value in sequence.items()
        },
        "ppi": ppi,
    }

    feature_report = {}
    for name, features in feature_dicts.items():
        dimensions = sorted({len(value) for value in features.values()})
        invalid = [
            gene
            for gene, value in features.items()
            if value.size == 0 or not np.isfinite(value).all()
        ]

        feature_report[name] = {
            "genes": len(features),
            "dimensions": dimensions,
            "invalid_gene_count": len(invalid),
            "invalid_gene_examples": invalid[:10],
        }

        print(f"\n{name.upper()}")
        print("Genes:", len(features))
        print("Dimensions:", dimensions)
        print("Invalid vectors:", len(invalid))

    valid_genes = set.intersection(
        *(set(features) for features in feature_dicts.values())
    )
    print("\nGenes shared across all feature sources:", len(valid_genes))

    dataset_report = {}
    dataset_pairs = {}
    for name, filename in [
        ("SL", "Human_SL_filtered.csv"),
        ("nonSL", "Human_nonSL.csv"),
    ]:
        path = data_root / filename
        df = pd.read_csv(path)

        required = {"n1.name", "n2.name"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Unexpected columns in {filename}: {df.columns.tolist()}"
            )

        pairs = list(
            zip(df["n1.name"].astype(str), df["n2.name"].astype(str))
        )
        dataset_pairs[name] = pairs
        retained = [
            (gene1, gene2)
            for gene1, gene2 in pairs
            if gene1 in valid_genes and gene2 in valid_genes
        ]

        try:
            reported_file = str(path.relative_to(ROOT))
        except ValueError:
            reported_file = str(path)

        dataset_report[name] = {
            "file": reported_file,
            "original_rows": len(df),
            "unique_ordered_pairs": len(set(pairs)),
            "gene_identifier_dtypes": {
                column: str(df[column].dtype) for column in PAIR_COLUMNS
            },
            "rows_with_both_genes_covered": len(retained),
            "rows_excluded_for_missing_features": len(df) - len(retained),
        }

        print(f"\n{name} DATASET")
        for key, value in dataset_report[name].items():
            print(f"{key}: {value}")

    pair_audit = audit_pair_relationships(
        dataset_pairs["SL"], dataset_pairs["nonSL"]
    )

    report = {
        "feature_sources": feature_report,
        "shared_gene_count": len(valid_genes),
        "feature_dimensions_per_gene": {
            name: feature_report[name]["dimensions"]
            for name in feature_report
        },
        "dataset_coverage": dataset_report,
        "pair_audit": pair_audit,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nSaved audit report:", output_path)
    print("Original datasets were not modified.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=str(ROOT.parent / "data"),
        help="Directory containing the biological data files.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data_audit" / "data_audit.json"),
        help="Path for the generated audit JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing audit JSON explicitly.",
    )
    args = parser.parse_args()
    run_audit(args.data_root, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
