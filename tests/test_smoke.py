import json
import tempfile
import unittest
import importlib
from pathlib import Path

import numpy as np
import torch

from src.data_pipeline import build_pair_features, load_pair_table
from src.model import Struct2SLMLP, load_checkpoint, save_checkpoint
from src.train import ensure_output_paths_available, resolve_data_path
from data_audit.audit_data import audit_pair_relationships, run_audit


class Struct2SLSmokeTests(unittest.TestCase):
    def assert_pair_table_error(self, csv_text, message):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                load_pair_table(path)

    def test_data_audit_import_is_side_effect_free(self):
        module = importlib.import_module("data_audit.audit_data")
        self.assertTrue(callable(module.main))
        self.assertTrue(callable(module.run_audit))

    def test_data_audit_refuses_existing_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "audit.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                FileExistsError, "Refusing to overwrite existing audit report"
            ):
                run_audit(Path(tmp) / "missing-data", output)

    def test_data_audit_allows_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "audit.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                run_audit(Path(tmp) / "missing-data", output, overwrite=True)

    def test_config_lists_all_recorded_metrics(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "experiment.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["evaluation"]["metrics"],
            [
                "accuracy",
                "roc_auc",
                "average_precision",
                "f1",
                "precision",
                "recall",
                "specificity",
            ],
        )

    def test_pair_audit_reports_unordered_duplicates_and_conflicts(self):
        report = audit_pair_relationships(
            [("A", "B"), ("B", "A"), ("A", "B"), ("C", "D")],
            [("D", "C"), ("C", "D"), ("E", "F"), ("E", "F")],
        )

        self.assertEqual(
            report["positive"]["duplicate_unordered_pair_groups"], 1
        )
        self.assertEqual(
            report["positive"]["duplicate_unordered_pair_rows"], 2
        )
        self.assertEqual(report["positive"]["reversed_pair_groups"], 1)
        self.assertEqual(
            report["negative"]["duplicate_unordered_pair_groups"], 2
        )
        self.assertEqual(report["negative"]["reversed_pair_groups"], 1)
        self.assertEqual(report["positive_negative_overlap_groups"], 1)
        self.assertEqual(
            report["positive_negative_overlap_examples"][0]["pair"],
            ["C", "D"],
        )

    def test_model_output_shape_and_logits(self):
        model = Struct2SLMLP()
        output = model(torch.zeros(3, 2560))
        self.assertEqual(tuple(output.shape), (3, 1))

    def test_pair_feature_construction(self):
        sources = {
            "structure": {"A": np.zeros(128), "B": np.ones(128)},
            "sequence": {"A": np.zeros(1024), "B": np.ones(1024)},
            "ppi": {"A": np.zeros(128), "B": np.ones(128)},
        }
        vector = build_pair_features(("A", "B"), sources)
        self.assertEqual(vector.shape, (2560,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.all(vector[:128] == 0))
        self.assertTrue(np.all(vector[128:1152] == 0))
        self.assertTrue(np.all(vector[1280:] == 1))

    def test_external_data_root_resolution(self):
        resolved = resolve_data_path("/tmp/struct2sl-data", "data/foo.csv")
        self.assertEqual(
            resolved,
            Path("/tmp/struct2sl-data/foo.csv").resolve(),
        )

    def test_output_protection_uses_selected_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ensure_output_paths_available(1, output_dir)

            results_dir = output_dir / "results"
            results_dir.mkdir()
            (results_dir / "run_1.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                FileExistsError, "Refusing to overwrite existing experiment outputs"
            ):
                ensure_output_paths_available(1, output_dir)

    def test_missing_gene_identifier_is_rejected(self):
        self.assert_pair_table_error(
            "n1.name,n2.name\n,BRCA1\n",
            "missing gene identifier",
        )

    def test_empty_gene_identifier_is_rejected(self):
        self.assert_pair_table_error(
            'n1.name,n2.name\n"",BRCA1\n',
            "missing gene identifier",
        )

    def test_whitespace_only_gene_identifier_is_rejected(self):
        self.assert_pair_table_error(
            "n1.name,n2.name\n   ,BRCA1\n",
            "empty or whitespace-only gene identifier",
        )

    def test_nonempty_gene_identifier_whitespace_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            path.write_text(
                "n1.name,n2.name\n  TP53  ,BRCA1  \n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_pair_table(path),
                [("  TP53  ", "BRCA1  ")],
            )

    def test_checkpoint_round_trip(self):
        model = Struct2SLMLP()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(model, path, metadata={"seed": 42})
            restored = Struct2SLMLP()
            metadata = load_checkpoint(restored, path)
            self.assertEqual(metadata["seed"], 42)
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
