import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline import (
    load_feature_sources,
    load_pair_table,
    build_labeled_dataset,
)
from src.model import Struct2SLMLP, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def resolve_data_path(data_root, configured_path):
    """Resolve a configured data path with optional external data storage.

    With no override, the historical layout is preserved: paths in the
    configuration are resolved relative to the parent project directory.
    When ``data_root`` is supplied, paths beginning with ``data/`` are
    resolved relative to that directory instead.
    """
    configured = Path(configured_path)
    if data_root is None:
        return PROJECT_ROOT / configured

    data_root = Path(data_root).expanduser().resolve()
    if configured.parts and configured.parts[0] == "data":
        configured = Path(*configured.parts[1:])
    return data_root / configured


def configure_logging(output_dir=ROOT):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "training.log"),
        ],
        force=True,
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Deterministic algorithms do not guarantee bit-for-bit results across
    # different hardware or PyTorch versions.
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_output_paths_available(run_count, output_dir=ROOT):
    """Refuse to overwrite outputs from an earlier experiment."""
    output_dir = Path(output_dir)
    output_paths = [output_dir / "results" / "summary.json"]
    for run in range(run_count):
        output_paths.extend([
            output_dir / "checkpoints" / f"run_{run + 1}.pt",
            output_dir / "results" / f"run_{run + 1}.json",
        ])

    existing = [path for path in output_paths if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing experiment outputs: " + paths
        )


def load_experiment_data(config, data_root=None):
    data = config["data"]

    features = load_feature_sources(
        resolve_data_path(data_root, data["structure_dir"]),
        resolve_data_path(data_root, data["sequence_file"]),
        resolve_data_path(data_root, data["ppi_file"]),
    )

    positives = load_pair_table(
        resolve_data_path(data_root, data["positive_pairs_file"])
    )
    negatives = load_pair_table(
        resolve_data_path(data_root, data["negative_pairs_file"])
    )

    X, y, report = build_labeled_dataset(
        features,
        positives,
        negatives,
        require_both_genes=True,
    )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    if X.ndim != 2 or X.shape[1] != data["input_size"]:
        raise ValueError(
            f"Expected feature matrix (*, {data['input_size']}), "
            f"received {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError("Feature and label counts do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values.")

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"Expected binary labels 0 and 1; found {np.unique(y)}"
        )

    logging.info("Data pipeline report: %s", report)
    logging.info("Feature matrix: %s", X.shape)
    logging.info(
        "Class counts: %s",
        {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    )

    return X, y


def make_splits(y, seed, test_size, validation_size):
    indices = np.arange(len(y))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=validation_size,
        random_state=seed,
        stratify=y[train_val_idx],
    )

    return train_idx, val_idx, test_idx


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


def predict_probabilities(model, loader, device):
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probabilities = torch.sigmoid(logits).squeeze(1)

            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.asarray(all_labels), np.asarray(all_probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiment.json"),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load and validate data without training.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Directory containing the biological data files. Paths in the "
            "configuration are resolved relative to this directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT),
        help=(
            "Directory under which results/, checkpoints/, and logs/ are "
            "written. Existing experiment outputs are never overwritten."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    training = config["training"]
    evaluation = config["evaluation"]
    run_count = config["reproducibility"]["num_repeated_runs"]

    if not args.check_only:
        ensure_output_paths_available(run_count, output_dir)

    configure_logging(output_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logging.info("Experiment: %s", config["experiment_name"])
    logging.info("Device: %s", device)

    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    X, y = load_experiment_data(config, data_root=args.data_root)

    train_idx, val_idx, test_idx = make_splits(
        y,
        training["seed"],
        training["test_size"],
        training["validation_size"],
    )

    # Fit preprocessing on training samples only to avoid leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)

    logging.info(
        "Split sizes: train=%d, validation=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    if args.check_only:
        logging.info(
            "CHECK PASSED: data, labels, splits, and scaling are valid."
        )
        return

    model_config = config["model"]
    batch_size = training["batch_size"]
    threshold = evaluation["classification_threshold"]

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y[train_idx].astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y[val_idx].astype(np.float32)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y[test_idx].astype(np.float32)),
    )

    output_dir.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("results").mkdir(parents=True, exist_ok=True)

    all_results = []

    for run in range(run_count):
        seed = training["seed"] + run
        seed_everything(seed)
        started = time.time()

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = Struct2SLMLP(
            input_size=model_config["input_size"]
            if "input_size" in model_config
            else config["data"]["input_size"],
            hidden_sizes=tuple(model_config["hidden_sizes"]),
            output_size=model_config["output_size"],
            dropout=model_config["dropout"],
        ).to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training["scheduler_factor"],
            patience=training["scheduler_patience"],
        )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_counter = 0

        logging.info(
            "========== RUN %d/%d | seed=%d ==========",
            run + 1, run_count, seed,
        )

        for epoch in range(training["max_epochs"]):
            model.train()
            total_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training["gradient_clip_norm"],
                )

                optimizer.step()
                total_loss += loss.item()

            model.eval()
            val_loss_sum = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    logits = model(inputs.to(device))
                    loss = criterion(
                        logits,
                        labels.to(device).unsqueeze(1),
                    )
                    val_loss_sum += loss.item()

            val_loss = val_loss_sum / len(val_loader)
            train_loss = total_loss / len(train_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(
                    "Run %d | epoch %d | train_loss=%.6f | "
                    "val_loss=%.6f | best_val_loss=%.6f",
                    run + 1, epoch + 1, train_loss, val_loss,
                    best_val_loss,
                )

            if patience_counter >= training["early_stopping_patience"]:
                logging.info(
                    "Early stopping run %d at epoch %d; best epoch=%d",
                    run + 1, epoch + 1, best_epoch,
                )
                break

        if best_state is None:
            raise RuntimeError("Training ended without a valid checkpoint.")

        model.load_state_dict(best_state)
        model.to(device)

        checkpoint_path = output_dir / "checkpoints" / f"run_{run + 1}.pt"
        save_checkpoint(
            model,
            str(checkpoint_path),
            metadata={
                "run": run + 1,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "experiment": config["experiment_name"],
            },
        )

        y_test, probabilities = predict_probabilities(
            model, test_loader, device
        )
        metrics = calculate_metrics(y_test, probabilities, threshold)
        metrics.update({
            "run": run + 1,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "elapsed_seconds": round(time.time() - started, 2),
        })
        all_results.append(metrics)

        logging.info("Run %d test metrics: %s", run + 1, metrics)

        with open(
            output_dir / "results" / f"run_{run + 1}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics, f, indent=2)

    metric_names = [
        "accuracy", "roc_auc", "average_precision", "f1",
        "precision", "recall", "specificity",
    ]
    summary = {
        "experiment": config["experiment_name"],
        "evaluation_threshold": threshold,
        "runs": all_results,
        "mean": {
            name: float(np.mean([row[name] for row in all_results]))
            for name in metric_names
        },
        "standard_deviation": {
            name: float(np.std([row[name] for row in all_results]))
            for name in metric_names
        },
        "note": (
            "Repeated runs use the same stratified train/validation/test "
            "partition with different model initialization and shuffle seeds. "
            "The test set is not used for checkpoint selection."
        ),
    }

    with open(
        output_dir / "results" / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    logging.info("FINAL MEAN METRICS: %s", summary["mean"])
    logging.info("FINAL STANDARD DEVIATIONS: %s", summary["standard_deviation"])
    logging.info("Results saved under %s", output_dir / "results")


if __name__ == "__main__":
    main()
