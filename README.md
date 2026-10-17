# Struct2SL-Reproduction

Independent corrected reproduction of the Struct2SL structure-enhanced
synthetic-lethality prediction workflow.

## Purpose

Synthetic lethality occurs when disrupting two genes together causes a greater loss of cell viability than disrupting either gene individually. Predicting synthetic-lethal gene pairs may help prioritize candidates for further investigation in cancer research.

This repository contains an independent corrected reproduction of Struct2SL, a
model that combines protein structure, sequence, and protein-protein
interaction (PPI) features for gene-pair classification. It is not the original
authors' repository.

## Relationship to the original method

This project uses the original Struct2SL paper and released code as references. The implementation includes corrections to several aspects of the original code, as described below.

The corrected implementation:

- returns raw logits and uses `BCEWithLogitsLoss`;
- fits the feature scaler on the training partition only;
- selects checkpoints using validation loss;
- controls five repeated-run seeds, 42 through 46;
- uses the supplied positive and known-negative tables without generating
  additional random negative pairs.

These changes mean that the results represent a corrected reproduction, not an exact replication of the released implementation.

Original implementation:
<https://github.com/hyr-hit/Struct2SL>

## Representations and data

Each gene has:

- a 128-dimensional structural feature vector;
- a 1,024-dimensional sequence feature vector;
- a 128-dimensional PPI feature vector.

The pair representation concatenates the structural, sequence, and PPI vectors
for gene 1, followed by the corresponding vectors for gene 2, producing 2,560
input features.

The recorded historical audit report lists 17,180 genes shared across all three
feature sources. It records the following usable pair-table counts:

- 23,749 positive pairs;
- 2,745 known-negative pairs from an original 2,899-row negative table;
- approximately 89.6% positive examples among usable pairs.

The processed datasets and feature archives are not included in this repository. Users must obtain the required files separately and verify the applicable access and redistribution terms before using or sharing them.

Expected external data layout:

```text
DATA_ROOT/
├── struct_features/*.txt
├── sequence_features.npz
├── ppi_features.npz
├── Human_SL_filtered.csv
└── Human_nonSL.csv
```

The files with `.npz` extensions in this dataset are pickle-serialized feature
dictionaries. The loader uses the format supplied by the project data.

## Model and training

The model is an MLP with dimensions:

```text
2560 -> 256 -> 128 -> 64 -> 1
```

ReLU activations and 0.5 dropout are applied after each hidden layer. The
output layer returns a raw logit.

Training uses:

- `BCEWithLogitsLoss`;
- Adam with learning rate `0.004`;
- weight decay `0.001`;
- batch size `64`;
- gradient clipping at `1.0`;
- `ReduceLROnPlateau`;
- validation-loss early stopping with patience `15`.

The `fit_scaler_on_training_data_only` and `save_best_checkpoint` configuration
fields record these protocol choices; the current training script enforces them
directly rather than treating them as runtime switches.

Five runs use seeds 42 through 46.

Using split-generation seed `42`, a stratified 20% test split is created once
before the five-run experiment. A validation split is taken from the remaining
data, and all runs reuse the same train, validation, and test partitions. The
five runs use seeds `42` through `46` for run-specific model initialization and
training shuffling. The reported standard deviations therefore describe
variation across runs on one shared test set, not variation across independent
test sets.

## Installation and checks

The `requirements.txt` file lists the runtime dependencies but does not pin their versions. The software versions recorded during the experiment are listed in `environment/environment.json`.

Create a compatible Python environment and install the listed packages. Install PyTorch using the command appropriate for your operating system and hardware, following the [official installation guide](https://pytorch.org/get-started/locally/).

From the repository root, run the CPU-only smoke tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Validate the external data without training:

```bash
python -m src.train \
  --check-only \
  --data-root /path/to/DATA_ROOT
```

With `--data-root`, the following files are resolved relative to the supplied
directory:

```text
DATA_ROOT/struct_features
DATA_ROOT/sequence_features.npz
DATA_ROOT/ppi_features.npz
DATA_ROOT/Human_SL_filtered.csv
DATA_ROOT/Human_nonSL.csv
```

Without `--data-root`, the historical surrounding-directory layout is used,
with paths resolved relative to the parent of the repository.

Run the data audit without replacing the recorded audit report:

```bash
python data_audit/audit_data.py \
  --data-root /path/to/DATA_ROOT \
  --output /tmp/struct2sl_data_audit.json
```

Existing audit output files are protected from replacement. Pass `--overwrite`
only when deliberately replacing a specified output file.

The smoke tests and audit command can be run without starting model training.

Run the complete five-run training and evaluation experiment with a new local
output directory:

```bash
python -m src.train \
  --data-root /path/to/DATA_ROOT \
  --output-dir local_runs/reproduction_01
```

The selected output directory receives `checkpoints/`, `results/`, and `logs/`
subdirectories. Existing checkpoint or result files are never overwritten. The
documented `local_runs/` directory is ignored by Git so that regenerated
checkpoints and logs remain local; the verified JSON files already present in
the repository's `results/` directory are the recorded reference results.

## Recorded five-run results

The values below are recomputed from `results/run_1.json` through
`results/run_5.json` and match `results/summary.json`. Standard deviations are
population standard deviations across the five runs.

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Accuracy | 94.67% | 0.21 percentage points |
| ROC-AUC | 95.37% | 0.16 percentage points |
| Average precision | 99.35% | 0.02 percentage points |
| F1-score | 97.07% | 0.11 percentage points |
| Precision | 95.86% | 0.41 percentage points |
| Recall | 98.30% | 0.23 percentage points |
| Specificity | 63.24% | 3.90 percentage points |

Average precision is calculated using scikit-learn’s `average_precision_score`. The results should be interpreted in the context of class imbalance and the model’s relatively low specificity.

## Limitations

- The processed data and feature archives are not included.
- Trained checkpoints and logs are not included by default.
- The fitted `StandardScaler` is not saved as a standalone inference artifact.
- Reproducing the experiment requires access to the same processed datasets and execution of the documented training pipeline.
- The five repeated runs share one test partition.
- The corrected implementation differs from the released Struct2SL code and
  should be reported as a corrected reproduction.

## Repository layout

```text
configs/       Experiment configuration
data_audit/    Data validation code and recorded audit
environment/   Recorded environment information
reports/       Detailed reproduction report
results/       Lightweight run metrics
src/           Data pipeline, model, and training code
tests/         CPU-only smoke tests
```

Raw datasets, feature archives, checkpoints, logs, Python caches, and private server information are intended to be excluded from the public repository.

## Citation

When using this reproduction, cite the original Struct2SL paper and acknowledge the released implementation. This repository provides a corrected reproduction of the method, not the original authors’ code.

> Huang Y, Yuan R, Li Y, Xing Z, Li J. Struct2SL: Synthetic lethality
> prediction based on AlphaFold2 structure information and Multilayer
> Perceptron. *Computational and Structural Biotechnology Journal*. 2025;
> 27:1570-1577. doi:10.1016/j.csbj.2025.04.012.
