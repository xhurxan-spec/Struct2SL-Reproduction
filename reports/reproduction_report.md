# Struct2SL-Reproduction: Corrected Reproduction Report

## Scope

This repository contains an independent corrected reproduction of the
Struct2SL synthetic-lethality prediction workflow. It is not the original
authors' repository and does not claim to reproduce every behavior of the
released implementation exactly.

## Data and representations

The experiment uses locally available processed datasets and feature representations for the Struct2SL reproduction. The exact dataset provenance and version have not been independently verified.
Structural, sequence, and PPI representations have 128,
1024, and 128 dimensions per gene, respectively. The pair representation is
ordered as structure, sequence, PPI for gene 1 followed by the same three
representations for gene 2, giving 2,560 features.

The historical audit report records 17,180 shared genes. It records 23,749
usable positive pairs and 2,899 negative rows, of which 2,745 have complete
feature coverage; 154 are excluded by the feature filter.
Raw biological data are intentionally excluded from this repository.

The JSON report uses an earlier audit schema and does not contain the
pair-relationship or identifier-dtype fields emitted by the current utility.
The source data required for a fresh audit are not included.

## Corrected protocol

The implementation uses an MLP with hidden sizes 256, 128, and 64, ReLU and
0.5 dropout after each hidden layer, and a one-unit logit output. Training uses
`BCEWithLogitsLoss`, Adam with learning rate 0.004 and weight decay 0.001,
gradient clipping at 1.0, a ReduceLROnPlateau scheduler, batch size 64, and
early stopping after 15 validation-loss non-improvements.

Using split-generation seed 42, a stratified 20% test split is created once
before the five seeded runs (42--46), followed by a stratified validation split
from the remaining data. The scaler is fitted only on the training partition.
The same train, validation, and test indices are reused for every run, while
model initialization and training shuffling vary by run seed. Checkpoints are
selected using validation loss; the test set is not used for checkpoint
selection. The runs therefore do not represent independent data partitions.

## Recorded results

The values below are recomputed from `results/run_1.json` through
`results/run_5.json` and match `results/summary.json`. Standard deviations are
population standard deviations across the five runs, not uncertainty across
independent test partitions.

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Accuracy | 94.67% | 0.21 percentage points |
| ROC-AUC | 95.37% | 0.16 percentage points |
| Average precision | 99.35% | 0.02 percentage points |
| F1-score | 97.07% | 0.11 percentage points |
| Precision | 95.86% | 0.41 percentage points |
| Recall | 98.30% | 0.23 percentage points |
| Specificity | 63.24% | 3.90 percentage points |

The low specificity relative to recall and the class imbalance should be
considered when interpreting these results. High average precision does not by
itself establish balanced classification performance.

## Differences from released Struct2SL code

The corrected track intentionally differs from the released implementation in
several ways: it returns logits and uses `BCEWithLogitsLoss` correctly, fits
scaling on training data only, selects checkpoints using validation loss,
controls repeated-run seeds, and does not generate additional random negative
pairs. These changes address specific methodological issues in the released
implementation, but mean that the results must be labelled a corrected
reproduction rather than an exact reproduction.

## Reproducibility limitations

The processed biological data and trained checkpoints are not distributed in
this repository. The fitted scaler is also not saved as a standalone inference
artifact. Exact independent regeneration therefore requires authorized access
to the data and execution of the documented pipeline. The reported repeated
runs share the same test partition and should not be interpreted as five
independent test-set evaluations.
