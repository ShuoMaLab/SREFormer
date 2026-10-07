# Dataset Preparation

This directory will contain dataset loaders and preprocessing utilities for:

- BraTS2017
- ACDC
- Synapse

The repository does not redistribute medical image datasets. Please download
each dataset from its official source and organize it according to the layout
shown in the root README.

## Expected Layout

```text
data/<DatasetName>/
├── images/
├── labels/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Dataset-specific preprocessing scripts will be added under `tools/`.
