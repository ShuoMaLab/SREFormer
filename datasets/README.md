# Dataset Preparation

This directory documents the dataset organization used by SREFormer.

| Dataset | Task | Classes |
| --- | --- | --- |
| BraTS2017 | Brain tumor segmentation | WT, TC, ET |
| ACDC | Cardiac MRI segmentation | RV, MYO, LV |
| Synapse | Multi-organ abdominal CT segmentation | Aorta, liver, left kidney, right kidney, gallbladder, pancreas, spleen, stomach |

The repository does not redistribute medical image datasets. Please download
each dataset from its official source and organize it according to the layout
shown below.

## Dataset Directory Structure

```text
data/brats2017_seg/
├── brats2017_raw_data/
│   └── train/
│       ├── imageTr/
│       ├── labelsTr/
│       └── imageTs/
└── BraTS2017_Training_Data/
```

```text
data/ACDC/
├── train/
│   ├── case_xxx_sliceED_0.npz
│   ├── case_xxx_sliceED_1.npz
│   ├── case_xxx_sliceES_0.npz
│   └── ...
├── valid/
│   ├── *.npz
│   └── ...
├── test/
│   ├── case_002_volume_ED.npz
│   ├── case_002_volume_ES.npz
│   ├── case_003_volume_ED.npz
│   ├── case_003_volume_ES.npz
│   └── ...
└── lists_ACDC/
    └── dataset split files
```

```text
data/synapse/
├── train/
│   ├── case0005.npz
│   ├── case0006.npz
│   ├── case0007.npz
│   ├── case0009.npz
│   └── ...
└── val/
    ├── case0001.npz
    ├── case0002.npz
    ├── case0003.npz
    ├── case0004.npz
    └── ...
```

Preprocessing scripts are not included in this repository. Please organize the processed datasets according to the layouts above.
