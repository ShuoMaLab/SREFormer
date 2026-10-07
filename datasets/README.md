# Dataset Preparation

This directory documents the dataset organization used by SREFormer.

| Dataset | Task | Classes | Download Link |
| --- | --- | --- | --- |
| BraTS2017 | Brain tumor segmentation | WT, TC, ET | https://drive.google.com/file/d/1LMrJRpcMjhsAT6tbstgB1GTdZBk5yKU8/view |
| ACDC | Cardiac MRI segmentation | RV, MYO, LV | https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html |
| Synapse | Multi-organ abdominal CT segmentation | Aorta, liver, left kidney, right kidney, gallbladder, pancreas, spleen, stomach | https://www.synapse.org/Synapse:syn3193805/wiki/89480 |

The repository does not redistribute medical image datasets. Please download
each dataset from its official source and organize it according to the layout
shown in the root README.

## BraTS2017 Layout

```text
data/brats2017_seg/
├── brats2017_raw_data/
│   └── train/
│       ├── imageTr/
│       ├── labelsTr/
│       └── imageTs/
└── BraTS2017_Training_Data/
```

Dataset-specific preprocessing scripts will be added under `tools/`.
