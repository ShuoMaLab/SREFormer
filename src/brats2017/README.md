# BraTS2017 Code Snapshot

This directory contains the current BraTS2017 implementation snapshot for SREFormer.

The code is being cleaned for the official release. The current snapshot keeps the main components available for inspection:

- `architectures/`: SegFormer3D/SREFormer-related architecture code.
- `dataloaders/`: BraTS2017 dataset loader.
- `augmentations/`: training augmentation utilities.
- `losses/`: Dice loss and segmentation losses.
- `optimizers/`: optimizer and scheduler helpers.
- `train_scripts/`: distributed training utilities.
- `metrics/`: segmentation metrics and profiling helpers.
- `experiments/exp_agent_stage34_register_bs2_dim256/`: BraTS2017 experiment configuration.

Large datasets and checkpoints are not included in Git.
