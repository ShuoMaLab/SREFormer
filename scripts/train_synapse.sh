#!/usr/bin/env bash
set -e

python train.py --dataset synapse --data_dir data/synapse --out_dir experiments/synapse
