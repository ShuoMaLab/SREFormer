#!/usr/bin/env bash
set -e

python test.py --dataset synapse --data_dir data/synapse --checkpoint checkpoints/synapse_best.pth
