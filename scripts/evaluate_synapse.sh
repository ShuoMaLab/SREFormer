#!/usr/bin/env bash
set -e

python test.py --config configs/synapse.yaml --checkpoint checkpoints/synapse_best.pth
