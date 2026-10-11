#!/usr/bin/env bash
set -e

python test.py --dataset brats2017 --checkpoint checkpoints/brats2017_best.pth
