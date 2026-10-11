#!/usr/bin/env bash
set -e

python test.py --dataset acdc --data_dir data/ACDC --checkpoint checkpoints/acdc_best.pth
