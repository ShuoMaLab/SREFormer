#!/usr/bin/env bash
set -e

python -m accelerate.commands.launch --config_file ./gpu_accelerate.yaml ./run_experiment.py --config ./experiments/brats_2017/exp_agent_stage34_register_bs2/config.yaml
