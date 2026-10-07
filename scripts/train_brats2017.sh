#!/usr/bin/env bash
set -e

cd brats2017
accelerate launch --config_file experiments/exp_agent_stage34_register_bs2_dim256/gpu_accelerate.yaml \
  experiments/exp_agent_stage34_register_bs2_dim256/run_experiment.py
