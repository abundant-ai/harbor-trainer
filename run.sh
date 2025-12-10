#!/bin/bash

# Increase file descriptor limit to prevent "Too many open files" errors
ulimit -n 65536

python -m src.train \
  model_name=deepseek-ai/DeepSeek-V3.1  \
  tasks_dir=./datasets/terminal-bench-2 \
  learning_rate=5e-4 \
  batch_size=1 \
  group_size=8 \
  n_parallel_envs=8 \
  max_tokens=1024 \
  max_turns=50 \
  temperature=0.7 \
  context_limit=32000 \
  proactive_summarization_threshold=2000 \
  enable_summarize=true \
  n_epochs=1 \
  loss_fn=ppo \
  environment_type=docker \
  wandb_project=train-ds \
  wandb_name=ds-run
