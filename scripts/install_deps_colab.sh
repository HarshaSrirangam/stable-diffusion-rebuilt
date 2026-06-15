#!/usr/bin/env bash
set -e

pip install -e . --no-deps -q
pip install transformers safetensors tqdm accelerate -q