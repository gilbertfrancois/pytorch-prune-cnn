#!/bin/bash

# This script is used to start a Docker container for the PyTorch pruning project.
# System requirements:
# - x86_64 architecture
# - Docker installed
# - NVIDIA GPU with CUDA support
# - NVIDIA Container Toolkit installed

# Build the docker image first with:
#    docker build -t pytorch-prune-cnn:latest .

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -it --rm pytorch-prune-cnn:latest bash
