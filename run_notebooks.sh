#!/bin/bash

# Check if a path was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <relative-path-to-notebooks>"
    exit 1
fi

# Get the relative path to the notebook directory
NOTEBOOKS_DIR="$1"

# Get the absolute path to the notebook directory
ABSOLUTE_NOTEBOOKS_DIR=$(realpath "$NOTEBOOKS_DIR")

# Run the Docker container
docker run --gpus all -it --rm \
    -v "$ABSOLUTE_NOTEBOOKS_DIR:/tf/notebooks" \
    -p 8888:8888 \
    tensorflow-jupyter-gpu

