#!/bin/bash

# activate trainium venv
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# Environment variables
export NEURON_COMPILE_CACHE_URL="s3://cs217-moe-neuron-cache"
export INPUT_WEIGHTS_PATH="weights/input_weights"
export OUTPUT_WEIGHTS_PATH="weights/output_weights"
export MODEL_WEIGHTS_PATH="weights/compiled_model_weights"
export S3_COMPILED_MODELS_URI=${MODEL_WEIGHTS_PATH}

echo "Checking for compiled Trainium graphs in ${MODEL_WEIGHTS_PATH}..."

# Check if there are any .pt files in the model weights directory
if ! ls ${MODEL_WEIGHTS_PATH}/*.pt 1> /dev/null 2>&1; then
    echo "Compiled graphs not found locally. Downloading from S3..."
    
    # Ensure the local directory exists just in case
    mkdir -p ${MODEL_WEIGHTS_PATH}
    
    # Sync the compiled models from the S3 bucket to the local directory
    aws s3 sync ${S3_COMPILED_MODELS_URI} ${MODEL_WEIGHTS_PATH}/
    
    echo "Download complete!"
else
    echo "Compiled graphs already exist locally. Skipping download."
fi