# Use NVIDIA CUDA 12.6 base image
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
ENV INSTANCE_DIR="/app/monster_training_images"
ENV VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
ENV OUTPUT_DIR="/app/monster_lora_model"
ENV CLASS_DIR="/app/monster_output_images"
ENV GCLOUD_BUCKET="stable-diff-tj"

# Install system dependencies and upgrade libstdc++6
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --only-upgrade libstdc++6 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
 
RUN mkdir -p /app/monster_output_images
RUN mkdir -p /app/monster_lora_model

COPY monster_training_images/ ./monster_training_images/
COPY distributed_raytrain.py ./distributed_raytrain.py
COPY __init__.py ./__init__.py
COPY train_dreambooth_lora_sdxl.py ./train_dreambooth_lora_sdxl.py

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install ray[default]
RUN pip3 install ray[train]
RUN pip3 install google-cloud-storage
