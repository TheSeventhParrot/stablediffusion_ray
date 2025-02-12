# Use NVIDIA CUDA 12.6 base image
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV MODEL_NAME="stabilityai/stable-diffusion-2"
ENV INSTANCE_DIR="/app/monster_training_images"
ENV OUTPUT_DIR="/app/monster_lora_model"
ENV VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
ENV CLASS_DIR="/app/monster_output_images"
ENV HF_TOKEN="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"
ENV CONFIG_YAML="/app/config.yaml"
ENV DB_SCIPT="/app/diffusers/examples/dreambooth/train_dreambooth.py"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install ray[default]

COPY monster_training_images/ ./monster_training_images/
COPY monster_output_images/ ./monster_output_images/
COPY monster_lora_model/ ./monster_lora_model/
COPY diffusers/ ./diffusers/
COPY config.yaml ./config.yaml
COPY distributed_train.py ./distributed_train.py
