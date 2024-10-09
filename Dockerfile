FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install Python, git, and other system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential

# Install numpy separately to avoid missing module error
RUN pip3 install numpy

# Upgrade pip, setuptools, and wheel to avoid compatibility issues
RUN pip3 install --upgrade pip setuptools wheel packaging

# Install torch separately to ensure it is available before flash_attn
RUN pip3 install torch

# Install remaining Python dependencies
RUN pip3 install \
    transformers \
    taoverse \
    flash_attn \
    accelerate \
    bittensor

# Copy the ML evaluator script and necessary modules
COPY run_eval.py /app/run_eval.py
COPY constants/ /app/constants/
COPY pretrain/ /app/pretrain/
COPY competitions/ /app/competitions/

# Set working directory
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["python3", "run_eval.py"]

