# Reproducible environment for Membership Inference Attacks research
# Base: official PyTorch image (CPU build so it runs on any machine;
# switch the tag to a CUDA build if a GPU is available).
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

LABEL maintainer="Independent Study - AI Security & Data Privacy"
LABEL description="PyTorch + ML Privacy Meter environment for membership inference experiments"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# Copy project files
COPY scripts /workspace/scripts
COPY notebooks /workspace/notebooks

EXPOSE 8888

# Default command: start a Jupyter server on 0.0.0.0:8888 with no token.
# For shared machines, set a token via the JUPYTER_TOKEN env var in
# docker-compose.yml instead of leaving it empty.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password="]
