# Jetson image based on NVIDIA L4T.
# Match L4T_VERSION to the host (`cat /etc/nv_tegra_release`). r36.2.0 is the
# JetPack 6.0 L4T base tag. A newer host JetPack needs the matching tag.
#
# TensorRT engines are built on the device. See docs/jetson.md before installing
# PyTorch or Ultralytics inside this image.

ARG L4T_VERSION=r36.2.0
FROM nvcr.io/nvidia/l4t-base:${L4T_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        python3-libnvinfer \
        libnvinfer-bin \
        libopenblas-dev \
        libjpeg-dev \
        zlib1g-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/edge-yolo

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir .

WORKDIR /work
ENTRYPOINT ["python3", "-m", "edge_yolo"]
