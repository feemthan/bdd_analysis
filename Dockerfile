FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    wget nano\
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

ADD ./pyproject.toml /app
RUN uv sync

COPY ./src /app/src
COPY ./*.py /app
ADD ./.env /app

RUN mkdir -p /root/.cache/torch/hub/checkpoints/
COPY ./fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

RUN mkdir -p /mlruns /artifacts

CMD ["uv", "run", "run_net.py"]