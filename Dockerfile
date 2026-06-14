FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    python3 \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
  && python3 -m pip install -r requirements.txt

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
