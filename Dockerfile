FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
  && python -m pip install -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
