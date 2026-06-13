import base64
import io
import os
import subprocess
import tempfile
import time
import urllib.request
import uuid
from typing import Any

import runpod

PIPELINES: dict[str, Any] = {}
WHISPER_MODELS: dict[str, Any] = {}
DEFAULT_VOICE_MAP = {
    "host": "ff_siwis",
    "analyst": "ff_siwis",
}


def get_pipeline(lang_code: str):
    if lang_code not in PIPELINES:
        from kokoro import KPipeline

        PIPELINES[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return PIPELINES[lang_code]


def synthesize_kokoro(text: str, voice: str, lang_code: str, speed: float):
    import numpy as np

    pipeline = get_pipeline(lang_code)
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio)
    if not chunks:
        raise ValueError("kokoro returned no audio")
    return np.concatenate(chunks)


def encode_audio(audio, audio_format: str) -> tuple[bytes, str]:
    import soundfile as sf

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, 24000, format="WAV")
    wav_bytes = wav_buffer.getvalue()

    if audio_format == "wav":
        return wav_bytes, "audio/wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "input.wav")
        mp3_path = os.path.join(tmpdir, "output.mp3")
        with open(wav_path, "wb") as file:
            file.write(wav_bytes)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                mp3_path,
            ],
            check=True,
        )
        with open(mp3_path, "rb") as file:
            return file.read(), "audio/mpeg"


def upload_to_r2(audio_bytes: bytes, key: str, content_type: str) -> str | None:
    import boto3

    bucket = os.getenv("R2_BUCKET")
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    public_base_url = os.getenv("R2_PUBLIC_BASE_URL")
    endpoint_url = os.getenv("R2_ENDPOINT_URL")

    if not bucket or not access_key_id or not secret_access_key:
        return None
    if not endpoint_url:
        if not account_id:
            return None
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    )
    client.put_object(Bucket=bucket, Key=key, Body=audio_bytes, ContentType=content_type)
    if public_base_url:
        return f"{public_base_url.rstrip('/')}/{key}"
    return None


def get_whisper_model(model_name: str, device: str, compute_type: str):
    cache_key = f"{model_name}|{device}|{compute_type}"
    if cache_key not in WHISPER_MODELS:
        from faster_whisper import WhisperModel

        WHISPER_MODELS[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    return WHISPER_MODELS[cache_key]


def write_audio_payload(payload: dict[str, Any], tmpdir: str) -> str:
    audio_url = str(payload.get("audio_url") or "").strip()
    audio_base64 = str(payload.get("audio_base64") or "").strip()

    audio_path = os.path.join(tmpdir, "input_audio")
    if audio_url:
        request = urllib.request.Request(
            audio_url,
            headers={
                "User-Agent": "TechPulse-AI-Services/1.0 (+https://runpod.io)",
                "Accept": "audio/mpeg,audio/*,*/*",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            audio_bytes = response.read()
        if not audio_bytes:
            raise ValueError("audio_url returned empty body")
        with open(audio_path, "wb") as file:
            file.write(audio_bytes)
        return audio_path

    if audio_base64:
        with open(audio_path, "wb") as file:
            file.write(base64.b64decode(audio_base64))
        return audio_path

    raise ValueError("audio_url or audio_base64 is required")


def handle_whisper_transcribe(payload: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    model_name = str(payload.get("model") or os.getenv("WHISPER_MODEL") or "large-v3-turbo")
    device = str(payload.get("device") or os.getenv("WHISPER_DEVICE") or "cuda")
    compute_type = str(payload.get("compute_type") or os.getenv("WHISPER_COMPUTE_TYPE") or "float16")
    language = payload.get("language") or "en"
    beam_size = int(payload.get("beam_size") or 5)
    vad_filter = bool(payload.get("vad_filter", True))
    initial_prompt = str(payload.get("initial_prompt") or "").strip() or None

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = write_audio_payload(payload, tmpdir)
        model = get_whisper_model(model_name, device=device, compute_type=compute_type)
        segments_iter, info = model.transcribe(
            audio_path,
            language=str(language) if language else None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
        )
        segments = [
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text.strip(),
            }
            for segment in segments_iter
            if segment.text and segment.text.strip()
        ]

    text = " ".join(segment["text"] for segment in segments).strip()
    return {
        "task": "transcribe.whisper",
        "provider": "runpod",
        "model": model_name,
        "language": getattr(info, "language", language),
        "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 4),
        "duration": round(float(getattr(info, "duration", 0) or 0), 3),
        "text": text,
        "segments": segments,
        "segment_count": len(segments),
        "duration_ms": int((time.time() - started_at) * 1000),
    }


def handle_kokoro_tts(payload: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > 7000:
        raise ValueError("text is too long; max 7000 characters")

    speaker = str(payload.get("speaker") or "host")
    voice = str(payload.get("voice") or DEFAULT_VOICE_MAP.get(speaker, "ff_siwis"))
    lang_code = str(payload.get("lang_code") or "f")
    audio_format = str(payload.get("format") or "mp3").lower()
    if audio_format not in {"mp3", "wav"}:
        raise ValueError("format must be mp3 or wav")

    speed = float(payload.get("speed") or 1.0)
    speed = max(0.65, min(1.35, speed))

    audio = synthesize_kokoro(text, voice=voice, lang_code=lang_code, speed=speed)
    audio_bytes, content_type = encode_audio(audio, audio_format)

    upload = bool(payload.get("upload_to_r2", True))
    key = str(payload.get("r2_key") or f"podcasts/runpod/kokoro/{uuid.uuid4().hex}.{audio_format}")
    audio_url = upload_to_r2(audio_bytes, key, content_type) if upload else None
    return_base64 = bool(payload.get("return_base64", not audio_url))

    result: dict[str, Any] = {
        "task": "tts.kokoro",
        "provider": "runpod",
        "model": "hexgrad/Kokoro-82M",
        "voice": voice,
        "speaker": speaker,
        "lang_code": lang_code,
        "format": audio_format,
        "content_type": content_type,
        "byte_length": len(audio_bytes),
        "r2_key": key if audio_url else None,
        "audio_url": audio_url,
        "duration_ms": int((time.time() - started_at) * 1000),
    }
    if return_base64:
        result["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
    return result


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    task = str(payload.get("task") or "tts.kokoro")
    try:
        if task == "tts.kokoro":
            return {"ok": True, "output": handle_kokoro_tts(payload)}
        if task == "transcribe.whisper":
            return {"ok": True, "output": handle_whisper_transcribe(payload)}
        raise ValueError(f"unsupported task: {task}")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "task": task}


runpod.serverless.start({"handler": handler})
