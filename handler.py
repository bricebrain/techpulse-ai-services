import base64
import io
import os
import subprocess
import tempfile
import time
import uuid
from typing import Any

import runpod

PIPELINES: dict[str, Any] = {}
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
        raise ValueError(f"unsupported task: {task}")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "task": task}


runpod.serverless.start({"handler": handler})
