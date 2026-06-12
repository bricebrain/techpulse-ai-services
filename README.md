# TechPulse AI Services

RunPod Serverless worker pour les modeles IA lourds mutualisables.

Phase 1 cible TechPulse uniquement :

- `tts.kokoro` : generation audio podcast via `hexgrad/Kokoro-82M`

Phases suivantes :

- `transcribe.whisper`
- `embed.bge_m3`
- `rerank.bge`

## Pourquoi ce service existe

Le Worker Cloudflare reste le gateway produit. RunPod ne fait que le compute IA lourd :

```text
Cloudflare Worker
-> RunPod Serverless
-> R2 pour les fichiers audio
-> D1/Neon pour les metadonnees
-> App
```

## Creation RunPod

Dans l'onboarding RunPod :

1. Choisir `Serverless`.
2. Choisir `Deploy your own production API`.
3. Choisir `Use your own Repository`.
4. Utiliser ce dossier comme repo ou construire/pousser une image Docker.

## Variables d'environnement RunPod

Pour R2, ajouter dans l'endpoint RunPod :

```bash
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=techpulse-podcasts
R2_PUBLIC_BASE_URL=https://...
```

`R2_PUBLIC_BASE_URL` est optionnel. Si absent, le worker retourne le base64 audio.

## Input RunPod

```json
{
  "input": {
    "task": "tts.kokoro",
    "text": "OpenAI et Nvidia accelerent la bataille de l infrastructure IA.",
    "speaker": "host",
    "voice": "ff_siwis",
    "lang_code": "f",
    "format": "mp3",
    "upload_to_r2": false,
    "return_base64": true
  }
}
```

## Sortie

```json
{
  "ok": true,
  "output": {
    "task": "tts.kokoro",
    "provider": "runpod",
    "model": "hexgrad/Kokoro-82M",
    "format": "mp3",
    "audio_url": null,
    "audio_base64": "...",
    "byte_length": 12345
  }
}
```

## Appel API RunPod

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "tts.kokoro",
      "text": "Test TechPulse.",
      "speaker": "host",
      "format": "mp3",
      "upload_to_r2": false,
      "return_base64": true
    }
  }'
```

## Budget safe

Pour usage personnel :

- Serverless uniquement.
- Max workers : `1`.
- Pas de worker chaud au debut.
- Pas de Pod permanent.
- Tester d'abord avec `upload_to_r2: false` et un texte court.
