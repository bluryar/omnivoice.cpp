# OmniVoice Server

A long-running HTTP server that loads the OmniVoice model once into memory and serves speech synthesis requests via an OpenAI-compatible REST API.

Unlike the CLI tool (which loads the model, generates audio, and exits), the server keeps the model resident in RAM/VRAM across requests — eliminating model load time on every call.

## Quick Start

```bash
# Build
cmake -B build
cmake --build build -j$(nproc)

# Run (CPU)
./build/omnivoice-server --model path/to/model.gguf

# Run (CUDA)
./build/omnivoice-server --model path/to/model.gguf --device cuda:0
```

The server listens on `0.0.0.0:8000` by default.

## Command-Line Flags

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | required | Path to the GGUF model file |
| `--device DEVICE` | cpu | Compute device: `cpu` or `cuda[:N]` (e.g. `cuda:0`) |
| `--backend BACKEND` | cpu | Compute backend: `cpu` or `cuda` (aliases: `rocm`, `hip` → `cuda`) |
| `--host HOST` | 0.0.0.0 | Address to bind to |
| `--port PORT` | 8000 | Port to listen on |
| `--threads N` | 4 | Number of CPU threads |
| `--voices-dir DIR` | none | Directory with voice presets (see [Voice Presets](#voice-presets)) |
| `--help` / `-h` | | Print usage and exit |

## Endpoints

### `POST /v1/audio/speech`

Generate audio from text. Returns a binary audio file (WAV, PCM, or MP3).

This is the primary endpoint, compatible with the OpenAI TTS API format.

#### OpenAI-Compatible Parameters

These follow the [OpenAI Create Speech API](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create/) convention:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `input` | string | yes | | Text to synthesize (max 4096 chars) |
| `voice` | string | no | | Voice specification (see [Voice Resolution](#voice-resolution)) |
| `instructions` | string | no | | Style/delivery instructions for voice design |
| `response_format` | string | no | `"wav"` | Output format: `wav`, `pcm`, or `mp3` |
| `speed` | float | no | `1.0` | Speech speed (0.25 – 4.0) |
| `model` | string | no | | Ignored (always uses the loaded model) |

#### OmniVoice Extension Parameters

These parameters give direct access to OmniVoice's generation controls:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | string | auto | Language code (e.g. `"ru"`, `"en"`, `"zh"`, `"ja"`). See [Supported Languages](languages.md). |
| `ref_audio` | string | | Path to reference WAV, MP3, or FLAC file for voice cloning |
| `ref_text` | string | | Transcription of the reference audio |
| `num_step` | int | 32 | Iterative unmasking steps. Higher = better quality but slower. Use 16 for faster inference. |
| `guidance_scale` | float | 2.0 | Classifier-free guidance scale |
| `t_shift` | float | 0.1 | Time-step shift for the noise schedule |
| `position_temperature` | float | 5.0 | Randomness for mask-position selection. 0 = deterministic, higher = more varied |
| `class_temperature` | float | 0.0 | Randomness for token sampling. 0 = greedy, higher = more varied |
| `layer_penalty_factor` | float | 5.0 | Penalty on deeper codebook layers |
| `duration` | float | | Fixed output duration in seconds (overrides `speed`) |
| `denoise` | bool | true | Prepend denoise token for cleaner speech |
| `preprocess_prompt` | bool | true | Preprocess reference audio (remove silences, add punctuation) |
| `postprocess_output` | bool | true | Post-process output (remove trailing silence) |
| `audio_chunk_duration` | float | 15.0 | Target chunk duration in seconds for long-form generation |
| `audio_chunk_threshold` | float | 30.0 | Duration threshold above which chunking activates |
| `seed` | int | 0 | Random seed for reproducible generation |

#### Voice Resolution

The `voice` parameter supports three modes:

**1. Voice Design (descriptive attributes)**

Pass a comma-separated string of speaker attributes. The model generates a matching voice on the fly — no reference audio needed.

```json
{
  "voice": "female, young adult, high pitch, british accent"
}
```

Supported attributes:

| Category | Values |
|---|---|
| Gender | `male`, `female` |
| Age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| Pitch | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| Style | `whisper` |
| English Accent | `american accent`, `british accent`, `australian accent`, etc. |
| Chinese Dialect | `东北话`, `四川话`, `河南话`, etc. |

See [Voice Design](voice-design.md) for the full list.

**2. Voice Preset (named voice)**

If `--voices-dir` is configured, pass a preset name. The server loads the matching reference audio and text.

```json
{
  "voice": "alice"
}
```

**3. Voice Cloning (explicit reference)**

Use `ref_audio` and `ref_text` directly for voice cloning:

```json
{
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "Transcription of the reference audio.",
  "input": "This text will be spoken in the cloned voice."
}
```

If neither `voice`, `instructions`, `ref_audio`, nor `instruct` are provided, the server defaults to `"male, British accent"`.

#### Response Format

The response is a binary audio file with the appropriate `Content-Type` header:

| Format | Content-Type | Description |
|---|---|---|
| `wav` | `audio/wav` | 16-bit PCM WAV (default) |
| `pcm` | `audio/pcm` | Raw 24kHz 16-bit signed little-endian samples (no header) |
| `mp3` | `audio/mpeg` | MP3 encoded at 64 kbps CBR |

> **Note:** Opus, AAC, and FLAC are listed in the OpenAI API spec but are not currently supported by the server. WAV, PCM, and MP3 are produced locally without shelling out to `ffmpeg`.

#### Error Responses

Errors are returned as JSON:

```json
{
  "error": {
    "message": "'input' is required and must be a non-empty string",
    "type": "invalid_request_error"
  }
}
```

| HTTP Status | Meaning |
|---|---|
| 400 | Invalid request (missing `input`, bad JSON, etc.) |
| 500 | Generation failure (model error, etc.) |

### `GET /v1/audio/voices`

List available voice presets from the `--voices-dir` directory.

```bash
curl http://localhost:8000/v1/audio/voices
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "object": "voice",
      "has_ref_text": true,
      "ref_text": "Hello, my name is Alice."
    },
    {
      "id": "bob",
      "object": "voice",
      "has_ref_text": true,
      "ref_text": "Hi, I'm Bob."
    }
  ]
}
```

### `GET /v1/models`

List available models.

```bash
curl http://localhost:8000/v1/models
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "omnivoice",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok"}
```

## Voice Presets

Voice presets are pairs of files in the `--voices-dir` directory:

```
voices/
  alice.wav       # Reference audio (WAV, MP3, or FLAC)
  alice.txt       # Transcription (single line of text)
  bob.wav
  bob.txt
```

To use a preset, pass its name (without extension) as the `voice` parameter:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "alice"}' \
  --output output.wav
```

## Examples

### Basic synthesis with voice design

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello! Welcome to OmniVoice.",
    "voice": "female, young adult, high pitch"
  }' \
  --output hello.wav
```

### With instructions (like gpt-4o-mini-tts)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The weather is lovely today.",
    "voice": "male",
    "instructions": "whisper, british accent, slow pace"
  }' \
  --output weather.wav
```

### Russian speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Привет! Как у тебя дела?",
    "voice": "female, young adult",
    "language": "ru"
  }' \
  --output russian.wav
```

### Voice cloning

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a completely new sentence.",
    "ref_audio": "/data/voices/reference.wav",
    "ref_text": "This is the transcription of the reference audio."
  }' \
  --output cloned.wav
```

### Fast generation (fewer steps)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Quick synthesis for real-time use.",
    "voice": "male",
    "num_step": 16
  }' \
  --output fast.wav
```

### Fixed duration

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "A fixed-length prompt.",
    "voice": "female",
    "duration": 5.0
  }' \
  --output fixed.wav
```

### Reproducible output with seed

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Same input, same output.",
    "voice": "male",
    "seed": 42,
    "position_temperature": 0
  }' \
  --output reproducible.wav
```

### All OmniVoice parameters

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Full parameter example.",
    "voice": "female, young adult",
    "language": "en",
    "response_format": "wav",
    "speed": 1.0,
    "num_step": 32,
    "guidance_scale": 2.0,
    "t_shift": 0.1,
    "position_temperature": 5.0,
    "class_temperature": 0.0,
    "layer_penalty_factor": 5.0,
    "denoise": true,
    "preprocess_prompt": true,
    "postprocess_output": true,
    "audio_chunk_duration": 15.0,
    "audio_chunk_threshold": 30.0,
    "seed": 12345
  }' \
  --output full.wav
```

### Raw PCM output (no WAV header)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Raw PCM output.",
    "voice": "male",
    "response_format": "pcm"
  }' \
  --output raw.pcm
```

## Architecture

```
┌─────────────────────────────────────────┐
│          omnivoice-server               │
│                                         │
│  ┌─────────┐    ┌──────────────────┐   │
│  │ httplib  │───▶│  JSON Parser    │   │
│  │ (HTTP)   │    └───────┬──────────┘   │
│  └─────────┘             │              │
│                          ▼              │
│                 ┌────────────────┐      │
│                 │ OmniVoice      │      │
│                 │ Runtime        │      │
│                 │ (model in RAM) │      │
│                 └───────┬────────┘      │
│                         │               │
│                         ▼               │
│                 ┌────────────────┐      │
│                 │ WAV / PCM      │      │
│                 │ Encoder        │      │
│                 └────────────────┘      │
└─────────────────────────────────────────┘
```

- **Model loaded once** at startup in `OmniVoiceRuntime` constructor
- **Requests serialized** via mutex (GPU inference is not thread-safe)
- **Zero external dependencies** beyond the CLI (uses inline JSON parser, no nlohmann)
- **cpp-httplib** ([github.com/yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib)) — header-only HTTP library

## Concurrency

The server serializes `generate()` calls with a mutex. This is intentional: the ggml compute graph reuses pre-allocated buffers, and concurrent GPU inference on the same context is not safe.

If you need parallel request handling (e.g. multiple GPUs), run multiple server instances on different ports with `--device cuda:0`, `--device cuda:1`, etc., behind a load balancer or round-robin proxy.

## Compatibility with OpenAI Client Libraries

The server implements the same `POST /v1/audio/speech` contract, so OpenAI client libraries can be pointed at it by changing the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.audio.speech.create(
    model="omnivoice",
    voice="female, young adult",
    input="Hello from the OpenAI Python SDK!",
)

response.stream_to_file("output.mp3")
```

> **Note:** The OpenAI SDK defaults to MP3 output, which is supported by this server via the vendored Shine encoder.
