# TTS Module

![License: MIT](https://img.shields.io/github/license/rookiemann/portable-tts-server) ![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue) ![Python](https://img.shields.io/badge/Python-3.10%20Portable-green) ![Models](https://img.shields.io/badge/TTS%20Models-10-orange) ![Stars](https://img.shields.io/github/stars/rookiemann/portable-tts-server)

**Portable multi-GPU text-to-speech server for Windows — 10 models, one-click install, zero dependencies.**

Send text in, get broadcast-ready audio out. Voice cloning, dialogue, emotion control, multilingual synthesis — across any combination of NVIDIA GPUs. No system Python, no Git, no FFmpeg, no Docker, no admin rights. Everything downloads automatically into one portable folder.

---

## Highlights

- **10 TTS models** — from 82M-parameter Kokoro to 7B-parameter Qwen, covering voice cloning, dialogue, emotion control, and multilingual synthesis
- **Multi-GPU inference** — pin workers to any detected GPU, run the same model on multiple GPUs simultaneously
- **Zero-install portable app** — embedded Python 3.10, portable Git, bundled FFmpeg; copy the folder to any Windows machine
- **Production audio pipeline** — hierarchical text chunking, 7-stage post-processing (de-reverb, highpass, de-ess, tempo, trim, LUFS normalization, peak limiting), multi-format export
- **Whisper verification** — optional transcription check scores synthesized audio against the original text
- **Auto-scaling workers** — workers spawn on first request, fail over to siblings, and get health-checked every 10 seconds
- **Full REST API** — 40+ endpoints with interactive Swagger docs at `/docs`
- **GUI environment manager** — install venvs, download models, and launch the server from a Tkinter interface

---

## Architecture

```
                        Port 8100
                    +---------------+
  Client --------->|   GATEWAY     |
                    | (FastAPI)     |
                    |               |
                    | - Pipeline    |  Text chunking, audio post-processing,
                    | - Job mgmt   |  whisper verification, assembly,
                    | - Load balance|  format conversion, job tracking
                    | - Worker mgmt|
                    +-------+-------+
                            |
            +---------------+---------------+------------------+
            |               |               |                  |
     Port 8101       Port 8102       Port 8103          Port 8104
   +-----------+   +-----------+   +-----------+   +-----------+
   | WORKER    |   | WORKER    |   | WORKER    |   | WORKER    |
   | xtts #1   |   | kokoro #1 |   | kokoro #2 |   | higgs #1  |
   | cuda:0    |   | cuda:0    |   | cuda:1    |   | cpu       |
   +-----------+   +-----------+   +-----------+   +-----------+
   coqui_env        unified_env     unified_env     higgs_env
```

- **Gateway** (port 8100) — orchestrates the full pipeline, delegates inference to workers via HTTP
- **Workers** (ports 8101-8200) — each runs one model on one device as an isolated subprocess
- Each worker injects only its own venv's `site-packages` — no cross-environment dependency conflicts
- Same model can run multiple instances across GPUs for concurrent inference
- Workers auto-spawn on first request if none exist for a model

---

## Quick Start

### Download

**[Download Latest Release (v3.0.2)](https://github.com/rookiemann/portable-tts-server/releases/latest/download/portable-tts-server-v3.0.2.zip)** — Extract anywhere and run `install.bat`.

### First-Time Setup
```batch
install.bat
```
Downloads embedded Python 3.10, portable Git, configures pip, installs gateway dependencies, and launches the GUI. From there, install model environments and download weights.

### Start the API Server
```batch
launcher.bat api
```
Gateway starts on `http://127.0.0.1:8100`. Workers auto-spawn when you send your first request. Interactive API docs at `http://127.0.0.1:8100/docs`.

### Generate Speech
```bash
curl -X POST http://127.0.0.1:8100/api/tts/kokoro \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Kokoro!", "voice": "af_heart"}'
```

### Voice Cloning (Reference Audio)

Models like XTTS, Chatterbox, Fish, and F5-TTS can clone a voice from a reference audio file. Encode it as base64:

```bash
# XTTS voice cloning
REF=$(base64 -w 0 my_voice.wav)
curl -X POST http://127.0.0.1:8100/api/tts/xtts \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hello in my voice!\", \"reference_audio\": \"$REF\"}"

# F5-TTS requires both reference audio AND its transcript
curl -X POST http://127.0.0.1:8100/api/tts/f5 \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"New text to say.\", \"reference_audio\": \"$REF\", \"reference_text\": \"The words spoken in the reference audio.\"}"

# XTTS with a built-in voice (no reference audio needed)
curl -X POST http://127.0.0.1:8100/api/tts/xtts \
  -H "Content-Type: application/json" \
  -d '{"text": "Using a built-in voice.", "voice": "Ana Florence", "mode": "builtin"}'

# Dia multi-speaker dialogue
curl -X POST http://127.0.0.1:8100/api/tts/dia \
  -H "Content-Type: application/json" \
  -d '{"text": "[S1] How are you? [S2] Great, thanks!"}'
```

### Other Commands
```batch
launcher.bat gui                # Launch environment manager GUI
launcher.bat api --port 8200    # Custom port
launcher.bat download --all     # Download all model weights
launcher.bat download --list    # List available models
launcher.bat setup              # Re-run full setup
```

---

## TTS Models

Ten models spanning voice cloning, dialogue, emotion control, and multilingual synthesis:

| Model | Params | Max Chars | Sample Rate | Key Capability |
|-------|--------|-----------|-------------|----------------|
| [**XTTS v2**](https://huggingface.co/coqui/XTTS-v2) | ~500M | 250 | 22050 Hz | Multilingual voice cloning from audio samples |
| [**Bark**](https://huggingface.co/suno/bark) | ~1B | 200 | 24000 Hz | Expressive audio with laughter, music, nonverbals |
| [**Fish Speech 1.5**](https://github.com/fishaudio/fish-speech) | ~500M | 250 | 24000 Hz | Fast neural TTS with V1.5 FireflyGAN codec |
| [**Kokoro**](https://huggingface.co/hexgrad/Kokoro-82M) | 82M | 500 | 24000 Hz | Lightweight and fast, 54 built-in voices |
| [**Dia 1.6B-0626**](https://huggingface.co/nari-labs/Dia-1.6B-0626) | 1.6B | 400 | 44100 Hz | Multi-speaker dialogue with `[S1]`/`[S2]` tags |
| [**Chatterbox**](https://huggingface.co/ResembleAI/chatterbox) | ~500M | 250 | 24000 Hz | Emotion and exaggeration control |
| [**F5-TTS**](https://huggingface.co/SWivid/F5-TTS) | ~300M | 135 | 24000 Hz | Diffusion-based, reference audio + transcript |
| [**Qwen2.5-Omni**](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | 7B | 300 | 24000 Hz | Multimodal LLM, speakers: Chelsie, Ethan |
| [**VibeVoice**](https://huggingface.co/microsoft/VibeVoice-1.5B) | 1.5B | 300 | 24000 Hz | 90-minute generation, multi-speaker via `Speaker N:` |
| [**Higgs Audio**](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) | 3B | 300 | 24000 Hz | Automatic prosody, ChatML format, runs on CPU |

### Voice Libraries

| Endpoint | Voices |
|----------|--------|
| `/api/tts/xtts/voices` | 58 built-in speakers |
| `/api/tts/kokoro/voices` | 54 voices (dynamic `.pt` scan) |
| `/api/tts/bark/voices` | 260+ speaker presets (`.npy` embeddings) |
| `/api/tts/qwen/voices` | Chelsie, Ethan |
| Other models | Reference audio instructions returned |

### Isolated Environments

Each model group runs in its own virtual environment to avoid dependency conflicts:

| Environment | Models | PyTorch | CUDA | transformers |
|-------------|--------|---------|------|-------------|
| `coqui_env` | XTTS, Bark | 2.6.0+cu124 | 12.4 | <5.0 |
| `unified_env` | Kokoro, Fish, Dia | 2.6.0+cu124 | 12.4 | flexible |
| `chatterbox_env` | Chatterbox | 2.6.0+cu124 | 12.4 | 4.46.3 |
| `f5tts_env` | F5-TTS | 2.8.0+cu126 | 12.6 | 5.1.0 |
| `qwen3_env` | Qwen2.5-Omni | 2.6.0+cu124 | 12.4 | 4.57.3 |
| `vibevoice_env` | VibeVoice | 2.5.1+cu121 | 12.1 | 4.51.3 |
| `higgs_env` | Higgs Audio | CPU-only | — | 4.47.0 |

---

## API Reference

Full interactive documentation is available at `http://localhost:8100/docs` when the server is running.

### TTS Inference

All TTS endpoints accept `POST` with JSON body:

```
POST /api/tts/xtts          POST /api/tts/bark         POST /api/tts/fish
POST /api/tts/kokoro         POST /api/tts/dia          POST /api/tts/chatterbox
POST /api/tts/f5             POST /api/tts/qwen         POST /api/tts/vibevoice
POST /api/tts/higgs
```

#### Request Parameters

```jsonc
{
  "text": "Hello, world!",              // Required
  "voice": "af_heart",                  // Model-specific voice ID
  "reference_audio": "<base64 wav>",    // For voice cloning models
  "reference_text": "transcript here",  // Required for F5-TTS and Higgs
  "mode": "cloned",                     // "cloned" (default) or "builtin" for XTTS
  "language": "en",
  "speed": 1.0,
  "temperature": 0.65,
  "repetition_penalty": 2.0,
  "output_format": "wav",               // wav, mp3, ogg, flac, m4a
  "de_reverb": 0.7,                     // 0.0-1.0
  "de_ess": 0.0,                        // 0.0-1.0
  "verify_whisper": false,              // Transcription verification
  "whisper_model": "base",              // tiny/base/small/medium/large
  "tolerance": 80,                      // Whisper similarity threshold (0-100)
  "save_path": "my_project/scene1",     // Optional output directory
  "skip_post_process": false,
  "auto_retry": 3,
  "device": "cuda:0"                    // Target GPU
}
```

#### Response

```jsonc
{
  "status": "completed",
  "job_id": "abc123-...",
  "filename": "kokoro_20260207_143000_final.wav",
  "saved_to": "output/jobs/abc123/kokoro_..._final.wav",
  "sample_rate": 24000,
  "duration_sec": 3.45,
  "format": "wav",
  "audio_base64": "<base64-encoded audio>",  // Omitted when save_path is set
  "chunks": 1,
  "whisper_result": null                     // Populated when verify_whisper=true
}
```

When `save_path` is provided, audio is saved to `projects_output/{save_path}/` and `audio_base64` is omitted from the response.

### Worker Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/workers/spawn` | Spawn worker: `{"model":"kokoro","device":"cuda:0"}` |
| `DELETE` | `/api/workers/{worker_id}` | Kill a specific worker |
| `GET` | `/api/workers` | List all workers with status, device, VRAM |
| `POST` | `/api/models/{model}/scale` | Scale to N instances: `{"count":2,"device":"cuda:0"}` |
| `POST` | `/api/models/{model}/load` | Load model (spawn a worker) |
| `POST` | `/api/models/{model}/unload` | Unload model (kill all its workers) |
| `GET` | `/api/devices` | GPU discovery with VRAM info |

#### Examples

```bash
# Spawn a Kokoro worker on GPU 0
curl -X POST http://127.0.0.1:8100/api/workers/spawn \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "device": "cuda:0"}'

# Run Kokoro on GPU 0 and XTTS on GPU 1 simultaneously
curl -X POST http://127.0.0.1:8100/api/workers/spawn -d '{"model":"xtts","device":"cuda:1"}'

# Scale Kokoro to 2 instances for parallel inference
curl -X POST http://127.0.0.1:8100/api/models/kokoro/scale \
  -H "Content-Type: application/json" \
  -d '{"count": 2, "device": "cuda:0"}'

# List all workers
curl http://127.0.0.1:8100/api/workers

# Kill a specific worker
curl -X DELETE http://127.0.0.1:8100/api/workers/kokoro-1
```

### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/jobs` | List all jobs (running, completed, failed) |
| `GET` | `/api/jobs/{job_id}` | Job status, chunks, progress metadata |
| `POST` | `/api/jobs/{job_id}/recover` | Resume interrupted job from last completed chunk |
| `POST` | `/api/tts/{model}/cancel` | Cancel running job(s) for a model |

### Whisper Verification

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/whisper` | Status, available sizes, loaded instances |
| `POST` | `/api/whisper/{size}/load` | Load model: tiny, base, small, medium, large |
| `POST` | `/api/whisper/{size}/unload` | Unload model and free VRAM |

### Discovery

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with loaded models and worker count |
| `GET` | `/api/models` | List all available models |
| `GET` | `/api/models/status` | Detailed status per model (workers, loaded) |
| `GET` | `/api/tts/{model}/voices` | Available voices for a model |
| `GET` | `/docs` | Interactive Swagger documentation |

---

## Audio Pipeline

### Text Processing
- **Hierarchical chunking** — splits on sentence boundaries first, then clauses, then words
- **Model-aware limits** — each model has a tuned character limit (135–500 chars)
- **Unicode normalization** — smart quotes, whitespace collapse, NFKC

### 7-Stage Post-Processing

Applied per chunk, fully configurable per request:

| Stage | Library | Description |
|-------|---------|-------------|
| 1. De-reverb | [noisereduce](https://github.com/timsainb/noisereduce) | Spectral gating using first 0.2s as noise profile |
| 2. Highpass | [scipy](https://scipy.org/) | 4th-order Butterworth at 80 Hz |
| 3. De-ess | scipy | Multiband Hilbert envelope, 3 kHz+ crossover, 4:1 compression |
| 4. Tempo | [pyrubberband](https://github.com/bmcfee/pyrubberband) | Time-stretch without pitch shift |
| 5. Trim | [pydub](https://github.com/jiaaro/pydub) | Silence detection with front/end protection zones |
| 6. LUFS | [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | Loudness normalization to -23 LUFS |
| 7. Peak limit | numpy | Hard clamp to 0.95 (prevents clipping) |

Each stage degrades gracefully — if a library is unavailable, that stage is skipped with a warning.

### Chunk Assembly & Export
- Configurable silence padding between chunks
- Front/end padding for natural spacing
- Format conversion via [FFmpeg](https://ffmpeg.org/): **WAV**, **MP3**, **OGG**, **FLAC**, **M4A**

### Whisper Verification
- Powered by [OpenAI Whisper](https://github.com/openai/whisper) running as a dedicated worker
- Transcribes generated audio and scores it against the original text
- Configurable similarity threshold (0–100%)
- Five model sizes from tiny (39M) to large (1.5B)

---

## How It Works

### Request Flow

```
1. Client POSTs text to /api/tts/{model}
2. Gateway normalizes and chunks text per model limits
3. Creates a tracked job in output/jobs/
4. For each chunk:
   a. Round-robin picks a ready worker (or auto-spawns one)
   b. POSTs to worker's /infer endpoint via HTTP
   c. Worker returns base64-encoded numpy audio
   d. Gateway applies 7-stage post-processing
   e. Optional Whisper verification
   f. Updates job metadata
5. Assembles chunks with silence padding
6. Converts to requested format via FFmpeg
7. Returns audio (base64 JSON or file path)
```

### Worker Lifecycle
- Workers are Python subprocesses, each running its own [FastAPI](https://fastapi.tiangolo.com/) server
- Each injects only its model's venv `site-packages` into `sys.path`
- Health checks run every 10 seconds; 3 consecutive failures trigger cleanup
- Dead workers are terminated, their ports released, and they're unregistered
- Failed inference automatically retries on a different worker (up to 3 attempts)

### Job Recovery
Jobs track per-chunk progress in `output/jobs/{job_id}/job.json`. If a long synthesis is interrupted, send `text: "##recover##"` with the `save_path` to resume from the last completed chunk.

---

## Configuration

Key settings in `config.py`:

```python
DEFAULT_API_PORT = 8100          # Gateway port

WORKER_PORT_MIN = 8101           # Worker port range
WORKER_PORT_MAX = 8200           # Up to 100 concurrent workers
WORKER_HEALTH_INTERVAL = 10      # Health check every 10 seconds
WORKER_STARTUP_TIMEOUT = 120     # Max wait for model loading (seconds)
WORKER_MAX_HEALTH_FAILURES = 3   # Failures before worker cleanup
WORKER_AUTO_SPAWN = True         # Spawn workers on first request
WORKER_DEFAULT_DEVICE = "cuda:0"
```

### Path Resolution (Portable-First)
1. **Python**: `python_embedded/python.exe` > system Python
2. **Git**: `git_portable/cmd/git.exe` > system Git
3. **FFmpeg**: `ffmpeg/bin/ffmpeg.exe` > `ffmpeg/*/bin/ffmpeg.exe` > system FFmpeg

---

## Directory Structure

```
tts_module/
├── install.bat                  One-click setup (downloads Python, Git, FFmpeg)
├── launcher.bat                 Command router (gui, api, download, setup)
├── requirements.txt             Gateway dependencies
├── config.py                    Central configuration
│
├── tts_api_server.py            Gateway: pipeline, jobs, worker management
├── tts_worker.py                Worker: single-model FastAPI inference server
├── worker_registry.py           Worker tracking: port pool, round-robin balancing
├── worker_manager.py            Worker lifecycle: spawn, kill, scale, health
├── job_manager.py               Job tracking: create, update, cancel, recover
│
├── text_utils.py                Hierarchical text chunking + normalization
├── audio_profiles.py            Per-model audio processing presets
├── audio_processing.py          7-stage post-processing pipeline
├── audio_assembler.py           Chunk assembly + FFmpeg format conversion
│
├── tts_manager.py               Tkinter GUI (environments, models, server)
├── download_all_models.py       HuggingFace model downloader
│
├── install_configs/             8 virtual environment configurations
│   ├── base.py                  Abstract base (pip install, git clone, verify)
│   ├── coqui_tts.py             XTTS, Bark
│   ├── unified_tts.py           Kokoro, Fish Speech, Dia
│   ├── chatterbox.py            Chatterbox
│   ├── f5_tts.py                F5-TTS
│   ├── qwen3_tts.py             Qwen2.5-Omni
│   ├── vibevoice.py             VibeVoice
│   ├── higgs_audio.py           Higgs Audio
│   └── orpheus.py               Orpheus (reserved)
│
├── python_embedded/             [Auto] Embedded Python 3.10.11
├── git_portable/                [Auto] Portable Git 2.47.1
├── ffmpeg/                      Portable FFmpeg
├── tts_models/                  [Auto] HuggingFace model cache
├── venvs/                       [Auto] 8 isolated virtual environments
├── voices/                      [Auto] Voice reference audio files
└── output/                      [Auto] Generated audio + logs
    ├── jobs/{job_id}/           Per-job chunks, metadata, final audio
    └── logs/                    Worker log files
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10 64-bit | Windows 11 |
| **GPU** | NVIDIA with CUDA (any) | 12 GB+ VRAM |
| **RAM** | 16 GB | 32 GB |
| **Disk** | ~2 GB (base install) | ~50 GB (all envs + models) |
| **Internet** | Required for setup | Required for setup |

CUDA versions used: **12.1**, **12.4**, **12.6** (depending on model environment). Higgs Audio runs on CPU.

### VRAM per Model (Approximate)

| Model | VRAM | Notes |
|-------|------|-------|
| Kokoro | ~1 GB | Lightweight, great for low-VRAM GPUs |
| F5-TTS | ~2 GB | |
| Chatterbox | ~3 GB | |
| XTTS v2 | ~3 GB | |
| Fish Speech | ~3 GB | |
| Bark | ~4 GB | |
| Dia | ~6 GB | |
| VibeVoice | ~6 GB | |
| Higgs Audio | CPU-only | Uses ~6 GB system RAM |
| Qwen2.5-Omni | ~14 GB | Requires RTX 3090 or better |

### External Dependency

[**espeak-ng**](https://github.com/espeak-ng/espeak-ng/releases) is required for Kokoro (phoneme synthesis). Automatically downloaded and configured by `install.bat`.

---

## GUI: Environment Manager

The Tkinter GUI (`tts_manager.py`) provides three tabs:

**Environments** — Install, remove, and manage the 8 virtual environments. Status indicators show installed (green), installing (yellow), not installed (gray), or error (red). Each environment has Install, Terminal (opens activated cmd), Remove, and Info buttons.

**Models** — Download model weights from HuggingFace Hub with progress tracking. Supports selective or bulk download.

**API Server** — Start/stop the gateway server, configure the port, and view endpoint documentation.

---

## Model-Specific Notes

### Bark
History prompt chaining is managed by the gateway (not the worker). History resets every 5 chunks to prevent voice drift. 260+ speaker presets via `.npy` embeddings.

### Dia
Uses the updated Dia-1.6B-0626 model which generates proper EOS tokens. The first 2–3 inferences after loading may produce near-silent audio (warmup). The gateway's RMS check rejects degenerate audio (<-40 dB) and retries automatically. Recommended params: `temperature=1.8`, `top_p=0.90`, `cfg_scale=3.0`.

### Fish Speech
Uses the V1.5 codec (FireflyArchitecture + FiniteScalarQuantize), not the S1-mini DAC codec. The LLAMA model generates 8-codebook tokens. Gateway default `repetition_penalty` (2.0) is auto-clamped to 1.1 for Fish.

### Qwen2.5-Omni
Very large model (~14 GB). First load takes ~12 minutes; subsequent loads use cached weights.

### Higgs Audio
Runs on CPU only. Inference is slow (~5 minutes per generation) but produces good quality with automatic prosody.

---

## Adding a New Model

1. Create a venv config in `install_configs/new_model.py`
2. Register it in `install_configs/__init__.py`
3. Add an inference function `_infer_newmodel()` in `tts_worker.py`
4. Add model entry to `load_model()` in `tts_worker.py`
5. Add audio profile in `audio_profiles.py`
6. Add text limit in `audio_profiles.py`
7. Add endpoint in `tts_api_server.py`

---

## References

### TTS Models

| Model | Repository | Paper / Card |
|-------|-----------|-------|
| XTTS v2 | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) | [HF: coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| Bark | [suno-ai/bark](https://github.com/suno-ai/bark) | [HF: suno/bark](https://huggingface.co/suno/bark) |
| Fish Speech | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) | [HF: fishaudio/fish-speech-1.5](https://huggingface.co/fishaudio/fish-speech-1.5) |
| Kokoro | [hexgrad/kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | [HF: hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| Dia | [nari-labs/dia](https://github.com/nari-labs/dia) | [HF: nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) |
| Chatterbox | [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | [HF: ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) |
| F5-TTS | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) | [HF: SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) |
| Qwen2.5-Omni | [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | [HF: Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) |
| VibeVoice | [microsoft/VibeVoice](https://huggingface.co/microsoft/VibeVoice-1.5B) | [HF: microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| Higgs Audio | [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) | [HF: bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) |

### Core Libraries

| Library | Purpose |
|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | Gateway and worker web framework |
| [OpenAI Whisper](https://github.com/openai/whisper) | Transcription verification |
| [HuggingFace Hub](https://github.com/huggingface/huggingface_hub) | Model downloads |
| [FFmpeg](https://ffmpeg.org/) | Audio format conversion |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) | Phoneme synthesis (Kokoro) |

### Audio Processing Libraries

| Library | Stage |
|---------|-------|
| [noisereduce](https://github.com/timsainb/noisereduce) | De-reverb (spectral gating) |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS loudness normalization |
| [pyrubberband](https://github.com/bmcfee/pyrubberband) | Tempo adjustment |
| [pydub](https://github.com/jiaaro/pydub) | Silence detection and trimming |
| [scipy](https://scipy.org/) | Highpass filter, de-essing |
| [soundfile](https://github.com/bastibe/python-soundfile) | WAV I/O |
| [numpy](https://numpy.org/) | Audio array processing |

---

## Troubleshooting

**"Worker failed to start" / model won't load**
- Ensure the model's virtual environment is installed (GUI > Environments tab)
- Check the worker log in `output/logs/` for the specific error
- Verify you have enough VRAM (see VRAM table above)

**Kokoro fails with phonemizer/espeak error**
- espeak-ng should be auto-installed by `install.bat`. If missing, re-run `install.bat` or manually download from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0)

**CUDA out of memory**
- Kill workers for other models first: `curl -X DELETE http://127.0.0.1:8100/api/workers/{id}`
- Use a smaller model (Kokoro ~1 GB vs Qwen ~14 GB)
- Or pin models to different GPUs: `{"device": "cuda:1"}`

**Dia produces silence on first requests**
- Normal warmup behavior. The gateway auto-retries up to 3 times. Audio quality improves after 2-3 inferences.

**Qwen takes forever to load**
- First load is ~12 minutes (downloads/caches 14 GB of weights). Subsequent loads use cached weights and are faster.

**Tempo/speed adjustment doesn't work**
- Rubberband must be installed. Re-run `install.bat` or check that `rubberband/` directory exists with `rubberband.exe`.

**Port already in use**
- Change the gateway port: `launcher.bat api --port 8200`
- Or kill whatever is using port 8100: `netstat -ano | findstr :8100`

**Audio sounds robotic / low quality**
- Try increasing `temperature` (more expressive) or adjusting model-specific params
- Disable post-processing to test raw output: `"skip_post_process": true`
- Ensure the model environment has CUDA PyTorch (not CPU-only), except Higgs which is CPU-only

---

## Version History

### v3.0.2 (2026-02-17)
- Fix venv creation failing with embedded Python (`python -m venv` → `python -m virtualenv`)
- Fix operator precedence bug in `run_git_clone()` editable install check
- Add `virtualenv` to requirements.txt for bootstrap

### v3.0.1 (2026-02-11)
- Fix `install.bat` PowerShell multi-line commands failing on fresh installs

### v3.0.0 (2026-02-06)
- Gateway + Worker architecture with subprocess isolation
- 10 TTS models integrated and verified
- Multi-GPU support with round-robin load balancing
- Dynamic worker management (spawn, kill, scale, failover)
- Auto-spawn workers on first request
- Health checks with dead worker cleanup
- Worker management API endpoints

### v2.0.0 (2026-02-05)
- Full TTS pipeline: chunking, post-processing, assembly
- 7-stage audio processing pipeline
- Whisper transcription verification
- Job management with cancellation and recovery
- Multi-chunk support with per-model text limits

### v1.0.0 (2026-02-05)
- Initial portable release
- Embedded Python 3.10, portable Git, FFmpeg
- 8 virtual environment configurations
- Tkinter GUI with environment, model, and server tabs
- HuggingFace model downloads

---

## License

This project is a gateway/orchestrator. Each integrated TTS model has its own license:

| Model | License |
|-------|---------|
| XTTS v2, Bark | [MPL 2.0](https://www.mozilla.org/en-US/MPL/2.0/) (Coqui) |
| Chatterbox | [MIT](https://opensource.org/licenses/MIT) |
| F5-TTS | [MIT](https://opensource.org/licenses/MIT) |
| Fish Speech | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Kokoro | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Dia | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Qwen2.5-Omni | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| VibeVoice | [MIT](https://opensource.org/licenses/MIT) |
| Higgs Audio | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

The gateway/orchestrator code is licensed under the [MIT License](LICENSE).

---

## Credits

Built with [Claude Opus 4.6](https://www.anthropic.com/claude) as a pair-programming partner.
