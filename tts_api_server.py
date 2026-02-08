"""
TTS API Server - Gateway + Worker Architecture

The gateway runs on port 8100 and orchestrates:
- Text chunking and normalization
- Audio post-processing (de-reverb, high-pass, de-ess, trim, normalize, peak limit)
- Whisper verification (via whisper worker subprocess)
- Job tracking with recovery and cancellation
- Format conversion (wav/mp3/ogg/flac/m4a)
- Worker lifecycle management (spawn/kill/scale/health)
- Load balancing across multiple worker instances
- Multi-GPU support

Workers are separate subprocesses, each running one TTS model instance.
The gateway delegates inference to workers via HTTP.

Run with: uvicorn tts_api_server:app --host 0.0.0.0 --port 8100
"""

import asyncio
import os
import re
import sys
import threading
from pathlib import Path

# Bootstrap: ensure project root is on sys.path for sibling imports
_BASE_DIR = Path(__file__).parent.resolve()
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import io
import uuid
import time
import base64
import logging
import soundfile as sf
from typing import Optional
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import (
    BASE_DIR, MODELS_DIR, VENVS_DIR, OUTPUT_DIR, FFMPEG_PATH,
    DEFAULT_API_HOST, DEFAULT_API_PORT, JOBS_DIR, PROJECTS_OUTPUT,
    VOICE_DIR, MAX_RETRIES, WHISPER_MODEL_SIZE, WHISPER_ENABLED,
    WHISPER_DEFAULT_TOLERANCE, WHISPER_AVAILABLE_MODELS, MAX_INFERENCE_WORKERS,
    WORKER_AUTO_SPAWN, WORKER_DEFAULT_DEVICE,
    WORKER_PORT_MIN, WORKER_PORT_MAX,
    setup_environment,
)
from text_utils import chunk_text_for_model
from audio_profiles import PROFILES
from audio_processing import post_process, verify_with_whisper
from audio_assembler import assemble_chunks, convert_format
from job_manager import JobManager
from worker_registry import WorkerRegistry
from worker_manager import WorkerManager

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup_environment()
OUTPUT_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_OUTPUT.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker infrastructure
# ---------------------------------------------------------------------------
registry = WorkerRegistry(WORKER_PORT_MIN, WORKER_PORT_MAX)
worker_manager = WorkerManager(registry)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start health check loop on startup, kill all workers on shutdown."""
    worker_manager.start_health_checks()
    logger.info("Gateway started - worker health checks active")
    yield
    logger.info("Gateway shutting down - killing all workers...")
    worker_manager.stop_health_checks()
    await worker_manager.kill_all_workers()
    logger.info("All workers stopped")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TTS API Gateway",
    description="Gateway server orchestrating TTS workers with full pipeline",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
job_manager = JobManager(JOBS_DIR)
executor = ThreadPoolExecutor(max_workers=MAX_INFERENCE_WORKERS)

# Track running job_id per model for cancellation
_running_jobs: dict[str, set[str]] = {}  # model -> set of active job_ids
_running_jobs_lock = threading.Lock()

# Bark history prompt state: { job_id: history_data }
# Gateway manages this, passes to worker per-chunk, receives updated history back
_bark_history: dict[str, object] = {}

# Whisper worker reference (worker_id of the dedicated whisper worker)
_whisper_worker_id: str | None = None

# Prevent concurrent auto-spawns for the same model
_spawn_locks: dict[str, asyncio.Lock] = {}
_spawn_locks_lock = threading.Lock()


def _ts():
    return time.strftime("%H:%M:%S")


# ============================================================
# Request models
# ============================================================
class PipelineTTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    language: Optional[str] = "en"
    speed: Optional[float] = 1.0
    temperature: Optional[float] = 0.65
    repetition_penalty: Optional[float] = 2.0
    de_reverb: Optional[float] = 0.7
    de_ess: Optional[float] = 0.0
    tolerance: Optional[float] = 80.0
    verify_whisper: Optional[bool] = False
    whisper_model: Optional[str] = None
    output_format: Optional[str] = "wav"
    save_path: Optional[str] = None
    skip_post_process: Optional[bool] = False
    auto_retry: Optional[int] = None
    device: Optional[str] = None
    mode: Optional[str] = "cloned"


class CancelRequest(BaseModel):
    job_id: Optional[str] = None


class SpawnRequest(BaseModel):
    model: str
    device: Optional[str] = None


class ScaleRequest(BaseModel):
    count: int
    device: Optional[str] = None


# ============================================================
# Health / discovery
# ============================================================
@app.get("/")
@app.get("/health")
async def health():
    workers = registry.all_workers()
    loaded_models = list(set(w.model for w in workers if w.status in ("ready", "busy")))
    return {
        "status": "ok",
        "message": "TTS API Gateway running",
        "loaded_models": loaded_models,
        "worker_count": len(workers),
    }


@app.get("/api/models")
async def list_models():
    return {
        "models": [
            {"id": "chatterbox", "name": "Chatterbox TTS", "env": "chatterbox_env"},
            {"id": "f5", "name": "F5-TTS", "env": "f5tts_env"},
            {"id": "xtts", "name": "XTTS v2", "env": "coqui_env"},
            {"id": "bark", "name": "Bark", "env": "coqui_env"},
            {"id": "fish", "name": "Fish Speech", "env": "unified_env"},
            {"id": "kokoro", "name": "Kokoro", "env": "unified_env"},
            {"id": "dia", "name": "Dia", "env": "unified_env"},
            {"id": "qwen", "name": "Qwen TTS", "env": "qwen3_env"},
            {"id": "vibevoice", "name": "VibeVoice", "env": "vibevoice_env"},
            {"id": "higgs", "name": "Higgs Audio", "env": "higgs_env"},
        ]
    }


@app.get("/api/models/status")
async def models_status():
    workers = registry.all_workers()
    status = {}
    for w in workers:
        if w.model not in status:
            status[w.model] = {"workers": [], "loaded": False}
        status[w.model]["workers"].append({
            "worker_id": w.worker_id,
            "device": w.device,
            "status": w.status,
        })
        if w.status in ("ready", "busy"):
            status[w.model]["loaded"] = True
    return {"models": status}


# ============================================================
# GPU / Device discovery
# ============================================================
@app.get("/api/devices")
async def list_devices():
    devices = worker_manager.detect_devices()
    return {"devices": devices}


# ============================================================
# Worker management endpoints
# ============================================================
@app.get("/api/workers")
async def list_workers():
    return {"workers": registry.to_dict_list()}


@app.post("/api/workers/spawn")
async def spawn_worker(req: SpawnRequest):
    try:
        worker = await worker_manager.spawn_worker(req.model, req.device)
        return {
            "status": "spawned",
            "worker_id": worker.worker_id,
            "model": worker.model,
            "port": worker.port,
            "device": worker.device,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workers/{worker_id}")
async def delete_worker(worker_id: str):
    killed = await worker_manager.kill_worker(worker_id)
    if not killed:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    return {"status": "killed", "worker_id": worker_id}


@app.post("/api/models/{model}/scale")
async def scale_model(model: str, req: ScaleRequest):
    try:
        workers = await worker_manager.scale_model(model, req.count, req.device)
        return {
            "status": "scaled",
            "model": model,
            "device": req.device or WORKER_DEFAULT_DEVICE,
            "count": len(workers),
            "workers": [w.worker_id for w in workers],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Model load/unload (now delegates to workers)
# ============================================================
@app.post("/api/models/{model}/load")
async def load_model(model: str, device: Optional[str] = None):
    """Load a model by spawning a worker for it."""
    existing = registry.get_ready_workers(model)
    if existing:
        return {"status": "already_loaded", "model": model,
                "workers": [w.worker_id for w in existing]}
    try:
        worker = await worker_manager.spawn_worker(model, device)
        return {"status": "loaded", "model": model, "worker_id": worker.worker_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/{model}/unload")
async def unload_model(model: str):
    """Unload a model by killing all its workers."""
    workers = registry.workers_for_model(model)
    if not workers:
        return {"status": "not_loaded", "model": model}

    killed = 0
    for w in workers:
        if await worker_manager.kill_worker(w.worker_id):
            killed += 1
    return {"status": "unloaded", "model": model, "workers_killed": killed}


# ============================================================
# Whisper management
# ============================================================
@app.get("/api/whisper")
async def whisper_info():
    global _whisper_worker_id
    whisper_workers = registry.workers_for_model("whisper")
    loaded_sizes = []

    # Query the whisper worker for its loaded sizes
    if whisper_workers:
        for w in whisper_workers:
            if w.status in ("ready", "busy"):
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.get(f"http://127.0.0.1:{w.port}/health")
                        if resp.status_code == 200:
                            loaded_sizes.append(w.worker_id)
                except Exception:
                    pass

    return {
        "available_models": WHISPER_AVAILABLE_MODELS,
        "default": WHISPER_MODEL_SIZE,
        "loaded": loaded_sizes,
        "enabled": WHISPER_ENABLED,
        "whisper_workers": [w.worker_id for w in whisper_workers],
    }


@app.post("/api/whisper/{size}/load")
async def load_whisper(size: str):
    global _whisper_worker_id
    if size not in WHISPER_AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size. Choose from: {list(WHISPER_AVAILABLE_MODELS.keys())}"
        )

    # Spawn a whisper worker if none exists
    whisper_workers = registry.workers_for_model("whisper")
    if not whisper_workers:
        try:
            worker = await worker_manager.spawn_worker("whisper")
            _whisper_worker_id = worker.worker_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to spawn whisper worker: {e}")

    return {"status": "loaded", "size": size, "info": WHISPER_AVAILABLE_MODELS[size]}


@app.post("/api/whisper/{size}/unload")
async def unload_whisper(size: str):
    # Kill whisper workers
    whisper_workers = registry.workers_for_model("whisper")
    for w in whisper_workers:
        await worker_manager.kill_worker(w.worker_id)
    return {"status": "unloaded", "size": size}


# ============================================================
# Job management endpoints
# ============================================================
@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": job_manager.list_jobs()}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/recover")
async def recover_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = Path(job.get("job_dir", JOBS_DIR / job_id))
    recovered = job_manager.recover_job(job_dir)
    if recovered is None:
        raise HTTPException(status_code=400, detail="Job not recoverable (already complete or missing)")
    model = recovered["model"]
    if model in PROFILES:
        executor.submit(_run_pipeline_recovery, model, recovered, job_dir)
        return {"status": "recovering", "job_id": recovered["job_id"],
                "resuming_from_chunk": recovered["chunks_completed"]}
    raise HTTPException(status_code=400, detail=f"Recovery not supported for model: {model}")


# ============================================================
# Cancel endpoints
# ============================================================
@app.post("/api/tts/{model}/cancel")
async def cancel_model_job(model: str, body: CancelRequest = CancelRequest()):
    if body.job_id:
        # Cancel a specific job
        if job_manager.request_cancel(body.job_id):
            return {"status": "cancel_requested", "job_id": body.job_id}
        return {"status": "job_not_found", "job_id": body.job_id}
    # Cancel all running jobs for this model
    with _running_jobs_lock:
        active = list(_running_jobs.get(model, set()))
    if active:
        cancelled = [jid for jid in active if job_manager.request_cancel(jid)]
        if cancelled:
            return {"status": "cancel_requested", "job_ids": cancelled}
    return {"status": "no_running_job", "model": model}


# ============================================================
# Pipeline TTS endpoints (backward compatible)
# ============================================================
@app.post("/api/tts/xtts")
async def tts_xtts(req: PipelineTTSRequest):
    return await _handle_pipeline_request("xtts", req)


@app.post("/api/tts/fish")
async def tts_fish(req: PipelineTTSRequest):
    return await _handle_pipeline_request("fish", req)


@app.post("/api/tts/kokoro")
async def tts_kokoro(req: PipelineTTSRequest):
    return await _handle_pipeline_request("kokoro", req)


@app.post("/api/tts/bark")
async def tts_bark(req: PipelineTTSRequest):
    return await _handle_pipeline_request("bark", req)


@app.post("/api/tts/chatterbox")
async def tts_chatterbox(req: PipelineTTSRequest):
    return await _handle_pipeline_request("chatterbox", req)


@app.post("/api/tts/f5")
async def tts_f5(req: PipelineTTSRequest):
    return await _handle_pipeline_request("f5", req)


@app.post("/api/tts/dia")
async def tts_dia(req: PipelineTTSRequest):
    return await _handle_pipeline_request("dia", req)


@app.post("/api/tts/qwen")
async def tts_qwen(req: PipelineTTSRequest):
    return await _handle_pipeline_request("qwen", req)


@app.post("/api/tts/vibevoice")
async def tts_vibevoice(req: PipelineTTSRequest):
    return await _handle_pipeline_request("vibevoice", req)


@app.post("/api/tts/higgs")
async def tts_higgs(req: PipelineTTSRequest):
    return await _handle_pipeline_request("higgs", req)


async def _handle_pipeline_request(model: str, req: PipelineTTSRequest):
    """Shared handler for all pipeline TTS requests."""
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Missing text")

    profile = PROFILES[model]
    params = req.model_dump()
    output_format = (req.output_format or "wav").lower()
    max_retries = req.auto_retry if req.auto_retry is not None else MAX_RETRIES

    # --- Recovery mode ---
    if raw_text.lower() == "##recover##":
        target = (req.save_path or "").strip()
        if not target:
            raise HTTPException(status_code=400, detail="Set save_path to the folder you want to recover")

        job_dir = _resolve_job_dir(target)
        recovered = job_manager.recover_job(job_dir)
        if recovered is None:
            raise HTTPException(status_code=400, detail="Job not recoverable or already finished")

        executor.submit(_run_pipeline_recovery, model, recovered, job_dir)
        return JSONResponse({
            "status": "recovering",
            "job_id": recovered["job_id"],
            "resuming_from_chunk": recovered["chunks_completed"],
            "total_chunks": recovered["total_chunks"],
        })

    # --- Ensure a worker exists for this model ---
    worker = registry.pick_worker(model)
    if not worker:
        if WORKER_AUTO_SPAWN:
            # Per-model lock prevents concurrent auto-spawns for the same model
            with _spawn_locks_lock:
                if model not in _spawn_locks:
                    _spawn_locks[model] = asyncio.Lock()
            async with _spawn_locks[model]:
                # Re-check after acquiring the lock — another request may have spawned one
                worker = registry.pick_worker(model)
                if not worker:
                    try:
                        device = req.device or WORKER_DEFAULT_DEVICE
                        worker = await worker_manager.spawn_worker(model, device)
                    except Exception as e:
                        raise HTTPException(
                            status_code=503,
                            detail=f"No worker available for '{model}' and auto-spawn failed: {e}"
                        )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"No worker available for model '{model}'. Spawn one first via POST /api/workers/spawn"
            )

    # --- New job ---
    text = raw_text
    chunks = chunk_text_for_model(text, model)

    save_path_input = (req.save_path or "").strip()
    job_dir, stem = _resolve_output_paths(model, save_path_input)
    job_dir.mkdir(parents=True, exist_ok=True)

    job = job_manager.create_job(
        model=model, text=text, chunks=chunks, params=params,
        output_format=output_format, sample_rate=profile["sample_rate"],
        stem=stem, job_dir=job_dir,
    )
    job_id = job["job_id"]

    logger.info("[%s %s] New job %s: %d chunks, format=%s",
                _ts(), model.upper(), job_id, len(chunks), output_format)

    # Run pipeline in thread pool (blocking HTTP calls to workers)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        _run_pipeline,
        model, job_id, chunks, 0, params, profile, job_dir, stem,
        output_format, max_retries, save_path_input,
    )
    return JSONResponse(result)


# ============================================================
# Pipeline core (runs in thread pool)
# ============================================================
def _run_pipeline(model: str, job_id: str, chunks: list[str],
                  start_from: int, params: dict, profile: dict,
                  job_dir: Path, stem: str, output_format: str,
                  max_retries: int, save_path_input: str) -> dict:
    """Execute the full TTS pipeline, delegating inference to workers via HTTP."""
    logger.info("[%s %s] Pipeline thread started for job %s (thread=%s, chunks=%d)",
                _ts(), model.upper(), job_id, threading.current_thread().name, len(chunks))

    with _running_jobs_lock:
        _running_jobs.setdefault(model, set()).add(job_id)
    sr = profile["sample_rate"]

    speed = float(params.get("speed", 1.0))
    de_reverb = float(params.get("de_reverb", 0.7))
    de_ess_raw = float(params.get("de_ess", 0.0))
    de_ess = de_ess_raw / 100.0 if de_ess_raw > 1.0 else de_ess_raw
    tolerance = float(params.get("tolerance", WHISPER_DEFAULT_TOLERANCE))
    do_verify_whisper = params.get("verify_whisper", False)
    skip_post_process = params.get("skip_post_process", False)

    try:
        # Bark: initialize history prompt state
        bark_base_history = None
        if model == "bark":
            voice_preset = params.get("voice") or "v2/en_speaker_6"
            bark_base_history = voice_preset
            _bark_history[job_id] = bark_base_history
            history_reset_every = profile.get("history_reset_every", 5)

        for i in range(start_from, len(chunks)):
            # Check cancellation
            if job_manager.is_cancelled(job_id):
                logger.info("[%s %s] Job %s cancelled at chunk %d",
                            _ts(), model.upper(), job_id, i)
                job_manager.fail_job(job_id, "Cancelled by user")
                return {"error": "Cancelled", "job_id": job_id, "cancelled_at_chunk": i}

            chunk_text = chunks[i]
            retry_count = 0

            while True:
                try:
                    # Bark: hybrid history - reset periodically
                    if model == "bark":
                        chunk_offset = i - start_from
                        if chunk_offset > 0 and chunk_offset % history_reset_every == 0:
                            logger.info("[%s BARK] Resetting history at chunk %d",
                                        _ts(), i)
                            _bark_history[job_id] = bark_base_history

                    # --- Inference via worker ---
                    infer_params = dict(params)
                    if model == "bark":
                        infer_params["history_prompt"] = _bark_history.get(job_id)

                    raw_data, chunk_sr = _infer_via_worker(model, chunk_text, infer_params, job_id)

                    # Prepend front pad if configured
                    front_pad = profile.get("front_pad_sec", 0.0)
                    if front_pad > 0:
                        pad = np.zeros(int(chunk_sr * front_pad), dtype=np.float32)
                        raw_data = np.concatenate([pad, raw_data])

                    # Save raw chunk
                    tmp_wav = OUTPUT_DIR / f"raw_{model}_{i}_{uuid.uuid4().hex}.wav"
                    sf.write(str(tmp_wav), raw_data, chunk_sr, subtype="PCM_16")

                    # Post-process
                    if not skip_post_process:
                        post_process(str(tmp_wav), profile, speed, de_reverb, de_ess)

                    # Whisper verification
                    if do_verify_whisper:
                        whisper_size = params.get("whisper_model") or WHISPER_MODEL_SIZE
                        passed, sim, transcript = _verify_whisper_via_worker(
                            str(tmp_wav), chunk_text,
                            params.get("language", "en"),
                            tolerance, whisper_size,
                        )
                        job_manager.update_chunk(
                            job_id, i, status="success" if passed else "failed",
                            whisper_transcript=transcript,
                            whisper_similarity=sim,
                            verification_passed=passed,
                        )
                        if not passed:
                            raise ValueError(
                                f"Whisper verification failed (similarity={sim:.3f})"
                            )

                    # Move to final chunk location
                    data, _ = sf.read(str(tmp_wav))
                    duration_sec = len(data) / chunk_sr

                    chunk_wav = job_dir / f"chunk_{i:03d}.wav"
                    tmp_wav.replace(chunk_wav)

                    job_manager.update_chunk(
                        job_id, i, status="success",
                        duration=duration_sec,
                        audio_file=chunk_wav.name,
                    )

                    logger.info("[%s %s] Chunk %03d -> %.2fs (success)",
                                _ts(), model.upper(), i, duration_sec)
                    break  # Success -> next chunk

                except Exception as e:
                    retry_count += 1
                    error_msg = str(e) or "Unknown error"

                    if retry_count > max_retries:
                        logger.error("[%s %s] Chunk %03d failed after %d retries: %s",
                                     _ts(), model.upper(), i, max_retries, error_msg)
                        job_manager.update_chunk(job_id, i, status="failed", error=error_msg)
                        job_manager.fail_job(
                            job_id, f"Chunk {i} failed after {max_retries} retries"
                        )
                        return {
                            "error": "generation_failed",
                            "reason": f"Chunk {i} failed after {max_retries} retries",
                            "failed_at_chunk": i,
                            "job_id": job_id,
                            "job_folder": str(job_dir.name),
                            "recover_command": f"##recover## (save_path: {job_dir.name})",
                        }

                    logger.warning("[%s %s] Chunk %03d retry %d/%d: %s",
                                   _ts(), model.upper(), i, retry_count, max_retries,
                                   error_msg)
                    time.sleep(1)

        # --- Assembly ---
        missing = [
            f"chunk_{i:03d}.wav" for i in range(len(chunks))
            if not (job_dir / f"chunk_{i:03d}.wav").exists()
        ]
        if missing:
            return {
                "status": "incomplete",
                "message": f"Missing {len(missing)} chunk(s)",
                "missing_count": len(missing),
                "job_id": job_id,
                "job_folder": str(job_dir.name),
                "recover_command": f"##recover## (save_path: {job_dir.name})",
            }

        chunk_files = [job_dir / f"chunk_{i:03d}.wav" for i in range(len(chunks))]

        assembled_wav = OUTPUT_DIR / f"assembled_{uuid.uuid4().hex}.wav"
        assemble_chunks(
            chunk_files, str(assembled_wav), sr,
            inter_pause=profile.get("inter_pause_sec", 0.25),
            front_pad=profile.get("padding_sec", 0.5),
            end_pad=profile.get("padding_sec", 0.5),
        )

        # Calculate duration from the assembled WAV (before format conversion,
        # since sf.read cannot read mp3/ogg/m4a)
        assembled_data, _ = sf.read(str(assembled_wav))
        total_duration = len(assembled_data) / sr
        del assembled_data

        # Format conversion
        final_filename = f"{stem}_final.{output_format}"
        final_path = job_dir / final_filename

        if output_format != "wav":
            convert_format(str(assembled_wav), str(final_path), output_format, FFMPEG_PATH)
            assembled_wav.unlink(missing_ok=True)
        else:
            assembled_wav.replace(final_path)

        job_manager.complete_job(job_id, final_filename, total_duration)

        logger.info("[%s %s] Job %s complete: %s (%.1fs)",
                    _ts(), model.upper(), job_id, final_filename, total_duration)

        resp = {
            "status": "completed",
            "job_id": job_id,
            "filename": final_filename,
            "saved_to": str(final_path),
            "sample_rate": sr,
            "duration_sec": round(total_duration, 3),
            "format": output_format,
        }

        if not save_path_input:
            resp["audio_base64"] = base64.b64encode(final_path.read_bytes()).decode("utf-8")

        return resp

    except Exception as e:
        error_str = str(e) or "Unknown error"
        logger.error("[%s %s] Unexpected error: %s", _ts(), model.upper(), error_str)
        job_manager.fail_job(job_id, error_str)
        return {
            "error": "generation_failed",
            "reason": error_str,
            "job_id": job_id,
            "job_folder": str(job_dir.name),
        }
    finally:
        with _running_jobs_lock:
            active = _running_jobs.get(model, set())
            active.discard(job_id)
            if not active:
                _running_jobs.pop(model, None)
        _bark_history.pop(job_id, None)
        job_manager.cleanup_cancel_flag(job_id)


def _run_pipeline_recovery(model: str, job: dict, job_dir: Path):
    """Resume a recovered job from where it left off."""
    chunks = [c["text"] for c in job["chunks"]]
    start_from = job["chunks_completed"]
    params = job["parameters"]
    profile = PROFILES[model]
    output_format = job.get("output_format", "wav")
    stem = job_dir.name

    _run_pipeline(
        model, job["job_id"], chunks, start_from, params,
        profile, job_dir, stem, output_format, MAX_RETRIES, str(job_dir),
    )


# ============================================================
# Worker-delegated inference
# ============================================================
def _finalize_worker(worker_id: str, timed_out: bool = False) -> None:
    """Check if a worker is still alive and mark ready or dead accordingly.

    If timed_out=True, the worker is assumed stuck (e.g. hung CUDA generate())
    and will be forcibly killed at the process level.
    """
    worker_info = registry.get(worker_id)
    if not worker_info:
        return

    if timed_out and worker_info.process and worker_info.process.poll() is None:
        # Worker process is alive but not responding — kill it
        logger.warning("Worker %s timed out — killing stuck process (pid=%s)",
                      worker_id, worker_info.process.pid)
        try:
            worker_info.process.terminate()
            try:
                worker_info.process.wait(timeout=5)
            except Exception:
                worker_info.process.kill()
                worker_info.process.wait(timeout=5)
        except Exception as e:
            logger.error("Failed to kill stuck worker %s: %s", worker_id, e)
        if worker_info.log_fh:
            try:
                worker_info.log_fh.close()
            except Exception:
                pass
            worker_info.log_fh = None
        worker_info.process = None
        registry.unregister(worker_id)
        registry.release_port(worker_info.port)
        return

    if worker_info.process and worker_info.process.poll() is not None:
        logger.warning("Worker %s process died (exit code %d)",
                      worker_id, worker_info.process.returncode)
        if worker_info.log_fh:
            try:
                worker_info.log_fh.close()
            except Exception:
                pass
            worker_info.log_fh = None
        registry.unregister(worker_id)
        registry.release_port(worker_info.port)
    else:
        registry.mark_ready(worker_id)


def _infer_via_worker(model: str, text: str, params: dict,
                      job_id: str | None = None) -> tuple:
    """Send inference request to a worker and decode the response.

    Retries with different workers on connection failures.
    Returns (numpy_array, sample_rate).
    """
    logger.info("[%s %s] _infer_via_worker called for job %s, text=%r",
                _ts(), model.upper(), job_id, text[:80] + "..." if len(text) > 80 else text)

    # Pre-serialize bark history (do once, reuse across retries)
    send_params = dict(params)
    if model == "bark" and "history_prompt" in send_params:
        hp = send_params["history_prompt"]
        if isinstance(hp, tuple) and len(hp) == 3 and hasattr(hp[0], 'dtype'):
            encoded = []
            for arr in hp:
                buf = io.BytesIO()
                np.save(buf, arr)
                encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            send_params["history_prompt"] = {"_bark_b64": encoded}

    max_worker_attempts = 3
    last_error = None
    tried_workers: set[str] = set()

    for attempt in range(max_worker_attempts):
        worker = registry.pick_worker(model)
        if not worker:
            break

        # Don't retry the same worker that just failed
        if worker.worker_id in tried_workers:
            # Try to find a different worker
            all_ready = registry.get_ready_workers(model)
            untried = [w for w in all_ready if w.worker_id not in tried_workers]
            if not untried:
                break
            worker = untried[0]

        tried_workers.add(worker.worker_id)
        worker_id = worker.worker_id
        registry.mark_busy(worker_id, job_id)

        infer_start = time.time()
        try:
            url = f"http://127.0.0.1:{worker.port}/infer"
            # Large models may need longer for first inference (includes model loading)
            infer_timeout = 900.0 if model in ("qwen", "higgs", "vibevoice", "dia", "fish") else 300.0
            logger.info("[%s %s] Sending POST %s (worker=%s, timeout=%.0fs)",
                        _ts(), model.upper(), url, worker_id, infer_timeout)

            with httpx.Client(timeout=infer_timeout) as client:
                resp = client.post(url, json={"text": text, "params": send_params})
            logger.info("[%s %s] Worker %s responded with status %d",
                        _ts(), model.upper(), worker_id, resp.status_code)

            if resp.status_code != 200:
                detail = resp.json().get("detail", resp.text) if resp.headers.get(
                    "content-type", "").startswith("application/json") else resp.text
                raise RuntimeError(
                    f"Worker {worker_id} returned {resp.status_code}: {detail}")

            data = resp.json()
            audio_b64 = data["audio_b64"]
            sample_rate = data["sample_rate"]

            buf = io.BytesIO(base64.b64decode(audio_b64))
            audio_array = np.load(buf, allow_pickle=False)

            # For bark, capture updated history
            if model == "bark" and "bark_history" in data and job_id:
                history_parts = []
                for part_b64 in data["bark_history"]:
                    hbuf = io.BytesIO(base64.b64decode(part_b64))
                    history_parts.append(np.load(hbuf, allow_pickle=False))
                _bark_history[job_id] = tuple(history_parts)

            # Success - mark worker ready
            registry.mark_ready(worker_id)
            return audio_array, sample_rate

        except (httpx.ReadTimeout, httpx.PoolTimeout, httpx.WriteTimeout) as e:
            # Inference timed out — worker is likely stuck in generate().
            # Kill the process so it doesn't hold GPU memory forever.
            elapsed = time.time() - infer_start
            logger.error("Worker %s inference TIMED OUT after %.0fs (attempt %d/%d): %s",
                        worker_id, elapsed, attempt + 1, max_worker_attempts, e)
            _finalize_worker(worker_id, timed_out=True)
            last_error = f"Inference timed out after {int(elapsed)}s"
            continue

        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                httpx.ConnectTimeout, httpx.RemoteProtocolError,
                ConnectionError) as e:
            # Worker connection failure - mark dead/ready, try next worker
            logger.warning("Worker %s connection failed (attempt %d/%d): %s",
                          worker_id, attempt + 1, max_worker_attempts, e)
            _finalize_worker(worker_id)
            last_error = str(e)
            continue

        except Exception:
            # Non-connection errors (inference failure, decode error) -
            # finalize worker and re-raise for the outer retry loop
            _finalize_worker(worker_id)
            raise

    if last_error:
        raise RuntimeError(
            f"All {len(tried_workers)} worker(s) for '{model}' failed: {last_error}")
    raise RuntimeError(f"No ready worker available for model '{model}'")


def _verify_whisper_via_worker(audio_path: str, expected_text: str,
                               language: str, tolerance: float,
                               whisper_size: str) -> tuple:
    """Verify audio via whisper worker, falling back to local whisper.

    Returns (passed: bool, similarity: float, transcript: str).
    """
    # Try whisper worker first
    whisper_workers = registry.get_ready_workers("whisper")
    if whisper_workers:
        worker = whisper_workers[0]
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"http://127.0.0.1:{worker.port}/transcribe",
                    json={"audio_path": audio_path, "size": whisper_size},
                )
            if resp.status_code == 200:
                result = resp.json()
                transcript = result.get("text", "")
                # Sanitize for comparison
                clean_expected = expected_text.lower().strip()
                clean_transcript = transcript.lower().strip()
                sim = SequenceMatcher(None, clean_expected, clean_transcript).ratio()
                passed = (sim * 100) >= tolerance
                return passed, sim, transcript
        except Exception as e:
            logger.warning("Whisper worker failed, falling back to local: %s", e)

    # Fallback: try local whisper (if torch available from some other source)
    try:
        import whisper
        model = whisper.load_model(whisper_size)
        return verify_with_whisper(audio_path, expected_text, language, tolerance, model)
    except ImportError:
        logger.warning("No whisper available (no worker, no local torch)")
        return True, 1.0, "(whisper unavailable)"


# ============================================================
# Path helpers
# ============================================================
def _resolve_output_paths(model: str, save_path_input: str) -> tuple[Path, str]:
    """Determine job directory and file stem from save_path."""
    if not save_path_input:
        stem = f"{model}_{int(time.time())}"
        job_dir = JOBS_DIR / f"temp_{stem}"
        return job_dir, stem

    if "/" in save_path_input or "\\" in save_path_input:
        full_path = Path(save_path_input).expanduser().resolve()
        if full_path.suffix:
            job_dir = full_path.parent
            stem = full_path.stem
        else:
            job_dir = full_path
            stem = full_path.name
    else:
        job_dir = PROJECTS_OUTPUT / save_path_input
        stem = save_path_input

    return job_dir, stem


def _resolve_job_dir(target: str) -> Path:
    """Resolve a recovery target to a job directory."""
    if "/" in target or "\\" in target:
        return Path(target).expanduser().resolve()
    candidate = PROJECTS_OUTPUT / target
    if (candidate / "job.json").exists():
        return candidate
    candidate = JOBS_DIR / target
    if (candidate / "job.json").exists():
        return candidate
    return PROJECTS_OUTPUT / target


# ============================================================
# Model-specific helper endpoints
# ============================================================
@app.get("/api/tts/xtts/voices")
async def xtts_voices():
    return {
        "voices": [
            "Aaron Dreschner", "Abrahan Mack", "Adde Michal", "Alexandra Hisakawa",
            "Alison Dietlinde", "Alma María", "Ana Florence", "Andrew Chipper",
            "Annmarie Nele", "Asya Anara", "Badr Odhiambo", "Baldur Sanjin",
            "Barbora MacLean", "Brenda Stern", "Camilla Holmström",
            "Chandra MacFarland", "Claribel Dervla", "Craig Gutsy",
            "Daisy Studious", "Damien Black", "Damjan Chapman",
            "Dionisio Schuyler", "Eugenio Mataracı", "Ferran Simen",
            "Filip Traverse", "Gilberto Mathias", "Gitta Nikolina", "Gracie Wise",
            "Henriette Usha", "Ige Behringer", "Ilkin Urbano",
            "Kazuhiko Atallah", "Kumar Dahl", "Lidiya Szekeres",
            "Lilya Stainthorpe", "Ludvig Milivoj", "Luis Moray", "Maja Ruoho",
            "Marcos Rudaski", "Narelle Moon", "Nova Hogarth", "Rosemary Okafor",
            "Royston Min", "Sofia Hellen", "Suad Qasim", "Szofi Granger",
            "Tammie Ema", "Tammy Grit", "Tanja Adelina", "Torcull Diarmuid",
            "Uta Obando", "Viktor Eka", "Viktor Menelaos", "Vjollca Johnnie",
            "Wulf Carlevaro", "Xavier Hayasaka", "Zacharie Aimilios",
            "Zofija Kendrick",
        ],
        "note": (
            "Built-in voices: set 'voice' to a speaker name and 'mode' to 'built-in'. "
            "For voice cloning: set 'voice' to a WAV file path and 'mode' to 'cloned' (default)."
        ),
    }


@app.get("/api/tts/kokoro/voices")
async def kokoro_voices():
    voices_dir = Path(MODELS_DIR) / "kokoro" / "voices"
    if voices_dir.exists():
        voices = sorted(p.stem for p in voices_dir.glob("*.pt"))
    else:
        voices = ["af_heart"]
    return {
        "voices": voices,
        "note": (
            "Prefix meanings: a=American, b=British, j=Japanese, z=Mandarin, "
            "e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese. "
            "f/m after prefix = female/male."
        ),
    }


@app.get("/api/tts/bark/voices")
async def bark_voices():
    # Scan speaker embeddings directory for all available speakers
    emb_dir = Path(MODELS_DIR) / "bark" / "speaker_embeddings"
    speakers: set[str] = set()
    if emb_dir.exists():
        for f in emb_dir.iterdir():
            m = re.match(r"(.+?)_(semantic|coarse|fine)_prompt\.npy", f.name)
            if m:
                speakers.add(m.group(1))
    # Also include v2/ prefixed versions (Bark internally resolves these)
    v2_speakers = sorted(s for s in speakers if not s.startswith("v2"))
    all_voices = [f"v2/{s}" for s in v2_speakers] + sorted(speakers)
    return {
        "voices": all_voices,
        "note": (
            "v2/ prefixed speakers are recommended (better quality). "
            "Languages: en, de, es, fr, hi, it, ja, ko, pl, pt, ru, tr, zh. "
            "Voice is kept consistent across chunks via history prompt chaining."
        ),
    }


@app.get("/api/tts/fish/voices")
async def fish_voices():
    return {
        "voices": [],
        "note": (
            "Fish Speech uses reference audio for voice cloning. "
            "Set 'reference_audio' to a WAV file path. "
            "A default reference voice is loaded automatically if none specified."
        ),
    }


@app.get("/api/tts/chatterbox/voices")
async def chatterbox_voices():
    return {
        "voices": [],
        "note": (
            "Chatterbox uses reference audio for voice cloning. "
            "Set 'reference_audio' to a WAV file path. "
            "Adjustable parameters: exaggeration (0-2, default 0.5), "
            "cfg_weight (0-1, default 0.5)."
        ),
    }


@app.get("/api/tts/f5/voices")
async def f5_voices():
    return {
        "voices": [],
        "note": (
            "F5-TTS requires reference audio + transcript for voice cloning. "
            "Set 'reference_audio' to a WAV file path and "
            "'reference_text' to the transcript of that audio."
        ),
    }


@app.get("/api/tts/dia/voices")
async def dia_voices():
    return {
        "voices": [],
        "note": (
            "Dia uses [S1] and [S2] speaker tags inline in text for dialogue. "
            "Set 'reference_audio' to a WAV file for voice cloning. "
            "Supports non-verbal expressions: (laughs), (sighs), (clears throat), etc."
        ),
        "example": "[S1] Hello there! (laughs) [S2] Hey, how are you?",
    }


@app.get("/api/tts/qwen/voices")
async def qwen_voices():
    return {
        "voices": ["Chelsie", "Ethan"],
        "note": (
            "Chelsie: Female, warm and clear voice (default). "
            "Ethan: Male, bright and energetic voice."
        ),
    }


@app.get("/api/tts/vibevoice/voices")
async def vibevoice_voices():
    return {
        "voices": [],
        "note": (
            "VibeVoice uses reference audio for voice cloning. "
            "Set 'reference_audio' to a WAV file path. "
            "Text can use 'Speaker N:' format for multi-speaker output."
        ),
    }


@app.get("/api/tts/higgs/voices")
async def higgs_voices():
    return {
        "voices": [],
        "note": (
            "Higgs Audio uses reference audio for voice cloning. "
            "Set 'reference_audio' to a WAV file and 'reference_text' to its transcript. "
            "Supports multi-speaker dialogue and auto-prosody."
        ),
    }


# ============================================================
# Server management
# ============================================================
def start_server(host: str = "0.0.0.0", port: int = 8100):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    start_server(args.host, args.port)
