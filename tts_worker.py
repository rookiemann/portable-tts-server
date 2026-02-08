"""
TTS Worker - A single-model inference server launched as a subprocess by the gateway.

Usage: python tts_worker.py --model kokoro --port 8102 --device cuda:0

Each worker loads exactly ONE model and exposes:
  GET  /health  - Status, device, VRAM info
  POST /infer   - Generate audio from text (returns base64 numpy array + sample rate)
  POST /load    - (Re)load the model
  POST /unload  - Unload model from GPU, process stays alive
"""

import argparse
import base64
import gc
import io
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: inject config before anything else
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()

# Ensure our project root is on sys.path so config, audio_profiles, etc. import
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import VENVS_DIR, MODELS_DIR, VOICE_DIR, OUTPUT_DIR, setup_environment

setup_environment()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [worker] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import model->venv mapping from shared config
from config import MODEL_VENV_MAP


def _inject_venv(model: str) -> None:
    """Insert the model's venv site-packages into sys.path."""
    venv_name = MODEL_VENV_MAP.get(model)
    if not venv_name:
        logger.warning("No venv mapping for model '%s'", model)
        return

    venv_dir = VENVS_DIR / venv_name
    site_packages = venv_dir / "Lib" / "site-packages"

    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
        logger.info("Injected venv site-packages: %s", site_packages)

    # Some models have source clones inside the venv
    extra_dirs = {
        "vibevoice": venv_dir / "VibeVoice",
        "higgs": venv_dir / "higgs-audio",
        "fish": venv_dir / "repos" / "fish-speech",
    }
    extra = extra_dirs.get(model)
    if extra and extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
        logger.info("Injected extra source dir: %s", extra)


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------
_model_name: str = ""
_device: str = "cuda:0"
_model_obj = None  # the loaded model object (varies per model)
_loaded: bool = False


# ============================================================
# Model loading
# ============================================================
def _resolve_device(device_str: str) -> str:
    """Validate and return the device string."""
    import torch
    if device_str == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning("CUDA not available in this venv's torch (%s), using CPU",
                       torch.__version__)
        return "cpu"
    if device_str.startswith("cuda:"):
        idx = int(device_str.split(":")[1])
        if idx < torch.cuda.device_count():
            return device_str
        logger.warning("Device %s not found (have %d GPUs), using cuda:0",
                       device_str, torch.cuda.device_count())
        return "cuda:0"
    return "cuda:0"


def load_model() -> None:
    """Load the model onto the configured device."""
    global _model_obj, _loaded

    if _loaded:
        logger.info("Model %s already loaded", _model_name)
        return

    device = _resolve_device(_device)
    logger.info("Loading model %s on %s...", _model_name, device)
    start = time.time()

    if _model_name == "xtts":
        from TTS.api import TTS
        _model_obj = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    elif _model_name == "bark":
        from TTS.api import TTS
        _model_obj = TTS("tts_models/multilingual/multi-dataset/bark").to(device)

    elif _model_name == "fish":
        import torch as _torch
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.vqgan.inference import load_model as _load_vqgan
        from fish_speech.inference_engine import TTSInferenceEngine

        fish_dir = MODELS_DIR / "fish-speech"
        _precision = _torch.bfloat16 if "cuda" in device else _torch.float32

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(fish_dir),
            device=device,
            precision=_precision,
            compile=False,
        )
        # Use V1.5 FireflyGAN VQ-GAN codec (matches our checkpoint)
        decoder_model = _load_vqgan(
            config_name="firefly_gan_vq",
            checkpoint_path=str(fish_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
            device=device,
        )
        _model_obj = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=_precision,
            compile=False,
        )

    elif _model_name == "kokoro":
        from kokoro import KPipeline
        _model_obj = KPipeline(lang_code="a")

    elif _model_name == "chatterbox":
        from chatterbox import ChatterboxTTS
        _model_obj = ChatterboxTTS.from_pretrained(device=device)

    elif _model_name == "f5":
        # Block torchcodec - it tries to load FFmpeg shared DLLs that
        # aren't available in our static ffmpeg build. Make importlib
        # unable to find it so transformers falls back to soundfile.
        import importlib.util
        _orig_find_spec = importlib.util.find_spec
        def _patched_find_spec(name, *a, **kw):
            if name == "torchcodec":
                return None
            return _orig_find_spec(name, *a, **kw)
        importlib.util.find_spec = _patched_find_spec
        try:
            from f5_tts.api import F5TTS
            _model_obj = F5TTS()
        finally:
            importlib.util.find_spec = _orig_find_spec

    elif _model_name == "dia":
        from dia.model import Dia
        import torch as _torch
        _dia_dtype = "bfloat16" if "cuda" in device else "float32"
        _model_obj = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626",
            compute_dtype=_dia_dtype,
            device=_torch.device(device),
        )

    elif _model_name == "qwen":
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        import torch
        model_id = "Qwen/Qwen2.5-Omni-7B"
        qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        _model_obj = (qwen_model, processor)

    elif _model_name == "vibevoice":
        import torch
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        model_id = "microsoft/VibeVoice-1.5B"
        vv_processor = VibeVoiceProcessor.from_pretrained(model_id)
        vv_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
            attn_implementation="sdpa",
        )
        vv_model.eval()
        vv_model.set_ddpm_inference_steps(num_steps=10)
        _model_obj = (vv_model, vv_processor)

    elif _model_name == "higgs":
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        _model_obj = HiggsAudioServeEngine(
            model_path, tokenizer_path, device=device,
        )

    elif _model_name == "whisper":
        import whisper
        # For whisper worker, model_obj is a dict of loaded sizes
        _model_obj = {}

    else:
        raise ValueError(f"Unknown model: {_model_name}")

    _loaded = True
    elapsed = time.time() - start
    logger.info("Model %s loaded in %.1fs", _model_name, elapsed)

    # Dia needs warmup inferences to produce coherent speech.
    # First 1-2 inferences after cold load produce garbage (hum/drone).
    if _model_name == "dia":
        _warmup_dia()


def _warmup_dia():
    """Run a warmup inference on Dia to prime CUDA kernels."""
    import numpy as np
    warmup_text = "[S1] Warmup inference, please discard this audio."
    try:
        t0 = time.time()
        data, sr = _infer_dia(warmup_text, {"max_new_tokens": 512})
        rms = float(np.sqrt(np.mean(data**2)))
        rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100
        logger.info("Dia warmup: %.1fs, rms=%.1fdB", time.time() - t0, rms_db)
    except Exception as e:
        logger.warning("Dia warmup failed (expected): %s", e)


def unload_model() -> None:
    """Unload the model from GPU and free memory."""
    global _model_obj, _loaded

    if not _loaded:
        return

    logger.info("Unloading model %s...", _model_name)
    _model_obj = None
    _loaded = False
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass

    logger.info("Model %s unloaded", _model_name)


# ============================================================
# Inference functions (moved from tts_api_server.py)
# ============================================================
def infer(text: str, params: dict) -> tuple:
    """Run model-specific inference. Returns (numpy_array, sample_rate).

    For Bark, params may contain 'history_prompt' and the response will
    include updated history in a separate field.
    """
    import numpy as np

    if not _loaded or _model_obj is None:
        raise RuntimeError(f"Model {_model_name} is not loaded")

    if _model_name == "xtts":
        return _infer_xtts(text, params)
    elif _model_name == "fish":
        return _infer_fish(text, params)
    elif _model_name == "kokoro":
        return _infer_kokoro(text, params)
    elif _model_name == "bark":
        return _infer_bark(text, params)
    elif _model_name == "chatterbox":
        return _infer_chatterbox(text, params)
    elif _model_name == "f5":
        return _infer_f5(text, params)
    elif _model_name == "dia":
        return _infer_dia(text, params)
    elif _model_name == "qwen":
        return _infer_qwen(text, params)
    elif _model_name == "vibevoice":
        return _infer_vibevoice(text, params)
    elif _model_name == "higgs":
        return _infer_higgs(text, params)
    else:
        raise ValueError(f"Inference not implemented for: {_model_name}")


def _infer_xtts(text: str, params: dict) -> tuple:
    import numpy as np
    voice = params.get("voice") or ""
    mode = params.get("mode") or "cloned"

    speaker_param = {}
    if voice:
        voice_path = VOICE_DIR / voice
        if mode == "cloned" and voice_path.exists():
            speaker_param["speaker_wav"] = str(voice_path)
        elif mode == "cloned" and Path(voice).is_file():
            speaker_param["speaker_wav"] = voice
        else:
            # Built-in speaker name (or mode explicitly set to built-in)
            speaker_param["speaker"] = voice

    wav = _model_obj.tts(
        text=text,
        language=params.get("language") or "en",
        temperature=float(params.get("temperature") or 0.65),
        speed=float(params.get("speed") or 1.0),
        repetition_penalty=float(params.get("repetition_penalty") or 2.0),
        split_sentences=False,
        **speaker_param,
    )
    data = np.array(wav, dtype=np.float32)
    sr = _model_obj.synthesizer.output_sample_rate
    return data, sr


def _infer_fish(text: str, params: dict) -> tuple:
    import numpy as np
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    engine = _model_obj

    top_p = float(params.get("top_p") or 0.8)
    temperature = float(params.get("temperature") or 0.8)
    repetition_penalty = float(params.get("repetition_penalty") or 1.1)
    # Fish Speech requires 0 < repetition_penalty < 2 (strict);
    # gateway default is 2.0 (for XTTS), so clamp to Fish's safe range
    if repetition_penalty >= 2.0:
        repetition_penalty = 1.1
    chunk_length = int(params.get("chunk_length") or 200)

    # Handle reference audio (optional voice cloning)
    references = []
    voice = params.get("voice") or params.get("reference_audio") or ""
    if voice:
        voice_path = VOICE_DIR / voice
        if voice_path.exists():
            voice = str(voice_path)
        if Path(voice).exists():
            with open(voice, "rb") as f:
                audio_bytes = f.read()
            ref_text = params.get("reference_text") or ""
            references.append(ServeReferenceAudio(audio=audio_bytes, text=ref_text))

    req = ServeTTSRequest(
        text=text,
        chunk_length=chunk_length,
        format="wav",
        references=references,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    audio_segments = []
    for result in engine.inference(req):
        if result.code == "final" and result.audio:
            _, audio_data = result.audio
            audio_segments.append(audio_data)
        elif result.code == "error" and result.error:
            raise result.error

    if not audio_segments:
        raise RuntimeError("Fish Speech returned no audio")

    data = np.concatenate(audio_segments, axis=0).astype(np.float32)
    return data, 24000


def _infer_kokoro(text: str, params: dict) -> tuple:
    import numpy as np
    voice = params.get("voice") or "af_heart"
    speed = float(params.get("speed") or 1.0)

    gen = _model_obj(text, voice=voice, speed=speed)
    audio_segments = [audio for _, _, audio in gen]

    if not audio_segments:
        raise RuntimeError("Kokoro returned no audio segments")

    raw_audio = np.concatenate(audio_segments, axis=0).astype(np.float32)
    return raw_audio, 24000


def _infer_bark(text: str, params: dict) -> tuple:
    """Bark inference with history prompt support.

    The gateway passes history_prompt through params and expects
    updated history back via the 'bark_history' field in the response.

    history_prompt from gateway is either:
      - None / string → first chunk, use (None, None, None)
      - list of 3 base64-encoded numpy arrays → decode to torch tensors
    """
    import numpy as np
    import torch

    bark_model = _model_obj.synthesizer.tts_model

    # Decode history prompt from gateway
    # Gateway sends either: None, string, or {"_bark_b64": [b64_1, b64_2, b64_3]}
    raw_history = params.get("history_prompt")
    if raw_history is None or isinstance(raw_history, str):
        # First chunk or string preset: no tensor history
        history = (None, None, None)
    elif isinstance(raw_history, dict) and "_bark_b64" in raw_history:
        # Base64-encoded numpy arrays from gateway
        parts = []
        for b64_str in raw_history["_bark_b64"]:
            buf = io.BytesIO(base64.b64decode(b64_str))
            arr = np.load(buf, allow_pickle=False)
            parts.append(torch.from_numpy(arr).to(bark_model.device))
        history = tuple(parts)
    else:
        history = (None, None, None)

    text_temp = float(params.get("temperature") or 0.7)
    waveform_temp = float(params.get("waveform_temperature") or 0.7)

    audio_arr, x_semantic, coarse, fine = bark_model.generate_audio(
        text,
        history_prompt=history,
        text_temp=text_temp,
        waveform_temp=waveform_temp,
    )

    if hasattr(audio_arr, 'numpy'):
        audio_arr = audio_arr.numpy()
    data = np.array(audio_arr, dtype=np.float32)

    # Store updated history for the gateway to retrieve
    global _bark_history_out
    _bark_history_out = (
        x_semantic.cpu().numpy() if hasattr(x_semantic, 'cpu') else np.array(x_semantic),
        coarse.cpu().numpy() if hasattr(coarse, 'cpu') else np.array(coarse),
        fine.cpu().numpy() if hasattr(fine, 'cpu') else np.array(fine),
    )

    return data, 24000


def _infer_chatterbox(text: str, params: dict) -> tuple:
    import numpy as np

    exaggeration = float(params.get("exaggeration") or 0.5)
    cfg_weight = float(params.get("cfg_weight") or 0.5)
    temperature = float(params.get("temperature") or 0.8)
    repetition_penalty = float(params.get("repetition_penalty") or 1.2)

    voice = params.get("voice") or params.get("reference_audio") or ""
    audio_prompt_path = None
    if voice:
        voice_path = VOICE_DIR / voice
        if voice_path.exists():
            audio_prompt_path = str(voice_path)
        elif Path(voice).exists():
            audio_prompt_path = voice

    wav = _model_obj.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    if hasattr(wav, 'cpu'):
        wav = wav.cpu()
    if hasattr(wav, 'numpy'):
        wav = wav.numpy()

    data = np.array(wav, dtype=np.float32).squeeze()
    return data, _model_obj.sr


def _infer_f5(text: str, params: dict) -> tuple:
    import numpy as np

    voice = params.get("voice") or params.get("reference_audio") or ""
    ref_text = params.get("reference_text") or ""
    # Sanitize: JSON null can become string "null" in some serialization paths
    if ref_text.lower() in ("null", "none"):
        ref_text = ""

    ref_file = None
    if voice:
        voice_path = VOICE_DIR / voice
        if voice_path.exists():
            ref_file = str(voice_path)
        elif Path(voice).exists():
            ref_file = voice

    if not ref_file:
        raise ValueError("F5-TTS requires reference audio (set 'voice' to a wav file)")

    speed = float(params.get("speed") or 1.0)
    seed = params.get("seed")
    if seed is not None:
        seed = int(seed)

    logger.info("F5 infer: ref_file=%s, ref_text=%r, gen_text=%r",
                ref_file, ref_text[:50] if ref_text else "(empty - will auto-transcribe)", text[:50])

    wav, sr, _spec = _model_obj.infer(
        ref_file=ref_file, ref_text=ref_text, gen_text=text,
        speed=speed, seed=seed, remove_silence=False,
    )

    data = np.array(wav, dtype=np.float32).squeeze()
    return data, sr


def _infer_dia(text: str, params: dict) -> tuple:
    import re
    import numpy as np
    import torch

    # Auto-fix lowercase dialogue tags: [s1] -> [S1], [s2] -> [S2]
    text = re.sub(r'\[s(\d)\]', lambda m: f'[S{m.group(1)}]', text)

    max_new_tokens = int(params.get("max_new_tokens") or 3072)
    cfg_scale = float(params.get("cfg_scale") or 3.0)
    top_p = float(params.get("top_p") or 0.90)
    top_k = int(params.get("top_k") or 50)
    temperature = float(params.get("temperature") or 1.8)

    voice = params.get("voice") or params.get("reference_audio") or ""
    audio_prompt = None
    if voice:
        voice_path = VOICE_DIR / voice
        if voice_path.exists():
            audio_prompt = str(voice_path)
        elif Path(voice).exists():
            audio_prompt = voice

    generate_kwargs = {
        "text": text,
        "max_tokens": max_new_tokens,
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "top_p": top_p,
        "cfg_filter_top_k": top_k,
    }
    if audio_prompt:
        generate_kwargs["audio_prompt_path"] = audio_prompt

    output = _model_obj.generate(**generate_kwargs, verbose=True)
    data = np.array(output, dtype=np.float32).squeeze()
    duration = len(data) / 44100 if data.ndim > 0 else 0
    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0
    rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100
    logger.info("Dia generate output: shape=%s duration=%.3fs rms=%.1fdB", data.shape, duration, rms_db)
    if duration < 0.5:
        raise RuntimeError(f"Dia generated degenerate audio ({duration:.3f}s) - will retry")
    if rms_db < -40:
        raise RuntimeError(f"Dia generated near-silent audio (rms={rms_db:.1f}dB) - will retry")
    return data, 44100


def _infer_qwen(text: str, params: dict) -> tuple:
    import numpy as np

    model_obj, processor = _model_obj

    speaker = params.get("voice") or "Chelsie"
    temperature = float(params.get("temperature") or 0.9)
    top_p = float(params.get("top_p") or 0.8)
    top_k = int(params.get("top_k") or 40)

    conversation = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Please read the following text aloud exactly as written: {text}",
            }],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(model_obj.device)

    text_ids, audio = model_obj.generate(
        **inputs, speaker=speaker, return_audio=True,
        talker_do_sample=True, talker_temperature=temperature,
        talker_top_p=top_p, talker_top_k=top_k,
    )

    if audio is None:
        raise RuntimeError("Qwen returned no audio output")

    data = audio.reshape(-1).detach().cpu().numpy().astype(np.float32)
    return data, 24000


def _infer_vibevoice(text: str, params: dict) -> tuple:
    import numpy as np
    import torch

    model_obj, processor = _model_obj

    if not text.strip().startswith("Speaker"):
        text = f"Speaker 1: {text}"

    voice = params.get("voice") or params.get("reference_audio") or ""
    voice_samples = []
    if voice:
        voice_path = VOICE_DIR / voice
        if voice_path.exists():
            voice_samples = [str(voice_path)]
        elif Path(voice).exists():
            voice_samples = [voice]

    cfg_scale = float(params.get("cfg_scale") or 1.3)

    inputs = processor(
        text=[text],
        voice_samples=[voice_samples] if voice_samples else None,
        padding=True, return_tensors="pt", return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model_obj.device)

    outputs = model_obj.generate(
        **inputs, cfg_scale=cfg_scale, tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        is_prefill=bool(voice_samples),
    )

    audio_tensor = outputs.speech_outputs[0]
    if hasattr(audio_tensor, "cpu"):
        audio_tensor = audio_tensor.cpu()
    if hasattr(audio_tensor, "float"):
        audio_tensor = audio_tensor.float()  # bfloat16 -> float32 before numpy
    if hasattr(audio_tensor, "numpy"):
        audio_tensor = audio_tensor.numpy()

    data = np.array(audio_tensor, dtype=np.float32).squeeze()
    return data, 24000


def _infer_higgs(text: str, params: dict) -> tuple:
    import numpy as np
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

    temperature = float(params.get("temperature") or 0.3)
    top_p = float(params.get("top_p") or 0.95)
    top_k = int(params.get("top_k") or 50)

    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "Audio is recorded from a quiet room.\n"
        "<|scene_desc_end|>"
    )

    messages = [Message(role="system", content=system_prompt)]

    voice = params.get("voice") or params.get("reference_audio") or ""
    ref_text = params.get("reference_text") or ""
    if voice:
        voice_path = VOICE_DIR / voice
        ref_path = str(voice_path) if voice_path.exists() else voice
        if Path(ref_path).exists():
            if ref_text:
                messages.append(Message(role="user", content=ref_text))
            messages.append(
                Message(role="assistant", content=AudioContent(audio_url=ref_path))
            )

    messages.append(Message(role="user", content=text))

    output = _model_obj.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=2048,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )

    if output.audio is None:
        raise RuntimeError("Higgs returned no audio output")

    data = np.array(output.audio, dtype=np.float32).squeeze()
    sr = output.sampling_rate or 24000
    return data, sr


# ============================================================
# Whisper inference (for whisper worker mode)
# ============================================================
def transcribe(audio_path: str, size: str = "base") -> dict:
    """Transcribe audio using whisper. Only available in whisper worker mode."""
    if _model_name != "whisper":
        raise RuntimeError("This worker is not a whisper worker")

    import whisper

    # Lazy-load the requested size
    if size not in _model_obj:
        logger.info("Loading whisper model size '%s'...", size)
        _model_obj[size] = whisper.load_model(size)
        logger.info("Whisper '%s' loaded", size)

    result = _model_obj[size].transcribe(audio_path)
    return {"text": result.get("text", ""), "language": result.get("language", "")}


# Module-level for bark history passing
_bark_history_out = None


# ============================================================
# FastAPI app
# ============================================================
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

worker_app = FastAPI(title="TTS Worker", version="1.0.0")


class InferRequest(BaseModel):
    text: str
    params: dict = {}


class TranscribeRequest(BaseModel):
    audio_path: str
    size: str = "base"


@worker_app.get("/health")
async def health():
    vram_used = 0
    vram_total = 0
    try:
        import torch
        if torch.cuda.is_available():
            # Get VRAM for the device this worker uses
            dev_idx = 0
            if _device.startswith("cuda:"):
                dev_idx = int(_device.split(":")[1])
            vram_used = torch.cuda.memory_allocated(dev_idx) // (1024 * 1024)
            vram_total = torch.cuda.get_device_properties(dev_idx).total_memory // (1024 * 1024)
    except (ImportError, RuntimeError, AttributeError):
        pass

    return {
        "status": "ready" if _loaded else "idle",
        "model": _model_name,
        "device": _device,
        "pid": os.getpid(),
        "vram_used_mb": vram_used,
        "vram_total_mb": vram_total,
    }


@worker_app.post("/load")
async def api_load():
    try:
        load_model()
        return {"status": "loaded", "model": _model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@worker_app.post("/unload")
async def api_unload():
    unload_model()
    return {"status": "unloaded", "model": _model_name}


@worker_app.post("/infer")
def api_infer(req: InferRequest):
    """Synchronous endpoint — FastAPI runs it in a thread pool,
    keeping the event loop free for /health checks during long inferences."""
    global _bark_history_out
    import numpy as np

    text_preview = req.text[:80] + "..." if len(req.text) > 80 else req.text
    logger.info("Received /infer request: text=%r", text_preview)

    if not _loaded:
        # Auto-load on first inference
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        _bark_history_out = None
        logger.info("Starting inference for model=%s ...", _model_name)
        infer_start = time.time()
        audio_data, sample_rate = infer(req.text, req.params)
        logger.info("Inference complete in %.1fs (sr=%d, samples=%d)",
                    time.time() - infer_start, sample_rate, len(audio_data))

        # Encode numpy array as base64
        buf = io.BytesIO()
        np.save(buf, audio_data)
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = {
            "audio_b64": audio_b64,
            "sample_rate": sample_rate,
            "dtype": str(audio_data.dtype),
        }

        # For bark, include updated history prompt
        if _model_name == "bark" and _bark_history_out is not None:
            history_parts = []
            for arr in _bark_history_out:
                hbuf = io.BytesIO()
                np.save(hbuf, np.array(arr) if not isinstance(arr, np.ndarray) else arr)
                history_parts.append(base64.b64encode(hbuf.getvalue()).decode("utf-8"))
            response["bark_history"] = history_parts

        return JSONResponse(response)

    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@worker_app.post("/transcribe")
async def api_transcribe(req: TranscribeRequest):
    """Whisper transcription endpoint (only for whisper workers)."""
    if _model_name != "whisper":
        raise HTTPException(status_code=400, detail="Not a whisper worker")

    if not _loaded:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load whisper: {e}")

    try:
        result = transcribe(req.audio_path, req.size)
        return result
    except Exception as e:
        logger.error("Transcribe error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main entry point
# ============================================================
def main():
    global _model_name, _device

    parser = argparse.ArgumentParser(description="TTS Worker Server")
    parser.add_argument("--model", required=True, help="Model to load (kokoro, xtts, etc.)")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (cuda:0, cuda:1, cpu)")
    parser.add_argument("--preload", action="store_true", help="Load model immediately on startup")
    args = parser.parse_args()

    _model_name = args.model
    _device = args.device

    # Inject the model's venv into sys.path
    _inject_venv(_model_name)

    logger.info("Starting worker: model=%s port=%d device=%s", _model_name, args.port, _device)

    if args.preload:
        load_model()

    import uvicorn
    uvicorn.run(worker_app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
