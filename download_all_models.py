"""
Download all TTS models to tts_models directory using HuggingFace Hub.
Run this script to pre-download all model weights.
"""

import os
import sys
from pathlib import Path

# Try to import from config, fall back to defaults
try:
    from config import MODELS_DIR, setup_environment
    setup_environment()
    MODELS_DIR = str(MODELS_DIR)
except ImportError:
    # Fallback if config not available
    MODELS_DIR = str(Path(__file__).parent / "tts_models")
    os.environ["HF_HOME"] = MODELS_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(MODELS_DIR, "hub")
    os.environ["TORCH_HOME"] = os.path.join(MODELS_DIR, "torch")
    os.environ["COQUI_TTS_CACHE"] = os.path.join(MODELS_DIR, "coqui")

from huggingface_hub import snapshot_download, hf_hub_download
import argparse

# Model registry - (repo_id, description, optional specific files)
MODELS = {
    # Chatterbox
    "chatterbox": {
        "repo": "ResembleAI/chatterbox",
        "desc": "Chatterbox TTS - emotion control, voice cloning",
    },

    # F5-TTS
    "f5-tts": {
        "repo": "SWivid/F5-TTS",
        "desc": "F5-TTS - SOTA diffusion TTS",
    },

    # Coqui XTTS v2
    "xtts-v2": {
        "repo": "coqui/XTTS-v2",
        "desc": "XTTS v2 - multilingual voice cloning",
    },

    # Qwen2.5-Omni - excluded (user has other plans for this model)
    # Orpheus - excluded (gated model, requires HF login)

    # Fish Speech
    "fish-speech": {
        "repo": "fishaudio/fish-speech-1.5",
        "desc": "Fish Speech 1.5 - fast TTS",
    },

    # Kokoro
    "kokoro": {
        "repo": "hexgrad/Kokoro-82M",
        "desc": "Kokoro 82M - lightweight TTS",
    },

    # Dia
    "dia": {
        "repo": "nari-labs/Dia-1.6B-0626",
        "desc": "Dia 1.6B-0626 - dialogue TTS",
    },

    # Bark (via Suno)
    "bark": {
        "repo": "suno/bark",
        "desc": "Bark - expressive TTS with laughter, music",
    },

    # OpenVoice
    "openvoice-v2": {
        "repo": "myshell-ai/OpenVoiceV2",
        "desc": "OpenVoice V2 - instant voice cloning",
    },
}


def download_model(model_key: str, models_dir: str = MODELS_DIR):
    """Download a single model."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return False

    info = MODELS[model_key]
    repo = info["repo"]
    desc = info["desc"]

    print(f"\n{'='*60}")
    print(f"Downloading: {model_key}")
    print(f"Repo: {repo}")
    print(f"Description: {desc}")
    print(f"{'='*60}")

    try:
        local_dir = snapshot_download(
            repo_id=repo,
            cache_dir=os.path.join(models_dir, "hub"),
            local_dir=os.path.join(models_dir, model_key),
            local_dir_use_symlinks=False,  # Windows compatibility
        )
        print(f"Downloaded to: {local_dir}")
        return True
    except Exception as e:
        print(f"ERROR downloading {model_key}: {e}")
        return False


def download_all(models_dir: str = MODELS_DIR):
    """Download all models."""
    os.makedirs(models_dir, exist_ok=True)

    print(f"Downloading all models to: {models_dir}")
    print(f"Total models: {len(MODELS)}")

    results = {}
    for key in MODELS:
        results[key] = download_model(key, models_dir)

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {key}: {status}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\nFailed downloads: {', '.join(failed)}")
        return False
    else:
        print("\nAll models downloaded successfully!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TTS models")
    parser.add_argument("--model", "-m", help="Specific model to download (or 'all')")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for key, info in MODELS.items():
            print(f"  {key}: {info['desc']}")
        sys.exit(0)

    if args.model and args.model != "all":
        success = download_model(args.model)
        sys.exit(0 if success else 1)
    else:
        success = download_all()
        sys.exit(0 if success else 1)
