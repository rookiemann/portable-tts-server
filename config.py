"""
TTS Module Configuration

Central configuration for the TTS Environment Manager with dynamic path resolution.
Supports embedded Python for portable installs.
"""
import shutil
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
VENVS_DIR = BASE_DIR / "venvs"
MODELS_DIR = BASE_DIR / "tts_models"
OUTPUT_DIR = BASE_DIR / "output"

# Environment paths
PYTHON_EMBEDDED_DIR = BASE_DIR / "python_embedded"
GIT_PORTABLE_DIR = BASE_DIR / "git_portable"
FFMPEG_DIR = BASE_DIR / "ffmpeg"


# Dynamic Python path resolution
def _resolve_python_path() -> Path:
    """Find the best available Python executable.

    Priority: embedded Python > system Python
    """
    # 1. Embedded Python (preferred for portable installs)
    embedded = PYTHON_EMBEDDED_DIR / "python.exe"
    if embedded.exists():
        return embedded

    # 2. System Python (last resort)
    system_python = shutil.which("python")
    if system_python:
        return Path(system_python)

    # 3. Return embedded path even if not yet downloaded
    #    (install.bat will create it)
    return embedded


def _resolve_git_path() -> str:
    """Find the best available git executable.

    Priority: portable Git > system Git
    """
    portable_git = GIT_PORTABLE_DIR / "cmd" / "git.exe"
    if portable_git.exists():
        return str(portable_git)
    return "git"


def _resolve_ffmpeg_path() -> str:
    """Find the best available ffmpeg executable.

    Priority: local ffmpeg (multiple possible locations) > system FFmpeg
    """
    # Check our ffmpeg directory for various structures
    possible_paths = [
        FFMPEG_DIR / "ffmpeg.exe",  # Direct in ffmpeg folder
        FFMPEG_DIR / "bin" / "ffmpeg.exe",  # In bin subfolder
    ]

    # Also check for extracted builds with version in name (e.g., ffmpeg-8.0.1-essentials_build)
    if FFMPEG_DIR.exists():
        for subdir in FFMPEG_DIR.iterdir():
            if subdir.is_dir():
                possible_paths.append(subdir / "bin" / "ffmpeg.exe")
                possible_paths.append(subdir / "ffmpeg.exe")

    for path in possible_paths:
        if path.exists():
            return str(path)

    return "ffmpeg"


def _get_ffmpeg_bin_dir() -> Path | None:
    """Get the directory containing ffmpeg binaries."""
    ffmpeg_path = _resolve_ffmpeg_path()
    if ffmpeg_path != "ffmpeg":
        return Path(ffmpeg_path).parent
    return None


PYTHON_PATH = _resolve_python_path()
GIT_PATH = _resolve_git_path()
FFMPEG_PATH = _resolve_ffmpeg_path()
FFMPEG_BIN_DIR = _get_ffmpeg_bin_dir()

# Whether we're running with embedded Python
USE_EMBEDDED = (PYTHON_EMBEDDED_DIR / "python.exe").exists()

# Set up environment variables
def setup_environment():
    """Configure environment variables for HuggingFace and FFmpeg."""
    # HuggingFace cache directories
    os.environ["HF_HOME"] = str(MODELS_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "hub")
    os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
    os.environ["COQUI_TTS_CACHE"] = str(MODELS_DIR / "coqui")

    # Add FFmpeg to PATH if available
    if FFMPEG_BIN_DIR:
        current_path = os.environ.get("PATH", "")
        if str(FFMPEG_BIN_DIR) not in current_path:
            os.environ["PATH"] = str(FFMPEG_BIN_DIR) + os.pathsep + current_path

    # Add portable Git to PATH if available
    if GIT_PORTABLE_DIR.exists():
        git_cmd_dir = GIT_PORTABLE_DIR / "cmd"
        if git_cmd_dir.exists():
            current_path = os.environ.get("PATH", "")
            if str(git_cmd_dir) not in current_path:
                os.environ["PATH"] = str(git_cmd_dir) + os.pathsep + current_path

    # Add Rubberband to PATH if available (for tempo adjustment)
    rubberband_dir = BASE_DIR / "rubberband"
    if rubberband_dir.exists():
        rubberband_bin = None
        if (rubberband_dir / "rubberband.exe").exists():
            rubberband_bin = rubberband_dir
        else:
            for sub in rubberband_dir.iterdir():
                if sub.is_dir() and (sub / "rubberband.exe").exists():
                    rubberband_bin = sub
                    break
        if rubberband_bin:
            current_path = os.environ.get("PATH", "")
            if str(rubberband_bin) not in current_path:
                os.environ["PATH"] = str(rubberband_bin) + os.pathsep + current_path

    # Add eSpeak NG to PATH if available (required for Kokoro)
    espeak_dir = BASE_DIR / "espeak_ng"
    if espeak_dir.exists():
        # Check for direct install or MSI-extracted subdirectory
        espeak_bin = None
        if (espeak_dir / "espeak-ng.exe").exists():
            espeak_bin = espeak_dir
        else:
            for sub in espeak_dir.iterdir():
                if sub.is_dir() and (sub / "espeak-ng.exe").exists():
                    espeak_bin = sub
                    break
        if espeak_bin:
            current_path = os.environ.get("PATH", "")
            if str(espeak_bin) not in current_path:
                os.environ["PATH"] = str(espeak_bin) + os.pathsep + current_path
            data_path = espeak_bin / "espeak-ng-data"
            if data_path.exists():
                os.environ["ESPEAK_DATA_PATH"] = str(data_path)


# UI Settings
WINDOW_TITLE = "TTS Module - Environment Manager"
WINDOW_SIZE = "1100x800"
APP_VERSION = "1.0.0"

# API Server settings
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8100

# Job management
JOBS_DIR = OUTPUT_DIR / "jobs"
PROJECTS_OUTPUT = BASE_DIR / "projects_output"
VOICE_DIR = BASE_DIR / "voices"
MAX_RETRIES = 3
MAX_JOB_AGE_HOURS = 72

# Whisper verification
WHISPER_MODEL_SIZE = "base"
WHISPER_ENABLED = True
WHISPER_DEFAULT_TOLERANCE = 80.0
WHISPER_AVAILABLE_MODELS = {
    "tiny":   {"params": "39M",  "vram": "~1GB",  "speed": "~10x", "note": "Fastest, least accurate"},
    "base":   {"params": "74M",  "vram": "~1GB",  "speed": "~7x",  "note": "Good balance (default)"},
    "small":  {"params": "244M", "vram": "~2GB",  "speed": "~4x",  "note": "Better accuracy"},
    "medium": {"params": "769M", "vram": "~5GB",  "speed": "~2x",  "note": "Near-best accuracy"},
    "large":  {"params": "1550M","vram": "~10GB", "speed": "~1x",  "note": "Best accuracy, slowest"},
}

# Server
MAX_INFERENCE_WORKERS = 2

# Worker management
WORKER_PORT_MIN = 8101
WORKER_PORT_MAX = 8200
WORKER_HEALTH_INTERVAL = 10  # seconds
WORKER_STARTUP_TIMEOUT = 120  # seconds (models can take time to load)
WORKER_MAX_HEALTH_FAILURES = 3
WORKER_AUTO_SPAWN = True  # auto-spawn a worker when request comes for unloaded model
WORKER_LOG_DIR = OUTPUT_DIR / "logs"
WORKER_DEFAULT_DEVICE = "cuda:0"  # default GPU for auto-spawned workers

# Model -> venv mapping (single source of truth, used by both gateway and workers)
MODEL_VENV_MAP = {
    "xtts": "coqui_env",
    "bark": "coqui_env",
    "fish": "unified_env",
    "kokoro": "unified_env",
    "dia": "unified_env",
    "chatterbox": "chatterbox_env",
    "f5": "f5tts_env",
    "qwen": "qwen3_env",
    "vibevoice": "vibevoice_env",
    "higgs": "higgs_env",
    "whisper": "unified_env",
}
