"""
TTS Environment Manager - Tkinter GUI Application

Manages multiple TTS model virtual environments, model downloads,
and API server for TTS inference.
"""

import os
import sys
from pathlib import Path

# Bootstrap: ensure project root is on sys.path for sibling imports
_BASE_DIR = Path(__file__).parent.resolve()
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import queue
import json
import shutil
import base64
import tempfile
import winsound
import requests
from collections import deque

# Import configuration
from config import (
    BASE_DIR, VENVS_DIR, MODELS_DIR, OUTPUT_DIR,
    PYTHON_PATH, GIT_PATH, FFMPEG_PATH, FFMPEG_BIN_DIR,
    PYTHON_EMBEDDED_DIR, USE_EMBEDDED,
    WINDOW_TITLE, WINDOW_SIZE, APP_VERSION,
    DEFAULT_API_HOST, DEFAULT_API_PORT,
    setup_environment
)

# Set up environment (HuggingFace cache, FFmpeg PATH, etc.)
setup_environment()

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Add install_configs to path
sys.path.insert(0, str(BASE_DIR))

from install_configs import ALL_CONFIGS
from install_configs.base import run_pip_install, run_git_clone, check_package_installed


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------
class _ToolTip:
    """Hover tooltip for Tkinter widgets."""

    def __init__(self, widget, text: str, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def _show(self):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw, text=self.text, background="#ffffe0", foreground="#333",
            relief=tk.SOLID, borderwidth=1, font=("Segoe UI", 9),
            wraplength=350, justify=tk.LEFT, padx=6, pady=4,
        )
        lbl.pack()


def _tip(widget, text: str):
    """Attach a tooltip to *widget* and return it (for call-chaining)."""
    _ToolTip(widget, text)
    return widget


# Unified model setup - maps each model to its venv and weights
# weights_repo=None means auto-download on first use
MODEL_SETUP = {
    "bark":       {"display": "Bark",           "desc": "Expressive TTS - laughter, music, emotions",  "env": "coqui_env",      "weights_repo": "suno/bark",                "weights_dir": "bark",        "weights_size": "~5GB",    "gated": False},
    "chatterbox": {"display": "Chatterbox",     "desc": "Emotion control, voice cloning",              "env": "chatterbox_env", "weights_repo": "ResembleAI/chatterbox",    "weights_dir": "chatterbox",  "weights_size": "~2GB",    "gated": False},
    "dia":        {"display": "Dia 1.6B",       "desc": "Dialogue TTS with [S1]/[S2] speaker tags",    "env": "unified_env",    "weights_repo": "nari-labs/Dia-1.6B-0626",  "weights_dir": "dia",         "weights_size": "~6GB",    "gated": False},
    "f5":         {"display": "F5-TTS",         "desc": "Diffusion TTS, reference audio cloning",      "env": "f5tts_env",      "weights_repo": "SWivid/F5-TTS",            "weights_dir": "f5-tts",      "weights_size": "~1.5GB",  "gated": False},
    "fish":       {"display": "Fish Speech 1.5","desc": "Fast TTS with voice cloning",                "env": "unified_env",    "weights_repo": "fishaudio/fish-speech-1.5","weights_dir": "fish-speech", "weights_size": "~1GB",    "gated": False},
    "higgs":      {"display": "Higgs Audio 3B", "desc": "Boson AI ChatML (CPU supported)",             "env": "higgs_env",      "weights_repo": None,                       "weights_dir": None,          "weights_size": None,      "gated": False},
    "kokoro":     {"display": "Kokoro 82M",     "desc": "Lightweight, fast, 54 built-in voices",       "env": "unified_env",    "weights_repo": "hexgrad/Kokoro-82M",       "weights_dir": "kokoro",      "weights_size": "~300MB",  "gated": False},
    "qwen":       {"display": "Qwen Omni 7B",   "desc": "Multimodal with speech output",               "env": "qwen3_env",      "weights_repo": None,                       "weights_dir": None,          "weights_size": None,      "gated": False},
    "vibevoice":  {"display": "VibeVoice",      "desc": "Speaker-conditioned TTS",                    "env": "vibevoice_env",  "weights_repo": None,                       "weights_dir": None,          "weights_size": None,      "gated": False},
    "whisper":    {"display": "Whisper",        "desc": "Speech recognition for verification",        "env": "unified_env",    "weights_repo": None,                       "weights_dir": None,          "weights_size": None,      "gated": False},
    "xtts":       {"display": "XTTS v2",        "desc": "Multilingual voice cloning, 58 built-in voices","env": "coqui_env",    "weights_repo": "coqui/XTTS-v2",            "weights_dir": "xtts-v2",     "weights_size": "~1.8GB",  "gated": False},
}

# Per-model recommended defaults for common + model-specific params
MODEL_DEFAULTS = {
    "xtts":       {"temperature": 0.65, "speed": 1.0, "repetition_penalty": 2.0},
    "fish":       {"temperature": 0.8, "repetition_penalty": 1.1, "top_p": 0.8},
    "kokoro":     {"speed": 1.0},
    "bark":       {"temperature": 0.7, "waveform_temperature": 0.7},
    "chatterbox": {"temperature": 0.8, "repetition_penalty": 1.2, "exaggeration": 0.5, "cfg_weight": 0.5},
    "f5":         {"speed": 1.0},
    "dia":        {"temperature": 1.8, "cfg_scale": 3.0, "top_p": 0.90, "top_k": 50},
    "qwen":       {"temperature": 0.9, "top_p": 0.8, "top_k": 40},
    "vibevoice":  {"cfg_scale": 1.3},
    "higgs":      {"temperature": 0.3, "top_p": 0.95, "top_k": 50},
}

# Common params already shown in the main Inference frame
_COMMON_PARAMS = {"temperature", "speed", "repetition_penalty"}

# Spinbox config: param -> (label, from_, to, increment, width, format_str)
PARAM_SPINBOX = {
    "temperature":          ("Temp",         0.0,  3.0,  0.05, 5, "%.2f"),
    "speed":                ("Speed",        0.5,  2.0,  0.1,  5, "%.1f"),
    "repetition_penalty":   ("Rep.Pen",      0.5,  5.0,  0.1,  5, "%.1f"),
    "top_p":                ("Top P",        0.0,  1.0,  0.05, 5, "%.2f"),
    "top_k":                ("Top K",        1,    200,  1,    5, "%d"),
    "cfg_scale":            ("CFG Scale",    0.0,  10.0, 0.5,  5, "%.1f"),
    "exaggeration":         ("Exaggeration", 0.0,  1.0,  0.05, 5, "%.2f"),
    "cfg_weight":           ("CFG Weight",   0.0,  1.0,  0.05, 5, "%.2f"),
    "waveform_temperature": ("Wave Temp",    0.0,  3.0,  0.05, 5, "%.2f"),
    "de_reverb":            ("De-reverb",    0.0,  1.0,  0.1,  5, "%.1f"),
    "de_ess":               ("De-ess",       0.0,  1.0,  0.1,  5, "%.1f"),
    "tolerance":            ("Tolerance",    0,    100,  5,    5, "%d"),
}

# ---------------------------------------------------------------------------
# Tooltip descriptions for every interactive control
# ---------------------------------------------------------------------------
TOOLTIPS = {
    # -- Inference params --
    "speed": (
        "Playback speed multiplier.\n"
        "1.0 = normal, 0.5 = half speed, 2.0 = double speed.\n"
        "Only affects models that support speed control (XTTS, Kokoro, F5)."
    ),
    "temperature": (
        "Controls randomness in speech generation.\n"
        "Lower = more consistent and predictable.\n"
        "Higher = more varied and expressive.\n"
        "Default is set automatically when you select a worker."
    ),
    "repetition_penalty": (
        "Penalizes the model for repeating the same words or sounds.\n"
        "Higher = less repetition but can sound less natural.\n"
        "Default varies per model (e.g. XTTS: 2.0, Fish: 1.1)."
    ),

    # -- Post-processing --
    "de_reverb": (
        "Removes room reverb / echo from generated audio.\n"
        "0.0 = no removal, 1.0 = maximum removal.\n"
        "Useful for cleaner, drier speech."
    ),
    "de_ess": (
        "Reduces harsh sibilance (sharp 's' and 'sh' sounds).\n"
        "0.0 = no de-essing, 1.0 = maximum reduction.\n"
        "Useful if the generated voice sounds too hissy."
    ),
    "format": (
        "Output audio file format.\n"
        "WAV = lossless (largest file), MP3 = compressed.\n"
        "OGG / FLAC = alternative formats. WAV recommended for quality."
    ),

    # -- Model-specific params --
    "top_p": (
        "Nucleus sampling \u2014 considers only the most likely tokens\n"
        "that together make up this cumulative probability.\n"
        "Lower = more focused output, Higher = more diverse.\n"
        "0.9 is a common balanced value."
    ),
    "top_k": (
        "Limits token selection to the top K most likely options.\n"
        "Lower = more predictable, Higher = more variety.\n"
        "Works alongside Top P for fine-grained sampling control."
    ),
    "cfg_scale": (
        "Classifier-Free Guidance strength.\n"
        "Higher = output follows the text prompt more closely.\n"
        "Lower = more creative / varied output.\n"
        "Dia default: 3.0, VibeVoice default: 1.3."
    ),
    "exaggeration": (
        "Controls emotional intensity (Chatterbox only).\n"
        "0.0 = neutral, flat delivery.\n"
        "1.0 = maximum emotional expression.\n"
        "Try 0.3\u20130.6 for natural-sounding emotion."
    ),
    "cfg_weight": (
        "Classifier-Free Guidance weight (Chatterbox only).\n"
        "Controls how strongly the model follows its conditioning.\n"
        "0.5 is a balanced default. Lower = more creative."
    ),
    "waveform_temperature": (
        "Controls diversity in Bark's waveform generation stage.\n"
        "Bark has two stages: semantic tokens (Temp) and\n"
        "acoustic waveform (Wave Temp).\n"
        "Higher = more varied audio texture."
    ),

    # -- Whisper & options --
    "verify_whisper": (
        "Run Whisper speech-to-text on the generated audio to\n"
        "verify it matches your input text. Chunks that fail\n"
        "verification will be automatically retried."
    ),
    "whisper_model": (
        "Whisper model size for verification.\n"
        "Tiny = fastest but least accurate.\n"
        "Base = good balance (recommended).\n"
        "Larger = slower but more accurate transcription."
    ),
    "tolerance": (
        "Minimum text similarity % for Whisper verification.\n"
        "80 = allows minor word differences (recommended).\n"
        "95 = very strict, rejects most paraphrasing.\n"
        "Chunks scoring below this are retried."
    ),
    "skip_post_process": (
        "Skip all audio post-processing (de-reverb, de-ess,\n"
        "trimming, loudness normalization, peak limiting).\n"
        "Outputs raw audio directly from the model.\n"
        "Useful for testing or when you want unprocessed output."
    ),

    # -- Voice section --
    "voice_dropdown": (
        "Select a built-in voice for this model.\n"
        "Each model has its own set of voices with\n"
        "different pitch, tone, and accent characteristics."
    ),
    "ref_audio": (
        "Path to a reference audio file for voice cloning.\n"
        "The model will match the speaker's voice from this sample.\n"
        "Best results: 5\u201315 seconds of clear speech, no background noise."
    ),
    "ref_audio_optional": (
        "Optional reference audio for voice cloning.\n"
        "If left empty, the model uses its default voice.\n"
        "Provide a WAV/MP3 with clear speech to clone a specific voice."
    ),
    "ref_text": (
        "Transcript of the reference audio.\n"
        "Must exactly match what is spoken in the audio file.\n"
        "Required by F5-TTS and Higgs to align the cloned voice."
    ),

    # -- Text input --
    "text_input": (
        "Enter the text you want converted to speech.\n"
        "Long text is automatically split into chunks sized for\n"
        "the selected model. Dia supports [S1] / [S2] tags for\n"
        "two-speaker dialogue."
    ),

    # -- Workers & API Server --
    "workers_tree": (
        "Active TTS workers. Click a row to select it for testing.\n"
        "Each worker runs one model instance on one GPU.\n"
        "Spawn workers from the API Server tab."
    ),
    "spawn_model": (
        "Select which TTS model to load.\n"
        "Each model has different strengths \u2014 hover over\n"
        "models in the Setup tab for details.\n"
        "Weights are downloaded on first use if not present."
    ),
    "spawn_device": (
        "GPU to load the model on.\n"
        "cuda:0 / cuda:1 = specific GPU, cpu = CPU (slow).\n"
        "Large models (Qwen 7B, Dia 1.6B) need 8\u201316 GB VRAM."
    ),
    "spawn_button": (
        "Start a new worker process for the selected model.\n"
        "Loads model weights into GPU memory (may take 10\u201360s).\n"
        "You can spawn multiple workers for the same model."
    ),
    "kill_selected": (
        "Terminate the selected worker and free its GPU memory.\n"
        "The worker process is killed and its port released."
    ),
    "kill_all": (
        "Terminate ALL running workers and free all GPU memory.\n"
        "Useful for freeing VRAM before loading different models."
    ),
    "restart_gateway": (
        "Restart the FastAPI gateway server.\n"
        "Workers continue running \u2014 only the gateway restarts.\n"
        "Use if the gateway becomes unresponsive."
    ),
    "port_entry": (
        "Gateway API port (default 8100).\n"
        "Change if port 8100 is already in use.\n"
        "Restart the gateway after changing."
    ),

    # -- Action buttons --
    "send_request": (
        "Send the text to the selected worker for TTS.\n"
        "Long text is automatically split into chunks.\n"
        "Results appear in Response History below."
    ),
    "play_button": "Play the selected audio through your default audio device.",
    "stop_button": "Stop audio playback.",
    "save_button": "Save the selected audio result to a file.",
    "clear_history": "Clear all entries from the response history.",
}


class TTSEnvironment:
    """Represents a single TTS environment."""

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.venv_path = VENVS_DIR / name
        self.status = "unknown"
        self.error_message = ""

    @property
    def python_path(self) -> Path:
        return self.venv_path / "Scripts" / "python.exe"

    @property
    def pip_path(self) -> Path:
        return self.venv_path / "Scripts" / "pip.exe"

    @property
    def activate_script(self) -> Path:
        return self.venv_path / "Scripts" / "activate.bat"

    def check_status(self) -> str:
        """Check the current status of this environment."""
        if not self.venv_path.exists():
            self.status = "not_installed"
            return self.status

        if not self.python_path.exists():
            self.status = "not_installed"
            return self.status

        if self.config.verify_package:
            if check_package_installed(str(self.pip_path), self.config.verify_package):
                self.status = "installed"
            else:
                self.status = "partial"
        else:
            self.status = "installed"

        return self.status


class LogRedirector:
    """Redirects log messages to a queue for thread-safe GUI updates."""

    def __init__(self, log_queue: queue.Queue):
        self.queue = log_queue

    def write(self, msg: str):
        if msg.strip():
            self.queue.put(msg)

    def flush(self):
        pass


class TTSManagerApp:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TTS Environment Manager")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)

        # State
        self.environments = {}
        self.log_queue = queue.Queue()
        self.current_installation = None
        self._cached_devices = []
        self._test_responses = []  # response history with audio paths
        self._log_entries = deque(maxlen=1000)  # ring buffer for log
        self._server_poll_id = None
        self._gateway_process = None
        self._gateway_port = DEFAULT_API_PORT

        # Initialize environments
        self._init_environments()

        # Build GUI
        self._build_gui()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start log queue processor
        self._process_log_queue()

        # Detect GPUs locally (no server needed)
        self._detect_local_devices()

        # Auto-start gateway
        self._start_gateway()

        # Start polling for gateway connection (delayed so mainloop is running)
        self.root.after(2000, self._start_server_polling)

        # Initial status check
        self._refresh_setup_status()

    def _init_environments(self):
        """Initialize environment objects from configs."""
        for name, config in ALL_CONFIGS.items():
            self.environments[name] = TTSEnvironment(name, config)

    def _build_gui(self):
        """Build the main GUI with tabs."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="TTS Environment Manager",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Setup
        self.setup_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.setup_tab, text="Setup")
        self._build_setup_tab()

        # Tab 2: API Server
        self.api_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.api_tab, text="API Server")
        self._build_api_tab()

        # Tab 3: Testing
        self.test_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.test_tab, text="Testing")
        self._build_test_tab()

        # Tab 4: Log
        self.log_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.log_tab, text="Log")
        self._build_log_tab()

    def _build_setup_tab(self):
        """Build the unified setup tab - models with their environments and weights."""
        # Info
        python_mode = "Embedded" if USE_EMBEDDED else "System"
        info_label = ttk.Label(
            self.setup_tab,
            text=f"Python ({python_mode}): {PYTHON_PATH}  |  Venvs: {VENVS_DIR}  |  Models: {MODELS_DIR}",
            font=("Segoe UI", 8), foreground="gray"
        )
        info_label.pack(anchor=tk.W, pady=(0, 5))

        # Model list (no canvas - 12 rows fits fine)
        models_frame = ttk.LabelFrame(self.setup_tab, text="Models", padding="5")
        models_frame.pack(fill=tk.X)

        self.setup_widgets = {}
        for model_id, info in MODEL_SETUP.items():
            self._create_setup_row(models_frame, model_id, info)

        # Actions frame
        actions_frame = ttk.Frame(self.setup_tab)
        actions_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(actions_frame, text="Install All", command=self._install_all_models).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="Refresh Status", command=self._refresh_setup_status).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="Open Venvs Folder", command=self._open_venvs_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="Open Models Folder", command=lambda: os.startfile(MODELS_DIR)).pack(side=tk.LEFT)

    def _build_log_tab(self):
        """Build the log tab with category filters."""
        # Filter row
        filter_frame = ttk.Frame(self.log_tab)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Show:").pack(side=tk.LEFT, padx=(0, 5))
        self._log_filters = {}
        for tag, label, default in [
            ("info", "Info", True), ("success", "Success", True),
            ("error", "Errors", True), ("warning", "Warnings", True),
            ("gateway", "Gateway", True),
        ]:
            var = tk.BooleanVar(value=default)
            self._log_filters[tag] = var
            ttk.Checkbutton(
                filter_frame, text=label, variable=var,
                command=self._rerender_log
            ).pack(side=tk.LEFT, padx=(0, 8))

        # Log text
        self.log_text = scrolledtext.ScrolledText(
            self.log_tab, font=("Consolas", 9), state=tk.DISABLED, wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("success", foreground="green")
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("gateway", foreground="#666666")

        # Bottom row
        bottom = ttk.Frame(self.log_tab)
        bottom.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(
            bottom, text="Last 1000 entries", foreground="gray",
            font=("Segoe UI", 8)
        ).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)

    def _build_api_tab(self):
        """Build the API server tab - gateway, workers, spawn."""
        # Gateway bar
        gw_frame = ttk.LabelFrame(self.api_tab, text="Gateway", padding="5")
        gw_frame.pack(fill=tk.X, pady=(0, 5))

        gw_row = ttk.Frame(gw_frame)
        gw_row.pack(fill=tk.X)

        self.server_status_canvas = tk.Canvas(gw_row, width=16, height=16, highlightthickness=0)
        self.server_status_canvas.pack(side=tk.LEFT, padx=(0, 5))
        self.server_status_circle = self.server_status_canvas.create_oval(2, 2, 14, 14, fill="gray", outline="")

        self.server_status_label = ttk.Label(gw_row, text="Starting...", font=("Segoe UI", 9))
        self.server_status_label.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(gw_row, text="Port:").pack(side=tk.LEFT, padx=(0, 3))
        self.port_var = tk.StringVar(value=str(DEFAULT_API_PORT))
        self.port_entry = ttk.Entry(gw_row, textvariable=self.port_var, width=6)
        self.port_entry.pack(side=tk.LEFT, padx=(0, 10))
        _tip(self.port_entry, TOOLTIPS["port_entry"])

        rg = ttk.Button(gw_row, text="Restart Gateway", command=self._restart_gateway)
        rg.pack(side=tk.LEFT, padx=(0, 10))
        _tip(rg, TOOLTIPS["restart_gateway"])
        ka = ttk.Button(gw_row, text="Kill All Workers", command=self._kill_all_workers)
        ka.pack(side=tk.RIGHT)
        _tip(ka, TOOLTIPS["kill_all"])

        # Workers section
        workers_frame = ttk.LabelFrame(self.api_tab, text="Workers", padding="5")
        workers_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        workers_btn_row = ttk.Frame(workers_frame)
        workers_btn_row.pack(fill=tk.X, pady=(0, 5))
        ks = ttk.Button(workers_btn_row, text="Kill Selected", command=self._kill_selected_worker)
        ks.pack(side=tk.RIGHT)
        _tip(ks, TOOLTIPS["kill_selected"])

        wrk_cols = ("worker_id", "model", "port", "device", "status")
        self.workers_tree = ttk.Treeview(workers_frame, columns=wrk_cols, show="headings", height=8)
        self.workers_tree.heading("worker_id", text="Worker ID")
        self.workers_tree.heading("model", text="Model")
        self.workers_tree.heading("port", text="Port")
        self.workers_tree.heading("device", text="Device")
        self.workers_tree.heading("status", text="Status")
        self.workers_tree.column("worker_id", width=120, minwidth=80)
        self.workers_tree.column("model", width=100, minwidth=70)
        self.workers_tree.column("port", width=70, minwidth=50)
        self.workers_tree.column("device", width=90, minwidth=60)
        self.workers_tree.column("status", width=80, minwidth=60)

        workers_scroll = ttk.Scrollbar(workers_frame, orient="vertical", command=self.workers_tree.yview)
        self.workers_tree.configure(yscrollcommand=workers_scroll.set)
        self.workers_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        workers_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Spawn section
        spawn_frame = ttk.LabelFrame(self.api_tab, text="Spawn Worker", padding="5")
        spawn_frame.pack(fill=tk.X)

        spawn_row = ttk.Frame(spawn_frame)
        spawn_row.pack(fill=tk.X)

        ttk.Label(spawn_row, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.spawn_model_var = tk.StringVar(value="kokoro")
        self.spawn_model_combo = ttk.Combobox(
            spawn_row, textvariable=self.spawn_model_var, state="readonly", width=14,
            values=["bark", "chatterbox", "dia", "f5", "fish", "higgs", "kokoro",
                    "qwen", "vibevoice", "whisper", "xtts"]
        )
        self.spawn_model_combo.pack(side=tk.LEFT, padx=(0, 15))
        _tip(self.spawn_model_combo, TOOLTIPS["spawn_model"])

        ttk.Label(spawn_row, text="Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.spawn_device_var = tk.StringVar()
        self.spawn_device_combo = ttk.Combobox(
            spawn_row, textvariable=self.spawn_device_var, state="readonly", width=14
        )
        self.spawn_device_combo.pack(side=tk.LEFT, padx=(0, 15))
        _tip(self.spawn_device_combo, TOOLTIPS["spawn_device"])

        sp_btn = ttk.Button(spawn_row, text="Spawn", command=self._spawn_worker)
        sp_btn.pack(side=tk.LEFT)
        _tip(sp_btn, TOOLTIPS["spawn_button"])

    # ============================================================
    # API Server Tab Actions
    # ============================================================
    def _update_device_combos(self, devices: list):
        """Update cached device list and spawn device combo."""
        self._cached_devices = devices
        device_ids = [d["id"] for d in devices]
        if not device_ids:
            device_ids = ["cuda:0", "cpu"]
        self.spawn_device_combo.config(values=device_ids)
        if not self.spawn_device_var.get() and device_ids:
            self.spawn_device_var.set(device_ids[0])

    def _refresh_workers(self):
        """Refresh the workers treeview."""
        def do_fetch():
            data = self._api_get("/api/workers")
            if data and "workers" in data:
                self.root.after(0, lambda: self._update_workers_display(data["workers"]))
            else:
                self.root.after(0, lambda: self.log("Could not fetch workers. Is the server running?", tag="warning"))

        thread = threading.Thread(target=do_fetch, daemon=True)
        thread.start()

    def _update_workers_display(self, workers: list):
        """Update workers treeview in-place (no flicker)."""
        self._update_tree_inplace(self.workers_tree, workers)

    def _update_tree_inplace(self, tree: ttk.Treeview, workers: list):
        """Update a treeview with worker data without clearing it."""
        incoming_ids = set()
        for w in workers:
            wid = w["worker_id"]
            incoming_ids.add(wid)
            values = (wid, w["model"], w["port"], w["device"], w["status"])
            if tree.exists(wid):
                tree.item(wid, values=values)
            else:
                tree.insert("", tk.END, iid=wid, values=values)
        # Remove workers no longer present
        for existing_id in tree.get_children():
            if existing_id not in incoming_ids:
                tree.delete(existing_id)

    def _kill_selected_worker(self):
        """Kill the selected worker."""
        sel = self.workers_tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a worker to kill.")
            return
        worker_id = sel[0]
        if not messagebox.askyesno("Confirm Kill", f"Kill worker '{worker_id}'?"):
            return

        def do_kill():
            result = self._api_delete(f"/api/workers/{worker_id}")
            if result:
                self.root.after(0, lambda: self.log(f"Worker '{worker_id}' killed.", tag="success"))
                self.root.after(500, self._refresh_workers)
            else:
                self.root.after(0, lambda: self.log(f"Failed to kill worker '{worker_id}'.", tag="error"))

        thread = threading.Thread(target=do_kill, daemon=True)
        thread.start()

    def _kill_all_workers(self):
        """Kill all workers via API."""
        children = self.workers_tree.get_children()
        if not children:
            messagebox.showinfo("No Workers", "No workers to kill.")
            return
        if not messagebox.askyesno("Confirm", f"Kill all {len(children)} workers?"):
            return

        self.log("Killing all workers...")

        def do_kill_all():
            try:
                resp = requests.get(self._api_url("/api/workers"), timeout=3)
                if resp.status_code == 200:
                    for w in resp.json().get("workers", []):
                        try:
                            requests.delete(
                                self._api_url(f"/api/workers/{w['worker_id']}"),
                                timeout=10
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            self.root.after(0, lambda: self.log("All workers killed.", tag="success"))

        thread = threading.Thread(target=do_kill_all, daemon=True)
        thread.start()

    def _spawn_worker(self):
        """Spawn a new worker."""
        model = self.spawn_model_var.get()
        device = self.spawn_device_var.get()
        if not model:
            messagebox.showwarning("No Model", "Select a model to spawn.")
            return

        self.log(f"Spawning {model} worker on {device}...")

        def do_spawn():
            result = self._api_post("/api/workers/spawn", {"model": model, "device": device}, timeout=960)
            if result:
                wid = result.get("worker_id", "unknown")
                self.root.after(0, lambda: self.log(f"Worker '{wid}' spawned on {device}.", tag="success"))
                self.root.after(1000, self._refresh_workers)
            else:
                self.root.after(0, lambda: self.log(f"Failed to spawn {model} worker.", tag="error"))

        thread = threading.Thread(target=do_spawn, daemon=True)
        thread.start()

    # ============================================================
    # Testing Tab
    # ============================================================
    def _build_test_tab(self):
        """Build the testing tab - select loaded worker, set params, send request."""
        # --- Loaded Workers ---
        workers_frame = ttk.LabelFrame(self.test_tab, text="Loaded Workers", padding="5")
        workers_frame.pack(fill=tk.X, pady=(0, 5))

        wrk_cols = ("worker_id", "model", "port", "device", "status")
        self.test_workers_tree = ttk.Treeview(
            workers_frame, columns=wrk_cols, show="headings", height=4
        )
        for col, label, w in [
            ("worker_id", "Worker ID", 120), ("model", "Model", 100),
            ("port", "Port", 70), ("device", "Device", 90), ("status", "Status", 80),
        ]:
            self.test_workers_tree.heading(col, text=label)
            self.test_workers_tree.column(col, width=w, minwidth=50)
        self.test_workers_tree.pack(fill=tk.X)
        self.test_workers_tree.bind(
            "<<TreeviewSelect>>", lambda e: self._on_test_worker_selected()
        )
        _tip(self.test_workers_tree, TOOLTIPS["workers_tree"])

        # Guidance hint shown when no workers are loaded
        self._no_workers_hint = ttk.Label(
            workers_frame,
            text="No workers loaded \u2014 go to API Server tab to spawn one, "
                 "or just send a request to auto-spawn.",
            foreground="gray", font=("Segoe UI", 8), wraplength=800,
        )
        self._no_workers_hint.pack(anchor=tk.W, pady=(3, 0))

        # --- Voice / Reference Controls (rebuilt on worker selection) ---
        self._test_voice_frame = ttk.LabelFrame(
            self.test_tab, text="Voice", padding="5"
        )
        self._test_voice_frame.pack(fill=tk.X, pady=(0, 5))
        self._test_voice_inner = ttk.Frame(self._test_voice_frame)
        self._test_voice_inner.pack(fill=tk.X)
        ttk.Label(
            self._test_voice_inner, text="Select a worker above", foreground="gray"
        ).pack(anchor=tk.W)

        # Voice / reference state
        self.test_voice_var = tk.StringVar()
        self.test_ref_audio_var = tk.StringVar()
        self.test_ref_text_var = tk.StringVar()

        # --- Text Input ---
        text_frame = ttk.LabelFrame(self.test_tab, text="Text", padding="5")
        text_frame.pack(fill=tk.X, pady=(0, 5))
        self.test_text_input = scrolledtext.ScrolledText(
            text_frame, height=4, font=("Consolas", 10), wrap=tk.WORD
        )
        self.test_text_input.pack(fill=tk.X)
        self.test_text_input.insert(
            "1.0", "Hello, this is a test of the text to speech system."
        )
        _tip(self.test_text_input, TOOLTIPS["text_input"])

        # --- Parameters ---
        params_frame = ttk.Frame(self.test_tab)
        params_frame.pack(fill=tk.X, pady=(0, 5))

        inf_frame = ttk.LabelFrame(params_frame, text="Inference", padding="5")
        inf_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        inf_grid = ttk.Frame(inf_frame)
        inf_grid.pack(fill=tk.X)
        ttk.Label(inf_grid, text="Speed:").grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
        self.test_speed_var = tk.StringVar(value="1.0")
        spd = ttk.Spinbox(inf_grid, textvariable=self.test_speed_var, from_=0.5, to=2.0,
                           increment=0.1, width=5, format="%.1f")
        spd.grid(row=0, column=1, padx=(0, 10))
        _tip(spd, TOOLTIPS["speed"])
        ttk.Label(inf_grid, text="Temp:").grid(row=0, column=2, sticky=tk.W, padx=(0, 3))
        self.test_temp_var = tk.StringVar(value="0.65")
        tmp = ttk.Spinbox(inf_grid, textvariable=self.test_temp_var, from_=0.0, to=3.0,
                           increment=0.05, width=5, format="%.2f")
        tmp.grid(row=0, column=3, padx=(0, 10))
        _tip(tmp, TOOLTIPS["temperature"])
        ttk.Label(inf_grid, text="Rep. Penalty:").grid(row=0, column=4, sticky=tk.W, padx=(0, 3))
        self.test_rep_penalty_var = tk.StringVar(value="2.0")
        rp = ttk.Spinbox(inf_grid, textvariable=self.test_rep_penalty_var, from_=0.5, to=5.0,
                          increment=0.1, width=5, format="%.1f")
        rp.grid(row=0, column=5)
        _tip(rp, TOOLTIPS["repetition_penalty"])

        pp_frame = ttk.LabelFrame(params_frame, text="Post-Processing", padding="5")
        pp_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        pp_grid = ttk.Frame(pp_frame)
        pp_grid.pack(fill=tk.X)
        ttk.Label(pp_grid, text="De-reverb:").grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
        self.test_dereverb_var = tk.StringVar(value="0.7")
        dr = ttk.Spinbox(pp_grid, textvariable=self.test_dereverb_var, from_=0.0, to=1.0,
                          increment=0.1, width=5, format="%.1f")
        dr.grid(row=0, column=1, padx=(0, 8))
        _tip(dr, TOOLTIPS["de_reverb"])
        ttk.Label(pp_grid, text="De-esser:").grid(row=0, column=2, sticky=tk.W, padx=(0, 3))
        self.test_deess_var = tk.StringVar(value="0.0")
        de = ttk.Spinbox(pp_grid, textvariable=self.test_deess_var, from_=0.0, to=1.0,
                          increment=0.1, width=5, format="%.1f")
        de.grid(row=0, column=3, padx=(0, 8))
        _tip(de, TOOLTIPS["de_ess"])
        ttk.Label(pp_grid, text="Format:").grid(row=0, column=4, sticky=tk.W, padx=(0, 3))
        self.test_format_var = tk.StringVar(value="wav")
        fmt = ttk.Combobox(
            pp_grid, textvariable=self.test_format_var, state="readonly",
            width=5, values=["wav", "mp3", "ogg", "flac"]
        )
        fmt.grid(row=0, column=5)
        _tip(fmt, TOOLTIPS["format"])

        # --- Model-Specific Parameters (dynamic, rebuilt on worker selection) ---
        self._model_params_frame = ttk.LabelFrame(
            self.test_tab, text="Model Parameters", padding="5"
        )
        # Don't pack yet - shown/hidden by _rebuild_model_params
        self._model_params_inner = ttk.Frame(self._model_params_frame)
        self._model_params_inner.pack(fill=tk.X)
        self._model_param_vars: dict[str, tk.StringVar] = {}

        # Whisper + options
        self._whisper_opt_row = ttk.Frame(self.test_tab)
        self._whisper_opt_row.pack(fill=tk.X, pady=(0, 5))
        opt_row = self._whisper_opt_row
        self.test_verify_var = tk.BooleanVar(value=False)
        vw_cb = ttk.Checkbutton(
            opt_row, text="Verify Whisper", variable=self.test_verify_var
        )
        vw_cb.pack(side=tk.LEFT, padx=(0, 10))
        _tip(vw_cb, TOOLTIPS["verify_whisper"])
        ttk.Label(opt_row, text="Whisper:").pack(side=tk.LEFT, padx=(0, 3))
        self.test_whisper_model_var = tk.StringVar(value="base")
        wm_cb = ttk.Combobox(
            opt_row, textvariable=self.test_whisper_model_var, state="readonly",
            width=7, values=["tiny", "base", "small", "medium", "large"]
        )
        wm_cb.pack(side=tk.LEFT, padx=(0, 10))
        _tip(wm_cb, TOOLTIPS["whisper_model"])
        ttk.Label(opt_row, text="Tolerance:").pack(side=tk.LEFT, padx=(0, 3))
        self.test_tolerance_var = tk.StringVar(value="80")
        tol = ttk.Spinbox(
            opt_row, textvariable=self.test_tolerance_var, from_=0, to=100,
            increment=5, width=5, format="%d"
        )
        tol.pack(side=tk.LEFT, padx=(0, 10))
        _tip(tol, TOOLTIPS["tolerance"])
        self.test_skip_pp_var = tk.BooleanVar(value=False)
        spp_cb = ttk.Checkbutton(
            opt_row, text="Skip Post-Processing", variable=self.test_skip_pp_var
        )
        spp_cb.pack(side=tk.LEFT)
        _tip(spp_cb, TOOLTIPS["skip_post_process"])

        # Send button
        self.test_send_btn = ttk.Button(
            self.test_tab, text="Send Request", command=self._send_test_request
        )
        self.test_send_btn.pack(anchor=tk.W, pady=(0, 5))
        _tip(self.test_send_btn, TOOLTIPS["send_request"])

        # --- Response History ---
        resp_frame = ttk.LabelFrame(self.test_tab, text="Response History", padding="5")
        resp_frame.pack(fill=tk.BOTH, expand=True)

        resp_cols = ("num", "model", "duration", "time", "sr", "status")
        self.test_resp_tree = ttk.Treeview(
            resp_frame, columns=resp_cols, show="headings", height=5
        )
        for col, label, w in [
            ("num", "#", 40), ("model", "Model", 100),
            ("duration", "Duration", 80), ("time", "Time", 80),
            ("sr", "SR", 60), ("status", "Status", 80),
        ]:
            self.test_resp_tree.heading(col, text=label)
            self.test_resp_tree.column(col, width=w, minwidth=30)
        resp_scroll = ttk.Scrollbar(
            resp_frame, orient="vertical", command=self.test_resp_tree.yview
        )
        self.test_resp_tree.configure(yscrollcommand=resp_scroll.set)
        self.test_resp_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        resp_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_resp_tree.bind(
            "<<TreeviewSelect>>", lambda e: self._on_response_selected()
        )

        # Audio controls
        audio_row = ttk.Frame(self.test_tab)
        audio_row.pack(fill=tk.X, pady=(5, 0))
        self.test_play_btn = ttk.Button(
            audio_row, text="Play", command=self._play_audio, state=tk.DISABLED
        )
        self.test_play_btn.pack(side=tk.LEFT, padx=(0, 5))
        _tip(self.test_play_btn, TOOLTIPS["play_button"])
        self.test_stop_btn = ttk.Button(
            audio_row, text="Stop", command=self._stop_audio, state=tk.DISABLED
        )
        self.test_stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        _tip(self.test_stop_btn, TOOLTIPS["stop_button"])
        self.test_save_btn = ttk.Button(
            audio_row, text="Save Audio...", command=self._save_audio, state=tk.DISABLED
        )
        self.test_save_btn.pack(side=tk.LEFT, padx=(0, 10))
        _tip(self.test_save_btn, TOOLTIPS["save_button"])
        clr_btn = ttk.Button(
            audio_row, text="Clear History", command=self._clear_response_history
        )
        clr_btn.pack(side=tk.RIGHT)
        _tip(clr_btn, TOOLTIPS["clear_history"])

    # --- Worker Selection & Voice Controls ---

    def _on_test_worker_selected(self):
        """Handle worker selection - fetch and display model-specific voice controls."""
        sel = self.test_workers_tree.selection()
        if not sel:
            return
        values = self.test_workers_tree.item(sel[0], "values")
        if not values:
            return
        model = values[1]  # model is 2nd column

        # Update model-specific params and defaults
        self._rebuild_model_params(model)

        # Clear voice controls and show loading
        for w in self._test_voice_inner.winfo_children():
            w.destroy()
        ttk.Label(
            self._test_voice_inner, text=f"Loading voices for {model}...",
            foreground="gray"
        ).pack(anchor=tk.W)

        # Reset fields
        self.test_ref_audio_var.set("")
        self.test_ref_text_var.set("")

        def do_fetch():
            data = self._api_get(f"/api/tts/{model}/voices")
            voices = data.get("voices", []) if data else []
            note = data.get("note", "") if data else "Could not fetch voices"
            self.root.after(0, lambda: self._build_voice_controls(model, voices, note))

        threading.Thread(target=do_fetch, daemon=True).start()

    def _build_voice_controls(self, model: str, voices: list, note: str):
        """Build model-specific voice/reference controls in the voice frame."""
        for w in self._test_voice_inner.winfo_children():
            w.destroy()

        # Models that need reference text alongside reference audio
        needs_ref_text = model in ("f5", "higgs")
        # Models where ref audio is optional (has default or uses inline tags)
        ref_optional = model in ("fish", "dia")
        # Models that need a ref audio picker (no built-in voices)
        needs_ref_audio = model in ("chatterbox", "f5", "fish", "higgs", "vibevoice")

        row = ttk.Frame(self._test_voice_inner)
        row.pack(fill=tk.X)

        if voices:
            # Built-in voices dropdown (kokoro, xtts, bark, qwen)
            ttk.Label(row, text="Voice:").pack(side=tk.LEFT, padx=(0, 5))
            self.test_voice_var.set(voices[0])
            vc = ttk.Combobox(
                row, textvariable=self.test_voice_var, state="readonly",
                width=24, values=voices
            )
            vc.pack(side=tk.LEFT, padx=(0, 15))
            _tip(vc, TOOLTIPS["voice_dropdown"])

        # Reference audio picker
        if needs_ref_audio or model == "xtts":
            if ref_optional:
                lbl = "Ref Audio (optional):"
            elif voices:  # xtts supports both built-in and cloning
                lbl = "Ref Audio (optional):"
            else:
                lbl = "Ref Audio:"
            is_optional = ref_optional or bool(voices)
            ra_tip = TOOLTIPS["ref_audio_optional"] if is_optional else TOOLTIPS["ref_audio"]
            ttk.Label(row, text=lbl).pack(side=tk.LEFT, padx=(0, 5))
            ra_entry = ttk.Entry(
                row, textvariable=self.test_ref_audio_var, width=30
            )
            ra_entry.pack(side=tk.LEFT, padx=(0, 5))
            _tip(ra_entry, ra_tip)
            br_btn = ttk.Button(
                row, text="Browse...", command=self._browse_ref_audio
            )
            br_btn.pack(side=tk.LEFT)
            _tip(br_btn, ra_tip)

        # Reference text field (F5 and Higgs need transcript of ref audio)
        if needs_ref_text:
            ref_text_row = ttk.Frame(self._test_voice_inner)
            ref_text_row.pack(fill=tk.X, pady=(3, 0))
            ttk.Label(ref_text_row, text="Ref Text:").pack(
                side=tk.LEFT, padx=(0, 5)
            )
            rt_entry = ttk.Entry(
                ref_text_row, textvariable=self.test_ref_text_var, width=70
            )
            rt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            _tip(rt_entry, TOOLTIPS["ref_text"])

        if note:
            ttk.Label(
                self._test_voice_inner, text=note, foreground="gray",
                font=("Segoe UI", 8), wraplength=800, justify=tk.LEFT
            ).pack(anchor=tk.W, pady=(3, 0))

    def _rebuild_model_params(self, model: str):
        """Rebuild model-specific parameter spinboxes and update common defaults."""
        # Clear existing model param widgets
        for w in self._model_params_inner.winfo_children():
            w.destroy()
        self._model_param_vars.clear()

        defaults = MODEL_DEFAULTS.get(model, {})

        # Update common params to model-recommended defaults
        if "temperature" in defaults:
            self.test_temp_var.set(str(defaults["temperature"]))
        if "speed" in defaults:
            self.test_speed_var.set(str(defaults["speed"]))
        if "repetition_penalty" in defaults:
            self.test_rep_penalty_var.set(str(defaults["repetition_penalty"]))

        # Filter to model-specific params (not already in the main Inference frame)
        extra_params = {k: v for k, v in defaults.items() if k not in _COMMON_PARAMS}

        if not extra_params:
            self._model_params_frame.pack_forget()
            return

        # Show the frame (insert before whisper options row)
        display = MODEL_SETUP.get(model, {}).get("display", model)
        self._model_params_frame.config(text=f"{display} Parameters")
        self._model_params_frame.pack(fill=tk.X, pady=(0, 5),
                                       before=self._whisper_opt_row)

        # Build spinboxes in a grid, 3 params per row
        grid = ttk.Frame(self._model_params_inner)
        grid.pack(fill=tk.X)
        col = 0
        row = 0
        for param_name, default_val in extra_params.items():
            cfg = PARAM_SPINBOX.get(param_name)
            if not cfg:
                continue
            label, from_, to, increment, width, fmt = cfg
            var = tk.StringVar(value=str(default_val))
            self._model_param_vars[param_name] = var

            ttk.Label(grid, text=f"{label}:").grid(
                row=row, column=col, sticky=tk.W, padx=(0, 3))
            spin = ttk.Spinbox(grid, textvariable=var, from_=from_, to=to,
                               increment=increment, width=width, format=fmt)
            spin.grid(row=row, column=col + 1, padx=(0, 10))
            tip_text = TOOLTIPS.get(param_name)
            if tip_text:
                _tip(spin, tip_text)
            col += 2
            if col >= 8:  # 4 params per row
                col = 0
                row += 1

    def _browse_ref_audio(self):
        """Browse for a reference audio file."""
        path = filedialog.askopenfilename(
            title="Select Reference Audio",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.test_ref_audio_var.set(path)

    # --- Send Request ---

    def _send_test_request(self):
        """Send a TTS request using the selected worker's model."""
        sel = self.test_workers_tree.selection()
        if not sel:
            messagebox.showwarning("No Worker", "Select a loaded worker from the list.")
            return

        values = self.test_workers_tree.item(sel[0], "values")
        model = values[1]

        text = self.test_text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty Text", "Please enter text to synthesize.")
            return

        voice = self.test_voice_var.get()
        ref_audio = self.test_ref_audio_var.get().strip()
        ref_text = self.test_ref_text_var.get().strip()

        # Validate required reference audio for models that need it
        if model in ("f5", "chatterbox", "vibevoice", "higgs") and not ref_audio:
            messagebox.showwarning(
                "Reference Audio Required",
                f"{model.upper()} requires a reference audio file.\n"
                "Use the Browse button to select one."
            )
            return

        try:
            params = {
                "text": text,
                "speed": float(self.test_speed_var.get()),
                "temperature": float(self.test_temp_var.get()),
                "repetition_penalty": float(self.test_rep_penalty_var.get()),
                "output_format": self.test_format_var.get(),
                "de_reverb": float(self.test_dereverb_var.get()),
                "de_ess": float(self.test_deess_var.get()),
                "tolerance": float(self.test_tolerance_var.get()),
                "verify_whisper": self.test_verify_var.get(),
                "whisper_model": self.test_whisper_model_var.get(),
                "skip_post_process": self.test_skip_pp_var.get(),
            }
        except ValueError as e:
            messagebox.showerror("Invalid Parameter", f"Check numeric parameters: {e}")
            return

        # Add model-specific params from dynamic spinboxes
        for param_name, var in self._model_param_vars.items():
            try:
                val = var.get()
                cfg = PARAM_SPINBOX.get(param_name)
                if cfg and "%d" in cfg[5]:
                    params[param_name] = int(float(val))
                else:
                    params[param_name] = float(val)
            except (ValueError, TypeError):
                pass

        if voice and voice not in ("(reference audio)", "Loading..."):
            params["voice"] = voice
        if ref_audio:
            params["reference_audio"] = ref_audio
        if ref_text:
            params["reference_text"] = ref_text

        self.test_send_btn.config(state=tk.DISABLED, text="Sending...")
        self.log(f"Sending TTS request: model={model}, text='{text[:50]}...'")

        def do_request():
            import time as _time
            start = _time.time()
            try:
                resp = requests.post(
                    self._api_url(f"/api/tts/{model}"),
                    json=params, timeout=600
                )
                elapsed = _time.time() - start
                if resp.status_code == 200:
                    data = resp.json()
                    self.root.after(
                        0, lambda: self._handle_test_response(data, elapsed, model)
                    )
                else:
                    detail = self._extract_error_detail(resp)
                    self.root.after(0, lambda: self._handle_test_error(
                        f"HTTP {resp.status_code}: {detail}", model
                    ))
            except requests.Timeout:
                self.root.after(0, lambda: self._handle_test_error(
                    "Request timed out (600s)", model
                ))
            except requests.ConnectionError:
                self.root.after(0, lambda: self._handle_test_error(
                    "Connection failed. Is the server running?", model
                ))
            except Exception as e:
                self.root.after(0, lambda: self._handle_test_error(str(e), model))

        threading.Thread(target=do_request, daemon=True).start()

    # --- Response Handling ---

    def _handle_test_response(self, data: dict, elapsed: float, model: str):
        """Handle successful TTS response - add to response history."""
        self.test_send_btn.config(state=tk.NORMAL, text="Send Request")

        duration = data.get("duration_sec", 0)
        sr = data.get("sample_rate", 0)
        fmt = data.get("format", "wav")
        status = data.get("status", "completed")

        # Decode audio to temp file
        audio_path = None
        audio_b64 = data.get("audio_base64")
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                suffix = f".{fmt}" if fmt else ".wav"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                audio_path = tmp.name
            except Exception as e:
                self.log(f"Failed to decode audio: {e}", tag="error")

        resp_num = len(self._test_responses) + 1
        self._test_responses.append({
            "num": resp_num, "model": model, "duration": duration,
            "elapsed": elapsed, "status": status, "audio_path": audio_path,
            "format": fmt, "sample_rate": sr,
        })

        iid = str(resp_num)
        self.test_resp_tree.insert("", 0, iid=iid, values=(
            resp_num, model, f"{duration:.2f}s", f"{elapsed:.2f}s", sr, status
        ))
        self.test_resp_tree.selection_set(iid)
        self.test_resp_tree.see(iid)

        self.log(
            f"TTS #{resp_num}: {model} - {duration:.2f}s audio in {elapsed:.2f}s",
            tag="success"
        )

    def _handle_test_error(self, msg: str, model: str = ""):
        """Handle TTS request error - add error entry to history."""
        self.test_send_btn.config(state=tk.NORMAL, text="Send Request")

        resp_num = len(self._test_responses) + 1
        self._test_responses.append({
            "num": resp_num, "model": model or "—", "duration": 0,
            "elapsed": 0, "status": "error", "audio_path": None,
            "format": None, "sample_rate": 0,
        })

        iid = str(resp_num)
        self.test_resp_tree.insert("", 0, iid=iid, values=(
            resp_num, model or "—", "—", "—", "—", "error"
        ))
        self.log(f"TTS request failed: {msg}", tag="error")

    def _on_response_selected(self):
        """Enable/disable audio controls based on selected response."""
        sel = self.test_resp_tree.selection()
        if not sel:
            self.test_play_btn.config(state=tk.DISABLED)
            self.test_stop_btn.config(state=tk.DISABLED)
            self.test_save_btn.config(state=tk.DISABLED)
            return

        entry = self._test_responses[int(sel[0]) - 1]
        has_audio = (
            entry["audio_path"] and os.path.exists(entry["audio_path"])
        )
        self.test_save_btn.config(state=tk.NORMAL if has_audio else tk.DISABLED)
        if has_audio and entry["format"] == "wav":
            self.test_play_btn.config(state=tk.NORMAL)
            self.test_stop_btn.config(state=tk.NORMAL)
        else:
            self.test_play_btn.config(state=tk.DISABLED)
            self.test_stop_btn.config(state=tk.DISABLED)

    # --- Audio Playback ---

    def _play_audio(self):
        """Play the selected response's audio."""
        sel = self.test_resp_tree.selection()
        if not sel:
            return
        entry = self._test_responses[int(sel[0]) - 1]
        if entry["audio_path"] and os.path.exists(entry["audio_path"]):
            try:
                winsound.PlaySound(
                    entry["audio_path"],
                    winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            except Exception as e:
                self.log(f"Playback failed: {e}", tag="warning")

    def _stop_audio(self):
        """Stop audio playback."""
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def _save_audio(self):
        """Save the selected response's audio to a user-chosen location."""
        sel = self.test_resp_tree.selection()
        if not sel:
            return
        entry = self._test_responses[int(sel[0]) - 1]
        if not entry["audio_path"] or not os.path.exists(entry["audio_path"]):
            return

        ext = f".{entry['format']}" if entry["format"] else ".wav"
        dest = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[("Audio files", f"*{ext}"), ("All files", "*.*")],
            title="Save Audio"
        )
        if dest:
            try:
                shutil.copy2(entry["audio_path"], dest)
                self.log(f"Audio saved to {dest}", tag="success")
            except Exception as e:
                self.log(f"Failed to save audio: {e}", tag="error")

    def _clear_response_history(self):
        """Clear response history and delete temp audio files."""
        for entry in self._test_responses:
            if entry.get("audio_path") and os.path.exists(entry["audio_path"]):
                try:
                    os.unlink(entry["audio_path"])
                except Exception:
                    pass
        self._test_responses.clear()
        for item in self.test_resp_tree.get_children():
            self.test_resp_tree.delete(item)
        self.test_play_btn.config(state=tk.DISABLED)
        self.test_stop_btn.config(state=tk.DISABLED)
        self.test_save_btn.config(state=tk.DISABLED)

    def _cleanup_test_audio(self):
        """Remove all temporary test audio files."""
        for entry in self._test_responses:
            if entry.get("audio_path") and os.path.exists(entry["audio_path"]):
                try:
                    os.unlink(entry["audio_path"])
                except Exception:
                    pass
        self._test_responses.clear()

    def _create_setup_row(self, parent: ttk.Frame, model_id: str, info: dict):
        """Create a row for a model in the setup tab."""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=2)

        canvas = tk.Canvas(row_frame, width=20, height=20, highlightthickness=0)
        canvas.pack(side=tk.LEFT, padx=(0, 5))
        status_circle = canvas.create_oval(4, 4, 16, 16, fill="gray", outline="")

        size_str = f" ({info['weights_size']})" if info["weights_size"] else ""
        name_label = ttk.Label(
            row_frame,
            text=f"{info['display']}",
            font=("Segoe UI", 10, "bold"),
            width=16, anchor=tk.W
        )
        name_label.pack(side=tk.LEFT, padx=(0, 5))

        desc_label = ttk.Label(
            row_frame,
            text=f"{info['desc']}{size_str}",
            width=44, anchor=tk.W
        )
        desc_label.pack(side=tk.LEFT, padx=(0, 5))

        status_label = ttk.Label(row_frame, text="Checking...", width=16, anchor=tk.W)
        status_label.pack(side=tk.LEFT, padx=(0, 5))

        install_btn = ttk.Button(
            row_frame, text="Install", width=8,
            command=lambda m=model_id: self._install_model(m)
        )
        install_btn.pack(side=tk.LEFT, padx=2)

        remove_btn = ttk.Button(
            row_frame, text="Remove", width=8,
            command=lambda m=model_id: self._remove_model(m)
        )
        remove_btn.pack(side=tk.LEFT, padx=2)

        self.setup_widgets[model_id] = {
            "canvas": canvas,
            "circle": status_circle,
            "status_label": status_label,
            "install_btn": install_btn,
            "remove_btn": remove_btn,
        }

    def _check_model_status(self, model_id: str) -> tuple:
        """Check if a model's env and weights are ready. Returns (status, label)."""
        info = MODEL_SETUP[model_id]
        env_name = info["env"]
        env = self.environments.get(env_name)

        env_ok = env and env.check_status() == "installed"

        if info["weights_repo"]:
            weights_path = MODELS_DIR / info["weights_dir"]
            try:
                weights_ok = weights_path.exists() and any(weights_path.iterdir())
            except (OSError, StopIteration):
                weights_ok = False
        else:
            weights_ok = True  # auto-download models

        if env_ok and weights_ok:
            return ("ready", "Ready")
        elif env_ok and not weights_ok:
            return ("partial", "Weights needed")
        elif not env_ok and weights_ok:
            return ("partial", "Env needed")
        else:
            return ("not_installed", "Not installed")

    def _update_setup_row(self, model_id: str, status: str, label: str):
        """Update a single model row in the setup tab."""
        widgets = self.setup_widgets.get(model_id)
        if not widgets:
            return

        colors = {
            "ready": "#32CD32",
            "partial": "#FFA500",
            "not_installed": "gray",
            "installing": "#FFD700",
            "error": "#FF4444",
        }
        color = colors.get(status, "gray")
        widgets["canvas"].itemconfig(widgets["circle"], fill=color)
        widgets["status_label"].config(text=label)

        is_busy = status == "installing"
        widgets["install_btn"].config(
            state=tk.DISABLED if is_busy else tk.NORMAL,
            text="Installing..." if is_busy else "Install"
        )
        widgets["remove_btn"].config(
            state=tk.DISABLED if is_busy or status == "not_installed" else tk.NORMAL
        )

    def _refresh_setup_status(self):
        """Check and update status for all models."""
        self.log("Checking model status...")
        for model_id in MODEL_SETUP:
            status, label = self._check_model_status(model_id)
            self._update_setup_row(model_id, status, label)
        self.log("Status check complete.", tag="success")

    def log(self, message: str, tag: str = "info"):
        """Add a message to the log."""
        self.log_queue.put((message, tag))

    def _process_log_queue(self):
        """Process messages from the log queue."""
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple):
                    message, tag = item
                else:
                    message, tag = item, "info"

                # Store in ring buffer
                self._log_entries.append((message, tag))

                # Only display if filter allows this tag
                filter_var = self._log_filters.get(tag, self._log_filters.get("info"))
                if filter_var and filter_var.get():
                    self._append_log_line(message, tag)
        except queue.Empty:
            pass

        self.root.after(100, self._process_log_queue)

    def _append_log_line(self, message: str, tag: str):
        """Append a single line to the log widget, enforcing line limit."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n", tag)
        # Trim widget to 1000 visible lines
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 1000:
            self.log_text.delete("1.0", f"{line_count - 1000}.0")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _rerender_log(self):
        """Re-render the log from the ring buffer with current filters applied."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for message, tag in self._log_entries:
            filter_var = self._log_filters.get(tag, self._log_filters.get("info"))
            if filter_var and filter_var.get():
                self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        """Clear the log text and buffer."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
        self._log_entries.clear()

    # ============================================================
    # Model Installation (env + weights)
    # ============================================================
    def _install_model(self, model_id: str):
        """Install a model: set up its venv and download weights."""
        if self.current_installation:
            messagebox.showwarning("Installation in Progress",
                                   "Please wait for the current installation to complete.")
            return

        info = MODEL_SETUP[model_id]

        # Warn about gated models
        if info.get("gated"):
            if not messagebox.askyesno(
                "Gated Model",
                f"'{info['display']}' requires HuggingFace authentication.\n\n"
                "1. Accept the terms at the model page\n"
                "2. Run 'huggingface-cli login' in a terminal\n\n"
                f"Repo: {info['weights_repo']}\n\nHave you completed these steps?"
            ):
                return

        self.current_installation = model_id
        self._update_setup_row(model_id, "installing", "Installing...")

        thread = threading.Thread(target=self._run_model_setup, args=(model_id,), daemon=True)
        thread.start()

    def _run_model_setup(self, model_id: str):
        """Run full model setup in background: install env + download weights."""
        info = MODEL_SETUP[model_id]
        env_name = info["env"]
        env = self.environments.get(env_name)

        try:
            # --- Step 1: Install venv if needed ---
            if env:
                env.check_status()
                if env.status != "installed":
                    self.log(f"\n{'='*50}")
                    self.log(f"Installing env '{env_name}' for {info['display']}")
                    self.log(f"{'='*50}")
                    self._run_env_install(env)
                    if env.status != "installed":
                        raise Exception(f"Environment '{env_name}' installation failed")
                else:
                    self.log(f"Environment '{env_name}' already installed.")
            else:
                self.log(f"Warning: No config found for env '{env_name}'", tag="warning")

            # --- Step 2: Download weights if needed ---
            if info["weights_repo"]:
                weights_path = MODELS_DIR / info["weights_dir"]
                try:
                    weights_exist = weights_path.exists() and any(weights_path.iterdir())
                except (OSError, StopIteration):
                    weights_exist = False

                if not weights_exist:
                    self.log(f"\nDownloading {info['display']} weights from {info['weights_repo']}...")
                    self._run_weights_download(model_id, info)
                else:
                    self.log(f"Weights for {info['display']} already downloaded.")
            else:
                self.log(f"{info['display']} weights auto-download on first use.")

            # Check final status
            status, label = self._check_model_status(model_id)
            self.root.after(0, lambda: self._update_setup_row(model_id, status, label))
            self.log(f"{info['display']} setup complete!", tag="success")

        except Exception as e:
            self.log(f"\nSetup failed for {info['display']}: {e}", tag="error")
            self.root.after(0, lambda: self._update_setup_row(model_id, "error", "Error"))

        finally:
            self.current_installation = None

    def _run_env_install(self, env: TTSEnvironment):
        """Install a venv (blocking, runs in background thread)."""
        try:
            VENVS_DIR.mkdir(exist_ok=True)

            if not env.venv_path.exists():
                self.log(f"Creating virtual environment at {env.venv_path}...")
                # Use virtualenv (pip-installed) instead of venv (stdlib) because
                # embedded Python does not include the venv module.
                result = subprocess.run(
                    [str(PYTHON_PATH), "-m", "virtualenv", str(env.venv_path)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise Exception(f"Failed to create venv: {result.stderr}")
                self.log("Virtual environment created.", tag="success")
            else:
                self.log("Virtual environment already exists.")

            pip_path = str(env.pip_path)

            for step in env.config.get_install_steps():
                self.log(f"\n> {step['description']}...")

                if step["type"] == "pip":
                    success, output = run_pip_install(
                        pip_path, step["args"], lambda msg: self.log(msg)
                    )
                    if not success:
                        raise Exception(f"Pip install failed: {output}")

                elif step["type"] == "git_clone":
                    repos_dir = env.venv_path / "repos"
                    repos_dir.mkdir(exist_ok=True)
                    success, output = run_git_clone(
                        step["url"], str(repos_dir), pip_path,
                        step.get("editable", True), lambda msg: self.log(msg)
                    )
                    if not success:
                        self.log(f"Warning: Git clone failed - {output}", tag="warning")

            if env.config.system_notes:
                self.log(f"\n{env.config.system_notes}", tag="warning")

            self.log(f"\n{env.config.display_name} environment installed!", tag="success")
            env.status = "installed"

        except Exception as e:
            self.log(f"\nEnvironment installation failed: {e}", tag="error")
            env.status = "error"
            env.error_message = str(e)
            raise

    def _run_weights_download(self, model_id: str, info: dict):
        """Download model weights (blocking, runs in background thread)."""
        # Find a venv with huggingface_hub installed
        hf_python = None
        # Prefer the model's own env first
        own_env = VENVS_DIR / info["env"] / "Scripts" / "python.exe"
        if own_env.exists():
            hf_python = str(own_env)
        else:
            for env_name in ["unified_env", "coqui_env", "chatterbox_env", "f5tts_env"]:
                env_path = VENVS_DIR / env_name / "Scripts" / "python.exe"
                if env_path.exists():
                    hf_python = str(env_path)
                    break

        if not hf_python:
            raise Exception("No environment with huggingface_hub found.")

        weights_dir = MODELS_DIR / info["weights_dir"]
        download_script = (
            f'import os\n'
            f'os.environ["HF_HOME"] = r"{MODELS_DIR}"\n'
            f'os.environ["HUGGINGFACE_HUB_CACHE"] = r"{MODELS_DIR / "hub"}"\n'
            f'from huggingface_hub import snapshot_download\n'
            f'snapshot_download(\n'
            f'    repo_id="{info["weights_repo"]}",\n'
            f'    local_dir=r"{weights_dir}",\n'
            f'    local_dir_use_symlinks=False\n'
            f')\n'
            f'print("Download complete!")\n'
        )

        result = subprocess.run(
            [hf_python, "-c", download_script],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise Exception(f"Download failed: {result.stderr[:500]}")

        self.log(f"Weights downloaded to {weights_dir}", tag="success")

    def _remove_model(self, model_id: str):
        """Remove a model's venv and/or weights."""
        info = MODEL_SETUP[model_id]
        env = self.environments.get(info["env"])

        # Check what can be removed
        has_env = env and env.venv_path.exists()
        has_weights = (info["weights_dir"] and
                       (MODELS_DIR / info["weights_dir"]).exists())

        if not has_env and not has_weights:
            messagebox.showinfo("Nothing to Remove", f"{info['display']} is not installed.")
            return

        # Check if other models share this env
        shared_models = [m for m, i in MODEL_SETUP.items()
                         if i["env"] == info["env"] and m != model_id]
        env_note = ""
        if shared_models and has_env:
            env_note = (f"\n\nNote: env '{info['env']}' is shared with: "
                        f"{', '.join(shared_models)}.\nRemoving it will affect those models too.")

        parts = []
        if has_env:
            parts.append(f"Environment: {env.venv_path}")
        if has_weights:
            parts.append(f"Weights: {MODELS_DIR / info['weights_dir']}")

        if not messagebox.askyesno(
            "Confirm Remove",
            f"Remove {info['display']}?\n\n" + "\n".join(parts) + env_note
        ):
            return

        self.log(f"Removing {info['display']}...")
        try:
            if has_weights:
                shutil.rmtree(MODELS_DIR / info["weights_dir"])
                self.log(f"Weights removed.", tag="success")
            if has_env:
                shutil.rmtree(env.venv_path)
                env.status = "not_installed"
                self.log(f"Environment '{info['env']}' removed.", tag="success")
        except Exception as e:
            self.log(f"Remove failed: {e}", tag="error")

        # Refresh all rows that share this env
        for m in [model_id] + shared_models:
            status, label = self._check_model_status(m)
            self._update_setup_row(m, status, label)

    def _install_all_models(self):
        """Install all models that aren't ready."""
        if self.current_installation:
            messagebox.showwarning("Installation in Progress",
                                   "Please wait for the current installation to complete.")
            return

        to_install = []
        for model_id in MODEL_SETUP:
            status, _ = self._check_model_status(model_id)
            if status != "ready":
                to_install.append(model_id)

        if not to_install:
            messagebox.showinfo("All Ready", "All models are already installed!")
            return

        names = "\n".join(f"- {MODEL_SETUP[m]['display']}" for m in to_install)
        if not messagebox.askyesno("Install All",
                                    f"Install {len(to_install)} models?\n\n{names}"):
            return

        def do_install_all():
            import time
            for model_id in to_install:
                self.current_installation = model_id
                self.root.after(0, lambda m=model_id: self._update_setup_row(
                    m, "installing", "Installing..."))
                self._run_model_setup(model_id)
                time.sleep(1)

        thread = threading.Thread(target=do_install_all, daemon=True)
        thread.start()

    # ============================================================
    # Gateway Management
    # ============================================================
    def _start_gateway(self):
        """Start the gateway server as a subprocess."""
        port = self.port_var.get()
        try:
            port = int(port)
        except ValueError:
            port = DEFAULT_API_PORT

        self._gateway_port = port
        server_python = str(PYTHON_PATH)

        if not os.path.exists(server_python):
            self.log(f"Embedded Python not found at: {server_python}", tag="error")
            return

        self.log(f"Starting gateway on port {port}...")

        try:
            self._gateway_process = subprocess.Popen(
                [server_python, "-m", "uvicorn", "tts_api_server:app",
                 "--host", "0.0.0.0", "--port", str(port)],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Monitor output in background
            thread = threading.Thread(target=self._monitor_gateway, daemon=True)
            thread.start()

            self.log(f"Gateway starting (PID {self._gateway_process.pid})...")
        except Exception as e:
            self.log(f"Failed to start gateway: {e}", tag="error")

    def _stop_gateway(self):
        """Stop the gateway and all its child processes."""
        if not self._gateway_process:
            return

        pid = self._gateway_process.pid
        self._stop_server_polling()

        # Kill the entire process tree
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, timeout=10,
            )
        except Exception:
            try:
                self._gateway_process.kill()
            except Exception:
                pass

        self._gateway_process = None
        self.server_status_canvas.itemconfig(self.server_status_circle, fill="gray")
        self.server_status_label.config(text="Gateway: stopped")

    def _restart_gateway(self):
        """Stop and restart the gateway with the current port setting."""
        self.log("Restarting gateway...")
        self._stop_gateway()

        # Delay to let port free up, then start (non-blocking)
        self.root.after(1500, self._finish_restart_gateway)

    def _finish_restart_gateway(self):
        """Complete gateway restart after port-release delay."""
        self._start_gateway()
        self._start_server_polling()
        self.log("Gateway restarted.", tag="success")

    def _monitor_gateway(self):
        """Monitor gateway stdout and log it."""
        proc = self._gateway_process
        if not proc or not proc.stdout:
            return
        try:
            for line in proc.stdout:
                if self._gateway_process is not proc:
                    break
                self.log(f"[Gateway] {line.strip()}", tag="gateway")
        except Exception:
            pass

    # ============================================================
    # Worker Cleanup
    # ============================================================
    def _kill_all_workers_sync(self):
        """Kill all workers via API (blocking, best-effort). For use in close handler."""
        try:
            resp = requests.get(self._api_url("/api/workers"), timeout=3)
            if resp.status_code == 200:
                workers = resp.json().get("workers", [])
                for w in workers:
                    try:
                        requests.delete(
                            self._api_url(f"/api/workers/{w['worker_id']}"),
                            timeout=5
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    # ============================================================
    # Local Device Detection
    # ============================================================
    def _detect_local_devices(self):
        """Detect GPUs via nvidia-smi locally (no server needed)."""
        devices = []
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        device_id = f"cuda:{parts[0]}"
                        devices.append({
                            "id": device_id,
                            "name": parts[1],
                            "vram_total_mb": int(float(parts[2])),
                            "vram_free_mb": int(float(parts[3])),
                            "workers": [],
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        devices.append({
            "id": "cpu", "name": "CPU",
            "vram_total_mb": 0, "vram_free_mb": 0, "workers": [],
        })

        self._update_device_combos(devices)

    # ============================================================
    # API Helpers
    # ============================================================
    def _api_url(self, path: str) -> str:
        """Build full API URL from path."""
        return f"http://localhost:{self._gateway_port}{path}"

    def _api_get(self, path: str, timeout: int = 10):
        """GET request to the API server. Returns dict or None."""
        try:
            resp = requests.get(self._api_url(path), timeout=timeout)
            if resp.status_code >= 400:
                detail = self._extract_error_detail(resp)
                self.log(f"API GET {path} -> {resp.status_code}: {detail}", tag="error")
                return None
            return resp.json()
        except Exception as e:
            self.log(f"API GET {path} failed: {e}", tag="error")
            return None

    def _api_post(self, path: str, json_body=None, timeout: int = 300):
        """POST request to the API server. Returns dict or None."""
        try:
            resp = requests.post(self._api_url(path), json=json_body, timeout=timeout)
            if resp.status_code >= 400:
                detail = self._extract_error_detail(resp)
                self.log(f"API POST {path} -> {resp.status_code}: {detail}", tag="error")
                return None
            return resp.json()
        except Exception as e:
            self.log(f"API POST {path} failed: {e}", tag="error")
            return None

    def _api_delete(self, path: str, timeout: int = 30):
        """DELETE request to the API server. Returns dict or None."""
        try:
            resp = requests.delete(self._api_url(path), timeout=timeout)
            if resp.status_code >= 400:
                detail = self._extract_error_detail(resp)
                self.log(f"API DELETE {path} -> {resp.status_code}: {detail}", tag="error")
                return None
            return resp.json()
        except Exception as e:
            self.log(f"API DELETE {path} failed: {e}", tag="error")
            return None

    @staticmethod
    def _extract_error_detail(resp) -> str:
        """Extract error detail from an HTTP error response."""
        try:
            data = resp.json()
            return data.get("detail", resp.text[:300])
        except Exception:
            return resp.text[:300]

    # ============================================================
    # Server Polling
    # ============================================================
    def _start_server_polling(self):
        """Start polling server status every 10 seconds."""
        self._stop_server_polling()
        self._poll_server_status()

    def _stop_server_polling(self):
        """Stop polling server status."""
        if self._server_poll_id is not None:
            self.root.after_cancel(self._server_poll_id)
            self._server_poll_id = None

    def _poll_server_status(self):
        """Poll server health and workers."""
        def do_poll():
            results = {}
            try:
                resp = requests.get(self._api_url("/health"), timeout=3)
                results["healthy"] = resp.status_code == 200
            except Exception:
                results["healthy"] = False

            if results["healthy"]:
                try:
                    resp = requests.get(self._api_url("/api/workers"), timeout=5)
                    if resp.status_code == 200:
                        results["workers"] = resp.json()
                except Exception:
                    pass

            try:
                self.root.after(0, lambda: self._handle_poll_results(results))
            except RuntimeError:
                pass  # mainloop exiting

        thread = threading.Thread(target=do_poll, daemon=True)
        thread.start()

        # Schedule next poll in 10 seconds
        self._server_poll_id = self.root.after(10000, self._poll_server_status)

    def _handle_poll_results(self, results: dict):
        """Handle poll results on the main thread."""
        if results.get("healthy"):
            self.server_status_canvas.itemconfig(self.server_status_circle, fill="#32CD32")
            self.server_status_label.config(text=f"Gateway: connected (port {self._gateway_port})")
        else:
            self.server_status_canvas.itemconfig(self.server_status_circle, fill="gray")
            self.server_status_label.config(text="Gateway: not connected")

        if "workers" in results:
            workers_data = results["workers"]
            worker_list = workers_data.get("workers", []) if isinstance(workers_data, dict) else workers_data
            self._update_workers_display(worker_list)
            # Also update the Testing tab workers treeview
            self._update_tree_inplace(self.test_workers_tree, worker_list)
            # Show/hide "no workers" hint
            if hasattr(self, "_no_workers_hint"):
                if self.test_workers_tree.get_children():
                    self._no_workers_hint.pack_forget()
                else:
                    self._no_workers_hint.pack(anchor=tk.W, pady=(3, 0))

    # ============================================================
    # Utilities
    # ============================================================
    def _show_model_info(self, model_id: str):
        """Show detailed info about a model."""
        info = MODEL_SETUP[model_id]
        env = self.environments.get(info["env"])
        status, status_label = self._check_model_status(model_id)

        text = f"Model: {info['display']}\n"
        text += f"Description: {info['desc']}\n"
        text += f"Status: {status_label}\n\n"
        text += f"Environment: {info['env']}\n"
        if env:
            text += f"  Path: {env.venv_path}\n"
            text += f"  Status: {env.check_status()}\n"
        if info["weights_repo"]:
            text += f"\nWeights: {info['weights_repo']}\n"
            text += f"  Size: {info['weights_size']}\n"
            text += f"  Path: {MODELS_DIR / info['weights_dir']}\n"
        else:
            text += f"\nWeights: auto-download on first use\n"

        messagebox.showinfo(f"{info['display']} Info", text)

    def _open_venvs_folder(self):
        """Open the venvs folder in Explorer."""
        VENVS_DIR.mkdir(exist_ok=True)
        os.startfile(VENVS_DIR)

    def _on_close(self):
        """Handle window close - kill gateway + all workers, cleanup, destroy."""
        self._stop_server_polling()
        self._stop_audio()
        self._cleanup_test_audio()
        self._kill_all_workers_sync()
        self._stop_gateway()
        self.root.destroy()

def main():
    """Main entry point."""
    if not os.path.exists(PYTHON_PATH):
        if USE_EMBEDDED:
            msg = (
                f"Embedded Python not found at:\n{PYTHON_PATH}\n\n"
                "Run install.bat to set up the embedded Python environment."
            )
        else:
            msg = (
                f"Python not found at:\n{PYTHON_PATH}\n\n"
                "Run install.bat to set up the embedded Python environment,\n"
                "or install Python 3.10+ on your system."
            )
        messagebox.showerror("Python Not Found", msg)
        return

    root = tk.Tk()

    try:
        root.iconbitmap(default="")
    except:
        pass

    style = ttk.Style()
    style.theme_use("clam")

    app = TTSManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
