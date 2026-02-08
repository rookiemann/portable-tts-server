"""Unified TTS installation configuration (Fish Speech + Kokoro + Dia)."""

from .base import InstallConfig


class UnifiedTTSConfig(InstallConfig):
    """Configuration for unified TTS environment with multiple compatible models."""

    def __init__(self):
        super().__init__()
        self.name = "unified_env"
        self.display_name = "Unified (Fish + Kokoro + Dia)"
        self.description = (
            "Three compatible models in one environment:\n"
            "- Fish Speech 1.5 / S1-mini: High quality, flexible\n"
            "- Kokoro: Lightweight (82M), fast\n"
            "- Dia: Dialogue-focused, 2-speaker support"
        )
        self.verify_package = "kokoro"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch with CUDA 12.1
            ["torch", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu121"],

            # Core shared dependencies
            ["transformers"],
            ["accelerate"],
            ["scipy"],
            ["numpy"],
            ["librosa"],
            ["soundfile"],

            # Kokoro TTS
            ["kokoro>=0.3.4"],

            # Dia TTS
            ["diarizors"],
        ]

        # Git repos to clone
        self.git_repos = [
            # Fish Speech
            ("https://github.com/fishaudio/fish-speech", True),
        ]

        self.system_notes = (
            "SYSTEM DEPENDENCY REQUIRED:\n"
            "Kokoro requires espeak-ng to be installed on your system.\n"
            "Windows: Download from https://github.com/espeak-ng/espeak-ng/releases\n"
            "Install espeak-ng and add it to your PATH."
        )
