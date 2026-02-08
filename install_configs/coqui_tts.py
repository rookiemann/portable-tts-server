"""Coqui-ai-TTS installation configuration."""

from .base import InstallConfig


class CoquiTTSConfig(InstallConfig):
    """Configuration for Coqui-ai-TTS environment (Idiap fork)."""

    def __init__(self):
        super().__init__()
        self.name = "coqui_env"
        self.display_name = "Coqui TTS"
        self.description = (
            "Coqui-ai-TTS (Idiap fork): Swiss army knife of TTS. "
            "Includes XTTS v2, Bark, OpenVoice, VITS, YourTTS, Tortoise, "
            "and many vocoders. Supports 1100+ languages via Fairseq."
        )
        self.verify_package = "coqui-tts"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch 2.6.0 with CUDA 12.4 (Coqui needs PyTorch 2.2+)
            ["torch==2.6.0", "torchaudio==2.6.0",
             "--index-url", "https://download.pytorch.org/whl/cu124"],

            # Coqui TTS with all optional dependencies
            ["coqui-tts[all]"],

            # Pin transformers <5.0 (Coqui incompatible with transformers 5.x)
            ["transformers>=4.47,<5.0"],
        ]

        # No git repos needed - pip installable
        self.git_repos = []

        self.system_notes = (
            "Includes: XTTS v2, Bark, OpenVoice, VITS, YourTTS, Tortoise. "
            "Models downloaded on first use from Coqui model registry. "
            "Run 'tts --list_models' to see all available models."
        )
