"""Chatterbox TTS installation configuration."""

from .base import InstallConfig


class ChatterboxConfig(InstallConfig):
    """Configuration for Chatterbox TTS environment."""

    def __init__(self):
        super().__init__()
        self.name = "chatterbox_env"
        self.display_name = "Chatterbox TTS"
        self.description = (
            "Chatterbox TTS by Resemble AI: High-quality voice cloning and TTS. "
            "Supports emotion/exaggeration control, voice cloning from short audio samples."
        )
        self.verify_package = "chatterbox-tts"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch 2.6.0 with CUDA 12.4 (must match chatterbox's pinned version)
            # Note: torch 2.6.0 not available for cu121, using cu124
            ["torch==2.6.0", "torchaudio==2.6.0",
             "--index-url", "https://download.pytorch.org/whl/cu124"],

            # Chatterbox TTS - will install all other pinned dependencies
            # (transformers==4.46.3, diffusers==0.29.0, librosa==0.11.0, etc.)
            ["chatterbox-tts"],
        ]

        # No git repos needed - pip installable
        self.git_repos = []

        self.system_notes = (
            "Chatterbox uses transformers==4.46.3 and torch==2.6.0. "
            "Model weights downloaded automatically from HuggingFace on first use."
        )
