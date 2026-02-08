"""Orpheus TTS installation configuration."""

from .base import InstallConfig


class OrpheusConfig(InstallConfig):
    """Configuration for Orpheus TTS environment."""

    def __init__(self):
        super().__init__()
        self.name = "orpheus_env"
        self.display_name = "Orpheus TTS"
        self.description = (
            "Orpheus TTS: Comparable to ElevenLabs quality. "
            "Zero-shot voice cloning, emotion tags. Requires vLLM."
        )
        self.verify_package = "vllm"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch with CUDA 12.1
            ["torch", "--index-url", "https://download.pytorch.org/whl/cu121"],

            # vLLM - specific version for Orpheus
            ["vllm==0.7.3"],

            # Orpheus TTS
            ["orpheus-speech"],

            # Additional dependencies
            ["soundfile"],
            ["scipy"],
            ["numpy"],
        ]

        # No git repos needed
        self.git_repos = []

        self.system_notes = (
            "WARNING: Orpheus requires vLLM which has strict dependencies. "
            "Python 3.12 is NOT supported. Uses significant GPU memory."
        )
