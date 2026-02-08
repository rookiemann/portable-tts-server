"""Qwen3-TTS installation configuration."""

from .base import InstallConfig


class Qwen3TTSConfig(InstallConfig):
    """Configuration for Qwen3-TTS environment."""

    def __init__(self):
        super().__init__()
        self.name = "qwen3_env"
        self.display_name = "Qwen3-TTS"
        self.description = (
            "Qwen3-TTS: 97ms latency, 3-second voice cloning, "
            "natural language voice control. Best all-rounder."
        )
        self.verify_package = "transformers"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch with CUDA 12.1
            ["torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu121"],

            # Core dependencies
            ["transformers==4.57.3"],
            ["accelerate"],
            ["scipy"],
            ["soundfile"],
            ["librosa"],

            # Qwen TTS specific
            ["funasr"],
            ["modelscope"],
        ]

        # Git repos to clone
        self.git_repos = [
            # Qwen2.5-Omni repo contains the TTS model code
            ("https://github.com/QwenLM/Qwen2.5-Omni", True),
        ]

        self.system_notes = (
            "Note: Qwen3-TTS model weights will be downloaded on first use. "
            "Requires ~10GB disk space for model weights."
        )
