"""Higgs Audio V2 installation configuration."""

from .base import InstallConfig


class HiggsAudioConfig(InstallConfig):
    """Configuration for Higgs Audio V2 environment."""

    def __init__(self):
        super().__init__()
        self.name = "higgs_env"
        self.display_name = "Higgs Audio V2"
        self.description = (
            "Higgs Audio V2: 75.7% win rate vs GPT-4o-mini TTS. "
            "SOTA on Seed-TTS benchmark. Auto-prosody, multi-speaker dialogue."
        )
        self.verify_package = "transformers"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch with CUDA 12.1
            ["torch", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu121"],

            # Core dependencies - specific transformers version
            ["transformers==4.47.0"],
            ["accelerate"],
            ["scipy"],
            ["soundfile"],
            ["librosa"],
            ["numpy"],
        ]

        # Git repos to clone
        self.git_repos = [
            ("https://github.com/boson-ai/higgs-audio", True),
        ]

        self.system_notes = (
            "Note: Higgs Audio V2 model weights (~10GB) download on first use from HuggingFace. "
            "Model: bosonai/higgs-audio-v2-generation-3B-base + bosonai/higgs-audio-v2-tokenizer"
        )
