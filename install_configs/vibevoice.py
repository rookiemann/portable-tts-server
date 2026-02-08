"""VibeVoice installation configuration."""

from .base import InstallConfig


class VibeVoiceConfig(InstallConfig):
    """Configuration for VibeVoice environment."""

    def __init__(self):
        super().__init__()
        self.name = "vibevoice_env"
        self.display_name = "VibeVoice"
        self.description = (
            "VibeVoice: Microsoft's TTS model. Strong emotion control, "
            "90-minute generation capability. Community fork available."
        )
        self.verify_package = "transformers"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch with CUDA 12.1 (torch 2.7.1 not available, using latest)
            ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"],

            # Core dependencies with specific versions
            ["transformers==4.51.3"],
            ["accelerate==1.6.0"],
            ["diffusers"],
            ["librosa"],
            ["soundfile"],
            ["gradio"],
            ["scipy"],
            ["numpy"],
        ]

        # Git repos to clone - community fork (original Microsoft repo removed)
        self.git_repos = [
            ("https://github.com/TheLocalLab/VibeVoice-Fork", True),
        ]

        self.system_notes = (
            "Note: This uses a community fork as Microsoft removed the original code. "
            "Model weights are still available on HuggingFace: microsoft/VibeVoice-1.5B"
        )
