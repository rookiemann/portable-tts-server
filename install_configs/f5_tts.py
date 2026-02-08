"""F5-TTS installation configuration."""

from .base import InstallConfig


class F5TTSConfig(InstallConfig):
    """Configuration for F5-TTS environment."""

    def __init__(self):
        super().__init__()
        self.name = "f5tts_env"
        self.display_name = "F5-TTS"
        self.description = (
            "F5-TTS: State-of-the-art diffusion-based TTS. "
            "Excellent zero-shot voice cloning, natural prosody, "
            "high-quality synthesis. Includes E2-TTS support."
        )
        self.verify_package = "f5_tts"

        # Installation packages in order
        self.pip_packages = [
            # PyTorch 2.8.0 with CUDA 12.6 (as recommended by F5-TTS)
            ["torch==2.8.0", "torchaudio==2.8.0",
             "--index-url", "https://download.pytorch.org/whl/cu126"],

            # F5-TTS package
            ["f5-tts"],
        ]

        # No git repos needed - pip installable
        self.git_repos = []

        self.system_notes = (
            "F5-TTS uses flow matching for high-quality synthesis. "
            "Model weights downloaded automatically on first use. "
            "CLI available: 'f5-tts_infer-cli' for inference."
        )
