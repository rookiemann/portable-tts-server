# TTS Environment Installation Configurations
from .base import InstallConfig, run_pip_install, run_git_clone
from .qwen3_tts import Qwen3TTSConfig
from .vibevoice import VibeVoiceConfig
from .higgs_audio import HiggsAudioConfig
from .orpheus import OrpheusConfig
from .unified_tts import UnifiedTTSConfig
from .chatterbox import ChatterboxConfig
from .coqui_tts import CoquiTTSConfig
from .f5_tts import F5TTSConfig

ALL_CONFIGS = {
    "qwen3_env": Qwen3TTSConfig(),
    "vibevoice_env": VibeVoiceConfig(),
    "higgs_env": HiggsAudioConfig(),
    "orpheus_env": OrpheusConfig(),
    "unified_env": UnifiedTTSConfig(),
    "chatterbox_env": ChatterboxConfig(),
    "coqui_env": CoquiTTSConfig(),
    "f5tts_env": F5TTSConfig(),
}
