# audio_processing.py
"""Unified 7-stage audio post-processing pipeline with graceful degradation.

Stages (each accepts/returns numpy array or operates in-place on file):
1. De-reverb        - noisereduce with first 0.2s as noise profile
2. 80Hz high-pass   - 4th-order Butterworth via scipy
3. De-esser         - Hilbert envelope follower, 3kHz crossover, 4:1 compression
4. Tempo adjustment - pyrubberband time-stretch (graceful skip if unavailable)
5. Silence trimming - pydub detect_silence with front/end protection zones
6. LUFS normalization - pyloudnorm to target LUFS
7. Peak limiting    - Hard clamp to clipping threshold
"""

import os
import time
import logging
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Stage 1: De-reverb
# ---------------------------------------------------------------------------
def _apply_de_reverb(data: np.ndarray, rate: int, strength: float = 0.7) -> np.ndarray:
    """Reduce reverb/room noise using first 0.2s as noise profile."""
    if strength <= 0.0 or len(data) <= int(rate * 0.2):
        return data
    try:
        import noisereduce as nr
        noise_clip = data[:int(rate * 0.2)]
        return nr.reduce_noise(y=data, sr=rate, y_noise=noise_clip, prop_decrease=strength)
    except ImportError:
        logger.warning("[%s POST] noisereduce not available, skipping de-reverb", _ts())
        return data


# ---------------------------------------------------------------------------
# Stage 2: High-pass filter
# ---------------------------------------------------------------------------
def _apply_highpass(data: np.ndarray, rate: int, cutoff: int = 80) -> np.ndarray:
    """4th-order Butterworth high-pass filter."""
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, cutoff, 'high', fs=rate, output='sos')
        return sosfiltfilt(sos, data)
    except ImportError:
        logger.warning("[%s POST] scipy not available, skipping high-pass", _ts())
        return data


# ---------------------------------------------------------------------------
# Stage 3: De-esser
# ---------------------------------------------------------------------------
def _apply_de_esser(data: np.ndarray, rate: int, strength: float = 0.0) -> np.ndarray:
    """Multiband de-esser: Hilbert envelope on 3kHz+ band, 4:1 compression."""
    if strength <= 0.0:
        return data
    strength = min(1.0, max(0.0, strength))
    try:
        from scipy.signal import butter, sosfiltfilt, hilbert
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        logger.warning("[%s POST] scipy not available, skipping de-esser", _ts())
        return data

    cutoff = 3000
    sos_high = butter(4, cutoff, 'high', fs=rate, output='sos')
    high = sosfiltfilt(sos_high, data)

    env = np.abs(hilbert(high))
    sigma = (rate * 5 / 1000) / 2.355
    env = gaussian_filter1d(env, sigma)

    env_db = 20 * np.log10(env + 1e-10)
    gain_db = np.where(env_db > -20, (env_db + 20) * (1 / 4 - 1), 0.0)
    gain = 10 ** (gain_db / 20.0)

    high_compressed = high * gain

    sos_low = butter(4, cutoff, 'low', fs=rate, output='sos')
    low = sosfiltfilt(sos_low, data)

    return (1 - strength) * data + strength * (low + high_compressed)


# ---------------------------------------------------------------------------
# Stage 4: Tempo adjustment
# ---------------------------------------------------------------------------
def _adjust_tempo(data: np.ndarray, rate: int, speed: float) -> np.ndarray:
    """Time-stretch without pitch change via pyrubberband."""
    if abs(speed - 1.0) < 1e-6:
        return data
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(data, rate, speed)
    except (ImportError, Exception) as e:
        logger.warning("[%s POST] Tempo adjust failed (%s), skipping", _ts(), e)
        return data


# ---------------------------------------------------------------------------
# Stage 5: Silence trimming
# ---------------------------------------------------------------------------
def _trim_silence(wav_path: str, profile: dict) -> None:
    """Trim leading/trailing silence with configurable protection zones. Modifies file in-place."""
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
    except ImportError:
        logger.warning("[%s POST] pydub not available, skipping silence trim", _ts())
        return

    trim_db = profile.get("trim_db", -35)
    min_silence = profile.get("min_silence_ms", 500)
    front_protect = profile.get("front_protect_ms", 100)
    end_protect = profile.get("end_protect_ms", 800)

    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        logger.warning("[%s POST] Failed to load audio for trimming: %s", _ts(), e)
        return

    sil = detect_silence(audio, min_silence_len=min_silence, silence_thresh=trim_db)

    start_trim = 0
    if sil and sil[0][0] == 0:
        front_ms = sil[0][1]
        start_trim = max(0, front_ms - front_protect)

    end_trim = 0
    if sil and sil[-1][1] == len(audio):
        tail_ms = len(audio) - sil[-1][0]
        end_trim = max(0, tail_ms - end_protect)

    if start_trim or end_trim:
        # Safety: ensure trimming doesn't produce empty audio
        result_end = len(audio) - end_trim
        if result_end <= start_trim:
            logger.warning("[%s POST] Trim would produce empty audio (start=%dms end=%dms total=%dms), skipping",
                           _ts(), start_trim, end_trim, len(audio))
            return
        trimmed = audio[start_trim:result_end]
        trimmed.export(wav_path, format="wav")
        logger.debug("[%s POST] Trimmed start=%dms end=%dms -> %dms", _ts(), start_trim, end_trim, len(trimmed))


# ---------------------------------------------------------------------------
# Stage 6: LUFS normalization
# ---------------------------------------------------------------------------
def _normalize_loudness(wav_path: str, target_lufs: float = -23.0) -> None:
    """Normalize integrated loudness to target LUFS. Modifies file in-place."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("[%s POST] pyloudnorm not available, skipping LUFS normalization", _ts())
        return

    try:
        data, rate = sf.read(wav_path)
    except Exception as e:
        logger.warning("[%s POST] Failed to load for normalization: %s", _ts(), e)
        return

    meter = pyln.Meter(rate)
    # pyloudnorm requires audio longer than 0.4s (block_size)
    min_samples = int(0.4 * rate) + 1
    if len(data) < min_samples:
        logger.warning("[%s POST] Audio too short for LUFS normalization (%d samples), skipping", _ts(), len(data))
        return
    loudness = meter.integrated_loudness(data)
    if np.isinf(loudness):
        logger.warning("[%s POST] Measured loudness is -inf, skipping normalization", _ts())
        return
    normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    sf.write(wav_path, normalized, rate, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Stage 7: Peak limiting
# ---------------------------------------------------------------------------
def _apply_peak_limit(wav_path: str, threshold: float = 0.95) -> None:
    """Hard clamp peaks to threshold. Modifies file in-place."""
    data, rate = sf.read(wav_path)
    peak = np.max(np.abs(data))
    if peak > threshold:
        data = data * (threshold / peak)
        sf.write(wav_path, data, rate, subtype="PCM_16")
        logger.debug("[%s POST] Peak limited %.6f -> %.3f", _ts(), peak, threshold)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def post_process(wav_path: str, profile: dict, speed: float = 1.0,
                 de_reverb: float = 0.7, de_ess: float = 0.0) -> str:
    """Run the full 7-stage post-processing pipeline on a WAV file.

    Args:
        wav_path: Path to the raw WAV file (modified in-place).
        profile: Audio profile dict from audio_profiles.PROFILES.
        speed: Playback speed factor (1.0 = original).
        de_reverb: Noise reduction strength (0.0 - 1.0).
        de_ess: De-esser strength (0.0 - 1.0).

    Returns:
        The same wav_path (file is modified in-place).
    """
    if not os.path.exists(wav_path):
        logger.warning("[%s POST] File not found: %s", _ts(), wav_path)
        return wav_path

    try:
        data, rate = sf.read(wav_path)
    except Exception as e:
        logger.warning("[%s POST] Failed to load %s: %s", _ts(), wav_path, e)
        return wav_path

    logger.info("[%s POST] Starting pipeline: speed=%.2f de_reverb=%.2f de_ess=%.2f",
                _ts(), speed, de_reverb, de_ess)

    # Stage 1: De-reverb
    data = _apply_de_reverb(data, rate, de_reverb)

    # Stage 2: High-pass
    data = _apply_highpass(data, rate, 80)

    # Stage 3: De-esser
    data = _apply_de_esser(data, rate, de_ess)

    # Stage 4: Tempo
    data = _adjust_tempo(data, rate, speed)

    # Save intermediate for pydub-based trimming
    sf.write(wav_path, data, rate, subtype="PCM_16")

    # Stage 5: Silence trimming
    _trim_silence(wav_path, profile)

    # Stage 6: LUFS normalization
    _normalize_loudness(wav_path, profile.get("lufs", -23.0))

    # Stage 7: Peak limiting
    _apply_peak_limit(wav_path, profile.get("clipping", 0.95))

    logger.info("[%s POST] Pipeline complete: %s", _ts(), wav_path)
    return wav_path


# ---------------------------------------------------------------------------
# Whisper verification
# ---------------------------------------------------------------------------
def verify_with_whisper(wav_path: str, original_text: str, language: str = "en",
                        tolerance: float = 80.0, whisper_model=None) -> tuple[bool, float, str]:
    """Verify generated audio matches expected text using Whisper.

    Args:
        wav_path: Path to the WAV file to verify.
        original_text: The text that was used to generate the audio.
        language: Language code for Whisper transcription.
        tolerance: Minimum similarity percentage (0-100) to pass.
        whisper_model: A loaded whisper model instance. If None, returns pass.

    Returns:
        Tuple of (passed, similarity_ratio, transcript).
        If whisper is unavailable, returns (True, 1.0, "").
    """
    if whisper_model is None:
        return (True, 1.0, "")

    try:
        import whisper
        from text_utils import sanitize_for_whisper
    except ImportError:
        logger.warning("[%s WHISPER] whisper not available, skipping verification", _ts())
        return (True, 1.0, "")

    try:
        data, _ = sf.read(wav_path)
        peak = np.max(np.abs(data))
        if peak > 1.0:
            logger.warning("[%s WHISPER] Audio clipped (peak=%.4f), rejecting", _ts(), peak)
            return (False, 0.0, "")
    except Exception as e:
        logger.warning("[%s WHISPER] Failed to read audio: %s", _ts(), e)
        return (False, 0.0, "")

    audio = whisper.load_audio(wav_path)
    result = whisper_model.transcribe(audio, language=language, fp16=False, word_timestamps=False)
    transcribed = result["text"].strip()

    orig_san = sanitize_for_whisper(original_text)
    trans_san = sanitize_for_whisper(transcribed)
    sim = SequenceMatcher(None, orig_san.split(), trans_san.split()).ratio()
    tolerance_norm = tolerance / 100.0
    passed = sim >= tolerance_norm

    logger.info("[%s WHISPER] Expected: \"%s\"", _ts(), original_text)
    logger.info("[%s WHISPER] Heard   : \"%s\"", _ts(), transcribed)
    logger.info("[%s WHISPER] Similarity %.4f >= %.2f -> %s",
                _ts(), sim, tolerance_norm, "PASS" if passed else "FAIL")

    return (passed, sim, transcribed)
