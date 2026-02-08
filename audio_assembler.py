# audio_assembler.py
"""Final audio assembly and format conversion."""

import subprocess
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def assemble_chunks(chunk_files: list[str | Path], output_path: str | Path,
                    sr: int, inter_pause: float = 0.25,
                    front_pad: float = 0.5, end_pad: float = 0.5) -> str:
    """Concatenate chunk WAV files with silence padding between them.

    Args:
        chunk_files: Ordered list of WAV file paths.
        output_path: Where to write the assembled WAV.
        sr: Sample rate.
        inter_pause: Seconds of silence between chunks.
        front_pad: Seconds of silence at the beginning.
        end_pad: Seconds of silence at the end.

    Returns:
        The output_path as a string.
    """
    parts = []
    for f in chunk_files:
        try:
            data, file_sr = sf.read(str(f))
            parts.append(data)
        except Exception as e:
            logger.warning("Failed to read chunk %s: %s", f, e)

    if not parts:
        logger.error("No audio chunks to assemble")
        # Write silence so downstream doesn't break
        silence = np.zeros(int(sr * 0.1), dtype=np.float32)
        sf.write(str(output_path), silence, sr, subtype="PCM_16")
        return str(output_path)

    inter = np.zeros(int(sr * inter_pause), dtype=np.float32)
    front = np.zeros(int(sr * front_pad), dtype=np.float32)
    end = np.zeros(int(sr * end_pad), dtype=np.float32)

    segments = [front, parts[0]]
    for p in parts[1:]:
        segments.append(inter)
        segments.append(p)
    segments.append(end)

    final_wav = np.concatenate(segments)
    sf.write(str(output_path), final_wav, sr, subtype="PCM_16")
    logger.info("Assembled %d chunks -> %s (%.1fs)", len(parts), output_path,
                len(final_wav) / sr)
    return str(output_path)


def _ffmpeg_args(fmt: str) -> list[str]:
    """Return ffmpeg encoding arguments for common output formats."""
    return {
        "mp3": ["-c:a", "libmp3lame", "-q:a", "0"],
        "ogg": ["-c:a", "libvorbis", "-q:a", "6"],
        "flac": ["-c:a", "flac", "-compression_level", "12"],
        "m4a": ["-c:a", "aac", "-b:a", "320k"],
    }.get(fmt, [])


def convert_format(input_path: str | Path, output_path: str | Path,
                   fmt: str = "wav", ffmpeg_path: str = "ffmpeg") -> str:
    """Convert audio file to the requested format via FFmpeg.

    Args:
        input_path: Source WAV file.
        output_path: Destination file path (should have the correct extension).
        fmt: Target format (wav, mp3, ogg, flac, m4a).
        ffmpeg_path: Path to the ffmpeg executable.

    Returns:
        The output_path as a string.
    """
    if fmt == "wav":
        # No conversion needed, just copy if different paths
        if str(input_path) != str(output_path):
            import shutil
            shutil.copy2(str(input_path), str(output_path))
        return str(output_path)

    cmd = [
        ffmpeg_path, "-y", "-i", str(input_path),
        *_ffmpeg_args(fmt),
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Converted %s -> %s", input_path, output_path)
    except FileNotFoundError:
        logger.error("FFmpeg not found at '%s', returning WAV", ffmpeg_path)
        import shutil
        shutil.copy2(str(input_path), str(output_path))
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg conversion failed: %s", e.stderr)
        import shutil
        shutil.copy2(str(input_path), str(output_path))

    return str(output_path)
