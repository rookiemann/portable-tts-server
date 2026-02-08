# job_manager.py
"""Job tracking, recovery, retry, and cancellation for TTS inference."""

import json
import uuid
import threading
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class JobManager:
    """Manages TTS job lifecycle: creation, progress tracking, cancellation, and recovery.

    Job directory structure:
        {jobs_dir}/{job_id}/
            job.json
            chunk_000.wav, chunk_001.wav, ...
            {stem}_final.{format}
    """

    def __init__(self, jobs_dir: Path):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._cancel_flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def create_job(self, model: str, text: str, chunks: list[str],
                   params: dict, output_format: str = "wav",
                   sample_rate: int = 24000, stem: str | None = None,
                   job_dir: Path | None = None) -> dict:
        """Create a new job with tracking metadata.

        Args:
            model: Model identifier (xtts, fish, kokoro).
            text: Full original input text.
            chunks: List of text chunks after splitting.
            params: Request parameters dict (saved for recovery).
            output_format: Target audio format.
            sample_rate: Audio sample rate.
            stem: Base name for the final output file.
            job_dir: Override job directory (for save_path jobs).

        Returns:
            The job payload dict.
        """
        job_id = str(uuid.uuid4())

        if job_dir is None:
            job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if stem is None:
            stem = f"{model}_{int(datetime.utcnow().timestamp())}"

        job_payload = {
            "job_id": job_id,
            "model": model,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "running",
            "input_text": text,
            "total_chunks": len(chunks),
            "chunks_completed": 0,
            "total_duration_sec": None,
            "sample_rate": sample_rate,
            "output_format": output_format,
            "final_file": None,
            "expected_files": [f"chunk_{i:03d}.wav" for i in range(len(chunks))]
                              + [f"{stem}_final.{output_format}"],
            "missing_files": [f"chunk_{i:03d}.wav" for i in range(len(chunks))]
                             + [f"{stem}_final.{output_format}"],
            "chunks": [
                {
                    "index": i,
                    "text": c,
                    "char_length": len(c),
                    "duration_sec": None,
                    "file": f"chunk_{i:03d}.wav",
                    "verification_passed": None,
                    "whisper_transcript": None,
                    "whisper_similarity": None,
                    "processing_error": None,
                }
                for i, c in enumerate(chunks)
            ],
            "parameters": params,
            "failure_reason": None,
            "job_dir": str(job_dir),
        }

        job_file = job_dir / "job.json"
        self._write_job(job_file, job_payload)

        # Set up cancellation flag
        with self._lock:
            self._cancel_flags[job_id] = threading.Event()

        logger.info("Created job %s (%s, %d chunks)", job_id, model, len(chunks))
        return job_payload

    def update_chunk(self, job_id: str, chunk_idx: int, status: str = "success",
                     duration: float | None = None, audio_file: str | None = None,
                     error: str | None = None,
                     whisper_transcript: str | None = None,
                     whisper_similarity: float | None = None,
                     verification_passed: bool | None = None) -> None:
        """Update a single chunk's status in job.json.

        Args:
            job_id: The job UUID.
            chunk_idx: Zero-based chunk index.
            status: 'success' or 'failed'.
            duration: Audio duration in seconds.
            audio_file: Filename of the chunk WAV.
            error: Error message if failed.
            whisper_transcript: Whisper transcription result.
            whisper_similarity: Similarity ratio.
            verification_passed: Whether Whisper verification passed.
        """
        job_file = self._find_job_file(job_id)
        if not job_file:
            return

        try:
            job = self._read_job(job_file)
            chunk = job["chunks"][chunk_idx]

            if status == "success":
                chunk["duration_sec"] = round(duration, 3) if duration else None
                chunk["verification_passed"] = verification_passed if verification_passed is not None else True
                chunk["processing_error"] = None
                job["chunks_completed"] = chunk_idx + 1
                chunk_file_name = f"chunk_{chunk_idx:03d}.wav"
                if chunk_file_name in job["missing_files"]:
                    job["missing_files"].remove(chunk_file_name)
            else:
                chunk["processing_error"] = error
                chunk["verification_passed"] = verification_passed if verification_passed is not None else False

            if whisper_transcript is not None:
                chunk["whisper_transcript"] = whisper_transcript
            if whisper_similarity is not None:
                chunk["whisper_similarity"] = round(whisper_similarity, 4)

            self._write_job(job_file, job)
        except Exception as e:
            logger.warning("Failed to update chunk %d of job %s: %s", chunk_idx, job_id, e)

    def complete_job(self, job_id: str, output_file: str, total_duration: float) -> None:
        """Mark a job as completed."""
        job_file = self._find_job_file(job_id)
        if not job_file:
            return
        try:
            job = self._read_job(job_file)
            job["status"] = "completed"
            job["final_file"] = output_file
            job["total_duration_sec"] = round(total_duration, 3)
            job["missing_files"] = []
            self._write_job(job_file, job)
            logger.info("Job %s completed: %s (%.1fs)", job_id, output_file, total_duration)
        except Exception as e:
            logger.warning("Failed to complete job %s: %s", job_id, e)

    def fail_job(self, job_id: str, reason: str) -> None:
        """Mark a job as failed."""
        job_file = self._find_job_file(job_id)
        if not job_file:
            return
        try:
            job = self._read_job(job_file)
            job["status"] = "failed"
            job["failure_reason"] = reason
            self._write_job(job_file, job)
            logger.warning("Job %s failed: %s", job_id, reason)
        except Exception as e:
            logger.warning("Failed to mark job %s as failed: %s", job_id, e)

    def recover_job(self, job_dir: str | Path) -> dict | None:
        """Load a failed/incomplete job for re-processing.

        Args:
            job_dir: Path to the job directory containing job.json.

        Returns:
            The job dict with status reset to 'running', or None if not recoverable.
        """
        job_dir = Path(job_dir)
        job_file = job_dir / "job.json"
        if not job_file.exists():
            logger.warning("No job.json found in %s", job_dir)
            return None

        try:
            job = self._read_job(job_file)
        except Exception as e:
            logger.error("job.json corrupted in %s: %s", job_dir, e)
            return None

        if job.get("chunks_completed", 0) >= job.get("total_chunks", 0):
            logger.info("Job in %s is already finished", job_dir)
            return None

        # Reset status for re-processing
        job["status"] = "running"
        job["failure_reason"] = None

        # Reset incomplete chunks
        for chunk in job["chunks"][job["chunks_completed"]:]:
            chunk["processing_error"] = None
            chunk["verification_passed"] = None
            chunk["whisper_transcript"] = None
            chunk["whisper_similarity"] = None

        self._write_job(job_file, job)

        # Set up cancellation flag
        job_id = job["job_id"]
        with self._lock:
            self._cancel_flags[job_id] = threading.Event()

        logger.info("Recovering job %s from chunk %d/%d",
                    job_id, job["chunks_completed"], job["total_chunks"])
        return job

    def is_cancelled(self, job_id: str) -> bool:
        """Thread-safe check if a job has been cancelled."""
        with self._lock:
            event = self._cancel_flags.get(job_id)
        if event is None:
            return False
        return event.is_set()

    def request_cancel(self, job_id: str) -> bool:
        """Set the cancel flag for a running job.

        Returns:
            True if the job was found and cancel was set, False otherwise.
        """
        with self._lock:
            event = self._cancel_flags.get(job_id)
        if event is None:
            return False
        event.set()
        logger.info("Cancel requested for job %s", job_id)
        return True

    def get_job(self, job_id: str) -> dict | None:
        """Load job data by ID."""
        job_file = self._find_job_file(job_id)
        if job_file:
            return self._read_job(job_file)
        return None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """List recent jobs with summary info."""
        jobs = []
        if not self.jobs_dir.exists():
            return jobs

        # Scan jobs_dir for job.json files
        job_dirs = sorted(self.jobs_dir.iterdir(), key=lambda p: p.stat().st_mtime
                          if p.is_dir() else 0, reverse=True)

        for d in job_dirs[:limit]:
            if not d.is_dir():
                continue
            job_file = d / "job.json"
            if job_file.exists():
                try:
                    job = self._read_job(job_file)
                    jobs.append({
                        "job_id": job.get("job_id"),
                        "model": job.get("model"),
                        "status": job.get("status"),
                        "total_chunks": job.get("total_chunks"),
                        "chunks_completed": job.get("chunks_completed"),
                        "timestamp": job.get("timestamp"),
                        "failure_reason": job.get("failure_reason"),
                    })
                except Exception:
                    pass

        # Also scan projects_output for jobs
        try:
            from config import PROJECTS_OUTPUT
            if PROJECTS_OUTPUT.exists():
                for d in sorted(PROJECTS_OUTPUT.iterdir(),
                                key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
                                reverse=True):
                    if not d.is_dir():
                        continue
                    job_file = d / "job.json"
                    if job_file.exists():
                        try:
                            job = self._read_job(job_file)
                            if not any(j["job_id"] == job.get("job_id") for j in jobs):
                                jobs.append({
                                    "job_id": job.get("job_id"),
                                    "model": job.get("model"),
                                    "status": job.get("status"),
                                    "total_chunks": job.get("total_chunks"),
                                    "chunks_completed": job.get("chunks_completed"),
                                    "timestamp": job.get("timestamp"),
                                    "failure_reason": job.get("failure_reason"),
                                    "job_dir": str(d),
                                })
                        except Exception:
                            pass
        except ImportError:
            pass

        return jobs[:limit]

    def cleanup_cancel_flag(self, job_id: str) -> None:
        """Remove the cancel flag for a completed/failed job."""
        with self._lock:
            self._cancel_flags.pop(job_id, None)

    # --- Internal helpers ---

    def _find_job_file(self, job_id: str) -> Path | None:
        """Find job.json by job_id, searching jobs_dir and projects_output."""
        # Direct path
        direct = self.jobs_dir / job_id / "job.json"
        if direct.exists():
            return direct

        # Scan all subdirs
        for d in self.jobs_dir.iterdir():
            if d.is_dir():
                jf = d / "job.json"
                if jf.exists():
                    try:
                        data = self._read_job(jf)
                        if data.get("job_id") == job_id:
                            return jf
                    except Exception:
                        pass

        # Check projects_output
        try:
            from config import PROJECTS_OUTPUT
            if PROJECTS_OUTPUT.exists():
                for d in PROJECTS_OUTPUT.iterdir():
                    if d.is_dir():
                        jf = d / "job.json"
                        if jf.exists():
                            try:
                                data = self._read_job(jf)
                                if data.get("job_id") == job_id:
                                    return jf
                            except Exception:
                                pass
        except ImportError:
            pass

        return None

    @staticmethod
    def _read_job(job_file: Path) -> dict:
        with open(job_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_job(job_file: Path, data: dict) -> None:
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
