"""Worker Registry - Tracks active workers, their status, and provides load balancing."""

import subprocess
import threading
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """State for a single worker subprocess."""
    worker_id: str          # e.g. "kokoro-1", "kokoro-2", "xtts-1"
    model: str              # "kokoro"
    port: int               # 8102
    device: str             # "cuda:0", "cuda:1", "cpu"
    process: subprocess.Popen | None = None
    log_fh: object = None  # open file handle for worker log (closed on cleanup)
    status: str = "starting"  # "starting" | "loading" | "ready" | "busy" | "dead"
    current_job: str | None = None
    last_health: float = 0.0
    health_failures: int = 0
    vram_used_mb: int = 0
    vram_total_mb: int = 0


class WorkerRegistry:
    """Thread-safe registry of all active workers with port pool and load balancing."""

    def __init__(self, port_min: int = 8101, port_max: int = 8200):
        self._lock = threading.Lock()
        self._workers: dict[str, WorkerInfo] = {}
        self._port_pool: set[int] = set(range(port_min, port_max + 1))
        self._model_counters: dict[str, int] = {}  # for generating worker IDs
        self._round_robin: dict[str, int] = {}  # round-robin index per model

    def allocate_port(self) -> int:
        """Allocate the next available port from the pool."""
        with self._lock:
            if not self._port_pool:
                raise RuntimeError("No available ports in pool (all 100 in use)")
            return self._port_pool.pop()

    def release_port(self, port: int) -> None:
        """Return a port to the pool."""
        with self._lock:
            self._port_pool.add(port)

    def next_worker_id(self, model: str) -> str:
        """Generate the next worker ID for a model (e.g. kokoro-1, kokoro-2)."""
        with self._lock:
            count = self._model_counters.get(model, 0) + 1
            self._model_counters[model] = count
            return f"{model}-{count}"

    def register(self, worker: WorkerInfo) -> None:
        """Add a worker to the registry."""
        with self._lock:
            self._workers[worker.worker_id] = worker
        logger.info("Registered worker %s (model=%s, port=%d, device=%s)",
                    worker.worker_id, worker.model, worker.port, worker.device)

    def unregister(self, worker_id: str) -> WorkerInfo | None:
        """Remove a worker from the registry. Returns the removed worker or None."""
        with self._lock:
            return self._workers.pop(worker_id, None)

    def get(self, worker_id: str) -> WorkerInfo | None:
        """Get a worker by ID."""
        with self._lock:
            return self._workers.get(worker_id)

    def get_ready_workers(self, model: str) -> list[WorkerInfo]:
        """Get all workers with status='ready' for a given model."""
        with self._lock:
            return [w for w in self._workers.values()
                    if w.model == model and w.status == "ready"]

    def pick_worker(self, model: str) -> WorkerInfo | None:
        """Round-robin among ready workers for a given model (single lock acquisition)."""
        with self._lock:
            ready = [w for w in self._workers.values()
                     if w.model == model and w.status == "ready"]
            if not ready:
                return None
            idx = self._round_robin.get(model, 0) % len(ready)
            self._round_robin[model] = idx + 1
            return ready[idx]

    def mark_busy(self, worker_id: str, job_id: str) -> None:
        """Mark a worker as busy with a specific job."""
        with self._lock:
            w = self._workers.get(worker_id)
            if w:
                w.status = "busy"
                w.current_job = job_id

    def mark_ready(self, worker_id: str) -> None:
        """Mark a worker as ready for new work."""
        with self._lock:
            w = self._workers.get(worker_id)
            if w:
                w.status = "ready"
                w.current_job = None

    def mark_dead(self, worker_id: str) -> None:
        """Mark a worker as dead."""
        with self._lock:
            w = self._workers.get(worker_id)
            if w:
                w.status = "dead"
                w.current_job = None

    def update_health(self, worker_id: str, vram_used_mb: int = 0,
                      vram_total_mb: int = 0) -> None:
        """Update health check data for a worker."""
        with self._lock:
            w = self._workers.get(worker_id)
            if w:
                w.last_health = time.time()
                w.health_failures = 0
                w.vram_used_mb = vram_used_mb
                w.vram_total_mb = vram_total_mb

    def record_health_failure(self, worker_id: str) -> int:
        """Record a health check failure. Returns the new failure count."""
        with self._lock:
            w = self._workers.get(worker_id)
            if w:
                w.health_failures += 1
                return w.health_failures
        return 0

    def all_workers(self) -> list[WorkerInfo]:
        """Get all registered workers."""
        with self._lock:
            return list(self._workers.values())

    def workers_for_model(self, model: str) -> list[WorkerInfo]:
        """Get all workers for a specific model (any status)."""
        with self._lock:
            return [w for w in self._workers.values() if w.model == model]

    def workers_on_device(self, device: str) -> list[WorkerInfo]:
        """Get all workers on a specific device/GPU."""
        with self._lock:
            return [w for w in self._workers.values() if w.device == device]

    def worker_count(self, model: str | None = None) -> int:
        """Count workers, optionally filtered by model."""
        with self._lock:
            if model:
                return sum(1 for w in self._workers.values() if w.model == model)
            return len(self._workers)

    def to_dict_list(self) -> list[dict]:
        """Serialize all workers to a list of dicts for API responses."""
        with self._lock:
            return [
                {
                    "worker_id": w.worker_id,
                    "model": w.model,
                    "port": w.port,
                    "device": w.device,
                    "status": w.status,
                    "current_job": w.current_job,
                    "pid": w.process.pid if w.process else None,
                    "vram_used_mb": w.vram_used_mb,
                    "vram_total_mb": w.vram_total_mb,
                }
                for w in self._workers.values()
            ]
