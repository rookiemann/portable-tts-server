"""Worker Manager - Spawns, monitors, and kills worker subprocesses."""

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

import httpx

from config import (
    BASE_DIR, PYTHON_PATH, WORKER_PORT_MIN, WORKER_PORT_MAX,
    WORKER_HEALTH_INTERVAL, WORKER_STARTUP_TIMEOUT,
    WORKER_MAX_HEALTH_FAILURES, WORKER_LOG_DIR, WORKER_DEFAULT_DEVICE,
    MODEL_VENV_MAP, setup_environment,
)
from worker_registry import WorkerRegistry, WorkerInfo

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages the lifecycle of TTS worker subprocesses."""

    def __init__(self, registry: WorkerRegistry | None = None):
        self.registry = registry or WorkerRegistry(WORKER_PORT_MIN, WORKER_PORT_MAX)
        self._health_task: asyncio.Task | None = None
        WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    async def spawn_worker(self, model: str, device: str | None = None) -> WorkerInfo:
        """Spawn a new worker subprocess for the given model.

        Args:
            model: Model identifier (kokoro, xtts, higgs, etc.)
            device: CUDA device string (cuda:0, cuda:1, cpu). Defaults to config.

        Returns:
            WorkerInfo for the spawned worker.

        Raises:
            RuntimeError: If worker fails to start or become ready.
        """
        device = device or WORKER_DEFAULT_DEVICE
        port = self.registry.allocate_port()
        worker_id = self.registry.next_worker_id(model)

        worker_script = str(BASE_DIR / "tts_worker.py")
        python_exe = str(PYTHON_PATH)

        cmd = [
            python_exe, worker_script,
            "--model", model,
            "--port", str(port),
            "--device", device,
        ]

        # Set up environment for the subprocess
        env = os.environ.copy()
        # Ensure HF/torch cache dirs are set
        from config import MODELS_DIR
        env["HF_HOME"] = str(MODELS_DIR)
        env["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "hub")
        env["TORCH_HOME"] = str(MODELS_DIR / "torch")
        env["COQUI_TTS_CACHE"] = str(MODELS_DIR / "coqui")
        env["PYTHONUNBUFFERED"] = "1"  # flush worker logs immediately

        log_file = WORKER_LOG_DIR / f"worker_{model}_{port}.log"
        logger.info("Spawning worker %s: model=%s port=%d device=%s",
                    worker_id, model, port, device)

        try:
            log_fh = open(log_file, "w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(BASE_DIR),
            )
        except Exception as e:
            self.registry.release_port(port)
            if 'log_fh' in locals():
                log_fh.close()
            raise RuntimeError(f"Failed to launch worker process: {e}")

        worker = WorkerInfo(
            worker_id=worker_id,
            model=model,
            port=port,
            device=device,
            process=process,
            log_fh=log_fh,
            status="starting",
        )
        self.registry.register(worker)

        # Wait for worker FastAPI to be up, then load the model
        try:
            await self._wait_for_healthy(worker)
            await self._load_model(worker)
        except Exception:
            # Clean up on failure
            await self._force_kill(worker)
            self.registry.unregister(worker_id)
            self.registry.release_port(port)
            raise

        return worker

    async def _wait_for_healthy(self, worker: WorkerInfo) -> None:
        """Poll worker /health until FastAPI responds (model may not be loaded yet)."""
        url = f"http://127.0.0.1:{worker.port}/health"
        deadline = time.time() + WORKER_STARTUP_TIMEOUT

        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.time() < deadline:
                if worker.process and worker.process.poll() is not None:
                    raise RuntimeError(
                        f"Worker {worker.worker_id} process exited with code "
                        f"{worker.process.returncode} during startup"
                    )

                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        logger.info("Worker %s FastAPI is up (pid=%d)",
                                    worker.worker_id,
                                    worker.process.pid if worker.process else 0)
                        return
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                    pass

                await asyncio.sleep(1.0)

        raise RuntimeError(
            f"Worker {worker.worker_id} FastAPI did not start within "
            f"{WORKER_STARTUP_TIMEOUT}s"
        )

    async def _load_model(self, worker: WorkerInfo) -> None:
        """Call POST /load on the worker to load the model into memory."""
        load_url = f"http://127.0.0.1:{worker.port}/load"
        health_url = f"http://127.0.0.1:{worker.port}/health"

        worker.status = "loading"
        logger.info("Loading model on worker %s ...", worker.worker_id)

        # /load blocks until the model is fully loaded; use a long timeout
        async with httpx.AsyncClient(timeout=900.0) as client:
            try:
                resp = await client.post(load_url)
                if resp.status_code != 200:
                    detail = resp.text[:200]
                    raise RuntimeError(
                        f"Worker {worker.worker_id} /load returned "
                        f"{resp.status_code}: {detail}"
                    )
            except httpx.TimeoutException:
                raise RuntimeError(
                    f"Worker {worker.worker_id} model load timed out (900s)"
                )

            # Confirm health reports ready
            try:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    data = resp.json()
                    worker.vram_used_mb = data.get("vram_used_mb", 0)
                    worker.vram_total_mb = data.get("vram_total_mb", 0)
            except Exception:
                pass

        worker.status = "ready"
        worker.last_health = time.time()
        logger.info("Worker %s model loaded and ready", worker.worker_id)

    async def kill_worker(self, worker_id: str) -> bool:
        """Gracefully stop and remove a worker.

        Returns True if worker was found and killed.
        """
        worker = self.registry.get(worker_id)
        if not worker:
            return False

        logger.info("Killing worker %s (port=%d)", worker_id, worker.port)

        # Try graceful unload first
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"http://127.0.0.1:{worker.port}/unload")
        except Exception:
            pass

        await self._force_kill(worker)
        self.registry.unregister(worker_id)
        self.registry.release_port(worker.port)
        return True

    async def _force_kill(self, worker: WorkerInfo) -> None:
        """Terminate the worker subprocess without blocking the event loop."""
        if worker.process:
            try:
                worker.process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(worker.process.wait), timeout=5
                    )
                except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                    worker.process.kill()
                    await asyncio.wait_for(
                        asyncio.to_thread(worker.process.wait), timeout=5
                    )
            except Exception as e:
                logger.warning("Error killing worker %s: %s", worker.worker_id, e)
            worker.process = None
        # Close the log file handle
        if worker.log_fh:
            try:
                worker.log_fh.close()
            except Exception:
                pass
            worker.log_fh = None

    async def kill_all_workers(self) -> int:
        """Kill all active workers. Returns count killed."""
        workers = self.registry.all_workers()
        count = 0
        for w in workers:
            if await self.kill_worker(w.worker_id):
                count += 1
        return count

    async def scale_model(self, model: str, count: int,
                          device: str | None = None) -> list[WorkerInfo]:
        """Ensure exactly `count` workers exist for this model on the specified device.

        Args:
            model: Model identifier.
            count: Desired number of workers.
            device: GPU device. If None, uses default.

        Returns:
            List of all workers for this model on this device after scaling.
        """
        device = device or WORKER_DEFAULT_DEVICE

        # Get current workers for this model on this device
        current = [w for w in self.registry.workers_for_model(model)
                   if w.device == device and w.status != "dead"]
        current_count = len(current)

        if current_count < count:
            # Spawn more
            for _ in range(count - current_count):
                await self.spawn_worker(model, device)
        elif current_count > count:
            # Kill excess (kill most recent first)
            to_kill = current[count:]
            for w in to_kill:
                await self.kill_worker(w.worker_id)

        return [w for w in self.registry.workers_for_model(model)
                if w.device == device and w.status != "dead"]

    async def _cleanup_dead_worker(self, worker: WorkerInfo) -> None:
        """Clean up a dead worker: terminate process, release port, unregister."""
        logger.info("Cleaning up dead worker %s (port=%d)", worker.worker_id, worker.port)
        await self._force_kill(worker)
        self.registry.unregister(worker.worker_id)
        self.registry.release_port(worker.port)

    async def health_check_loop(self) -> None:
        """Periodic health check for all workers. Runs as an asyncio task."""
        logger.info("Worker health check loop started (interval=%ds)",
                    WORKER_HEALTH_INTERVAL)

        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    await asyncio.sleep(WORKER_HEALTH_INTERVAL)
                    workers = self.registry.all_workers()
                    to_cleanup = []

                    for w in workers:
                        if w.status == "dead":
                            to_cleanup.append(w)
                            continue
                        if w.status in ("busy", "starting", "loading"):
                            continue

                        # Check if process is still alive
                        if w.process and w.process.poll() is not None:
                            logger.warning("Worker %s process died (exit code %d)",
                                          w.worker_id, w.process.returncode)
                            to_cleanup.append(w)
                            continue

                        try:
                            resp = await client.get(
                                f"http://127.0.0.1:{w.port}/health"
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                self.registry.update_health(
                                    w.worker_id,
                                    vram_used_mb=data.get("vram_used_mb", 0),
                                    vram_total_mb=data.get("vram_total_mb", 0),
                                )
                            else:
                                failures = self.registry.record_health_failure(w.worker_id)
                                if failures >= WORKER_MAX_HEALTH_FAILURES:
                                    logger.warning(
                                        "Worker %s failed %d health checks, marking dead",
                                        w.worker_id, failures)
                                    to_cleanup.append(w)
                        except Exception:
                            failures = self.registry.record_health_failure(w.worker_id)
                            if failures >= WORKER_MAX_HEALTH_FAILURES:
                                logger.warning(
                                    "Worker %s unreachable %d times, marking dead",
                                    w.worker_id, failures)
                                to_cleanup.append(w)

                    for w in to_cleanup:
                        await self._cleanup_dead_worker(w)

                except asyncio.CancelledError:
                    logger.info("Health check loop cancelled")
                    break
                except Exception as e:
                    logger.error("Health check loop error: %s", e)

    def start_health_checks(self) -> None:
        """Start the health check background task."""
        if self._health_task is None or self._health_task.done():
            loop = asyncio.get_event_loop()
            self._health_task = loop.create_task(self.health_check_loop())

    def stop_health_checks(self) -> None:
        """Stop the health check background task."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()

    def detect_devices(self) -> list[dict]:
        """Detect available GPU devices using nvidia-smi.

        Returns list of device dicts with id, name, vram_total_mb, vram_free_mb,
        and list of worker_ids currently on each device.
        """
        devices = []

        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        idx = parts[0]
                        device_id = f"cuda:{idx}"
                        workers_on = [w.worker_id
                                      for w in self.registry.workers_on_device(device_id)]
                        devices.append({
                            "id": device_id,
                            "name": parts[1],
                            "vram_total_mb": int(float(parts[2])),
                            "vram_free_mb": int(float(parts[3])),
                            "workers": workers_on,
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("nvidia-smi not available: %s", e)

        # Always include CPU
        cpu_workers = [w.worker_id for w in self.registry.workers_on_device("cpu")]
        devices.append({
            "id": "cpu",
            "name": "CPU",
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "workers": cpu_workers,
        })

        return devices
