"""
memory_monitor.py
==================
Approximate memory measurement utilities for CPU (via psutil) and NVIDIA GPU
(via torch.cuda memory functions).

IMPORTANT — these measurements are approximate. Actual memory usage depends
on the operating system, memory allocator behavior, model loading strategy,
inference backend, batch size, numeric precision (fp32/fp16/int8), and the
specific GPU/CPU hardware in use. Treat these numbers as directional
indicators for comparing models under identical conditions on the same
machine, not as absolute guarantees for a production deployment.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _get_process_rss_mb() -> float:
    """Current process resident set size (RAM actually in use), in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def _get_gpu_memory_mb() -> float:
    """Current allocated GPU memory for this process, in MB. 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return 0.0


class MemoryMonitor:
    """
    Usage:
        mon = MemoryMonitor(device="cpu")   # or "cuda"
        mon.start()
        ... run inference ...
        stats = mon.stop()
        # stats = {"initial_mb": ..., "peak_mb": ..., "increase_mb": ...}
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.initial_mb = None
        self.peak_mb = None

    def _current_mb(self) -> float:
        if self.device == "cuda":
            return _get_gpu_memory_mb()
        return _get_process_rss_mb()

    def start(self):
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass
        self.initial_mb = self._current_mb()
        self.peak_mb = self.initial_mb

    def sample(self):
        """Call periodically or after inference to update the peak reading."""
        current = self._current_mb()
        if self.device == "cuda":
            try:
                import torch
                current = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except ImportError:
                pass
        self.peak_mb = max(self.peak_mb, current)

    def stop(self) -> dict:
        self.sample()
        return {
            "initial_mb": round(self.initial_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "increase_mb": round(self.peak_mb - self.initial_mb, 2),
        }


if __name__ == "__main__":
    mon = MemoryMonitor(device="cpu")
    mon.start()
    _ = [x ** 2 for x in range(2_000_000)]  # allocate some memory
    stats = mon.stop()
    print("Memory stats (approximate):", stats)
