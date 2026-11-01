"""glibc heap control for the session server.

A session holds MB-sized buffers: the backend's response bodies and the base64 routed-expert payloads of every turn.
glibc serves a block above its mmap threshold with mmap (returned to the OS on free), but the threshold starts at
128 KiB and rises, up to 32 MiB, each time such a block is freed. After the first sessions end, these buffers come from
the heap, whose freed pages stay resident: in train67 (2026-11-01) the 32 session servers held 39 -> ~70 GiB at B=16
through every training step, the host-memory peak of a colocated run. With the threshold fixed and a trim after each
session ends, the freed buffers go back to the OS (a replay of the pattern: 3.1 GB resident after free by default,
~20 MB with this).
"""

import ctypes
import logging

logger = logging.getLogger(__name__)

_M_MMAP_THRESHOLD = -3   # mallopt parameter (glibc malloc.h)

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:   # not glibc: nothing to tune
    _libc = None


def bound_heap(threshold: int = 1 << 20) -> bool:
    """Serve every allocation of `threshold` bytes or more with mmap, a fixed threshold (setting it also stops glibc
    from raising it). Returns whether glibc accepted it."""
    return bool(_libc is not None and _libc.mallopt(_M_MMAP_THRESHOLD, threshold))


def trim() -> None:
    """Give the heap's free pages back to the OS (malloc_trim(0))."""
    if _libc is not None:
        _libc.malloc_trim(0)
