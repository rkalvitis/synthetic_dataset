"""Process memory: read the resident set size, and cap it (`resources.max_ram_gb`).

- Windows: a Job Object with JOB_OBJECT_LIMIT_PROCESS_MEMORY is a hard cap on committed memory; an allocation
  beyond it fails with MemoryError instead of swapping the machine to death.
- Linux / macOS: no hard cap is set (RLIMIT_AS breaks CUDA's large virtual reservations); the generator's
  watchdog (`check_limit`) aborts cleanly when the RSS goes over the limit instead.
Nothing here needs psutil.
"""
from __future__ import annotations

import ctypes
import os
import sys

GB = 1024 ** 3


def rss_bytes() -> int:
    """Current resident set size of this process in bytes (0 if unknown)."""
    if sys.platform == "win32":
        return _win_mem()[0]
    try:
        if sys.platform == "linux":
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)   # macOS reports peak, in bytes
    except Exception:  # noqa: BLE001
        return 0


def peak_rss_bytes() -> int:
    """Peak resident set size so far, in bytes (0 if unknown)."""
    if sys.platform == "win32":
        return _win_mem()[1]
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(ru) if sys.platform == "darwin" else int(ru) * 1024
    except Exception:  # noqa: BLE001
        return 0


def _win_mem() -> tuple[int, int]:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        h = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize), int(pmc.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001
        pass
    return 0, 0


def limit_memory(max_gb: float | None) -> str:
    """Apply a hard RAM cap where the platform allows it. Returns a one-line description of what was done."""
    if not max_gb:
        return "no RAM limit"
    if sys.platform != "win32":
        return f"RAM limit {max_gb:g} GB: watchdog only (no hard cap on {sys.platform})"
    try:
        return _win_job_limit(float(max_gb))
    except Exception as e:  # noqa: BLE001
        return f"RAM limit {max_gb:g} GB: watchdog only (job object failed: {e})"


def _win_job_limit(max_gb: float) -> str:
    k32 = ctypes.windll.kernel32
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JobObjectExtendedLimitInformation = 9

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                  "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    # 64-bit handles: declare the signatures, otherwise ctypes truncates them to int
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(f"CreateJobObjectW failed ({ctypes.GetLastError()})")
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(max_gb * GB)
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(f"SetInformationJobObject failed ({ctypes.GetLastError()})")
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        raise OSError(f"AssignProcessToJobObject failed ({ctypes.GetLastError()}); already in a job without nesting?")
    return f"RAM limit {max_gb:g} GB: hard cap (Windows job object) + watchdog"


def check_limit(max_gb: float | None, what: str = "") -> None:
    """Raise MemoryError with a helpful message if the RSS is above the limit."""
    if not max_gb:
        return
    rss = rss_bytes()
    if rss > max_gb * GB:
        raise MemoryError(f"{what}: process uses {rss / GB:.1f} GB > resources.max_ram_gb={max_gb:g}. "
                          f"Lower resources.sprite_cache_gb / occluder_cache_gb or compose.image_size.")


def fmt(bytes_: int) -> str:
    return f"{bytes_ / GB:.2f} GB" if bytes_ else "n/a"
