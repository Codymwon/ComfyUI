import operator
import os
import sys
import tempfile
import uuid

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _kernel32.DeviceIoControl.restype = wintypes.BOOL
    _kernel32.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.SetFileInformationByHandle.restype = wintypes.BOOL


elif sys.platform == "linux":
    if hasattr(os, "memfd_create"):
        _linux_memfd_create = os.memfd_create
        _MFD_CLOEXEC = getattr(os, "MFD_CLOEXEC", 1)
    else:
        import ctypes
        _libc = ctypes.CDLL(None, use_errno=True)
        if hasattr(_libc, "memfd_create"):
            _libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
            _libc.memfd_create.restype = ctypes.c_int

            def _linux_memfd_create(name, flags):
                name_bytes = name.encode("utf-8") if isinstance(name, str) else name
                fd = _libc.memfd_create(name_bytes, flags)
                if fd < 0:
                    err = ctypes.get_errno()
                    raise OSError(err, os.strerror(err))
                return fd

            _MFD_CLOEXEC = 1
        else:
            import platform
            machine = platform.machine().lower()
            _syscall_map = {"x86_64": 319, "amd64": 319, "aarch64": 279, "arm64": 279, "i386": 356, "i686": 356, "armv7l": 385}
            _sys_memfd_create = _syscall_map.get(machine, None)
            if _sys_memfd_create is not None:
                def _linux_memfd_create(name, flags):
                    name_bytes = name.encode("utf-8") if isinstance(name, str) else name
                    fd = _libc.syscall(_sys_memfd_create, name_bytes, flags)
                    if fd < 0:
                        err = ctypes.get_errno()
                        raise OSError(err, os.strerror(err))
                    return fd

                _MFD_CLOEXEC = 1
            else:
                _linux_memfd_create = None
                _MFD_CLOEXEC = 1


def create_anonymous_file(size=0):
    """Return an unbuffered binary file owned by the caller, initially sized in bytes.

    Uses Linux memfd or a Windows temporary-cache/delete-on-close file.
    Windows may still write to disk; Linux memfd may be swapped.
    Initial sizing is sparse with zero-filled holes; Windows requires a filesystem
    supporting sparse files. Keep the file open through all I/O and close
    mappings before closing it. No persistent pathname is provided.
    """
    size = operator.index(size)
    if size < 0 or size > 0x7fffffffffffffff:
        raise ValueError("size must fit a nonnegative signed 64-bit file offset")

    if os.name == "nt":
        flags = os.O_TEMPORARY | os.O_SHORT_LIVED
        path = os.path.join(tempfile.gettempdir(), f"aimdo-{uuid.uuid4().hex}")
        file = open(path, "x+b", buffering=0,
                    opener=lambda path, mode: os.open(path, mode | flags, 0o600))
    elif sys.platform == "linux":
        file = None
        if _linux_memfd_create is not None:
            try:
                file = open("aimdo", "w+b", buffering=0,
                            opener=lambda name, mode: _linux_memfd_create(name, _MFD_CLOEXEC))
            except OSError as err:
                if err.errno != 38:  # ENOSYS
                    raise
        if file is None:
            path = os.path.join(tempfile.gettempdir(), f"aimdo-{uuid.uuid4().hex}")
            file = open(path, "x+b", buffering=0)
            try:
                os.unlink(path)
            except OSError:
                pass
    else:
        raise NotImplementedError("anonymous files require Linux or Windows")

    try:
        if os.name == "nt":
            handle = msvcrt.get_osfhandle(file.fileno())
            transferred = wintypes.DWORD()
            # FSCTL_SET_SPARSE: unwritten ranges stay unallocated and read as zeros.
            if not _kernel32.DeviceIoControl(handle, 0x900c4, None, 0, None, 0,
                                             ctypes.byref(transferred), None):
                raise ctypes.WinError(ctypes.get_last_error())
            # FileEndOfFileInfo avoids the CRT truncate implementation's zero-fill loop.
            end = ctypes.c_longlong(size)
            if not _kernel32.SetFileInformationByHandle(handle, 6,
                                                       ctypes.byref(end), ctypes.sizeof(end)):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            file.truncate(size)
        return file
    except BaseException:
        file.close()
        raise
