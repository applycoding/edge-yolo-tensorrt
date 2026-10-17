"""Small CUDA memory helpers for TensorRT execution and INT8 calibration.

cuda-python has shipped two import paths. JetPack 6 images may have either
`cuda.bindings.runtime` or the older `cuda.cudart` module.
"""

from typing import Any

import numpy as np

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart
    except ImportError:
        cudart = None


def require_cuda():
    if cudart is None:
        raise RuntimeError(
            "cuda-python is not installed. On Jetson, install the CUDA Python "
            "bindings that match the JetPack CUDA version."
        )
    return cudart


class DeviceBuffer:
    """One reusable device allocation. TensorRT reads it until the next batch."""

    def __init__(self) -> None:
        self._ptr = 0
        self._nbytes = 0

    def upload(self, host: np.ndarray) -> int:
        array = np.ascontiguousarray(host)
        if array.nbytes > self._nbytes:
            self.close()
            self._ptr = malloc(array.nbytes)
            self._nbytes = array.nbytes
        memcpy_host_to_device(self._ptr, array)
        return int(self._ptr)

    def close(self) -> None:
        if self._ptr:
            free(self._ptr)
            self._ptr = 0
            self._nbytes = 0


def malloc(nbytes: int) -> int:
    runtime = require_cuda()
    pointer = _check(runtime.cudaMalloc(nbytes))
    return int(pointer)


def free(pointer: int) -> None:
    runtime = require_cuda()
    _check(runtime.cudaFree(pointer))


def memcpy_host_to_device(pointer: int, host: np.ndarray) -> None:
    runtime = require_cuda()
    _check(
        runtime.cudaMemcpy(
            pointer,
            host,
            host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
    )


def memcpy_device_to_host(host: np.ndarray, pointer: int) -> None:
    runtime = require_cuda()
    _check(
        runtime.cudaMemcpy(
            host,
            pointer,
            host.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
    )


def stream_create() -> int:
    runtime = require_cuda()
    return int(_check(runtime.cudaStreamCreate()))


def stream_synchronize(stream: int) -> None:
    runtime = require_cuda()
    _check(runtime.cudaStreamSynchronize(stream))


def stream_destroy(stream: int) -> None:
    runtime = require_cuda()
    _check(runtime.cudaStreamDestroy(stream))


def _check(result: Any) -> Any:
    runtime = require_cuda()
    if isinstance(result, tuple):
        error = result[0]
        payload = result[1:]
    else:
        error = result
        payload = ()
    if int(error) != int(runtime.cudaError_t.cudaSuccess):
        name = runtime.cudaGetErrorName(error)
        if isinstance(name, tuple):
            name = name[-1]
        raise RuntimeError(f"CUDA error: {name}")
    if len(payload) == 1:
        return payload[0]
    if len(payload) > 1:
        return payload
    return None
