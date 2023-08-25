# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import sys
from ctypes import CDLL, byref, c_char_p, c_uint
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

# nvml constants
NVML_SUCCESS = 0
NVML_DRIVER_NOT_LOADED = 9

# Some constants taken from cuda.h
CUDA_SUCCESS = 0

_is_windows: bool = sys.platform.startswith("win")
_is_wsl: bool = "WSL_DISTRO_NAME" in os.environ


def _load_nv_library(*libnames):
    for lib in libnames:
        try:
            return CDLL(lib)
        except OSError:
            continue


_cuda_lib = _nvml_lib = None

_init_pid = None
_gpu_count = None

_no_device_warned = False


class NVError(Exception):
    def __init__(self, msg, *args, errno=None):
        self._errno = errno
        super().__init__(msg or "Unknown error", *args)

    def __str__(self):
        return f"({self._errno}) {super().__str__()}"

    @property
    def errno(self):
        return self._errno

    @property
    def message(self):
        return super().__str__()


class NVDeviceAPIError(NVError):
    pass


class NVMLAPIError(NVError):
    pass


def _cu_check_error(result):
    if result != CUDA_SUCCESS:
        _error_str = c_char_p()
        _cuda_lib.cuGetErrorString(result, byref(_error_str))
        err_value = _error_str.value.decode() if _error_str.value is not None else None
        raise NVDeviceAPIError(err_value, errno=result)


_nvmlErrorString = None


def _nvml_check_error(result):
    global _nvmlErrorString
    if _nvmlErrorString is None:
        _nvmlErrorString = _nvml_lib.nvmlErrorString
        _nvmlErrorString.restype = c_char_p

    if result != NVML_SUCCESS:
        _error_str = _nvmlErrorString(result)
        raise NVMLAPIError(_error_str.decode(), errno=result)


def _init_nvml():
    global _nvml_lib, _no_device_warned
    if _init_pid == os.getpid():
        return

    nvml_paths = [
        "libnvidia-ml.so",
        "libnvidia-ml.so.1",
        "libnvidia-ml.dylib",
        "nvml.dll",
    ]
    if _is_windows:
        nvml_paths.append(
            os.path.join(
                os.getenv("ProgramFiles", "C:/Program Files"),
                "NVIDIA Corporation/NVSMI/nvml.dll",
            )
        )
    if _is_wsl:
        nvml_paths = ["/usr/lib/wsl/lib/libnvidia-ml.so.1"] + nvml_paths
    _nvml_lib = _load_nv_library(*nvml_paths)

    if _nvml_lib is None:
        return
    try:
        _nvml_check_error(_nvml_lib.nvmlInit_v2())
    except NVMLAPIError as ex:
        if ex.errno == NVML_DRIVER_NOT_LOADED:
            _nvml_lib = None
            if not _no_device_warned:
                logger.warning(
                    "Failed to load libnvidia-ml: %s, no CUDA device will be enabled",
                    ex.message,
                )
                _no_device_warned = True
        else:
            logger.exception("Failed to initialize libnvidia-ml.")
        return


def get_device_count() -> Optional[int]:
    global _gpu_count

    if _gpu_count is not None:
        return _gpu_count

    _init_nvml()
    if _nvml_lib is None:
        return None

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if not devices or devices == "-1":
            _gpu_count = 0
        else:
            _gpu_count = len(devices.split(","))
    else:
        n_gpus = c_uint()
        _cu_check_error(_nvml_lib.nvmlDeviceGetCount(byref(n_gpus)))
        _gpu_count = n_gpus.value
    return _gpu_count


def cpu_count():
    if "MARS_CPU_TOTAL" in os.environ:
        _cpu_total = int(math.ceil(float(os.environ["MARS_CPU_TOTAL"])))
    else:
        _cpu_total = psutil.cpu_count(logical=True)
    return _cpu_total


def cuda_count():
    return get_device_count() or 0
