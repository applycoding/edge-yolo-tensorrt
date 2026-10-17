import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    machine: str
    processor: str
    tegra_release: str | None
    gpu_name: str | None

    @property
    def label(self) -> str:
        if self.tegra_release:
            name = self.gpu_name or "Jetson"
            return f"{name} | {self.tegra_release}"
        if self.gpu_name:
            return self.gpu_name
        return self.processor or self.machine


class SystemDeviceSource:
    def machine(self) -> str:
        return platform.machine()

    def processor(self) -> str:
        return platform.processor()

    def tegra_release(self) -> str | None:
        return read_tegra_release(Path("/etc/nv_tegra_release"))

    def gpu_name(self) -> str | None:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return lines[0] if lines else None


def read_tegra_release(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = " ".join(path.read_text(encoding="utf-8").split())
    return text or None


def probe_device(source: SystemDeviceSource | None = None) -> DeviceInfo:
    reader = source or SystemDeviceSource()
    return DeviceInfo(
        machine=reader.machine(),
        processor=reader.processor(),
        tegra_release=reader.tegra_release(),
        gpu_name=reader.gpu_name(),
    )
