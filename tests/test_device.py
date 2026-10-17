from pathlib import Path

from edge_yolo.device import DeviceInfo, SystemDeviceSource, probe_device, read_tegra_release


def test_tegra_release_reader(tmp_path: Path):
    missing = tmp_path / "absent"
    assert read_tegra_release(missing) is None
    present = tmp_path / "nv_tegra_release"
    present.write_text("# R36 , REVISION: 2.0\n", encoding="utf-8")
    assert read_tegra_release(present) == "# R36 , REVISION: 2.0"


def test_device_label_prefers_the_board_then_the_gpu():
    jetson = DeviceInfo("aarch64", "ARM", "# R36 REVISION: 2.0", "Orin")
    gpu = DeviceInfo("x86_64", "cpu", None, "Example GPU")
    cpu = DeviceInfo("x86_64", "cpu", None, None)
    assert jetson.label == "Orin | # R36 REVISION: 2.0"
    assert gpu.label == "Example GPU"
    assert cpu.label == "cpu"


def test_probe_uses_the_injected_source():
    class Source(SystemDeviceSource):
        def machine(self) -> str:
            return "aarch64"

        def processor(self) -> str:
            return "arm"

        def tegra_release(self) -> str | None:
            return "R36"

        def gpu_name(self) -> str | None:
            return None

    info = probe_device(Source())
    assert info.label == "Jetson | R36"


def test_gpu_name_skips_a_missing_nvidia_smi(monkeypatch):
    monkeypatch.setattr("edge_yolo.device.shutil.which", lambda _name: None)
    assert SystemDeviceSource().gpu_name() is None
