from pathlib import Path

import edge_yolo
import edge_yolo.cli
import edge_yolo.detectors
import edge_yolo.export_tensorrt

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_without_optional_runtimes():
    assert edge_yolo.__version__
    assert edge_yolo.cli.main
    assert edge_yolo.detectors.open_detector
    assert edge_yolo.export_tensorrt.build_engine


def test_ci_workflow_lints_and_tests():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check ." in text
    assert "ruff format --check ." in text
    assert "pytest" in text
    assert "ubuntu-latest" in text


def test_results_table_has_no_measurements():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Results", 1)[1].split("##", 1)[0]
    table = [line for line in section.splitlines() if line.startswith("|")]
    assert len(table) >= 3
    _header, separator, *rows = table
    assert "---" in separator
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells
        assert all(cell == "" for cell in cells)
