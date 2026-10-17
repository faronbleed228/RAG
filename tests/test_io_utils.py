"""Check UTF-8 JSON writes and preservation of previous results on failure."""

from pathlib import Path
from typing import Any

import pytest

from src import io_utils


def test_json_is_written_as_utf8(tmp_path: Path) -> None:
    """Preserve Unicode text when replacing an existing result file."""
    path = tmp_path / "results.json"
    path.write_text("old result", encoding="utf-8")
    content = '{"question": "ёжик 🐈"}'
    io_utils.write_json(path, content)
    assert path.read_bytes() == content.encode("utf-8")
    assert list(tmp_path.iterdir()) == [path]


def test_failed_write_preserves_previous_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the old file if writing the temporary replacement stops midway."""
    path = tmp_path / "results.json"
    path.write_text("old result", encoding="utf-8")
    original = io_utils.NamedTemporaryFile

    def failing_file(*args: Any, **kwargs: Any) -> Any:
        """Write a partial result and simulate a disk failure."""
        output = original(*args, **kwargs)
        original_write = output.write

        def fail_write(content: str) -> int:
            """Fail after part of the new JSON has reached the file."""
            original_write(content[:3])
            raise OSError("simulated disk failure")

        monkeypatch.setattr(output, "write", fail_write)
        return output

    monkeypatch.setattr(io_utils, "NamedTemporaryFile", failing_file)
    with pytest.raises(OSError, match="simulated disk failure"):
        io_utils.write_json(path, '{"new": "result"}')
    assert path.read_text(encoding="utf-8") == "old result"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_preserves_previous_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep old results and remove the temporary file if replacement fails."""
    path = tmp_path / "results.json"
    path.write_text("old result", encoding="utf-8")

    def fail_replace(source: Path, target: str | Path) -> Path:
        """Simulate a destination that cannot be replaced."""
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        io_utils.write_json(path, '{"new": "result"}')
    assert path.read_text(encoding="utf-8") == "old result"
    assert list(tmp_path.iterdir()) == [path]
