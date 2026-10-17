"""Write serialized pipeline JSON without truncating existing results."""

from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path: Path, content: str) -> None:
    """Write UTF-8 JSON atomically through a sibling temporary file."""
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp",
                                delete=False) as output:
            temporary_path = Path(output.name)
            output.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
