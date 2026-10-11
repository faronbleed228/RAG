"""Provide isolated corpora and indexes for pipeline tests."""

from collections.abc import Iterator
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.indexing.index import index_data


@pytest.fixture
def corpus_dir() -> Iterator[Path]:
    """Create a small corpus inside the permitted project directory."""
    project_root = Path(__file__).resolve().parent.parent
    with TemporaryDirectory(
        prefix=".pytest-corpus-", dir=project_root
    ) as name:
        directory = Path(name)
        (directory / "cats.md").write_text(
            "# Cat care\n\nCats communicate by purring softly. 🐈\n",
            encoding="utf-8",
        )
        (directory / "dogs.md").write_text(
            "# Dog care\n\nDogs communicate by barking loudly. 🐕\n",
            encoding="utf-8",
        )
        (directory / "birds.py").write_text(
            '"""Bird helpers."""\n\n\n'
            'def bird_sound() -> str:\n'
            '    """Return the distinctive sound of a bird."""\n'
            '    return "chirping"\n',
            encoding="utf-8",
        )
        yield directory


@pytest.fixture
def indexed_dir(corpus_dir: Path, tmp_path: Path) -> Path:
    """Build a real index outside the project without touching user data."""
    directory = tmp_path / "processed"
    index_data(
        input_directory=str(corpus_dir),
        index_directory=str(directory),
    )
    return directory


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """Create ordered questions with IDs that differ from corpus order."""
    path = tmp_path / "questions sample.json"
    path.write_text(
        json.dumps({"rag_questions": [
            {"question_id": "dog-id", "question": "barking"},
            {"question_id": "cat-id", "question": "Where is purring? 🐈"},
            {"question_id": "bird-id", "question": "chirping"},
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
