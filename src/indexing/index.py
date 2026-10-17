"""Build BM25 and JSONL artifacts and publish them with rollback support."""

from .chunker import BaseChunk, DataChunk
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from shutil import rmtree
from ..errors import IndexingException
import bm25s


def index_data(max_chunk_size: int = 2000,
               input_directory: str | None = None,
               index_directory: str | None = None) -> None:
    """Build both artifacts before publishing them, restoring on failure.

    Args:
        max_chunk_size: Maximum chunk length, from 10 to 2000 characters.
        input_directory: Corpus inside the project root. None selects
            data/raw/vllm-0.10.1.
        index_directory: Output folder. None selects data/processed.

    Stage chunks.jsonl and bm25 together before replacing the saved files.
    Publication uses rollback, not a concurrent-reader snapshot.
    """
    index_path = _resolve_index_directory(index_directory)
    chunker = BaseChunk(max_chunk_size)
    chunker.update_path(input_directory)
    chunk_list: list[DataChunk] = chunker.chunk_loop()

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rag-index-",
                            dir=index_path.parent) as directory:
        staged = Path(directory)
        load_chunks(chunk_list, directory)
        create_search_index(chunk_list, directory)
        _publish_index(staged, index_path)
    print(f"Ingestion complete! Indexed {len(chunk_list)} chunks "
          f"under {index_path}")


def _resolve_index_directory(index_directory: str | None) -> Path:
    """Resolve the index destination without treating blank paths as cwd."""
    if index_directory is None:
        return Path(__file__).resolve().parent.parent.parent / \
            "data" / "processed"
    if not isinstance(index_directory, str):
        raise IndexingException("Index directory has to be a string")
    if not index_directory.strip():
        raise IndexingException("Index path has to be not empty")
    return Path(index_directory).resolve()


def _publish_index(staged: Path, destination: Path) -> None:
    """Publish the artifacts with rollback on errors and interruptions."""
    destination.mkdir(parents=True, exist_ok=True)
    names = ("chunks.jsonl", "bm25")
    backup = Path(mkdtemp(prefix=".rag-backup-", dir=destination))
    remove_backup = False
    try:
        try:
            for name in names:
                current = destination / name
                if current.exists() or current.is_symlink():
                    current.replace(backup / name)
                (staged / name).replace(current)
        except BaseException:
            # Also restore the old pair when the user interrupts publication.
            try:
                for name in reversed(names):
                    current = destination / name
                    if not (staged / name).exists() and current.exists():
                        current.replace(staged / name)
                    previous = backup / name
                    if previous.exists() or previous.is_symlink():
                        previous.replace(current)
            except OSError as error:
                raise IndexingException(
                    f"Could not restore the index; backup kept at {backup}"
                ) from error
            remove_backup = True
            raise
        remove_backup = True
    finally:
        if remove_backup:
            rmtree(backup, ignore_errors=True)


def load_chunks(chunk_list: list[DataChunk],
                chunks_directory: str | None) -> None:
    """Write validated chunks to their internal JSONL storage."""
    if len(chunk_list) == 0:
        raise IndexingException("Chunk list is empty....")
    try:
        if chunks_directory is None:
            data_path = Path(__file__).resolve(
            ).parent.parent.parent / "data" / "processed" / "chunks.jsonl"
        else:
            if chunks_directory.strip() == "":
                raise IndexingException("Chunks path has to be not empty")
            data_path = Path(chunks_directory, "chunks.jsonl").resolve()
    except (TypeError, AttributeError) as e:
        raise IndexingException(
            "Wrong value type "
            f"for chunks directory: {type(chunks_directory)}") from e

    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as output:
        for chunk in chunk_list:
            output.write(chunk.model_dump_json() + "\n")
    new_chunk_list: list[DataChunk] = []
    with data_path.open("r", encoding="utf_8") as input:
        for line in input:
            if line.strip():
                chunk = DataChunk.model_validate_json(line)
                new_chunk_list.append(chunk)


def create_search_index(chunk_list: list[DataChunk],
                        index_directory: str | None) -> None:
    """Build a BM25 index, rejecting corpora without searchable tokens."""
    try:
        if index_directory is None:
            data_path = Path(__file__).resolve(
            ).parent.parent.parent / "data" / "processed" / "bm25"
        else:
            if index_directory.strip() == "":
                raise IndexingException("Index path has to be not empty")
            data_path = Path(index_directory, "bm25").resolve()
    except (TypeError, AttributeError) as e:
        raise IndexingException(
            "Wrong value type "
            f"for index directory: {type(index_directory)}") from e
    texts: list[str] = [get_more_token(chunk.content,
                        chunk.min_source.file_path)
                        for chunk in chunk_list]
    if len(texts) == 0:
        raise IndexingException("All data files are empty")
    corpus_tokens = bm25s.tokenize(
        texts,
        stopwords=["what", "which", "who", "does", "do", "did"],
        return_ids=False,
        show_progress=True,
        leave=True,
        lower=True,
    )
    if not any(corpus_tokens):
        raise IndexingException("All indexed files have an empty vocabulary")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, leave_progress=True)
    data_path.mkdir(parents=True, exist_ok=True)
    retriever.save(data_path)


def get_more_token(content: str, path: str | None = None) -> str:
    """Keep original words and append their snake_case and CamelCase parts."""
    words: list[str] = content.split()
    if path is not None:
        path_list = path.split("/")
        file_path = "/".join(path_list[2:])
        words.insert(0, file_path)
    total_words: list[str] = words.copy()
    for word in words:
        if len(word.strip("_").split("_")) > 1:
            total_words.extend(word.split("_"))
        camelcase: list[str] = []
        last_index = 0
        for index in range(len(word)):
            if (index != 0
                    and (word[index].isupper() and word[index-1].islower())):
                camelcase.append(word[last_index:index])
                last_index = index
            if (index != 0 and index != len(word) - 1
                    and (word[index].isupper() and word[index-1].isupper())
                    and word[index+1].islower()):
                camelcase.append(word[last_index:index])
                last_index = index
        if len(camelcase) != 0:
            camelcase.append(word[last_index:])
            total_words.extend(camelcase)
            camelcase = []

    return " ".join(total_words)
