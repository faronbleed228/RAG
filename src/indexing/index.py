from .chunker import BaseChunk, DataChunk
from pathlib import Path
from ..errors import IndexingException
import bm25s


def index_data(max_chunk_size: int = 2000,
               input_directory: str | None = None,
               index_directory: str | None = None) -> None:

    chunker = BaseChunk(max_chunk_size)
    chunker.update_path(input_directory)
    chunk_list: list[DataChunk] = chunker.chunk_loop()

    load_chunks(chunk_list, index_directory)
    create_search_index(chunk_list,
                        index_directory)


def load_chunks(chunk_list: list[DataChunk],
                chunks_directory: str | None) -> None:
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

    data_path.mkdir(parents=True, exist_ok=True)

    texts: list[str] = [get_more_token(chunk.content) for chunk in chunk_list]
    if len(texts) == 0:
        raise IndexingException("All data files are empty")
    corpus_tokens = bm25s.tokenize(
        texts, stopwords=["what", "which", "who", "does", "do", "did"], return_ids=False, show_progress=True, leave=True, lower=True)
    print(
        f"Ingestion complete! Indexed {len(corpus_tokens)} "
        f"under {data_path}")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(data_path)


def get_more_token(content: str) -> str:
    words: list[str] = content.split()
    total_words: list[str] = content.split()
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
                    and (word[index].isupper() and word[index-1].isupper()) and word[index+1].islower()):
                camelcase.append(word[last_index:index])
                last_index = index
        if len(camelcase) != 0:
            camelcase.append(word[last_index:])
            total_words.extend(camelcase)
            camelcase = []

    return " ".join(total_words)
