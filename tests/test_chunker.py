import pytest
from src.indexing.index import index_data
from src.indexing.chunker import BaseChunk, DataChunk
from pathlib import Path


def test_chunker() -> None:
    chunker = BaseChunk(2000)
    chunker.data_path = Path(
        __file__).resolve().parent.parent / \
        "tests" / "fixtures"
    chunk_list = chunker.chunk_loop()

    last_chunk_path = ""
    cur_chunk_path = ""
    collected_chunks: list[list[DataChunk]] = []
    cur_file_chunks: list[DataChunk] = []
    for chunk in chunk_list:
        last_chunk_path = cur_chunk_path
        cur_chunk_path = chunk.min_source.file_path
        if last_chunk_path == cur_chunk_path or last_chunk_path == "":
            cur_file_chunks.append(chunk)
        else:
            collected_chunks.append(cur_file_chunks)
            cur_file_chunks.clear()
            cur_file_chunks.append(chunk)
        start = chunk.min_source.first_character_index
        end = chunk.min_source.last_character_index

        with open(cur_chunk_path, "r", encoding="utf-8") as fd:
            source = fd.read()
            assert source[start:end] == chunk.content

    for total_chunks in collected_chunks:
        print((total_chunks))
        content = [chunk.content for chunk in total_chunks]
        path = total_chunks[0].min_source.file_path
        merged_content = "".join(content)
        with open(path, "r", encoding="utf-8") as fd:
            source = fd.read()
            assert source == merged_content
