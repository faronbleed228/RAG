from .errors import IndexingException
from pydantic import BaseModel
from pathlib import Path
import regex as re


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class DataChunk(BaseModel):
    index: int
    content: str
    min_source: MinimalSource


class IndexChunks():
    def __init__(self) -> None:
        self.chunk_list: list[DataChunk] = []

        self.absolute_path: Path = Path(__file__).resolve().parent.parent
        self.data_path: Path = Path(__file__).resolve().parent.parent / \
            "data" / "raw" / "vllm-0.10.1"

        self.start_index: int = 0
        self.end_index: int = 0
        self.general_index: int = 0

        self.remaining_chunk: str = ""

    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size not in range(1, 2001):
            raise IndexingException(
                "max_chunk_size has to be in range 1-2000")
        if not self.data_path.is_dir():
            raise IndexingException(
                f"Data directory does not exist: {self.data_path}")
        for file in self.data_path.rglob("*.md"):
            with open(file, mode="r", encoding="utf-8") as fd:
                while chunk := fd.read(max_chunk_size -
                                       len(self.remaining_chunk)):
                    self._get_data_chunk(self.remaining_chunk+chunk, file)

                return
                self.general_index += 1

    def _get_data_chunk(self, chunk: str, file_path: Path) -> None:
        relative_path = str(file_path.relative_to(self.absolute_path))
        self.start_index = self.end_index

        last_title_index: int | None = None
        last_paragraph_index: int | None = None

        line_list = chunk.splitlines(True)
        for index, line in enumerate(line_list):
            if re.match(r"^#{1,3}(?!#)[ \t]+", line):
                last_title_index = index
            if line == "/n":
                last_paragraph_index = index

        if last_title_index:
            logical_chunk = "".join(line_list[:last_title_index])
            self.remaining_chunk = "".join(line_list[last_title_index:])
            chunk_len = len(logical_chunk)
            self.end_index += chunk_len
        elif last_paragraph_index:
            logical_chunk = "".join(line_list[:last_paragraph_index])
            self.remaining_chunk = "".join(line_list[last_paragraph_index:])
            chunk_len = len(logical_chunk)
            self.end_index += chunk_len
        else:
            logical_chunk = "".join(line_list)
            self.remaining_chunk = ""

        print(logical_chunk, end="")
        data_chunk = DataChunk(index=self.general_index, content=logical_chunk,
                               min_source=MinimalSource(
                                   file_path=relative_path,
                                   first_character_index=self.start_index,
                                   last_character_index=self.end_index)
                               )
        self.chunk_list.append(data_chunk)
