from .errors import IndexingException
from pydantic import BaseModel
from pathlib import Path
import regex as re
import ast
from typing import Literal


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class DataChunk(BaseModel):
    index: int
    content: str
    min_source: MinimalSource


class IndexChunks():
    def __init__(self, max_chunk_size: int) -> None:
        self.chunk_list: list[DataChunk] = []
        self.max_chunk_size: int = max_chunk_size

        self.absolute_path: Path = Path(__file__).resolve().parent.parent
        self.data_path: Path = Path(__file__).resolve().parent.parent / \
            "data" / "raw" / "vllm-0.10.1"
        self.relative_path: str

        self.start_index: int = 0
        self.end_index: int = 0
        self.general_index: int = 0

        self.remaining_chunk: str = ""

        self.code_chunk: bool = False

    def index(self) -> None:
        if self.max_chunk_size not in range(1, 2001):
            raise IndexingException(
                "max_chunk_size has to be in range 1-2000")
        if not self.data_path.is_dir():
            raise IndexingException(
                f"Data directory does not exist: {self.data_path}")
        for file in self.data_path.rglob("*.md"):
            with open(file, mode="r", encoding="utf-8") as fd:
                while chunk := fd.read(self.max_chunk_size -
                                       len(self.remaining_chunk)):
                    self._get_md_chunk(self.remaining_chunk+chunk, file)
                    self.general_index += 1

                if self.remaining_chunk:
                    self._get_md_chunk(self.remaining_chunk, file, True)
                    self.general_index += 1

                self._reset()
        for file in self.data_path.glob("chunking_sample.py"):
            with open(file, mode="r", encoding="utf-8") as fd:
                while chunk := fd.read():
                    self._get_code_chunk(chunk, file)
                    self.general_index += 1
                if self.remaining_chunk:
                    pass
                self._reset()

        for chunk in self.chunk_list:
            pass
            # print(chunk.index, chunk.min_source.first_character_index,
            #     chunk.min_source.last_character_index)

    def _reset(self) -> None:
        self.end_index = 0
        self.start_index = self.end_index
        self.code_chunk = False
        self.remaining_chunk = ""

    def _get_md_chunk(self, chunk: str, file_path: Path,
                      last_chunk: bool = False) -> None:
        self.relative_path = str(file_path.relative_to(self.absolute_path))
        self.start_index = self.end_index

        last_title_index: int = -1
        last_paragraph_index: int = -1
        last_code_index: int = -1

        line_list = chunk.splitlines(True)
        last_code_state: bool = self.code_chunk
        for index, line in enumerate(line_list):
            if re.match(r"^#{1,6}(?!#)[ \t]+", line) and not self.code_chunk:
                last_title_index = index
            if line == "\n" and not self.code_chunk:
                last_paragraph_index = index
            if line.startswith("```"):
                self.code_chunk = not self.code_chunk
                last_code_index = index

        if self.code_chunk:
            last_logical_index = max(last_title_index, last_paragraph_index)
        else:
            last_logical_index = max(
                last_paragraph_index, last_title_index, last_code_index)

        if last_logical_index < last_code_index and last_logical_index != -1:
            self.code_chunk = False

        if last_logical_index == -1 or last_chunk:
            self.end_index += len(chunk)
            data_chunk = DataChunk(index=self.general_index, content=chunk,
                                   min_source=MinimalSource(
                                       file_path=self.relative_path,
                                       first_character_index=self.start_index,
                                       last_character_index=self.end_index))
            self.chunk_list.append(data_chunk)
            self.remaining_chunk = ""
            return

        else:
            logical_chunk = "".join(line_list[:max(last_logical_index, 1)])
            self.remaining_chunk = "".join(
                line_list[max(last_logical_index, 1):])
            chunk_len = len(logical_chunk)
            self.end_index += chunk_len
            self._check_code_chunk(line_list, last_code_state, max(
                last_logical_index, 1))

        data_chunk = DataChunk(index=self.general_index, content=logical_chunk,
                               min_source=MinimalSource(
                                   file_path=self.relative_path,
                                   first_character_index=self.start_index,
                                   last_character_index=self.end_index)
                               )

        self.chunk_list.append(data_chunk)

    def _check_code_chunk(self,
                          line_list: list[str],
                          start_state: bool, end_index: int = -1) -> None:
        if end_index == -1:
            for line in line_list[:]:
                if line.startswith("```"):
                    start_state = not start_state
        else:
            for line in line_list[:end_index]:
                if line.startswith("```"):
                    start_state = not start_state
        self.code_chunk = start_state

    def _get_code_chunk(self, chunk: str, file_path: Path) -> None:
        self.relative_path: str = str(
            file_path.relative_to(self.absolute_path))
        tree = ast.parse(chunk)
        line_list = chunk.splitlines(keepends=True)

        last_state: type | None = None
        cur_state: type | None = None
        new_logical_node: bool = False
        import_types: bool = False

        last_line_index = 0
        first_line_index = last_line_index

        cur_chunk_size: int = 0

        for node in tree.body:

            last_state = cur_state
            cur_state = type(node)

            if (last_state == ast.ImportFrom and cur_state == ast.Import or
                    cur_state == ast.ImportFrom and last_state == ast.Import):
                import_types = True
            else:
                import_types = False

            last_line_index = node.lineno - 1

            if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                 ast.AsyncFunctionDef)):
                new_logical_node = True
                if node.decorator_list:
                    last_line_index = min(min(
                        decor_index.lineno
                        for decor_index in node.decorator_list),
                        node.lineno) - 1

            cur_chunk_size = len("".join(
                line_list[first_line_index:last_line_index]))
            print(cur_chunk_size)
            if cur_chunk_size >= self.max_chunk_size:
                print(cur_state)
                print(last_state)
                self._split_code_chunk(
                    first_line_index, last_line_index,
                    line_list, last_state)
                first_line_index = last_line_index
                new_logical_node = False

            elif ((last_state != cur_state or new_logical_node)
                    and import_types is False
                    and last_line_index != first_line_index):
                self._get_code_data_chunk(
                    line_list, first_line_index,
                    last_line_index)

                first_line_index = last_line_index
                new_logical_node = False

        if self.end_index != len(chunk):
            self._get_code_data_chunk(
                line_list, first_line_index,
                len(line_list))

    def _get_code_data_chunk(self, line_list: list[str],
                             first_line_index: int,
                             last_line_index: int) -> None:
        content = "".join(
            line_list[first_line_index:last_line_index])
        self.end_index += len(content)

        data_chunk = DataChunk(index=self.general_index,
                               content=content,
                               min_source=MinimalSource(
                                   file_path=self.relative_path,
                                   first_character_index=self.start_index,
                                   last_character_index=self.end_index))
        self.chunk_list.append(data_chunk)

        print(
            f"Data chunk indexes: {data_chunk.min_source.first_character_index}, {data_chunk.min_source.last_character_index}")
        print(
            f"Data chunk content:\n{content}", end="")
        self.general_index += 1
        self.start_index = self.end_index

    def _split_code_chunk(self, start_line_index: int, end_line_index: int,
                          line_list: list[str], last_state: type | None) -> None:
        print("splitingggg")
        last_logical_index: int = -1

        if isinstance(last_state, ast.ClassDef):
            for index, line in enumerate(line_list[start_line_index:end_line_index]):
                if line.startswith(("def")):
                    last_logical_index = index + start_line_index
                    break

        elif (isinstance(last_state, (ast.FunctionDef, ast.AsyncFunctionDef))
              or last_logical_index == -1):
            print("aaaa")
            for index, line in enumerate(line_list[start_line_index:end_line_index]):
                clean_line = line.lstrip(" \t").rstrip("\n")
                if (clean_line.startswith(("for", "while", "try", "if"))
                        and clean_line.endswith(":")):
                    last_logical_index = index + start_line_index
                    break

        self._get_code_data_chunk(
            line_list, start_line_index, last_logical_index)
        if len("".join(line_list[last_logical_index:end_line_index])) > self.max_chunk_size:
            pass
            # self._split_code_chunk(
            #   last_logical_index, end_line_index, line_list, last_state)

        self._get_code_data_chunk(
            line_list, last_logical_index, end_line_index)
