from ..errors import ChunkingException
from pydantic import BaseModel
from pathlib import Path
import regex as re
import ast
from tqdm import tqdm


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class DataChunk(BaseModel):
    index: int
    content: str
    min_source: MinimalSource


class BaseChunk():
    def __init__(self, max_chunk_size: int) -> None:
        self.chunk_list: list[DataChunk] = []
        self.max_chunk_size: int = max_chunk_size

        self.absolute_path: Path = Path(
            __file__).resolve().parent.parent.parent
        self.data_path: Path = Path(
            __file__).resolve().parent.parent.parent / \
            "data" / "raw" / "vllm-0.10.1"
        self.test_path: Path = Path(__file__).resolve(
        ).parent.parent / "tests" / "fixtures"
        self.relative_path: str

        self.start_index: int = 0
        self.end_index: int = 0
        self.general_index: int = 0

        self.remaining_chunk: str = ""

        self.code_chunk: bool = False

    def update_path(self, input_directory: str | None):
        if input_directory is None:
            return
        try:
            if input_directory.strip() == "":
                raise ChunkingException("Input path has to be not empty")
            self.data_path = Path(input_directory).resolve()
            self.data_path.relative_to(self.absolute_path)
            if not self.data_path.exists():
                raise ChunkingException(
                    f"Input directory doesnt_exists: {input_directory}")
        except (TypeError, AttributeError) as e:
            raise ChunkingException(
                "Wrong value type "
                f"for raw_data: {type(input_directory)}") from e
        except ValueError as e:
            raise ChunkingException(
                "Raw data cant be located outside of program folder"
            ) from e

    def chunk_loop(self) -> list[DataChunk]:
        if self.max_chunk_size not in range(10, 2001):
            raise ChunkingException(
                "max_chunk_size has to be in range 10-2000")
        if not self.data_path.is_dir():
            raise ChunkingException(
                f"Data directory does not exist: {self.data_path}")
        try:
            md_files: list[Path] = [
                file for file in
                self.data_path.rglob("*.md") if file.is_file()]

            for file in tqdm(md_files, desc="Chunking MD Files",
                             total=len(md_files)):
                with open(file, mode="r", encoding="utf-8") as fd:
                    while chunk := fd.read(self.max_chunk_size -
                                           len(self.remaining_chunk)):
                        if chunk != "":
                            self._get_md_chunk(
                                self.remaining_chunk+chunk, file)

                    if self.remaining_chunk:
                        self._get_md_chunk(self.remaining_chunk, file, True)
                    self._reset()

            code_files: list[Path] = [
                file for file in self.data_path.rglob("*.py") if file.is_file()]

            for file in tqdm(code_files, desc="Chunking Python Files",
                             total=len(code_files)):
                with open(file, mode="r", encoding="utf-8") as fd:
                    chunk = fd.read()
                    if chunk != "":
                        self._get_code_chunk(chunk, file)
                        self._reset()

            txt_files: list[Path] = [
                file for file in
                self.data_path.rglob("*.txt") if file.is_file()]

            for file in tqdm(txt_files, desc="Chunking Txt Files",
                             total=len(txt_files)):
                with open(file, "r", encoding="utf-8") as fd:
                    while chunk := (fd.read(self.max_chunk_size
                                            - len(self.remaining_chunk))):
                        if chunk != "":
                            self._get_txt_chunk(
                                self.remaining_chunk + chunk, file)

                    if self.remaining_chunk:
                        self._get_txt_chunk(self.remaining_chunk, file, True)
                    self._reset()

            return self.chunk_list
        except UnicodeDecodeError as e:
            raise ChunkingException(f"Unicode Error for file: {file},"
                                    f" encoding:{e.encoding}, "
                                    f"reason:{e.reason}") from e

    def _reset(self) -> None:
        self.end_index = 0
        self.start_index = self.end_index
        self.code_chunk = False
        self.remaining_chunk = ""

    def _get_txt_chunk(self, chunk: str,
                       file_path: Path, last_chunk: bool = False) -> None:
        self.relative_path = str(file_path.relative_to(self.absolute_path))
        self.start_index = self.end_index

        line_list = chunk.splitlines(True)
        total_lenght = 0
        last_logical_index = -1
        if last_chunk:
            self._create_text_chunk(chunk)
            return
        for index, line in enumerate(line_list):
            total_lenght += len(line)
            if line.endswith("\n"):
                last_logical_index = index
        if last_logical_index != -1:
            logical_chunk = "".join(line_list[:max(last_logical_index, 1)])
            self.remaining_chunk = "".join(
                line_list[max(last_logical_index, 1):])
            self._create_text_chunk(logical_chunk)
        else:
            self._create_text_chunk(chunk)
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
            self._create_text_chunk(chunk)
            self.remaining_chunk = ""
            return

        else:
            logical_chunk = "".join(line_list[:max(last_logical_index, 1)])
            self.remaining_chunk = "".join(
                line_list[max(last_logical_index, 1):])
            self._check_code_chunk(line_list, last_code_state, max(
                last_logical_index, 1))
        self._create_text_chunk(logical_chunk)

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
        self.relative_path = str(
            file_path.relative_to(self.absolute_path))

        try:
            tree = ast.parse(chunk)
        except SyntaxError as e:
            raise ChunkingException(
                f"Syntax error in {self.relative_path}, "
                "line:{e.lineno}, msg: {e.msg}") from e

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

            import_types = (last_state in (ast.ImportFrom, ast.Import)
                            and cur_state in (ast.Import, ast.ImportFrom))

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
            if cur_chunk_size >= self.max_chunk_size:
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
            self._split_code_chunk(first_line_index,
                                   len(line_list), line_list, cur_state)

    def _get_code_data_chunk(self, line_list: list[str],
                             first_line_index: int,
                             last_line_index: int,
                             content: str | None = None) -> None:
        if content is None:
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
        self.general_index += 1
        self.start_index = self.end_index

    def _split_code_chunk(self, start_line_index: int, end_line_index: int,
                          line_list: list[str],
                          last_state: type | None) -> None:
        if (len("".join(line_list[start_line_index:end_line_index]))
                <= self.max_chunk_size):
            self._get_code_data_chunk(
                line_list, start_line_index, end_line_index)
            return
        last_logical_index: int = -1
        if last_state == ast.ClassDef:
            for index, line in enumerate(
                    line_list[start_line_index:end_line_index]):
                clean_line = line.lstrip(" \t").rstrip("\n")
                if clean_line.startswith(("def ")):
                    if (len("".join(
                        line_list[start_line_index:index + start_line_index]))
                            > self.max_chunk_size):
                        break
                    else:
                        last_logical_index = index + start_line_index

        elif last_state in (ast.FunctionDef, ast.AsyncFunctionDef):
            for index, line in enumerate(
                    line_list[start_line_index:end_line_index]):
                clean_line = line.lstrip(" \t").rstrip("\n")
                if (clean_line.startswith(("for", "while", "try", "if"))
                        and clean_line.endswith(":")):
                    if (len
                        ("".join(line_list
                                 [start_line_index:index + start_line_index]))
                            > self.max_chunk_size):
                        break
                    else:
                        last_logical_index = index + start_line_index
        if last_logical_index == -1:
            for index, line in enumerate(
                    line_list[start_line_index:end_line_index]):
                if (len("".join(
                    line_list[start_line_index:index + start_line_index]))
                        > self.max_chunk_size):
                    break
                else:
                    last_logical_index = index + start_line_index

        if last_logical_index > start_line_index:
            self._get_code_data_chunk(
                line_list, start_line_index, last_logical_index)

            if (len("".join(line_list[last_logical_index:end_line_index]))
                    > self.max_chunk_size):
                self._split_code_chunk(
                    last_logical_index, end_line_index, line_list, last_state)
            else:
                self._get_code_data_chunk(
                    line_list, last_logical_index, end_line_index)
        else:
            content = line_list[start_line_index][:self.max_chunk_size]
            self._get_code_data_chunk(
                line_list, start_line_index, last_logical_index, content)
            cur_split = self.max_chunk_size
            while (cur_split + self.max_chunk_size
                   < len(line_list[start_line_index])):
                content = line_list[start_line_index][cur_split:cur_split +
                                                      self.max_chunk_size]
                self._get_code_data_chunk(
                    line_list, start_line_index, last_logical_index, content)
                cur_split += self.max_chunk_size
            if cur_split < len(line_list[start_line_index]):
                content = line_list[start_line_index][cur_split:]
                self._get_code_data_chunk(
                    line_list, start_line_index, last_logical_index, content)
            if (start_line_index + 1 != end_line_index):
                if (len("".join(line_list[start_line_index+1: end_line_index]))
                        > self.max_chunk_size):

                    self._split_code_chunk(
                        start_line_index+1, end_line_index,
                        line_list, last_state)
                else:
                    self._get_code_data_chunk(
                        line_list, start_line_index+1, end_line_index)

    def _create_text_chunk(self,  content: str) -> None:
        self.end_index += len(content)
        data_chunk = DataChunk(index=self.general_index, content=content,
                               min_source=MinimalSource(
                                   file_path=self.relative_path,
                                   first_character_index=self.start_index,
                                   last_character_index=self.end_index)
                               )
        self.chunk_list.append(data_chunk)
        self.general_index += 1
