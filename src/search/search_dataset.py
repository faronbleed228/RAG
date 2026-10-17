"""Run batch retrieval and serialize results with their original questions."""

from .search import search
from ..indexing.chunker import MinimalSource, DataChunk
from ..errors import SearchException
from ..io_utils import write_json
from ..dataset_classes import (RagDataset, MinimalSearchResults,
                               StudentSearchResults, UnansweredQuestion)

from pathlib import Path
from typing import Any


def search_dataset(dataset_path: str | None = None,
                   k: int = 10, save_directory: str | None = None,
                   index_directory: str | None = None) -> None:
    """Search every dataset question and save a StudentSearchResults JSON.

    Args:
        dataset_path: Input RagDataset JSON. None selects the public docs
            dataset under data/datasets/UnansweredQuestions.
        k: Number of sources per question, limited to 1-100 and chunk count.
        save_directory: Output folder. None selects
            data/output/search_results/UnansweredQuestions.
        index_directory: Index folder, or None for data/processed.

    The output keeps the input filename, question IDs and question order.
    """
    questions_path, save_path = _check_input(dataset_path, save_directory, k)
    output_name: Path = (questions_path.relative_to(questions_path.parent))
    rag_dataset = _load_questions(questions_path)
    results = _collect_results(rag_dataset, k, index_directory)
    _dump_results(save_path, results, output_name, k)


def _dump_results(save_dir: Path,
                  results: list[MinimalSearchResults],
                  output_name: Path, k: int) -> None:
    """Write the result envelope, overwriting the same output filename."""
    complete_path = Path(save_dir, output_name)
    student_search = StudentSearchResults(search_results=results, k=k)
    json_str = student_search.model_dump_json(indent=2)
    write_json(complete_path, json_str)
    print(complete_path)


def _collect_results(rag_dataset: RagDataset, k: int,
                     index_dir: str | None) -> list[MinimalSearchResults]:
    """Retrieve all dataset questions together using a single index load."""
    question_list: list[UnansweredQuestion] = []
    for question in rag_dataset.rag_questions:
        if not isinstance(question, UnansweredQuestion):
            raise SearchException(
                "Wrong format of UnansweredQuestion please "
                "check your data and try again")
        question_list.append(question)
    results: list[MinimalSearchResults] = []
    q_list = [q.question for q in question_list]
    if len(q_list) == 0:
        raise SearchException("Empty query dataset")
    chunk_list, documents = search(q_list, k, index_dir)
    results = _create_result(question_list, documents, chunk_list)

    return (results)


def _create_result(question_list: list[UnansweredQuestion],
                   documents: Any,
                   chunk_list: list[DataChunk]) -> list[MinimalSearchResults]:
    """Map ranking rows to question IDs and their ordered source locations."""
    results: list[MinimalSearchResults] = []
    min_source_list: list[MinimalSource] = []
    for question, document in zip(question_list, documents):
        query = question.question
        id = question.question_id
        for index in document:
            chunk = chunk_list[index]
            min_source_list.append(chunk.min_source)
        result = MinimalSearchResults(
            question_id=id, question=query, retrieved_sources=min_source_list)
        results.append(result)
        min_source_list = []
    return results


def _check_input(dataset_path: str | None,
                 save_directory: str | None, k: int) -> tuple[Path, Path]:
    """Validate paths and k, create the output folder and return both paths."""
    try:
        if dataset_path is None:
            questions_path = Path(
                __file__).resolve().parent.parent.parent / "data" \
                / "datasets" / "UnansweredQuestions" \
                / "dataset_docs_public.json"
        else:
            if dataset_path.strip() == "":
                raise SearchException("Dataset path has to be not empty")
            questions_path = Path(dataset_path).resolve()
        if not questions_path.exists() or not questions_path.is_file():
            raise SearchException(
                "Couldnt resolve dataset file path. "
                "Please check your path and try again")
    except (TypeError, AttributeError) as e:
        raise SearchException(
            f"Wrong type for dataset path ({type(dataset_path)})"
        ) from e

    try:
        if save_directory is None:
            search_save_dir = Path(
                __file__).resolve().parent.parent.parent / "data" / "output" \
                / "search_results"/"UnansweredQuestions"
        else:
            if save_directory.strip() == "":
                raise SearchException(
                    "Save directory path has to be not empty")
            search_save_dir = Path(save_directory).resolve()
    except (TypeError, AttributeError) as e:
        raise SearchException(
            f"Wrong type for save directory ({type(save_directory)})"
        ) from e
    if not isinstance(k, int) or isinstance(k, bool):
        raise SearchException("--k parameter has to be an int")
    if k not in range(1, 101):
        raise SearchException(
            "--k parameter has to be in range 1-100 and "
            "at less than amount of chunks")
    search_save_dir.mkdir(parents=True, exist_ok=True)
    return (questions_path, search_save_dir)


def _load_questions(questions_path: Path) -> RagDataset:
    """Read a UTF-8 JSON document and validate its RagDataset envelope."""
    rag_dataset: RagDataset = RagDataset.model_validate_json(
        questions_path.read_text(encoding="utf-8"))

    return rag_dataset
