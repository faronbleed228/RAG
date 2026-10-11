from pathlib import Path
from ..errors import SearchException
from ..dataset_classes import (
    StudentSearchResults, RagDataset,
    AnsweredQuestion, MinimalSource)


def evaluate(student_search_results_path: str | None = None,
             dataset_path: str | None = None) -> None:
    results_path, data_path = _validate_input(
        student_search_results_path, dataset_path)
    search_dataset = _get_search_dataset(results_path)
    rag_dataset = _get_answered(data_path)


def _get_search_dataset(result_path: Path) -> StudentSearchResults:
    search_dataset = StudentSearchResults.model_validate_json(
        result_path.read_text(encoding="utf-8"))
    if len(search_dataset.search_results) == 0:
        raise SearchException(
            "student_search_results_are empty. check your path")
    return search_dataset


def _get_answered(answers_path: Path) -> RagDataset:
    rag_dataset = RagDataset.model_validate_json(
        answers_path.read_text(encoding="utf-8"))
    if len(rag_dataset.rag_questions) == 0:
        raise SearchException("Answered Questions must not be empty")
    return rag_dataset


def _recall(search_dataset: StudentSearchResults,
            rag_dataset: RagDataset) -> None:
    for answer in rag_dataset.rag_questions:
        if not isinstance(answer, AnsweredQuestion):
            raise SearchException("Wrong data format in Answered Question")
        sources = answer.sources


def _get_iou(ideal_source: MinimalSource,
             retrieved_source: MinimalSource) -> float:
    if ideal_source.file_path != retrieved_source.file_path:
        return 0
    ideal_start = ideal_source.first_character_index
    ideal_end = ideal_source.last_character_index
    sys_start = retrieved_source.first_character_index
    sys_end = retrieved_source.last_character_index
    intersection = max(0, min(ideal_end, sys_end)
                       - max(ideal_start, sys_start))
    union = ((ideal_end - ideal_start) + (sys_end - sys_start) - intersection)
    iou = intersection / union
    return iou


def _validate_input(student_search_results_path: str | None,
                    dataset_path: str | None) -> tuple[Path, Path]:
    try:
        if student_search_results_path is None:
            results_path = Path(__file__).resolve(
            ).parent.parent.parent / "data" / "output" / "search_result" \
                / "dataset_docs_public.json"
        else:
            if student_search_results_path.strip() == "":
                raise SearchException(
                    "student_search_results_path path has to be not empty")
            results_path = Path(student_search_results_path).resolve()
        if not results_path.is_file() or not results_path.exists():
            raise SearchException(
                "Couldnt resolve student_search_results_path file path. "
                "Please check your path and try again")
    except (TypeError, AttributeError) as e:
        raise SearchException(
            "Wrong type for student_search_results_path "
            f"path ({type(student_search_results_path)})"
        ) from e

    try:
        if dataset_path is None:
            data_path = Path(__file__).resolve(
            ).parent.parent.parent / "data" / "output" / "search_result" \
                / "dataset_docs_public.json"
        else:
            if dataset_path.strip() == "":
                raise SearchException(
                    "dataset_path path has to be not empty")
            data_path = Path(dataset_path).resolve()
        if not data_path.is_file() or not data_path.exists():
            raise SearchException(
                "Couldnt resolve dataset_path file path. "
                "Please check your path and try again")
    except (TypeError, AttributeError) as e:
        raise SearchException(
            "Wrong type for dataset_path "
            f"path ({type(dataset_path)})"
        ) from e

    return (results_path, data_path)
