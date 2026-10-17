"""Measure retrieval recall using exact paths and character-span overlap."""

from pathlib import Path
from ..errors import SearchException
from ..dataset_classes import (
    StudentSearchResults, RagDataset,
    AnsweredQuestion, MinimalSource)


def evaluate(student_search_results_path: str | None = None,
             dataset_path: str | None = None, k: int | None = None) -> None:
    """Report recall at the requested k, defaulting to the saved result k.

    Args:
        student_search_results_path: Saved retrieval JSON. None selects the
            public docs results under data/output/search_results.
        dataset_path: Answered RagDataset JSON. None selects the public docs
            references under data/datasets/AnsweredQuestions.
        k: Positive cutoff no greater than the saved k, or None to use it.

    Print the mean per-question recall. Question IDs and order must match.
    A source match requires the exact path and character-span IoU >= 0.05.
    """
    results_path, data_path = _validate_input(
        student_search_results_path, dataset_path)
    search_dataset = _get_search_dataset(results_path)
    rag_dataset = _get_answered(data_path)
    selected_k = search_dataset.k if k is None else k
    recall = _recall_at_k(search_dataset, rag_dataset, selected_k)
    print(f"Recall_{selected_k} = {recall}")


def _get_search_dataset(result_path: Path) -> StudentSearchResults:
    """Load retrieval results and reject an empty question list."""
    search_dataset = StudentSearchResults.model_validate_json(
        result_path.read_text(encoding="utf-8"))
    if len(search_dataset.search_results) == 0:
        raise SearchException(
            "student_search_results_are empty. check your path")
    return search_dataset


def _get_answered(answers_path: Path) -> RagDataset:
    """Load the reference dataset and reject an empty question list."""
    rag_dataset = RagDataset.model_validate_json(
        answers_path.read_text(encoding="utf-8"))
    if len(rag_dataset.rag_questions) == 0:
        raise SearchException("Answered Questions must not be empty")
    return rag_dataset


def _recall_at_k(search_dataset: StudentSearchResults,
                 rag_dataset: RagDataset, k: int) -> float:
    """Average per-question source recall over the first k retrieved spans."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise SearchException("--k has to be a positive integer")
    if k > search_dataset.k:
        raise SearchException(
            "--k cannot exceed k in the saved search results")
    if not rag_dataset.rag_questions:
        raise SearchException("Answered Questions must not be empty")
    if len(search_dataset.search_results) != len(rag_dataset.rag_questions):
        raise SearchException(
            "Amount of questions has to be equal "
            "in student_search_results and dataset")
    total_recall: float = 0

    for answer, search_results in zip(rag_dataset.rag_questions,
                                      search_dataset.search_results):
        if not isinstance(answer, AnsweredQuestion):
            raise SearchException("Wrong data format in Answered Question")
        if len(answer.sources) == 0:
            raise SearchException("Data must have at least one source")
        if answer.question_id != search_results.question_id:
            raise SearchException(
                "Questio Ids are not matching check your data")

        matches = 0
        for ideal_source in answer.sources:

            if _get_iou(ideal_source, search_results.retrieved_sources[:k]):
                matches += 1
        total_recall += matches / len(answer.sources)

    return total_recall / len(rag_dataset.rag_questions)


def _get_iou(ideal_source: MinimalSource,
             retrieved_sources: list[MinimalSource]) -> bool:
    """Check whether any supplied span matches the source at IoU >= 0.05."""
    for retrieved_source in retrieved_sources:
        if retrieved_source.file_path != ideal_source.file_path:
            continue
        else:
            ideal_start = ideal_source.first_character_index
            ideal_end = ideal_source.last_character_index
            sys_start = retrieved_source.first_character_index
            sys_end = retrieved_source.last_character_index
            if (not (0 <= ideal_start < ideal_end)
                    or not (0 <= sys_start < sys_end)):
                raise SearchException(
                    "Incorrect start/end boundaries check your data")
            intersection = max(0, min(ideal_end, sys_end)
                               - max(ideal_start, sys_start))
            union = ((ideal_end - ideal_start) +
                     (sys_end - sys_start) - intersection)
            iou = intersection / union
            if iou >= 0.05:

                return True
    return False


def _validate_input(student_search_results_path: str | None,
                    dataset_path: str | None) -> tuple[Path, Path]:
    """Resolve result and reference paths and require both files to exist."""
    try:
        if student_search_results_path is None:
            results_path = Path(__file__).resolve(
            ).parent.parent.parent / "data" / "output" / "search_results" \
                / "UnansweredQuestions" / "dataset_docs_public.json"
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
            data_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data" / "datasets" / "AnsweredQuestions"
                / "dataset_docs_public.json"
            )
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
