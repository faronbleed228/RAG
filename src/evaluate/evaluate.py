from pathlib import Path
from ..errors import SearchException


def evaluate(student_search_results_path: str | None = None,
             dataset_path: str | None = None) -> None:
    results_path, data_path = _validate_input(
        student_search_results_path, dataset_path)


def _recall() -> None:
    pass


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
