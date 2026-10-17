"""Generate batch answers from saved rankings and original source files."""

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from pathlib import Path
import tqdm
from ..errors import SearchException
from ..io_utils import write_json
from ..dataset_classes import (
    StudentSearchResults, StudentSearchResultsAndAnswer,
    MinimalAnswer, MinimalSource, DataChunk, MinimalSearchResults)
from .answer import _build_promt


def answer_dataset(
    student_search_results_path: str | None = None,
    save_directory: str | None = None,
) -> None:
    """Generate answers from saved retrieval results and write a JSON file.

    Args:
        student_search_results_path: StudentSearchResults JSON. None selects
            the public docs results under data/output/search_results.
        save_directory: Output folder, or None for data/output/answers.

    Keep question IDs, retrieved sources, k and the input filename. Source
    paths are read relative to the working directory; run from project root.
    """
    search_path, save_path = _check_input(
        student_search_results_path, save_directory)
    search_results = _load_search_results(search_path)
    answer_results = _get_dataset_answers(search_results)
    output_name = search_path.relative_to(search_path.parent)
    _save_answer_results(answer_results, search_results.k,
                         save_path, output_name)


def _save_answer_results(answer_results: list[MinimalAnswer],
                         k: int, save_path: Path, output_name: Path) -> None:
    """Write the answer envelope, overwriting the same output filename."""
    output_path = Path(save_path, output_name)
    student_results = StudentSearchResultsAndAnswer(
        search_results=answer_results, k=k)
    answer_json = student_results.model_dump_json(indent=2)
    write_json(output_path, answer_json)


def _check_input(student_search_results_path: str | None,
                 save_directory: str | None) -> tuple[Path, Path]:
    """Resolve the saved ranking file and create the output directory."""
    try:
        if student_search_results_path is None:
            questions_path = Path(
                __file__).resolve().parent.parent.parent / "data" \
                / "output" / "search_results" / "UnansweredQuestions" \
                / "dataset_docs_public.json"
        else:
            if student_search_results_path.strip() == "":
                raise SearchException(
                    "student_search_results_path path has to be not empty")
            questions_path = Path(student_search_results_path).resolve()
        if not questions_path.exists() or not questions_path.is_file():
            raise SearchException(
                "Couldnt resolve student_search_results_path file path. "
                "Please check your path and try again")
    except (TypeError, AttributeError) as e:
        raise SearchException(
            "Wrong type for student_search_results_path path "
            f"({type(student_search_results_path)})"
        ) from e

    try:
        if save_directory is None:
            search_save_dir = Path(
                __file__).resolve().parent.parent.parent / "data" / "output" \
                / "answers"
        else:
            if save_directory.strip() == "":
                raise SearchException(
                    "Save directory path has to be not empty")
            search_save_dir = Path(save_directory).resolve()
    except (TypeError, AttributeError) as e:
        raise SearchException(
            f"Wrong type for save directory ({type(save_directory)})"
        ) from e

    search_save_dir.mkdir(parents=True, exist_ok=True)
    return (questions_path, search_save_dir)


def _load_search_results(search_path: Path) -> StudentSearchResults:
    """Validate the retrieval envelope, requiring results and positive k."""
    search_results = StudentSearchResults.model_validate_json(
        search_path.read_text(encoding="utf-8"))
    if len(search_results.search_results) == 0:
        raise SearchException("There must be at least one search result.")
    if any(not result.retrieved_sources
           for result in search_results.search_results):
        raise SearchException("Each question must have a retrieved source")
    if search_results.k <= 0:
        raise SearchException("Wrong k parameter in search results.")

    return search_results


def _get_dataset_answers(
    search_results: StudentSearchResults,
) -> list[MinimalAnswer]:
    """Load Qwen3 once and generate answers in the saved question order.

    Reconstruct each question's excerpts from the original files and reserve
    250 new tokens within the model's configured context window.
    """
    model_name = "Qwen/Qwen3-0.6B"
    answer_results: list[MinimalAnswer] = []

    tokeniser: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name)
    config = AutoConfig.from_pretrained(model_name)
    max_new_tokens = 250
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name, config=config,
        torch_dtype=torch.float32, use_safetensors=True)
    model.eval()

    for search in tqdm.tqdm(search_results.search_results,
                            desc="Processing Answers",
                            total=len(search_results.search_results)):
        query = search.question
        sources = search.retrieved_sources
        chunk_list = _build_data_from_source(sources)
        inputs = _build_promt(
            query, chunk_list, tokeniser, config.max_position_embeddings,
            max_new_tokens)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(**inputs,
                                        max_new_tokens=max_new_tokens)

        answer_ids = output_ids[0][input_len:]
        output = tokeniser.decode(answer_ids, skip_special_tokens=True)
        min_answer = _get_minimal_answer(output, search)
        answer_results.append(min_answer)

    return answer_results


def _get_minimal_answer(answer: str,
                        minimal_search: MinimalSearchResults) -> MinimalAnswer:
    """Attach answer text, preserving the question and retrieved sources."""
    question = minimal_search.question
    id = minimal_search.question_id
    sources = minimal_search.retrieved_sources

    return MinimalAnswer(
        question_id=id, question=question,
        retrieved_sources=sources, answer=answer)


def _build_data_from_source(sources: list[MinimalSource]) -> list[DataChunk]:
    """Read UTF-8 source slices, requiring valid exclusive character ranges."""
    data_list: list[DataChunk] = []
    index = 0

    for source in sources:
        start = source.first_character_index
        end = source.last_character_index
        path = Path(source.file_path)
        text = path.read_text(encoding="utf-8")
        if not (0 <= start < end <= len(text)):
            raise SearchException(
                "Wrong data in chunk(start and end chars). "
                "Check your data and try again")
        content = text[start:end]
        data_chunk = DataChunk(index=index, content=content, min_source=source)
        data_list.append(data_chunk)
        index += 1

    return data_list
