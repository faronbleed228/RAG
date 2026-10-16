import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from pathlib import Path
import tqdm
from ..errors import SearchException
from ..dataset_classes import (
    StudentSearchResults, StudentSearchResultsAndAnswer,
    MinimalAnswer, MinimalSource, DataChunk, MinimalSearchResults)
from .answer import _build_promt


def answer_dataset(student_search_results_path: str | None = None,  save_directory: str | None = None) -> None:
    search_path, save_path = _check_input(
        student_search_results_path, save_directory)
    search_results = _load_search_results(search_path)
    answer_results = _get_dataset_answers(search_results)
    output_name = search_path.relative_to(search_path.parent)
    _save_answer_results(answer_results, search_results.k,
                         save_path, output_name)


def _save_answer_results(answer_results: list[MinimalAnswer],
                         k: int, save_path: Path, output_name: Path) -> None:
    output_path = Path(save_path, output_name)
    student_results = StudentSearchResultsAndAnswer(
        search_results=answer_results, k=k)
    answer_json = student_results.model_dump_json(indent=2)
    output_path.write_text(answer_json)


def _check_input(student_search_results_path: str | None,
                 save_directory: str | None) -> tuple[Path, Path]:

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
            f"Wrong type for student_search_results_path path ({type(student_search_results_path)})"
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
    search_results = StudentSearchResults.model_validate_json(
        search_path.read_text(encoding="utf-8"))
    if len(search_results.search_results) == 0:
        raise SearchException("There must be at least one search result.")
    if search_results.k <= 0:
        raise SearchException("Wrong k parameter in search results.")

    return search_results


def _get_dataset_answers(search_results: StudentSearchResults) -> list[MinimalAnswer]:
    model_name = "Qwen/Qwen3-0.6B"
    answer_results: list[MinimalAnswer] = []

    tokeniser: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, use_safetensors=True)
    model.eval()

    for search in tqdm.tqdm(search_results.search_results,
                            desc="Processing Answers",
                            total=len(search_results.search_results)):
        query = search.question
        sources = search.retrieved_sources
        chunk_list = _build_data_from_source(sources)
        master_prompt = _build_promt(query, chunk_list)
        messages = [{
            "role": "user",
            "content": master_prompt
        }]
        text = str(tokeniser.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False))
        inputs = tokeniser(text, return_tensors="pt", add_special_tokens=False)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=250)

        output_ids = output_ids[0][input_len:]
        output = tokeniser.decode(output_ids, skip_special_tokens=True)
        min_answer = _get_minimal_answer(output, search)
        answer_results.append(min_answer)

    return answer_results


def _get_minimal_answer(answer: str,
                        minimal_search: MinimalSearchResults) -> MinimalAnswer:
    question = minimal_search.question
    id = minimal_search.question_id
    sources = minimal_search.retrieved_sources

    return MinimalAnswer(
        question_id=id, question=question,
        retrieved_sources=sources, answer=answer)


def _build_data_from_source(sources: list[MinimalSource]) -> list[DataChunk]:
    data_list: list[DataChunk] = []
    index = 0

    for source in sources:
        start = source.first_character_index
        end = source.last_character_index
        path = Path(source.file_path)
        text = path.read_text(encoding="utf-8")
        content = text[start:end]
        data_chunk = DataChunk(index=index, content=content, min_source=source)
        data_list.append(data_chunk)
        index += 1

    return data_list
