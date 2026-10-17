*This project has been created as part of the 42 curriculum by ypopovyc.*

# RAG against the machine

## Description

A local Retrieval-Augmented Generation system for the vLLM 0.10.1 codebase and
documentation: split files into chunks, retrieve relevant passages with BM25,
and generate answers with source references using `Qwen/Qwen3-0.6B`.

## Instructions

Use Python **3.12** (the project's pinned version) and
[uv](https://docs.astral.sh/uv/). Run commands from the repository root.
Place the provided vLLM corpus in `data/raw/vllm-0.10.1/` and the question
datasets in `data/datasets/{UnansweredQuestions,AnsweredQuestions}/`.

```bash
make install
uv run python -m src --help

# Build the index, then search or answer one question.
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How does vLLM support LoRA?" --k 5
uv run python -m src answer "How does vLLM support LoRA?" --k 5
```

Generation uses CPU and float32 weights. The first answer command downloads the
model from Hugging Face; allow internet access and several GB of free disk space.

For a complete dataset workflow:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Use `dataset_code_public.json` for code questions. Outputs keep the input
filename and overwrite existing files; separate directories by dataset scope.

| Command | Behavior and optional paths |
| --- | --- |
| `index` | Saves `chunks.jsonl` and `bm25/` under `data/processed/`; accepts `--input_directory` and `--index_directory`. |
| `search <query>` | Prints ranked `file_path[start:end]` locations; accepts `--index_dir`. |
| `search_dataset` | Saves search results as JSON; accepts `--index_directory`. |
| `answer <query>` | Retrieves context and prints the generated answer; accepts `--index_dir`. |
| `answer_dataset` | Reads saved search results and original files, then saves answers as JSON. |
| `evaluate` | Prints recall@k; uses the saved `k` by default, or a smaller cutoff via `--k`. |

The default `k` is 10; allowed values are integers from 1 to 100, up to the chunk
count. Chunk sizes range from 10 to 2000 characters. Custom corpora must be inside
the repository. Keep sources unchanged between search and batch generation.

## Architecture and design

```text
UTF-8 .py / .md / .txt → chunking → chunks.jsonl + BM25 index
Question(s) → retrieval → source excerpts → Qwen3 → answer(s)
                       ↘ reference sources → recall@k
```

The CLI is in `src/__main__.py`; pipeline stages are in `src/indexing/`,
`src/search/`, `src/answer/` and `src/evaluate/`. Shared Pydantic models live in
`src/dataset_classes.py`; tests and edge-case fixtures are in `tests/`.

- **Chunking:** Python uses AST boundaries for imports, functions and classes,
  including decorators. Markdown prefers headings, paragraphs and triple-backtick
  fences; text prefers lines. Oversized blocks fall back to character slicing.
  Chunks do not overlap and store exact relative paths and `[start:end)` spans.
- **Retrieval:** `bm25s` ranks chunks by term frequency, inverse document
  frequency and length normalization (Lucene BM25, `k1=1.5`, `b=0.75`). Original
  identifiers are supplemented with snake_case and CamelCase components.
  Tokens are lowercased; indexing removes six question words, queries use English
  stopwords. String tokens keep vocabulary matching consistent.
- **Generation:** the prompt includes labelled sources and requests citations,
  concise answers and acknowledgement of missing evidence. Thinking is disabled.
  Batch search loads the index once; batch generation loads the model once.
  The full chat template must fit the model context with 250/256 tokens reserved
  for generation; excess sources are omitted and overlong questions rejected.
  Pydantic validates data between stages.

Index rebuilding stages both artifacts before publication and restores the old
pair if publication fails. Unrelated files in the index directory are preserved.

## Data formats

Datasets use `{"rag_questions": [...]}`; questions contain `question_id` and
`question`, with `sources` and `answer` for references. IDs and order are preserved.
Batch outputs use `{"search_results": [...], "k": 5}`: each result contains
`question_id`, `question` and `retrieved_sources`; answer results add `answer`.
Sources contain `file_path`, `first_character_index` and `last_character_index`.
See [the models](src/dataset_classes.py) for the complete schemas.

## Performance and limitations

Evaluation averages per-question recall. A match requires an exact source path
and character-range IoU ≥ 0.05. `evaluate` requires matching question order and
IDs. It defaults to the saved `k`; pass `--k 5` to evaluate recall@5 from a deeper
ranking. Requested cutoffs cannot exceed the saved `k`. It does not call the
moulinette.

Local run on **2026-10-17**, macOS x86_64, Python 3.12.14, fresh index with
23,683 chunks of at most 2000 characters, `k=5`:

| Metric | Measured | Required |
| --- | --- | --- |
| Docs recall@5, 100 public questions | 80.0% | ≥ 80% |
| Code recall@5, 99 public questions | 68.7% | ≥ 50% |
| Full corpus indexing | 17.8 s | ≤ 300 s |
| Batch retrieval, 200 queries | 5.2 s | ≤ 90 s |

Timings include Python CLI startup; the 200-query batch used all 199 public
questions plus one repeated question. These are local results, not official
grading scores. Other chunk sizes and generation performance were not benchmarked.

Challenges include preserving Unicode offsets, splitting oversized code and
matching questions to identifiers. Character-slice tests, format-specific
chunking and identifier expansion address these cases. Remaining limitations:
heuristic Markdown fences, lexical retrieval without semantic matching and
incomplete handling of model errors. Index publication uses rollback and does
not provide a simultaneous snapshot for concurrent searches during replacement.

## Development

```bash
make test
make test TEST_PATH=tests/test_chunker.py TEST_ARGS="-q"
make lint
make run ARGS="search 'LoRA' --k 5"
make debug ARGS="search 'LoRA' --k 5"
make clean
```

Tests cover chunking, corrupted indexes, rollback, JSON validation and writes,
prompt budgets, recall@k and CLI errors; model inference is checked manually.
`make clean`
preserves `.git`, `.venv` and `data`. Keep large data, weights and outputs out of
new commits.

Verification on 2026-10-17: **161 tests passed** (`make test TEST_ARGS="-q"`)
and `make lint` passed. Single and batch generation were also checked on small
examples using cached Qwen3 weights, including real-tokenizer budget checks.

## Resources

- [Project subject](RAG.md): requirements, data models and evaluation rules.
- [Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401): background on RAG.
- [Introduction to Information Retrieval: BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html): ranking fundamentals.
- [BM25S](https://github.com/xhluca/bm25s), [Python AST](https://docs.python.org/3.12/library/ast.html), [Pydantic](https://docs.pydantic.dev/latest/) and [Python Fire](https://github.com/google/python-fire): implementation references.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): generation model and usage.

AI was used for learning, code review, concept explanations and documentation,
plus Makefile improvements and fixes to chunking, validation, JSON writes,
index publication, recall@k and prompt budgets with regression tests.
