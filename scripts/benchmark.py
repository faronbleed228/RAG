"""Rebuild the bundled corpus index and record source-span recall; no LLM."""
import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing.index import index_data  # noqa: E402
from src.search.search_dataset import search_dataset  # noqa: E402
from src.evaluate.evaluate import (  # noqa: E402
    _get_answered, _get_search_dataset, _recall_at_k, evaluate)


def checksum(paths: list[Path]) -> str:
    """Hash sorted relative paths and their content digests."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/retrieval.json"))
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error("Run from the repository root")
    corpus = ROOT / "data/raw/vllm-0.10.1"
    corpus_files = [p for p in corpus.rglob("*") if p.is_file()]
    datasets = ROOT / "data/datasets"
    report = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": checksum(list((ROOT / "src").rglob("*.py"))),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in (
            "numpy", "bm25s", "pydantic", "regex", "scipy")},
        "corpus": str(corpus.relative_to(ROOT)),
        "corpus_file_count": len(corpus_files),
        "corpus_sha256": checksum(corpus_files),
        "max_chunk_size": 2000, "k": 5, "iou_threshold": 0.05,
        "inference": "not run",
        "results": {},
    }
    with TemporaryDirectory(prefix="rag-benchmark-") as directory:
        index = Path(directory) / "index"
        results = Path(directory) / "results"
        index_data(2000, str(corpus), str(index))
        report["chunks"] = sum(1 for _ in (
            index / "chunks.jsonl").open(encoding="utf-8"))
        for name in ("docs", "code"):
            filename = f"dataset_{name}_public.json"
            query = datasets / "UnansweredQuestions" / filename
            reference = datasets / "AnsweredQuestions" / filename
            search_dataset(str(query), 5, str(results), str(index))
            saved = results / filename
            rankings = _get_search_dataset(saved)
            expected = _get_answered(reference)
            output = io.StringIO()
            with redirect_stdout(output):
                evaluate(str(saved), str(reference), 5)
            report["results"][name] = {
                "questions": len(expected.rag_questions),
                "recall_at_5": _recall_at_k(rankings, expected, 5),
                "evaluate_stdout": output.getvalue().strip(),
                "datasets_sha256": checksum([query, reference]),
                "rankings_sha256": hashlib.sha256(
                    saved.read_bytes()).hexdigest(),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
