"""Define the Pydantic data exchanged between RAG pipeline stages."""

from pydantic import BaseModel, Field, field_validator
import uuid


class MinimalSource(BaseModel):
    """Locate an excerpt by project-relative path and character offsets.

    The first index is inclusive and the last index is exclusive, matching
    Python string slicing after reading the source as UTF-8 text.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class DataChunk(BaseModel):
    """Store a chunk's corpus index, original text and source location."""

    index: int
    content: str
    min_source: MinimalSource


class UnansweredQuestion(BaseModel):
    """Store a question and generate a UUID when its ID is omitted."""

    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Extend a question with its reference sources and reference answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Wrap answered or unanswered questions under the rag_questions key."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Pair a question and its ID with source locations in retrieval order."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        """Reject blank questions while preserving the original text."""
        if not question.strip():
            raise ValueError("Question must not be empty or whitespace")
        return question


class StudentSearchResults(BaseModel):
    """Wrap batch retrieval results with the requested ranking depth k."""

    search_results: list[MinimalSearchResults]
    k: int = Field(strict=True, ge=1, le=100)


class MinimalAnswer(MinimalSearchResults):
    """Extend a retrieval result with the generated answer text."""

    answer: str


class StudentSearchResultsAndAnswer(BaseModel):
    """Wrap batch answers and retrieved sources with their ranking depth."""

    search_results: list[MinimalAnswer]
    k: int = Field(strict=True, ge=1, le=100)
