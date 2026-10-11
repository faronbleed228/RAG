from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class DataChunk(BaseModel):
    index: int
    content: str
    min_source: MinimalSource


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int
