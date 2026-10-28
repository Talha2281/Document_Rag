from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

class SourceItem(BaseModel):
    source: str
    page: int | None = None
    excerpt: str
    relevance_score: float | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    steps: list[str]
    retries: int
    abstained: bool
    contexts: list[str] = []

class EvaluationCase(BaseModel):
    question: str = Field(min_length=3)
    expected_keywords: list[str] = []
    expected_source: str | None = None

class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1, max_length=30)
