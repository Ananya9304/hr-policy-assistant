from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    document: str
    filename: str
    section: str
    section_path: str
    kind: str
    text: str

    def embed_text(self) -> str:
        return f"{self.document} | {self.section_path}\n{self.text}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float
    fused_score: float


class LLMCitation(BaseModel):
    source_id: str = Field(description="Label of the source, e.g. 'S2'")
    quote: str = Field(description="Exact sentence or table row copied from that source")


class LLMAnswer(BaseModel):
    answerable: bool
    answer: str = ""
    citations: list[LLMCitation] = []


class AskRequest(BaseModel):
    question: str = Field(..., max_length=1000)


class Status(str, Enum):
    answered = "answered"
    refused = "refused"
    error = "error"


class RefusalReason(str, Enum):
    empty_query = "empty_query"
    no_documents = "no_documents"
    low_retrieval_score = "low_retrieval_score"
    model_declined = "model_declined"
    unverified_citations = "unverified_citations"
    llm_unavailable = "llm_unavailable"


class Citation(BaseModel):
    document: str
    filename: str
    section: str
    section_path: str
    quote: str


class AskResponse(BaseModel):
    status: Status
    answer: str
    citations: list[Citation] = []
    reason: Optional[RefusalReason] = None
    top_score: Optional[float] = None


class DocumentInfo(BaseModel):
    doc_id: str
    document: str
    filename: str
    version: int
    chunks: int
    uploaded_at: str