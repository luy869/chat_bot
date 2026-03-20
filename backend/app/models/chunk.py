from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    content: str
    document_id: str
    source_file: str

    heading_path: list[str]
    page_number: int | None

    chunk_index: int
    token_count: int

    metadata: dict[str, Any] | None = None