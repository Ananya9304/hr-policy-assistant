
import os
import sqlite3
import threading
from datetime import datetime, timezone

import numpy as np

from app.models import Chunk, DocumentInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    document    TEXT NOT NULL,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section      TEXT NOT NULL,
    section_path TEXT NOT NULL,
    kind         TEXT NOT NULL,
    text         TEXT NOT NULL,
    embedding    BLOB NOT NULL
);
"""


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self._cache = None  # (chunks, matrix); rebuilt after every write

    # ------------------------------------------------------------ writes
    def upsert_document(self, document: str, filename: str, sha256: str,
                        chunks: list[Chunk], vectors: np.ndarray) -> tuple[DocumentInfo, bool]:
        """Insert or replace a document. Returns (info, changed)."""
        doc_id = chunks[0].doc_id
        with self.lock, self.db:
            row = self.db.execute("SELECT sha256, version FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            if row and row[0] == sha256:
                return self.get_document(doc_id), False        # same file uploaded again: no-op
            version = row[1] + 1 if row else 1
            # Replace in one transaction: readers never see a half-indexed policy
            self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self.db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                (doc_id, document, filename, sha256, version, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            self.db.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                [(c.chunk_id, doc_id, c.section, c.section_path, c.kind, c.text, v.astype(np.float32).tobytes())
                 for c, v in zip(chunks, vectors)],
            )
            self._cache = None
        return self.get_document(doc_id), True

    def delete_document(self, doc_id: str) -> bool:
        with self.lock, self.db:
            cur = self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self._cache = None
        return cur.rowcount > 0

    # ------------------------------------------------------------ reads
    def get_document(self, doc_id: str) -> DocumentInfo | None:
        row = self.db.execute(
            """SELECT d.doc_id, d.document, d.filename, d.version, COUNT(c.chunk_id), d.uploaded_at
               FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
               WHERE d.doc_id=? GROUP BY d.doc_id""", (doc_id,)).fetchone()
        return self._info(row) if row else None

    def list_documents(self) -> list[DocumentInfo]:
        rows = self.db.execute(
            """SELECT d.doc_id, d.document, d.filename, d.version, COUNT(c.chunk_id), d.uploaded_at
               FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
               GROUP BY d.doc_id ORDER BY d.document""").fetchall()
        return [self._info(r) for r in rows]

    def all_chunks(self) -> tuple[list[Chunk], np.ndarray]:
        """All chunks plus an (n, dim) matrix of their vectors, cached in memory."""
        with self.lock:
            if self._cache is None:
                rows = self.db.execute(
                    """SELECT c.chunk_id, c.doc_id, d.document, d.filename, c.section, c.section_path,
                              c.kind, c.text, c.embedding
                       FROM chunks c JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.chunk_id""").fetchall()
                chunks = [Chunk(chunk_id=r[0], doc_id=r[1], document=r[2], filename=r[3], section=r[4],
                                section_path=r[5], kind=r[6], text=r[7]) for r in rows]
                matrix = np.vstack([np.frombuffer(r[8], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 0))
                self._cache = (chunks, matrix)
            return self._cache

    @staticmethod
    def _info(r) -> DocumentInfo:
        return DocumentInfo(doc_id=r[0], document=r[1], filename=r[2], version=r[3], chunks=r[4], uploaded_at=r[5])
