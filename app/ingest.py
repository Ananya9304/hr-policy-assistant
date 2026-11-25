import hashlib
import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from app.config import MAX_CHUNK_CHARS
from app.models import Chunk

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}

MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.!?]{1,80})$")
META_LINE = re.compile(r"^\*\*(.+?):\*\*\s*(.+)$")
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")


class IngestError(ValueError):
    pass


@dataclass
class Section:
    path: list[str]
    lines: list[str] = field(default_factory=list)


def extract_text(filename: str, data: bytes) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED:
        raise IngestError(f"Unsupported file type '{ext}'. Use .md, .txt or .pdf.")
    if not data.strip():
        raise IngestError("The file is empty.")

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise IngestError(f"Could not read PDF: {exc}") from exc
        if not text.strip():
            raise IngestError("PDF has no extractable text (is it a scanned image?).")
        return text

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestError("File is not readable text (unknown encoding).")


def _heading(line: str, is_markdown: bool):
    m = MD_HEADING.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    if not is_markdown:
        m = NUMBERED_HEADING.match(line.strip())
        if m:
            return m.group(1).count(".") + 2, line.strip()
    return None


def split_sections(text: str, is_markdown: bool) -> tuple[str | None, dict, list[Section]]:
    title, meta = None, {}
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current = Section(path=[])

    for raw in text.splitlines():
        line = raw.rstrip()
        h = _heading(line, is_markdown)
        if h:
            level, heading = h
            if level == 1 and title is None:
                title = heading
                continue
            if current.lines:
                sections.append(current)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            current = Section(path=[s[1] for s in stack])
            continue

        m = META_LINE.match(line.strip())
        if m and not stack:
            meta[m.group(1).strip().lower()] = m.group(2).strip()
            continue
        current.lines.append(line)

    if current.lines:
        sections.append(current)
    return title, meta, sections


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _split_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    for line in lines:
        kind = "table" if line.strip().startswith("|") else "text"
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(line)
        else:
            blocks.append((kind, [line]))
    return blocks


def _table_rows(lines: list[str]) -> list[str]:
    rows = [l for l in lines if l.strip() and not TABLE_SEPARATOR.match(l.strip())]
    if len(rows) < 2:
        return [" ".join(rows)] if rows else []
    header = _cells(rows[0])
    out = []
    for row in rows[1:]:
        cells = _cells(row)
        out.append(" | ".join(f"{h}: {c}" for h, c in zip(header, cells)))
    return out


def _pack_paragraphs(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > limit:
            pieces.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        pieces.append(buf)
    return pieces


def _clean(text: str) -> str:
    text = text.replace("**", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def make_doc_id(filename: str) -> str:
    return hashlib.sha1(filename.lower().encode()).hexdigest()[:12]


def chunk_document(filename: str, data: bytes) -> tuple[str, dict, list[Chunk]]:
    text = extract_text(filename, data)
    is_markdown = filename.lower().endswith((".md", ".markdown"))
    title, meta, sections = split_sections(text, is_markdown)

    document = meta.get("document title") or title or filename.rsplit(".", 1)[0]
    doc_id = make_doc_id(filename)
    chunks: list[Chunk] = []

    for sec in sections:
        section = sec.path[-1] if sec.path else "General"
        section_path = " > ".join(sec.path) if sec.path else "General"
        for kind, lines in _split_blocks(sec.lines):
            if kind == "table":
                pieces = [(p, "table_row") for p in _table_rows(lines)]
            else:
                body = _clean("\n".join(lines))
                pieces = [(p, "text") for p in _pack_paragraphs(body, MAX_CHUNK_CHARS)] if body else []
            for piece, piece_kind in pieces:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}-{len(chunks):03d}",
                    doc_id=doc_id,
                    document=document,
                    filename=filename,
                    section=section,
                    section_path=section_path,
                    kind=piece_kind,
                    text=piece,
                ))

    if not chunks:
        raise IngestError("No readable policy text found in the file.")
    return document, meta, chunks