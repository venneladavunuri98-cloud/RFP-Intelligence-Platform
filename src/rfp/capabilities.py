"""Capability store ingestion (guide §2.3): python -m rfp.capabilities [path]

Splits data/capabilities.md by '## ' sections, embeds each, and reloads the
capabilities table (full replace, so re-running is idempotent).
"""
import re
import sys

from . import config, db, llm


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """[(section_title, content)] for every '## ' heading with non-empty body."""
    parts = re.split(r"^##\s+(.+)$", markdown, flags=re.M)
    sections = []
    for title, body in zip(parts[1::2], parts[2::2]):
        body = body.strip()
        if body:
            sections.append((title.strip(), body))
    return sections


def ingest(path=None, uri=None, embed=llm.embed_documents) -> int:
    path = path or config.ROOT / "data" / "capabilities.md"
    sections = split_sections(open(path, encoding="utf-8").read())
    vectors = embed([f"{title}\n{content}" for title, content in sections])
    if len(vectors) != len(sections) or any(len(v) != config.EMBED_DIM for v in vectors):
        raise ValueError("embedding count/dimension mismatch")
    with db.connect(uri) as conn, conn.transaction():
        conn.execute("TRUNCATE capabilities RESTART IDENTITY")
        for (title, content), vec in zip(sections, vectors):
            conn.execute("INSERT INTO capabilities (section, content, embedding) "
                         "VALUES (%s, %s, %s::vector)", (title, content, str(vec)))
    return len(sections)


if __name__ == "__main__":
    db.init_db()
    print(f"ingested {ingest(sys.argv[1] if len(sys.argv) > 1 else None)} capability sections")
