"""
FUNDSCREENING - page-aware text chunker

Input:
    extracted_text/pages/<document_id>/page_*.txt

Output:
    graph_output/chunk_results/<document_id>_chunks.json

Preserves document and page provenance for later LLM extraction.
Run from project root:
    python src/chunker.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = PROJECT_ROOT / "extracted_text" / "pages"
OUTPUT_DIR = PROJECT_ROOT / "graph_output" / "chunk_results"

TARGET_CHARS = 6500
MAX_CHARS = 8000
MIN_CHARS = 2200
OVERLAP_CHARS = 400


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)\.txt$", path.name, re.I)
    if not match:
        raise ValueError(f"Invalid page filename: {path.name}")
    return int(match.group(1))


def read_page(path: Path) -> tuple[int, str]:
    number = page_number(path)
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"^\s*===== PAGE \d+ =====\s*",
        "",
        text,
        count=1,
        flags=re.I,
    )
    return number, clean_text(text)


def split_oversized(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= MAX_CHARS:
        return [text]

    pieces: list[str] = []
    remaining = text

    while len(remaining) > MAX_CHARS:
        candidate = remaining[:MAX_CHARS]
        cut = candidate.rfind("\n\n")

        if cut < MAX_CHARS * 0.55:
            cut = candidate.rfind(". ")
        if cut < MAX_CHARS * 0.55:
            cut = candidate.rfind("\n")
        if cut < MAX_CHARS * 0.55:
            cut = MAX_CHARS

        piece = clean_text(remaining[:cut])
        if piece:
            pieces.append(piece)

        remaining = remaining[cut:].lstrip()

    if remaining:
        pieces.append(remaining)

    return pieces


def looks_like_heading(block: str) -> bool:
    """Conservative heading detection; avoids treating normal prose as headings."""
    if "\n" in block or not block or len(block) > 140:
        return False
    s = block.strip()
    if re.match(r"^\d+(?:\.\d+)*[\).]?\s+\S+", s):
        return True
    if re.match(r"^[A-Z][A-Z0-9 &/\-]{4,100}$", s):
        return True
    if s.endswith(":") and 2 <= len(s.split()) <= 12:
        return True
    common = re.compile(
        r"^(executive summary|investment thesis|investment criteria|"
        r"company overview|business overview|market|financials?|"
        r"financial analysis|valuation|transaction|deal|risk|risks|"
        r"audit|due diligence|conclusion|appendix|recommendations?)$",
        re.I,
    )
    return bool(common.match(s))


def is_table_or_list(block: str) -> bool:
    lines = block.splitlines()
    if len(lines) < 2:
        return False
    bullet_count = sum(bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", x)) for x in lines)
    pipe_count = sum("|" in x for x in lines)
    return bullet_count >= 2 or pipe_count >= 2


def split_oversized_semantic(text: str) -> list[str]:
    text = clean_text(text)
    if len(text) <= MAX_CHARS:
        return [text]
    pieces = []
    remaining = text
    while len(remaining) > MAX_CHARS:
        candidate = remaining[:MAX_CHARS]
        cuts = [candidate.rfind("\n\n"), candidate.rfind(". "), candidate.rfind("; "), candidate.rfind("\n")]
        cut = max((x for x in cuts if x >= int(MAX_CHARS * 0.55)), default=MAX_CHARS)
        piece = clean_text(remaining[:cut])
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def page_units(page: int, text: str) -> list[dict[str, Any]]:
    blocks = [clean_text(x) for x in re.split(r"\n\s*\n", text)]
    blocks = [x for x in blocks if x]
    units = []
    section = ""
    for block in blocks:
        if looks_like_heading(block):
            section = block.strip()
            units.append({"page": page, "text": section, "section": section, "kind": "heading"})
            continue
        kind = "table_or_list" if is_table_or_list(block) else "paragraph"
        for piece in split_oversized_semantic(block):
            units.append({"page": page, "text": piece, "section": section, "kind": kind})
    return units


def make_chunks(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    current = []
    current_len = 0
    current_section = ""

    def flush():
        nonlocal current, current_len, current_section
        if not current:
            return
        body = "\n\n".join(x["text"] for x in current).strip()
        if body:
            section = current_section or current[0].get("section", "")
            if section and not body.startswith(section):
                body = f"[SECTION: {section}]\n\n{body}"
            chunks.append({
                "start_page": current[0]["page"],
                "end_page": current[-1]["page"],
                "section": section,
                "text": body,
            })
        current, current_len, current_section = [], 0, ""

    for unit in units:
        text = unit["text"]
        if not text:
            continue

        # Never put a new major heading into the previous completed section
        # when that section already has enough content.
        if unit["kind"] == "heading" and current and current_len >= MIN_CHARS:
            previous = current[-1]["text"]
            previous_page = current[-1]["page"]
            previous_section = current[-1].get("section", current_section)
            flush()
            overlap = clean_text(previous[-OVERLAP_CHARS:])
            if overlap:
                current.append({
                    "page": previous_page,
                    "text": "[CONTEXT OVERLAP]\n" + overlap,
                    "section": previous_section,
                    "kind": "overlap",
                })
                current_len = len(overlap) + 20

        addition = len(text) + (2 if current else 0)
        if current and current_len + addition > TARGET_CHARS and current_len >= MIN_CHARS:
            previous = current[-1]["text"]
            previous_page = current[-1]["page"]
            previous_section = current[-1].get("section", current_section)
            flush()
            overlap = clean_text(previous[-OVERLAP_CHARS:])
            if overlap:
                current.append({
                    "page": previous_page,
                    "text": "[CONTEXT OVERLAP]\n" + overlap,
                    "section": previous_section,
                    "kind": "overlap",
                })
                current_len = len(overlap) + 20

        current.append(unit)
        current_len += addition
        current_section = unit.get("section", current_section)

        if current_len >= MAX_CHARS:
            flush()

    flush()
    return chunks

def process_document(directory: Path) -> dict[str, Any]:
    document_id = directory.name
    files = sorted(directory.glob("page_*.txt"), key=page_number)

    if not files:
        raise ValueError(f"No page files found in {directory}")

    units: list[dict[str, Any]] = []
    pages_read: list[int] = []
    empty_pages: list[int] = []

    for file in files:
        page, text = read_page(file)
        pages_read.append(page)

        if text:
            units.extend(page_units(page, text))
        else:
            empty_pages.append(page)

    raw = make_chunks(units)

    chunks = []
    source_file = f"documents/{document_id}.pdf"

    for index, item in enumerate(raw):
        chunks.append({
            "document_id": document_id,
            "source_file": source_file,
            "chunk_id": f"{document_id}_chunk_{index + 1:04d}",
            "chunk_index": index,
            "start_page": item["start_page"],
            "end_page": item["end_page"],
            "section": item.get("section", ""),
            "character_count": len(item["text"]),
            "text": item["text"],
        })

    return {
        "document_id": document_id,
        "source_file": source_file,
        "page_count": len(files),
        "pages_read": pages_read,
        "empty_pages": empty_pages,
        "chunk_count": len(chunks),
        "target_chars": TARGET_CHARS,
        "max_chars": MAX_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "strategy": "page-aware, paragraph-aware, conservative section-aware chunking",
        "chunks": chunks,
    }


def main() -> None:
    if not PAGES_DIR.exists():
        raise SystemExit(
            f"Missing {PAGES_DIR}. Run src/pdf_extractor.py first."
        )

    directories = sorted(p for p in PAGES_DIR.iterdir() if p.is_dir())

    if not directories:
        raise SystemExit(f"No document folders found in {PAGES_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FUNDSCREENING - PAGE-AWARE CHUNKING")
    print("=" * 70)

    failures = 0

    for directory in directories:
        print(f"\nChunking: {directory.name}")

        try:
            result = process_document(directory)
            output = OUTPUT_DIR / f"{result['document_id']}_chunks.json"

            output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print(
                f"  pages={result['page_count']} "
                f"chunks={result['chunk_count']} "
                f"empty_pages={len(result['empty_pages'])}"
            )
            print(f"  output={output.relative_to(PROJECT_ROOT)}")

        except Exception as exc:
            failures += 1
            print(f"  FAILED: {exc}")

    print("\n" + "=" * 70)

    if failures:
        print(f"Finished with {failures} failure(s).")
        raise SystemExit(1)

    print("Chunking completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
