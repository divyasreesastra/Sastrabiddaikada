"""
Generic, ontology-driven graph extractor.

Design goals
------------
1. The ontology in schema/nodes.txt and schema/relationships.txt is the only
   source of truth for node/relationship types and properties.
2. No document-specific node names, keywords, metric names, or relationship
   names are hard-coded in the extraction logic.
3. A document does NOT need to contain every ontology node type or relationship.
   Missing types are reported as NOT_FOUND/NOT_APPLICABLE; placeholders are
   never created.
4. Chunk extraction is local. The complete document/graph is never sent to the
   LLM in one prompt.
5. Entity consolidation is generic and works for every node type.
6. Relationship recovery is generic and driven only by the ontology plus
   source-grounded evidence.
7. Disconnected nodes are not automatically deleted. A generic verifier checks
   support, duplicate/alias status, and explicit dependencies. Only
   source-grounded ontology-valid changes are applied.
8. Final node properties are strictly filtered to nodes.txt.
9. Every final relationship must have source-grounded evidence.
10. Final validation reports the actual ontology coverage; it does not demand
    that all ontology types occur in every document.

Expected project layout
-----------------------
project/
  src/
    graph_extractor_generic.py
  schema/
    nodes.txt
    relationships.txt
  graph_output/
    chunk_results/
      *_chunks.json
  .env

Environment
-----------
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=<your deployment>

Run
---
python src/graph_extractor_generic.py

Output
------
graph_output/graph_extract.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


# ============================================================================
# PATHS / SETTINGS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = PROJECT_ROOT / "graph_output" / "chunk_results"
SCHEMA_DIR = PROJECT_ROOT / "schema"
NODES_SCHEMA = SCHEMA_DIR / "nodes.txt"
RELATIONSHIPS_SCHEMA = SCHEMA_DIR / "relationships.txt"
OUTPUT_DIR = PROJECT_ROOT / "graph_output"
OUTPUT_FILE = OUTPUT_DIR / "graph_extract.json"

MAX_RETRIES = 4
RETRY_BASE_SECONDS = 3
EXTRACTION_MAX_TOKENS = 14000
FEEDBACK_MAX_TOKENS = 12000

# Maximum amount of source text placed in one verification/recovery request.
# This is deliberately bounded so the design scales to larger documents.
MAX_CONTEXT_CHARS = 30000

# Number of nodes in one generic consolidation/verification batch.
NODE_BATCH_SIZE = 20

# Maximum source chunks retrieved for a relationship signature review.
RELATIONSHIP_CHUNK_LIMIT = 12


# ============================================================================
# GENERIC SYSTEM PROMPTS
# ============================================================================

EXTRACTION_SYSTEM = r"""
You are a precise ontology-constrained knowledge-graph extraction engine.

The supplied files define the complete ontology. Treat them as the only
authority for node types, node properties, relationship types, and relationship
endpoint signatures.

Extract facts from ONLY the supplied source chunk.

RULES
-----
1. Never invent facts.
2. Never infer a relationship merely because two entities occur in the same
   paragraph or chunk.
3. Never create a node type outside the supplied ontology.
4. Never create a relationship type outside the supplied ontology.
5. Use ONLY properties defined for the corresponding node type.
6. Preserve source terminology and meaning.
7. Extract entities/facts that are explicitly supported by the source.
8. A measurable fact should be represented using the FinancialMetric node type
   only when FinancialMetric exists in the supplied ontology AND the source
   fact semantically belongs to that type. Do not assume every number is a
   metric.
9. Do not create a node merely because the ontology contains that type.
10. Do not create placeholder nodes.
11. If the same real-world entity appears in multiple chunks, use a stable
    semantic node_id where possible. The downstream consolidation stage will
    also reconcile aliases.
12. A relationship is allowed only when its source and target node types match
    an allowed ontology signature AND the source text explicitly supports the
    relationship.
13. Every extracted node and relationship must contain at least one short,
    verbatim evidence quote from the supplied source chunk.
14. Keep evidence tied to the supplied document/chunk/page metadata.
15. Do not turn a conclusion or recommendation into a fact unless the source
    explicitly states it.
16. Do not output screening judgments such as good/bad/pass/fail unless those
    are literally stated and represented by an allowed schema property.
17. Return JSON only.

OUTPUT
------
{
  "nodes": [
    {
      "node_id": "semantic_or_stable_id",
      "node_type": "AllowedType",
      "properties": {},
      "evidence": [
        {
          "document_id": "...",
          "source_file": "...",
          "page": 1,
          "chunk_id": "...",
          "quote": "short exact quote"
        }
      ]
    }
  ],
  "relationships": [
    {
      "relationship_id": "stable_relationship_id",
      "relationship_type": "AllowedRelationship",
      "source_node_id": "...",
      "target_node_id": "...",
      "properties": {},
      "evidence": [
        {
          "document_id": "...",
          "source_file": "...",
          "page": 1,
          "chunk_id": "...",
          "quote": "short exact quote"
        }
      ]
    }
  ],
  "warnings": []
}
""".strip()


CONSOLIDATION_SYSTEM = r"""
You are a generic entity-resolution and graph-quality verifier.

You are given:
- the authoritative ontology,
- a batch of node instances of ONE node type,
- source evidence for those instances.

Your job is NOT to invent entities.

For every supplied node:
1. Determine whether it is supported by the supplied evidence.
2. Determine whether another supplied node is an alias/duplicate of the same
   real-world or semantic entity.
3. Do not merge two observations merely because their names are similar.
   Different values, periods, dates, scenarios, versions, or other explicit
   distinctions may represent legitimate separate instances.
4. Identify explicit dependencies/relationships among supplied nodes only when
   the source text supports them and the ontology permits them.
5. Do not create new node instances.
6. Do not delete nodes yourself; return decisions for the deterministic
   post-processor.
7. A proposed relationship must use a relationship type allowed for the two
   endpoint node types and must include a short verbatim evidence quote.
8. Return JSON only.

OUTPUT
------
{
  "node_reviews": [
    {
      "node_id": "...",
      "supported": true,
      "classification": "independent|duplicate|observation|unsupported",
      "duplicate_of": null,
      "reason": "..."
    }
  ],
  "relationship_candidates": [
    {
      "relationship_type": "...",
      "source_node_id": "...",
      "target_node_id": "...",
      "properties": {},
      "evidence": [
        {
          "chunk_id": "...",
          "quote": "..."
        }
      ]
    }
  ]
}
""".strip()


RELATIONSHIP_SYSTEM = r"""
You are a generic ontology relationship auditor.

Use ONLY the supplied ontology, canonical node catalog, and source chunks.

Find relationships explicitly stated or directly expressed by the source.

Rules:
1. Do not infer relationships from simple co-occurrence.
2. Do not invent endpoints.
3. Use only the supplied canonical node IDs.
4. Use only relationship types explicitly allowed by the supplied ontology.
5. Every accepted relationship must have a short verbatim source quote.
6. If a relationship type is not supported by the source, return nothing for it.
7. A missing relationship is not an error; absence of evidence means no edge.
8. Return JSON only.

OUTPUT
------
{
  "relationships": [
    {
      "relationship_type": "...",
      "source_node_id": "...",
      "target_node_id": "...",
      "properties": {},
      "evidence": [
        {
          "chunk_id": "...",
          "quote": "..."
        }
      ]
    }
  ]
}
""".strip()


COVERAGE_SYSTEM = r"""
You are a generic ontology coverage auditor.

The ontology may contain node types and relationship types that are not
present in a particular document. Do NOT invent missing entities.

Review the supplied source chunks and report only:
- additional source-supported nodes of the requested type, if any;
- additional source-supported relationships for the requested endpoint
  signature, if any.

Every returned item requires a verbatim source quote.

Do not use document-specific assumptions. Do not create placeholders.
Return JSON only.
""".strip()


# ============================================================================
# ENVIRONMENT / CLIENT
# ============================================================================

def load_environment() -> None:
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def validate_environment() -> None:
    missing = []
    if not env("AZURE_OPENAI_ENDPOINT"):
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not env("AZURE_OPENAI_API_KEY"):
        missing.append("AZURE_OPENAI_API_KEY")
    if not env("AZURE_OPENAI_DEPLOYMENT"):
        missing.append("AZURE_OPENAI_DEPLOYMENT")

    if missing:
        raise SystemExit(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    if AzureOpenAI is None:
        raise SystemExit(
            "The 'openai' package is missing. Install/upgrade it before running "
            "this extractor."
        )


def create_client() -> Any:
    return AzureOpenAI(
        azure_endpoint=env("AZURE_OPENAI_ENDPOINT"),
        api_key=env("AZURE_OPENAI_API_KEY"),
        api_version=env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


# ============================================================================
# FILE / SCHEMA HELPERS
# ============================================================================

def read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_DIR.exists():
        raise SystemExit(f"Chunk directory not found: {CHUNKS_DIR}")

    files = sorted(CHUNKS_DIR.glob("*_chunks.json"))
    if not files:
        raise SystemExit(f"No *_chunks.json files found in {CHUNKS_DIR}")

    chunks: list[dict[str, Any]] = []

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for chunk in payload.get("chunks", []):
            c = dict(chunk)
            c["_chunk_file"] = str(path.relative_to(PROJECT_ROOT))
            chunks.append(c)

    return chunks


def parse_nodes_schema(text: str) -> tuple[list[str], dict[str, set[str]]]:
    """
    Generic parser for nodes.txt.

    It intentionally does not know the names of node types.
    """
    node_types: list[str] = []
    properties: dict[str, set[str]] = {}

    current: str | None = None
    in_properties = False

    heading = re.compile(
        r"^\s*(?:\d+[\.\)]\s*)?([A-Za-z][A-Za-z0-9_]*)\s*$"
    )

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            continue

        if re.match(r"^properties\s*:\s*$", line, re.I):
            in_properties = True
            continue

        # Common schema formats:
        # "1. Company"
        # "Company"
        # "### Company"
        candidate = line.lstrip("#").strip()
        m = heading.match(candidate)

        if m and not line.startswith("-"):
            name = m.group(1)
            # Avoid interpreting generic section headings as node types.
            if name.lower() in {
                "properties", "node", "nodes", "schema", "description"
            }:
                continue

            # A node heading normally precedes a Properties section.
            if name not in node_types:
                current = name
                node_types.append(name)
                properties[name] = set()
                in_properties = False
                continue

        if current and in_properties:
            pm = re.match(
                r"^[-*]\s*([A-Za-z][A-Za-z0-9_]*)"
                r"(?:\s*\(.*)?$",
                line,
            )
            if pm:
                properties[current].add(pm.group(1))

    if not node_types:
        raise RuntimeError("No node types could be parsed from nodes.txt")

    return node_types, properties


def parse_relationship_schema(
    text: str,
    node_types: set[str],
) -> tuple[dict[tuple[str, str], set[str]], list[dict[str, str]]]:
    """
    Generic parser for relationships.txt.

    IMPORTANT: parsing stops at the relationship-property convention section.
    Therefore properties such as description/status/value cannot accidentally
    become relationship types.
    """
    allowed: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows: list[dict[str, str]] = {}

    current_signature: tuple[str, str] | None = None

    stop_patterns = {
        "general relationship property convention",
        "relationship property convention",
    }

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.lower() in stop_patterns:
            break

        # Signature: Company -> Deal
        m = re.match(
            r"^([A-Za-z][A-Za-z0-9_]*)\s*->\s*"
            r"([A-Za-z][A-Za-z0-9_]*)$",
            line,
        )
        if m:
            s, t = m.group(1), m.group(2)
            if s in node_types and t in node_types:
                current_signature = (s, t)
                allowed[current_signature]  # initialize
            else:
                current_signature = None
            continue

        if current_signature:
            rm = re.match(
                r"^[-*]\s*([A-Z][A-Z0-9_]*)\s*$",
                line,
            )
            if rm:
                rel = rm.group(1)
                allowed[current_signature].add(rel)

    rows_list = [
        {
            "source_type": s,
            "relationship_type": rel,
            "target_type": t,
        }
        for (s, t), rels in allowed.items()
        for rel in sorted(rels)
    ]

    return dict(allowed), rows_list


def load_ontology() -> dict[str, Any]:
    nodes_text = read_text(NODES_SCHEMA)
    rel_text = read_text(RELATIONSHIPS_SCHEMA)

    node_types, node_properties = parse_nodes_schema(nodes_text)
    rel_allowed, rel_rows = parse_relationship_schema(
        rel_text, set(node_types)
    )

    rel_types = sorted({
        r["relationship_type"] for r in rel_rows
    })

    return {
        "nodes_text": nodes_text,
        "relationships_text": rel_text,
        "node_types": node_types,
        "node_properties": node_properties,
        "rel_allowed": rel_allowed,
        "rel_rows": rel_rows,
        "relationship_types": rel_types,
    }


# ============================================================================
# JSON / TEXT HELPERS
# ============================================================================

def parse_json_response(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object.")
        value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object.")

    return value


def norm_text(value: Any) -> str:
    s = str(value or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_identity_text(value: Any) -> str:
    s = norm_text(value)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(
        r"\b(inc|incorporated|ltd|limited|llc|corp|corporation|company|co)\b",
        " ",
        s,
    )
    return re.sub(r"\s+", " ", s).strip()


def stable_hash(*parts: Any, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def evidence_key(item: dict[str, Any]) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


def dedup_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []

    for item in items:
        if not isinstance(item, dict):
            continue
        key = evidence_key(item)
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


def quote_exists(quote: str, chunks: list[dict[str, Any]]) -> bool:
    q = re.sub(r"\s+", " ", str(quote or "")).strip().lower()
    if len(q) < 8:
        return False

    for c in chunks:
        text = re.sub(r"\s+", " ", str(c.get("text", ""))).strip().lower()
        if q in text:
            return True

    return False


def valid_evidence(
    evidence: Any,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []

    out = []
    for e in evidence:
        if not isinstance(e, dict):
            continue
        quote = str(e.get("quote") or "").strip()
        if quote and quote_exists(quote, chunks):
            out.append({
                "document_id": e.get("document_id"),
                "source_file": e.get("source_file"),
                "page": e.get("page"),
                "chunk_id": e.get("chunk_id"),
                "quote": quote,
            })
    return dedup_evidence(out)


def chunk_header(c: dict[str, Any]) -> str:
    return (
        f"--- chunk={c.get('chunk_id')} "
        f"document={c.get('document_id')} "
        f"pages={c.get('start_page')}-{c.get('end_page')} ---"
    )


def bounded_context(
    chunks: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    blocks = []
    used = 0

    for c in chunks:
        block = (
            f"\n{chunk_header(c)}\n"
            f"{c.get('text', '')}\n"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)

    return "".join(blocks)


# ============================================================================
# LLM CALL
# ============================================================================

def llm_json(
    client: Any,
    system: str,
    user: str,
    max_tokens: int,
) -> dict[str, Any]:
    last: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=env("AZURE_OPENAI_DEPLOYMENT"),
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return parse_json_response(
                response.choices[0].message.content or "{}"
            )
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last}")


# ============================================================================
# GENERIC EXTRACTION
# ============================================================================

def schema_for_prompt(ontology: dict[str, Any]) -> str:
    lines = ["NODE TYPES AND PROPERTIES"]

    for t in ontology["node_types"]:
        props = sorted(ontology["node_properties"].get(t, set()))
        lines.append(f"- {t}: {', '.join(props) if props else '(no properties listed)'}")

    lines.append("")
    lines.append("ALLOWED RELATIONSHIP SIGNATURES")

    for sig, rels in sorted(ontology["rel_allowed"].items()):
        lines.append(
            f"- {sig[0]} -> {sig[1]}: {', '.join(sorted(rels))}"
        )

    return "\n".join(lines)


def extraction_prompt(
    chunk: dict[str, Any],
    ontology: dict[str, Any],
) -> str:
    return f"""
AUTHORITATIVE ONTOLOGY
{schema_for_prompt(ontology)}

SOURCE METADATA
document_id: {chunk.get("document_id")}
source_file: {chunk.get("source_file")}
chunk_id: {chunk.get("chunk_id")}
pages: {chunk.get("start_page")}-{chunk.get("end_page")}

SOURCE TEXT
====================
{chunk.get("text", "")}
====================

Extract every source-supported entity and relationship that belongs to the
authoritative ontology.

IMPORTANT:
- Do not try to populate node types that are absent from this chunk.
- Do not create placeholders for ontology types.
- Do not use any property not listed for the node type.
- A relationship must be explicitly supported by the source.
- Evidence quotes must be verbatim.
- If a numeric fact is not semantically a FinancialMetric, represent it using
  another appropriate ontology node/property instead of forcing a metric.
- If the ontology has no suitable node type for a fact, omit it rather than
  inventing a node type.

Return JSON only.
""".strip()


def normalize_extraction(
    result: dict[str, Any],
    chunk: dict[str, Any],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    valid_types = set(ontology["node_types"])

    nodes = []
    for n in result.get("nodes", []) if isinstance(result.get("nodes"), list) else []:
        if not isinstance(n, dict):
            continue

        node_id = str(n.get("node_id") or "").strip()
        node_type = str(n.get("node_type") or "").strip()

        if not node_id or node_type not in valid_types:
            continue

        props = n.get("properties")
        if not isinstance(props, dict):
            props = {}

        allowed = ontology["node_properties"].get(node_type, set())
        props = {
            k: v for k, v in props.items()
            if k in allowed
        }

        evidence = []
        for e in n.get("evidence", []) if isinstance(n.get("evidence"), list) else []:
            if not isinstance(e, dict):
                continue
            quote = str(e.get("quote") or "").strip()
            if not quote:
                continue
            evidence.append({
                "document_id": e.get("document_id", chunk.get("document_id")),
                "source_file": e.get("source_file", chunk.get("source_file")),
                "page": e.get("page", chunk.get("start_page")),
                "chunk_id": e.get("chunk_id", chunk.get("chunk_id")),
                "quote": quote,
            })

        evidence = valid_evidence(evidence, [chunk])
        if not evidence:
            # An extracted node without source-grounded evidence is discarded.
            continue

        nodes.append({
            "node_id": node_id,
            "node_type": node_type,
            "properties": props,
            "evidence": evidence,
        })

    node_ids = {n["node_id"] for n in nodes}

    relationships = []
    for r in result.get("relationships", []) if isinstance(result.get("relationships"), list) else []:
        if not isinstance(r, dict):
            continue

        rid = str(r.get("relationship_id") or "").strip()
        rt = str(r.get("relationship_type") or "").strip()
        sid = str(r.get("source_node_id") or "").strip()
        tid = str(r.get("target_node_id") or "").strip()

        if not rid or not rt or sid not in node_ids or tid not in node_ids:
            continue

        s = next(n for n in nodes if n["node_id"] == sid)
        t = next(n for n in nodes if n["node_id"] == tid)

        if rt not in ontology["rel_allowed"].get(
            (s["node_type"], t["node_type"]), set()
        ):
            continue

        props = r.get("properties")
        if not isinstance(props, dict):
            props = {}

        evidence = valid_evidence(
            r.get("evidence"),
            [chunk],
        )
        if not evidence:
            continue

        relationships.append({
            "relationship_id": rid,
            "relationship_type": rt,
            "source_node_id": sid,
            "target_node_id": tid,
            "properties": props,
            "evidence": evidence,
        })

    return {
        "nodes": nodes,
        "relationships": relationships,
        "warnings": result.get("warnings", [])
        if isinstance(result.get("warnings"), list)
        else [],
    }


def extract_all_chunks(
    client: Any,
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    failures = []

    for index, chunk in enumerate(chunks, 1):
        print(
            f"[{index}/{len(chunks)}] "
            f"{chunk.get('chunk_id')} "
            f"pages {chunk.get('start_page')}-{chunk.get('end_page')}"
        )

        try:
            result = llm_json(
                client,
                EXTRACTION_SYSTEM,
                extraction_prompt(chunk, ontology),
                EXTRACTION_MAX_TOKENS,
            )
            normalized = normalize_extraction(result, chunk, ontology)
            normalized["document_id"] = chunk.get("document_id")
            normalized["chunk_id"] = chunk.get("chunk_id")
            results.append(normalized)
            print(
                f"  extracted nodes={len(normalized['nodes'])} "
                f"relationships={len(normalized['relationships'])}"
            )
        except Exception as exc:
            failures.append({
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "error": str(exc),
            })
            print(f"  FAILED: {exc}")

    return results, failures


# ============================================================================
# MERGE CHUNK RESULTS
# ============================================================================

def merge_chunk_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    relationships: dict[str, dict[str, Any]] = {}
    warnings = []

    for result in results:
        warnings.extend(result.get("warnings", []))

        for node in result.get("nodes", []):
            nid = node["node_id"]

            if nid not in nodes:
                nodes[nid] = dict(node)
                nodes[nid]["properties"] = dict(node.get("properties", {}))
                nodes[nid]["evidence"] = list(node.get("evidence", []))
                continue

            existing = nodes[nid]
            for k, v in node.get("properties", {}).items():
                if k not in existing["properties"] or existing["properties"][k] in ("", None, []):
                    existing["properties"][k] = v

            existing["evidence"] = dedup_evidence(
                existing.get("evidence", []) + node.get("evidence", [])
            )

        for rel in result.get("relationships", []):
            # IDs generated by different chunks may differ. The actual semantic
            # identity is the endpoint/type triple; that is handled below.
            key = (
                rel["source_node_id"],
                rel["relationship_type"],
                rel["target_node_id"],
            )

            existing = relationships.get(str(key))

            if existing is None:
                relationships[str(key)] = dict(rel)
                relationships[str(key)]["properties"] = dict(rel.get("properties", {}))
                relationships[str(key)]["evidence"] = list(rel.get("evidence", []))
            else:
                for k, v in rel.get("properties", {}).items():
                    if k not in existing["properties"] or existing["properties"][k] in ("", None, []):
                        existing["properties"][k] = v
                existing["evidence"] = dedup_evidence(
                    existing.get("evidence", []) + rel.get("evidence", [])
                )

    final_relationships = []
    for key, rel in relationships.items():
        source, rt, target = key.strip("()").split(", ", 2)
        rel["relationship_id"] = "rel_" + stable_hash(
            source, rt, target
        )
        final_relationships.append(rel)

    return {
        "nodes": list(nodes.values()),
        "relationships": final_relationships,
        "warnings": sorted(set(map(str, warnings))),
    }


# ============================================================================
# GENERIC NODE IDENTITY / ENTITY CONSOLIDATION
# ============================================================================

def identity_fields_for_type(
    node_type: str,
    properties: dict[str, Any],
) -> list[tuple[str, str]]:
    """
    Generic identity discovery.

    There is intentionally no list such as:
      FinancialMetric -> metric_name
      Company -> name

    Instead, identity candidates are derived from property names defined by
    the ontology and from actual populated properties.

    Strong identifiers are *_id / id / code / key.
    Name-like identifiers are *_name / name / title.
    Other populated descriptive fields are not used as identity.
    """
    items = []

    for key, value in properties.items():
        if value in (None, "", [], {}):
            continue

        k = key.lower()

        if (
            k == "id"
            or k.endswith("_id")
            or k.endswith("code")
            or k.endswith("_key")
        ):
            items.append(("strong", norm_identity_text(value)))

    for key, value in properties.items():
        if value in (None, "", [], {}):
            continue

        k = key.lower()

        if (
            k == "name"
            or k.endswith("_name")
            or k in {"title", "label"}
        ):
            items.append(("name", norm_identity_text(value)))

    return items


def node_display_name(node: dict[str, Any]) -> str:
    p = node.get("properties", {})
    candidates = []

    for key, value in p.items():
        k = key.lower()
        if value not in (None, "", [], {}) and (
            k == "name"
            or k.endswith("_name")
            or k in {"title", "label"}
        ):
            candidates.append(str(value))

    if candidates:
        return candidates[0]

    # Fall back to any useful textual property, without assuming a document
    # specific property.
    for key, value in p.items():
        if isinstance(value, str) and value.strip():
            return value[:100]

    return node.get("node_id", "")


def merge_node_properties(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    out = dict(left)

    for key, value in right.items():
        if key not in out or out[key] in (None, "", [], {}):
            out[key] = value

    return out


def deterministic_exact_consolidation(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """
    First consolidation pass.

    Only merges nodes when a strong identifier or exact normalized name clearly
    identifies the same node type. It never uses metric-specific or
    document-specific rules.
    """
    canonical: dict[tuple[Any, ...], dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    merges = []

    for node in graph["nodes"]:
        node_type = node["node_type"]
        props = node.get("properties", {})
        identity = identity_fields_for_type(node_type, props)

        strong = [v for kind, v in identity if kind == "strong" and v]
        names = [v for kind, v in identity if kind == "name" and v]

        # Strong identifier wins. Otherwise normalized name is used.
        if strong:
            key = (node_type, "strong", strong[0])
        elif names:
            key = (node_type, "name", names[0])
        else:
            # No semantic identity: keep as a separate node. This is important
            # for observation-like nodes where merging would be unsafe.
            key = (
                node_type,
                "opaque",
                node.get("node_id"),
            )

        if key not in canonical:
            canonical[key] = node
            aliases[node["node_id"]] = node["node_id"]
        else:
            target = canonical[key]
            aliases[node["node_id"]] = target["node_id"]

            target["properties"] = merge_node_properties(
                target.get("properties", {}),
                node.get("properties", {}),
            )
            target["evidence"] = dedup_evidence(
                target.get("evidence", [])
                + node.get("evidence", [])
            )

            merges.append({
                "from": node["node_id"],
                "to": target["node_id"],
                "reason": "same node type and exact normalized strong identifier/name",
            })

    graph["nodes"] = list(canonical.values())
    graph["relationships"] = remap_relationships(
        graph.get("relationships", []),
        aliases,
    )

    graph["deterministic_consolidation"] = {
        "merged_count": len(merges),
        "merges": merges,
    }

    return graph


def remap_relationships(
    relationships: list[dict[str, Any]],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}

    for rel in relationships:
        sid = aliases.get(
            rel.get("source_node_id"),
            rel.get("source_node_id"),
        )
        tid = aliases.get(
            rel.get("target_node_id"),
            rel.get("target_node_id"),
        )
        rt = rel.get("relationship_type")

        if not sid or not tid or not rt:
            continue

        key = (sid, rt, tid)

        if key not in merged:
            item = dict(rel)
            item["source_node_id"] = sid
            item["target_node_id"] = tid
            item["relationship_id"] = "rel_" + stable_hash(*key)
            merged[key] = item
        else:
            merged[key]["properties"] = merge_node_properties(
                merged[key].get("properties", {}),
                rel.get("properties", {}),
            )
            merged[key]["evidence"] = dedup_evidence(
                merged[key].get("evidence", [])
                + rel.get("evidence", [])
            )

    return list(merged.values())


# ============================================================================
# GENERIC LLM CONSOLIDATION
# ============================================================================

def node_catalog(nodes: list[dict[str, Any]]) -> str:
    lines = []
    for n in nodes:
        lines.append(
            f"{n['node_id']} | {n['node_type']} | "
            f"{json.dumps(n.get('properties', {}), ensure_ascii=False)}"
        )
    return "\n".join(lines)


def retrieve_for_nodes(
    batch: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Generic retrieval based on the actual values present in node properties.
    No domain keyword dictionary is used.
    """
    terms = []

    for node in batch:
        for value in node.get("properties", {}).values():
            if isinstance(value, str):
                cleaned = norm_text(value)
                if len(cleaned) >= 4:
                    terms.append(cleaned[:160])

    scored = []
    for chunk in chunks:
        text = norm_text(chunk.get("text", ""))
        score = 0

        for term in terms:
            if term and term in text:
                score += 1

        if score:
            scored.append((score, len(text), chunk))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in scored[:limit]]


def consolidate_with_llm(
    client: Any,
    graph: dict[str, Any],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """
    Generic alias/duplicate review.

    This does not hard-code any node type. It processes whatever node types
    actually exist in the ontology and graph.
    """
    reviews = []
    candidate_relationships = []
    calls = 0

    for node_type in ontology["node_types"]:
        typed = [
            n for n in graph["nodes"]
            if n["node_type"] == node_type
        ]

        for start in range(0, len(typed), NODE_BATCH_SIZE):
            batch = typed[start:start + NODE_BATCH_SIZE]
            if not batch:
                continue

            relevant = retrieve_for_nodes(batch, chunks)

            # Always include direct evidence chunks first.
            evidence_ids = {
                e.get("chunk_id")
                for n in batch
                for e in n.get("evidence", [])
                if e.get("chunk_id")
            }

            direct = [
                c for c in chunks
                if c.get("chunk_id") in evidence_ids
            ]

            seen = {c.get("chunk_id") for c in direct}
            relevant = direct + [
                c for c in relevant
                if c.get("chunk_id") not in seen
            ]
            relevant = relevant[:10]

            if not relevant:
                continue

            user = f"""
ONTOLOGY
{schema_for_prompt(ontology)}

NODE TYPE UNDER REVIEW
{node_type}

CANONICAL CANDIDATE NODES
{node_catalog(batch)}

SOURCE EVIDENCE
{bounded_context(relevant)}

Review every supplied node. Return duplicate/alias decisions only when
supported by the evidence. Also return explicit ontology-valid relationships
among supplied nodes if the source states them.
""".strip()

            result = llm_json(
                client,
                CONSOLIDATION_SYSTEM,
                user,
                FEEDBACK_MAX_TOKENS,
            )

            reviews.extend(
                result.get("node_reviews", [])
                if isinstance(result.get("node_reviews"), list)
                else []
            )
            candidate_relationships.extend(
                result.get("relationship_candidates", [])
                if isinstance(result.get("relationship_candidates"), list)
                else []
            )
            calls += 1

    return {
        "calls": calls,
        "node_reviews": reviews,
        "relationship_candidates": candidate_relationships,
    }


def apply_llm_consolidation(
    graph: dict[str, Any],
    report: dict[str, Any],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply ONLY high-confidence duplicate decisions returned by the verifier.

    The verifier cannot create nodes. It can only nominate an existing supplied
    node as duplicate_of another existing supplied node.
    """
    node_ids = {n["node_id"] for n in graph["nodes"]}
    aliases: dict[str, str] = {}
    applied = []
    rejected = []

    for review in report.get("node_reviews", []):
        if not isinstance(review, dict):
            continue

        nid = review.get("node_id")
        dup = review.get("duplicate_of")

        if not nid or nid not in node_ids:
            continue

        if not dup:
            continue

        if dup not in node_ids or dup == nid:
            rejected.append({
                "node_id": nid,
                "duplicate_of": dup,
                "reason": "target is not another existing node",
            })
            continue

        # Require explicit duplicate classification.
        if review.get("classification") != "duplicate":
            rejected.append({
                "node_id": nid,
                "duplicate_of": dup,
                "reason": "verifier did not classify as duplicate",
            })
            continue

        source = next(n for n in graph["nodes"] if n["node_id"] == nid)
        target = next(n for n in graph["nodes"] if n["node_id"] == dup)

        if source["node_type"] != target["node_type"]:
            rejected.append({
                "node_id": nid,
                "duplicate_of": dup,
                "reason": "node types differ",
            })
            continue

        aliases[nid] = dup
        applied.append({
            "from": nid,
            "to": dup,
            "reason": review.get("reason", ""),
        })

    if aliases:
        by_id = {n["node_id"]: n for n in graph["nodes"]}

        for old_id, new_id in aliases.items():
            by_id[new_id]["properties"] = merge_node_properties(
                by_id[new_id].get("properties", {}),
                by_id[old_id].get("properties", {}),
            )
            by_id[new_id]["evidence"] = dedup_evidence(
                by_id[new_id].get("evidence", [])
                + by_id[old_id].get("evidence", [])
            )

        graph["nodes"] = [
            n for n in graph["nodes"]
            if n["node_id"] not in aliases
        ]
        graph["relationships"] = remap_relationships(
            graph["relationships"],
            aliases,
        )

    graph["llm_consolidation_applied"] = {
        "merged_count": len(applied),
        "merges": applied,
        "rejected": rejected,
    }

    return graph


# ============================================================================
# GENERIC RELATIONSHIP RECOVERY
# ============================================================================

def node_evidence_chunks(
    nodes: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = {
        e.get("chunk_id")
        for n in nodes
        for e in n.get("evidence", [])
        if e.get("chunk_id")
    }

    return [c for c in chunks if c.get("chunk_id") in ids]


def retrieve_for_endpoint_sets(
    source_nodes: list[dict[str, Any]],
    target_nodes: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    limit: int = RELATIONSHIP_CHUNK_LIMIT,
) -> list[dict[str, Any]]:
    """
    Generic retrieval. It uses the actual canonical node values, never a
    hard-coded domain vocabulary.
    """
    nodes = source_nodes + target_nodes

    direct = node_evidence_chunks(nodes, chunks)

    terms = []
    for n in nodes:
        for value in n.get("properties", {}).values():
            if isinstance(value, str):
                v = norm_text(value)
                if len(v) >= 4:
                    terms.append(v[:160])

    scored = []
    for c in chunks:
        text = norm_text(c.get("text", ""))
        score = sum(1 for term in terms if term and term in text)
        if score:
            scored.append((score, len(text), c))

    scored.sort(key=lambda x: (-x[0], x[1]))

    result = []
    seen = set()

    for c in direct:
        cid = c.get("chunk_id")
        if cid not in seen:
            result.append(c)
            seen.add(cid)

    for _, _, c in scored:
        if len(result) >= limit:
            break
        cid = c.get("chunk_id")
        if cid not in seen:
            result.append(c)
            seen.add(cid)

    return result[:limit]


def relationship_catalog(
    source_nodes: list[dict[str, Any]],
    target_nodes: list[dict[str, Any]],
) -> str:
    return node_catalog(source_nodes + target_nodes)


def recover_relationships(
    client: Any,
    graph: dict[str, Any],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """
    Review each ontology endpoint signature that is applicable to the current
    graph.

    The LLM sees only:
      - source node instances,
      - target node instances,
      - allowed relationship types for that signature,
      - relevant source chunks.

    It never sees the entire graph.
    """
    node_by_type = defaultdict(list)
    for n in graph["nodes"]:
        node_by_type[n["node_type"]].append(n)

    existing = {
        (
            r["source_node_id"],
            r["relationship_type"],
            r["target_node_id"],
        )
        for r in graph["relationships"]
    }

    accepted = []
    rejected = []
    reviews = []

    for (source_type, target_type), allowed_types in sorted(
        ontology["rel_allowed"].items()
    ):
        sources = node_by_type.get(source_type, [])
        targets = node_by_type.get(target_type, [])

        if not sources or not targets:
            reviews.append({
                "signature": f"{source_type}->{target_type}",
                "status": "not_applicable",
                "reason": "One or both endpoint node types are absent.",
            })
            continue

        relevant = retrieve_for_endpoint_sets(
            sources,
            targets,
            chunks,
            RELATIONSHIP_CHUNK_LIMIT,
        )

        if not relevant:
            reviews.append({
                "signature": f"{source_type}->{target_type}",
                "status": "no_relevant_source_chunks",
            })
            continue

        user = f"""
ONTOLOGY
{schema_for_prompt(ontology)}

ENDPOINT SIGNATURE
{source_type} -> {target_type}

ALLOWED RELATIONSHIP TYPES
{json.dumps(sorted(allowed_types))}

CANONICAL SOURCE NODES
{node_catalog(sources)}

CANONICAL TARGET NODES
{node_catalog(targets)}

SOURCE CHUNKS
{bounded_context(relevant)}

Find only relationships explicitly supported by these source chunks.
Do not assume every possible relationship exists.
""".strip()

        result = llm_json(
            client,
            RELATIONSHIP_SYSTEM,
            user,
            FEEDBACK_MAX_TOKENS,
        )

        local = 0

        for r in result.get("relationships", []) if isinstance(result.get("relationships"), list) else []:
            if not isinstance(r, dict):
                continue

            rt = str(r.get("relationship_type") or "").strip()
            sid = str(r.get("source_node_id") or "").strip()
            tid = str(r.get("target_node_id") or "").strip()

            if rt not in allowed_types:
                rejected.append({
                    "relationship": r,
                    "reason": "relationship type not allowed for signature",
                })
                continue

            if not any(n["node_id"] == sid for n in sources):
                rejected.append({
                    "relationship": r,
                    "reason": "source endpoint not in supplied catalog",
                })
                continue

            if not any(n["node_id"] == tid for n in targets):
                rejected.append({
                    "relationship": r,
                    "reason": "target endpoint not in supplied catalog",
                })
                continue

            evidence = valid_evidence(
                r.get("evidence"),
                chunks,
            )

            if not evidence:
                rejected.append({
                    "relationship": r,
                    "reason": "evidence quote not found in source chunks",
                })
                continue

            key = (sid, rt, tid)

            if key in existing:
                continue

            rel = {
                "relationship_id": "rel_" + stable_hash(*key),
                "relationship_type": rt,
                "source_node_id": sid,
                "target_node_id": tid,
                "properties": (
                    r.get("properties", {})
                    if isinstance(r.get("properties"), dict)
                    else {}
                ),
                "evidence": evidence,
            }

            graph["relationships"].append(rel)
            existing.add(key)
            accepted.append(rel)
            local += 1

        reviews.append({
            "signature": f"{source_type}->{target_type}",
            "status": "reviewed",
            "allowed_relationship_types": sorted(allowed_types),
            "relationships_added": local,
        })

    return {
        "relationships_accepted": len(accepted),
        "relationships": accepted,
        "rejected": rejected,
        "signature_reviews": reviews,
    }


# ============================================================================
# GENERIC INSTANCE VERIFICATION
# ============================================================================

def verify_instances(
    client: Any,
    graph: dict[str, Any],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """
    Generic instance verification.

    This deliberately does not contain a special case for FinancialMetric or
    any other domain type.
    """
    all_reviews = []
    all_candidates = []
    calls = 0

    for node_type in ontology["node_types"]:
        typed = [
            n for n in graph["nodes"]
            if n["node_type"] == node_type
        ]

        for start in range(0, len(typed), NODE_BATCH_SIZE):
            batch = typed[start:start + NODE_BATCH_SIZE]
            if not batch:
                continue

            relevant = retrieve_for_nodes(batch, chunks)

            direct = node_evidence_chunks(batch, chunks)
            seen = {c.get("chunk_id") for c in direct}

            relevant = direct + [
                c for c in relevant
                if c.get("chunk_id") not in seen
            ]
            relevant = relevant[:10]

            if not relevant:
                continue

            user = f"""
ONTOLOGY
{schema_for_prompt(ontology)}

NODE TYPE
{node_type}

NODE INSTANCES
{node_catalog(batch)}

SOURCE CHUNKS
{bounded_context(relevant)}

For every supplied node, determine support and whether it is an independent
instance, duplicate/alias, observation, or unsupported.

Also identify explicit dependencies/relationships among these supplied nodes
when the ontology allows them.

Do not create nodes.
Do not infer relationships from co-occurrence.
""".strip()

            result = llm_json(
                client,
                CONSOLIDATION_SYSTEM,
                user,
                FEEDBACK_MAX_TOKENS,
            )

            all_reviews.extend(
                result.get("node_reviews", [])
                if isinstance(result.get("node_reviews"), list)
                else []
            )
            all_candidates.extend(
                result.get("relationship_candidates", [])
                if isinstance(result.get("relationship_candidates"), list)
                else []
            )
            calls += 1

    # Apply only source-grounded relationship candidates.
    node_map = {n["node_id"]: n for n in graph["nodes"]}
    existing = {
        (
            r["source_node_id"],
            r["relationship_type"],
            r["target_node_id"],
        )
        for r in graph["relationships"]
    }

    accepted = []
    rejected = []

    for r in all_candidates:
        if not isinstance(r, dict):
            continue

        sid = str(r.get("source_node_id") or "")
        tid = str(r.get("target_node_id") or "")
        rt = str(r.get("relationship_type") or "")

        s = node_map.get(sid)
        t = node_map.get(tid)

        if not s or not t:
            rejected.append({
                "relationship": r,
                "reason": "unknown endpoint",
            })
            continue

        if rt not in ontology["rel_allowed"].get(
            (s["node_type"], t["node_type"]),
            set(),
        ):
            rejected.append({
                "relationship": r,
                "reason": "relationship not allowed by endpoint types",
            })
            continue

        evidence = valid_evidence(r.get("evidence"), chunks)
        if not evidence:
            rejected.append({
                "relationship": r,
                "reason": "no source-grounded evidence",
            })
            continue

        key = (sid, rt, tid)
        if key in existing:
            continue

        rel = {
            "relationship_id": "rel_" + stable_hash(*key),
            "relationship_type": rt,
            "source_node_id": sid,
            "target_node_id": tid,
            "properties": (
                r.get("properties", {})
                if isinstance(r.get("properties"), dict)
                else {}
            ),
            "evidence": evidence,
        }

        graph["relationships"].append(rel)
        existing.add(key)
        accepted.append(rel)

    return {
        "calls": calls,
        "node_reviews": all_reviews,
        "relationship_candidates": len(all_candidates),
        "relationships_accepted": len(accepted),
        "relationships_rejected": rejected,
    }


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def enforce_schema_properties(
    graph: dict[str, Any],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    removed = []

    for n in graph["nodes"]:
        allowed = ontology["node_properties"].get(
            n["node_type"],
            set(),
        )

        props = n.get("properties", {})
        for key in list(props):
            if key not in allowed:
                removed.append({
                    "node_id": n["node_id"],
                    "node_type": n["node_type"],
                    "property": key,
                    "value": props[key],
                })
                del props[key]

    graph["schema_property_cleanup"] = {
        "removed_count": len(removed),
        "removed": removed,
    }

    return graph


def validate_graph(
    graph: dict[str, Any],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    node_counts = {
        t: 0 for t in ontology["node_types"]
    }

    node_by_id = {}

    errors = []
    warnings = []

    for n in graph["nodes"]:
        nid = n.get("node_id")
        nt = n.get("node_type")

        if nid in node_by_id:
            errors.append(f"duplicate node_id: {nid}")
        node_by_id[nid] = n

        if nt not in node_counts:
            errors.append(f"unknown node type: {nt}")
        else:
            node_counts[nt] += 1

        allowed = ontology["node_properties"].get(nt, set())
        bad_props = sorted(
            set(n.get("properties", {})) - allowed
        )

        if bad_props:
            errors.append(
                f"unsupported properties on {nid}: {bad_props}"
            )

        if not n.get("evidence"):
            errors.append(f"node without evidence: {nid}")

    relationship_counts = {
        rt: 0 for rt in ontology["relationship_types"]
    }

    valid_relationships = 0
    orphan_relationships = []

    for r in graph["relationships"]:
        sid = r.get("source_node_id")
        tid = r.get("target_node_id")
        rt = r.get("relationship_type")

        s = node_by_id.get(sid)
        t = node_by_id.get(tid)

        if not s or not t:
            orphan_relationships.append(r)
            continue

        allowed = ontology["rel_allowed"].get(
            (s["node_type"], t["node_type"]),
            set(),
        )

        if rt not in allowed:
            errors.append(
                f"invalid relationship signature: "
                f"{s['node_type']} -[{rt}]-> {t['node_type']}"
            )
            continue

        evidence = [
            e for e in r.get("evidence", [])
            if e.get("quote")
        ]

        if not evidence:
            errors.append(
                f"relationship without evidence: {r.get('relationship_id')}"
            )
            continue

        relationship_counts[rt] += 1
        valid_relationships += 1

    found_nodes = [
        t for t in ontology["node_types"]
        if node_counts[t] > 0
    ]

    missing_nodes = [
        t for t in ontology["node_types"]
        if node_counts[t] == 0
    ]

    found_relationships = [
        rt for rt in ontology["relationship_types"]
        if relationship_counts[rt] > 0
    ]

    missing_relationships = [
        rt for rt in ontology["relationship_types"]
        if relationship_counts[rt] == 0
    ]

    # A graph can legitimately contain disconnected nodes, but a disconnected
    # node is explicitly surfaced for review.
    degree = defaultdict(int)

    for r in graph["relationships"]:
        degree[r["source_node_id"]] += 1
        degree[r["target_node_id"]] += 1

    disconnected = [
        {
            "node_id": n["node_id"],
            "node_type": n["node_type"],
            "name": node_display_name(n),
        }
        for n in graph["nodes"]
        if degree[n["node_id"]] == 0
    ]

    if disconnected:
        warnings.append(
            f"{len(disconnected)} nodes have degree 0 after relationship recovery."
        )

    return {
        "node_type_counts": node_counts,
        "relationship_type_counts": relationship_counts,
        "found_node_types": found_nodes,
        "missing_node_types": missing_nodes,
        "found_relationship_types": found_relationships,
        "missing_relationship_types": missing_relationships,
        "node_count": len(graph["nodes"]),
        "relationship_count": len(graph["relationships"]),
        "valid_relationship_count": valid_relationships,
        "orphan_relationship_count": len(orphan_relationships),
        "orphan_relationships": orphan_relationships,
        "disconnected_node_count": len(disconnected),
        "disconnected_nodes": disconnected,
        "errors": errors,
        "warnings": warnings,
    }


def final_clean(graph: dict[str, Any]) -> dict[str, Any]:
    ids = {n["node_id"] for n in graph["nodes"]}

    relationships = []
    seen = set()

    for r in graph["relationships"]:
        sid = r.get("source_node_id")
        tid = r.get("target_node_id")
        rt = r.get("relationship_type")

        if sid not in ids or tid not in ids:
            continue

        key = (sid, rt, tid)
        if key in seen:
            continue

        seen.add(key)

        r["relationship_id"] = "rel_" + stable_hash(*key)
        relationships.append(r)

    graph["relationships"] = relationships

    for n in graph["nodes"]:
        n["id"] = n["node_id"]
        n["label"] = n["node_type"]

    for r in graph["relationships"]:
        r["source_id"] = r["source_node_id"]
        r["target_id"] = r["target_node_id"]

    return graph


# ============================================================================
# OPTIONAL DRY-RUN / CHUNK REPORT
# ============================================================================

def print_schema_report(ontology: dict[str, Any]) -> None:
    print("=" * 78)
    print("ONTOLOGY")
    print("=" * 78)
    print(f"Node types          : {len(ontology['node_types'])}")
    print(f"Relationship types   : {len(ontology['relationship_types'])}")
    print(f"Relationship sigs    : {len(ontology['rel_allowed'])}")

    for t in ontology["node_types"]:
        print(
            f"  {t}: "
            f"{len(ontology['node_properties'].get(t, set()))} properties"
        )


def print_chunk_report(chunks: list[dict[str, Any]]) -> None:
    print("=" * 78)
    print("CHUNK REPORT")
    print("=" * 78)

    by_document = defaultdict(list)
    for c in chunks:
        by_document[c.get("document_id")].append(c)

    for document_id, items in by_document.items():
        print(f"\nDOCUMENT: {document_id}")

        for c in items:
            text = re.sub(r"\s+", " ", c.get("text", "")).strip()
            print(
                f"  {c.get('chunk_id')} "
                f"pages={c.get('start_page')}-{c.get('end_page')} "
                f"chars={len(c.get('text', ''))} "
                f"preview={text[:120]}"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic ontology-driven Azure OpenAI graph extractor."
    )
    parser.add_argument(
        "--report-chunks",
        action="store_true",
        help="Print chunk boundaries and exit.",
    )
    parser.add_argument(
        "--no-llm-consolidation",
        action="store_true",
        help="Skip generic LLM duplicate/alias review.",
    )
    parser.add_argument(
        "--no-instance-verification",
        action="store_true",
        help="Skip generic instance verification.",
    )
    args = parser.parse_args()

    load_environment()

    ontology = load_ontology()
    chunks = load_chunks()

    print_schema_report(ontology)

    if args.report_chunks:
        print_chunk_report(chunks)
        return

    validate_environment()
    client = create_client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("GENERIC ONTOLOGY-DRIVEN GRAPH EXTRACTION")
    print("=" * 78)
    print(f"Documents/chunks     : {len(chunks)}")
    print(f"Node types           : {len(ontology['node_types'])}")
    print(f"Relationship types   : {len(ontology['relationship_types'])}")
    print(f"Relationship sigs    : {len(ontology['rel_allowed'])}")

    # ----------------------------------------------------------------------
    # 1. Local chunk extraction
    # ----------------------------------------------------------------------
    chunk_results, failures = extract_all_chunks(
        client,
        chunks,
        ontology,
    )

    graph = merge_chunk_results(chunk_results)

    print(
        f"\nAfter chunk merge: "
        f"{len(graph['nodes'])} nodes / "
        f"{len(graph['relationships'])} relationships"
    )

    # ----------------------------------------------------------------------
    # 2. Ontology filter + deterministic consolidation
    # ----------------------------------------------------------------------
    graph = deterministic_exact_consolidation(graph)

    # Remove invalid endpoint/signature relationships after consolidation.
    graph = final_clean(graph)

    print(
        f"After deterministic consolidation: "
        f"{len(graph['nodes'])} nodes / "
        f"{len(graph['relationships'])} relationships"
    )

    # ----------------------------------------------------------------------
    # 3. Generic LLM duplicate/alias verification
    # ----------------------------------------------------------------------
    consolidation_report = {
        "calls": 0,
        "node_reviews": [],
        "relationship_candidates": [],
    }

    if not args.no_llm_consolidation:
        print("\nRunning generic entity consolidation review...")
        consolidation_report = consolidate_with_llm(
            client,
            graph,
            chunks,
            ontology,
        )
        graph = apply_llm_consolidation(
            graph,
            consolidation_report,
            chunks,
            ontology,
        )

        print(
            "LLM consolidation merges applied: "
            f"{graph['llm_consolidation_applied']['merged_count']}"
        )

    # ----------------------------------------------------------------------
    # 4. Generic relationship recovery
    # ----------------------------------------------------------------------
    print("\nRunning ontology-driven relationship recovery...")
    relationship_report = recover_relationships(
        client,
        graph,
        chunks,
        ontology,
    )

    print(
        "Relationships recovered: "
        f"{relationship_report['relationships_accepted']}"
    )

    # ----------------------------------------------------------------------
    # 5. Generic instance verification
    # ----------------------------------------------------------------------
    instance_report = {
        "calls": 0,
        "node_reviews": [],
        "relationship_candidates": 0,
        "relationships_accepted": 0,
        "relationships_rejected": [],
    }

    if not args.no_instance_verification:
        print("\nRunning generic instance verification...")
        instance_report = verify_instances(
            client,
            graph,
            chunks,
            ontology,
        )

        print(
            "Instance-verification relationships recovered: "
            f"{instance_report['relationships_accepted']}"
        )

    # ----------------------------------------------------------------------
    # 6. Final deterministic consolidation + schema property enforcement
    # ----------------------------------------------------------------------
    graph = deterministic_exact_consolidation(graph)
    graph = enforce_schema_properties(graph, ontology)
    graph = final_clean(graph)

    # ----------------------------------------------------------------------
    # 7. Final validation
    # ----------------------------------------------------------------------
    validation = validate_graph(graph, ontology)

    graph["ontology_coverage"] = {
        "node_types": {
            "required": ontology["node_types"],
            "found": validation["found_node_types"],
            "not_found": validation["missing_node_types"],
        },
        "relationship_types": {
            "required": ontology["relationship_types"],
            "found": validation["found_relationship_types"],
            "not_found": validation["missing_relationship_types"],
        },
        "note": (
            "Ontology coverage is descriptive. A node or relationship type "
            "may legitimately be absent from a document. No placeholder "
            "nodes or edges are created to satisfy coverage."
        ),
    }

    graph["feedback"] = {
        "entity_consolidation": consolidation_report,
        "relationship_recovery": relationship_report,
        "instance_verification": instance_report,
    }

    graph["validation"] = validation

    graph["extraction_summary"] = {
        "chunks_discovered": len(chunks),
        "chunks_succeeded": len(chunk_results),
        "chunks_failed": len(failures),
        "failures": failures,
        "nodes_count": len(graph["nodes"]),
        "relationships_count": len(graph["relationships"]),
        "node_type_counts": validation["node_type_counts"],
        "relationship_type_counts": validation["relationship_type_counts"],
        "disconnected_node_count": validation["disconnected_node_count"],
    }

    graph["schema"] = {
        "node_type_count": len(ontology["node_types"]),
        "relationship_type_count": len(ontology["relationship_types"]),
        "relationship_signature_count": len(ontology["rel_allowed"]),
    }

    OUTPUT_FILE.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # 8. Console summary
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL GRAPH")
    print("=" * 78)
    print(f"Nodes               : {validation['node_count']}")
    print(f"Relationships        : {validation['relationship_count']}")
    print(
        f"Node types found    : "
        f"{len(validation['found_node_types'])}/"
        f"{len(ontology['node_types'])}"
    )
    print(
        f"Relationship types  : "
        f"{len(validation['found_relationship_types'])}/"
        f"{len(ontology['relationship_types'])}"
    )
    print(
        f"Disconnected nodes  : "
        f"{validation['disconnected_node_count']}"
    )
    print(
        f"Validation errors   : "
        f"{len(validation['errors'])}"
    )
    print(f"Output              : {OUTPUT_FILE}")

    if validation["missing_node_types"]:
        print(
            "\nNode types not found in this document:"
            f" {validation['missing_node_types']}"
        )

    if validation["missing_relationship_types"]:
        print(
            "\nRelationship types not found in this document:"
            f" {validation['missing_relationship_types']}"
        )

    if validation["errors"]:
        print("\nValidation errors:")
        for error in validation["errors"][:30]:
            print(f"  - {error}")

    if failures:
        print(
            f"\nWARNING: {len(failures)} chunks failed extraction. "
            "Review extraction_summary.failures."
        )

    print("=" * 78)

    # Fail only for structural validation errors, not because an ontology type
    # is absent from a document.
    if validation["errors"] or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
