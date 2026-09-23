"""
Generic ontology-driven graph extraction - v2.

This version is deliberately designed around the ontology files, not around
the current three documents.

INPUT
-----
graph_output/chunk_results/*_chunks.json
schema/nodes.txt
schema/relationships.txt

OUTPUT
------
graph_output/graph_extract.json
graph_output/validation_report.json
graph_output/extraction_audit.json

RUN
---
python src/graph_extractor_final_generic_v2.py

DESIGN
------
1. Parse the ontology dynamically.
2. Run an exhaustive local extraction pass on every chunk.
3. Run a second recall/completeness extraction pass on every chunk.
4. Merge local results using ontology-aware semantic keys, not LLM-generated
   temporary IDs.
5. Run generic LLM entity-resolution review.
6. Run relationship extraction separately from node extraction.
7. Run a generic ontology coverage/recovery pass for node types that have not
   yet been observed.
8. Run relationship recovery again after node recovery.
9. Run generic instance verification.
10. Enforce the schema and validate the final graph.

IMPORTANT
---------
- No document names are hard-coded.
- No entity names are hard-coded.
- No domain keyword dictionary is used.
- No metric-name list is used by the Python logic.
- No ontology type is artificially instantiated.
- Missing ontology types are reported as not found.
- A missing type is NOT treated as an error.
- Evidence is mandatory for retained nodes and relationships.
- The complete graph is never sent to the LLM.
- The complete document is never sent to the LLM in one request.
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


# =============================================================================
# PATHS / SETTINGS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schema"
NODES_FILE = SCHEMA_DIR / "nodes.txt"
RELATIONSHIPS_FILE = SCHEMA_DIR / "relationships.txt"

CHUNK_DIR = ROOT / "graph_output" / "chunk_results"
OUTPUT_DIR = ROOT / "graph_output"
GRAPH_FILE = OUTPUT_DIR / "graph_extract.json"
VALIDATION_FILE = OUTPUT_DIR / "validation_report.json"
AUDIT_FILE = OUTPUT_DIR / "extraction_audit.json"

MAX_RETRIES = 4
RETRY_BASE = 3
EXTRACTION_MAX_TOKENS = 14000
RECALL_MAX_TOKENS = 14000
RELATIONSHIP_MAX_TOKENS = 12000
VERIFY_MAX_TOKENS = 12000

# Local source context limit for one LLM request.
MAX_CONTEXT_CHARS = int(os.getenv("GRAPH_MAX_CONTEXT_CHARS", "28000"))

# Number of node instances sent to entity resolution / verification at once.
NODE_BATCH_SIZE = int(os.getenv("GRAPH_NODE_BATCH_SIZE", "16"))

# Number of source chunks used for a relationship recovery request.
REL_CHUNK_LIMIT = int(os.getenv("GRAPH_REL_CHUNK_LIMIT", "10"))

# Second extraction pass is ON by default because the previous implementation
# was recall-poor. Set GRAPH_RECALL_PASS=false only for a deliberate cheap run.
RECALL_PASS = os.getenv("GRAPH_RECALL_PASS", "true").lower() == "true"

# Coverage recovery can be disabled for a deliberately cheap run.
COVERAGE_PASS = os.getenv("GRAPH_COVERAGE_PASS", "true").lower() == "true"


# =============================================================================
# GENERIC PROMPTS
# =============================================================================

EXTRACTION_SYSTEM = r"""
You are an exhaustive knowledge-graph extraction engine.

The ontology supplied in the user message is the ONLY structural authority.

Your task is extraction, not summarization.

For the supplied source chunk:
- identify every distinct source-supported entity/fact that belongs to an
  allowed node type;
- capture the useful properties that the source explicitly provides;
- capture explicit relationships only when the source expresses the relation;
- preserve separate observations when their values, periods, scenarios,
  versions, reporting bases, or other source distinctions make them distinct;
- do not invent facts;
- do not create a node merely because an ontology type exists;
- do not create placeholder nodes;
- do not use properties outside the allowed property list;
- do not use relationship types outside the allowed signatures;
- every retained node needs evidence;
- every retained relationship needs evidence;
- evidence quotes must be short verbatim excerpts from the supplied chunk.

VERY IMPORTANT FOR RECALL:
Extract all relevant instances, not only the most prominent entity.
For example, if a table contains many distinct rows that correspond to an
allowed node type, inspect the rows individually.

Do not reduce a table to one summary node when the ontology supports multiple
instances.

Do not merge two observations merely because they have the same conceptual
name.

Return JSON only.

For each node also return:
"canonical_key": "a concise semantic identity key for this instance"

The canonical_key must distinguish materially different observations when
needed (for example different reporting periods, values, scenarios, versions,
or bases), while using the same key for repeated mentions of the same entity.
Do not put arbitrary chunk/page IDs into canonical_key.
"""


RECALL_SYSTEM = r"""
You are a graph-extraction completeness auditor.

The first extraction pass may have missed entities. Re-scan the supplied source
chunk independently.

The ontology in the prompt is the only authority.

Find source-supported instances that a normal extraction pass could overlook,
including:
- entities mentioned in tables;
- repeated but materially distinct observations;
- criteria/thresholds;
- risks and findings;
- entities embedded in narrative text;
- entities introduced in headings or notes;
- secondary entities that participate in explicit relationships.

Do not invent anything and do not create placeholders.

For every returned node provide a verbatim evidence quote.

Return ONLY newly observed candidates. It is acceptable to return an empty list
if the first pass already captured everything.

Return JSON only.
"""


RELATIONSHIP_SYSTEM = r"""
You are an exhaustive relationship extraction engine.

The ontology supplied in the prompt is the only authority.

You are given:
- a source node catalog;
- a target node catalog;
- allowed relationship types for one endpoint signature;
- source chunks.

Find every relationship explicitly supported by the source.

Do NOT infer relationships merely because:
- two nodes occur in the same chunk;
- one node's name appears near another;
- a property happens to mention another entity.

A relationship must be semantically expressed by the source.

Return every supported relationship, not just one or two examples.

Every returned relationship must use canonical node IDs from the supplied
catalog and must include a short verbatim evidence quote.

Return JSON only.
"""


ENTITY_REVIEW_SYSTEM = r"""
You are a generic entity-resolution and instance-verification engine.

The ontology is authoritative.

For every supplied node instance:
1. Decide whether the source evidence supports it.
2. Classify it as:
   - independent
   - duplicate
   - observation
   - unsupported
3. If it is a duplicate of another supplied node, identify duplicate_of.
4. Do not merge observations simply because their names match.
5. A difference in period, date, value, scenario, reporting basis, version,
   or other explicit context can make two observations legitimately distinct.
6. Identify explicit ontology-valid missing relationships among the supplied
   nodes when the source supports them.
7. Never invent a node.
8. Every proposed relationship requires evidence.

Return JSON only.
"""


COVERAGE_SYSTEM = r"""
You are an ontology coverage and recall auditor.

The supplied ontology can contain types that are absent from a document.

For the requested node type:
- inspect the supplied source chunks;
- find every explicit source-supported instance of that node type that is not
  already represented in the supplied catalog;
- do not invent placeholders;
- do not assume that the node type must exist.

For a requested relationship signature:
- inspect the source chunks;
- find every explicit source-supported relationship using the allowed types;
- use only canonical node IDs supplied by the caller.

Evidence is mandatory.

Return JSON only.
"""


# =============================================================================
# ENVIRONMENT
# =============================================================================

def load_environment() -> None:
    if load_dotenv:
        load_dotenv(ROOT / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def validate_environment() -> None:
    missing = [
        name for name in (
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT",
        )
        if not get_env(name)
    ]

    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    if AzureOpenAI is None:
        raise SystemExit(
            "The openai package is unavailable. Install/upgrade openai "
            "before running this script."
        )


def create_client() -> Any:
    return AzureOpenAI(
        azure_endpoint=get_env("AZURE_OPENAI_ENDPOINT"),
        api_key=get_env("AZURE_OPENAI_API_KEY"),
        api_version=get_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


# =============================================================================
# ONTOLOGY PARSING
# =============================================================================

def read_file(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_nodes_schema(text: str) -> tuple[list[str], dict[str, set[str]]]:
    """
    Parse numbered top-level node definitions.

    Example:
        1. Company
        Purpose: ...
        Properties:
        - name
        - industry
    """
    node_types: list[str] = []
    properties: dict[str, set[str]] = {}

    current: str | None = None
    in_properties = False

    heading_re = re.compile(
        r"^\s*(\d+)\.\s*([A-Za-z][A-Za-z0-9_]*)\s*$"
    )
    property_re = re.compile(
        r"^\s*-\s*([A-Za-z][A-Za-z0-9_]*)\s*$"
    )

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            continue

        heading = heading_re.match(line)
        if heading:
            current = heading.group(2)
            node_types.append(current)
            properties[current] = set()
            in_properties = False
            continue

        if current and re.match(
            r"^Properties\s*:\s*$",
            line,
            re.IGNORECASE,
        ):
            in_properties = True
            continue

        if current and in_properties:
            match = property_re.match(line)
            if match:
                properties[current].add(match.group(1))
            elif not line.startswith("-"):
                in_properties = False

    if not node_types:
        raise RuntimeError(
            "No numbered node types were found in schema/nodes.txt"
        )

    return node_types, properties


def parse_relationship_schema(
    text: str,
    node_types: set[str],
) -> tuple[
    dict[tuple[str, str], set[str]],
    list[dict[str, str]],
]:
    """
    Parse relationship blocks until the relationship-property convention.

    This is deliberately generic and does not contain the actual relationship
    names.
    """
    allowed: dict[tuple[str, str], set[str]] = defaultdict(set)

    current: tuple[str, str] | None = None

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.upper().startswith(
            "GENERAL RELATIONSHIP PROPERTY CONVENTION"
        ):
            break

        signature = re.match(
            r"^([A-Za-z][A-Za-z0-9_]*)\s*->\s*"
            r"([A-Za-z][A-Za-z0-9_]*)$",
            line,
        )

        if signature:
            source = signature.group(1)
            target = signature.group(2)

            if source in node_types and target in node_types:
                current = (source, target)
                allowed[current]
            else:
                current = None

            continue

        if current:
            relationship = re.match(
                r"^-\s*([A-Z][A-Z0-9_]*)$",
                line,
            )
            if relationship:
                allowed[current].add(
                    relationship.group(1)
                )

    rows = [
        {
            "source_type": source,
            "relationship_type": relationship,
            "target_type": target,
        }
        for (source, target), relationships in allowed.items()
        for relationship in sorted(relationships)
    ]

    if not rows:
        raise RuntimeError(
            "No relationship definitions were parsed from "
            "schema/relationships.txt"
        )

    return dict(allowed), rows


def load_ontology() -> dict[str, Any]:
    nodes_text = read_file(NODES_FILE)
    relationships_text = read_file(RELATIONSHIPS_FILE)

    node_types, node_properties = parse_nodes_schema(nodes_text)

    rel_allowed, rel_rows = parse_relationship_schema(
        relationships_text,
        set(node_types),
    )

    relationship_types = sorted({
        row["relationship_type"]
        for row in rel_rows
    })

    return {
        "nodes_text": nodes_text,
        "relationships_text": relationships_text,
        "node_types": node_types,
        "node_properties": node_properties,
        "rel_allowed": rel_allowed,
        "rel_rows": rel_rows,
        "relationship_types": relationship_types,
    }


def ontology_prompt(ontology: dict[str, Any]) -> str:
    lines = ["NODE TYPES"]

    for node_type in ontology["node_types"]:
        props = sorted(
            ontology["node_properties"].get(node_type, set())
        )
        lines.append(
            f"{node_type}: "
            + ", ".join(props)
        )

    lines.append("")
    lines.append("RELATIONSHIP SIGNATURES")

    for signature, relationships in sorted(
        ontology["rel_allowed"].items()
    ):
        lines.append(
            f"{signature[0]} -> {signature[1]}: "
            + ", ".join(sorted(relationships))
        )

    return "\n".join(lines)


# =============================================================================
# CHUNKS
# =============================================================================

def load_chunks() -> list[dict[str, Any]]:
    if not CHUNK_DIR.exists():
        raise SystemExit(
            f"Chunk directory not found: {CHUNK_DIR}\n"
            "Run the chunker first."
        )

    files = sorted(
        CHUNK_DIR.glob("*_chunks.json")
    )

    if not files:
        raise SystemExit(
            f"No *_chunks.json files found in {CHUNK_DIR}"
        )

    chunks = []

    for path in files:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        for raw in payload.get("chunks", []):
            chunk = dict(raw)
            chunk["_chunk_file"] = str(path)
            chunks.append(chunk)

    return chunks


def chunk_context(
    chunks: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    output = []
    used = 0

    for chunk in chunks:
        block = (
            f"\n--- "
            f"{chunk.get('document_id')} / "
            f"{chunk.get('chunk_id')} / "
            f"pages {chunk.get('start_page')}-"
            f"{chunk.get('end_page')} ---\n"
            f"{chunk.get('text', '')}\n"
        )

        if used + len(block) > max_chars:
            break

        output.append(block)
        used += len(block)

    return "".join(output)


# =============================================================================
# TEXT / EVIDENCE
# =============================================================================

def normalize_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def identity_text(value: Any) -> str:
    value = normalize_text(value).lower()
    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )
    value = re.sub(
        r"\s+",
        " ",
        value,
    )
    return value.strip()


def hash_id(*parts: Any) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    )
    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:20]


def evidence_exists(
    quote: str,
    chunks: list[dict[str, Any]],
) -> bool:
    q = normalize_text(quote).lower()

    if len(q) < 8:
        return False

    for chunk in chunks:
        text = normalize_text(
            chunk.get("text", "")
        ).lower()

        if q in text:
            return True

    return False


def clean_evidence(
    evidence: Any,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []

    result = []
    seen = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        quote = normalize_text(
            item.get("quote")
        )

        if not quote:
            continue

        if not evidence_exists(
            quote,
            chunks,
        ):
            continue

        normalized = {
            "document_id": item.get("document_id"),
            "source_file": item.get("source_file"),
            "page": item.get("page"),
            "chunk_id": item.get("chunk_id"),
            "quote": quote,
        }

        key = json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=False,
        )

        if key not in seen:
            seen.add(key)
            result.append(normalized)

    return result


def merge_evidence(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = set()
    result = []

    for item in first + second:
        key = json.dumps(
            item,
            sort_keys=True,
            ensure_ascii=False,
        )

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


# =============================================================================
# LLM
# =============================================================================

def parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
        )
        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "LLM did not return a JSON object."
            )

        result = json.loads(
            text[start:end + 1]
        )

    if not isinstance(result, dict):
        raise ValueError(
            "LLM response must be a JSON object."
        )

    return result


def call_llm(
    client: Any,
    system: str,
    user: str,
    max_tokens: int,
) -> dict[str, Any]:
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = (
                client.chat.completions.create(
                    model=get_env(
                        "AZURE_OPENAI_DEPLOYMENT"
                    ),
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format={
                        "type": "json_object"
                    },
                    messages=[
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                )
            )

            return parse_json(
                response.choices[0]
                .message
                .content or "{}"
            )

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_BASE
                    * (2 ** (attempt - 1))
                )

    raise RuntimeError(
        f"LLM call failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# =============================================================================
# NODE NORMALIZATION
# =============================================================================

def node_semantic_key(
    node: dict[str, Any],
) -> tuple[Any, ...]:
    """
    Generic merge key.

    Preferred identity is the LLM-produced canonical_key. The model is told
    to include only identity-defining context in that key, so observation
    distinctions (period/value/basis/etc.) can be preserved when they matter.

    If no canonical_key is supplied, keep the occurrence distinct. This is
    intentionally conservative: it is better for the LLM entity-resolution
    pass to merge two true aliases than for Python to destroy two observations
    before verification.
    """
    canonical_key = normalize_text(
        node.get("canonical_key")
    )

    if canonical_key:
        return (
            node["node_type"],
            "canonical_key",
            identity_text(canonical_key),
        )

    return (
        node["node_type"],
        "source_instance",
        node.get(
            "_source_instance_id",
            node.get("node_id"),
        ),
    )


def node_display_name(
    node: dict[str, Any],
) -> str:
    properties = node.get(
        "properties",
        {},
    )

    for key, value in properties.items():
        key_l = key.lower()

        if (
            value not in (
                None,
                "",
                [],
                {},
            )
            and (
                key_l == "name"
                or key_l.endswith("_name")
                or key_l in {
                    "title",
                    "label",
                }
            )
        ):
            return str(value)

    for value in properties.values():
        if isinstance(value, str) and value.strip():
            return value[:120]

    return str(
        node.get("node_id", "")
    )


def normalize_node(
    raw: dict[str, Any],
    chunk: dict[str, Any],
    ontology: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    node_type = normalize_text(
        raw.get("node_type")
    )

    if node_type not in ontology["node_types"]:
        return None

    properties = raw.get(
        "properties",
        {},
    )

    if not isinstance(properties, dict):
        properties = {}

    allowed = ontology[
        "node_properties"
    ].get(
        node_type,
        set(),
    )

    properties = {
        key: value
        for key, value in properties.items()
        if key in allowed
        and value not in (
            None,
            "",
            [],
            {},
        )
    }

    evidence = clean_evidence(
        raw.get("evidence"),
        [chunk],
    )

    if not evidence:
        return None

    source_instance_id = (
        f"{chunk.get('document_id')}|"
        f"{chunk.get('chunk_id')}|"
        f"{raw.get('node_id') or hash_id(node_type, json.dumps(properties, sort_keys=True))}"
    )

    node = {
        "node_id": (
            "node_"
            + hash_id(
                node_type,
                source_instance_id,
            )
        ),
        "canonical_key": normalize_text(raw.get("canonical_key")) or None,
        "node_type": node_type,
        "properties": properties,
        "evidence": evidence,
        "_source_instance_id": source_instance_id,
    }

    # Preserve the LLM ID only as audit metadata, never as semantic identity.
    if raw.get("node_id"):
        node["_llm_node_id"] = str(
            raw["node_id"]
        )

    return node


# =============================================================================
# CHUNK EXTRACTION
# =============================================================================

def extraction_user(
    chunk: dict[str, Any],
    ontology: dict[str, Any],
) -> str:
    return f"""
AUTHORITATIVE ONTOLOGY
=====================
{ontology_prompt(ontology)}

SOURCE METADATA
===============
document_id: {chunk.get('document_id')}
source_file: {chunk.get('source_file')}
chunk_id: {chunk.get('chunk_id')}
pages: {chunk.get('start_page')}-{chunk.get('end_page')}

SOURCE CHUNK
============
{chunk.get('text', '')}

TASK
====
Extract all source-supported nodes and explicit relationships.

For node extraction:
- inspect narrative text, headings, tables, notes, and lists;
- return distinct instances;
- preserve distinct observations when the source distinguishes them;
- use only allowed properties.

For relationships:
- use only canonical node IDs returned in THIS response;
- use only allowed relationship types;
- only create an edge when the source explicitly supports it.

Return:
{{
  "nodes": [...],
  "relationships": [...],
  "warnings": []
}}
""".strip()


def recall_user(
    chunk: dict[str, Any],
    ontology: dict[str, Any],
    existing_catalog: str,
) -> str:
    return f"""
AUTHORITATIVE ONTOLOGY
=====================
{ontology_prompt(ontology)}

SOURCE CHUNK
============
{chunk.get('text', '')}

ALREADY CAPTURED FROM THIS CHUNK
=================================
{existing_catalog or '(none)'}

COMPLETENESS TASK
=================
Independently rescan the source for source-supported node instances that may
have been missed.

Pay particular attention to:
- table rows;
- separate values/periods/bases;
- explicit criteria and thresholds;
- explicit risks;
- audit/accounting/control findings;
- named people, organizations, markets, deals, and other entities;
- entities appearing only in notes or conclusions.

Return ONLY additional nodes not already represented above.

Every returned node must have evidence from this chunk and a semantic
canonical_key using the same identity rules as the first extraction pass.

Return:
{{
  "nodes": [...],
  "warnings": []
}}
""".strip()


def normalize_relationship(
    raw: dict[str, Any],
    nodes: list[dict[str, Any]],
    chunk: dict[str, Any],
    ontology: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    relationship_type = normalize_text(
        raw.get("relationship_type")
    )
    source_id = normalize_text(
        raw.get("source_node_id")
    )
    target_id = normalize_text(
        raw.get("target_node_id")
    )

    node_map = {
        n.get("_llm_node_id"): n
        for n in nodes
        if n.get("_llm_node_id")
    }

    source = node_map.get(
        source_id
    )
    target = node_map.get(
        target_id
    )

    if not source or not target:
        return None

    allowed = ontology[
        "rel_allowed"
    ].get(
        (
            source["node_type"],
            target["node_type"],
        ),
        set(),
    )

    if relationship_type not in allowed:
        return None

    evidence = clean_evidence(
        raw.get("evidence"),
        [chunk],
    )

    if not evidence:
        return None

    properties = raw.get(
        "properties",
        {},
    )

    if not isinstance(properties, dict):
        properties = {}

    return {
        "relationship_id": (
            "rel_"
            + hash_id(
                source["node_id"],
                relationship_type,
                target["node_id"],
            )
        ),
        "relationship_type": relationship_type,
        "source_node_id": source["node_id"],
        "target_node_id": target["node_id"],
        "properties": properties,
        "evidence": evidence,
    }


def extract_chunk_once(
    client: Any,
    chunk: dict[str, Any],
    ontology: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    result = call_llm(
        client,
        system_prompt,
        user_prompt,
        max_tokens,
    )

    nodes = []

    raw_nodes = result.get(
        "nodes",
        [],
    )

    if isinstance(raw_nodes, list):
        for raw in raw_nodes:
            node = normalize_node(
                raw,
                chunk,
                ontology,
            )

            if node:
                nodes.append(node)

    relationships = []

    raw_relationships = result.get(
        "relationships",
        [],
    )

    if isinstance(raw_relationships, list):
        for raw in raw_relationships:
            relationship = normalize_relationship(
                raw,
                nodes,
                chunk,
                ontology,
            )

            if relationship:
                relationships.append(
                    relationship
                )

    return {
        "nodes": nodes,
        "relationships": relationships,
        "warnings": (
            result.get("warnings", [])
            if isinstance(
                result.get("warnings", []),
                list,
            )
            else []
        ),
    }


def node_catalog_short(
    nodes: list[dict[str, Any]],
) -> str:
    return "\n".join(
        f"{n['node_id']} | "
        f"{n['node_type']} | "
        f"{json.dumps(n.get('properties', {}), ensure_ascii=False)}"
        for n in nodes
    )


def extract_all_chunks(
    client: Any,
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    results = []
    failures = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"[{index}/{len(chunks)}] "
            f"{chunk.get('document_id')} / "
            f"{chunk.get('chunk_id')}"
        )

        try:
            primary = extract_chunk_once(
                client,
                chunk,
                ontology,
                EXTRACTION_SYSTEM,
                extraction_user(
                    chunk,
                    ontology,
                ),
                EXTRACTION_MAX_TOKENS,
            )

            all_nodes = list(
                primary["nodes"]
            )
            all_relationships = list(
                primary["relationships"]
            )

            print(
                f"  primary: "
                f"nodes={len(all_nodes)}, "
                f"relationships={len(all_relationships)}"
            )

            if RECALL_PASS:
                catalog = node_catalog_short(
                    all_nodes
                )

                recall_result = call_llm(
                    client,
                    RECALL_SYSTEM,
                    recall_user(
                        chunk,
                        ontology,
                        catalog,
                    ),
                    RECALL_MAX_TOKENS,
                )

                recall_nodes = []

                for raw in recall_result.get(
                    "nodes",
                    []
                ) if isinstance(
                    recall_result.get("nodes"),
                    list,
                ) else []:
                    node = normalize_node(
                        raw,
                        chunk,
                        ontology,
                    )

                    if node:
                        recall_nodes.append(
                            node
                        )

                all_nodes.extend(
                    recall_nodes
                )

                print(
                    f"  recall: "
                    f"nodes_added={len(recall_nodes)}"
                )

            results.append({
                "document_id": chunk.get(
                    "document_id"
                ),
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "nodes": all_nodes,
                "relationships": all_relationships,
            })

        except Exception as exc:
            failures.append({
                "document_id": chunk.get(
                    "document_id"
                ),
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "error": str(exc),
            })

            print(
                f"  FAILED: {exc}"
            )

    return results, failures


# =============================================================================
# GLOBAL NODE MERGE
# =============================================================================

def merge_properties(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    result = dict(first)

    for key, value in second.items():
        if key not in result:
            result[key] = value
            continue

        if result[key] in (
            None,
            "",
            [],
            {},
        ):
            result[key] = value

    return result


def merge_nodes(
    chunk_results: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
]:
    registry: dict[
        tuple[Any, ...],
        dict[str, Any],
    ] = {}

    aliases: dict[str, str] = {}

    for result in chunk_results:
        for node in result["nodes"]:
            key = node_semantic_key(
                node
            )

            # A source_instance key is intentionally unique to the occurrence.
            if key not in registry:
                registry[key] = dict(node)
                registry[key]["properties"] = dict(
                    node.get("properties", {})
                )
                registry[key]["evidence"] = list(
                    node.get("evidence", [])
                )
            else:
                canonical = registry[key]

                canonical["properties"] = merge_properties(
                    canonical.get("properties", {}),
                    node.get("properties", {}),
                )

                canonical["evidence"] = merge_evidence(
                    canonical.get("evidence", []),
                    node.get("evidence", []),
                )

            aliases[
                node["node_id"]
            ] = registry[key]["node_id"]

    nodes = list(
        registry.values()
    )

    return nodes, aliases


def remap_relationships(
    relationships: list[dict[str, Any]],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    registry = {}

    for relationship in relationships:
        source = aliases.get(
            relationship.get(
                "source_node_id"
            )
        )

        target = aliases.get(
            relationship.get(
                "target_node_id"
            )
        )

        relationship_type = relationship.get(
            "relationship_type"
        )

        if not source or not target:
            continue

        key = (
            source,
            relationship_type,
            target,
        )

        if key not in registry:
            registry[key] = dict(
                relationship
            )
            registry[key][
                "source_node_id"
            ] = source
            registry[key][
                "target_node_id"
            ] = target
            registry[key][
                "relationship_id"
            ] = (
                "rel_"
                + hash_id(*key)
            )
        else:
            registry[key]["properties"] = merge_properties(
                registry[key].get(
                    "properties",
                    {},
                ),
                relationship.get(
                    "properties",
                    {},
                ),
            )

            registry[key]["evidence"] = merge_evidence(
                registry[key].get(
                    "evidence",
                    [],
                ),
                relationship.get(
                    "evidence",
                    [],
                ),
            )

    return list(
        registry.values()
    )


# =============================================================================
# RETRIEVAL
# =============================================================================

def direct_evidence_chunks(
    nodes: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = {
        evidence.get("chunk_id")
        for node in nodes
        for evidence in node.get(
            "evidence",
            [],
        )
        if evidence.get("chunk_id")
    }

    return [
        chunk
        for chunk in chunks
        if chunk.get("chunk_id") in ids
    ]


def retrieve_chunks_for_nodes(
    nodes: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Generic lexical retrieval from actual node values.
    There is no domain vocabulary list.
    """
    direct = direct_evidence_chunks(
        nodes,
        chunks,
    )

    terms = []

    for node in nodes:
        for value in node.get(
            "properties",
            {},
        ).values():
            if isinstance(value, str):
                value = normalize_text(
                    value
                )

                if len(value) >= 4:
                    terms.append(
                        value[:150]
                    )

    scored = []

    for chunk in chunks:
        text = normalize_text(
            chunk.get("text", "")
        ).lower()

        score = sum(
            1
            for term in terms
            if term.lower() in text
        )

        if score:
            scored.append(
                (
                    score,
                    len(text),
                    chunk,
                )
            )

    scored.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    output = []
    seen = set()

    for chunk in direct:
        cid = chunk.get(
            "chunk_id"
        )

        if cid not in seen:
            seen.add(cid)
            output.append(chunk)

    for _, _, chunk in scored:
        if len(output) >= limit:
            break

        cid = chunk.get(
            "chunk_id"
        )

        if cid not in seen:
            seen.add(cid)
            output.append(chunk)

    return output[:limit]


# =============================================================================
# GENERIC ENTITY RESOLUTION
# =============================================================================

def entity_resolution(
    client: Any,
    graph_nodes: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Generic LLM review.

    The LLM only sees a bounded batch of nodes and their source evidence.
    """
    reviews = []
    merges = []
    rejected_merges = []
    calls = 0

    for node_type in ontology["node_types"]:
        typed = [
            node
            for node in graph_nodes
            if node["node_type"] == node_type
        ]

        for start in range(
            0,
            len(typed),
            NODE_BATCH_SIZE,
        ):
            batch = typed[
                start:start + NODE_BATCH_SIZE
            ]

            if not batch:
                continue

            relevant = retrieve_chunks_for_nodes(
                batch,
                chunks,
                limit=10,
            )

            if not relevant:
                continue

            user = f"""
ONTOLOGY
========
{ontology_prompt(ontology)}

NODE TYPE UNDER REVIEW
======================
{node_type}

CANONICAL NODE INSTANCES
========================
{node_catalog_short(batch)}

SOURCE EVIDENCE
===============
{chunk_context(relevant)}

Review every supplied node.

For each node decide:
- supported?
- independent, duplicate, observation, or unsupported?
- if duplicate, duplicate_of another supplied canonical node ID.

Also return explicit missing relationships among these supplied nodes when the
source supports them and the ontology permits them.
""".strip()

            result = call_llm(
                client,
                ENTITY_REVIEW_SYSTEM,
                user,
                VERIFY_MAX_TOKENS,
            )

            reviews.extend(
                result.get(
                    "node_reviews",
                    [],
                )
                if isinstance(
                    result.get(
                        "node_reviews"
                    ),
                    list,
                )
                else []
            )

            calls += 1

    node_map = {
        node["node_id"]: node
        for node in graph_nodes
    }

    # Apply only explicit duplicate decisions.
    aliases = {}

    for review in reviews:
        if not isinstance(
            review,
            dict,
        ):
            continue

        node_id = review.get(
            "node_id"
        )
        duplicate_of = review.get(
            "duplicate_of"
        )

        if (
            review.get(
                "classification"
            ) != "duplicate"
        ):
            continue

        if node_id not in node_map:
            continue

        if duplicate_of not in node_map:
            rejected_merges.append({
                "node_id": node_id,
                "duplicate_of": duplicate_of,
                "reason": "duplicate target does not exist",
            })
            continue

        if (
            node_map[node_id]["node_type"]
            != node_map[duplicate_of]["node_type"]
        ):
            rejected_merges.append({
                "node_id": node_id,
                "duplicate_of": duplicate_of,
                "reason": "node types differ",
            })
            continue

        aliases[node_id] = duplicate_of

    if aliases:
        for old_id, new_id in aliases.items():
            node_map[new_id]["properties"] = merge_properties(
                node_map[new_id].get(
                    "properties",
                    {},
                ),
                node_map[old_id].get(
                    "properties",
                    {},
                ),
            )

            node_map[new_id]["evidence"] = merge_evidence(
                node_map[new_id].get(
                    "evidence",
                    [],
                ),
                node_map[old_id].get(
                    "evidence",
                    [],
                ),
            )

        graph_nodes = [
            node
            for node in graph_nodes
            if node["node_id"] not in aliases
        ]

    applied = [
        {
            "from": old_id,
            "to": new_id,
        }
        for old_id, new_id in aliases.items()
    ]

    return graph_nodes, {
        "calls": calls,
        "node_reviews": reviews,
        "merges_applied": applied,
        "merges_rejected": rejected_merges,
    }


# =============================================================================
# COVERAGE / NODE RECOVERY
# =============================================================================

def coverage_recovery(
    client: Any,
    graph_nodes: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Generic recovery for node types that are currently absent.

    No keyword dictionary is used.

    For each missing node type, all available source chunks are reviewed in
    bounded batches. This is intentionally more recall-oriented than lexical
    retrieval because the previous implementation missed whole node classes.
    """
    added = []
    reports = []

    existing_types = {
        node["node_type"]
        for node in graph_nodes
    }

    missing_types = [
        node_type
        for node_type in ontology["node_types"]
        if node_type not in existing_types
    ]

    for node_type in missing_types:
        type_added = 0

        # Bounded batches over the entire document set.
        for start in range(
            0,
            len(chunks),
            max(
                1,
                min(
                    6,
                    len(chunks),
                ),
            ),
        ):
            batch_chunks = chunks[
                start:start + 6
            ]

            if not batch_chunks:
                continue

            existing_catalog = "\n".join(
                f"{node['node_id']} | "
                f"{node['node_type']} | "
                f"{node_display_name(node)}"
                for node in graph_nodes
                if node["node_type"] == node_type
            )

            user = f"""
AUTHORITATIVE ONTOLOGY
======================
{ontology_prompt(ontology)}

REQUESTED NODE TYPE
===================
{node_type}

ALREADY KNOWN INSTANCES OF THIS TYPE
=====================================
{existing_catalog or '(none)'}

SOURCE CHUNKS
=============
{chunk_context(batch_chunks)}

Find every explicit source-supported instance of {node_type} in these chunks
that is not already represented.

Do not create placeholders.
Do not infer.
Return an empty list when none exists.

Return:
{{
  "nodes": [],
  "warnings": []
}}
""".strip()

            result = call_llm(
                client,
                COVERAGE_SYSTEM,
                user,
                RECALL_MAX_TOKENS,
            )

            for raw in (
                result.get("nodes", [])
                if isinstance(
                    result.get("nodes"),
                    list,
                )
                else []
            ):
                # Determine the actual source chunk by testing evidence against
                # the current batch.
                candidate = None

                for chunk in batch_chunks:
                    candidate = normalize_node(
                        raw,
                        chunk,
                        ontology,
                    )

                    if candidate:
                        break

                if not candidate:
                    continue

                # Avoid exact semantic duplicates.
                key = node_semantic_key(
                    candidate
                )

                if any(
                    node_semantic_key(
                        node
                    ) == key
                    for node in graph_nodes
                ):
                    continue

                graph_nodes.append(
                    candidate
                )
                added.append(candidate)
                type_added += 1

        reports.append({
            "node_type": node_type,
            "nodes_added": type_added,
        })

    return graph_nodes, {
        "missing_types_before": missing_types,
        "nodes_added": len(added),
        "by_type": reports,
    }


# =============================================================================
# RELATIONSHIP EXTRACTION
# =============================================================================

def relationship_node_catalog(
    nodes: list[dict[str, Any]],
) -> str:
    return "\n".join(
        f"{node['node_id']} | "
        f"{node['node_type']} | "
        f"{json.dumps(node.get('properties', {}), ensure_ascii=False)}"
        for node in nodes
    )


def relationship_retrieval(
    source_nodes: list[dict[str, Any]],
    target_nodes: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    nodes = source_nodes + target_nodes

    direct = direct_evidence_chunks(
        nodes,
        chunks,
    )

    terms = []

    for node in nodes:
        for value in node.get(
            "properties",
            {},
        ).values():
            if isinstance(value, str):
                value = normalize_text(
                    value
                )

                if len(value) >= 4:
                    terms.append(
                        value[:150]
                    )

    scored = []

    for chunk in chunks:
        text = normalize_text(
            chunk.get("text", "")
        ).lower()

        score = sum(
            1
            for term in terms
            if term.lower() in text
        )

        if score:
            scored.append(
                (
                    score,
                    len(text),
                    chunk,
                )
            )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    result = []
    seen = set()

    for chunk in direct:
        cid = chunk.get(
            "chunk_id"
        )

        if cid not in seen:
            seen.add(cid)
            result.append(chunk)

    for _, _, chunk in scored:
        if len(result) >= REL_CHUNK_LIMIT:
            break

        cid = chunk.get(
            "chunk_id"
        )

        if cid not in seen:
            seen.add(cid)
            result.append(chunk)

    return result[:REL_CHUNK_LIMIT]


def recover_relationships(
    client: Any,
    graph_nodes: list[dict[str, Any]],
    graph_relationships: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    node_by_type = defaultdict(list)

    for node in graph_nodes:
        node_by_type[
            node["node_type"]
        ].append(node)

    existing = {
        (
            relationship[
                "source_node_id"
            ],
            relationship[
                "relationship_type"
            ],
            relationship[
                "target_node_id"
            ],
        )
        for relationship in graph_relationships
    }

    added = []
    reviews = []

    for signature, allowed_types in sorted(
        ontology["rel_allowed"].items()
    ):
        source_type, target_type = signature

        sources = node_by_type.get(
            source_type,
            [],
        )

        targets = node_by_type.get(
            target_type,
            [],
        )

        if not sources or not targets:
            reviews.append({
                "signature": (
                    f"{source_type}->{target_type}"
                ),
                "status": "not_applicable",
                "reason": "endpoint type absent",
            })
            continue

        # Batch endpoint nodes for scalability.
        source_batches = [
            sources[i:i + NODE_BATCH_SIZE]
            for i in range(
                0,
                len(sources),
                NODE_BATCH_SIZE,
            )
        ]

        target_batches = [
            targets[i:i + NODE_BATCH_SIZE]
            for i in range(
                0,
                len(targets),
                NODE_BATCH_SIZE,
            )
        ]

        signature_added = 0

        for source_batch in source_batches:
            for target_batch in target_batches:
                relevant = relationship_retrieval(
                    source_batch,
                    target_batch,
                    chunks,
                )

                if not relevant:
                    continue

                user = f"""
ONTOLOGY
========
{ontology_prompt(ontology)}

ENDPOINT SIGNATURE
==================
{source_type} -> {target_type}

ALLOWED RELATIONSHIP TYPES
==========================
{json.dumps(sorted(allowed_types))}

SOURCE NODES
============
{relationship_node_catalog(source_batch)}

TARGET NODES
============
{relationship_node_catalog(target_batch)}

SOURCE CHUNKS
=============
{chunk_context(relevant)}

Extract EVERY explicit relationship supported by these chunks for this
signature. Do not infer from co-occurrence.
""".strip()

                result = call_llm(
                    client,
                    RELATIONSHIP_SYSTEM,
                    user,
                    RELATIONSHIP_MAX_TOKENS,
                )

                raw_relationships = (
                    result.get(
                        "relationships",
                        [],
                    )
                    if isinstance(
                        result.get(
                            "relationships"
                        ),
                        list,
                    )
                    else []
                )

                node_map = {
                    node["node_id"]: node
                    for node in (
                        source_batch
                        + target_batch
                    )
                }

                for raw in raw_relationships:
                    if not isinstance(
                        raw,
                        dict,
                    ):
                        continue

                    source_id = normalize_text(
                        raw.get(
                            "source_node_id"
                        )
                    )
                    target_id = normalize_text(
                        raw.get(
                            "target_node_id"
                        )
                    )
                    relationship_type = normalize_text(
                        raw.get(
                            "relationship_type"
                        )
                    )

                    source = node_map.get(
                        source_id
                    )
                    target = node_map.get(
                        target_id
                    )

                    if not source or not target:
                        continue

                    if (
                        source["node_type"],
                        target["node_type"],
                    ) != signature:
                        continue

                    if relationship_type not in allowed_types:
                        continue

                    evidence = clean_evidence(
                        raw.get(
                            "evidence"
                        ),
                        relevant,
                    )

                    if not evidence:
                        continue

                    key = (
                        source_id,
                        relationship_type,
                        target_id,
                    )

                    if key in existing:
                        continue

                    relationship = {
                        "relationship_id": (
                            "rel_"
                            + hash_id(*key)
                        ),
                        "relationship_type": relationship_type,
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                        "properties": (
                            raw.get(
                                "properties",
                                {},
                            )
                            if isinstance(
                                raw.get(
                                    "properties",
                                    {},
                                ),
                                dict,
                            )
                            else {}
                        ),
                        "evidence": evidence,
                    }

                    graph_relationships.append(
                        relationship
                    )

                    existing.add(
                        key
                    )

                    added.append(
                        relationship
                    )

                    signature_added += 1

        reviews.append({
            "signature": (
                f"{source_type}->{target_type}"
            ),
            "status": "reviewed",
            "allowed_types": sorted(
                allowed_types
            ),
            "relationships_added": signature_added,
        })

    return graph_relationships, {
        "relationships_added": len(added),
        "relationships": added,
        "signature_reviews": reviews,
    }


# =============================================================================
# INSTANCE VERIFICATION
# =============================================================================

def verify_instances(
    client: Any,
    graph_nodes: list[dict[str, Any]],
    graph_relationships: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    reviews = []
    candidates = []
    calls = 0

    for node_type in ontology["node_types"]:
        typed = [
            node
            for node in graph_nodes
            if node["node_type"] == node_type
        ]

        for start in range(
            0,
            len(typed),
            NODE_BATCH_SIZE,
        ):
            batch = typed[
                start:start + NODE_BATCH_SIZE
            ]

            if not batch:
                continue

            relevant = retrieve_chunks_for_nodes(
                batch,
                chunks,
                limit=10,
            )

            if not relevant:
                continue

            user = f"""
ONTOLOGY
========
{ontology_prompt(ontology)}

NODE TYPE
=========
{node_type}

CANONICAL NODES
===============
{node_catalog_short(batch)}

SOURCE CHUNKS
=============
{chunk_context(relevant)}

For EVERY node:
- determine supported/unsupported;
- classify independent/duplicate/observation/unsupported;
- if duplicate, provide duplicate_of using another supplied canonical node ID.

Also identify explicit missing relationships among the supplied nodes when
allowed by the ontology.

Do not invent relationships.
""".strip()

            result = call_llm(
                client,
                ENTITY_REVIEW_SYSTEM,
                user,
                VERIFY_MAX_TOKENS,
            )

            reviews.extend(
                result.get(
                    "node_reviews",
                    [],
                )
                if isinstance(
                    result.get(
                        "node_reviews"
                    ),
                    list,
                )
                else []
            )

            candidates.extend(
                result.get(
                    "relationship_candidates",
                    [],
                )
                if isinstance(
                    result.get(
                        "relationship_candidates"
                    ),
                    list,
                )
                else []
            )

            calls += 1

    node_map = {
        node["node_id"]: node
        for node in graph_nodes
    }

    existing = {
        (
            relationship[
                "source_node_id"
            ],
            relationship[
                "relationship_type"
            ],
            relationship[
                "target_node_id"
            ],
        )
        for relationship in graph_relationships
    }

    accepted = []
    rejected = []

    for raw in candidates:
        if not isinstance(
            raw,
            dict,
        ):
            continue

        source_id = normalize_text(
            raw.get(
                "source_node_id"
            )
        )
        target_id = normalize_text(
            raw.get(
                "target_node_id"
            )
        )
        relationship_type = normalize_text(
            raw.get(
                "relationship_type"
            )
        )

        source = node_map.get(
            source_id
        )
        target = node_map.get(
            target_id
        )

        if not source or not target:
            rejected.append({
                "relationship": raw,
                "reason": "unknown endpoint",
            })
            continue

        allowed = ontology[
            "rel_allowed"
        ].get(
            (
                source["node_type"],
                target["node_type"],
            ),
            set(),
        )

        if relationship_type not in allowed:
            rejected.append({
                "relationship": raw,
                "reason": "ontology mismatch",
            })
            continue

        evidence = clean_evidence(
            raw.get(
                "evidence"
            ),
            chunks,
        )

        if not evidence:
            rejected.append({
                "relationship": raw,
                "reason": "missing source evidence",
            })
            continue

        key = (
            source_id,
            relationship_type,
            target_id,
        )

        if key in existing:
            continue

        relationship = {
            "relationship_id": (
                "rel_"
                + hash_id(*key)
            ),
            "relationship_type": relationship_type,
            "source_node_id": source_id,
            "target_node_id": target_id,
            "properties": (
                raw.get(
                    "properties",
                    {},
                )
                if isinstance(
                    raw.get(
                        "properties",
                        {},
                    ),
                    dict,
                )
                else {}
            ),
            "evidence": evidence,
        }

        graph_relationships.append(
            relationship
        )
        existing.add(key)
        accepted.append(
            relationship
        )

    return {
        "calls": calls,
        "node_reviews": reviews,
        "relationship_candidates": len(
            candidates
        ),
        "relationships_accepted": len(
            accepted
        ),
        "relationships_rejected": rejected,
    }


# =============================================================================
# SCHEMA ENFORCEMENT / VALIDATION
# =============================================================================

def enforce_properties(
    graph_nodes: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> list[dict[str, Any]]:
    removed = []

    for node in graph_nodes:
        allowed = ontology[
            "node_properties"
        ].get(
            node["node_type"],
            set(),
        )

        for key in list(
            node.get(
                "properties",
                {},
            )
        ):
            if key not in allowed:
                removed.append({
                    "node_id": node["node_id"],
                    "node_type": node["node_type"],
                    "property": key,
                })

                del node[
                    "properties"
                ][key]

    return removed


def validate_graph(
    graph_nodes: list[dict[str, Any]],
    graph_relationships: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    node_counts = {
        node_type: 0
        for node_type in ontology[
            "node_types"
        ]
    }

    relationship_counts = {
        relationship_type: 0
        for relationship_type in ontology[
            "relationship_types"
        ]
    }

    errors = []
    warnings = []

    node_map = {}

    for node in graph_nodes:
        node_id = node.get(
            "node_id"
        )
        node_type = node.get(
            "node_type"
        )

        if node_id in node_map:
            errors.append(
                f"duplicate node_id: {node_id}"
            )

        node_map[node_id] = node

        if node_type not in node_counts:
            errors.append(
                f"unknown node type: {node_type}"
            )
        else:
            node_counts[node_type] += 1

        allowed = ontology[
            "node_properties"
        ].get(
            node_type,
            set(),
        )

        invalid_properties = sorted(
            set(
                node.get(
                    "properties",
                    {},
                )
            ) - allowed
        )

        if invalid_properties:
            errors.append(
                f"unsupported properties on "
                f"{node_id}: "
                f"{invalid_properties}"
            )

        if not node.get(
            "evidence"
        ):
            errors.append(
                f"node without evidence: "
                f"{node_id}"
            )

    seen_relationships = set()

    for relationship in graph_relationships:
        source_id = relationship.get(
            "source_node_id"
        )
        target_id = relationship.get(
            "target_node_id"
        )
        relationship_type = relationship.get(
            "relationship_type"
        )

        source = node_map.get(
            source_id
        )
        target = node_map.get(
            target_id
        )

        if not source or not target:
            errors.append(
                f"orphan relationship: "
                f"{relationship.get('relationship_id')}"
            )
            continue

        signature = (
            source["node_type"],
            target["node_type"],
        )

        if relationship_type not in ontology[
            "rel_allowed"
        ].get(
            signature,
            set(),
        ):
            errors.append(
                f"invalid relationship: "
                f"{source['node_type']} "
                f"-[{relationship_type}]-> "
                f"{target['node_type']}"
            )
            continue

        key = (
            source_id,
            relationship_type,
            target_id,
        )

        if key in seen_relationships:
            errors.append(
                f"duplicate relationship: {key}"
            )

        seen_relationships.add(key)

        if not relationship.get(
            "evidence"
        ):
            errors.append(
                f"relationship without evidence: "
                f"{relationship.get('relationship_id')}"
            )
        else:
            relationship_counts[
                relationship_type
            ] += 1

    found_node_types = [
        node_type
        for node_type in ontology[
            "node_types"
        ]
        if node_counts[node_type] > 0
    ]

    missing_node_types = [
        node_type
        for node_type in ontology[
            "node_types"
        ]
        if node_counts[node_type] == 0
    ]

    found_relationship_types = [
        relationship_type
        for relationship_type in ontology[
            "relationship_types"
        ]
        if relationship_counts[
            relationship_type
        ] > 0
    ]

    missing_relationship_types = [
        relationship_type
        for relationship_type in ontology[
            "relationship_types"
        ]
        if relationship_counts[
            relationship_type
        ] == 0
    ]

    degree = defaultdict(int)

    for relationship in graph_relationships:
        degree[
            relationship[
                "source_node_id"
            ]
        ] += 1

        degree[
            relationship[
                "target_node_id"
            ]
        ] += 1

    disconnected = [
        {
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "name": node_display_name(node),
        }
        for node in graph_nodes
        if degree[node["node_id"]] == 0
    ]

    if disconnected:
        warnings.append(
            f"{len(disconnected)} nodes have no relationships."
        )

    return {
        "valid": not errors,
        "node_count": len(
            graph_nodes
        ),
        "relationship_count": len(
            graph_relationships
        ),
        "node_type_counts": node_counts,
        "relationship_type_counts": relationship_counts,
        "found_node_types": found_node_types,
        "missing_node_types": missing_node_types,
        "found_relationship_types": found_relationship_types,
        "missing_relationship_types": missing_relationship_types,
        "disconnected_node_count": len(
            disconnected
        ),
        "disconnected_nodes": disconnected,
        "errors": errors,
        "warnings": warnings,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def serialize_graph(
    graph_nodes: list[dict[str, Any]],
    graph_relationships: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []

    for node in graph_nodes:
        clean = {
            "id": node["node_id"],
            "label": node["node_type"],
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "properties": node.get(
                "properties",
                {},
            ),
            "evidence": node.get(
                "evidence",
                [],
            ),
        }

        nodes.append(clean)

    relationships = []

    for relationship in graph_relationships:
        clean = {
            "id": relationship[
                "relationship_id"
            ],
            "relationship_id": relationship[
                "relationship_id"
            ],
            "relationship_type": relationship[
                "relationship_type"
            ],
            "source_id": relationship[
                "source_node_id"
            ],
            "target_id": relationship[
                "target_node_id"
            ],
            "source_node_id": relationship[
                "source_node_id"
            ],
            "target_node_id": relationship[
                "target_node_id"
            ],
            "properties": relationship.get(
                "properties",
                {},
            ),
            "evidence": relationship.get(
                "evidence",
                [],
            ),
        }

        relationships.append(clean)

    return nodes, relationships


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generic ontology-driven graph extraction v2"
        )
    )

    parser.add_argument(
        "--report-chunks",
        action="store_true",
    )

    args = parser.parse_args()

    load_environment()

    ontology = load_ontology()
    chunks = load_chunks()

    print("=" * 78)
    print("GENERIC ONTOLOGY GRAPH EXTRACTOR V2")
    print("=" * 78)
    print(
        f"Node types         : "
        f"{len(ontology['node_types'])}"
    )
    print(
        f"Relationship types : "
        f"{len(ontology['relationship_types'])}"
    )
    print(
        f"Signatures         : "
        f"{len(ontology['rel_allowed'])}"
    )
    print(
        f"Chunks             : "
        f"{len(chunks)}"
    )

    if args.report_chunks:
        for chunk in chunks:
            print(
                f"{chunk.get('document_id')} / "
                f"{chunk.get('chunk_id')} / "
                f"pages "
                f"{chunk.get('start_page')}-"
                f"{chunk.get('end_page')} / "
                f"chars={len(chunk.get('text', ''))}"
            )
        return 0

    validate_environment()
    client = create_client()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # PASS 1 + PASS 2: LOCAL EXTRACTION
    # -------------------------------------------------------------------------

    chunk_results, failures = extract_all_chunks(
        client,
        chunks,
        ontology,
    )

    graph_nodes, aliases = merge_nodes(
        chunk_results
    )

    all_relationships = []

    for result in chunk_results:
        all_relationships.extend(
            result["relationships"]
        )

    graph_relationships = remap_relationships(
        all_relationships,
        aliases,
    )

    print(
        f"\nAfter local extraction/merge: "
        f"{len(graph_nodes)} nodes / "
        f"{len(graph_relationships)} relationships"
    )

    # -------------------------------------------------------------------------
    # PASS 3: GENERIC ENTITY RESOLUTION
    # -------------------------------------------------------------------------

    graph_nodes, entity_report = entity_resolution(
        client,
        graph_nodes,
        chunks,
        ontology,
    )

    # Re-run relationship remapping is not necessary here because the entity
    # resolver only removes duplicate nodes after explicitly merging them.
    # Rebuild existing relationship endpoints from the applied merge map.
    merge_map = {
        item["from"]: item["to"]
        for item in entity_report[
            "merges_applied"
        ]
    }

    if merge_map:
        graph_relationships = remap_relationships(
            graph_relationships,
            merge_map,
        )

    print(
        f"After entity resolution: "
        f"{len(graph_nodes)} nodes / "
        f"{len(graph_relationships)} relationships"
    )

    # -------------------------------------------------------------------------
    # PASS 4: NODE COVERAGE RECOVERY
    # -------------------------------------------------------------------------

    coverage_report = {
        "enabled": COVERAGE_PASS,
        "missing_types_before": [],
        "nodes_added": 0,
        "by_type": [],
    }

    if COVERAGE_PASS:
        graph_nodes, coverage_report = coverage_recovery(
            client,
            graph_nodes,
            chunks,
            ontology,
        )

        print(
            f"Coverage recovery added: "
            f"{coverage_report['nodes_added']} nodes"
        )

    # -------------------------------------------------------------------------
    # PASS 5: RELATIONSHIP RECOVERY
    # -------------------------------------------------------------------------

    print(
        "\nRunning ontology-driven relationship extraction..."
    )

    graph_relationships, relationship_report = recover_relationships(
        client,
        graph_nodes,
        graph_relationships,
        chunks,
        ontology,
    )

    print(
        f"Relationship recovery added: "
        f"{relationship_report['relationships_added']}"
    )

    # -------------------------------------------------------------------------
    # PASS 6: INSTANCE VERIFICATION
    # -------------------------------------------------------------------------

    print(
        "\nRunning generic instance verification..."
    )

    instance_report = verify_instances(
        client,
        graph_nodes,
        graph_relationships,
        chunks,
        ontology,
    )

    print(
        f"Instance verification added: "
        f"{instance_report['relationships_accepted']}"
    )

    # -------------------------------------------------------------------------
    # PASS 7: FINAL PROPERTY / GRAPH CLEANUP
    # -------------------------------------------------------------------------

    removed_properties = enforce_properties(
        graph_nodes,
        ontology,
    )

    # Deduplicate relationships one last time.
    graph_relationships = remap_relationships(
        graph_relationships,
        {
            node["node_id"]: node["node_id"]
            for node in graph_nodes
        },
    )

    validation = validate_graph(
        graph_nodes,
        graph_relationships,
        ontology,
    )

    nodes_output, relationships_output = serialize_graph(
        graph_nodes,
        graph_relationships,
    )

    graph_output = {
        "schema_version": "generic-ontology-v2",
        "ontology": {
            "node_types": ontology[
                "node_types"
            ],
            "relationship_types": ontology[
                "relationship_types"
            ],
            "relationship_signatures": [
                {
                    "source": row[
                        "source_type"
                    ],
                    "relationship": row[
                        "relationship_type"
                    ],
                    "target": row[
                        "target_type"
                    ],
                }
                for row in ontology[
                    "rel_rows"
                ]
            ],
        },
        "nodes": nodes_output,
        "relationships": relationships_output,
        "ontology_coverage": {
            "node_types_found": validation[
                "found_node_types"
            ],
            "node_types_not_found": validation[
                "missing_node_types"
            ],
            "relationship_types_found": validation[
                "found_relationship_types"
            ],
            "relationship_types_not_found": validation[
                "missing_relationship_types"
            ],
        },
        "validation": validation,
        "extraction_summary": {
            "chunks_total": len(chunks),
            "chunks_successful": len(
                chunk_results
            ),
            "chunks_failed": len(
                failures
            ),
            "nodes": len(
                nodes_output
            ),
            "relationships": len(
                relationships_output
            ),
        },
    }

    audit_output = {
        "extraction_failures": failures,
        "entity_resolution": entity_report,
        "coverage_recovery": coverage_report,
        "relationship_recovery": relationship_report,
        "instance_verification": instance_report,
        "schema_property_cleanup": {
            "removed_count": len(
                removed_properties
            ),
            "removed": removed_properties,
        },
    }

    GRAPH_FILE.write_text(
        json.dumps(
            graph_output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    VALIDATION_FILE.write_text(
        json.dumps(
            validation,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    AUDIT_FILE.write_text(
        json.dumps(
            audit_output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print(
        f"Nodes               : "
        f"{validation['node_count']}"
    )
    print(
        f"Relationships       : "
        f"{validation['relationship_count']}"
    )
    print(
        f"Node types          : "
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
    print(
        f"Failed chunks       : "
        f"{len(failures)}"
    )
    print(
        f"Graph               : "
        f"{GRAPH_FILE}"
    )
    print(
        f"Audit               : "
        f"{AUDIT_FILE}"
    )

    if validation["missing_node_types"]:
        print(
            "\nNode types not found:"
        )
        for item in validation[
            "missing_node_types"
        ]:
            print(
                f"  - {item}"
            )

    if validation["missing_relationship_types"]:
        print(
            "\nRelationship types not found:"
        )
        for item in validation[
            "missing_relationship_types"
        ]:
            print(
                f"  - {item}"
            )

    if validation["errors"]:
        print(
            "\nValidation errors:"
        )
        for error in validation[
            "errors"
        ][:30]:
            print(
                f"  - {error}"
            )

    # Missing ontology types are NOT failures.
    # Actual structural errors and failed source extraction calls are.
    if validation["errors"] or failures:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
