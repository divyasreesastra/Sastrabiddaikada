"""
Fundscreening - LLM Graph Extractor

Reads chunk JSON files, applies Azure OpenAI extraction, merges chunk results,
removes orphaned relationships, merges duplicate node names, and asks the LLM
only to review truly disconnected nodes before writing the final graph.

Writes:
    graph_output/graph_extract.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import hashlib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = PROJECT_ROOT / "graph_output" / "chunk_results"
SCHEMA_DIR = PROJECT_ROOT / "schema"
NODES_SCHEMA = SCHEMA_DIR / "nodes.txt"
RELATIONSHIPS_SCHEMA = SCHEMA_DIR / "relationships.txt"
OUTPUT_DIR = PROJECT_ROOT / "graph_output"
OUTPUT_FILE = OUTPUT_DIR / "graph_extract.json"

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ---------------------------------------------------------------------------
# LLM SETTINGS
# ---------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION",
    "2024-10-21",
)
AZURE_OPENAI_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT",
    "gpt-4o",
)

MAX_RETRIES = 4
RETRY_BASE_SECONDS = 3
MAX_OUTPUT_TOKENS = 12000

SEMANTIC_REPAIR_MAX_ATTEMPTS = 3
SEMANTIC_REPAIR_MAX_CHUNKS_PER_TYPE = 8


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""
You are a highly precise graph extraction engine for a private-equity
fund-screening application.

Extract ONLY facts explicitly supported by the supplied document chunk.

The application uses a FINALIZED graph schema supplied below. The schema is
the structural boundary.

CRITICAL RULES

1. Never invent facts.
2. Never infer a relationship merely because two entities appear together.
3. Never create a node type that is not in the supplied schema.
4. Never create a relationship type that is not in the supplied schema.
5. Preserve the source meaning and terminology.
6. Extract all relevant facts, not just the examples in the property lists.
7. FinancialMetric is intentionally GENERIC. If the document contains a
   financial, operating, KPI, ratio, valuation, cash-flow, leverage or other
   measurable metric that is relevant to screening, extract it as a
   FinancialMetric even when its metric_name is NOT one of the example names.
8. AuditFinding is for audit/accounting/control observations. If the document
   discusses audit findings, quality-of-earnings adjustments, control issues,
   qualified opinions, restatements, or going-concern matters, extract them
   as AuditFinding nodes with appropriate severity and status.
9. Do not turn every metric into a new property on FinancialMetric.
   Use:
      metric_name
      value
      unit
      currency
      period
      fiscal_year
      basis
      status
      category
      source
   and put the actual metric name/value in those fields.
9. Keep numbers numeric whenever possible. Do not change units unless a
   normalized value can be safely derived from the supplied text.
10. Preserve currency and units.
11. If a fact is qualitative, use the relevant node/property rather than
    forcing it into FinancialMetric.
12. Every important extracted node/fact/relationship must contain evidence.
13. Evidence quote must be a SHORT verbatim excerpt from the supplied chunk.
14. Do not include unsupported conclusions such as "safe investment",
    "unsafe", "good", "bad", "pass", or "fail". Screening comes later.
15. If the chunk does not support an item, omit it.
16. If the same entity appears in multiple chunks, use a stable entity key so
    the downstream merger can reconcile it.
17. Return JSON only. No markdown. No explanation outside JSON.

OUTPUT FORMAT

{
  "nodes": [
    {
      "node_id": "stable_id",
      "node_type": "AllowedSchemaNode",
      "properties": {
        "property_name": "value"
      },
      "evidence": [
        {
          "document_id": "document_id",
          "source_file": "path",
          "page": 1,
          "chunk_id": "chunk_id",
          "quote": "short exact source quote"
        }
      ]
    }
  ],
  "relationships": [
    {
      "relationship_id": "stable_id",
      "relationship_type": "AllowedRelationship",
      "source_node_id": "stable_id",
      "target_node_id": "stable_id",
      "properties": {},
      "evidence": [
        {
          "document_id": "document_id",
          "source_file": "path",
          "page": 1,
          "chunk_id": "chunk_id",
          "quote": "short exact source quote"
        }
      ]
    }
  ],
  "warnings": []
}
"""


# ---------------------------------------------------------------------------
# ENVIRONMENT / CLIENT
# ---------------------------------------------------------------------------

def load_environment() -> None:
    """Load .env if python-dotenv is available."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")


def validate_environment() -> None:
    missing = []

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        missing.append("AZURE_OPENAI_ENDPOINT")

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        missing.append("AZURE_OPENAI_API_KEY")

    if missing:
        raise SystemExit(
            "Missing required environment variable(s): "
            + ", ".join(missing)
        )


def create_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv(
            "AZURE_OPENAI_API_VERSION",
            AZURE_OPENAI_API_VERSION,
        ),
    )


# ---------------------------------------------------------------------------
# FILE / SCHEMA HELPERS
# ---------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_schema() -> str:
    nodes = read_text_file(NODES_SCHEMA)
    relationships = read_text_file(RELATIONSHIPS_SCHEMA)

    return (
        "=== ALLOWED NODE TYPES AND PROPERTIES ===\n"
        f"{nodes}\n\n"
        "=== ALLOWED RELATIONSHIP TYPES ===\n"
        f"{relationships}"
    )



def parse_schema_contract() -> dict[str, Any]:
    """Parse only numbered top-level node headings; property names are not node types."""
    nodes_text = read_text_file(NODES_SCHEMA)
    rel_text = read_text_file(RELATIONSHIPS_SCHEMA)
    node_types = []
    node_properties = defaultdict(set)
    current = None
    in_props = False

    for raw in nodes_text.splitlines():
        line = raw.strip()
        m = re.match(r"^\s*\d+\.\s*([A-Za-z][A-Za-z0-9_]*)\s*$", line)
        if m:
            current = m.group(1)
            if current not in node_types:
                node_types.append(current)
            in_props = False
            continue
        if current and line.lower() == "properties:":
            in_props = True
            continue
        if current and in_props:
            pm = re.match(r"^-\s*([A-Za-z][A-Za-z0-9_]*)\s*$", line)
            if pm:
                node_properties[current].add(pm.group(1))
            elif line and not line.startswith("-"):
                in_props = False

    rels = defaultdict(set)
    current_sig = None
    for raw in rel_text.splitlines():
        line = raw.strip()
        m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*->\s*([A-Za-z][A-Za-z0-9_]*)\s*$", line)
        if m:
            current_sig = (m.group(1), m.group(2))
            continue
        if current_sig:
            rm = re.match(r"^-\s*([A-Z][A-Z0-9_]*)\s*$", line)
            if rm:
                rels[current_sig].add(rm.group(1))

    return {
        "node_types": node_types,
        "node_properties": {k: sorted(v) for k, v in node_properties.items()},
        "relationship_types": {f"{a}->{b}": sorted(v) for (a,b), v in rels.items()},
    }


def normalize_identity_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"\b(company|incorporated|inc|corp|corporation|ltd|limited|llc)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_node_name(node: dict[str, Any]) -> str:
    p = node.get("properties") or {}
    t = node.get("node_type")
    if t == "FinancialMetric":
        return str(p.get("metric_name") or p.get("name") or node.get("node_id") or "")
    if t == "Deal":
        return str(p.get("deal_name") or p.get("name") or node.get("node_id") or "")
    if t == "AuditFinding":
        return str(p.get("finding_id") or p.get("name") or p.get("description") or node.get("node_id") or "")
    return str(p.get("name") or p.get("legal_name") or node.get("node_id") or "")


def metric_identity(node: dict[str, Any]) -> tuple:
    p = node.get("properties") or {}
    return (
        normalize_identity_text(p.get("metric_name") or p.get("name")),
        normalize_identity_text(p.get("period") or p.get("fiscal_year") or p.get("quarter")),
        normalize_identity_text(p.get("basis")),
        normalize_identity_text(p.get("status")),
        normalize_identity_text(p.get("currency")),
        normalize_identity_text(p.get("unit")),
        str(p.get("value") if p.get("value") is not None else p.get("normalized_value")),
    )


def entity_key(node: dict[str, Any]) -> tuple:
    t = node.get("node_type", "")
    if t == "FinancialMetric":
        return (t, metric_identity(node))
    return (t, normalize_identity_text(get_node_name(node)))


def merge_property_values(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if v in (None, "", [], {}):
            continue
        if k not in dst or dst[k] in (None, "", [], {}):
            dst[k] = v
        elif isinstance(dst[k], list):
            vals = dst[k] if isinstance(dst[k], list) else [dst[k]]
            incoming = v if isinstance(v, list) else [v]
            for item in incoming:
                if item not in vals:
                    vals.append(item)


def resolve_entities(graph: dict[str, Any]) -> dict[str, Any]:
    """Conservative entity resolution; never collapses distinct metric observations."""
    canonical = {}
    redirect = {}
    merged = []

    for node in graph.get("nodes", []):
        key = entity_key(node)
        if key not in canonical:
            canonical[key] = node
            continue
        target = canonical[key]
        old_id = node.get("node_id")
        new_id = target.get("node_id")
        if old_id != new_id:
            redirect[old_id] = new_id
            merged.append({"node_type": node.get("node_type"), "from": old_id, "to": new_id})
        merge_property_values(target.setdefault("properties", {}), node.get("properties") or {})
        target["evidence"] = deduplicate_evidence((target.get("evidence") or []) + (node.get("evidence") or []))

    def resolve(nid):
        seen = set()
        while nid in redirect and nid not in seen:
            seen.add(nid)
            nid = redirect[nid]
        return nid

    nodes = []
    seen_ids = set()
    for node in canonical.values():
        nid = resolve(node.get("node_id"))
        node["node_id"] = nid
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append(node)

    rels = {}
    valid_ids = set(seen_ids)
    for rel in graph.get("relationships", []):
        sid = resolve(rel.get("source_node_id") or rel.get("source_id") or rel.get("source"))
        tid = resolve(rel.get("target_node_id") or rel.get("target_id") or rel.get("target"))
        if sid not in valid_ids or tid not in valid_ids:
            continue
        rel["source_node_id"] = sid
        rel["target_node_id"] = tid
        key = (sid, rel.get("relationship_type"), tid)
        if key not in rels:
            rels[key] = rel
        else:
            merge_property_values(rels[key].setdefault("properties", {}), rel.get("properties") or {})
            rels[key]["evidence"] = deduplicate_evidence((rels[key].get("evidence") or []) + (rel.get("evidence") or []))

    graph["nodes"] = nodes
    graph["relationships"] = list(rels.values())
    graph["entity_resolution"] = {"merged_count": len(merged), "merged_examples": merged[:100]}
    return graph


def validate_graph_against_schema(graph: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    allowed_nodes = set(contract["node_types"])
    allowed_rels = set()
    for sig, rels in contract["relationship_types"].items():
        a, b = sig.split("->")
        for r in rels:
            allowed_rels.add((a, b, r))

    counts = Counter()
    invalid_nodes = []
    invalid_props = []
    node_by_id = {}

    for node in graph.get("nodes", []):
        t = node.get("node_type")
        counts[t] += 1
        node_by_id[node.get("node_id")] = node
        if t not in allowed_nodes:
            invalid_nodes.append(node.get("node_id"))
        props = set((node.get("properties") or {}).keys())
        allowed_props = set(contract["node_properties"].get(t, []))
        invalid_props.extend([{"node_id": node.get("node_id"), "property": x} for x in sorted(props - allowed_props)])

    invalid_rels = []
    orphaned = []
    for rel in graph.get("relationships", []):
        s = node_by_id.get(rel.get("source_node_id"))
        t = node_by_id.get(rel.get("target_node_id"))
        if not s or not t:
            orphaned.append(rel.get("relationship_id"))
            continue
        sig = (s.get("node_type"), t.get("node_type"), rel.get("relationship_type"))
        if sig not in allowed_rels:
            invalid_rels.append({"relationship_id": rel.get("relationship_id"), "signature": sig})

    missing_evidence = [n.get("node_id") for n in graph.get("nodes", []) if not n.get("evidence")]
    missing_evidence += [r.get("relationship_id") for r in graph.get("relationships", []) if not r.get("evidence")]
    found = sorted(t for t in counts if t in allowed_nodes)
    missing = sorted(allowed_nodes - set(found))

    return {
        "required_base_node_types": contract["node_types"],
        "required_base_node_type_count": len(contract["node_types"]),
        "found_base_node_types": found,
        "found_base_node_type_count": len(found),
        "missing_base_node_types": missing,
        "node_type_counts": dict(sorted(counts.items())),
        "invalid_node_ids": invalid_nodes,
        "invalid_properties": invalid_props,
        "invalid_relationships": invalid_rels,
        "orphan_relationships": orphaned,
        "missing_evidence": missing_evidence,
        "errors": (
            [f"Unknown node types: {invalid_nodes}"] if invalid_nodes else []
        ) + (
            [f"{len(invalid_rels)} invalid relationship signatures"] if invalid_rels else []
        ) + (
            [f"{len(orphaned)} orphan relationships"] if orphaned else []
        ),
        "warnings": (
            [f"Missing base node types: {missing}"] if missing else []
        ) + (
            [f"{len(invalid_props)} properties outside predefined schema"] if invalid_props else []
        ),
    }


def relevant_chunks_for_type(chunks, node_type):
    terms = {
        "InvestmentFund": ["fund", "mandate", "investment", "exclusion"],
        "Company": ["company", "business", "customer", "product", "employee"],
        "Deal": ["deal", "transaction", "valuation", "purchase", "acquisition"],
        "FinancialMetric": ["arr", "revenue", "ebitda", "margin", "growth", "cash", "debt", "ratio"],
        "InvestmentCriterion": ["criterion", "criteria", "threshold", "minimum", "maximum", "screen"],
        "RiskFactor": ["risk", "concentration", "dependency", "exposure", "concern"],
        "Market": ["market", "tam", "sam", "sector", "geography", "competitor"],
        "AuditFinding": ["audit", "finding", "control", "restatement", "qualified", "going concern", "normalized"],
    }
    scored = []
    for c in chunks:
        text = str(c.get("text", "")).lower()
        score = sum(text.count(x) for x in terms.get(node_type, []))
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:SEMANTIC_REPAIR_MAX_CHUNKS_PER_TYPE]]


def semantic_repair(client, graph, chunks, schema_text, contract):
    logs = []
    for attempt in range(1, SEMANTIC_REPAIR_MAX_ATTEMPTS + 1):
        report = validate_graph_against_schema(graph, contract)
        missing = report["missing_base_node_types"]
        if not missing:
            break
        progress = False
        for wanted in missing:
            relevant = relevant_chunks_for_type(chunks, wanted)
            if not relevant:
                logs.append({"attempt": attempt, "type": wanted, "status": "no_relevant_chunks"})
                continue
            excerpts = "\n\n".join(
                f"DOCUMENT={c.get('document_id')} CHUNK={c.get('chunk_id')} PAGES={c.get('start_page')}-{c.get('end_page')}\n{c.get('text','')}"
                for c in relevant
            )
            prompt = f"""
Targeted evidence repair for missing base node type: {wanted}

This is NOT a quota. If the excerpts do not explicitly support a {wanted},
return an empty nodes list. Never fabricate a placeholder.

Use only this finalized schema:
{schema_text}

EXCERPTS:
{excerpts}

Return JSON only in the normal graph extraction format. Every node needs
short verbatim evidence. Only add relationships explicitly supported.
""".strip()
            try:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                result = normalize_result(parse_json_response(response.choices[0].message.content or ""), relevant[0])
                new_nodes = [n for n in result["nodes"] if n.get("node_type") == wanted]
                if new_nodes:
                    graph["nodes"].extend(new_nodes)
                    graph["relationships"].extend(result["relationships"])
                    progress = True
                    logs.append({"attempt": attempt, "type": wanted, "status": "evidence_found", "node_count": len(new_nodes)})
                else:
                    logs.append({"attempt": attempt, "type": wanted, "status": "no_supported_evidence"})
            except Exception as exc:
                logs.append({"attempt": attempt, "type": wanted, "status": "error", "error": str(exc)})
        graph = resolve_entities(graph)
        if not progress:
            break
    graph["base_node_completeness"] = validate_graph_against_schema(graph, contract)
    graph["base_node_completeness"]["repair_attempts"] = logs
    return graph


def load_chunk_files() -> list[dict[str, Any]]:
    if not CHUNKS_DIR.exists():
        raise SystemExit(
            f"Chunk directory not found: {CHUNKS_DIR}\n"
            "Run src/chunker.py first."
        )

    files = sorted(CHUNKS_DIR.glob("*_chunks.json"))

    if not files:
        raise SystemExit(
            f"No chunk JSON files found in: {CHUNKS_DIR}\n"
            "Run src/chunker.py first."
        )

    chunks: list[dict[str, Any]] = []

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))

        for chunk in payload.get("chunks", []):
            chunk["_chunk_file"] = str(path.relative_to(PROJECT_ROOT))
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# JSON EXTRACTION
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse strict JSON, with a conservative fallback for accidental
    markdown fences or surrounding whitespace.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Find the outermost JSON object as a last-resort recovery.
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain valid JSON.")

        value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object.")

    value.setdefault("nodes", [])
    value.setdefault("relationships", [])
    value.setdefault("warnings", [])

    return value


# ---------------------------------------------------------------------------
# VALIDATION / NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_string(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_evidence(
    evidence: Any,
    chunk: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        evidence = []

    normalized = []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        quote = normalize_string(item.get("quote"))
        if not quote:
            continue

        normalized.append(
            {
                "document_id": item.get(
                    "document_id",
                    chunk.get("document_id"),
                ),
                "source_file": item.get(
                    "source_file",
                    chunk.get("source_file"),
                ),
                "page": item.get("page"),
                "chunk_id": item.get(
                    "chunk_id",
                    chunk.get("chunk_id"),
                ),
                "quote": quote,
            }
        )

    # Guarantee at least one provenance record for extracted objects.
    if not normalized:
        normalized.append(
            {
                "document_id": chunk.get("document_id"),
                "source_file": chunk.get("source_file"),
                "page": chunk.get("start_page"),
                "chunk_id": chunk.get("chunk_id"),
                "quote": None,
            }
        )

    return normalized


def normalize_result(
    result: dict[str, Any],
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize LLM output without changing its business meaning.

    Detailed schema validation happens in graph_validator.py.
    """
    nodes = result.get("nodes", [])
    relationships = result.get("relationships", [])
    warnings = result.get("warnings", [])

    if not isinstance(nodes, list):
        nodes = []

    if not isinstance(relationships, list):
        relationships = []

    if not isinstance(warnings, list):
        warnings = []

    normalized_nodes = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_id = normalize_string(node.get("node_id"))
        node_type = normalize_string(node.get("node_type"))

        if not node_id or not node_type:
            continue

        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        normalized_nodes.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "properties": properties,
                "evidence": normalize_evidence(
                    node.get("evidence"),
                    chunk,
                ),
            }
        )

    normalized_relationships = []

    for rel in relationships:
        if not isinstance(rel, dict):
            continue

        rel_id = normalize_string(rel.get("relationship_id"))
        rel_type = normalize_string(rel.get("relationship_type"))
        source_id = normalize_string(rel.get("source_node_id"))
        target_id = normalize_string(rel.get("target_node_id"))

        if not rel_id or not rel_type or not source_id or not target_id:
            continue

        properties = rel.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        normalized_relationships.append(
            {
                "relationship_id": rel_id,
                "relationship_type": rel_type,
                "source_node_id": source_id,
                "target_node_id": target_id,
                "properties": properties,
                "evidence": normalize_evidence(
                    rel.get("evidence"),
                    chunk,
                ),
            }
        )

    return {
        "nodes": normalized_nodes,
        "relationships": normalized_relationships,
        "warnings": [str(w) for w in warnings],
    }


# ---------------------------------------------------------------------------
# PROMPT BUILDING
# ---------------------------------------------------------------------------

def build_user_prompt(
    chunk: dict[str, Any],
    schema_text: str,
) -> str:
    start_page = chunk.get("start_page")
    end_page = chunk.get("end_page")

    if start_page is None:
        page_text = "unknown"
    elif start_page == end_page:
        page_text = str(start_page)
    else:
        page_text = f"{start_page}-{end_page}"

    return f"""
DOCUMENT METADATA
document_id: {chunk.get("document_id")}
source_file: {chunk.get("source_file")}
chunk_id: {chunk.get("chunk_id")}
chunk_index: {chunk.get("chunk_index")}
page(s): {page_text}

FINALIZED GRAPH SCHEMA
{schema_text}

DOCUMENT TEXT
------------------------------
{chunk.get("text", "")}
------------------------------

Extract the complete set of relevant facts supported by this chunk.

SPECIAL REQUIREMENT FOR INVESTMENT FUNDS:
If the chunk discusses a fund, investment vehicle, or LP mandate, extract as
InvestmentFund with: fund_name, stage, sector, mandate, investment_criteria,
exclusions, and decision_framework. Include decision rules and thresholds.

SPECIAL REQUIREMENT FOR COMPANIES:
If the chunk describes a target company, extract as Company with: name,
legal_name, sector, industry, stage, headquarters, key products, competitive
advantages, ownership structure, and employee_count. Preserve org context.

SPECIAL REQUIREMENT FOR DEALS:
If the chunk describes a transaction, investment structure, or capital deployment,
extract as Deal with: deal_id, deal_name, structure, valuation, transaction_size,
date, parties involved, and investment_terms. Link to Fund and Company nodes.

SPECIAL REQUIREMENT FOR FINANCIAL METRICS:
Do not restrict to a small list. Extract ALL meaningful financial/operating/KPI/
valuation metrics with: metric_name, value, unit, currency, period, category,
status. Examples include ARR, growth rates, margins, ratios, multiples, cash flow.

SPECIAL REQUIREMENT FOR INVESTMENT CRITERIA:
If the chunk lists screening requirements, thresholds, or diligence items,
extract as InvestmentCriterion with: name, criterion_type (Mandatory/Soft),
category, description, numeric_value/target if applicable, mandatory flag.

SPECIAL REQUIREMENT FOR RISK FACTORS:
If the chunk identifies risks, concerns, dependencies, or adverse conditions,
extract as RiskFactor with: name, category, description, severity (High/Medium/Low),
status, and mitigation/monitoring strategies if mentioned.

SPECIAL REQUIREMENT FOR MARKETS:
If the chunk discusses geography, sectors, competitive landscape, or market context,
extract as Market with: name, market_type (Geography/Sector/Competitive), description,
size/potential, key players, growth_rate, or regulatory environment.

SPECIAL REQUIREMENT FOR AUDIT FINDINGS:
If the chunk discusses audit findings, quality-of-earnings adjustments, control
issues, qualified opinions, restatements, or going-concern matters, extract as
AuditFinding with: finding_id, finding_type, category, description, severity,
status, and source_text capturing the audit observation.

SPECIAL REQUIREMENT FOR RELATIONSHIPS:
Use only the allowed relationship types in the schema. Select them dynamically
according to the actual meaning in the text. Do not force unsupported relationships.
Ensure both source and target nodes are explicitly extracted (no orphaned endpoints).

Return JSON only.
""".strip()


# ---------------------------------------------------------------------------
# LLM CALL
# ---------------------------------------------------------------------------

def extract_chunk(
    client: AzureOpenAI,
    chunk: dict[str, Any],
    schema_text: str,
) -> dict[str, Any]:

    user_prompt = build_user_prompt(chunk, schema_text)

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            content = response.choices[0].message.content or ""
            result = parse_json_response(content)

            return normalize_result(result, chunk)

        except Exception as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            wait_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))

            print(
                f"      Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"      Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"LLM extraction failed for {chunk.get('chunk_id')}: "
        f"{last_error}"
    )


# ---------------------------------------------------------------------------
# MERGING
# ---------------------------------------------------------------------------

def merge_results(
    chunk_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge chunk-level results by stable node/relationship IDs.

    Properties from later chunks fill missing values but do not overwrite
    existing non-empty values blindly. Evidence is accumulated.
    """
    node_map: dict[str, dict[str, Any]] = {}
    relationship_map: dict[str, dict[str, Any]] = {}

    warnings: list[str] = []
    source_documents: set[str] = set()

    for item in chunk_results:
        source_documents.add(item["document_id"])

        warnings.extend(item.get("warnings", []))

        for node in item.get("nodes", []):
            node_id = node["node_id"]

            if node_id not in node_map:
                node_map[node_id] = node
                continue

            existing = node_map[node_id]

            for key, value in node.get("properties", {}).items():
                if (
                    key not in existing["properties"]
                    or existing["properties"][key] in (None, "", [])
                ):
                    existing["properties"][key] = value

            existing_evidence = existing.setdefault("evidence", [])
            existing_evidence.extend(node.get("evidence", []))

        for rel in item.get("relationships", []):
            rel_id = rel["relationship_id"]

            if rel_id not in relationship_map:
                relationship_map[rel_id] = rel
                continue

            existing = relationship_map[rel_id]

            for key, value in rel.get("properties", {}).items():
                if (
                    key not in existing["properties"]
                    or existing["properties"][key] in (None, "", [])
                ):
                    existing["properties"][key] = value

            existing_evidence = existing.setdefault("evidence", [])
            existing_evidence.extend(rel.get("evidence", []))

    # De-duplicate evidence while preserving order.
    for node in node_map.values():
        node["evidence"] = deduplicate_evidence(node.get("evidence", []))

    for rel in relationship_map.values():
        rel["evidence"] = deduplicate_evidence(rel.get("evidence", []))

    return {
        "schema_version": "final-8-base-nodes",
        "source_documents": sorted(source_documents),
        "nodes": list(node_map.values()),
        "relationships": list(relationship_map.values()),
        "warnings": sorted(set(warnings)),
    }


def deduplicate_evidence(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []

    for item in evidence:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    load_environment()
    validate_environment()

    schema_text = load_schema()
    schema_contract = parse_schema_contract()
    chunks = load_chunk_files()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = create_client()

    print("=" * 72)
    print("FUNDSCREENING - LLM GRAPH EXTRACTION")
    print("=" * 72)
    print(f"Azure deployment : {AZURE_OPENAI_DEPLOYMENT}")
    print(f"Chunks discovered: {len(chunks)}")
    print()

    chunk_results = []
    failures = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{index}")

        print(
            f"[{index}/{len(chunks)}] "
            f"{chunk.get('document_id')} | "
            f"{chunk_id} | "
            f"{chunk.get('character_count', len(chunk.get('text', '')))} chars"
        )

        try:
            result = extract_chunk(
                client=client,
                chunk=chunk,
                schema_text=schema_text,
            )

            # Keep source metadata outside the model response.
            result["document_id"] = chunk.get("document_id")
            result["chunk_id"] = chunk.get("chunk_id")

            chunk_results.append(result)

            print(
                f"    nodes={len(result['nodes'])}, "
                f"relationships={len(result['relationships'])}"
            )

        except Exception as exc:
            failure = {
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "error": str(exc),
            }
            failures.append(failure)

            print(f"    FAILED: {exc}")

    final_graph = merge_results(chunk_results)

    # Evidence-aware entity resolution removes chunk-generated duplicates
    # without collapsing distinct metric observations.
    final_graph = resolve_entities(final_graph)

    # If a base type is genuinely absent, ask the LLM to search only relevant
    # source chunks. This is a repair pass, not a node-count quota.
    final_graph = semantic_repair(
        client,
        final_graph,
        chunks,
        schema_text,
        schema_contract,
    )

    # Enforce the predefined node properties. Unknown model-generated keys are
    # dropped rather than leaking into Neo4j. Preserve the common source alias.
    for node in final_graph.get("nodes", []):
        ntype = node.get("node_type")
        props = node.get("properties") or {}
        if "source" in props and "source_text" not in props:
            props["source_text"] = props["source"]
        allowed = set(schema_contract["node_properties"].get(ntype, []))
        node["properties"] = {k: v for k, v in props.items() if k in allowed}

    # Final strict relationship validation. Never create placeholder endpoints.
    node_by_id = {n.get("node_id"): n for n in final_graph.get("nodes", [])}
    allowed_sigs = set()
    for sig, rels in schema_contract["relationship_types"].items():
        a, b = sig.split("->")
        for rel_type in rels:
            allowed_sigs.add((a, b, rel_type))

    final_relationships = []
    filtered_relationships = []
    seen_rel_keys = set()

    for rel in final_graph.get("relationships", []):
        sid = rel.get("source_node_id") or rel.get("source_id") or rel.get("source")
        tid = rel.get("target_node_id") or rel.get("target_id") or rel.get("target")
        sn = node_by_id.get(sid)
        tn = node_by_id.get(tid)
        if not sn or not tn:
            filtered_relationships.append({
                "relationship_id": rel.get("relationship_id"),
                "reason": "orphan_endpoint",
            })
            continue
        sig = (sn.get("node_type"), tn.get("node_type"), rel.get("relationship_type"))
        if sig not in allowed_sigs:
            filtered_relationships.append({
                "relationship_id": rel.get("relationship_id"),
                "reason": "relationship_signature_not_allowed",
                "signature": sig,
            })
            continue
        key = (sid, rel.get("relationship_type"), tid)
        if key in seen_rel_keys:
            continue
        seen_rel_keys.add(key)
        final_relationships.append(rel)

    final_graph["relationships"] = final_relationships

    # Loader-compatible aliases.
    for n in final_graph.get("nodes", []):
        n["id"] = n.get("node_id")
        n["label"] = n.get("node_type")
    for r in final_graph.get("relationships", []):
        r["source_id"] = r.get("source_node_id")
        r["target_id"] = r.get("target_node_id")
        r["source"] = r.get("source_node_id")
        r["target"] = r.get("target_node_id")

    validation = validate_graph_against_schema(final_graph, schema_contract)
    final_graph["base_node_completeness"] = validation
    final_graph["schema_contract"] = {
        "base_node_types": schema_contract["node_types"],
        "relationship_signatures": schema_contract["relationship_types"],
    }
    final_graph["extraction_summary"] = {
        "chunks_discovered": len(chunks),
        "chunks_succeeded": len(chunk_results),
        "chunks_failed": len(failures),
        "failure_details": failures,
        "nodes_count": len(final_graph["nodes"]),
        "relationships_count": len(final_graph["relationships"]),
        "node_type_counts": dict(sorted(Counter(n.get("node_type") for n in final_graph["nodes"]).items())),
        "relationship_type_counts": dict(sorted(Counter(r.get("relationship_type") for r in final_graph["relationships"]).items())),
        "relationships_filtered": len(filtered_relationships),
        "ontology_base_node_count": len(schema_contract["node_types"]),
        "ontology_base_node_types": schema_contract["node_types"],
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            final_graph,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)

    if failures:
        print("EXTRACTION FINISHED WITH SOME FAILURES")
    else:
        print("EXTRACTION FINISHED SUCCESSFULLY")

    print("=" * 72)
    print(f"Chunks succeeded : {len(chunk_results)}")
    print(f"Chunks failed    : {len(failures)}")
    print(f"Final nodes       : {len(final_graph['nodes'])}")
    print(f"Final relationships: {len(final_graph['relationships'])}")
    print(
        f"Base node types   : "
        f"{validation['found_base_node_type_count']}/"
        f"{validation['required_base_node_type_count']}"
    )
    print(f"Missing base types: {validation['missing_base_node_types']}")
    print(f"Output            : {OUTPUT_FILE}")

    if failures:
        print()
        print(
            "The failed chunks are recorded in extraction_summary.failure_details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
