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
      description
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
If the chunk discusses a fund, extract an InvestmentFund using ONLY properties
from nodes.txt. Use mandate_notes for detailed mandate/decision-framework text
and create InvestmentCriterion nodes for explicit screening rules.

SPECIAL REQUIREMENT FOR COMPANIES:
Use ONLY Company properties from nodes.txt. Put key products/services into
products_services; do not invent key_products or other properties.

SPECIAL REQUIREMENT FOR DEALS:
Use ONLY Deal properties from nodes.txt. Map valuation to entry_valuation,
enterprise_value, equity_value, or purchase_price when the source supports that
meaning. Do not invent transaction_size, valuation, currency, unit, or investment_terms
properties.

SPECIAL REQUIREMENT FOR FINANCIAL METRICS:
Extract all meaningful financial/operating/KPI/valuation metrics, but use ONLY
the FinancialMetric properties defined in nodes.txt.

SPECIAL REQUIREMENT FOR INVESTMENT CRITERIA:
Create InvestmentCriterion nodes for explicit screening requirements, thresholds,
minimums, maximums, preferences, exclusions, and rationale. Use ONLY its defined
properties.

SPECIAL REQUIREMENT FOR RISK FACTORS:
Create RiskFactor nodes for explicit risks, concerns, dependencies, and adverse
conditions. Use ONLY its defined properties.

SPECIAL REQUIREMENT FOR MARKETS:
Create Market nodes for explicit market/geography/sector/competitive context.
Use ONLY its defined properties; put unsupported extra detail into description.

SPECIAL REQUIREMENT FOR AUDIT FINDINGS:
Create AuditFinding nodes for explicit audit/accounting/control observations.
Use ONLY its defined properties.

SPECIAL REQUIREMENT FOR RELATIONSHIPS:
Use only relationship types from relationships.txt. A relationship requires
explicit semantic support in the source, not simple co-occurrence.

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
# FINAL ONTOLOGY / FEEDBACK / INSTANCE REVIEW
# ---------------------------------------------------------------------------

EXPECTED_NODE_TYPES = [
    "InvestmentFund", "Company", "Deal", "FinancialMetric",
    "InvestmentCriterion", "RiskFactor", "Market", "AuditFinding",
]


def parse_ontology(nodes_text: str, relationships_text: str) -> dict[str, Any]:
    """Parse the actual ontology. Relationship-property convention is not ontology."""
    node_types = []
    node_properties = {}
    current = None
    in_properties = False
    heading_re = re.compile(r"^\s*(\d+)\.\s*([A-Za-z][A-Za-z0-9_]*)\s*$")

    for line in nodes_text.splitlines():
        m = heading_re.match(line)
        if m:
            current = m.group(2)
            node_types.append(current)
            node_properties[current] = set()
            in_properties = False
            continue
        if current and re.match(r"^\s*Properties\s*:\s*$", line, re.I):
            in_properties = True
            continue
        if current and in_properties:
            pm = re.match(r"^\s*-\s*([A-Za-z][A-Za-z0-9_]*)\s*$", line)
            if pm:
                node_properties[current].add(pm.group(1))
            elif line.strip() and not line.lstrip().startswith("-"):
                in_properties = False

    rel_allowed = {}
    rel_rows = []
    current_sig = None
    for line in relationships_text.splitlines():
        if line.strip().upper() == "GENERAL RELATIONSHIP PROPERTY CONVENTION":
            break
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*->\s*([A-Za-z][A-Za-z0-9_]*)\s*$", line)
        if m:
            current_sig = (m.group(1), m.group(2))
            rel_allowed.setdefault(current_sig, set())
            continue
        if current_sig:
            rm = re.match(r"^\s*-\s*([A-Z][A-Z0-9_]*)\s*$", line)
            if rm:
                rel = rm.group(1)
                rel_allowed[current_sig].add(rel)
                rel_rows.append({
                    "source_type": current_sig[0],
                    "relationship_type": rel,
                    "target_type": current_sig[1],
                })

    if node_types != EXPECTED_NODE_TYPES:
        raise RuntimeError(f"Expected 8 node types, resolved: {node_types}")
    if len(rel_rows) != 45:
        raise RuntimeError(
            f"Expected 45 relationship types from relationships.txt, resolved {len(rel_rows)}."
        )
    return {
        "node_types": node_types,
        "node_properties": node_properties,
        "rel_allowed": rel_allowed,
        "rel_rows": rel_rows,
    }


def norm_name(value: Any) -> str:
    s = str(value or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(inc|incorporated|ltd|limited|llc|corp|corporation|co)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def node_display_name(node: dict[str, Any]) -> str:
    p = node.get("properties", {})
    t = node.get("node_type")
    if t == "FinancialMetric":
        return str(p.get("metric_name") or node.get("node_id"))
    if t == "Deal":
        return str(p.get("deal_name") or node.get("node_id"))
    if t == "AuditFinding":
        return str(p.get("finding_id") or p.get("description") or node.get("node_id"))
    return str(p.get("name") or node.get("node_id"))


def merge_props(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k not in out or out[k] in (None, "", []):
            out[k] = v
    return out



def filter_to_ontology(graph: dict[str, Any], ontology: dict[str, Any]) -> dict[str, Any]:
    """Drop model output outside the finalized ontology before any feedback pass."""
    valid_types = set(ontology["node_types"])
    nodes = [n for n in graph.get("nodes", []) if n.get("node_type") in valid_types]
    node_map = {n.get("node_id"): n for n in nodes}
    rels = []
    rejected = []
    for r in graph.get("relationships", []):
        s = node_map.get(r.get("source_node_id"))
        t = node_map.get(r.get("target_node_id"))
        if not s or not t:
            rejected.append({"relationship_id": r.get("relationship_id"), "reason": "unknown endpoint"})
            continue
        allowed = ontology["rel_allowed"].get((s["node_type"], t["node_type"]), set())
        if r.get("relationship_type") not in allowed:
            rejected.append({"relationship_id": r.get("relationship_id"), "reason": "relationship signature/type not allowed"})
            continue
        rels.append(r)
    graph["nodes"] = nodes
    graph["relationships"] = rels
    graph["ontology_filter"] = {"rejected_relationships": rejected, "rejected_relationship_count": len(rejected)}
    return graph

def conservative_entity_resolution(graph: dict[str, Any]) -> dict[str, Any]:
    """Deterministic only. No LLM-driven merge is performed here."""
    canonical = {}
    aliases = {}
    merges = []

    def key(n):
        t = n["node_type"]
        p = n.get("properties", {})
        if t == "FinancialMetric":
            # Preserve different periods/bases/status/values as separate observations.
            return (t, norm_name(p.get("metric_name")), norm_name(p.get("period")),
                    norm_name(p.get("fiscal_year")), norm_name(p.get("basis")),
                    norm_name(p.get("status")), norm_name(p.get("value")),
                    norm_name(p.get("currency")), norm_name(p.get("unit")))
        if t == "AuditFinding":
            return (t, norm_name(p.get("finding_id") or p.get("description")))
        if t == "Deal":
            return (t, norm_name(p.get("deal_name") or n["node_id"]))
        return (t, norm_name(p.get("name") or n["node_id"]))

    for n in graph.get("nodes", []):
        k = key(n)
        if k not in canonical:
            canonical[k] = n
            aliases[n["node_id"]] = n["node_id"]
        else:
            target = canonical[k]
            aliases[n["node_id"]] = target["node_id"]
            target["properties"] = merge_props(target.get("properties", {}), n.get("properties", {}))
            target["evidence"] = deduplicate_evidence(target.get("evidence", []) + n.get("evidence", []))
            merges.append({"from": n["node_id"], "to": target["node_id"], "reason": "deterministic identity key"})

    rels = []
    seen = set()
    for r in graph.get("relationships", []):
        r["source_node_id"] = aliases.get(r.get("source_node_id"), r.get("source_node_id"))
        r["target_node_id"] = aliases.get(r.get("target_node_id"), r.get("target_node_id"))
        k = (r.get("source_node_id"), r.get("relationship_type"), r.get("target_node_id"))
        if k in seen:
            continue
        seen.add(k)
        r["relationship_id"] = "rel_" + hashlib.sha1("|".join(map(str, k)).encode()).hexdigest()[:16]
        rels.append(r)

    graph["nodes"] = list(canonical.values())
    graph["relationships"] = rels
    graph["entity_resolution"] = {"merged_count": len(merges), "merges": merges}
    return graph


def ontology_audit(graph: dict[str, Any], ontology: dict[str, Any]) -> dict[str, Any]:
    node_counts = {t: 0 for t in ontology["node_types"]}
    for n in graph.get("nodes", []):
        if n.get("node_type") in node_counts:
            node_counts[n["node_type"]] += 1

    rel_counts = {r["relationship_type"]: 0 for r in ontology["rel_rows"]}
    errors = []
    node_ids = {n.get("node_id") for n in graph.get("nodes", [])}
    for n in graph.get("nodes", []):
        if n.get("node_type") not in ontology["node_types"]:
            errors.append(f"Unknown node type: {n.get('node_type')}")
        allowed_props = ontology["node_properties"].get(n.get("node_type"), set())
        bad = sorted(set(n.get("properties", {})) - allowed_props)
        if bad:
            errors.append(f"Unsupported properties on {n.get('node_id')}: {bad}")

    valid_relationships = []
    orphaned = []
    for r in graph.get("relationships", []):
        s = next((n for n in graph["nodes"] if n.get("node_id") == r.get("source_node_id")), None)
        t = next((n for n in graph["nodes"] if n.get("node_id") == r.get("target_node_id")), None)
        if not s or not t:
            orphaned.append(r)
            continue
        allowed = ontology["rel_allowed"].get((s["node_type"], t["node_type"]), set())
        if r.get("relationship_type") not in allowed:
            errors.append(f"Invalid relationship signature: {s['node_type']} -> {r.get('relationship_type')} -> {t['node_type']}")
            continue
        evidence = [e for e in r.get("evidence", []) if e.get("quote")]
        if not evidence:
            errors.append(f"Relationship without evidence: {r.get('relationship_id')}")
            continue
        rel_counts[r["relationship_type"]] = rel_counts.get(r["relationship_type"], 0) + 1
        valid_relationships.append(r)

    found_nodes = [t for t in ontology["node_types"] if node_counts[t] > 0]
    missing_nodes = [t for t in ontology["node_types"] if node_counts[t] == 0]
    found_rels = [r["relationship_type"] for r in ontology["rel_rows"] if rel_counts.get(r["relationship_type"], 0) > 0]
    missing_rels = [r for r in ontology["rel_rows"] if rel_counts.get(r["relationship_type"], 0) == 0]

    return {
        "node_type_counts": node_counts,
        "found_node_types": found_nodes,
        "missing_node_types": missing_nodes,
        "relationship_type_counts": rel_counts,
        "found_relationship_types": found_rels,
        "missing_relationship_types": missing_rels,
        "errors": errors,
        "orphan_relationship_count": len(orphaned),
    }


def generic_llm_json(client: AzureOpenAI, system: str, user: str, max_tokens: int = 10000) -> dict[str, Any]:
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return parse_json_response(response.choices[0].message.content or "{}")
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
    raise RuntimeError(f"LLM feedback call failed: {last}")


def select_relevant_chunks(chunks: list[dict[str, Any]], needles: list[str], limit: int = 8) -> list[dict[str, Any]]:
    scored = []
    nn = [norm_name(x) for x in needles if norm_name(x)]
    for c in chunks:
        text = norm_name(c.get("text", ""))
        score = sum(1 for x in nn if len(x) >= 3 and x in text)
        if score:
            scored.append((score, len(text), c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored[:limit]]


def context_for_chunks(chunks: list[dict[str, Any]], max_chars: int = 26000) -> str:
    parts, used = [], 0
    for c in chunks:
        block = f"\n--- {c.get('chunk_id')} pages {c.get('start_page')}-{c.get('end_page')} ---\n{c.get('text','')}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts)


NODE_FEEDBACK_KEYWORDS = {
    "InvestmentFund": ["fund", "mandate", "investment strategy", "investment criteria", "ticket size"],
    "Company": ["company", "business", "headquarters", "products", "customers"],
    "Deal": ["deal", "transaction", "investment", "purchase", "acquisition", "valuation"],
    "FinancialMetric": ["revenue", "arr", "ebitda", "margin", "cash flow", "debt", "growth", "%"],
    "InvestmentCriterion": ["criterion", "criteria", "threshold", "minimum", "maximum", "screen", "exclude"],
    "RiskFactor": ["risk", "concentration", "dependency", "exposure", "concern", "mitigation"],
    "Market": ["market", "tam", "sam", "sector", "industry", "competitive", "geography"],
    "AuditFinding": ["audit", "finding", "control", "restatement", "qualified", "going concern", "quality of earnings", "adjustment"],
}


def node_feedback(client, graph, chunks, ontology):
    audit = ontology_audit(graph, ontology)
    rounds = []
    unavailable = []
    for round_no in range(1, 3):
        missing = audit["missing_node_types"]
        if not missing:
            break
        added = 0
        results = []
        for node_type in missing:
            relevant = select_relevant_chunks(chunks, NODE_FEEDBACK_KEYWORDS[node_type], limit=10)
            if not relevant:
                # For a small corpus, do not let keyword retrieval itself cause a false
                # "unavailable" result. Review the complete chunk set once.
                relevant = chunks[:12]
            if not relevant:
                unavailable.append({"node_type": node_type, "reason": "No source chunks are available."})
                continue
            system = """You are a completeness auditor for a strict 8-node knowledge graph. Extract ONLY the requested node type when explicitly supported by the source. Never invent. Every returned node needs a verbatim evidence quote. Return JSON only."""
            user = f"""REQUESTED NODE TYPE: {node_type}\n\nALLOWED SCHEMA:\n{ontology['node_types']}\n\nPROPERTIES FOR {node_type}:\n{sorted(ontology['node_properties'][node_type])}\n\nSOURCE CHUNKS:\n{context_for_chunks(relevant)}\n\nReturn {{\"available\": true/false, \"nodes\": [...], \"reason\": \"...\"}}. If unsupported, return available=false. Do not create placeholders."""
            result = generic_llm_json(client, system, user, 8000)
            valid_nodes = []
            for n in result.get("nodes", []) if isinstance(result.get("nodes"), list) else []:
                if n.get("node_type") != node_type or not n.get("node_id"):
                    continue
                if not any(e.get("quote") for e in n.get("evidence", []) if isinstance(e, dict)):
                    continue
                valid_nodes.append(n)
            graph["nodes"].extend(valid_nodes)
            added += len(valid_nodes)
            results.append({"node_type": node_type, "available": bool(valid_nodes), "nodes_added": len(valid_nodes), "reason": result.get("reason")})
            if not valid_nodes:
                unavailable.append({"node_type": node_type, "reason": result.get("reason", "No explicit evidence found after feedback pass.")})
        if added:
            graph = conservative_entity_resolution(graph)
        rounds.append({"round": round_no, "missing_before": missing, "nodes_added": added, "results": results})
        audit = ontology_audit(graph, ontology)
        if added == 0:
            break
    return graph, {"rounds": rounds, "unavailable": unavailable}


def build_node_catalog(nodes: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{n['node_id']} | {n['node_type']} | {node_display_name(n)}"
        for n in nodes
    )


def relationship_feedback(client, graph, chunks, ontology, only_missing: bool = False):
    """Review each node-type signature once; optionally focus on still-missing relationship types."""
    results = []
    accepted = 0
    rejected = []
    node_by_id = {n["node_id"]: n for n in graph["nodes"]}
    existing = {(r["source_node_id"], r["relationship_type"], r["target_node_id"]) for r in graph["relationships"]}
    audit_now = ontology_audit(graph, ontology)
    missing_types = {x["relationship_type"] for x in audit_now["missing_relationship_types"]}

    for signature, allowed_types in ontology["rel_allowed"].items():
        review_types = set(allowed_types) & missing_types if only_missing else set(allowed_types)
        if only_missing and not review_types:
            continue
        source_type, target_type = signature
        sources = [n for n in graph["nodes"] if n["node_type"] == source_type]
        targets = [n for n in graph["nodes"] if n["node_type"] == target_type]
        if not sources or not targets:
            results.append({"signature": f"{source_type}->{target_type}", "status": "not_applicable", "reason": "One endpoint node type is absent."})
            continue

        names = [node_display_name(n) for n in sources + targets]
        relevant = select_relevant_chunks(chunks, names, limit=10)
        if not relevant:
            # Use evidence chunks from the endpoint nodes as fallback.
            ids = {e.get("chunk_id") for n in sources + targets for e in n.get("evidence", []) if e.get("chunk_id")}
            relevant = [c for c in chunks if c.get("chunk_id") in ids][:10]
        if not relevant:
            results.append({"signature": f"{source_type}->{target_type}", "status": "no_relevant_chunks"})
            continue

        catalog = build_node_catalog(sources + targets)
        system = """You are a strict relationship auditor. Use only explicit evidence from the supplied source chunks. Do not infer from co-occurrence. Return only relationships whose source and target node IDs are in the supplied catalog and whose relationship type is allowed. Each relationship must include a short verbatim quote from the source."""
        user = f"""SIGNATURE: {source_type} -> {target_type}\nALLOWED RELATIONSHIP TYPES UNDER REVIEW: {sorted(review_types)}\n\nCANONICAL NODES:\n{catalog}\n\nSOURCE CHUNKS:\n{context_for_chunks(relevant)}\n\nReturn JSON: {{\"relationships\":[{{\"relationship_type\":\"...\",\"source_node_id\":\"...\",\"target_node_id\":\"...\",\"properties\":{{}},\"evidence\":[{{\"chunk_id\":\"...\",\"quote\":\"...\"}}]}}]}}. Do not return unsupported relationship types."""
        result = generic_llm_json(client, system, user, 9000)
        local_added = 0
        for r in result.get("relationships", []) if isinstance(result.get("relationships"), list) else []:
            sid, tid, rt = r.get("source_node_id"), r.get("target_node_id"), r.get("relationship_type")
            s, t = node_by_id.get(sid), node_by_id.get(tid)
            if not s or not t or rt not in review_types:
                rejected.append({"relationship": r, "reason": "endpoint or ontology mismatch"})
                continue
            key = (sid, rt, tid)
            if key in existing:
                continue
            evidence = [e for e in r.get("evidence", []) if isinstance(e, dict) and e.get("quote") and evidence_quote_exists(e.get("quote"), chunks)]
            if not evidence:
                rejected.append({"relationship": r, "reason": "evidence quote not found in source"})
                continue
            graph["relationships"].append({
                "relationship_id": "rel_" + hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:16],
                "relationship_type": rt,
                "source_node_id": sid,
                "target_node_id": tid,
                "properties": r.get("properties", {}) if isinstance(r.get("properties"), dict) else {},
                "evidence": evidence,
            })
            existing.add(key)
            local_added += 1
        accepted += local_added
        results.append({"signature": f"{source_type}->{target_type}", "status": "reviewed", "relationships_added": local_added, "allowed_types": sorted(allowed_types)})
    return {"signatures": results, "relationships_accepted": accepted, "rejected": rejected}


def evidence_quote_exists(quote: str, chunks: list[dict[str, Any]]) -> bool:
    q = re.sub(r"\s+", " ", quote or "").strip().lower()
    if len(q) < 8:
        return False
    return any(q in re.sub(r"\s+", " ", c.get("text", "")).strip().lower() for c in chunks)


def instance_verification(client, graph, chunks, ontology):
    """Batch-review instances, then use the same evidence gate for proposed missing edges."""
    reviews = []
    candidates = []
    node_types = ontology["node_types"]
    calls = 0
    for node_type in node_types:
        typed = [n for n in graph["nodes"] if n["node_type"] == node_type]
        for start in range(0, len(typed), 12):
            batch = typed[start:start + 12]
            if not batch:
                continue
            evidence_ids = {
                e.get("chunk_id")
                for n in batch
                for e in n.get("evidence", [])
                if isinstance(e, dict) and e.get("chunk_id")
            }
            relevant = [c for c in chunks if c.get("chunk_id") in evidence_ids]
            if len(relevant) < 3:
                extra = select_relevant_chunks(chunks, [node_display_name(n) for n in batch], limit=8)
                seen_ids = {c.get("chunk_id") for c in relevant}
                relevant.extend(c for c in extra if c.get("chunk_id") not in seen_ids)
            relevant = relevant[:8]
            if not relevant:
                continue
            catalog = build_node_catalog(batch)
            system = """You are the final instance-level graph verifier. For EVERY supplied node instance, determine whether the source explicitly supports it. Determine whether it is an independent graph instance, a duplicate/alias of another supplied instance, or an observation/attribute that should remain a node only because the ontology defines that node type. Also identify explicit dependencies among supplied instances and propose only ontology-valid missing relationships. Do not invent or merge automatically. Every proposed relationship must have a verbatim source quote. Return JSON only."""
            user = f"""ONTOLOGY:\n{schema_prompt({'nodes_text': open(NODES_SCHEMA, encoding='utf-8').read(), 'relationships_text': open(RELATIONSHIPS_SCHEMA, encoding='utf-8').read()})}\n\nNODE INSTANCES:\n{catalog}\n\nSOURCE CHUNKS:\n{context_for_chunks(relevant)}\n\nReturn {{\"node_reviews\":[{{\"node_id\":\"...\",\"supported\":true,\"independent\":true,\"reason\":\"...\"}}],\"relationship_candidates\":[{{\"relationship_type\":\"...\",\"source_node_id\":\"...\",\"target_node_id\":\"...\",\"properties\":{{}},\"evidence\":[{{\"chunk_id\":\"...\",\"quote\":\"...\"}}]}}]}}"""
            result = generic_llm_json(client, system, user, 9000)
            reviews.extend(result.get("node_reviews", []) if isinstance(result.get("node_reviews"), list) else [])
            candidates.extend(result.get("relationship_candidates", []) if isinstance(result.get("relationship_candidates"), list) else [])
            calls += 1

    node_map = {n["node_id"]: n for n in graph["nodes"]}
    existing = {(r["source_node_id"], r["relationship_type"], r["target_node_id"]) for r in graph["relationships"]}
    accepted = 0
    rejected = []
    for r in candidates:
        sid, tid, rt = r.get("source_node_id"), r.get("target_node_id"), r.get("relationship_type")
        s, t = node_map.get(sid), node_map.get(tid)
        if not s or not t:
            rejected.append({"relationship": r, "reason": "unknown endpoint"})
            continue
        if rt not in ontology["rel_allowed"].get((s["node_type"], t["node_type"]), set()):
            rejected.append({"relationship": r, "reason": "not allowed for endpoint types"})
            continue
        key = (sid, rt, tid)
        if key in existing:
            continue
        evidence = [e for e in r.get("evidence", []) if isinstance(e, dict) and evidence_quote_exists(e.get("quote"), chunks)]
        if not evidence:
            rejected.append({"relationship": r, "reason": "no source-grounded evidence"})
            continue
        graph["relationships"].append({
            "relationship_id": "rel_" + hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:16],
            "relationship_type": rt,
            "source_node_id": sid,
            "target_node_id": tid,
            "properties": r.get("properties", {}) if isinstance(r.get("properties"), dict) else {},
            "evidence": evidence,
        })
        existing.add(key)
        accepted += 1
    return {"calls": calls, "node_reviews": reviews, "relationship_candidates": len(candidates), "relationships_accepted": accepted, "relationships_rejected": rejected, "note": "LLM verification does not delete or merge instances automatically; deterministic entity resolution is used for merges, while this report records support/independence/dependencies and source-backed missing edges."}



def enforce_schema_properties(graph: dict[str, Any], ontology: dict[str, Any]) -> None:
    removed = []
    for n in graph.get("nodes", []):
        allowed = ontology["node_properties"].get(n.get("node_type"), set())
        props = n.get("properties", {})
        for key in list(props):
            if key not in allowed:
                removed.append({"node_id": n.get("node_id"), "property": key, "value": props[key], "reason": "not defined in nodes.txt"})
                del props[key]
    graph["schema_property_cleanup"] = {"removed_count": len(removed), "removed": removed}

def final_clean(graph: dict[str, Any], ontology: dict[str, Any]) -> None:
    ids = {n["node_id"] for n in graph["nodes"]}
    graph["relationships"] = [r for r in graph["relationships"] if r.get("source_node_id") in ids and r.get("target_node_id") in ids]
    for n in graph["nodes"]:
        n["id"] = n["node_id"]
        n["label"] = n["node_type"]
    for r in graph["relationships"]:
        r["source_id"] = r["source_node_id"]
        r["target_id"] = r["target_node_id"]
        r["source"] = r["source_node_id"]
        r["target"] = r["target_node_id"]

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    load_environment()
    validate_environment()
    schema_text = load_schema()
    ontology = parse_ontology(
        read_text_file(NODES_SCHEMA),
        read_text_file(RELATIONSHIPS_SCHEMA),
    )
    chunks = load_chunk_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = create_client()

    print("=" * 78)
    print("FUNDSCREENING - FINAL GRAPH EXTRACTION")
    print("=" * 78)
    print(f"Node types          : {len(ontology['node_types'])}")
    print(f"Relationship types   : {len(ontology['rel_rows'])}")
    print(f"Relationship sigs    : {len(ontology['rel_allowed'])}")
    print(f"Chunks               : {len(chunks)}")

    chunk_results, failures = [], []
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk.get('chunk_id')} pages {chunk.get('start_page')}-{chunk.get('end_page')}")
        try:
            result = extract_chunk(client, chunk, schema_text)
            result["document_id"] = chunk.get("document_id")
            result["chunk_id"] = chunk.get("chunk_id")
            chunk_results.append(result)
            print(f"  nodes={len(result['nodes'])} relationships={len(result['relationships'])}")
        except Exception as exc:
            failures.append({"document_id": chunk.get("document_id"), "chunk_id": chunk.get("chunk_id"), "error": str(exc)})
            print(f"  FAILED: {exc}")

    graph = merge_results(chunk_results)
    graph = filter_to_ontology(graph, ontology)
    graph = conservative_entity_resolution(graph)
    initial = ontology_audit(graph, ontology)

    print(f"Initial graph: {len(graph['nodes'])} nodes / {len(graph['relationships'])} relationships")
    print(f"Missing node types: {initial['missing_node_types']}")

    graph, node_feedback_report = node_feedback(client, graph, chunks, ontology)
    graph = conservative_entity_resolution(graph)

    print("Running relationship-signature feedback...")
    relationship_feedback_rounds = []
    relationship_feedback_rounds.append(relationship_feedback(client, graph, chunks, ontology, only_missing=False))
    after_rel_1 = ontology_audit(graph, ontology)
    if after_rel_1["missing_relationship_types"]:
        relationship_feedback_rounds.append(relationship_feedback(client, graph, chunks, ontology, only_missing=True))
    relationship_feedback_report = {"rounds": relationship_feedback_rounds}

    print("Running instance verification / missing-edge analysis...")
    instance_report = instance_verification(client, graph, chunks, ontology)

    graph = conservative_entity_resolution(graph)
    enforce_schema_properties(graph, ontology)
    final = ontology_audit(graph, ontology)
    final_clean(graph, ontology)
    final = ontology_audit(graph, ontology)

    graph["ontology_coverage"] = {
        "node_types": {
            "required": ontology["node_types"],
            "found": final["found_node_types"],
            "unavailable_after_feedback": final["missing_node_types"],
        },
        "relationship_types": {
            "required_count": len(ontology["rel_rows"]),
            "found_count": len(final["found_relationship_types"]),
            "found": final["found_relationship_types"],
            "not_found_after_feedback": final["missing_relationship_types"],
            "note": "A missing relationship type is not treated as an error by itself. It means no evidence-backed instance was found after the bounded feedback passes. If endpoint node types were absent, the relationship is not_applicable."
        },
    }
    graph["feedback"] = {
        "node_type_feedback": node_feedback_report,
        "relationship_feedback": relationship_feedback_report,
        "instance_verification": instance_report,
    }
    graph["validation"] = {
        "initial": initial,
        "final": final,
    }
    graph["extraction_summary"] = {
        "chunks_discovered": len(chunks),
        "chunks_succeeded": len(chunk_results),
        "chunks_failed": len(failures),
        "failure_details": failures,
        "nodes_count": len(graph["nodes"]),
        "relationships_count": len(graph["relationships"]),
        "node_type_counts": final["node_type_counts"],
        "relationship_type_counts": final["relationship_type_counts"],
    }

    OUTPUT_FILE.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("FINAL GRAPH COMPLETE")
    print(f"Nodes              : {len(graph['nodes'])}")
    print(f"Relationships       : {len(graph['relationships'])}")
    print(f"Node types found    : {len(final['found_node_types'])}/8")
    print(f"Relationship types  : {len(final['found_relationship_types'])}/45")
    print(f"Unavailable nodes   : {final['missing_node_types']}")
    print(f"Unavailable rels    : {len(final['missing_relationship_types'])}")
    print(f"Output              : {OUTPUT_FILE}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
