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

# Bounded semantic repair. Values are configuration, not graph-size quotas.
SEMANTIC_REPAIR_MAX_ATTEMPTS = int(os.getenv("GRAPH_SEMANTIC_REPAIR_MAX_ATTEMPTS", "3"))
SEMANTIC_REPAIR_MAX_CHUNKS_PER_TYPE = int(os.getenv("GRAPH_SEMANTIC_REPAIR_MAX_CHUNKS_PER_TYPE", "8"))
SEMANTIC_REPAIR_MAX_TOTAL_CHUNKS = int(os.getenv("GRAPH_SEMANTIC_REPAIR_MAX_TOTAL_CHUNKS", "24"))


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
# SCHEMA-DRIVEN SEMANTIC COMPLETENESS REPAIR
# ---------------------------------------------------------------------------

def parse_base_node_types(nodes_text: str) -> list[str]:
    """Read required/base node types from the schema; never hard-code them."""
    found=[]; active=False
    for raw in nodes_text.splitlines():
        line=raw.strip()
        if not active:
            if re.search(r"FINAL\s+GRAPH\s+SCHEMA\s*[—-]\s*BASE\s+NODES", line, re.I):
                active=True
            continue
        if line and re.match(r"^(?:FINAL\s+GRAPH\s+SCHEMA|ALLOWED\s+RELATIONSHIPS)", line, re.I):
            break
        m=re.match(r"^(?:\d+\.\s*|[-*]\s*)\**([A-Za-z][A-Za-z0-9_]*)\**\s*$", line)
        if m and m.group(1) not in found:
            found.append(m.group(1))
    if not found:
        raise SystemExit("Could not discover base node types from nodes.txt.")
    return found


def parse_allowed_node_types(nodes_text: str) -> set[str]:
    result=set()
    for raw in nodes_text.splitlines():
        m=re.match(r"^(?:\d+\.\s*|[-*]\s*)\**([A-Za-z][A-Za-z0-9_]*)\**\s*$", raw.strip())
        if m and m.group(1) not in {"Properties","Attributes"}:
            result.add(m.group(1))
    return result


def parse_allowed_relationship_types(rels_text: str) -> set[str]:
    result=set()
    for raw in rels_text.splitlines():
        m=re.match(r"^[-*]\s*([A-Z][A-Z0-9_]*)\s*$", raw.strip())
        if m: result.add(m.group(1))
    return result


def missing_base_node_types(nodes: list[dict[str,Any]], base_types: list[str]) -> list[str]:
    present={str(n.get("node_type")) for n in nodes if isinstance(n,dict) and n.get("node_type")}
    return [t for t in base_types if t not in present]


def schema_terms_for_type(node_type: str, nodes_text: str) -> set[str]:
    """Derive retrieval terms from the schema instead of document-specific keywords."""
    terms=set(re.findall(r"[a-z0-9]+", re.sub(r"(?<!^)(?=[A-Z])", " ", node_type).lower()))
    lines=nodes_text.splitlines(); active=False
    for raw in lines:
        line=raw.strip()
        if re.match(rf"^(?:\d+\.\s*|[-*]\s*)\**{re.escape(node_type)}\**\s*$", line):
            active=True; continue
        if active:
            if re.match(r"^(?:\d+\.\s*|[-*]\s*)\**[A-Za-z][A-Za-z0-9_]*\**\s*$", line): break
            for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", line.lower()):
                if w not in {"properties","attributes","description","optional","required","string","number","boolean","list"}:
                    terms.add(w.replace("_"," "))
    return {t for t in terms if len(t)>=3}


def select_repair_chunks(chunks, missing_types, nodes_text):
    selected={}
    for typ in missing_types:
        terms=schema_terms_for_type(typ,nodes_text)
        scored=[]
        for idx,ch in enumerate(chunks):
            body=str(ch.get("text","")).lower(); score=0
            for term in terms:
                hits=len(re.findall(rf"\b{re.escape(term)}\b",body))
                score += min(hits,10)
            scored.append((score,idx,ch))
        scored.sort(key=lambda x:(-x[0],x[1]))
        positive=[x[2] for x in scored if x[0]>0]
        selected[typ]=(positive or [x[2] for x in scored[:2]])[:SEMANTIC_REPAIR_MAX_CHUNKS_PER_TYPE]
    return selected


SEMANTIC_REPAIR_SYSTEM_PROMPT = r"""
You are a targeted semantic repair engine for a document graph extractor.
The supplied nodes.txt and relationships.txt are authoritative.

Find only evidence-backed instances of the missing base node types. The
number of nodes is NOT fixed. Do not create a node merely because its type is
missing. Do not create placeholders, generic/unknown entities, or inferred
facts. If the supplied text does not explicitly support a type, return no node
for that type.

Use only predefined properties for each node type. Preserve source wording,
values, units and context. Reuse an existing node_id when the same entity is
already present. Every node needs a short verbatim evidence quote.

Relationships are optional in this repair pass. Only return a relationship
when its type and endpoint signature are allowed and explicitly supported.

Return JSON only in this shape:
{"nodes":[{"node_id":"...","node_type":"...","properties":{},"evidence":[{"document_id":"...","source_file":"...","page":1,"chunk_id":"...","quote":"..."}]}],"relationships":[],"warnings":[]}
""".strip()


def build_repair_prompt(missing_types, chunks, schema_text, existing_nodes):
    existing=[{"node_id":n.get("node_id"),"node_type":n.get("node_type"),"properties":n.get("properties",{})} for n in existing_nodes]
    source=[]
    for ch in chunks:
        source.append(
            f"SOURCE CHUNK\ndocument_id: {ch.get('document_id')}\nsource_file: {ch.get('source_file')}\nchunk_id: {ch.get('chunk_id')}\npage: {ch.get('start_page')}\nTEXT:\n{ch.get('text','')}\nEND SOURCE CHUNK"
        )
    return ("MISSING BASE NODE TYPES\n"+json.dumps(missing_types,ensure_ascii=False,indent=2)+
            "\n\nSCHEMA\n"+schema_text+
            "\n\nEXISTING NODES\n"+json.dumps(existing,ensure_ascii=False,indent=2)+
            "\n\nSOURCE CHUNKS\n"+"\n\n".join(source)+
            "\n\nExtract only explicitly supported missing node instances. The graph size must remain document-driven.")


def call_semantic_repair(client, prompt):
    last=None
    for attempt in range(1,MAX_RETRIES+1):
        try:
            response=client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type":"json_object"},
                messages=[{"role":"system","content":SEMANTIC_REPAIR_SYSTEM_PROMPT},{"role":"user","content":prompt}],
            )
            return parse_json_response(response.choices[0].message.content or "")
        except Exception as exc:
            last=exc
            if attempt==MAX_RETRIES: break
            time.sleep(RETRY_BASE_SECONDS*(2**(attempt-1)))
    raise RuntimeError(f"Semantic repair failed: {last}")


def conservative_merge_repair(graph, repair):
    nodes=graph.setdefault("nodes",[]); rels=graph.setdefault("relationships",[])
    by_id={str(n.get("node_id")):n for n in nodes if isinstance(n,dict) and n.get("node_id")}
    # Only merge by an explicit stable identity property; otherwise keep the
    # model's stable id. This avoids global fuzzy canonicalization.
    identity_fields=("name","fund_name","legal_name","deal_name","criterion_id","finding_id","metric_name")
    by_identity={}
    for n in nodes:
        p=n.get("properties",{}) if isinstance(n,dict) else {}
        if not isinstance(p,dict): continue
        vals=[str(p[k]).strip().lower() for k in identity_fields if p.get(k) not in (None,"",[])]
        if vals: by_identity[(str(n.get("node_type")),"|".join(vals))]=n
    redirects={}
    for n in repair.get("nodes",[]):
        if not isinstance(n,dict) or not n.get("node_id") or not n.get("node_type"): continue
        nid=str(n["node_id"]); existing=by_id.get(nid)
        p=n.get("properties",{}) if isinstance(n.get("properties",{}),dict) else {}
        vals=[str(p[k]).strip().lower() for k in identity_fields if p.get(k) not in (None,"",[])]
        if existing is None and vals:
            existing=by_identity.get((str(n["node_type"]),"|".join(vals)))
            if existing: redirects[nid]=str(existing.get("node_id"))
        if existing is None:
            clean={"node_id":nid,"node_type":str(n["node_type"]),"properties":p,"evidence":n.get("evidence",[])}
            nodes.append(clean); by_id[nid]=clean
            if vals: by_identity[(str(n["node_type"]),"|".join(vals))]=clean
        else:
            for k,v in p.items():
                if existing.setdefault("properties",{}).get(k) in (None,"",[]): existing["properties"][k]=v
            existing.setdefault("evidence",[]).extend(n.get("evidence",[]))
    rel_by_id={str(r.get("relationship_id")):r for r in rels if isinstance(r,dict) and r.get("relationship_id")}
    for r in repair.get("relationships",[]):
        if not isinstance(r,dict): continue
        s=redirects.get(str(r.get("source_node_id")),str(r.get("source_node_id","")))
        t=redirects.get(str(r.get("target_node_id")),str(r.get("target_node_id","")))
        if not s or not t or s not in by_id or t not in by_id: continue
        rr=dict(r); rr["source_node_id"]=s; rr["target_node_id"]=t
        rid=str(rr.get("relationship_id",""))
        if not rid: continue
        if rid not in rel_by_id:
            rels.append(rr); rel_by_id[rid]=rr
    for n in nodes: n["evidence"]=deduplicate_evidence(n.get("evidence",[]))
    for r in rels: r["evidence"]=deduplicate_evidence(r.get("evidence",[]))
    return graph


def run_semantic_completeness_repair(client, graph, chunks, nodes_text, rels_text):
    base_types=parse_base_node_types(nodes_text)
    allowed_nodes=parse_allowed_node_types(nodes_text)
    allowed_rels=parse_allowed_relationship_types(rels_text)
    schema_text="=== ALLOWED NODE TYPES AND PROPERTIES ===\n"+nodes_text+"\n\n=== ALLOWED RELATIONSHIP TYPES ===\n"+rels_text
    history=[]
    for attempt in range(1,SEMANTIC_REPAIR_MAX_ATTEMPTS+1):
        missing=missing_base_node_types(graph.get("nodes",[]),base_types)
        if not missing: break
        selected=select_repair_chunks(chunks,missing,nodes_text)
        chosen=[]; seen=set()
        for typ in missing:
            for ch in selected.get(typ,[]):
                key=str(ch.get("chunk_id",ch.get("_chunk_file","")))
                if key in seen: continue
                seen.add(key); chosen.append(ch)
                if len(chosen)>=SEMANTIC_REPAIR_MAX_TOTAL_CHUNKS: break
            if len(chosen)>=SEMANTIC_REPAIR_MAX_TOTAL_CHUNKS: break
        before=len(graph.get("nodes",[]))
        try:
            repair=call_semantic_repair(client,build_repair_prompt(missing,chosen,schema_text,graph.get("nodes",[])))
            repair["nodes"]=[n for n in repair.get("nodes",[]) if isinstance(n,dict) and n.get("node_type") in allowed_nodes]
            repair["relationships"]=[r for r in repair.get("relationships",[]) if isinstance(r,dict) and r.get("relationship_type") in allowed_rels]
            graph=conservative_merge_repair(graph,repair)
            after=len(graph.get("nodes",[])); missing_after=missing_base_node_types(graph.get("nodes",[]),base_types)
            history.append({"attempt":attempt,"missing_before":missing,"selected_chunk_count":len(chosen),"nodes_added":after-before,"missing_after":missing_after})
        except Exception as exc:
            history.append({"attempt":attempt,"missing_before":missing,"selected_chunk_count":len(chosen),"error":str(exc),"missing_after":missing})
    missing=missing_base_node_types(graph.get("nodes",[]),base_types)
    completeness={"required_base_node_types":base_types,"found_base_node_types":[t for t in base_types if t not in missing],"missing_base_node_types":missing,"required_base_node_type_count":len(base_types),"found_base_node_type_count":len(base_types)-len(missing),"complete":not missing,"semantic_repair_attempts":len(history),"semantic_repair_history":history,"note":"Base types are ontology requirements, not node-count quotas. No placeholder nodes are created."}
    graph["base_node_completeness"]=completeness
    return graph,completeness


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    load_environment()
    validate_environment()

    nodes_schema_text = read_text_file(NODES_SCHEMA)
    relationships_schema_text = read_text_file(RELATIONSHIPS_SCHEMA)
    schema_text = (
        "=== ALLOWED NODE TYPES AND PROPERTIES ===\n"
        f"{nodes_schema_text}\n\n"
        "=== ALLOWED RELATIONSHIP TYPES ===\n"
        f"{relationships_schema_text}"
    )
    # Validate schema structure before any extraction calls.
    base_node_types = parse_base_node_types(nodes_schema_text)
    allowed_node_types = parse_allowed_node_types(nodes_schema_text)
    allowed_relationship_types = parse_allowed_relationship_types(relationships_schema_text)
    if any(t not in allowed_node_types for t in base_node_types):
        raise SystemExit("nodes.txt is inconsistent: a base node type is not in the node vocabulary.")
    if not allowed_relationship_types:
        raise SystemExit("Could not parse relationship types from relationships.txt.")

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

    # Targeted semantic completeness repair. This reruns only the missing
    # base types, with a hard upper bound, and never changes the document-driven
    # number of instances into a fixed quota.
    final_graph, completeness = run_semantic_completeness_repair(
        client, final_graph, chunks, nodes_schema_text, relationships_schema_text
    )

    # Filter relationships that reference non-existent nodes (schema-strict mode).
    # Do NOT create placeholder nodes; instead, discard orphaned relationships.
    existing_ids = {n.get("node_id") for n in final_graph.get("nodes", [])}
    
    orphaned_rels = []
    filtered_relationships = []
    
    for r in final_graph.get("relationships", []):
        source_id = r.get("source_node_id") or r.get("source_id") or r.get("source")
        target_id = r.get("target_node_id") or r.get("target_id") or r.get("target")
        
        # Keep relationship only if both endpoints exist in the node set.
        if source_id and target_id and source_id in existing_ids and target_id in existing_ids:
            filtered_relationships.append(r)
        else:
            orphaned_rels.append({
                "rel_type": r.get("relationship_type", "unknown"),
                "source": source_id,
                "target": target_id,
                "reason": f"source_exists={source_id in existing_ids}, target_exists={target_id in existing_ids}"
            })
    
    if orphaned_rels:
        print(f"\n[WARNING] Filtered {len(orphaned_rels)} orphaned relationship(s):")
        for orphaned in orphaned_rels[:10]:  # Show first 10
            print(f"  {orphaned['rel_type']}: {orphaned['source']} -> {orphaned['target']} ({orphaned['reason']})")
        if len(orphaned_rels) > 10:
            print(f"  ... and {len(orphaned_rels) - 10} more")
    
    final_graph["relationships"] = filtered_relationships

    # Add loader-friendly alias fields for broader compatibility.
    for n in final_graph.get("nodes", []):
        if "node_id" in n and "id" not in n:
            n["id"] = n.get("node_id")
        if "node_type" in n and "label" not in n:
            n["label"] = n.get("node_type")

    for r in final_graph.get("relationships", []):
        # source/target aliases
        if "source_node_id" in r and "source_id" not in r:
            r["source_id"] = r.get("source_node_id")
        if "target_node_id" in r and "target_id" not in r:
            r["target_id"] = r.get("target_node_id")
        if "source_node_id" in r and "source" not in r:
            r["source"] = r.get("source_node_id")
        if "target_node_id" in r and "target" not in r:
            r["target"] = r.get("target_node_id")

    final_graph["extraction_summary"] = {
        "chunks_discovered": len(chunks),
        "chunks_succeeded": len(chunk_results),
        "chunks_failed": len(failures),
        "failure_details": failures,
        "nodes_count": len(final_graph["nodes"]),
        "relationships_count": len(final_graph["relationships"]),
        "base_node_completeness": completeness,
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
    print(f"Final nodes      : {len(final_graph['nodes'])}")
    print(f"Final relationships: {len(final_graph['relationships'])}")
    print(
        "Base node coverage: "
        f"{completeness['found_base_node_type_count']}/"
        f"{completeness['required_base_node_type_count']}"
    )
    if completeness["missing_base_node_types"]:
        print("Base types still missing (no placeholders): " + ", ".join(completeness["missing_base_node_types"]))
    print(f"Output           : {OUTPUT_FILE}")

    if failures:
        print()
        print(
            "The failed chunks are recorded in extraction_summary.failure_details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()