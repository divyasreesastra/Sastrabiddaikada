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
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    load_environment()
    validate_environment()

    schema_text = load_schema()
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
    print(f"Output           : {OUTPUT_FILE}")

    if failures:
        print()
        print(
            "The failed chunks are recorded in extraction_summary.failure_details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
