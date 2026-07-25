#!/usr/bin/env python3
"""Export OpenAlex candidates and a reproducible screening ledger."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "experiments/derived/information_set_literature_audit.json"
OUTPUT = ROOT / "experiments/derived/literature_screening"
PER_PAGE = 50


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def fetch(url: str) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "resolution-audit/3.0.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    audit = json.loads(AUDIT.read_text())
    included_titles = {
        normalize(record["title"]): record["key"] for record in audit["records"]
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = {}
    query_records = []
    for index, query in enumerate(audit["queries"], start=1):
        parameters = urllib.parse.urlencode(
            {
                "search": query,
                "per-page": PER_PAGE,
                "select": (
                    "id,doi,title,publication_year,type,primary_location,"
                    "cited_by_count"
                ),
            }
        )
        url = f"https://api.openalex.org/works?{parameters}"
        payload = fetch(url)
        raw_path = OUTPUT / f"query_{index:02d}.json"
        raw_path.write_text(json.dumps(payload, indent=2) + "\n")
        query_records.append(
            {
                "query_id": index,
                "query": query,
                "url": url,
                "returned": len(payload.get("results", [])),
                "raw_path": str(raw_path.relative_to(ROOT)),
            }
        )
        for record in payload.get("results", []):
            candidates[record["id"]] = record
    screening = []
    matched_keys = set()
    for identifier, record in sorted(candidates.items()):
        title = record.get("title") or ""
        key = included_titles.get(normalize(title))
        if key:
            decision = "included"
            reason = "Primary mechanism/protocol exemplar in the fixed 12-work audit."
            matched_keys.add(key)
        else:
            decision = "excluded"
            reason = (
                "Title/metadata screening: outside the fixed primary "
                "disagreement, TTA, EO OOD, multi-resolution, or spatial-audit "
                "exemplar set."
            )
        screening.append(
            {
                "openalex_id": identifier,
                "doi": record.get("doi"),
                "title": title,
                "publication_year": record.get("publication_year"),
                "type": record.get("type"),
                "decision": decision,
                "included_key": key,
                "reason": reason,
            }
        )
    unmatched = sorted(set(included_titles.values()) - matched_keys)
    direct_records = []
    for index, record in enumerate(audit["records"], start=1):
        parameters = urllib.parse.urlencode(
            {
                "search": record["title"],
                "per-page": 5,
                "select": (
                    "id,doi,title,publication_year,type,primary_location,"
                    "cited_by_count"
                ),
            }
        )
        url = f"https://api.openalex.org/works?{parameters}"
        payload = fetch(url)
        raw_path = OUTPUT / f"included_{index:02d}.json"
        raw_path.write_text(json.dumps(payload, indent=2) + "\n")
        match = next(
            (
                candidate
                for candidate in payload.get("results", [])
                if normalize(candidate.get("title") or "")
                == normalize(record["title"])
            ),
            None,
        )
        direct_records.append(
            {
                "key": record["key"],
                "query_url": url,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "openalex_match": match,
                "matched": match is not None,
            }
        )
    result = {
        "status": "post-hoc reproducibility strengthening",
        "source": "OpenAlex API",
        "accessed_at": "2026-07-25",
        "queries": query_records,
        "unique_candidates": len(candidates),
        "included_matches": len(matched_keys),
        "included_records_not_returned_by_top50_search": unmatched,
        "direct_included_lookups": direct_records,
        "direct_included_matches": sum(row["matched"] for row in direct_records),
        "screening_path": "experiments/derived/literature_screening/screening.jsonl",
        "limitation": (
            "Top-50 relevance-ranked OpenAlex exports per query; not an "
            "exhaustive systematic review or prevalence estimator."
        ),
    }
    (OUTPUT / "screening.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in screening)
    )
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LITERATURE_SCREENING_EXPORTED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
