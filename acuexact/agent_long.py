"""Read long, continuous articles in bounded page groups without losing provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .agent import (CLASSIFY_SYSTEM, _class_schema, _model, _validate_classification,
                    _title_key, run_pdf_agent)
from .extraction import extract_quotes
from .export import write_result
from .pdf import file_sha256


def merge_parts(parts):
    """Keep each occurrence; lift local offsets and region IDs into one document."""
    merged = copy.deepcopy(parts[0])
    merged.update(quotes=[], excluded=[], segments=[], warnings=[])
    merged["audit"].update(selected_pages=[], regions=[], model_calls=[], parts=[], calls_used=0)
    merged["audit"].pop("classification_id_aliases", None)
    cursor, boundaries = 0, []
    for number, part in enumerate(parts, 1):
        prefix = f"part-{number:04d}:"
        if number > 1:
            boundaries.append(cursor)
        source = copy.deepcopy(part)
        for segment in source.get("segments", []):
            segment["region_id"] = prefix + segment["region_id"]
            merged["segments"].append(segment)
        for kind in ("quotes", "excluded"):
            for quote in source.get(kind, []):
                quote["part_quote_id"] = quote["id"]
                quote["article_id"] = merged["article"]["id"]
                visual = quote.get("offset_space", "").startswith("vision_transcription:")
                if visual:
                    quote["offset_space"] = "vision_transcription:" + prefix + quote["offset_space"].partition(":")[2]
                else:
                    quote["char_start"] += cursor
                    quote["char_end"] += cursor
                for evidence in quote.get("evidence", []):
                    if "region_id" in evidence:
                        evidence["region_id"] = prefix + evidence["region_id"]
                    if evidence.get("offset_space", "").startswith("vision_transcription:"):
                        evidence["offset_space"] = quote["offset_space"]
                    elif not visual:
                        for key in ("char_start", "char_end"):
                            if key in evidence:
                                evidence[key] += cursor
                merged[kind].append(quote)
        for warning in source.get("warnings", []):
            if warning.get("code") in {"unclosed_quote", "unmatched_closing_quote"}:
                continue
            if "region_id" in warning:
                warning["region_id"] = prefix + warning["region_id"]
            if "offset" in warning:
                warning["offset"] += cursor
            merged["warnings"].append(dict(warning, part=number))
        for region in source["audit"].get("regions", []):
            if "id" in region:
                region["id"] = prefix + region["id"]
            merged["audit"]["regions"].append(region)
        merged["audit"]["selected_pages"].extend(source["audit"]["selected_pages"])
        merged["audit"]["model_calls"].extend(source["audit"].get("model_calls", []))
        merged["audit"]["calls_used"] += source["audit"].get("calls_used", 0)
        merged["audit"]["parts"].append({"part": number, "pages": source["audit"]["selected_pages"],
            "status": source["status"], "classification_id_aliases": source["audit"].get("classification_id_aliases", [])})
        cursor += sum(len(s["text"]) + 1 for s in source.get("segments", []))
    # Re-extract across joins so quotation marks on opposite sides of a group
    # boundary are paired exactly as they would be for a short document.
    joined = extract_quotes(merged["segments"], merged["article"])
    merged["warnings"].extend(w for w in joined["warnings"] if w.get("code") in {"unclosed_quote", "unmatched_closing_quote"})
    for kind in ("quotes", "excluded"):
        additions = [q for q in joined[kind] if any(q["char_start"] < b < q["char_end"] for b in boundaries)]
        for quote in additions:
            for existing in ("quotes", "excluded"):
                merged[existing] = [old for old in merged[existing] if old.get("offset_space", "").startswith("vision_transcription:")
                    or not (quote["char_start"] <= old["char_start"] and old["char_end"] <= quote["char_end"])]
            quote["review_flags"].extend(["cross_part_quote_requires_review", "automatic_article_boundary_requires_review"])
            quote["source_enriched"] = {"book": None, "chapter": None, "evidence": None, "url": None, "status": "待查证"}
            merged[kind].append(quote)
    merged["quotes"].sort(key=lambda q: (min((e.get("page", 0) for e in q.get("evidence", [])), default=0), q["char_start"]))
    for index, quote in enumerate(merged["quotes"] + merged["excluded"], 1):
        quote["id"] = f"{merged['article']['id']}:q{index:04d}"
    merged["audit"].update(retained=len(merged["quotes"]), excluded=len(merged["excluded"]),
                              long_document_parts=len(parts))
    merged["status"] = "needs_review"
    return merged


def run_long_pdf_agent(pdf_path, prompt, client, *, total_pages, directory, target_title,
                       max_calls=80, on_event=None, cancelled=None, pages_per_part=10):
    from .agent_batch import pipeline_complete
    parts, merged = [], None
    source_digest = file_sha256(pdf_path)
    directory = Path(directory)
    for start in range(1, total_pages + 1, pages_per_part):
        if cancelled and cancelled():
            break
        number = len(parts) + 1
        destination = directory / "parts" / f"part-{number:04d}"
        cached = destination / "result.json"
        pages = list(range(start, min(start + pages_per_part, total_pages + 1)))
        part = json.loads(cached.read_text(encoding="utf-8")) if cached.exists() else None
        valid_cache = (part and pipeline_complete(part) and part["audit"].get("selected_pages") == pages
            and part["audit"].get("pdf_sha256") == source_digest
            and part["audit"].get("research_prompt") == prompt
            and _title_key(part["article"]["title"]) == _title_key(merged["article"]["title"] if merged else target_title)
            and all(c.get("model") == _model(client) for c in part["audit"].get("model_calls", [])))
        if not valid_cache:
            if part and not pipeline_complete(part):
                for checkpoint in (directory / "checkpoints").glob("*.json"):
                    value = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if ("regions" in value and not part.get("audit", {}).get("regions")) or "results" in value or "missing_quotes" in value:
                        checkpoint.unlink()
            part = run_pdf_agent(pdf_path, prompt, client, target_title=merged["article"]["title"] if merged else target_title,
                page_numbers=pages, max_pages=20, max_calls=max_calls, transcribe_pages=False,
                transcribe_clean_regions=False, image_plan_clean_pages=False,
                continuation_of=merged, on_event=on_event, cancelled=cancelled)
            write_result(part, destination)
        parts.append(part)
        merged = merge_parts(parts)
        if not pipeline_complete(part):
            break
    if merged is None:
        raise ValueError("Long-document processing cancelled before the first part")
    complete = len(merged["audit"]["selected_pages"]) == total_pages and all(pipeline_complete(p) for p in parts)
    if not complete:
        merged["warnings"].append({"code": "agent_incomplete", "detail": "长文献尚未完成全部页面；已读证据保留。"})
    else:
        dimensions = merged["audit"]["classification_dimensions"]
        pending = [q for q in merged["quotes"] if "relevance" not in q]
        for start in range(0, len(pending), 20):
            batch = pending[start:start + 20]
            aliases = {f"q{index + 1}": quote for index, quote in enumerate(batch)}
            merged["audit"].setdefault("joined_classification_id_aliases", []).append(
                {alias: quote["id"] for alias, quote in aliases.items()})
            call_start = len(client.calls)
            try:
                if cancelled and cancelled():
                    raise ValueError("Cancelled")
                payload = {"user_prompt": prompt, "article": merged["article"], "dimensions": dimensions,
                    "quotes": [dict(id=alias, text=q["text"], source_explicit=q["source_explicit"])
                               for alias, q in aliases.items()]}
                response = client.generate_json(CLASSIFY_SYSTEM, payload, _class_schema(dimensions, aliases), schema_name="classify_joined_quotes")
                mapping = _validate_classification(response, aliases, dimensions)
                for alias, q in aliases.items():
                    q["relevance"] = dict(mapping[alias], method="llm_agent", model=_model(client))
                    q["relevance"].pop("id", None)
            except Exception:
                merged["warnings"].append({"code": "agent_incomplete", "detail": "跨段引文尚未完成模型分类。"})
            finally:
                for call in client.calls[call_start:]:
                    safe = {k: call[k] for k in ("provider", "model", "status", "usage", "request_id", "http_status", "attempt", "repair") if k in call}
                    merged["audit"]["model_calls"].append(dict(safe, stage="classify_joined_quotes"))
                merged["audit"]["calls_used"] += len(client.calls) - call_start
    for q in merged["quotes"]:
        if "relevance" not in q:
            q["relevance"] = {"label": "待分类", "dimensions": [], "reason": "跨段引文需续跑分类。",
                              "method": "unclassified", "needs_review": True}
    joined_count = sum("cross_part_quote_requires_review" in q["review_flags"] for q in merged["quotes"])
    merged["audit"].update(call_budget_per_part=max_calls,
        call_budget=max_calls * len(parts) + 3 * ((joined_count + 19) // 20))
    return merged
