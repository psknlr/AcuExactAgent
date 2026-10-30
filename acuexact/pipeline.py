"""Explicit stage orchestration with immutable extraction and separate enrichment."""
import copy
import json
from pathlib import Path

from .classification import classify_quote, enrich_source
from .extraction import extract_quotes
from .pdf import read_regions, file_sha256, text_quality_flags


def analyze(segments, article, *, use_llm=False, model=None, catalog=None, disease_names=None, lookup=False):
    scoped = copy.deepcopy(segments)
    for segment in scoped:
        segment.setdefault("quality_flags", text_quality_flags(segment["text"]))
    result = extract_quotes(scoped, article, disease_names=disease_names)
    quotes = []
    for original in result["quotes"]:
        quote = classify_quote(original, use_llm=use_llm, model=model)
        if quote["text"] != original["text"] or quote["source_explicit"] != original["source_explicit"]:
            raise ValueError("Classification attempted to alter extraction evidence.")
        quote = enrich_source(quote, catalog or [])
        if lookup and (not quote["source_explicit"].get("book") or not quote["source_explicit"].get("chapter")):
            from .retrieval import lookup_source
            quote = lookup_source(quote, enable_network=True)
        quotes.append(quote)
    result["quotes"] = quotes
    if any(q.get("relevance", {}).get("needs_review") for q in quotes):
        result["status"] = "needs_review"
    result["audit"]["stages"] = ["scoped_input", "extract_all_quotes", "classify_without_filtering", "enrich_separate_fields", "export"]
    result["audit"]["classification_mode"] = "llm" if use_llm else "rules"
    result["audit"]["gold_used_in_inference"] = False
    result["audit"]["source_lookup_enabled"] = lookup
    return result


def analyze_pdf(pdf, manifest, **kwargs):
    segments = read_regions(pdf, manifest, ocr=kwargs.pop("ocr", False))
    result = analyze(segments, manifest["article"], **kwargs)
    result["audit"].update(pdf_sha256=file_sha256(pdf), regions=manifest["regions"],
                           boundary_status="approved", boundary_reviewer=manifest.get("reviewer", "unspecified"))
    result["segments"] = segments
    return result


def demo():
    sample = json.loads((Path(__file__).parent / "data" / "demo.json").read_text(encoding="utf-8"))
    result = analyze(sample["segments"], sample["article"])
    result["audit"].update(demo=True, synthetic=True, boundary_status="synthetic_approved_regions",
                           outside_article_segments=len(sample["outside_article_segments"]),
                           disclaimer="合成测试文本，非古籍校勘材料；离线规则输出不代表人工金标准性能。")
    result["demo_input"] = "\n".join(x["text"] for x in sample["segments"])
    return result
