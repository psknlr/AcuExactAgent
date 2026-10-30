"""Deterministic quotation extraction. No model may rewrite extracted characters."""
from __future__ import annotations

import hashlib
import re

DISEASE_NAMES = frozenset({"肩痛", "肩不举", "肩不舉"})
PAIRS = {"“": "”", "「": "」", "『": "』"}


def normalize_layout(text: str) -> str:
    """Remove line wrapping, keeping spelling, punctuation, variant characters and spaces."""
    return re.sub(r"[\r\n]+", "", text).strip()


def explicit_source(text: str, start: int, end: int) -> dict:
    """Bind only local syntactic attributions; never supply an unstated book/chapter."""
    empty = {"book": None, "chapter": None, "evidence": None}
    before = text[max(0, start - 240):start]
    after = text[end:min(len(text), end + 140)]
    # A sentence boundary cuts off unrelated previous mentions. A title may contain a dot.
    before = re.split(r"[。！？]", before)[-1]
    pattern = r"《([^》]+)》(?:\s*(?:中|有|曾|亦|又|则|则有|记|记载|记述|载|云|曰|言|谓|说|指出|认为|提到|记曰|云曰|论述|明确指出|所言|所说|所云|定位为)){0,5}\s*[:：,，]?\s*$"
    match = re.search(pattern, before)
    if not match:
        # A bare title after a quote may introduce the NEXT quote; require attribution syntax.
        match = (re.match(r"\s*[（(]\s*(?:见|出自|引自)?\s*《([^》]+)》\s*[）)]", after)
                 or re.match(r"\s*(?:出自|引自|见|——)\s*《([^》]+)》", after))
    if not match:
        return empty
    title = match.group(1).strip()
    parts = re.split(r"[·•‧・]", title, maxsplit=1)
    return {"book": parts[0], "chapter": parts[1] if len(parts) > 1 else None,
            "evidence": match.group(0).strip()}


def extract_quotes(segments: list[dict], article: dict, *, disease_names=None) -> dict:
    """Segments are ordered, already scoped to ONE article, with page/bbox/text evidence.

    Offsets address the joined, unmodified text. Nested quotes stay inside their parent;
    duplicate occurrences remain separate. Unbalanced delimiters are never fabricated.
    """
    excluded_names = DISEASE_NAMES if disease_names is None else frozenset(disease_names)
    full, spans = "", []
    for index, segment in enumerate(segments):
        if index:
            full += "\n"
        begin = len(full)
        full += segment["text"]
        spans.append((begin, len(full), segment))
    stack, pairs, warnings = [], [], []
    for segment in segments:
        for flag in segment.get("quality_flags", []):
            warnings.append({"code": flag, "page": segment.get("page"), "region_id": segment.get("region_id")})
    for i, char in enumerate(full):
        if char in PAIRS:
            stack.append((char, i))
        elif char in PAIRS.values():
            if stack and PAIRS[stack[-1][0]] == char:
                opening, begin = stack.pop()
                pairs.append((begin, i + 1, opening))
            else:
                warnings.append({"code": "unmatched_closing_quote", "offset": i, "character": char})
    for opening, begin in stack:
        warnings.append({"code": "unclosed_quote", "offset": begin, "character": opening,
                         "excerpt": full[begin:begin + 160]})
    # Keep outermost balanced spans, including a balanced span inside a damaged/unclosed one.
    outer, enclosing_end = [], -1
    for pair in sorted(pairs, key=lambda p: (p[0], -p[1])):
        if pair[0] >= enclosing_end:
            outer.append(pair)
            enclosing_end = pair[1]
    pairs = outer
    quotes, excluded = [], []
    for begin, end, opening in pairs:
        raw = full[begin + 1:end - 1]
        clean = normalize_layout(raw)
        evidence = []
        for lo, hi, segment in spans:
            if lo < end and hi > begin:
                ev = {k: segment[k] for k in ("page", "bbox", "region_id", "method") if k in segment}
                ev.update(text=segment["text"][max(0, begin - lo):min(hi, end) - lo],
                          char_start=max(begin, lo), char_end=min(end, hi))
                evidence.append(ev)
        quote = {"id": f"{article['id']}:q{len(quotes) + len(excluded) + 1:04d}",
                 "article_id": str(article["id"]), "text": clean, "text_raw": raw,
                 "delimiter": opening + PAIRS[opening], "char_start": begin, "char_end": end,
                 "source_explicit": explicit_source(full, begin, end), "evidence": evidence,
                 "review_flags": []}
        if any(unclosed_start < begin for _, unclosed_start in stack):
            quote["review_flags"].append("recovered_with_unclosed_quote_context")
        if not clean:
            quote["review_flags"].append("empty_quote")
        if any(s.get("method", "") == "ocr" for s in evidence):
            quote["review_flags"].append("ocr_requires_visual_verification")
        if any(c in raw for c in ("�", "\ufffd")) or "(cid:" in raw:
            quote["review_flags"].append("suspect_pdf_characters")
        if quote["source_explicit"]["book"] is None:
            quote["review_flags"].append("source_not_explicit")
        if clean in excluded_names:
            quote["exclusion_reason"] = "exact_disease_name_allowlist"
            excluded.append(quote)
        else:
            quotes.append(quote)
    if not pairs:
        warnings.append({"code": "no_balanced_quotes", "detail": "Verify the rendered article; absence of text-layer quotes is not proof of absence."})
    return {"schema_version": "1.0", "article": article, "quotes": quotes, "excluded": excluded,
            "status": "needs_review" if warnings or any(q["review_flags"] for q in quotes) else "extracted",
            "warnings": warnings, "audit": {"engine": "balanced-delimiters-v1",
            "text_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest(),
            "total_balanced_pairs": len(pairs), "retained": len(quotes), "excluded": len(excluded),
            "disease_allowlist": sorted(excluded_names), "normalization": "strip outer whitespace; remove CR/LF only"}}
