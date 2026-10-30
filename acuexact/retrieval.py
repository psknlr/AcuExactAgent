"""Optional, evidence-first source lookup. A web match never becomes attribution.

MediaWiki API contracts: https://www.mediawiki.org/wiki/API:Search and
https://www.mediawiki.org/wiki/API:Parsing_wikitext . Search is followed by
rendered-page retrieval, because search snippets alone cannot verify a quotation.
"""
from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import json
import re
from urllib.parse import quote as urlquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

API = "https://zh.wikisource.org/w/api.php"
MAX_REQUESTS = 3
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def _allowed_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "zh.wikisource.org"
            or parsed.path != "/w/api.php" or parsed.fragment):
        raise ValueError("Source lookup only permits the zh.wikisource.org HTTPS API.")


class _FixedOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _allowed_url(newurl)
        # Do not allow redirects to silently expand the three-request budget.
        raise ValueError("Source lookup redirects are disabled; review the API endpoint.")


def _fetch_json(url, timeout):
    _allowed_url(url)
    request = Request(url, headers={
        "User-Agent": "AcuExactAgent/0.1 (https://github.com/psknlr/AcuExactAgent)",
        "Accept": "application/json",
    })
    with build_opener(_FixedOriginRedirect()).open(request, timeout=timeout) as response:
        _allowed_url(response.geturl())
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("Wikisource response exceeds the lookup size limit.")
    return json.loads(data.decode("utf-8"))


class _PlainText(HTMLParser):
    """Convert the parsed article HTML, excluding scripts, navigation and footnotes."""
    BLOCKS = {"p", "div", "section", "li", "tr", "h1", "h2", "h3", "h4", "br"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    SKIP_CLASSES = {"reference", "mw-editsection", "toc", "navbox", "noprint"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        skip = (any(item[1] for item in self.stack) or tag in {"script", "style", "nav"}
                or bool(self.SKIP_CLASSES.intersection(attributes.get("class", "").split())))
        if tag in self.BLOCKS and not skip:
            self.parts.append("\n")
        if tag not in self.VOID:
            self.stack.append((tag, skip))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag in self.BLOCKS and not any(item[1] for item in self.stack):
            self.parts.append("\n")

    def handle_data(self, data):
        if not any(item[1] for item in self.stack):
            self.parts.append(data)


def _plain_text(html):
    parser = _PlainText()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts).strip()


def _matching_evidence(text, quotation):
    """Whitespace may differ; spelling, punctuation and variants must match exactly."""
    positions = [i for i, char in enumerate(text) if not char.isspace()]
    compact = "".join(text[i] for i in positions)
    needle = re.sub(r"\s+", "", quotation)
    start = compact.find(needle)
    if start < 0:
        return None
    left, right = positions[start], positions[start + len(needle) - 1] + 1
    return text[max(0, left - 100):min(len(text), right + 100)]


def lookup_source(quote: dict, max_results=3, *, enable_network=False, fetch=None, timeout=10.0) -> dict:
    """Return a deep copy with review-only ``source_candidates``.

    Network access requires ``enable_network=True``. An injected
    ``fetch(url, timeout) -> dict`` enables deterministic offline tests. One search
    and at most two page fetches are allowed, even when max_results is larger.
    Neither source_explicit nor source_enriched is changed. A missing result is
    inconclusive: Wikisource coverage and punctuation/variant differences limit
    recall. The caller must separately review any book/chapter inferred from titles.
    """
    result = deepcopy(quote)
    result.setdefault("source_candidates", [])
    audit = {"provider": "zh.wikisource.org", "status": "disabled", "requests": 0,
             "checked_pages": 0, "match_policy": "exact_except_whitespace",
             "request_limit": MAX_REQUESTS}
    result["source_lookup"] = audit
    if not enable_network:
        return result
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        raise ValueError("max_results must be a positive integer.")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 30:
        raise ValueError("timeout must be greater than zero and at most 30 seconds.")
    result["source_candidates"] = []
    quotation = str(quote.get("text", "")).strip()
    compact = re.sub(r"\s+", "", quotation)
    audit["status"] = "pending_review"
    if len(compact) < 4:
        audit["reason"] = "quotation_too_short_for_reliable_lookup"
        return result
    if len(compact) > 12000:
        audit["reason"] = "quotation_exceeds_lookup_limit"
        return result
    fetcher = fetch or _fetch_json

    def request(**params):
        if audit["requests"] >= MAX_REQUESTS:
            raise ValueError("Lookup request budget exhausted.")
        url = API + "?" + urlencode({"format": "json", "formatversion": 2, **params})
        _allowed_url(url)
        audit["requests"] += 1
        data = fetcher(url, timeout)
        if not isinstance(data, dict) or "error" in data:
            raise ValueError("Wikisource API returned an error or invalid JSON object.")
        return data

    try:
        # Quoted search reduces unrelated hits. Full quotation verification below
        # still uses every character, including when the search prefix is shortened.
        query = '"' + compact[:120].replace('"', "") + '"'
        search = request(action="query", list="search", srsearch=query,
                         srnamespace=0, srwhat="text", srlimit=min(max_results, 2), srprop="")
        query_result = search.get("query", {})
        if not isinstance(query_result, dict):
            raise ValueError("Invalid Wikisource query response.")
        hits = query_result.get("search", [])
        if not isinstance(hits, list):
            raise ValueError("Invalid Wikisource search results.")
        audit["search_hits"] = len(hits)
        for hit in hits[:min(max_results, MAX_REQUESTS - 1)]:
            if not isinstance(hit, dict):
                continue
            pageid = hit.get("pageid")
            if isinstance(pageid, bool) or not isinstance(pageid, int) or pageid < 1:
                continue
            parsed = request(action="parse", pageid=pageid, prop="text|revid",
                             disableeditsection=1, disabletoc=1).get("parse", {})
            audit["checked_pages"] += 1
            if not isinstance(parsed, dict):
                raise ValueError("Invalid Wikisource parsed-page response.")
            html = parsed.get("text", "")
            if isinstance(html, dict):
                html = html.get("*", "")
            if not isinstance(html, str):
                continue
            evidence = _matching_evidence(_plain_text(html), quotation)
            title = parsed.get("title", hit.get("title", ""))
            if evidence is None or not isinstance(title, str) or not title.strip():
                continue
            book, separator, chapter = title.partition("/")
            candidate = {"book": book.strip(), "chapter": chapter.strip() if separator else None,
                         "url": "https://zh.wikisource.org/wiki/" + urlquote(title.replace(" ", "_"), safe="/"),
                         "evidence": evidence, "status": "candidate_requires_review",
                         "title": title, "page_id": pageid, "title_inference": True,
                         "match_type": "exact_except_whitespace"}
            revision = parsed.get("revid")
            if isinstance(revision, int) and not isinstance(revision, bool) and revision > 0:
                candidate["revision_id"] = revision
                candidate["revision_url"] = "https://zh.wikisource.org/w/index.php?oldid=" + str(revision)
            result["source_candidates"].append(candidate)
        audit["status"] = "candidates_require_review" if result["source_candidates"] else "pending_review"
        audit["reason"] = "candidate_titles_need_bibliographic_review" if result["source_candidates"] else "no_exact_match_in_checked_pages"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        audit["status"] = "lookup_error_requires_review"
        audit["error"] = str(exc)[:300]
    return result
