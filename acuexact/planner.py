"""EndNote record to archive filename candidates. Similarity never approves scope."""
from difflib import SequenceMatcher
from pathlib import PurePosixPath
import re


def title_key(value):
    value = re.sub(r"^\d+[\s_.-]*", "", value)
    return re.sub(r"[\W_]+", "", value).casefold()


def plan_records(records, inventory, limit=3):
    entries = inventory.get("entries", [])
    pdfs = [e["path"] for e in entries if e["path"].lower().endswith(".pdf")]
    result = []
    for record in records:
        title = title_key(record.get("title", ""))
        link_names = {PurePosixPath(x.replace("\\", "/")).name for x in record.get("pdf_links", [])}
        scored = []
        for path in pdfs:
            candidate = PurePosixPath(path)
            score = SequenceMatcher(None, title, title_key(candidate.stem)).ratio() if title else 0
            match_type = "title_similarity"
            if candidate.name in link_names:
                score, match_type = 1.0, "endnote_attachment_basename"
            if score > .35:
                scored.append({"archive_member": path, "score": round(score, 4), "match_type": match_type})
        scored.sort(key=lambda x: (-x["score"], x["archive_member"]))
        result.append({"article": record, "boundary_status": "pending_review", "regions": [],
                       "candidates": scored[:limit],
                       "review_reason": "Confirm title/author and exact page regions against the EndNote record; filename similarity is insufficient."})
    return result
