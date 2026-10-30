"""UTF-8 BOM TSV for direct Excel paste/import. JSON remains lossless."""
import csv
import io
import json
from pathlib import Path

HEADERS = ["条号", "文章ID", "古籍条文", "强弱相关类别", "相关维度", "归类理由",
           "原文明示书名", "原文明示篇章", "查证补全书名", "查证补全篇章", "查证链接", "查证状态", "页码", "待审标记"]


def excel_safe(value):
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def to_tsv(quotes):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(HEADERS)
    for q in quotes:
        r, s, e = q.get("relevance", {}), q.get("source_explicit", {}), q.get("source_enriched", {})
        writer.writerow(map(excel_safe, [q.get("id"), q.get("article_id"), q.get("text"),
            r.get("label"), "；".join(r.get("dimensions", [])), r.get("reason"), s.get("book"),
            s.get("chapter"), e.get("book"), e.get("chapter"), e.get("url"), e.get("status", "待查证"),
            ",".join(str(p) for p in sorted({x["page"] for x in q.get("evidence", []) if "page" in x})),
            "；".join(q.get("review_flags", []))]))
    return buffer.getvalue()


def write_result(result, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "quotes.tsv").write_text(to_tsv(result["quotes"]), encoding="utf-8-sig")
    return out
