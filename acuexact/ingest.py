"""Read-only archive inventories and portable EndNote export adapters."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import xml.etree.ElementTree as ET


def inventory_zip(path):
    with zipfile.ZipFile(path) as z:
        entries = [{"path": i.filename, "bytes": i.file_size, "compressed_bytes": i.compress_size}
                   for i in z.infolist() if not i.is_dir()]
    return {"archive": Path(path).name, "files": len(entries),
            "uncompressed_bytes": sum(i["bytes"] for i in entries),
            "extensions": dict(Counter(PurePosixPath(i["path"]).suffix.lower() or "(none)" for i in entries)),
            "entries": entries}


def extract_selected(path, members, output, max_bytes=512 * 1024 * 1024):
    """Never extractall: reject traversal, symlinks, oversize and duplicate names."""
    root = Path(output).resolve()
    if len(set(members)) != len(members):
        raise ValueError("Repeated selected archive member.")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        counts = Counter(archive.namelist())
        selected = [archive.getinfo(name) for name in members]
        if sum(i.file_size for i in selected) > max_bytes:
            raise ValueError("Selected members exceed extraction byte budget.")
        checked = []
        for info in selected:
            name = info.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or any(":" in p for p in relative.parts):
                raise ValueError(f"Unsafe archive path: {name}")
            if counts[info.filename] != 1 or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Ambiguous duplicate member or symlink in archive.")
            target = root.joinpath(*relative.parts).resolve()
            if root not in target.parents or target.exists():
                raise ValueError("Refusing overwrite or out-of-root archive extraction.")
            checked.append((info, target))
        for info, target in checked:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as dest:
                written = 0
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > info.file_size:
                        raise ValueError("Archive member exceeds declared size.")
                    dest.write(chunk)
    return [str(t) for _, t in checked]


def parse_endnote(path):
    """Accept XML, RIS, or EndNote tagged text. Binary ENL is not guessed."""
    path = Path(path)
    if path.suffix.lower() in {".enl", ".enlx"}:
        raise ValueError("Export selected EndNote records as XML (recommended), RIS, or EndNote Export TXT; native ENL is not a portable record export.")
    data = path.read_text(encoding="utf-8-sig")
    if data.lstrip().startswith("<"):
        # Entity expansion must not be accepted from arbitrary exported XML.
        if "<!DOCTYPE" in data.upper() or "<!ENTITY" in data.upper():
            raise ValueError("DTD/entity declarations are not accepted.")
        root = ET.fromstring(data)
        records = []
        for i, element in enumerate(root.findall(".//record")):
            def txt(pattern):
                found = element.find(pattern)
                return "".join(found.itertext()).strip() if found is not None else ""
            records.append({"id": txt("rec-number") or str(i + 1), "title": txt("titles/title"),
                            "authors": ["".join(a.itertext()) for a in element.findall("contributors/authors/author")],
                            "year": txt("dates/year"), "pages": txt("pages"),
                            "pdf_links": ["".join(u.itertext()) for u in element.findall(".//pdf-urls/url")]})
        return validate_records(records)
    records, current, authors, links = [], {}, [], []
    ris = bool(re.search(r"^TY  -", data, re.M))
    def finish():
        if current:
            current.setdefault("id", str(len(records) + 1))
            current.setdefault("title", "")
            current["authors"], current["pdf_links"] = list(authors), list(links)
            records.append(dict(current))
            current.clear(); authors.clear(); links.clear()
    for line in data.splitlines():
        if ris:
            match = re.match(r"^([A-Z0-9]{2})  -\s?(.*)$", line)
        else:
            match = re.match(r"^%([A-Za-z0-9#@>])\s*(.*)$", line)
        if not match:
            if not ris and not line.strip():
                finish()
            continue
        key, value = match.groups()
        if key in {"ER"}:
            finish()
        elif key in ({"TI", "T1"} if ris else {"T"}):
            current["title"] = value
        elif key in ({"ID"} if ris else {"F"}):
            current["id"] = value
        elif key in ({"AU", "A1"} if ris else {"A"}):
            authors.append(value)
        elif key in ({"PY", "Y1"} if ris else {"D"}):
            current["year"] = value
        elif key in ({"L1", "L2"} if ris else {">"}):
            links.append(value)
        elif key in ({"SP"} if ris else {"P"}):
            current["pages"] = value
    finish()
    return validate_records(records)


def validate_records(records):
    if not records:
        raise ValueError("No EndNote records detected; expected XML, RIS or tagged export text.")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("EndNote record IDs must be unique.")
    if any(not str(r.get("title", "")).strip() for r in records):
        raise ValueError("Every EndNote record requires a nonempty title; refusing to infer article identity.")
    return records


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
