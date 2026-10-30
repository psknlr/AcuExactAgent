"""Page geometry is part of the evidence, not discarded preprocessing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from contextlib import contextmanager
import tempfile


@contextmanager
def open_pdf(path):
    """Some VeryPDF files expose zero pages to pdfminer; normalize with pypdf once."""
    import pdfplumber
    document = pdfplumber.open(path)
    if document.pages:
        try:
            yield document
        finally:
            document.close()
        return
    document.close()
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(path)
    if not reader.pages:
        raise ValueError("PDF has zero readable pages.")
    with tempfile.TemporaryDirectory(prefix="acuexact-pdf-") as directory:
        repaired = Path(directory) / "normalized.pdf"
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.write(repaired)
        with pdfplumber.open(repaired) as document:
            if not document.pages:
                raise ValueError("PDF normalization failed; manual review required.")
            yield document


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_pdf(path, output, title="", max_pages=None):
    """Render pages, expose text and horizontal separators, propose UNAPPROVED bounds."""
    import pdfplumber
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pages, candidates = [], []
    target = re.sub(r"\s+", "", title)
    with open_pdf(path) as document:
        for index, page in enumerate(document.pages):
            if max_pages is not None and index >= max_pages:
                break
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            separators = [[round(line["x0"], 2), round(line["top"], 2),
                           round(line["x1"], 2), round(line["bottom"], 2)]
                          for line in page.edges if abs(line["top"] - line["bottom"]) < 2
                          and line["x1"] - line["x0"] > page.width * .3]
            png = out / f"page-{index + 1:04d}.png"
            page.to_image(resolution=100).save(png)
            pages.append({"page": index + 1, "width": page.width, "height": page.height,
                          "text": text, "separators": separators, "preview": png.name,
                          "needs_ocr": len(re.sub(r"\s", "", text)) < 30,
                          "text_quality_flags": text_quality_flags(text)})
            if target and target in re.sub(r"\s+", "", text):
                candidates.append(index + 1)
    report = {"pdf": str(Path(path).resolve()), "pdf_sha256": file_sha256(path),
              "title": title, "title_candidate_pages": candidates, "pages": pages,
              "note": "Horizontal rules and title matches are hints only. Confirm article and column regions visually."}
    (out / "inspection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def text_quality_flags(text):
    flags = []
    if "(cid:" in text or "�" in text:
        flags.append("unmapped_glyphs")
    if any(0xE000 <= ord(c) <= 0xF8FF for c in text):
        flags.append("private_use_characters")
    if any((ord(c) < 32 and c not in "\n\r\t") or 0x7F <= ord(c) < 0xA0 for c in text):
        flags.append("control_characters")
    if text.count("“") != text.count("”"):
        flags.append("unbalanced_curly_quotes")
    return flags


def read_regions(path, manifest, *, ocr=False):
    """Read only approved, ordered rectangles. PDF coords are points from top-left.

    For two columns, list left and right as separate regions in reading order. Same-page
    articles must use separate bounds; title matching alone cannot approve the region.
    """
    import pdfplumber
    if manifest.get("boundary_status") != "approved":
        raise ValueError("Article boundaries are not approved. Inspect page PNGs and approve the ordered regions first.")
    if not manifest.get("article", {}).get("id") or not manifest["article"].get("title"):
        raise ValueError("A manifest requires article.id and article.title from the EndNote record.")
    regions = manifest.get("regions", [])
    if not regions:
        raise ValueError("No article regions selected; refusing whole-PDF fallback.")
    actual_hash = file_sha256(path)
    if not manifest.get("pdf_sha256"):
        raise ValueError("An approved manifest requires pdf_sha256 to bind bounds to the reviewed PDF.")
    if manifest["pdf_sha256"] != actual_hash:
        raise ValueError("PDF hash differs from reviewed manifest.")
    segments = []
    with open_pdf(path) as document:
        seen = []
        for index, region in enumerate(regions):
            page_number = region.get("page")
            if not isinstance(page_number, int) or not 1 <= page_number <= len(document.pages):
                raise ValueError(f"Invalid page in region {index + 1}")
            page = document.pages[page_number - 1]
            bbox = region.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Every region requires an explicit [x0, top, x1, bottom] rectangle.")
            x0, top, x1, bottom = map(float, bbox)
            if not (0 <= x0 < x1 <= page.width + .01 and 0 <= top < bottom <= page.height + .01):
                raise ValueError(f"Out-of-page or empty region: {bbox}")
            for oldpage, old in seen:
                if oldpage == page_number and min(x1, old[2]) > max(x0, old[0]) + .5 and min(bottom, old[3]) > max(top, old[1]) + .5:
                    raise ValueError("Overlapping article regions would duplicate quotes.")
            seen.append((page_number, [x0, top, x1, bottom]))
            crop = page.crop((x0, top, x1, bottom))
            text = crop.extract_text(x_tolerance=2, y_tolerance=3) or ""
            method = "pdf_text"
            pdf_text = text
            verified_text = region.get("verified_text")
            if verified_text is not None:
                if not isinstance(verified_text, str) or not verified_text.strip() or not region.get("text_reviewer"):
                    raise ValueError("A corrected transcription requires nonempty verified_text and text_reviewer.")
                text, method = verified_text, "human_verified_transcription"
            low_text = len(re.sub(r"\s+", "", text)) < 10
            if verified_text is None and (low_text or ocr):
                if low_text and not ocr:
                    raise ValueError(f"Region {index + 1} on page {page_number} has insufficient text. Use --ocr or supply verified text; no silent empty result.")
                try:
                    import pytesseract
                    text = pytesseract.image_to_string(crop.to_image(resolution=300).original,
                                                       lang="chi_sim+chi_tra", config="--psm 6")
                except Exception as exc:
                    raise RuntimeError("OCR needs Tesseract plus chi_sim and chi_tra language data and the [ocr] extra.") from exc
                method = "ocr"
                if not text.strip():
                    raise ValueError(f"OCR returned no text on page {page_number}; review required.")
            segments.append({"page": page_number, "bbox": bbox, "region_id": region.get("id", f"r{index + 1}"),
                             "text": text, "pdf_text_original": pdf_text, "method": method,
                             "text_reviewer": region.get("text_reviewer"), "quality_flags": text_quality_flags(text)})
    return segments
