import tempfile
import unittest
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from acuexact.pdf import read_regions, file_sha256, open_pdf


def fixture(path):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 40 700 Td (TARGET_ARTICLE_TEXT) Tj ET\nBT /F1 12 Tf 340 300 Td (NEIGHBOUR_ARTICLE_TEXT) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


class PdfScopeTests(unittest.TestCase):
    def test_approved_region_excludes_neighbour_same_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two.pdf"
            fixture(path)
            manifest = {"article": {"id": "7", "title": "TARGET"}, "boundary_status": "approved",
                        "pdf_sha256": file_sha256(path), "regions": [{"page": 1, "bbox": [0, 0, 300, 300]}]}
            segments = read_regions(path, manifest)
            self.assertIn("TARGET_ARTICLE_TEXT", segments[0]["text"])
            self.assertNotIn("NEIGHBOUR", segments[0]["text"])
            manifest["boundary_status"] = "pending_review"
            with self.assertRaises(ValueError):
                read_regions(path, manifest)
            manifest["boundary_status"] = "approved"
            manifest["pdf_sha256"] = "changed"
            with self.assertRaises(ValueError):
                read_regions(path, manifest)

    def test_overlap_and_invalid_bounds_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two.pdf"
            fixture(path)
            manifest = {"article": {"id": "7", "title": "TARGET"}, "boundary_status": "approved",
                        "pdf_sha256": file_sha256(path), "regions": [{"page": 1, "bbox": [0, 0, 300, 300]}]}
            manifest["regions"].append({"page": 1, "bbox": [0, 0, 300, 300]})
            with self.assertRaises(ValueError):
                read_regions(path, manifest)
            manifest["regions"] = [{"page": 1, "bbox": [0, 0, 900, 900]}]
            with self.assertRaises(ValueError):
                read_regions(path, manifest)


if __name__ == "__main__":
    unittest.main()
