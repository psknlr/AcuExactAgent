import tempfile
import unittest
from pathlib import Path
import zipfile
from acuexact.ingest import extract_selected, parse_endnote
from acuexact.gold import import_gold


class IngestTests(unittest.TestCase):
    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("../escape.txt", "no")
            with self.assertRaises(ValueError):
                extract_selected(path, ["../escape.txt"], Path(temp) / "out")

    def test_zip_explicit_selection_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("a.txt", "yes"); z.writestr("b.txt", "no")
            extract_selected(path, ["a.txt"], Path(temp) / "out")
            self.assertFalse((Path(temp) / "out/b.txt").exists())

    def test_endnote_xml_and_ris(self):
        with tempfile.TemporaryDirectory() as temp:
            xml = Path(temp) / "en.xml"
            xml.write_text('<xml><records><record><rec-number>8</rec-number><titles><title><style>肩痛研究</style></title></titles><urls><pdf-urls><url>internal-pdf://a.pdf</url></pdf-urls></urls></record></records></xml>', encoding="utf-8")
            self.assertEqual(parse_endnote(xml)[0]["id"], "8")
            self.assertEqual(parse_endnote(xml)[0]["pdf_links"], ["internal-pdf://a.pdf"])
            ris = Path(temp) / "en.ris"
            ris.write_text('TY  - JOUR\nID  - 8\nTI  - 肩痛研究\nAU  - 某人\nER  - \n', encoding="utf-8")
            self.assertEqual(parse_endnote(ris)[0]["title"], "肩痛研究")

    def test_gold_requires_article_id(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gold.tsv"
            path.write_text("文章ID\t古籍条文\n\t不能猜文章\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                import_gold(path)

    def test_duplicate_endnote_xml_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.xml"
            record = '<record><rec-number>1</rec-number><titles><title>same</title></titles></record>'
            path.write_text('<xml><records>' + record * 2 + '</records></xml>', encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_endnote(path)


if __name__ == "__main__":
    unittest.main()
