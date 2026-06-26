"""Tests de l'export PAGE XML (polygones + transcription)."""

from htr import page_xml as PX


def _lignes():
    return [
        {
            "id": "line_0000",
            "poly": [(0, 0), (100, 0), (100, 30), (0, 30)],
            "baseline": [(0, 30), (100, 30)],
            "texte": "creature qui croira ⁊ sera baptiziez",
            "conf": 0.84,
        }
    ]


def test_export_page_xml_structure():
    xml = PX.exporter_page_xml(_lignes(), "page_demo.jpg", largeur=1396, hauteur=2512)
    assert isinstance(xml, bytes)
    texte = xml.decode("utf-8")
    assert texte.startswith("<?xml")
    assert 'imageWidth="1396"' in texte
    assert 'imageHeight="2512"' in texte
    assert "line_0000" in texte
    assert "baptiziez" in texte  # transcription présente, signes Unicode conservés
    assert "TextLine" in texte and "Baseline" in texte
