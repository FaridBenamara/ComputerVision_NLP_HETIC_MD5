"""Export PAGE XML (PRImA 2019) des lignes segmentées + transcrites.

Format standard d'encodage de structure de page, compatible eScriptorium / Kraken
(brief CV §Périmètre 6 et §Contraintes 6 : réutilisabilité des données de
segmentation, repère origine en haut à gauche, unité en pixels).
"""

from __future__ import annotations

from typing import Any, Sequence

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def _points(poly: Sequence[Sequence[float]]) -> str:
    """Sérialise une liste de points (x, y) au format PAGE ``x1,y1 x2,y2 …``."""
    return " ".join(f"{int(x)},{int(y)}" for x, y in poly)


def exporter_page_xml(
    lignes: list[dict[str, Any]],
    nom_image: str,
    largeur: int,
    hauteur: int,
    createur: str = "htr-catmus-french-2026",
) -> bytes:
    """Construit un document PAGE XML à partir de lignes ``{poly, baseline, texte, conf}``.

    Args:
        lignes: Lignes avec au moins ``id`` et ``poly`` ; ``baseline``, ``texte``,
            ``conf`` sont optionnels.
        nom_image: Nom du fichier image référencé (attribut ``imageFilename``).
        largeur: Largeur de l'image source en pixels.
        hauteur: Hauteur de l'image source en pixels.
        createur: Valeur du champ ``Metadata/Creator``.

    Returns:
        Le document XML sérialisé en ``bytes`` (UTF-8, avec déclaration).

    Raises:
        ImportError: Si ``lxml`` n'est pas installé.

    Example:
        >>> xml = exporter_page_xml(lignes, "page.jpg", 1396, 2512)
        >>> xml.startswith(b"<?xml")
        True
    """
    try:
        from lxml import etree
    except ImportError as exc:  # pragma: no cover
        raise ImportError("exporter_page_xml nécessite 'lxml'.") from exc

    root = etree.Element("PcGts", nsmap={None: NS})
    etree.SubElement(etree.SubElement(root, "Metadata"), "Creator").text = createur
    page = etree.SubElement(
        root, "Page", imageFilename=nom_image,
        imageWidth=str(largeur), imageHeight=str(hauteur),
    )
    region = etree.SubElement(page, "TextRegion", id="r0")
    etree.SubElement(
        region, "Coords",
        points=_points([(0, 0), (largeur, 0), (largeur, hauteur), (0, hauteur)]),
    )
    for ligne in lignes:
        tl = etree.SubElement(region, "TextLine", id=ligne["id"])
        etree.SubElement(tl, "Coords", points=_points(ligne["poly"]))
        if ligne.get("baseline"):
            etree.SubElement(tl, "Baseline", points=_points(ligne["baseline"]))
        if ligne.get("texte") is not None:
            te = etree.SubElement(tl, "TextEquiv")
            if ligne.get("conf") is not None:
                te.set("conf", f"{float(ligne['conf']):.3f}")
            etree.SubElement(te, "Unicode").text = ligne["texte"]

    return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
