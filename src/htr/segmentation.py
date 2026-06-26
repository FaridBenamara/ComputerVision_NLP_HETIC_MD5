"""Segmentation en lignes de texte.

Deux stratégies :
    - ``segmenter_par_projection`` : profil horizontal d'encre (rapide, sans
      dépendance lourde). Baseline / fallback. Limite connue et **assumée** :
      sur-/sous-segmente les pages denses ou multi-colonnes — c'est le maillon
      faible identifié, qui pilote le CER final.
    - ``segmenter_kraken_blla`` : wrapper Kraken BLLA (baselines + polygones de
      qualité, compatible eScriptorium/PAGE XML). Recommandé pour les vraies pages.

Chaque ligne est un dict ``{"id", "poly", "baseline"}`` avec ``poly`` =
liste de points (x, y) dans le repère image (origine en haut à gauche, px).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def segmenter_par_projection(
    image: np.ndarray,
    seuil_encre: float = 0.02,
    ecart_min: int = 6,
    hauteur_min: int = 12,
    lissage: int = 3,
) -> list[dict[str, Any]]:
    """Segmente en lignes via le profil de projection horizontale.

    On binarise (Otsu inversé), on calcule la quantité d'encre par ligne de
    pixels, on lisse, puis on coupe une ligne dès qu'on observe ``ecart_min``
    lignes quasi-blanches consécutives. Robuste aux petits inter-mots, mais
    limité sur les écritures denses (lignes qui se touchent).

    Args:
        image: Page RGB (H, W, 3) ou niveaux de gris (H, W), uint8.
        seuil_encre: Fraction du pic d'encre au-dessus de laquelle une ligne de
            pixels est considérée « texte ».
        ecart_min: Nb de lignes blanches consécutives confirmant une fin de ligne.
        hauteur_min: Hauteur minimale (px) d'une ligne retenue.
        lissage: Largeur du noyau de moyenne mobile sur le profil.

    Returns:
        Liste de dicts ``{"id", "poly", "baseline"}`` ordonnés de haut en bas.

    Example:
        >>> lignes = segmenter_par_projection(page_rgb)
        >>> lignes[0]["poly"]      # [(0,y0),(W-1,y0),(W-1,y1),(0,y1)]
    """
    gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    binv = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    profil = np.convolve(
        binv.sum(axis=1).astype(float), np.ones(lissage) / lissage, mode="same"
    )
    est_texte = profil > seuil_encre * profil.max()
    largeur = image.shape[1]

    bornes: list[tuple[int, int]] = []
    debut, ecart = None, 0
    for y in range(len(est_texte) + 1):
        texte = est_texte[y] if y < len(est_texte) else False
        if texte:
            if debut is None:
                debut = y
            ecart = 0
        elif debut is not None:
            ecart += 1
            if ecart >= ecart_min:
                fin = y - ecart
                if fin - debut >= hauteur_min:
                    bornes.append((debut, fin))
                debut, ecart = None, 0

    return [
        {
            "id": f"line_{i:04d}",
            "poly": [(0, y0), (largeur - 1, y0), (largeur - 1, y1), (0, y1)],
            "baseline": [(0, y1), (largeur - 1, y1)],
        }
        for i, (y0, y1) in enumerate(bornes)
    ]


def segmenter_kraken_blla(
    chemin_image: str,
    modele=None,
) -> list[dict[str, Any]]:
    """Segmente une page avec Kraken BLLA (baselines + polygones de qualité).

    Wrapper paresseux : Kraken (Linux, lourd) n'est importé qu'à l'appel, pour
    que le reste du module reste utilisable sans cette dépendance. Recommandé
    sur les vraies pages de manuscrit ; les polygones produits sont exportables
    en PAGE XML compatible eScriptorium.

    Args:
        chemin_image: Chemin vers l'image de page (TIFF/JPEG/PNG).
        modele: Modèle de segmentation Kraken déjà chargé (``vgsl.TorchVGSLModel``)
            ou None pour le modèle BLLA par défaut.

    Returns:
        Liste de dicts ``{"id", "poly", "baseline"}`` (mêmes clés que la version
        projection), prête pour l'export PAGE XML / le data contract.

    Raises:
        ImportError: Si ``kraken`` n'est pas installé (cf. extra ``[kraken]``).

    Example:
        >>> lignes = segmenter_kraken_blla("page.jpg")   # doctest: +SKIP
    """
    try:
        from kraken import blla
        from kraken.lib import vgsl
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "segmenter_kraken_blla nécessite 'kraken' (pip install kraken). "
            "Voir l'extra [kraken] du pyproject."
        ) from exc

    image = Image.open(chemin_image).convert("RGB")
    if modele is None:
        modele = vgsl.TorchVGSLModel.load_model(blla.DEFAULT_MODEL)
    resultat = blla.segment(image, model=modele)

    lignes: list[dict[str, Any]] = []
    for i, ligne in enumerate(resultat.lines):
        poly = [tuple(map(int, p)) for p in (ligne.boundary or [])]
        base = [tuple(map(int, p)) for p in (ligne.baseline or [])]
        lignes.append({"id": f"line_{i:04d}", "poly": poly, "baseline": base})
    return lignes
