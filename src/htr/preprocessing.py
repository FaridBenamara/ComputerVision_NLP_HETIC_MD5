"""Prétraitement documentaire : deskew, CLAHE, binarisation de Sauvola.

Chaîne définie dans le brief CV (§Périmètre 2 et Étape 2) : correction
d'inclinaison, amélioration de contraste par CLAHE, binarisation adaptative
de Sauvola — adaptée aux fonds anciens non uniformes où un seuil global échoue.

Toutes les fonctions sont *pures* (entrée → sortie, sans état global) et
paramétrables, donc directement testables (cf. tests/test_preprocessing.py).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def _en_gris(image: np.ndarray) -> np.ndarray:
    """Convertit une image RGB/RGBA en niveaux de gris uint8 ; passe-plat sinon."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.uint8, copy=False)


def corriger_inclinaison(
    image_gris: np.ndarray,
    angle_max: float = 15.0,
    pas: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Corrige l'inclinaison par maximisation de la variance du profil horizontal.

    Principe : on binarise (Otsu inversé), puis pour chaque angle candidat on
    fait tourner l'image et on mesure la variance de la projection horizontale
    (somme d'encre par ligne). Le bon angle aligne les lignes de texte, ce qui
    maximise le contraste entre lignes de texte et inter-lignes — donc la variance.

    Args:
        image_gris: Image en niveaux de gris (uint8) ou RGB (convertie en gris).
        angle_max: Amplitude de recherche en degrés, de ``-angle_max`` à ``+angle_max``.
        pas: Pas angulaire en degrés.

    Returns:
        Tuple ``(image_corrigee, angle)`` : l'image redressée (même dtype que
        l'entrée gris) et l'angle appliqué en degrés (positif = horaire inverse).

    Example:
        >>> img, angle = corriger_inclinaison(page_gris, angle_max=10, pas=1.0)
    """
    gris = _en_gris(image_gris)
    binv = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = binv.shape
    centre = (w / 2.0, h / 2.0)

    meilleur_angle, meilleur_score = 0.0, -1.0
    for angle in np.arange(-angle_max, angle_max + pas / 2, pas):
        mat = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rot = cv2.warpAffine(
            binv, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        score = float(rot.sum(axis=1, dtype=np.float64).var())
        if score > meilleur_score:
            meilleur_score, meilleur_angle = score, float(angle)

    mat = cv2.getRotationMatrix2D(centre, meilleur_angle, 1.0)
    corrige = cv2.warpAffine(
        gris, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return corrige, meilleur_angle


def ameliorer_contraste(
    image_gris: np.ndarray,
    clip_limit: float = 2.0,
    taille_tuile: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Améliore le contraste local par CLAHE (égalisation d'histogramme adaptative).

    Args:
        image_gris: Image niveaux de gris uint8 (ou RGB, convertie en gris).
        clip_limit: Plafond de contraste ; plus haut = plus agressif.
        taille_tuile: Grille de tuiles (lignes, colonnes) pour l'adaptation locale.

    Returns:
        Image niveaux de gris uint8 à contraste rehaussé, mêmes dimensions.
    """
    gris = _en_gris(image_gris)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=taille_tuile)
    return clahe.apply(gris)


def binariser_sauvola(
    image_gris: np.ndarray,
    taille_fenetre: int = 25,
    k: float = 0.2,
) -> np.ndarray:
    """Binarisation adaptative de Sauvola (seuil local par fenêtre glissante).

    Robuste aux fonds anciens non uniformes (taches, jaunissement, éclairage
    inégal) là où un seuil global (Otsu) échouerait.

    Args:
        image_gris: Image niveaux de gris uint8 (ou RGB, convertie en gris).
        taille_fenetre: Côté de la fenêtre locale (impair ; forcé impair sinon).
        k: Paramètre de Sauvola (typiquement 0.2–0.5). Plus haut = plus sélectif.

    Returns:
        Image binaire uint8 (0 ou 255), texte en noir (0) sur fond blanc (255).

    Raises:
        ValueError: Si ``taille_fenetre`` < 3.
    """
    if taille_fenetre < 3:
        raise ValueError("taille_fenetre doit être >= 3")
    if taille_fenetre % 2 == 0:
        taille_fenetre += 1  # threshold_sauvola exige une fenêtre impaire
    gris = _en_gris(image_gris)
    seuil = threshold_sauvola(gris, window_size=taille_fenetre, k=k)
    binaire = (gris > seuil).astype(np.uint8) * 255
    return binaire


def pretraiter(
    image: np.ndarray,
    deskew: bool = True,
    clahe: bool = True,
    sauvola: bool = True,
    angle_max: float = 15.0,
    clip_limit: float = 2.0,
    taille_tuile: tuple[int, int] = (8, 8),
    taille_fenetre: int = 25,
    k: float = 0.2,
) -> dict[str, Any]:
    """Applique la chaîne de prétraitement complète et paramétrable.

    Ordre : niveaux de gris → deskew → CLAHE → Sauvola. Chaque étape est
    optionnelle. Toutes les images intermédiaires sont renvoyées pour pouvoir
    illustrer l'effet de chaque brique (utile en présentation / article).

    Args:
        image: Image source RGB (H, W, 3) ou niveaux de gris (H, W), uint8.
        deskew: Active la correction d'inclinaison.
        clahe: Active l'amélioration de contraste CLAHE.
        sauvola: Active la binarisation de Sauvola.
        angle_max: Amplitude de recherche d'angle (deskew).
        clip_limit: Plafond CLAHE.
        taille_tuile: Grille de tuiles CLAHE.
        taille_fenetre: Fenêtre Sauvola.
        k: Paramètre k de Sauvola.

    Returns:
        Dict ``{"image", "angle", "etapes"}`` où ``image`` est la sortie finale,
        ``angle`` l'inclinaison corrigée (deg, 0.0 si deskew off) et ``etapes``
        un dict des images intermédiaires (``gris``, ``deskew``, ``clahe``,
        ``sauvola``) effectivement produites.

    Example:
        >>> res = pretraiter(page_rgb)
        >>> binaire = res["image"]        # image binaire finale
        >>> res["angle"]                  # inclinaison corrigée
    """
    gris = _en_gris(image)
    etapes: dict[str, np.ndarray] = {"gris": gris}
    courant = gris
    angle = 0.0

    if deskew:
        courant, angle = corriger_inclinaison(courant, angle_max=angle_max)
        etapes["deskew"] = courant
    if clahe:
        courant = ameliorer_contraste(
            courant, clip_limit=clip_limit, taille_tuile=taille_tuile
        )
        etapes["clahe"] = courant
    if sauvola:
        courant = binariser_sauvola(courant, taille_fenetre=taille_fenetre, k=k)
        etapes["sauvola"] = courant

    return {"image": courant, "angle": angle, "etapes": etapes}
