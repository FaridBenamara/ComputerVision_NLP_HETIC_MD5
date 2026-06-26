"""Tests du prétraitement : deskew, CLAHE, Sauvola, chaîne complète.

Exigence brief CV §5 : tester les transformations du prétraitement (formes,
types, plages de valeurs).
"""

import cv2
import numpy as np
import pytest

from htr import preprocessing as P


def _image_rayures(taille=200, periode=20, epaisseur=6):
    """Fabrique une image avec des rayures horizontales (lignes de texte simulées)."""
    img = np.full((taille, taille), 255, dtype=np.uint8)
    for y in range(0, taille, periode):
        img[y : y + epaisseur, :] = 0
    return img


def _variance_profil(gris):
    binv = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return float(binv.sum(axis=1, dtype=np.float64).var())


def test_deskew_image_alignee_angle_proche_zero():
    img = _image_rayures()
    _, angle = P.corriger_inclinaison(img, angle_max=10, pas=0.5)
    assert abs(angle) <= 1.0


def test_deskew_recupere_inclinaison_connue():
    img = _image_rayures()
    h, w = img.shape
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)  # on incline de +5°
    incline = cv2.warpAffine(img, mat, (w, h), borderValue=255)

    corrige, angle = P.corriger_inclinaison(incline, angle_max=12, pas=0.5)
    # Le redressement doit tourner dans le sens inverse (~ -5°)...
    assert -8.5 <= angle <= -1.5
    # ... et ne pas dégrader la variance du profil (lignes mieux alignées).
    assert _variance_profil(corrige) >= _variance_profil(incline) * 0.99


def test_clahe_forme_type_et_effet():
    rampe = np.tile(np.linspace(100, 150, 200).astype(np.uint8), (200, 1))
    out = P.ameliorer_contraste(rampe, clip_limit=3.0)
    assert out.shape == rampe.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, rampe)  # le contraste local a été modifié


def test_sauvola_binaire():
    img = _image_rayures()
    binaire = P.binariser_sauvola(img, taille_fenetre=25, k=0.2)
    assert binaire.shape == img.shape
    assert binaire.dtype == np.uint8
    assert set(np.unique(binaire)).issubset({0, 255})


def test_sauvola_fenetre_invalide():
    with pytest.raises(ValueError):
        P.binariser_sauvola(_image_rayures(), taille_fenetre=2)


def test_pretraiter_chaine_complete():
    img_rgb = cv2.cvtColor(_image_rayures(), cv2.COLOR_GRAY2RGB)
    res = P.pretraiter(img_rgb, deskew=True, clahe=True, sauvola=True)
    assert set(res) == {"image", "angle", "etapes"}
    assert isinstance(res["angle"], float)
    assert set(res["etapes"]) == {"gris", "deskew", "clahe", "sauvola"}
    assert set(np.unique(res["image"])).issubset({0, 255})  # sortie binaire


def test_pretraiter_etapes_optionnelles():
    img = _image_rayures()
    res = P.pretraiter(img, deskew=False, clahe=False, sauvola=False)
    assert res["angle"] == 0.0
    assert set(res["etapes"]) == {"gris"}
    assert np.array_equal(res["image"], res["etapes"]["gris"])
