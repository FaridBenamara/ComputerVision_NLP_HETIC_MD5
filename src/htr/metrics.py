"""Métriques d'évaluation HTR : CER, WER, IC bootstrap, McNemar, IoU.

Protocoles définis dans le brief CV (§Critères, Étape 4) :
    - CER global (distance de Levenshtein / longueur référence) — métrique principale.
    - WER global (Word Error Rate).
    - Intervalle de confiance à 95 % par bootstrap (N = 1000).
    - Test de McNemar pour comparer deux variantes (ex. r=16 vs r=32).
    - IoU entre polygones prédits et référence (qualité de segmentation).
"""

from __future__ import annotations

from math import comb
from typing import Callable, Sequence

import editdistance
import numpy as np


def _liste(x: str | Sequence[str]) -> list[str]:
    """Normalise une str unique ou une séquence de str en liste de str."""
    return [x] if isinstance(x, str) else list(x)


def cer(refs: str | Sequence[str], hyps: str | Sequence[str]) -> float:
    """Character Error Rate global (micro-moyenne sur le corpus).

    CER = somme des distances de Levenshtein / somme des longueurs de référence.

    Args:
        refs: Référence(s) — str ou séquence de str.
        hyps: Hypothèse(s) alignée(s) sur ``refs``.

    Returns:
        CER dans [0, +inf[ (généralement [0, 1]). 0.0 = transcription parfaite.

    Example:
        >>> round(cer("creature", "cieratur"), 3)
        0.5
    """
    refs, hyps = _liste(refs), _liste(hyps)
    dist = sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
    longueur = sum(max(len(r), 1) for r in refs)
    return dist / longueur


def wer(refs: str | Sequence[str], hyps: str | Sequence[str]) -> float:
    """Word Error Rate global (micro-moyenne, distance d'édition sur les mots).

    Args:
        refs: Référence(s).
        hyps: Hypothèse(s).

    Returns:
        WER (float). Plus sensible que le CER aux erreurs lexicales.
    """
    refs, hyps = _liste(refs), _liste(hyps)
    dist = sum(editdistance.eval(r.split(), h.split()) for r, h in zip(refs, hyps))
    longueur = sum(max(len(r.split()), 1) for r in refs)
    return dist / longueur


def cer_par_ligne(refs: Sequence[str], hyps: Sequence[str]) -> np.ndarray:
    """CER ligne par ligne (utile pour calibrer le seuil de needs_review).

    Args:
        refs: Références.
        hyps: Hypothèses.

    Returns:
        Tableau numpy des CER par ligne.
    """
    return np.array(
        [editdistance.eval(r, h) / max(len(r), 1) for r, h in zip(refs, hyps)],
        dtype=float,
    )


def bootstrap_ic_cer(
    refs: Sequence[str],
    hyps: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Intervalle de confiance bootstrap sur le CER global (ré-échantillonnage).

    On ré-échantillonne les lignes avec remise N fois et on recalcule le CER
    global à chaque fois ; l'IC est donné par les percentiles de la distribution.

    Args:
        refs: Références.
        hyps: Hypothèses.
        n_resamples: Nombre de ré-échantillons bootstrap (brief : 1000).
        seed: Graine du générateur (reproductibilité).
        alpha: Niveau ; 0.05 → IC à 95 %.

    Returns:
        Dict ``{"cer", "ic_bas", "ic_haut", "alpha", "n_resamples"}``.

    Example:
        >>> r = bootstrap_ic_cer(refs, hyps, n_resamples=1000)
        >>> r["ic_bas"], r["cer"], r["ic_haut"]
    """
    refs, hyps = _liste(refs), _liste(hyps)
    dists = np.array([editdistance.eval(r, h) for r, h in zip(refs, hyps)], dtype=float)
    longs = np.array([max(len(r), 1) for r in refs], dtype=float)
    n = len(refs)
    rng = np.random.default_rng(seed)

    scores = np.empty(n_resamples, dtype=float)
    indices = np.arange(n)
    for i in range(n_resamples):
        ech = rng.choice(indices, size=n, replace=True)
        scores[i] = dists[ech].sum() / longs[ech].sum()

    return {
        "cer": float(dists.sum() / longs.sum()),
        "ic_bas": float(np.percentile(scores, 100 * alpha / 2)),
        "ic_haut": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "alpha": alpha,
        "n_resamples": n_resamples,
    }


def mcnemar(
    refs: Sequence[str],
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    est_correct: Callable[[str, str], bool] | None = None,
) -> dict[str, float | int]:
    """Test de McNemar (exact) comparant deux systèmes ligne par ligne.

    Pour chaque ligne on décide si A et B sont « corrects » (par défaut : égalité
    exacte après strip). On compte les discordances : ``n10`` (A bon, B mauvais)
    et ``n01`` (A mauvais, B bon). Sous H0, n01 et n10 suivent une binomiale(n, 0.5) ;
    la p-value exacte bilatérale est ``2 * P(X <= min(n01, n10))`` plafonnée à 1.

    Args:
        refs: Références.
        hyps_a: Hypothèses du système A.
        hyps_b: Hypothèses du système B.
        est_correct: Critère ``(ref, hyp) -> bool``. Défaut : égalité exacte.

    Returns:
        Dict ``{"n01", "n10", "n_discordants", "statistique", "p_value"}``.
        ``statistique`` est le chi² de McNemar avec correction de continuité.

    Example:
        >>> mcnemar(refs, preds_r16, preds_r32)["p_value"]
    """
    refs, hyps_a, hyps_b = _liste(refs), _liste(hyps_a), _liste(hyps_b)
    if est_correct is None:
        est_correct = lambda r, h: r.strip() == h.strip()  # noqa: E731

    a_ok = [est_correct(r, h) for r, h in zip(refs, hyps_a)]
    b_ok = [est_correct(r, h) for r, h in zip(refs, hyps_b)]
    n10 = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    n01 = sum(1 for x, y in zip(a_ok, b_ok) if not x and y)
    n = n01 + n10

    if n == 0:
        return {"n01": 0, "n10": 0, "n_discordants": 0, "statistique": 0.0, "p_value": 1.0}

    k = min(n01, n10)
    p_value = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2**n))
    statistique = (abs(n01 - n10) - 1) ** 2 / n  # correction de continuité
    return {
        "n01": n01,
        "n10": n10,
        "n_discordants": n,
        "statistique": float(statistique),
        "p_value": float(p_value),
    }


def iou_polygones(
    poly_a: Sequence[Sequence[float]],
    poly_b: Sequence[Sequence[float]],
) -> float:
    """Intersection-over-Union entre deux polygones, par rastérisation de masques.

    Implémentation sans dépendance géométrique lourde : on rasterise les deux
    polygones dans un canevas commun (cv2.fillPoly) et on calcule
    intersection/union sur les masques binaires.

    Args:
        poly_a: Polygone A, liste de points ``(x, y)``.
        poly_b: Polygone B, liste de points ``(x, y)``.

    Returns:
        IoU dans [0, 1]. 1.0 = recouvrement parfait, 0.0 = disjoints.

    Example:
        >>> iou_polygones([(0,0),(10,0),(10,10),(0,10)],
        ...               [(0,0),(10,0),(10,10),(0,10)])
        1.0
    """
    import cv2  # import paresseux : seule IoU dépend d'OpenCV ici

    a = np.asarray(poly_a, dtype=np.int64)
    b = np.asarray(poly_b, dtype=np.int64)
    origine = np.vstack([a, b]).min(axis=0)
    a0, b0 = a - origine, b - origine
    largeur, hauteur = (np.vstack([a0, b0]).max(axis=0) + 1).tolist()

    masque_a = np.zeros((int(hauteur), int(largeur)), dtype=np.uint8)
    masque_b = np.zeros_like(masque_a)
    cv2.fillPoly(masque_a, [a0.astype(np.int32)], 1)
    cv2.fillPoly(masque_b, [b0.astype(np.int32)], 1)

    inter = int(np.logical_and(masque_a, masque_b).sum())
    union = int(np.logical_or(masque_a, masque_b).sum())
    return inter / union if union > 0 else 0.0
