"""Data contract : JSON livré au Volet 2 (NLP), construction + validation jsonschema.

Le JSON est le point de départ obligatoire de tout le NLP (brief NLP §1). Chaque
ligne porte le texte, sa géométrie (polygone dans le repère page), la confiance
globale ET par caractère, des candidats alternatifs par caractère, et le drapeau
``needs_review`` — pensé pour la correction NLP guidée par la confiance.

Le schéma est versionné dans ``schemas/data_contract.schema.json`` et validé
systématiquement avant toute manipulation (brief NLP §1 : « Validez le schéma
avec jsonschema avant toute manipulation »).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

_SCHEMA_DEFAUT = Path(__file__).resolve().parents[2] / "schemas" / "data_contract.schema.json"


def construire_ligne(
    idx: int,
    texte: str,
    polygone: Sequence[Sequence[float]],
    confiance: float,
    confiance_char: Sequence[float],
    candidats_char: Sequence[Sequence[str]],
    image_source: str,
    metadonnees: dict[str, Any] | None = None,
    baseline: Sequence[Sequence[float]] | None = None,
    seuil_needs_review: float = 0.70,
) -> dict[str, Any]:
    """Construit l'entrée JSON d'une ligne transcrite conforme au data contract.

    Args:
        idx: Index de la ligne (sert à fabriquer l'``id`` ``line_00042``).
        texte: Transcription de la ligne.
        polygone: Contour de la ligne dans le repère page (liste de points (x, y)).
        confiance: Confiance globale de la ligne dans [0, 1].
        confiance_char: Confiance par caractère (alignée sur ``texte``).
        candidats_char: Candidats alternatifs par caractère (ex. [["a","e"], ...]).
        image_source: Cote/identifiant de l'image source (``shelfmark``).
        metadonnees: Métadonnées libres (siècle, script_type…).
        baseline: Ligne de base optionnelle (liste de points (x, y)).
        seuil_needs_review: Seuil de confiance sous lequel ``needs_review`` = True
            (calibré à 0.70 dans le projet ; voir l'EDA de calibration).

    Returns:
        Dict d'une ligne, prêt à insérer dans le document du data contract.
    """
    return {
        "id": f"line_{idx:05d}",
        "image_source": image_source,
        "texte": texte,
        "polygone": [list(p) for p in polygone],
        "baseline": [list(p) for p in baseline] if baseline else None,
        "confiance": round(float(confiance), 4),
        "confiance_char": [round(float(c), 4) for c in confiance_char],
        "candidats_char": [list(c) for c in candidats_char],
        "needs_review": bool(confiance < seuil_needs_review or len(texte.strip()) == 0),
        "metadonnees": metadonnees or {},
    }


def construire_document(
    lignes: list[dict[str, Any]],
    corpus: str,
    modele: str,
    train_sha256: str,
    langue: str = "French",
    seuil_needs_review: float = 0.70,
) -> dict[str, Any]:
    """Assemble le document complet du data contract (métadonnées + lignes).

    Args:
        lignes: Lignes produites par ``construire_ligne``.
        corpus: Nom du corpus source (ex. "CATMuS Medieval (français)").
        modele: Identifiant du modèle HTR (ex. "trocr-fr-lora (r=16)").
        train_sha256: SHA-256 du split d'entraînement (anti-contamination).
        langue: Langue dominante du corpus.
        seuil_needs_review: Seuil de confiance utilisé pour ``needs_review``.

    Returns:
        Document dict conforme à ``schemas/data_contract.schema.json``.
    """
    return {
        "metadata": {
            "corpus": corpus,
            "langue": langue,
            "modele": modele,
            "systeme_coordonnees": {"origine": "haut-gauche", "unite": "pixels"},
            "seuil_needs_review": seuil_needs_review,
            "train_sha256": train_sha256,
        },
        "lignes": lignes,
    }


def valider(document: dict[str, Any], chemin_schema: str | Path | None = None) -> None:
    """Valide un document du data contract contre le schéma JSON.

    Args:
        document: Document produit par ``construire_document``.
        chemin_schema: Chemin du schéma ; défaut : ``schemas/data_contract.schema.json``.

    Raises:
        jsonschema.ValidationError: Si le document ne respecte pas le schéma.
        ImportError: Si ``jsonschema`` n'est pas installé.

    Example:
        >>> valider(doc)            # ne lève rien si conforme   # doctest: +SKIP
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise ImportError("valider nécessite 'jsonschema' (pip install jsonschema).") from exc

    chemin = Path(chemin_schema) if chemin_schema else _SCHEMA_DEFAUT
    schema = json.loads(Path(chemin).read_text(encoding="utf-8"))
    jsonschema.validate(instance=document, schema=schema)


def ecrire(document: dict[str, Any], chemin: str | Path) -> Path:
    """Écrit le document JSON sur disque en UTF-8 (signes médiévaux préservés).

    Args:
        document: Document à sérialiser.
        chemin: Chemin de sortie (.json).

    Returns:
        Le ``Path`` du fichier écrit.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return chemin
