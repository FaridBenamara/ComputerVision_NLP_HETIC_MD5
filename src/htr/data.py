"""Chargement et nettoyage du corpus CATMuS (sous-ensemble français).

- Sélection ciblée des fichiers parquet ``L-Fre*`` (≈20 Go au lieu de ≈96 Go).
- Normalisation Unicode NFC + nettoyage des espaces (les signes médiévaux —
  ⁊, q̃, ꝑ … — sont **préservés** : choix éditorial semi-diplomatique, la
  résolution des abréviations est laissée au Volet 2 / NLP).
- Hash SHA-256 du split d'entraînement (anti-contamination, brief CV §3 et §Critères).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Sequence

_ESPACES = re.compile(r"[ \t ]+")


def nettoyer_texte(texte: str | None) -> str:
    """Normalise un texte de ligne : Unicode NFC + espaces de bord/multiples.

    On applique NFC (forme composée canonique) pour que des graphies
    équivalentes soient encodées de façon identique, on réduit les suites
    d'espaces à une seule et on retire les espaces de bord. Les caractères
    médiévaux (abréviations, tilde nasal) sont conservés tels quels.

    Args:
        texte: Texte brut d'une ligne (peut être None).

    Returns:
        Texte normalisé (str). Chaîne vide si l'entrée est None/vide.

    Example:
        >>> nettoyer_texte("  créature  ⁊  ")   # 'é' décomposé + espaces
        'créature ⁊'
    """
    if not texte:
        return ""
    texte = unicodedata.normalize("NFC", texte)
    texte = _ESPACES.sub(" ", texte)
    return texte.strip()


def nettoyer_corpus(
    textes: Sequence[str],
    longueur_min: int = 1,
) -> tuple[list[int], dict[str, int]]:
    """Filtre les lignes vides ou trop courtes, avec rapport avant/après.

    Args:
        textes: Liste des textes de référence d'un split.
        longueur_min: Longueur minimale (en caractères, après nettoyage) pour garder.

    Returns:
        Tuple ``(indices_gardes, rapport)`` où ``indices_gardes`` est la liste
        des indices conservés et ``rapport`` = ``{"avant", "apres", "rejetes"}``.

    Example:
        >>> gardes, rapport = nettoyer_corpus(["abc", "", " "])
        >>> rapport
        {'avant': 3, 'apres': 1, 'rejetes': 2}
    """
    gardes = [i for i, t in enumerate(textes) if len(nettoyer_texte(t)) >= longueur_min]
    rapport = {
        "avant": len(textes),
        "apres": len(gardes),
        "rejetes": len(textes) - len(gardes),
    }
    return gardes, rapport


def sha256_split(
    textes: Sequence[str],
    shelfmarks: Sequence[str] | None = None,
) -> str:
    """Calcule le SHA-256 d'un split (scellement anti-contamination).

    Le hash dépend du couple (cote du manuscrit, texte) de chaque ligne, dans
    l'ordre fourni. À calculer une fois sur le test, à reporter, et ne plus
    regarder le test jusqu'au rendu final.

    Args:
        textes: Textes des lignes du split.
        shelfmarks: Cotes (``shelfmark``) alignées ; défaut : chaînes vides.

    Returns:
        Empreinte SHA-256 hexadécimale (str de 64 caractères).
    """
    if shelfmarks is None:
        shelfmarks = [""] * len(textes)
    h = hashlib.sha256()
    for cote, texte in zip(shelfmarks, textes):
        h.update(f"{cote}\t{texte}\n".encode("utf-8"))
    return h.hexdigest()


def charger_catmus_francais(repo: str = "CATMuS/medieval") -> Any:
    """Charge UNIQUEMENT le sous-ensemble français de CATMuS Medieval.

    Repère les parquet ``L-Fre*`` par split officiel (``train``/``dev``/``test``)
    et les charge via 🤗 datasets, en garantissant que la colonne ``im`` est
    décodée comme image PIL. Conçu pour Colab (téléchargement + cache HF).

    Args:
        repo: Identifiant du dataset Hugging Face. Défaut ``"CATMuS/medieval"``.

    Returns:
        Un ``DatasetDict`` avec les clés ``train`` / ``validation`` / ``test``.

    Raises:
        ImportError: Si ``datasets`` / ``huggingface_hub`` ne sont pas installés.
        RuntimeError: Si aucun fichier français n'est trouvé (motif inattendu).

    Example:
        >>> ds = charger_catmus_francais()           # doctest: +SKIP
        >>> {sp: len(ds[sp]) for sp in ds}           # doctest: +SKIP
        {'train': 49201, 'validation': 2712, 'test': 4135}
    """
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement Colab
        raise ImportError(
            "charger_catmus_francais nécessite 'datasets' et 'huggingface_hub' "
            "(disponibles dans le notebook Colab)."
        ) from exc

    fichiers = HfApi().list_repo_files(repo, repo_type="dataset")
    data_files = {
        split: [
            f
            for f in fichiers
            if f.startswith(dossier + "/") and "L-Fre" in f and f.endswith(".parquet")
        ]
        for dossier, split in [("train", "train"), ("dev", "validation"), ("test", "test")]
    }
    if not all(data_files.values()):
        raise RuntimeError(
            "Aucun fichier 'L-Fre' trouvé — motif de filtrage inattendu, "
            "vérifier la structure du dataset avant de continuer."
        )

    ds = load_dataset(repo, data_files=data_files)
    if not isinstance(ds["train"].features["im"], HFImage):
        for split in list(ds.keys()):
            ds[split] = ds[split].cast_column("im", HFImage())
    return ds
