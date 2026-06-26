"""Reproductibilité : graine unique pour random / numpy / torch.

La reproductibilité est un critère d'évaluation à part entière (brief CV §3).
On appelle ``fixer_seeds(42)`` en début de chaque script ou notebook.
"""

from __future__ import annotations

import os
import random


def fixer_seeds(seed: int = 42) -> int:
    """Fixe toutes les sources d'aléatoire pour rendre les runs reproductibles.

    Couvre ``random``, ``numpy`` et (si présent) ``torch`` CPU + CUDA, ainsi que
    ``PYTHONHASHSEED``. ``torch`` est importé paresseusement pour que la fonction
    reste utilisable dans un environnement sans deep learning (tests CPU).

    Args:
        seed: Graine entière à appliquer partout. Défaut 42.

    Returns:
        La graine effectivement appliquée (utile pour la journalisation).

    Example:
        >>> fixer_seeds(42)
        42
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy est une dépendance de base
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Environnement sans torch (ex. CI légère) : on ne bloque pas.
        pass

    return seed
