"""Volet 1 — Computer Vision & HTR (projet MD5, Master Data/IA HETIC).

Modules :
    seeds         : reproductibilité (fixer_seeds).
    preprocessing : deskew, CLAHE, binarisation de Sauvola (chaîne paramétrable).
    segmentation  : segmentation de lignes (projection + wrapper Kraken BLLA).
    metrics       : CER, WER, IC bootstrap, test de McNemar, IoU polygones.
    data          : chargement CATMuS français, normalisation NFC, hash SHA-256.
    data_contract : construction et validation jsonschema du JSON livré au NLP.
    model         : TrOCR + LoRA (build / train / save / inférence riche).
"""

__all__ = [
    "seeds",
    "preprocessing",
    "segmentation",
    "metrics",
    "data",
    "data_contract",
    "model",
]

__version__ = "0.1.0"
