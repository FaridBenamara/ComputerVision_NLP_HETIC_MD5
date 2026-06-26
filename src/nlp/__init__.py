"""Volet 2 — NLP (à venir).

Prendra en entrée le data contract JSON produit par le Volet 1 (``dataset_nlp/``)
et enchaînera : normalisation (règles déterministes NFC/u-v/i-j/tilde/abréviations
puis correction MLM CamemBERT guidée par la confiance), NER (CamemBERT médiéval,
schéma BIO PER/LOC/DATE/ORG/TITLE), POS + lemmes (stanza ``frm``), extraction de
relations par règles, graphe NetworkX et export TEI-XML.

Le démarrage est séquentiel par conception : le Volet 2 commence quand le Volet 1
livre un texte fidèle, localisé et calibré — c'est le rôle de ce dépôt.
"""

__all__: list[str] = []
