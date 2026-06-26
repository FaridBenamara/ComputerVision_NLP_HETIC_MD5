# Conventions NLP (Volet 2) — esquisse

> Volet 2 **non encore implémenté**. Ce document fige les choix dès maintenant (brief NLP §2/§5)
> et sera rempli au fur et à mesure. Le point de départ obligatoire est le data contract JSON
> du Volet 1 (`dataset_nlp/dataset_nlp.json`), **validé par jsonschema** avant toute manipulation.

## 1. Normalisation (règles d'abord, IA ensuite)
Ordre prévu, chaque règle documentée + impact **chiffré** (CER relatif avant/après) :
1. Unicode **NFC** (déjà appliqué en amont).
2. Substitution **u/v** et **i/j** selon le contexte.
3. Résolution du **tilde nasal** (ex. `ãte` → `ante`).
4. **Table d'abréviations** adaptée au corpus (ex. `q̃` → `que`, `⁊` → `et`, `ꝑ` → `per/par`…).
5. **Correction guidée par la confiance** : aux positions où `confiance_char < 0,70` et où
   plusieurs `candidats_char` existent, choisir via un modèle de langue **CamemBERT (MLM)** —
   `almanach/camembert-base` ou un modèle médiéval (ex. `magistermilitum/roberta-multilingual-medieval-ner`).

> Pas de vérité terrain absolue → **évaluation relative** (distances entre versions successives
> d'une même transcription), façon outil *Evaluation-HTR*.

## 2. NER — schéma BIO
- Classes : **PER, LOC, DATE, ORG** + éventuellement **TITLE** (roi, comte, évêque…).
- Modèle : CamemBERT médiéval fine-tuné légèrement (étendre `num_labels` si TITLE).
- **Alignement des labels** sur les sous-tokens : `-100` pour les word-pieces de continuation
  (point critique, à montrer dans le code et la présentation).
- Échantillon annoté manuellement 200–300 tokens minimum ; évaluation **seqeval** (F1 micro + par type).

## 3. POS, lemmes, relations, graphe, TEI
- POS + lemmes : **stanza** modèle `frm` (moyen français) ou `pie-extended`.
- Relations : règles lexico-syntaxiques simples (regex sur séquences de labels NER).
- Graphe : **NetworkX** sur un petit échantillon (5–10 pages).
- Export **TEI-XML** : `<persName>`, `<placeName>`, `<date>` dans un fichier TEI valide.

## Choix à justifier en soutenance
- Pourquoi semi-diplomatique + NFC ? (cf. `CONVENTIONS_TRANSCRIPTION.md`)
- Pourquoi ce schéma BIO et cette gestion des abréviations ambiguës ?
