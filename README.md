# Transcription automatique de manuscrits médiévaux français — HTR + NLP

Projet **MD5** (Master Data/IA, HETIC) · Module *Vision par ordinateur* + *Natural Language Processing*.
Pipeline complet **image de manuscrit → texte localisé et calibré → analyse linguistique**, sur le
corpus **CATMuS Medieval** (sous-ensemble français).

- **Volet 1 — CV/HTR** : prétraitement, segmentation, transcription (TrOCR + LoRA), data contract JSON.
- **Volet 2 — NLP** : normalisation, NER, POS, graphe, export TEI *(en cours)*.

## Résultats (Volet 1)

| Indicateur (test scellé CATMuS, n=500) | Valeur | Seuil brief | Statut |
|---|---|---|---|
| CER — Character Error Rate | **13,46 %** · IC95 % [12,70 ; 14,28] | < 15 % | ✅ atteint |
| WER — Word Error Rate | 43,80 % | < 25 % | ⚠️ à améliorer |
| Lignes `needs_review` (seuil calibré 0,70) | 18,7 % | < 20 % | ✅ atteint |
| Baseline TrOCR sans fine-tuning | 76,7 % | — | référence (÷5,7) |

> Honnêteté : éval sur un **échantillon de 500 lignes** du test (pas le test complet) ;
> la **segmentation page entière** reste le maillon faible (projection : IoU/CER dégradés
> sur pages denses) — d'où le passage à **Kraken BLLA** dans le notebook renforcé.

## Structure du dépôt

```
src/htr/            modules Volet 1 (preprocessing, segmentation, metrics, data, data_contract, model, page_xml)
src/nlp/            Volet 2 (à venir)
schemas/            data_contract.schema.json (jsonschema du JSON livré au NLP)
tests/              suite pytest (preprocessing, métriques, normalisation, schéma, PAGE XML)
notebooks/          volet1_pipeline.ipynb (Colab A100, renforcé) · computer_vision_volet1.ipynb (run d'origine)
dataset_nlp/        data contract JSON généré (livrable NLP)
segmentations/      PAGE XML des pages segmentées
experiments/        journal.jsonl (journal des runs)
docs/               présentation, supports
*.md                README, MODEL_CARD, CONVENTIONS_*, DATA_SOURCES
```

## Installation

Python ≥ 3.10.

```bash
pip install -e ".[dev]"          # cœur CPU + pytest (preprocessing, métriques, schéma…)
pip install -e ".[train]"        # + TrOCR/LoRA (transformers 4.46.3, datasets 3.1.0, peft, accelerate)
pip install -e ".[kraken]"       # + segmentation Kraken BLLA (Linux)
```

## Reproduire les résultats

1. **Tests** (doivent être verts) :
   ```bash
   pytest -q
   ```
2. **Pipeline complet sur GPU** : ouvrir `notebooks/volet1_pipeline.ipynb` dans **Google Colab (A100)**
   et exécuter les cellules dans l'ordre. Le notebook clone ce dépôt, installe le package,
   lance `pytest`, charge CATMuS français, illustre le prétraitement, segmente + mesure l'IoU,
   évalue le test scellé (CER/WER + IC bootstrap), compare r=16 vs r=32 (McNemar) et génère le
   data contract validé par jsonschema.
3. **Reproductibilité** : `seed = 42` partout (`htr.seeds.fixer_seeds`), dépendances figées
   (`pyproject.toml`, extra `[train]`), **SHA-256 du train** scellé :
   `11a7b277d73e53c540aba23fdd2f7533ea7068f790e16e4699a5d69fcc87c26c`.

## Organisation de l'équipe

| Rôle | Membre |
|---|---|
| Responsable technique (structure, reproductibilité, CI) | Farid Benamara |
| Responsable CV (prétraitement, segmentation) | Ouassim Megrad |
| Responsable expérimentation (modèle, métriques, données) | Abdennour Touat |
| Responsable données / documentation (data contract, conventions) | Khaled Sellani |

Établissement : HETIC — Master Data/IA — Module MD5 « Vision par ordinateur ».

## Liens

- Modèle HuggingFace : _à publier_ (`trocr-fr-lora`, lien à ajouter)
- Corpus : [CATMuS Medieval](https://huggingface.co/datasets/CATMuS/medieval)
- Cours : [Analyse philologique](https://github.com/Septentrion/CoursAnalysePhilologique) ·
  [TALN manuscrits anciens](https://github.com/Septentrion/CoursTALN-ManuscritsAnciens)

Voir aussi `MODEL_CARD.md`, `CONVENTIONS_TRANSCRIPTION.md`, `CONVENTIONS_NLP.md`, `DATA_SOURCES.md`.
