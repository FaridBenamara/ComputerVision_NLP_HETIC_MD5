# Conventions de transcription (Volet 1 — HTR)

## Niveau de transcription : **semi-diplomatique**
On reste fidèle à la graphie du manuscrit tout en garantissant un encodage cohérent.

| Choix | Règle | Pourquoi |
|---|---|---|
| Encodage | Unicode **NFC** (forme composée canonique) systématique | Graphies équivalentes encodées de façon identique ; comparaisons et métriques fiables |
| Abréviations | **Conservées** telles quelles (⁊, q̃, ꝑ, ꝗ, tilde nasal…) | La résolution (`q̃` → `que`) est un choix linguistique laissé au **Volet 2 / NLP** |
| Casse | Conservée telle qu'écrite | Pas de sur-interprétation éditoriale au niveau HTR |
| Espaces | Espaces de bord supprimés, suites d'espaces réduites à une | Bruit de segmentation, sans toucher au contenu |
| u/v, i/j | **Non** normalisés au niveau HTR | Variantes graphiques d'époque ; normalisation = règle déterministe du Volet 2 |
| Lacunes / illisible | Marquées via `needs_review` (confiance < 0,70) plutôt que par un symbole éditorial | Décision déléguée à la relecture humaine guidée par la confiance |

## Implémentation
- `htr.data.nettoyer_texte` : NFC + nettoyage des espaces, **signes médiévaux préservés**
  (testé dans `tests/test_data.py`).
- Le data contract conserve le texte semi-diplomatique ET fournit, par caractère, la confiance
  et les candidats alternatifs pour la correction NLP.

## Système de coordonnées (polygones)
- Origine **en haut à gauche**, unité **pixels** (déclaré dans `metadata.systeme_coordonnees`
  du data contract et dans le PAGE XML).
