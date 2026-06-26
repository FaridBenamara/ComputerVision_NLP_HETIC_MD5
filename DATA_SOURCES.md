# Sources de données, licences et attribution

## Corpus principal — CATMuS Medieval
- **Source** : <https://huggingface.co/datasets/CATMuS/medieval>
- **Sous-ensemble utilisé** : français (fichiers `L-Fre*`), splits officiels `gen_split`
  (train 49 201 / validation 2 712 / test 4 135 lignes).
- **Nature** : lignes de manuscrits médiévaux déjà transcrites par des spécialistes,
  images majoritairement issues de la **BnF / Gallica**.
- **Projet** : CATMuS (Consistent Approach to Transcribing ManuScript) fédérant CREMMA,
  GalliCorpora, HTRomance, DEEDS.
- **Licence** : corpus de référence sous **licence ouverte** (vérifier la mention exacte sur
  la fiche HuggingFace et la citer dans l'article ; n'utiliser que des données libres pour
  l'entraînement — brief CV §7).
- **Restriction projet** : restreint au français pour cohérence linguistique du décodeur et
  téléchargement ciblé (~20 Go au lieu de ~96 Go).

## Modèles pré-entraînés
- `microsoft/trocr-base-handwritten` — <https://huggingface.co/microsoft/trocr-base-handwritten>
  (usage recherche).
- Kraken BLLA (segmentation de lignes) — <https://kraken.re> ; modèles HTR-United.

## Test de généralisation (hors corpus)
- **Codex Manesse** (allemand, vers 1300) — utilisé uniquement comme test qualitatif de
  généralisation visuelle (jamais vu à l'entraînement, pas dans les métriques).

## Scellement / anti-contamination
- SHA-256 du train : `11a7b277d73e53c540aba23fdd2f7533ea7068f790e16e4699a5d69fcc87c26c`
- Seed global : 42.

## À faire avant rendu
- [ ] Citer la **licence exacte** de CATMuS Medieval et de chaque manuscrit réutilisé.
- [ ] Ajouter le lien du modèle publié sur HuggingFace.
- [ ] Vérifier les droits des images BnF/Gallica réutilisées dans l'article.
