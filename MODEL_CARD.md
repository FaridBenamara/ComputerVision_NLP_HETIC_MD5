# Model Card — `trocr-fr-lora` (HTR manuscrits médiévaux français)

## Résumé
Adaptateur **LoRA** spécialisant `microsoft/trocr-base-handwritten` sur la transcription
de **lignes** de manuscrits médiévaux **français** (CATMuS Medieval). Encodeur ViT +
décodeur autoregressif ; seules de petites matrices de bas rang dans l'attention sont
entraînées (≈0,63 % des paramètres).

## Données d'entraînement
- **Corpus** : CATMuS Medieval, sous-ensemble français (fichiers `L-Fre*`).
- **Splits officiels** (`gen_split`) : 49 201 train / 2 712 validation / 4 135 test.
- **Convention** : semi-diplomatique, abréviations conservées, normalisation Unicode NFC.
- **Scellement** : SHA-256 du train `11a7b277d73e53c540aba23fdd2f7533ea7068f790e16e4699a5d69fcc87c26c`, seed 42.

## Entraînement
| Hyperparamètre | Valeur |
|---|---|
| Base | `microsoft/trocr-base-handwritten` |
| LoRA | r = 16, α = 32, dropout = 0,05, cibles `query/value/q_proj/v_proj` |
| Params entraînés | 2 113 536 / 336 035 328 (0,63 %) |
| Optim | lr 1e-4, warmup 0,05, 3 epochs, batch 16, bf16 |
| Matériel / durée | 1× A100 40 Go, ~50 min (9 228 steps) |
| Courbe val | CER 32,7 % (800 steps) → 16,6 % en fin d'entraînement |

Variante en cours : **r = 32, α = 64 + augmentation** (contraste, rotation ±2°, bruit) —
comparaison r=16 vs r=32 par **test de McNemar** (notebook renforcé).

## Performances (test scellé, n = 500)
| Métrique | Valeur |
|---|---|
| CER | **13,46 %** (IC95 % bootstrap [12,70 ; 14,28], N=1000) |
| WER | 43,80 % |
| Baseline (sans fine-tuning) | CER 76,7 % |

## Limitations & biais
- **Biais corpus** : pic 13ᵉ–15ᵉ s., écriture *Textualis* dominante → moins fiable sur
  écritures rares et 11ᵉ–12ᵉ s.
- **WER élevé** (43,8 %) : cohérent avec le médiéval (une lettre fausse invalide le mot).
- **Segmentation** : le modèle transcrit des **lignes** ; la qualité sur page entière dépend
  fortement de la segmentation amont (projection grossière → CER dégradé ; passage à Kraken BLLA).
- **Éval** : sur un échantillon de 500 lignes, pas le test complet.
- Erreurs résiduelles typiques : confusions de lettres visuellement proches (c/e, n/u).

## Usage prévu
Transcription assistée de manuscrits médiévaux français (recherche, humanités numériques).
Sortie destinée au **Volet 2 NLP** via le data contract (confiance + candidats par caractère,
drapeau `needs_review` calibré à 0,70). Usage recherche non commerciale.
