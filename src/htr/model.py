"""Modèle HTR : TrOCR (microsoft/trocr-base-handwritten) spécialisé par LoRA.

On part d'un encodeur visuel ViT + décodeur déjà pré-entraînés sur de l'écriture
manuscrite, et on n'entraîne que de petites matrices LoRA de bas rang dans
l'attention (encodeur ET décodeur) : ≈0,6 % des paramètres, adaptateur de
quelques Mo, moins de surapprentissage, entraînement sur une seule A100.

Toutes les fonctions importent torch/transformers/peft paresseusement : le
module reste importable (pour la doc / l'introspection) hors environnement GPU.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

MODEL_ID = "microsoft/trocr-base-handwritten"


def construire_trocr_lora(
    model_id: str = MODEL_ID,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] = ("query", "value", "q_proj", "v_proj"),
) -> tuple[Any, Any]:
    """Construit un modèle TrOCR enveloppé de LoRA + son processor.

    Configure les tokens spéciaux (nécessaires en entraînement ET génération),
    aligne ``vocab_size`` sur le décodeur, puis injecte LoRA sur l'attention.

    Args:
        model_id: Checkpoint TrOCR de base.
        r: Rang LoRA (8/16/32…). Le run de référence utilise r=16.
        lora_alpha: Facteur d'échelle LoRA (ratio alpha/r = 2 conseillé).
        lora_dropout: Dropout sur les adaptateurs.
        target_modules: Modules d'attention ciblés (ViT ``query/value`` + TrOCR ``q_proj/v_proj``).

    Returns:
        Tuple ``(model, processor)`` ; ``model`` est un ``PeftModel`` prêt à entraîner.

    Raises:
        ImportError: Si transformers/peft ne sont pas installés.
    """
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("construire_trocr_lora nécessite transformers et peft.") from exc

    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    tok = processor.tokenizer
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = tok.cls_token_id
        cfg.pad_token_id = tok.pad_token_id
        cfg.eos_token_id = tok.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model = get_peft_model(
        model,
        LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=list(target_modules),
        ),
    )
    model.print_trainable_parameters()
    return model, processor


def sauver_adaptateurs(model: Any, processor: Any, chemin: str) -> None:
    """Sauve les adaptateurs LoRA + le processor (léger, ~quelques Mo).

    ``save_embedding_layers=False`` est requis : il évite la détection auto
    d'embedding qui fait planter la sauvegarde d'un VisionEncoderDecoderModel PEFT.

    Args:
        model: PeftModel entraîné.
        processor: TrOCRProcessor associé.
        chemin: Dossier de destination (créé si besoin).
    """
    model.save_pretrained(chemin, save_embedding_layers=False)
    processor.save_pretrained(chemin)


def collate_factory(processor: Any) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Fabrique une fonction de collation (images PIL -> pixel_values ; textes -> labels).

    Les tokens de padding des labels sont remplacés par ``-100`` pour être
    ignorés par la perte (alignement labels/tokenisation — point critique).

    Args:
        processor: TrOCRProcessor (fournit feature extractor + tokenizer).

    Returns:
        Fonction ``collate(batch) -> {"pixel_values", "labels"}``.
    """
    tok = processor.tokenizer
    pad = tok.pad_token_id

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [ex["im"].convert("RGB") for ex in batch]
        textes = [ex["text"] for ex in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        labels = tok(textes, padding=True, return_tensors="pt").input_ids
        labels[labels == pad] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    return collate


def metrics_factory(tokenizer: Any) -> Callable[[Any], dict[str, float]]:
    """Fabrique un ``compute_metrics`` robuste (CER/WER) pour le Seq2SeqTrainer.

    Args:
        tokenizer: Tokenizer TrOCR.

    Returns:
        Fonction ``compute_metrics(pred) -> {"cer", "wer"}``.
    """
    import numpy as np

    from . import metrics as M

    vocab, pad = tokenizer.vocab_size, tokenizer.pad_token_id

    def _clean(ids: Any) -> Any:
        ids = np.asarray(ids[0] if isinstance(ids, tuple) else ids)
        return np.where((ids < 0) | (ids >= vocab), pad, ids)

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_str = tokenizer.batch_decode(_clean(pred.predictions), skip_special_tokens=True)
        labels = np.where(np.asarray(pred.label_ids) == -100, pad, pred.label_ids)
        label_str = tokenizer.batch_decode(_clean(labels), skip_special_tokens=True)
        return {"cer": M.cer(label_str, pred_str), "wer": M.wer(label_str, pred_str)}

    return compute_metrics


def transcrire(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    device: str = "cuda",
    batch_size: int = 16,
    max_new_tokens: int = 128,
) -> list[str]:
    """Transcrit une liste d'images PIL (lignes) en texte.

    Args:
        model: Modèle TrOCR (PeftModel ou VED) en mode eval.
        processor: TrOCRProcessor.
        images: Images PIL de lignes de texte.
        device: "cuda" ou "cpu".
        batch_size: Taille de lot d'inférence.
        max_new_tokens: Longueur max générée par ligne.

    Returns:
        Liste des transcriptions (une str par image).
    """
    import torch

    model.eval()
    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            pv = processor(images=batch, return_tensors="pt").pixel_values.to(device)
            ids = model.generate(pv, max_new_tokens=max_new_tokens)
            preds.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return preds


def transcrire_riche(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    device: str = "cuda",
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> list[dict[str, Any]]:
    """Transcrit avec confiance par caractère et candidats top-2 (pour le NLP).

    Pour chaque position générée, la probabilité softmax du token top-1 est
    attribuée à chacun de ses caractères, et on retient les 2 meilleurs candidats
    de token. Ces signaux alimentent la correction NLP guidée par la confiance.

    Args:
        model: Modèle TrOCR en eval.
        processor: TrOCRProcessor.
        images: Images PIL de lignes.
        device: "cuda" ou "cpu".
        batch_size: Taille de lot.
        max_new_tokens: Longueur max générée.

    Returns:
        Liste de dicts ``{"texte", "conf_char", "cand_char", "conf"}`` par ligne.
    """
    import numpy as np
    import torch

    tok = processor.tokenizer
    model.eval()
    resultats: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            pv = processor(images=batch, return_tensors="pt").pixel_values.to(device)
            out = model.generate(
                pv, max_new_tokens=max_new_tokens, output_scores=True,
                return_dict_in_generate=True,
            )
            probs = [torch.softmax(s, dim=-1) for s in out.scores]
            for b in range(len(batch)):
                gen = out.sequences[b][1:]  # on saute decoder_start
                chars, confs, cands = [], [], []
                for t, tid in enumerate(gen):
                    if t >= len(probs):
                        break
                    tid = int(tid)
                    if tid in (tok.pad_token_id, tok.eos_token_id):
                        break
                    top2 = torch.topk(probs[t][b], 2)
                    piece = tok.decode([tid], skip_special_tokens=True)
                    for c in piece:
                        chars.append(c)
                        confs.append(round(float(top2.values[0]), 4))
                        cands.append([
                            tok.decode([int(top2.indices[0])]).strip(),
                            tok.decode([int(top2.indices[1])]).strip(),
                        ])
                resultats.append({
                    "texte": "".join(chars),
                    "conf_char": confs,
                    "cand_char": cands,
                    "conf": round(float(np.mean(confs)), 4) if confs else 0.0,
                })
    return resultats
