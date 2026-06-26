"""Tests du data contract : construction + validation jsonschema.

Exigence brief CV §5 / NLP §1 : cohérence du schéma JSON du data contract.
"""

import jsonschema
import pytest

from htr import data_contract as DC


def _ligne_valide(idx=42, conf=0.91):
    return DC.construire_ligne(
        idx=idx,
        texte="creature qui croira",
        polygone=[[0, 0], [100, 0], [100, 30], [0, 30]],
        confiance=conf,
        confiance_char=[0.99, 0.62],
        candidats_char=[["c", "e"], ["r", "n"]],
        image_source="Paris, BnF, fr. 413",
        metadonnees={"siecle": 13, "script_type": "Textualis"},
        baseline=[[0, 30], [100, 30]],
    )


def _document_valide():
    return DC.construire_document(
        lignes=[_ligne_valide()],
        corpus="CATMuS Medieval (français)",
        modele="trocr-fr-lora (r=16)",
        train_sha256="11a7b277d73e53c540aba23fdd2f7533ea7068f790e16e4699a5d69fcc87c26c",
    )


def test_construire_ligne_id_et_review():
    ligne = _ligne_valide(idx=42, conf=0.91)
    assert ligne["id"] == "line_00042"
    assert ligne["needs_review"] is False


def test_needs_review_si_confiance_basse():
    ligne = _ligne_valide(conf=0.50)
    assert ligne["needs_review"] is True


def test_document_valide_passe_le_schema():
    DC.valider(_document_valide())  # ne doit rien lever


def test_document_invalide_rejete():
    doc = _document_valide()
    del doc["metadata"]["train_sha256"]  # champ requis manquant
    with pytest.raises(jsonschema.ValidationError):
        DC.valider(doc)


def test_sha256_malforme_rejete():
    doc = _document_valide()
    doc["metadata"]["train_sha256"] = "pas-un-hash"
    with pytest.raises(jsonschema.ValidationError):
        DC.valider(doc)
