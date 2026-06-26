"""Tests du nettoyage / normalisation du corpus et du scellement SHA-256."""

from htr import data as D


def test_nettoyer_texte_applique_nfc():
    # 'é' décomposé (e + accent combinant U+0301) -> 'é' composé (U+00E9)
    decompose = "créature"
    assert D.nettoyer_texte(decompose) == "créature"


def test_nettoyer_texte_espaces():
    assert D.nettoyer_texte("  a   b  ") == "a b"


def test_nettoyer_texte_preserve_signes_medievaux():
    # Le tironien ⁊ et le tilde combinant doivent être conservés (semi-diplomatique)
    out = D.nettoyer_texte("q̃ ⁊ a")
    assert "⁊" in out
    assert "̃" in out  # tilde nasal préservé


def test_nettoyer_texte_none():
    assert D.nettoyer_texte(None) == ""


def test_nettoyer_corpus_rapport():
    gardes, rapport = D.nettoyer_corpus(["abc", "", "  "])
    assert gardes == [0]
    assert rapport == {"avant": 3, "apres": 1, "rejetes": 2}


def test_sha256_deterministe():
    textes = ["alpha", "beta"]
    cotes = ["BnF fr. 1", "BnF fr. 2"]
    h1 = D.sha256_split(textes, cotes)
    h2 = D.sha256_split(textes, cotes)
    assert h1 == h2
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)


def test_sha256_change_si_texte_change():
    base = D.sha256_split(["a", "b"])
    autre = D.sha256_split(["a", "c"])
    assert base != autre
