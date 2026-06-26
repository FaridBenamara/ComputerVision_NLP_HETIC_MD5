"""Tests des métriques : CER, WER, bootstrap, McNemar, IoU."""

import numpy as np

from htr import metrics as M


def test_cer_identique_est_zero():
    assert M.cer("creature qui croira", "creature qui croira") == 0.0


def test_cer_une_substitution():
    assert M.cer("abc", "abd") == 1 / 3


def test_wer_un_mot_faux():
    assert M.wer("a b c", "a x c") == 1 / 3


def test_cer_par_ligne_longueur():
    refs = ["abc", "defg"]
    hyps = ["abc", "defg"]
    out = M.cer_par_ligne(refs, hyps)
    assert isinstance(out, np.ndarray)
    assert len(out) == 2
    assert out[0] == 0.0


def test_bootstrap_ic_encadre_et_reproductible():
    refs = [f"ligne numero {i}" for i in range(40)]
    hyps = [r if i % 3 else r + "x" for i, r in enumerate(refs)]
    a = M.bootstrap_ic_cer(refs, hyps, n_resamples=500, seed=42)
    b = M.bootstrap_ic_cer(refs, hyps, n_resamples=500, seed=42)
    assert a == b  # même graine => même résultat (reproductibilité)
    assert a["ic_bas"] <= a["cer"] <= a["ic_haut"]


def test_mcnemar_a_parfait_b_nul():
    refs = [str(i) for i in range(10)]
    a = list(refs)                 # A : 10/10 corrects
    b = ["?" for _ in refs]        # B : 0/10 corrects
    res = M.mcnemar(refs, a, b)
    assert res["n10"] == 10 and res["n01"] == 0
    assert res["p_value"] < 0.05


def test_mcnemar_systemes_identiques():
    refs = [str(i) for i in range(10)]
    res = M.mcnemar(refs, refs, refs)
    assert res["n_discordants"] == 0
    assert res["p_value"] == 1.0


def test_iou_identique_et_disjoint():
    carre = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert M.iou_polygones(carre, carre) == 1.0
    loin = [(100, 100), (110, 100), (110, 110), (100, 110)]
    assert M.iou_polygones(carre, loin) == 0.0
