"""
Tests de la séance 4 — drawdown.

Nouveauté de ce fichier : des tests d'INVARIANCE. On vérifie non seulement
que le calcul est juste, mais que la mesure réagit — ou ne réagit pas — aux
transformations auxquelles elle devrait être sensible.

C'est le niveau de test qui distingue une mesure comprise d'une mesure
simplement codée. Deux exemples ici :

  • la volatilité est invariante par permutation des rendements ;
    le drawdown, non. On teste les deux ensemble.

  • inverser l'ordre d'une série préserve EXACTEMENT le maximum drawdown.
    Ce n'est pas évident, c'est démontrable, et un test le verrouille.
"""

from __future__ import annotations

from itertools import pairwise, permutations

import numpy as np
import pandas as pd
import pytest

from nkrisk.data import GBMSpec, generate_prices
from nkrisk.drawdown import (
    calmar_ratio,
    drawdown_curve,
    drawdown_episodes,
    expected_max_drawdown,
    max_drawdown,
    ulcer_index,
)
from nkrisk.returns import annualized_volatility, log_returns
from nkrisk.types import InsufficientData
from nkrisk.var import historical_var


def _prix(lr: np.ndarray, start: str = "2020-01-01") -> pd.Series:
    """Reconstruit une série de prix à partir de rendements log."""
    v = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(lr))])
    return pd.Series(v, index=pd.bdate_range(start, periods=len(v)))


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — PROPRIÉTÉS ET CAS CALCULABLES À LA MAIN
# ══════════════════════════════════════════════════════════════════════


def test_serie_monotone_croissante_na_aucun_drawdown():
    p = pd.Series([100.0, 101, 103, 107, 112],
                  index=pd.bdate_range("2020-01-01", periods=5))
    assert max_drawdown(p).depth == 0.0
    assert (drawdown_curve(p) == 0).all()
    assert ulcer_index(p) == 0.0


def test_drawdown_calculable_a_la_main():
    """
    100 → 120 → 60 → 130.
    Sommet 120, creux 60, donc −50 %. Retour au-dessus de 120 au dernier point.
    """
    p = pd.Series([100.0, 120, 60, 130],
                  index=pd.bdate_range("2020-01-01", periods=4))
    e = max_drawdown(p)
    assert abs(e.depth - (-0.5)) < 1e-12
    assert e.peak_idx == 1 and e.trough_idx == 2 and e.recovery_idx == 3
    assert e.decline_days == 1 and e.recovery_days == 1
    assert e.recovered
    # Perdre 50 % exige de gagner 100 %.
    assert abs(e.gain_required - 1.0) < 1e-12


def test_asymetrie_perte_gain():
    """La table de l'asymétrie, vérifiée point par point."""
    attendu = {-0.10: 0.1111111, -0.20: 0.25, -1/3: 0.5, -0.50: 1.0, -0.80: 4.0}
    for perte, gain in attendu.items():
        p = pd.Series([100.0, 100 * (1 + perte)],
                      index=pd.bdate_range("2020-01-01", periods=2))
        assert abs(max_drawdown(p).gain_required - gain) < 1e-6


def test_drawdown_toujours_negatif_ou_nul():
    p = generate_prices(GBMSpec(0.08, 0.25), 2000, seed=1)
    dd = drawdown_curve(p)
    assert (dd <= 1e-15).all()
    assert dd.iloc[0] == 0.0


def test_non_recuperation_rend_none_et_pas_zero():
    """
    Un épisode non refermé doit retourner None, jamais une date inventée.
    Confondre « pas encore revenu » et « revenu aujourd'hui » fausse toutes
    les statistiques de durée.
    """
    p = pd.Series([100.0, 150, 80, 90, 95],
                  index=pd.bdate_range("2020-01-01", periods=5))
    e = max_drawdown(p)
    assert e.recovery is None
    assert e.recovery_idx is None
    assert e.recovery_days is None
    assert e.total_days is None
    assert not e.recovered


def test_prix_negatif_et_donnees_insuffisantes_rejetes():
    with pytest.raises(ValueError, match="négatifs"):
        max_drawdown(pd.Series([100.0, -5.0],
                               index=pd.bdate_range("2020-01-01", periods=2)))
    with pytest.raises(InsufficientData):
        max_drawdown(pd.Series([100.0], index=pd.bdate_range("2020-01-01", periods=1)))


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — INVARIANCE : CE QUI DÉPEND DU CHEMIN ET CE QUI N'EN DÉPEND PAS
# ══════════════════════════════════════════════════════════════════════


def test_la_volatilite_est_invariante_par_permutation_pas_le_drawdown():
    """
    ★ LE TEST QUI FONDE LA SÉANCE ★

    Mêmes rendements, ordres différents. Volatilité, VaR et CVaR sont
    identiques au flottant près — elles ne décrivent que la distribution.
    Le drawdown, lui, varie du simple au double ou plus.
    """
    base = np.random.default_rng(7).normal(0.0002, 0.0126, 1260)
    rng = np.random.default_rng(1)

    vols, vars_, mdds = [], [], []
    for _ in range(40):
        p = _prix(rng.permutation(base))
        lr = log_returns(p)
        vols.append(annualized_volatility(lr).value)
        vars_.append(historical_var(lr, confidence=0.99).var)
        mdds.append(max_drawdown(p).depth)

    # La distribution ne bouge pas : écart relatif sous 1e-12
    assert (max(vols) - min(vols)) / np.mean(vols) < 1e-12
    assert (max(vars_) - min(vars_)) / np.mean(vars_) < 1e-12

    # Le chemin, lui, change tout : au moins 20 % d'écart entre extrêmes
    assert (min(mdds) / max(mdds)) > 1.2, (
        f"drawdowns trop proches : {min(mdds):.3f} à {max(mdds):.3f}"
    )


def test_inverser_la_serie_preserve_exactement_le_drawdown():
    """
    ★ UNE IDENTITÉ QUI SURPREND, ET QUI SE DÉMONTRE ★

    En rendements log, le drawdown maximal est le maximum, sur toutes les
    fenêtres [i,j], de −(somme des rendements sur la fenêtre).

    Inverser la série envoie la fenêtre [i,j] sur [n−j, n−i], dont la
    somme est la même. L'ensemble des sommes de fenêtres est donc
    identique, donc son maximum aussi.

    Conséquence : le drawdown dépend du chemin, mais pas de son SENS.
    """
    for seed in range(12):
        lr = np.random.default_rng(seed).normal(0.0002, 0.012, 400)
        a = max_drawdown(_prix(lr)).depth
        b = max_drawdown(_prix(lr[::-1])).depth
        assert abs(a - b) < 1e-12, f"seed {seed} : {a} vs {b}"


def _mdd_rapide(log_returns: np.ndarray) -> np.ndarray:
    """
    Oracle vectorisé : maximum drawdown de chaque LIGNE d'une matrice de
    rendements log.

    ★ POURQUOI CE N'EST PAS UN CONTOURNEMENT DU TEST ★

    Le test d'énumération appelait `max_drawdown` sur les 40 320
    permutations, en construisant chaque fois une `pd.Series` avec un
    index de dates ouvrables. Mesuré : 6,9 secondes, dont moins de 1 %
    de calcul réel. Tout le reste était de l'allocation pandas dans une
    boucle chaude.

    La correction naïve — réécrire le test en numpy — testerait une
    réimplémentation au lieu de l'API publique. Inacceptable.

    Schéma retenu : **l'oracle vérifié**. On contrôle d'abord que cette
    fonction donne exactement le même résultat que `max_drawdown`, puis
    on s'en sert pour la recherche exhaustive. La revendication reste
    prouvée sur les 40 320 ordres, et l'API publique reste testée.

    6,94 s → 0,02 s, soit 351×, à résultat identique au flottant près.
    """
    # Le 1.0 initial n'est pas cosmétique : `_prix` démarre la série à 100
    # avant d'appliquer le premier rendement, donc ce point est un sommet
    # candidat. Sans lui, une baisse en tout début de série est invisible.
    # C'est exactement l'écart que le test de concordance a attrapé.
    eq = np.exp(np.cumsum(log_returns, axis=1))
    eq = np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1)
    return (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)


def test_loracle_rapide_concorde_avec_lapi_publique():
    """Condition de validité du test suivant : sans cet accord, l'oracle
    ne prouverait rien sur `max_drawdown`."""
    rng = np.random.default_rng(5)
    lots = rng.normal(0, 0.02, size=(40, 12))
    rapide = _mdd_rapide(lots)
    for i, lr in enumerate(lots):
        assert abs(rapide[i] - max_drawdown(_prix(lr)).depth) < 1e-12


def test_trier_les_gains_en_premier_maximise_le_drawdown():
    """
    ★ DÉMONTRÉ PAR ÉNUMÉRATION EXHAUSTIVE ★

    Parmi toutes les permutations des mêmes rendements, celle qui place
    tous les gains d'abord produit le drawdown maximal : elle porte le
    sommet aussi haut que possible avant d'enchaîner toutes les pertes.
    Les 40 320 ordres de 8 rendements sont énumérés en totalité.
    """
    for seed in (11, 23, 57):
        v = np.random.default_rng(seed).normal(0, 0.03, 8)
        ordres = np.array(list(permutations(v)))
        pire = float(_mdd_rapide(ordres).min())
        tri = max_drawdown(_prix(np.sort(v)[::-1])).depth  # via l'API publique
        assert abs(pire - tri) < 1e-12, f"seed {seed} : {pire} vs {tri}"


def test_episodes_tries_et_coherents_avec_le_maximum():
    p = generate_prices(GBMSpec(0.08, 0.30), 2520, seed=5)
    eps = drawdown_episodes(p, min_depth=0.05)
    assert len(eps) >= 3
    assert all(eps[i].depth <= eps[i + 1].depth for i in range(len(eps) - 1))
    assert abs(eps[0].depth - max_drawdown(p).depth) < 1e-12
    for e in eps:
        assert e.peak_idx <= e.trough_idx
        if e.recovery_idx is not None:
            assert e.trough_idx < e.recovery_idx


def test_ulcere_distingue_la_duree_a_profondeur_egale():
    """
    ★ TEST ÉCRIT APRÈS QUE LE PRÉCÉDENT EUT ÉTÉ DÉMOLI ★

    J'avais d'abord testé que l'indice d'ulcère est plus STABLE que le
    maximum drawdown, par analogie avec CVaR contre VaR. La mesure a dit
    l'inverse : coefficient de variation 0,47 contre 0,30. L'analogie
    était fausse.

    Voici la propriété qui, elle, est vraie et qui justifie l'indice :
    deux trajectoires de MÊME maximum drawdown mais de durées sous l'eau
    différentes ont des ulcères différents. C'est précisément ce que le
    maximum ne voit pas.
    """
    def idx(n):
        return pd.bdate_range("2020-01-01", periods=n)

    # Chute de 30 %, remontée immédiate
    rapide = pd.Series([100.0, 70.0] + [100.0] * 40, index=idx(42))
    # Même chute de 30 %, mais quarante jours au fond
    lente = pd.Series([100.0] + [70.0] * 40 + [100.0], index=idx(42))

    assert abs(max_drawdown(rapide).depth - max_drawdown(lente).depth) < 1e-12
    assert ulcer_index(lente) > ulcer_index(rapide) * 3, (
        f"ulcère rapide {ulcer_index(rapide):.4f} vs lent {ulcer_index(lente):.4f}"
    )


def test_ulcere_est_plus_disperse_que_le_maximum():
    """
    Verrouille le résultat qui a corrigé ma docstring, pour qu'une future
    réécriture ne réintroduise pas l'affirmation inverse.
    """
    u, m = [], []
    for seed in range(80):
        p = generate_prices(GBMSpec(0.08, 0.20), 1260, seed=seed)
        u.append(ulcer_index(p))
        m.append(abs(max_drawdown(p).depth))
    def cv(x):
        return float(np.std(x) / np.mean(x))
    assert cv(u) > cv(m), (
        f"ulcère {cv(u):.3f} devrait être plus dispersé que maximum {cv(m):.3f}"
    )


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — CALIBRATION DE LA SIMULATION
# ══════════════════════════════════════════════════════════════════════


def test_le_drawdown_espere_croit_avec_lhorizon():
    """
    Même avec une dérive POSITIVE, la médiane du drawdown s'approfondit
    avec la durée de détention. Plus on détient longtemps, plus on
    multiplie les occasions de traverser une forte baisse.
    """
    medianes = [expected_max_drawdown(0.20, ans, drift=0.08,
                                      n_sim=1500, seed=4).value
                for ans in (1, 2, 5, 10, 20)]
    for a, b in pairwise(medianes):
        assert b < a, f"non monotone : {medianes}"
    assert medianes[0] < -0.10
    assert medianes[-1] < -0.40


def test_lintervalle_du_drawdown_est_enorme():
    """
    Le point le plus important de la séance : le maximum drawdown ne
    discrimine presque rien. Sur cinq ans, l'intervalle à 90 % couvre un
    facteur supérieur à deux.
    """
    e = expected_max_drawdown(0.20, 5, drift=0.08, n_sim=4000, seed=9)
    assert e.ci_low < e.value < e.ci_high  # ci_low = le plus profond
    rapport = e.ci_low / e.ci_high
    assert rapport > 2.0, f"rapport seulement {rapport:.2f}"
    assert not e.is_meaningful()


def test_plus_de_volatilite_donne_un_drawdown_plus_profond():
    vals = [expected_max_drawdown(v, 5, drift=0.08, n_sim=1200, seed=2).value
            for v in (0.10, 0.20, 0.35)]
    assert vals[0] > vals[1] > vals[2]


def test_simulation_reproductible():
    a = expected_max_drawdown(0.20, 3, n_sim=800, seed=77)
    b = expected_max_drawdown(0.20, 3, n_sim=800, seed=77)
    assert a.value == b.value and a.ci_low == b.ci_low


def test_calmar_coherent_avec_ses_composantes():
    p = generate_prices(GBMSpec(0.12, 0.20), 2520, seed=3)
    c = calmar_ratio(p)
    n = len(p) - 1
    annuel = float(p.iloc[-1] / p.iloc[0]) ** (252 / n) - 1
    assert abs(c - annuel / abs(max_drawdown(p).depth)) < 1e-12
