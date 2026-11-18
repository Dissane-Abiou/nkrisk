"""
Tests de la séance 5 — portefeuille et tests de scénario.

Ce fichier teste surtout des DÉCISIONS DE CONCEPTION, pas des formules :
l'immutabilité, la conformité au protocole, la composition, la
préservation d'un invariant mathématique sous transformation.

C'est le type de test qu'on écrit quand l'architecture est le produit.
"""

from __future__ import annotations

import itertools
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from nkrisk.covariance import CovarianceEstimate, ledoit_wolf
from nkrisk.data import GBMSpec, generate_correlated_prices
from nkrisk.portfolio import Portfolio, equal_weighted
from nkrisk.returns import simple_returns
from nkrisk.stress import (
    CRISE_2008,
    Combined,
    CorrelationShock,
    PriceShock,
    Shock,
    VolatilityShock,
    stress,
    stress_table,
)

NOMS = ["TECH_A", "TECH_B", "TECH_C", "TECH_D", "BANQUE", "OR", "OBLIG"]
VOLS = np.array([0.35, 0.38, 0.32, 0.40, 0.25, 0.18, 0.06])
CORR = np.array([
    [1.00, 0.85, 0.82, 0.88, 0.45, 0.05, -0.10],
    [0.85, 1.00, 0.80, 0.86, 0.42, 0.03, -0.08],
    [0.82, 0.80, 1.00, 0.83, 0.40, 0.06, -0.12],
    [0.88, 0.86, 0.83, 1.00, 0.44, 0.04, -0.09],
    [0.45, 0.42, 0.40, 0.44, 1.00, 0.10, -0.15],
    [0.05, 0.03, 0.06, 0.04, 0.10, 1.00, 0.20],
    [-0.10, -0.08, -0.12, -0.09, -0.15, 0.20, 1.00]])


@pytest.fixture(scope="module")
def cov() -> CovarianceEstimate:
    specs = {n: GBMSpec(0.08, float(v)) for n, v in zip(NOMS, VOLS, strict=True)}
    prix = generate_correlated_prices(specs, CORR, 252 * 8, seed=2024)
    return ledoit_wolf(simple_returns(prix))


@pytest.fixture
def ptf(cov) -> Portfolio:
    w = pd.Series([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10], index=NOMS)
    return Portfolio(weights=w, covariance=cov, name="départ")


# ══════════════════════════════════════════════════════════════════════
#  L'OBJET PORTEFEUILLE
# ══════════════════════════════════════════════════════════════════════


def test_validation_a_la_construction(cov):
    """L'invariant « ces poids vont avec cette covariance » est appliqué ici,
    une fois, et plus jamais ailleurs."""
    with pytest.raises(ValueError, match="poids absents"):
        Portfolio(weights=pd.Series([0.5, 0.5], index=NOMS[:2]), covariance=cov)

    with pytest.raises(ValueError, match="actifs absents de la covariance"):
        w = pd.Series([1 / 8] * 8, index=[*NOMS, "INCONNU"])
        Portfolio(weights=w, covariance=cov)

    with pytest.raises(ValueError, match="somment"):
        Portfolio(weights=pd.Series([0.2] * 7, index=NOMS), covariance=cov)

    with pytest.raises(ValueError, match="non positive"):
        Portfolio(weights=pd.Series([1 / 7] * 7, index=NOMS),
                  covariance=cov, value=0.0)


def test_les_poids_sont_realignes_sur_la_covariance(cov):
    """
    Le désalignement d'indices est l'erreur la plus insidieuse du domaine :
    elle produit un résultat plausible et complètement faux. On réaligne
    explicitement au lieu de supposer le même ordre.
    """
    melange = list(reversed(NOMS))
    w = pd.Series([0.10, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], index=melange)
    p = Portfolio(weights=w, covariance=cov)

    assert list(p.weights.index) == list(cov.assets)
    assert abs(p.weights["OBLIG"] - 0.10) < 1e-12
    # même portefeuille économique, saisi dans l'autre ordre → même risque
    direct = Portfolio(
        weights=pd.Series([0.15] * 6 + [0.10], index=NOMS), covariance=cov)
    assert abs(p.volatility - direct.volatility) < 1e-12


def test_le_portefeuille_est_immuable(ptf):
    # FrozenInstanceError et pas Exception : un `raises(Exception)` aveugle
    # passerait aussi sur un AttributeError dû à une faute de frappe dans
    # le nom du champ, et le test ne prouverait plus rien.
    with pytest.raises(FrozenInstanceError):
        ptf.value = 2.0
    avant = ptf.weights.copy()
    autre = ptf.with_weights(pd.Series([1 / 7] * 7, index=NOMS))
    pd.testing.assert_series_equal(ptf.weights, avant)
    assert autre is not ptf


def test_equipondere_somme_a_un(cov):
    p = equal_weighted(cov)
    assert abs(p.weights.sum() - 1.0) < 1e-12
    assert len(set(p.weights.round(12))) == 1


# ══════════════════════════════════════════════════════════════════════
#  LE PROTOCOLE
# ══════════════════════════════════════════════════════════════════════


def test_tous_les_chocs_respectent_le_protocole():
    for s in (VolatilityShock(2.0), CorrelationShock(0.5),
              PriceShock({"OR": -0.1}), CRISE_2008):
        assert isinstance(s, Shock)


def test_une_classe_externe_est_un_choc_sans_rien_heriter(ptf):
    """
    ★ CE QUE LE TYPAGE STRUCTUREL ACHÈTE ★

    Cette classe n'importe rien de nkrisk.stress et n'hérite de rien.
    Elle est pourtant un Shock valide, parce qu'elle a la bonne forme.
    Une classe de base abstraite aurait imposé un héritage.
    """

    class MonChoc:
        name = "choc maison"

        def apply(self, p: Portfolio) -> Portfolio:
            return p.rescaled(p.value * 0.5)

    c = MonChoc()
    assert isinstance(c, Shock)
    r = stress(ptf, c)
    assert abs(r.pnl - (-0.5)) < 1e-12


# ══════════════════════════════════════════════════════════════════════
#  CHOC DE VOLATILITÉ
# ══════════════════════════════════════════════════════════════════════


def test_choc_uniforme_multiplie_exactement_la_volatilite(ptf):
    for k in (0.5, 1.5, 2.0, 3.0):
        apres = VolatilityShock(k).apply(ptf)
        assert abs(apres.volatility / ptf.volatility - k) < 1e-12


def test_choc_de_volatilite_laisse_les_correlations_intactes(ptf):
    """
    Le piège silencieux : appliquer le facteur directement à Σ revient à
    l'appliquer au carré ET déforme les corrélations. On passe donc par
    la décomposition Σ = D·C·D.
    """
    avant = ptf.covariance.correlation().to_numpy()
    apres = VolatilityShock({"TECH_A": 3.0, "OR": 0.5}).apply(ptf)
    np.testing.assert_allclose(apres.covariance.correlation().to_numpy(),
                               avant, rtol=1e-12, atol=1e-14)


def test_choc_cible_ne_touche_que_les_actifs_nommes(ptf):
    apres = VolatilityShock({"OBLIG": 2.0}).apply(ptf)
    sd_av = np.sqrt(np.diag(ptf.covariance.matrix))
    sd_ap = np.sqrt(np.diag(apres.covariance.matrix))
    i = list(ptf.assets).index("OBLIG")
    for j in range(len(sd_av)):
        attendu = 2.0 if j == i else 1.0
        assert abs(sd_ap[j] / sd_av[j] - attendu) < 1e-12


def test_multiplicateur_invalide_rejete():
    with pytest.raises(ValueError, match="positif"):
        VolatilityShock(0.0)
    with pytest.raises(ValueError, match="positif"):
        VolatilityShock({"OR": -1.0})


def test_actif_inconnu_rejete(ptf):
    with pytest.raises(ValueError, match="inconnus"):
        VolatilityShock({"BITCOIN": 2.0}).apply(ptf)


# ══════════════════════════════════════════════════════════════════════
#  CHOC DE CORRÉLATION — LA CONTRAINTE QUI A DICTÉ L'ALGORITHME
# ══════════════════════════════════════════════════════════════════════


def test_la_combinaison_convexe_reste_valide_a_toute_intensite(ptf):
    """
    ★ L'INVARIANT QUI JUSTIFIE LE CHOIX D'IMPLÉMENTATION ★

    Une matrice de corrélation doit rester semi-définie positive, sinon
    elle ne décrit aucun monde possible. La combinaison convexe le
    garantit par construction ; on le vérifie sur toute la plage et pour
    plusieurs cibles, y compris négatives.
    """
    for toward in (-0.15, 0.0, 0.5, 0.9, 1.0):  # −0,15 > −1/6, la borne à 7 actifs
        for a in np.linspace(0, 1, 21):
            apres = CorrelationShock(float(a), toward=toward).apply(ptf)
            lam = np.linalg.eigvalsh(apres.covariance.matrix).min()
            assert lam >= -1e-10, f"toward={toward} α={a} → λmin={lam:.2e}"
            diag = apres.covariance.correlation().to_numpy().diagonal()
            np.testing.assert_allclose(diag, 1.0, atol=1e-12)


def test_lapproche_naive_casserait_la_validite():
    """
    ★ LE CONTRE-EXEMPLE QUI DOCUMENTE POURQUOI ON NE FAIT PAS SIMPLE ★

    Ajouter un delta à toutes les corrélations est la façon intuitive de
    « monter les corrélations ». Elle produit une matrice invalide dès
    +0,3 sur cette matrice. Ce test fige le contre-exemple pour que
    personne ne « simplifie » l'implémentation un jour.
    """
    assert np.linalg.eigvalsh(CORR).min() > 0  # la matrice de départ est saine

    naive = np.clip(CORR + 0.30, -1, 1)
    np.fill_diagonal(naive, 1.0)
    assert np.linalg.eigvalsh(naive).min() < 0, (
        "l'approche naïve devrait casser la validité à +0,30")


def test_intensite_nulle_est_lidentite(ptf):
    apres = CorrelationShock(0.0).apply(ptf)
    np.testing.assert_allclose(apres.covariance.matrix,
                               ptf.covariance.matrix, rtol=1e-14)


def test_intensite_maximale_atteint_la_cible(ptf):
    apres = CorrelationShock(1.0, toward=0.9).apply(ptf)
    c = apres.covariance.correlation().to_numpy()
    hors_diag = c[~np.eye(len(c), dtype=bool)]
    np.testing.assert_allclose(hors_diag, 0.9, atol=1e-12)


def test_monter_les_correlations_augmente_le_risque(ptf):
    vols = [CorrelationShock(a, toward=0.95).apply(ptf).volatility
            for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for x, y in itertools.pairwise(vols):
        assert y > x, f"non monotone : {vols}"


def test_decorreler_detruit_moins_de_paris_quen_corréler(ptf):
    """La décorrélation doit AUGMENTER le nombre effectif de paris."""
    base = ptf.decompose().effective_bets
    decor = CorrelationShock(0.8, toward=0.0).apply(ptf).decompose().effective_bets
    concen = CorrelationShock(0.8, toward=0.95).apply(ptf).decompose().effective_bets
    assert decor > base > concen


def test_intensite_hors_bornes_rejetee():
    with pytest.raises(ValueError, match="intensity"):
        CorrelationShock(1.5)
    with pytest.raises(ValueError, match="toward"):
        CorrelationShock(0.5, toward=2.0)


# ══════════════════════════════════════════════════════════════════════
#  CHOC DE PRIX
# ══════════════════════════════════════════════════════════════════════


def test_pnl_est_la_moyenne_ponderee_des_mouvements(ptf):
    moves = {"TECH_A": -0.40, "OR": 0.10, "OBLIG": -0.02}
    attendu = sum(ptf.weights[a] * r for a, r in moves.items())
    r = stress(ptf, PriceShock(moves))
    assert abs(r.pnl - attendu) < 1e-12


def test_les_poids_derivent_apres_un_choc_de_prix(ptf):
    """
    ★ L'EFFET QU'ON OUBLIE ★

    Après une chute inégale, la composition n'est plus celle qu'on avait
    choisie. Ignorer cette dérive fausse toute mesure faite ensuite.
    """
    apres = PriceShock({n: -0.45 for n in NOMS if n.startswith("TECH")}
                       ).apply(ptf)
    assert abs(apres.weights.sum() - 1.0) < 1e-12
    for n in NOMS:
        if n.startswith("TECH"):
            assert apres.weights[n] < ptf.weights[n]
        else:
            assert apres.weights[n] > ptf.weights[n]


def test_choc_de_prix_ne_touche_pas_le_modele_de_risque(ptf):
    apres = PriceShock({"TECH_A": -0.3}).apply(ptf)
    np.testing.assert_allclose(apres.covariance.matrix,
                               ptf.covariance.matrix, rtol=1e-14)


def test_perte_totale_rejetee():
    with pytest.raises(ValueError, match="hors du domaine"):
        PriceShock({"TECH_A": -1.0})


# ══════════════════════════════════════════════════════════════════════
#  COMPOSITION
# ══════════════════════════════════════════════════════════════════════


def test_combiner_equivaut_a_appliquer_en_sequence(ptf):
    a, b = VolatilityShock(2.0), CorrelationShock(0.6, toward=0.9)
    sequentiel = b.apply(a.apply(ptf))
    combine = Combined([a, b]).apply(ptf)
    np.testing.assert_allclose(combine.covariance.matrix,
                               sequentiel.covariance.matrix, rtol=1e-14)
    assert abs(combine.volatility - sequentiel.volatility) < 1e-12


def test_un_combine_se_compose_avec_un_autre(ptf):
    """`Combined` est lui-même un Shock : aucun cas particulier n'est requis."""
    interne = Combined([VolatilityShock(1.5), CorrelationShock(0.3)])
    externe = Combined([interne, VolatilityShock(2.0)])
    assert isinstance(externe, Shock)
    attendu = VolatilityShock(2.0).apply(interne.apply(ptf))
    np.testing.assert_allclose(externe.apply(ptf).covariance.matrix,
                               attendu.covariance.matrix, rtol=1e-14)


def test_tous_les_chocs_commutent(ptf):
    """
    ★ PROPRIÉTÉ DÉCOUVERTE EN TESTANT L'INVERSE ★

    J'avais écrit dans la documentation qu'un choc de prix ne commute
    avec aucun autre, parce qu'il fait dériver les poids. Faux : l'écart
    mesuré est exactement nul.

    La raison tient à la séparation de l'état. Un choc de prix touche
    (poids, valeur) ; un choc de risque touche (covariance). Deux
    transformations agissant sur des parts disjointes commutent
    nécessairement.

    Ce test fige la propriété. Elle tombera le jour où on ajoutera un
    choc couplé — par exemple « baisse de 3 écarts-types », dont
    l'ampleur dépend de la volatilité courante.
    """
    v = VolatilityShock(2.0)
    c = CorrelationShock(0.5, toward=0.9)
    p = PriceShock({"TECH_A": -0.5, "OR": 0.2})

    import itertools
    refs = None
    for ordre in itertools.permutations([v, c, p]):
        r = Combined(list(ordre)).apply(ptf)
        etat = (r.volatility, r.value, tuple(r.weights.round(12)))
        if refs is None:
            refs = etat
        else:
            assert abs(etat[0] - refs[0]) < 1e-12
            assert abs(etat[1] - refs[1]) < 1e-12
            assert etat[2] == refs[2]


def test_cible_de_correlation_impossible_rejetee(ptf):
    """
    ★ LA BORNE QUI A RÉVÉLÉ UN TROU DANS MA PREUVE ★

    Une corrélation uniforme ρ n'est réalisable sur n actifs que si
    ρ ≥ −1/(n−1). À 7 actifs la borne vaut −0,167. Ma preuve de validité
    supposait la cible SDP sans jamais le vérifier ; un balayage de toute
    la plage a trouvé le cas.

    Ce n'est pas un artefact numérique mais un fait financier : on ne
    peut pas avoir beaucoup d'actifs tous fortement anticorrélés.
    """
    n = ptf.n_assets
    borne = -1.0 / (n - 1)

    with pytest.raises(ValueError, match="impossible pour"):
        CorrelationShock(0.5, toward=borne - 0.05).apply(ptf)

    # juste au-dessus de la borne, le choc passe et reste valide
    ok = CorrelationShock(1.0, toward=borne + 0.01).apply(ptf)
    assert np.linalg.eigvalsh(ok.covariance.matrix).min() >= -1e-10

    # et la borne se relâche quand il y a moins d'actifs
    assert -1.0 / (3 - 1) < -1.0 / (20 - 1)


def test_combine_vide_est_lidentite(ptf):
    r = Combined([]).apply(ptf)
    np.testing.assert_allclose(r.covariance.matrix, ptf.covariance.matrix)
    assert abs(r.value - ptf.value) < 1e-15


# ══════════════════════════════════════════════════════════════════════
#  RÉSULTAT ET TABLEAU
# ══════════════════════════════════════════════════════════════════════


def test_le_scenario_ne_modifie_pas_loriginal(ptf):
    """Le « avant » reste intact : c'est la raison d'être de l'immutabilité."""
    vol_initiale = ptf.volatility
    poids = ptf.weights.copy()
    stress(ptf, CRISE_2008)
    stress(ptf, PriceShock({"TECH_A": -0.5}))
    assert abs(ptf.volatility - vol_initiale) < 1e-15
    pd.testing.assert_series_equal(ptf.weights, poids)


def test_diversification_perdue_bornee(ptf):
    for s in (CorrelationShock(0.0), CorrelationShock(1.0, toward=0.99),
              VolatilityShock(2.0), CRISE_2008):
        d = stress(ptf, s).diversification_lost
        assert 0.0 <= d <= 1.0


def test_un_portefeuille_mieux_diversifie_perd_davantage(cov):
    """
    ★ LE RÉSULTAT CONTRE-INTUITIF DE LA SÉANCE ★

    Sous un choc de corrélation, le portefeuille le mieux diversifié voit
    sa volatilité augmenter DAVANTAGE en relatif — parce qu'il avait plus
    de diversification à perdre. La crise détruit précisément ce qui
    protégeait.

    Son niveau absolu reste meilleur ; c'est le facteur qui est pire.
    """
    C = cov.matrix
    n = len(cov.assets)
    w = np.full(n, 1 / n)
    for _ in range(500):  # parité de risque
        for i in range(n):
            b = sum(C[i][j] * w[j] for j in range(n) if j != i)
            w[i] = (-b + np.sqrt(b * b + 4 * C[i][i] / n)) / (2 * C[i][i])
    w = w / w.sum()

    plat = equal_weighted(cov)
    parite = Portfolio(pd.Series(w, index=list(cov.assets)), cov,
                       name="parité de risque")
    assert parite.decompose().effective_bets > plat.decompose().effective_bets

    choc = CorrelationShock(1.0, toward=0.90)
    r_plat, r_par = stress(plat, choc), stress(parite, choc)

    assert r_par.vol_ratio > r_plat.vol_ratio       # pire en relatif
    assert r_par.vol_after < r_plat.vol_after       # meilleur en absolu


def test_tableau_de_scenarios(ptf):
    df = stress_table(ptf, [VolatilityShock(2.0), CRISE_2008])
    assert len(df) == 2
    assert {"P&L", "vol avant", "vol après", "facteur"} <= set(df.columns)
    assert abs(df.loc["choc de volatilité", "facteur"] - 2.0) < 1e-12
