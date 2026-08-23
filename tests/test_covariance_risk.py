"""
Tests de la séance 2 — covariance et décomposition du risque.

Même hiérarchie qu'à la séance 1 : propriétés → calibration → couverture.
S'ajoute ici un quatrième niveau, le plus fort quand il est disponible :

  Niveau 4 — validation croisée contre une implémentation indépendante

Notre Ledoit-Wolf est comparé à celui de scikit-learn, écrit par d'autres
gens, à partir du même article. Deux bugs identiques et indépendants sont
extrêmement improbables ; une concordance à 10⁻¹⁸ près est donc une
preuve très forte de correction.

Quand une implémentation de référence existe, s'en priver est une faute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nkrisk.covariance import CovarianceEstimate, ledoit_wolf, sample_covariance
from nkrisk.data import GBMSpec, generate_correlated_prices
from nkrisk.returns import log_returns, simple_returns
from nkrisk.risk import (
    decompose_risk,
    effective_number_of_bets,
    equal_weights,
    minimum_variance_weights,
    portfolio_volatility,
)


def _returns(n_assets: int, n_days: int, seed: int, rho: float = 0.3) -> pd.DataFrame:
    """Rendements corrélés avec une structure connue."""
    specs = {
        f"A{i}": GBMSpec(mu=0.08, sigma=0.20) for i in range(n_assets)
    }
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    prices = generate_correlated_prices(specs, corr, n_days, seed=seed)
    return simple_returns(prices)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — PROPRIÉTÉS
# ══════════════════════════════════════════════════════════════════════


def test_covariance_est_symetrique():
    cov = sample_covariance(_returns(8, 500, seed=1))
    np.testing.assert_allclose(cov.matrix, cov.matrix.T, rtol=1e-14)


def test_ledoit_wolf_est_definie_positive_meme_quand_n_depasse_t():
    """
    LA propriété qui justifie l'existence de la contraction.

    Avec 60 actifs et 40 observations, l'empirique est mathématiquement
    singulière — son rang ne peut pas dépasser 40. La contractée, elle,
    reste inversible.
    """
    r = _returns(60, 40, seed=2)

    emp = sample_covariance(r)
    lw = ledoit_wolf(r)

    assert not emp.is_invertible, "l'empirique devrait être singulière ici"
    assert lw.is_invertible, "la contractée doit rester inversible"
    assert np.linalg.eigvalsh(lw.matrix).min() > 0


def test_shrinkage_est_dans_zero_un():
    for n_assets, n_days in [(3, 2000), (20, 300), (100, 120)]:
        lw = ledoit_wolf(_returns(n_assets, n_days, seed=3))
        assert 0.0 <= lw.shrinkage <= 1.0


def test_shrinkage_augmente_quand_les_donnees_se_rarefient():
    """
    δ se lit comme un rapport bruit/signal. Moins de données → plus de
    bruit → contraction plus forte. La direction doit être monotone.
    """
    deltas = [
        ledoit_wolf(_returns(30, t, seed=4)).shrinkage
        for t in (2000, 1000, 500, 250, 100)
    ]
    for precedent, suivant in zip(deltas, deltas[1:]):
        assert suivant >= precedent - 1e-9, f"non monotone : {deltas}"


def test_correlation_a_une_diagonale_unitaire():
    cov = sample_covariance(_returns(6, 800, seed=5))
    c = cov.correlation()
    np.testing.assert_allclose(np.diag(c.to_numpy()), 1.0, rtol=1e-12)
    assert (c.to_numpy() >= -1.0 - 1e-9).all()
    assert (c.to_numpy() <= 1.0 + 1e-9).all()


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════


def test_covariance_retrouve_la_correlation_vraie():
    """
    On génère avec ρ = 0,50 imposé et on vérifie qu'on le retrouve.
    Impossible à faire avec des données réelles.
    """
    r = _returns(5, 252 * 20, seed=6, rho=0.50)
    c = sample_covariance(r).correlation().to_numpy()
    hors_diag = c[~np.eye(5, dtype=bool)]
    assert abs(hors_diag.mean() - 0.50) < 0.02


def test_le_conditionnement_se_degrade_avec_n_sur_t():
    """
    Le cœur du problème de la séance 2, mesuré.
    Plus N/T est grand, plus la matrice empirique est mal conditionnée.
    """
    conds = [
        sample_covariance(_returns(n, 120, seed=7)).condition_number
        for n in (40, 70, 100, 118)
    ]
    for precedent, suivant in zip(conds, conds[1:]):
        assert suivant > precedent, f"conditionnement non croissant : {conds}"

    # La pathologie ne mord vraiment que quand N/T approche 1. À N/T=0.33
    # le conditionnement reste modeste (~75) ; à N/T=0.98 il explose au
    # million. Ma première version de ce test visait N/T=0.40 et exigeait
    # 1e4 — l'attente était fausse, pas le code.
    assert conds[-1] > 1e5, f"N/T=0.98 devrait exploser : {conds[-1]:,.0f}"


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — L'IDENTITÉ D'EULER
# ══════════════════════════════════════════════════════════════════════


def test_les_contributions_somment_exactement_au_risque_total():
    """
    ★ L'IDENTITÉ QUI FONDE TOUTE LA DÉCOMPOSITION ★

    σ_p = Σ_i w_i · ∂σ_p/∂w_i

    Ce n'est pas une approximation : c'est le théorème d'Euler appliqué à
    une fonction homogène de degré 1. L'égalité doit tenir à la précision
    machine. Si elle ne tient pas, l'algèbre est fausse quelque part et
    tous les chiffres produits sont invalides.
    """
    rng = np.random.default_rng(8)
    for seed in range(20):
        r = _returns(10, 500, seed=seed + 100)
        cov = ledoit_wolf(r)

        brut = rng.random(10)
        w = pd.Series(brut / brut.sum(), index=list(cov.assets))

        d = decompose_risk(w, cov)
        assert abs(d.component.sum() - d.portfolio_volatility) < 1e-12
        assert abs(d.percent.sum() - 1.0) < 1e-12


def test_contributions_coherentes_avec_le_calcul_direct():
    """La volatilité issue de la décomposition = celle calculée directement."""
    r = _returns(12, 600, seed=9)
    cov = ledoit_wolf(r)
    w = equal_weights(cov)

    d = decompose_risk(w, cov)
    direct = portfolio_volatility(w, cov)
    assert abs(d.portfolio_volatility - direct) < 1e-12


def test_correlation_parfaite_annule_la_diversification():
    """
    Cas limite vérifiable à la main : si tous les actifs sont
    parfaitement corrélés et de même volatilité, le portefeuille a la
    volatilité d'un actif isolé, le ratio de diversification vaut 1 et
    le nombre effectif de paris vaut 1 — quel que soit le nombre de lignes.
    """
    n = 8
    specs = {f"A{i}": GBMSpec(mu=0.08, sigma=0.20) for i in range(n)}
    corr = np.full((n, n), 0.9999)
    np.fill_diagonal(corr, 1.0)
    prices = generate_correlated_prices(specs, corr, 2000, seed=10)

    cov = sample_covariance(simple_returns(prices))
    d = decompose_risk(equal_weights(cov), cov)

    assert abs(d.diversification_ratio - 1.0) < 0.01
    # C'est CE test qui a démoli ma première formule d'ENB (inverse de
    # Herfindahl sur les contributions), laquelle retournait 8 ici.
    assert abs(d.effective_bets - 1.0) < 0.1
    assert abs(d.portfolio_volatility - 0.20) < 0.02


def test_enb_decroit_quand_la_correlation_monte():
    """
    ★ LE TEST QUI A RÉVÉLÉ LE DOMAINE DE VALIDITÉ DE L'INDICATEUR ★

    Propriété attendue : plus les actifs sont corrélés, moins il y a de
    paris indépendants. L'ENB doit décroître de façon monotone avec ρ.

    Ce test tourne sur un portefeuille RÉALISTE — volatilités hétérogènes,
    poids inégaux. C'est essentiel : ma première version utilisait des
    actifs identiques équipondérés, cas où l'ENB est mathématiquement
    dégénéré (il saute de 10 à 1 entre ρ=0 et ρ=0,001, parce que le
    portefeuille est exactement aligné sur la première composante
    principale). Le test échouait, et l'attente était fausse, pas le code.

    Voir la note de domaine de validité dans effective_number_of_bets.
    """
    n = 10
    sd = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.12, 0.18, 0.22])
    rng = np.random.default_rng(0)
    brut = rng.random(n)
    w = brut / brut.sum()

    valeurs = []
    for rho in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        sigma = corr * np.outer(sd, sd)
        np.fill_diagonal(sigma, sd**2)
        valeurs.append(effective_number_of_bets(w, sigma))

    for precedent, suivant in zip(valeurs, valeurs[1:]):
        assert suivant <= precedent + 1e-9, f"non monotone : {valeurs}"

    assert valeurs[0] > 4.0, f"décorrélé devrait donner plusieurs paris : {valeurs[0]:.2f}"
    assert valeurs[-1] < 1.05, f"quasi parfaitement corrélé → 1 pari : {valeurs[-1]:.2f}"


def test_actifs_independants_donnent_n_paris():
    """
    Cas ρ = 0 exact : N actifs indépendants équipondérés → ENB = N.

    Attention, c'est le point de discontinuité documenté : à ρ = 0,001 ce
    même portefeuille retourne 1. Le test verrouille la valeur au point
    exact, pas la continuité autour — qui n'existe pas.
    """
    n = 10
    sigma = np.eye(n) * 0.04
    w = np.full(n, 1.0 / n)
    assert abs(effective_number_of_bets(w, sigma) - n) < 1e-6


def test_poids_mal_alignes_rejetes():
    """
    Un actif manquant doit lever une exception, jamais être silencieusement
    ignoré ou réordonné. Le désalignement d'indices est l'erreur la plus
    insidieuse du domaine : elle produit un résultat plausible et faux.
    """
    cov = ledoit_wolf(_returns(5, 400, seed=12))
    w = pd.Series([0.5, 0.5], index=["A0", "A1"])
    with pytest.raises(ValueError, match="poids absents"):
        decompose_risk(w, cov)


def test_poids_ne_sommant_pas_a_un_rejetes():
    cov = ledoit_wolf(_returns(4, 400, seed=13))
    w = pd.Series([0.3, 0.3, 0.3, 0.3], index=list(cov.assets))
    with pytest.raises(ValueError, match="somment"):
        decompose_risk(w, cov)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 4 — VALIDATION CROISÉE CONTRE SCIKIT-LEARN
# ══════════════════════════════════════════════════════════════════════


def test_ledoit_wolf_concorde_avec_scikit_learn():
    """
    ★ LA VALIDATION LA PLUS FORTE DISPONIBLE ★

    scikit-learn implémente le même article (Ledoit & Wolf 2004), écrit
    par d'autres personnes, sans contact avec notre code. Une concordance
    à la précision machine sur δ ET sur la matrice complète, dans quatre
    régimes N/T différents, rend l'hypothèse d'un bug commun négligeable.

    Le test est ignoré si scikit-learn n'est pas installé — une
    dépendance de confort ne doit jamais casser la suite.
    """
    sk = pytest.importorskip("sklearn.covariance")

    rng = np.random.default_rng(14)
    for n_assets, n_obs in [(5, 500), (30, 252), (80, 120), (150, 100)]:
        x = rng.normal(size=(n_obs, n_assets)) * 0.01
        df = pd.DataFrame(x, columns=[f"A{i}" for i in range(n_assets)])

        notre = ledoit_wolf(df)
        sk_cov, sk_delta = sk.ledoit_wolf(x, assume_centered=False)

        assert abs(notre.shrinkage - sk_delta) < 1e-12, (
            f"N={n_assets} T={n_obs} : δ nous={notre.shrinkage}, "
            f"sklearn={sk_delta}"
        )
        np.testing.assert_allclose(notre.matrix, sk_cov, rtol=1e-10, atol=1e-18)


# ══════════════════════════════════════════════════════════════════════
#  LE DANGER DE L'INVERSION
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_la_contraction_stabilise_les_poids_de_variance_minimale():
    """
    ★ LA DÉMONSTRATION QUI JUSTIFIE TOUTE LA SÉANCE ★

    Le portefeuille de variance minimale inverse la covariance. On mesure
    l'instabilité des poids obtenus sur des échantillons indépendants
    tirés de la MÊME distribution : idéalement, les poids devraient être
    identiques, puisque la vérité sous-jacente ne change pas.

    Avec l'empirique, ils partent dans tous les sens. Avec la contractée,
    ils tiennent. C'est la mesure directe du « maximiseur d'erreur ».
    """
    n_assets, n_days = 110, 120

    poids_emp, poids_lw = [], []
    for seed in range(30):
        r = _returns(n_assets, n_days, seed=seed + 500)
        poids_emp.append(minimum_variance_weights(sample_covariance(r)).to_numpy())
        poids_lw.append(minimum_variance_weights(ledoit_wolf(r)).to_numpy())

    # Instabilité = écart-type des poids d'un échantillon à l'autre
    instab_emp = float(np.std(np.array(poids_emp), axis=0).mean())
    instab_lw = float(np.std(np.array(poids_lw), axis=0).mean())

    assert instab_lw < instab_emp / 3, (
        f"la contraction devrait diviser l'instabilité par au moins 3 : "
        f"empirique={instab_emp:.4f}, ledoit-wolf={instab_lw:.4f}"
    )

    # Et l'empirique produit des positions vendeuses extrêmes
    levier_emp = float(np.mean([np.abs(w).sum() for w in poids_emp]))
    levier_lw = float(np.mean([np.abs(w).sum() for w in poids_lw]))
    assert levier_emp > 10.0, (
        f"à N/T=0.92 l'empirique devrait produire un levier aberrant, "
        f"obtenu {levier_emp:.1f}"
    )
    assert levier_lw < levier_emp / 3
