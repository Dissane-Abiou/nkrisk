"""
Tests du module returns.

CE QUE TU DOIS COMPRENDRE DE CE FICHIER
=======================================

Tu m'as demandé « sans aucune erreur ». Voici la seule façon connue de
s'en approcher : on ne PROMET pas l'absence de bug, on la MESURE.

Il y a trois niveaux de tests ici, du plus faible au plus fort :

  Niveau 1 — tests de propriété
    Vérifient des identités mathématiques qui doivent tenir quelles que
    soient les données. Ex : ln(1+r) doit toujours redonner le rendement
    log. Ça attrape les fautes de frappe.

  Niveau 2 — tests de calibration
    On génère des données dont on CONNAÎT la vraie volatilité, et on
    vérifie que l'estimateur la retrouve. Ça attrape les erreurs de
    formule et les facteurs d'échelle oubliés (le √252 inversé, par ex.).

  Niveau 3 — test de couverture       ← LE PLUS IMPORTANT
    On vérifie que l'intervalle de confiance à 95 % contient réellement
    la vraie valeur dans 95 % des cas. On simule mille fois et on compte.
    Ça valide non pas l'estimation, mais L'INCERTITUDE ELLE-MÊME.

Presque personne n'écrit le niveau 3. C'est pourtant le seul qui garantit
que ton moteur ne ment pas sur ce qu'il sait. Si tu ne retiens qu'une
chose de cette séance, retiens celle-là — et mets-la sur ton CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nkrisk import (
    Estimate,
    InsufficientData,
    annualized_mean_return,
    annualized_volatility,
    compound_annualize,
    geometric_return,
    log_returns,
    sharpe_ratio,
    simple_returns,
    volatility_drag,
)
from nkrisk.data import GBMSpec, generate_prices


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — PROPRIÉTÉS MATHÉMATIQUES
# ══════════════════════════════════════════════════════════════════════


def test_les_deux_rendements_sont_coherents():
    """r = e^l - 1 doit tenir exactement, à la précision machine près."""
    prices = generate_prices(GBMSpec(mu=0.08, sigma=0.20), n_days=500, seed=1)
    r = simple_returns(prices)
    l = log_returns(prices)
    np.testing.assert_allclose(r.to_numpy(), np.expm1(l.to_numpy()), rtol=1e-12)


def test_rendements_log_additifs_dans_le_temps():
    """
    LA propriété qui justifie l'existence des rendements log.
    La somme des log-rendements doit égaler le log du rendement total.
    """
    prices = generate_prices(GBMSpec(mu=0.10, sigma=0.25), n_days=1000, seed=2)
    l = log_returns(prices)
    somme = float(l.sum())
    total = float(np.log(prices.iloc[-1] / prices.iloc[0]))
    assert abs(somme - total) < 1e-10


def test_rendements_simples_additifs_entre_actifs():
    """
    La propriété symétrique : le rendement d'un portefeuille est la moyenne
    pondérée des rendements SIMPLES. Ce test échouerait avec des log.
    C'est pour ça qu'il faut les deux.
    """
    a = generate_prices(GBMSpec(mu=0.08, sigma=0.20), n_days=300, seed=3)
    b = generate_prices(GBMSpec(mu=0.05, sigma=0.15), n_days=300, seed=4)
    w_a, w_b = 0.6, 0.4

    valeur_ptf = w_a * a / a.iloc[0] + w_b * b / b.iloc[0]
    r_ptf_direct = simple_returns(valeur_ptf)

    r_a, r_b = simple_returns(a), simple_returns(b)
    # Poids qui dérivent avec les prix : on refait le calcul proprement.
    poids_a = w_a * (a / a.iloc[0]).shift(1)
    poids_b = w_b * (b / b.iloc[0]).shift(1)
    total = (poids_a + poids_b).dropna()
    r_ptf_agrege = (
        (poids_a.reindex(r_a.index) * r_a + poids_b.reindex(r_b.index) * r_b)
        / total.reindex(r_a.index)
    ).dropna()

    np.testing.assert_allclose(
        r_ptf_direct.to_numpy(), r_ptf_agrege.to_numpy(), rtol=1e-10
    )


def test_prix_negatif_rejete():
    prices = pd.Series([100.0, 50.0, -10.0])
    with pytest.raises(ValueError, match="négatifs"):
        log_returns(prices)


def test_nan_rejete_plutot_que_propage():
    """
    Un NaN ne doit JAMAIS traverser silencieusement le moteur.
    C'est le principe « fail loud » : mieux vaut planter que mentir.
    """
    prices = pd.Series([100.0, np.nan, 102.0, 103.0])
    with pytest.raises(ValueError, match="NaN"):
        simple_returns(prices)


def test_donnees_insuffisantes_leve_exception():
    prices = generate_prices(GBMSpec(mu=0.08, sigma=0.20), n_days=10, seed=5)
    with pytest.raises(InsufficientData):
        annualized_volatility(log_returns(prices), min_obs=20)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — CALIBRATION CONTRE UNE VÉRITÉ CONNUE
# ══════════════════════════════════════════════════════════════════════


def test_volatilite_retrouve_la_vraie_valeur():
    """
    On génère 20 ans de données avec σ = 20 % exactement.
    L'estimateur doit retrouver 20 % à 2 % près en relatif.

    C'est ce test qui attrape les erreurs de facteur d'échelle : si tu
    avais écrit /√252 au lieu de ×√252, tu obtiendrais 0,079 % et le
    test hurlerait.
    """
    vraie_sigma = 0.20
    prices = generate_prices(
        GBMSpec(mu=0.08, sigma=vraie_sigma), n_days=252 * 20, seed=42
    )
    est = annualized_volatility(log_returns(prices))
    assert abs(est.value - vraie_sigma) / vraie_sigma < 0.02


def test_intervalle_de_volatilite_est_asymetrique():
    """
    Piège classique : l'intervalle du chi² n'est PAS symétrique autour de
    l'estimation ponctuelle. La borne haute est plus éloignée que la basse.

    Si quelqu'un « simplifie » un jour le code en écrivant value ± k·SE,
    ce test échouera et signalera la régression.
    """
    prices = generate_prices(GBMSpec(mu=0.08, sigma=0.20), n_days=252, seed=7)
    est = annualized_volatility(log_returns(prices))
    dist_basse = est.value - est.ci_low
    dist_haute = est.ci_high - est.value
    assert dist_haute > dist_basse
    assert est.ci_low < est.value < est.ci_high


def test_incertitude_decroit_avec_les_donnees():
    """
    Propriété fondamentale : plus de données → intervalle plus étroit,
    en 1/√n. On vérifie la direction ET l'ordre de grandeur.
    """
    spec = GBMSpec(mu=0.08, sigma=0.20)
    e1 = annualized_volatility(log_returns(generate_prices(spec, 252, seed=11)))
    e4 = annualized_volatility(log_returns(generate_prices(spec, 252 * 4, seed=11)))

    assert e4.relative_uncertainty < e1.relative_uncertainty
    # 4× plus de données → intervalle ~2× plus étroit (√4 = 2)
    ratio = e1.relative_uncertainty / e4.relative_uncertainty
    assert 1.7 < ratio < 2.3, f"ratio attendu ≈2, obtenu {ratio:.2f}"


def test_trainee_de_volatilite_est_reelle():
    """
    Vérifie que rendement géométrique ≈ arithmétique - σ²/2 sur des
    données réelles générées. C'est le passage de la théorie au chiffre.
    """
    spec = GBMSpec(mu=0.10, sigma=0.30)
    prices = generate_prices(spec, n_days=252 * 30, seed=99)

    geo = geometric_return(prices)
    arith = annualized_mean_return(simple_returns(prices)).value
    vol = annualized_volatility(log_returns(prices)).value

    predit = volatility_drag(arith, vol)
    # L'approximation est du second ordre : on tolère 1,5 point d'écart.
    assert abs(geo - predit) < 0.015
    # Et la traînée doit être NÉGATIVE : le géométrique est sous l'arithmétique.
    assert geo < arith


@pytest.mark.slow
def test_trainee_positive_meme_a_faible_volatilite():
    """
    ★ TEST DE NON-RÉGRESSION D'UN BUG RÉEL ★

    Historique : la première version de examples/seance1.py comparait un
    rendement arithmétique annualisé LINÉAIREMENT (× 252) à un rendement
    géométrique annualisé par COMPOSITION. L'écart de convention valait
    environ 0,33 point, alors que l'effet à mesurer (la traînée à σ = 5 %)
    ne vaut que 0,13 point.

    Résultat : la traînée sortait NÉGATIVE. Le tableau démontrait
    l'inverse de la réalité, sans qu'aucune exception ne soit levée.

    Aucun de mes autres tests ne l'attrapait, parce qu'ils utilisaient
    tous σ ≥ 20 %, régime où l'effet réel (2 pts) écrase l'artefact.

    Ce test verrouille le régime dangereux : faible volatilité, où le bug
    de convention domine. La traînée doit être POSITIVE et proche de σ²/2.
    """
    for sigma in (0.05, 0.10):
        geos, aris = [], []
        for seed in range(200):
            p = generate_prices(
                GBMSpec(mu=0.08, sigma=sigma), n_days=252 * 30, seed=seed + 900
            )
            geos.append(geometric_return(p))
            aris.append(compound_annualize(float(np.mean(simple_returns(p)))))

        trainee = float(np.mean(aris)) - float(np.mean(geos))
        attendu = sigma**2 / 2

        assert trainee > 0, (
            f"σ={sigma:.0%} : traînée = {trainee:.4%}, devrait être positive. "
            f"Regarde si les deux rendements sont annualisés de la même façon."
        )
        assert abs(trainee - attendu) < 0.004, (
            f"σ={sigma:.0%} : traînée = {trainee:.4%}, attendue ≈ {attendu:.4%}"
        )


def test_les_deux_annualisations_different_de_facon_mesurable():
    """
    Rend l'écart de convention EXPLICITE plutôt que latent.
    Si quelqu'un les confond un jour, ce test documente l'ampleur.
    """
    r_quotidien = 0.08 / 252
    lineaire = r_quotidien * 252
    composee = compound_annualize(r_quotidien)

    assert composee > lineaire
    ecart = composee - lineaire
    assert 0.002 < ecart < 0.005, f"écart de convention = {ecart:.4%}"


def test_sharpe_dun_an_ne_prouve_rien():
    """
    Le test le plus pédagogique du fichier.

    On prend une stratégie réellement bonne (Sharpe vrai = 0,5) et un an
    de données. On vérifie que l'intervalle de confiance CONTIENT ZÉRO —
    c'est-à-dire qu'après un an, on ne peut pas prouver qu'elle vaut
    mieux que rien.

    Ce n'est pas un défaut de notre code. C'est une propriété du monde.
    """
    # μ/σ = 0,10/0,20 = 0,5 de Sharpe annualisé
    prices = generate_prices(GBMSpec(mu=0.10, sigma=0.20), n_days=252, seed=123)
    est = sharpe_ratio(simple_returns(prices))
    assert est.ci_low < 0 < est.ci_high, (
        f"IC = [{est.ci_low:.2f}, {est.ci_high:.2f}] — devrait contenir 0"
    )


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — TEST DE COUVERTURE  ★ LE PLUS IMPORTANT ★
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_couverture_de_lintervalle_de_volatilite():
    """
    ★ CE TEST VALIDE L'HONNÊTETÉ DU MOTEUR ★

    Un intervalle de confiance à 95 % n'est correct que s'il contient la
    vraie valeur dans 95 % des cas. Ce n'est pas une évidence : c'est une
    affirmation testable, et beaucoup d'implémentations la ratent.

    Protocole :
      1. On fixe la vraie volatilité à 20 %.
      2. On génère 1000 échantillons indépendants d'un an chacun.
      3. Pour chacun, on calcule l'intervalle à 95 %.
      4. On compte la proportion qui contient 0,20.
      5. Cette proportion doit être ≈ 95 %.

    Si on obtient 80 %, notre moteur ment : il prétend une précision qu'il
    n'a pas, et un utilisateur prendrait des décisions sur du sable.
    Si on obtient 99,9 %, on est trop prudents et l'outil devient inutile.

    Bande d'acceptation : avec 1000 tirages, l'erreur-type sur une
    proportion de 0,95 est √(0,95·0,05/1000) ≈ 0,7 %. On accepte donc
    [93 %, 97 %], soit environ ±3 erreurs-types.
    """
    vraie_sigma = 0.20
    n_simulations = 1000
    couvertures = 0

    for seed in range(n_simulations):
        prices = generate_prices(
            GBMSpec(mu=0.08, sigma=vraie_sigma), n_days=252, seed=seed
        )
        est = annualized_volatility(log_returns(prices), confidence=0.95)
        if est.ci_low <= vraie_sigma <= est.ci_high:
            couvertures += 1

    taux = couvertures / n_simulations
    assert 0.93 <= taux <= 0.97, (
        f"Couverture observée : {taux:.1%}, attendue ≈95 %. "
        f"L'intervalle de confiance est mal calibré."
    )


@pytest.mark.slow
def test_couverture_de_lintervalle_de_moyenne():
    """Même protocole pour l'estimateur de rendement moyen."""
    spec = GBMSpec(mu=0.08, sigma=0.20)
    # Espérance vraie du rendement SIMPLE quotidien, annualisée.
    vraie_moyenne = spec.mu

    n_simulations = 1000
    couvertures = 0
    for seed in range(n_simulations):
        prices = generate_prices(spec, n_days=252 * 3, seed=seed + 50_000)
        est = annualized_mean_return(simple_returns(prices), confidence=0.95)
        if est.ci_low <= vraie_moyenne <= est.ci_high:
            couvertures += 1

    taux = couvertures / n_simulations
    assert 0.92 <= taux <= 0.98, f"Couverture : {taux:.1%}, attendue ≈95 %"


# ══════════════════════════════════════════════════════════════════════
#  LE TYPE Estimate LUI-MÊME
# ══════════════════════════════════════════════════════════════════════


def test_estimate_refuse_les_etats_incoherents():
    """« Make illegal states unrepresentable » — appliqué."""
    with pytest.raises(ValueError, match="intervalle inversé"):
        Estimate(0.20, 0.25, 0.15, 0.95, 252, "test")

    with pytest.raises(ValueError, match="hors de son propre intervalle"):
        Estimate(0.50, 0.15, 0.25, 0.95, 252, "test")

    with pytest.raises(ValueError, match="confidence"):
        Estimate(0.20, 0.15, 0.25, 1.5, 252, "test")


def test_estimate_signale_quand_il_ne_sait_pas():
    """Une estimation trop imprécise doit se déclarer non fiable."""
    precise = Estimate(0.20, 0.19, 0.21, 0.95, 5000, "test")
    vague = Estimate(0.20, 0.05, 0.35, 0.95, 20, "test")

    assert precise.is_meaningful()
    assert not vague.is_meaningful()
    assert "PEU FIABLE" in vague.format()
    assert "PEU FIABLE" not in precise.format()
