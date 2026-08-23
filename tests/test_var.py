"""
Tests de la séance 3 — VaR et CVaR.

Le test central de ce fichier est celui de COUVERTURE, encore : on
vérifie qu'une VaR à 99 % est effectivement dépassée 1 % du temps, sur
des données dont on connaît la vraie distribution.

C'est la troisième fois que ce schéma revient dans le projet. Ce n'est
pas une coïncidence : **la seule façon de valider une affirmation
probabiliste est de compter empiriquement à quelle fréquence elle tient.**
Si tu ne retiens qu'une méthode de tout ce projet, retiens celle-là.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nkrisk.data import (
    GBMSpec,
    StudentTSpec,
    generate_prices,
    generate_student_t_prices,
)
from nkrisk.returns import log_returns
from nkrisk.var import (
    cornish_fisher_var,
    historical_var,
    kupiec_test,
    parametric_var,
    subadditivity_counterexample,
)


def _normal(n: int, seed: int, sigma: float = 0.20) -> pd.Series:
    return log_returns(generate_prices(GBMSpec(0.08, sigma), n, seed=seed))


def _fat(n: int, seed: int, sigma: float = 0.20, df: float = 4.0) -> pd.Series:
    return log_returns(
        generate_student_t_prices(StudentTSpec(0.08, sigma, df), n, seed=seed)
    )


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — PROPRIÉTÉS
# ══════════════════════════════════════════════════════════════════════


def test_cvar_depasse_toujours_var():
    """
    Propriété structurelle : la CVaR est la moyenne des pertes AU-DELÀ de
    la VaR, elle lui est donc toujours supérieure ou égale. Vrai pour les
    trois méthodes, sur données normales comme à queues épaisses.
    """
    for donnees in (_normal(2000, 1), _fat(2000, 1)):
        for methode in (historical_var, parametric_var, cornish_fisher_var):
            t = methode(donnees, confidence=0.99)
            assert t.cvar >= t.var, f"{t.method} : CVaR {t.cvar} < VaR {t.var}"
            assert t.tail_ratio >= 1.0


def test_var_croit_avec_le_niveau_de_confiance():
    """Une VaR à 99 % doit dépasser une VaR à 95 %, qui dépasse celle à 90 %."""
    r = _normal(3000, seed=2)
    vars_ = [historical_var(r, confidence=c).var for c in (0.90, 0.95, 0.99)]
    for precedent, suivant in zip(vars_, vars_[1:]):
        assert suivant > precedent, f"non monotone : {vars_}"


def test_donnees_insuffisantes_refusees():
    from nkrisk.types import InsufficientData

    r = _normal(100, seed=3)
    with pytest.raises(InsufficientData):
        historical_var(r, min_obs=250)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — CALIBRATION CONTRE LA VÉRITÉ THÉORIQUE
# ══════════════════════════════════════════════════════════════════════


def test_var_parametrique_exacte_sur_donnees_normales():
    """
    Sur des données réellement normales, la VaR paramétrique doit
    retrouver le quantile théorique. C'est son seul régime de validité,
    et on vérifie qu'elle y est juste.
    """
    sigma_annuel = 0.20
    r = _normal(252 * 40, seed=4, sigma=sigma_annuel)
    sigma_quot = sigma_annuel / np.sqrt(252)

    t = parametric_var(r, confidence=0.99)
    attendu = -stats.norm.ppf(0.01) * sigma_quot  # ≈ 2.326 σ

    assert abs(t.var - attendu) / attendu < 0.03


def test_ratio_de_queue_gaussien_vaut_environ_1_15():
    """
    Valeur de référence calculable à la main : pour une normale à 99 %,
    CVaR/VaR ≈ 1,15. Si ce chiffre dérive, la formule de CVaR gaussienne
    est fausse.
    """
    r = _normal(252 * 40, seed=5)
    t = parametric_var(r, confidence=0.99)
    assert 1.12 < t.tail_ratio < 1.18, f"ratio = {t.tail_ratio:.3f}"


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — LE MENSONGE DE LA VaR PARAMÉTRIQUE
# ══════════════════════════════════════════════════════════════════════


def test_le_penalite_de_queue_epaisse_croit_avec_le_niveau():
    """
    ★ LA DÉCOUVERTE LA PLUS DÉRANGEANTE DE LA SÉANCE ★

    Mon attente initiale était que la VaR paramétrique sous-estime
    fortement le risque dès 99 %. Le test a échoué, et le calcul
    théorique explique pourquoi — la vérité est bien pire que ce que
    j'avais écrit.

    Quantiles d'une Student(4) NORMALISÉE à variance unitaire, rapportés
    à ceux de la normale :

        niveau    normale   student(4)   rapport
        90 %        1,282        1,084      0,85   ← plus PETIT !
        95 %        1,645        1,507      0,92   ← plus PETIT !
        99 %        2,326        2,649      1,14
        99,5 %      2,576        3,256      1,26
        99,9 %      3,090        5,072      1,64
        99,99 %     3,719        9,216      2,48

    Aux niveaux de confiance USUELS — 90 %, 95 % — la distribution à
    queues épaisses a un quantile PLUS FAIBLE que la normale. À variance
    égale, elle concentre davantage de masse près de zéro, ce qui doit
    bien être compensé quelque part : dans les extrêmes.

    CONSÉQUENCE PRATIQUE, ET ELLE EST GLAÇANTE : une banque qui mesure sa
    VaR à 95 % ne voit strictement rien d'anormal. À 99 %, l'écart n'est
    que de 14 % — attribuable au bruit d'estimation. Le danger n'apparaît
    qu'à 99,9 % et au-delà, c'est-à-dire précisément sur les événements
    qui font faillir les institutions.

    **Le modèle ne se trompe pas un peu partout : il se trompe énormément,
    mais seulement là où on ne regarde pas.**

    C'est le mode de défaillance le plus dangereux qui soit, parce qu'il
    est invisible à tous les contrôles de routine.
    """
    from scipy import stats as st

    rapports = []
    for c in (0.90, 0.95, 0.99, 0.999):
        p = 1 - c
        q_norm = -st.norm.ppf(p)
        q_t = -st.t.ppf(p, df=4) / np.sqrt(4 / 2)
        rapports.append(q_t / q_norm)

    # Le rapport croît strictement avec le niveau de confiance
    for precedent, suivant in zip(rapports, rapports[1:]):
        assert suivant > precedent, f"non monotone : {rapports}"

    # Aux niveaux usuels, la queue épaisse est TROMPEUSEMENT rassurante
    assert rapports[0] < 1.0, f"à 90 % le rapport devrait être < 1 : {rapports[0]:.2f}"
    assert rapports[1] < 1.0, f"à 95 % le rapport devrait être < 1 : {rapports[1]:.2f}"

    # Et il explose dans l'extrême
    assert rapports[-1] > 1.5, f"à 99,9 % : {rapports[-1]:.2f}"


def test_la_cvar_detecte_la_queue_bien_avant_la_var():
    """
    ★ L'ARGUMENT QUANTITATIF EN FAVEUR DE LA CVaR ★

    Puisque la VaR ne voit rien avant 99,9 %, que faire ? Regarder la
    CVaR, qui moyenne TOUTE la queue au lieu de lire un seul point.

    Pénalité de queue épaisse, Student(4) rapportée à la normale :

        niveau      VaR    CVaR
        95 %       0,92    1,10
        99 %       1,14    1,39
        99,9 %     1,64    2,03

    À 99 %, la VaR signale un écart de 14 %, la CVaR de 39 %. La CVaR
    détecte le danger à un niveau de confiance où la VaR le manque encore.

    C'est, avec la sous-additivité, la seconde raison pour laquelle Bâle
    est passé de la VaR à l'expected shortfall dans le cadre FRTB.
    """
    from scipy import stats as st

    for c, mini in ((0.95, 1.05), (0.99, 1.30)):
        p = 1 - c
        es_norm = st.norm.pdf(st.norm.ppf(p)) / p
        ps = np.linspace(1e-9, p, 100_000)
        es_t = -np.mean(st.t.ppf(ps, df=4)) / np.sqrt(2)

        q_norm = -st.norm.ppf(p)
        q_t = -st.t.ppf(p, df=4) / np.sqrt(2)

        penalite_cvar = es_t / es_norm
        penalite_var = q_t / q_norm

        assert penalite_cvar > penalite_var, (
            f"à {c:.0%} la CVaR devrait révéler plus que la VaR : "
            f"{penalite_cvar:.2f} vs {penalite_var:.2f}"
        )
        assert penalite_cvar > mini


def test_ratio_de_queue_revele_lepaisseur():
    """CVaR/VaR distingue nettement une queue normale d'une queue épaisse."""
    ratio_norm = historical_var(_normal(252 * 20, seed=7)).tail_ratio
    ratio_fat = historical_var(_fat(252 * 20, seed=7)).tail_ratio
    assert ratio_fat > ratio_norm * 1.3, f"{ratio_norm:.2f} vs {ratio_fat:.2f}"


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 4 — COUVERTURE : LA VaR TIENT-ELLE SA PROMESSE ?
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_couverture_var_historique_sur_donnees_normales():
    """
    ★ LE TEST DE COUVERTURE, TROISIÈME APPARITION ★

    Protocole hors échantillon, comme en production :
      1. estimer la VaR à 99 % sur 1 000 jours,
      2. l'appliquer aux 1 000 jours SUIVANTS, jamais vus,
      3. compter les dépassements,
      4. vérifier qu'ils tournent autour de 1 %.

    Estimer et tester sur les mêmes données serait de la triche — c'est
    l'erreur qui fait qu'un modèle passe tous les backtests internes puis
    explose en production.
    """
    taux = []
    for seed in range(40):
        r = _normal(2000, seed=seed + 3000)
        calib, test = r.iloc[:1000], r.iloc[1000:]
        v = historical_var(calib, confidence=0.99).var
        taux.append(kupiec_test(test, v, confidence=0.99).breach_rate)

    moyen = float(np.mean(taux))
    assert 0.005 < moyen < 0.020, f"taux moyen de dépassement = {moyen:.3%}"


@pytest.mark.slow
def test_la_parametrique_echoue_le_backtest_sur_queues_epaisses():
    """
    ★ LA DÉMONSTRATION LA PLUS ACCABLANTE DU PROJET ★

    Sur des données à queues épaisses, la VaR paramétrique promet 1 % de
    dépassements et en produit nettement plus. Le test de Kupiec doit la
    rejeter dans une large majorité des cas.

    C'est exactement ce qui se produit sur les marchés réels — et c'est
    la raison pour laquelle Bâle a fini par abandonner la VaR au profit
    de l'expected shortfall.
    """
    rejets_param = 0
    rejets_hist = 0
    n = 40

    for seed in range(n):
        r = _fat(3000, seed=seed + 4000)
        calib, test = r.iloc[:1500], r.iloc[1500:]

        v_p = parametric_var(calib, confidence=0.99).var
        v_h = historical_var(calib, confidence=0.99).var

        if kupiec_test(test, v_p, confidence=0.99).rejected:
            rejets_param += 1
        if kupiec_test(test, v_h, confidence=0.99).rejected:
            rejets_hist += 1

    # La paramétrique doit être rejetée bien plus souvent que l'historique
    assert rejets_param > rejets_hist, (
        f"paramétrique rejetée {rejets_param}/{n}, "
        f"historique {rejets_hist}/{n}"
    )
    assert rejets_param >= n * 0.4, (
        f"la paramétrique devrait échouer souvent sur queues épaisses : "
        f"{rejets_param}/{n}"
    )


def test_kupiec_accepte_un_modele_correct():
    """Contrôle de bon sens : un taux de dépassement exact ne doit pas être rejeté."""
    rng = np.random.default_rng(11)
    # 1 000 jours, exactement 10 dépassements attendus à 99 %
    r = pd.Series(rng.normal(0, 0.01, 1000))
    v = -float(np.quantile(r, 0.01))
    res = kupiec_test(r, v, confidence=0.99)
    assert not res.rejected
    assert abs(res.n_breaches - 10) <= 2


def test_kupiec_rejette_un_modele_trop_optimiste():
    """Une VaR délibérément deux fois trop faible doit être rejetée."""
    rng = np.random.default_rng(12)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    v = -float(np.quantile(r, 0.01)) / 2
    assert kupiec_test(r, v, confidence=0.99).rejected


# ══════════════════════════════════════════════════════════════════════
#  L'INCOHÉRENCE MATHÉMATIQUE
# ══════════════════════════════════════════════════════════════════════


def test_la_var_viole_la_sous_additivite():
    """
    ★ CALCULABLE À LA MAIN, ET DÉVASTATEUR ★

    Deux obligations indépendantes à 4 % de défaut, VaR à 95 %.
      • seule       : P(perte) = 4 % < 5 %  →  VaR = 0
      • combinées   : P(≥1 défaut) = 7,84 % > 5 %  →  VaR = 0,5

    La VaR affirme donc que diversifier a créé du risque à partir de rien.
    Elle n'est pas une mesure de risque cohérente au sens d'Artzner,
    Delbaen, Eber & Heath (1999).
    """
    res = subadditivity_counterexample(
        default_probability=0.04, confidence=0.95
    )

    assert res["var_actif_seul"] == 0.0
    assert res["var_portefeuille_combine"] > 0.0
    assert res["sous_additivite_violee"] is True
    assert abs(res["proba_au_moins_un_defaut"] - 0.0784) < 1e-9


def test_pas_de_violation_quand_le_defaut_est_frequent():
    """
    Contrôle négatif — tout aussi important que le test positif.

    Avec 10 % de défaut, la perte individuelle dépasse déjà le seuil de
    5 %, donc VaR(seule) > 0 et le paradoxe disparaît. Si ce test
    échouait, cela signifierait que notre fonction fabrique la violation
    au lieu de la constater.
    """
    res = subadditivity_counterexample(
        default_probability=0.10, confidence=0.95
    )
    assert res["var_actif_seul"] > 0.0
    assert res["sous_additivite_violee"] is False
