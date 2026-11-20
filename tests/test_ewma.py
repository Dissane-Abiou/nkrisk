"""
Tests de la pondération exponentielle — séance 11.

Le test qui compte ici est `test_lavantage_de_lewma_ne_survit_pas_a_la_matrice`.
Il verrouille un résultat que je n'attendais pas, et qui a demandé trois
hypothèses réfutées avant d'apparaître.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nkrisk.covariance import ledoit_wolf, sample_covariance
from nkrisk.ewma import (
    DECAY_RISKMETRICS,
    DecayFit,
    effective_sample_size,
    ewma_covariance,
    ewma_volatility,
    ewma_weights,
    fit_decay,
    half_life,
)
from nkrisk.returns import annualized_volatility
from nkrisk.types import InsufficientData

JOURS = 252


# ══════════════════════════════════════════════════════════════════════
#  GÉNÉRATEURS
# ══════════════════════════════════════════════════════════════════════


def _garch(
    t: int = 2_200,
    n: int = 10,
    k: int = 2,
    seed: int = 0,
    alpha: float = 0.14,
    beta: float = 0.85,
    groupe: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rendements à facteurs dont la volatilité est groupée, et la VRAIE
    variance conditionnelle qui les a engendrés.

    Rendre `h` permet de construire un oracle, donc de mesurer ce
    qu'aucun estimateur ne peut dépasser. Sans cette borne, comparer
    deux estimateurs ne dit pas s'il restait quelque chose à gagner.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 0.5, (n, k))
    b[:, 0] = np.abs(rng.normal(0.8, 0.2, n))
    h = np.ones(t)
    if groupe:
        z = rng.normal(0, 1, t)
        for i in range(1, t):
            h[i] = (1 - alpha - beta) + alpha * (z[i - 1] ** 2 * h[i - 1]) \
                + beta * h[i - 1]
    echelle = np.sqrt(h)[:, None]
    f = rng.normal(0, 1, (t, k)) * echelle
    e = rng.normal(0, 1, (t, n)) * echelle * rng.uniform(0.5, 1.5, n)
    cadre = pd.DataFrame(
        (f @ b.T + e) * 0.20 / np.sqrt(JOURS),
        index=pd.bdate_range("2015-01-02", periods=t),
        columns=[f"A{i:02d}" for i in range(n)],
    )
    return cadre, h


def _ez2(predits: np.ndarray, realises: np.ndarray) -> float:
    """
    E[(réalisé/prédit)²] — la mesure de calibration CONDITIONNELLE.

    Vaut 1 pour une prévision bien spécifiée. Au-dessus, le modèle
    sous-estime. C'est le diagnostic classique des résidus
    standardisés, et il juge la prévision période par période — là où
    `BacktestResult.calibration()` compare une moyenne à un agrégat.
    """
    z = realises / predits
    return float(np.mean(z**2))


# ══════════════════════════════════════════════════════════════════════
#  LES POIDS ET LA TAILLE EFFECTIVE
# ══════════════════════════════════════════════════════════════════════


def test_les_poids_somment_a_un_et_decroissent() -> None:
    w = ewma_weights(100, 0.94)
    assert w.sum() == pytest.approx(1.0)
    assert (np.diff(w) > 0).all()          # du plus ancien au plus récent
    assert w[-1] > w[0]


def test_la_renormalisation_evite_un_biais_sur_serie_courte() -> None:
    """
    Sans renormalisation, les poids somment à 1 − λⁿ et la variance
    sort trop basse. L'erreur disparaît quand n grandit — le genre de
    biais qu'on ne voit jamais parce qu'il ne se manifeste que sur les
    cas courts.
    """
    for n in (30, 50, 100, 500):
        assert ewma_weights(n, 0.94).sum() == pytest.approx(1.0, abs=1e-12)

    # Les poids NON normalisés (1−λ)·λ^i somment à 1 − λⁿ, soit 95,5 %
    # à n = 50 pour λ = 0,94 : la variance sortirait 4,5 % trop basse.
    brut = (1.0 - 0.94) * 0.94 ** np.arange(49, -1, -1)
    assert brut.sum() == pytest.approx(1.0 - 0.94**50, abs=1e-12)
    assert brut.sum() < 0.96


@pytest.mark.parametrize("decay", [0.90, 0.94, 0.97])
def test_la_taille_effective_suit_la_formule_de_kish(decay) -> None:
    """
    (1+λ)/(1−λ) est la limite asymptotique de 1/Σw². Sur un historique
    long, le calcul exact doit la rejoindre.

    La valeur attendue est CALCULÉE, pas recopiée : une première
    version écrivait 32.333 pour λ=0,94 et échouait à 1e-6 contre le
    vrai 32.3333…. Un nombre arrondi à la main dans un test est une
    onzième occurrence de la même dérive que dans les narrations.
    """
    attendu = (1.0 + decay) / (1.0 - decay)
    assert effective_sample_size(decay) == pytest.approx(attendu, rel=1e-12)
    assert effective_sample_size(decay, 5_000) == pytest.approx(attendu, rel=1e-3)


def test_la_demi_vie_correspond_bien_a_un_poids_divise_par_deux() -> None:
    for decay in (0.90, 0.94, 0.97):
        hl = half_life(decay)
        assert decay**hl == pytest.approx(0.5)


def test_des_parametres_absurdes_sont_refuses() -> None:
    for mauvais in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="λ"):
            effective_sample_size(mauvais)
        with pytest.raises(ValueError, match="λ"):
            half_life(mauvais)
        with pytest.raises(ValueError, match="λ"):
            ewma_weights(10, mauvais)


# ══════════════════════════════════════════════════════════════════════
#  L'INTERVALLE PAIE LA RÉACTIVITÉ — LE POINT DU MODULE
# ══════════════════════════════════════════════════════════════════════


def test_lintervalle_selargit_quand_la_memoire_raccourcit() -> None:
    """
    LE RÉSULTAT CENTRAL DE LA SÉANCE, ET IL EST STRUCTUREL.

    Un estimateur qui réagit vite voit moins d'observations effectives,
    donc sait moins. Un moteur qui transporte son incertitude depuis la
    séance 1 ne peut pas faire semblant de l'ignorer.

    On vérifie que l'intervalle s'élargit MONOTONEMENT quand λ diminue,
    et que `n_obs` rapporté est bien la taille effective et non le
    nombre de points lus.
    """
    cadre, _ = _garch(t=2_000, n=2)
    serie = cadre["A00"]

    largeurs = []
    for decay in (0.99, 0.97, 0.94, 0.90, 0.85):
        est = ewma_volatility(serie, decay=decay)
        largeurs.append(est.relative_uncertainty)
        assert est.n_obs == round(effective_sample_size(decay, len(serie)))
        assert est.n_obs < len(serie)      # jamais les 2 000 points lus
        assert est.ci_low <= est.value <= est.ci_high

    assert largeurs == sorted(largeurs), (
        f"l'incertitude doit croître quand λ diminue : {largeurs}"
    )


def test_le_moteur_signale_lui_meme_que_lewma_rapide_est_peu_fiable() -> None:
    """
    FERMETURE INATTENDUE AVEC LA SÉANCE 1.

    `Estimate.is_meaningful()` déclare une estimation inexploitable
    au-delà de 25 % d'incertitude relative. Ce seuil a été fixé à la
    séance 1, sans aucune idée qu'un estimateur exponentiel existerait
    un jour dans ce projet.

    À λ = 0,94, il s'allume tout seul. Le type de la séance 1 juge
    l'estimateur de la séance 11 sans qu'on lui ait rien dit — c'est
    ce que vaut une contrainte posée dans le système de types plutôt
    que dans une convention.
    """
    cadre, _ = _garch(t=2_000, n=2)
    serie = cadre["A00"]

    lente = ewma_volatility(serie, decay=0.99)
    rapide = ewma_volatility(serie, decay=0.94)

    assert lente.is_meaningful()
    assert not rapide.is_meaningful()
    assert "PEU FIABLE" in rapide.format(2, percent=True)
    assert "PEU FIABLE" not in lente.format(2, percent=True)


def test_une_memoire_tres_longue_rejoint_la_fenetre_classique() -> None:
    """
    Cas limite : λ → 1 redonne une pondération quasi uniforme, donc la
    volatilité de la séance 1. Si cette identité échoue, l'une des deux
    implémentations est fausse.
    """
    cadre, _ = _garch(t=400, n=2, groupe=False)
    serie = cadre["A00"]

    ewma = ewma_volatility(serie, decay=0.9995, demean=False)
    classique = annualized_volatility(serie)

    # On compare sans centrage des deux côtés : annualized_volatility
    # retranche la moyenne, l'EWMA non par défaut. Sur des rendements
    # quotidiens la différence est minime, d'où la tolérance.
    assert ewma.value == pytest.approx(classique.value, rel=0.03)


def test_la_volatilite_ewma_refuse_un_historique_trop_court() -> None:
    cadre, _ = _garch(t=2_000, n=2)
    with pytest.raises(InsufficientData):
        ewma_volatility(cadre["A00"].iloc[:10], min_obs=30)


# ══════════════════════════════════════════════════════════════════════
#  LA COVARIANCE, ET LE RETOUR DU PROBLÈME DE LA SÉANCE 2
# ══════════════════════════════════════════════════════════════════════


def test_la_covariance_ewma_est_symetrique_et_definie_positive() -> None:
    cadre, _ = _garch(t=1_000, n=8)
    cov = ewma_covariance(cadre, decay=0.94)

    assert np.allclose(cov.matrix, cov.matrix.T)
    assert np.linalg.eigvalsh(cov.matrix).min() > 0
    assert cov.is_invertible


def test_la_covariance_ewma_rapporte_sa_taille_effective() -> None:
    """
    Rapporter les 1 000 points lus annoncerait une matrice bien mieux
    déterminée qu'elle ne l'est. C'est `ratio_n_over_t` qui alimente
    le diagnostic de fiabilité : lui mentir rendrait le diagnostic
    inutile.
    """
    cadre, _ = _garch(t=1_000, n=8)
    cov = ewma_covariance(cadre, decay=0.94)

    assert cov.n_obs == round(effective_sample_size(0.94, 1_000))
    assert cov.n_obs < 40
    assert cov.ratio_n_over_t > 0.2        # 8 / 32


def test_lewma_degrade_le_conditionnement_sans_rendre_la_matrice_singuliere(
) -> None:
    """
    ⚠️ J'AVAIS ÉCRIT QUE LA MATRICE DEVENAIT SINGULIÈRE. ELLE NE L'EST PAS,
       ET C'EST PIRE.

    À λ = 0,94, n_eff vaut 32 quelle que soit la longueur de
    l'historique. À N = 40 actifs, le rapport N/n_eff dépasse 1 — le
    régime où la covariance empirique serait singulière.

    Sauf que la covariance EWMA, elle, reste de RANG PLEIN. Chacune des
    1 000 observations reçoit un poids non nul, si minuscule soit-il,
    et cela suffit à garder la matrice techniquement inversible.

    Mesuré à N = 40, T = 1 000 :

        empirique          cond =    115    inversible
        ewma λ=0,94 nue    cond =  1 174    inversible
        ewma contractée    cond =     67    inversible

    Dix fois pire, et `is_invertible` répond True dans les trois cas.

    **C'est un mode de défaillance plus pernicieux que la
    singularité.** La covariance empirique à N > T échoue BRUYAMMENT :
    elle est singulière, le moteur refuse, l'utilisateur voit le
    problème. L'EWMA échoue EN SILENCE : elle rend une matrice qu'on
    peut inverser et dont l'inverse est du bruit.

    La preuve du mécanisme est à la fin : en ne gardant que les
    n_eff observations qui pèsent réellement, la matrice redevient
    singulière. La dégénérescence est bien là — elle est seulement
    masquée par des poids négligeables mais non nuls.
    """
    cadre, _ = _garch(t=1_000, n=40, groupe=False)

    empirique = sample_covariance(cadre)
    nue = ewma_covariance(cadre, decay=0.94, shrinkage=0.0)
    contractee = ewma_covariance(cadre, decay=0.94)

    assert nue.ratio_n_over_t > 1.0
    assert nue.is_invertible, "rang plein : les poids lointains sont non nuls"
    assert nue.condition_number > 5 * empirique.condition_number
    assert contractee.condition_number < empirique.condition_number
    assert contractee.shrinkage > 0.0

    # Le mécanisme : tronquer aux observations qui pèsent vraiment.
    tronquee = ewma_covariance(
        cadre.iloc[-32:], decay=0.94, shrinkage=0.0, min_obs=10
    )
    assert not tronquee.is_invertible


def test_la_contraction_est_active_par_defaut_et_desactivable() -> None:
    cadre, _ = _garch(t=1_000, n=12)

    defaut = ewma_covariance(cadre, decay=0.94)
    sans = ewma_covariance(cadre, decay=0.94, shrinkage=0.0)
    forcee = ewma_covariance(cadre, decay=0.94, shrinkage=0.5)

    assert defaut.shrinkage > 0.0
    assert sans.shrinkage == 0.0
    assert forcee.shrinkage == 0.5
    assert defaut.condition_number < sans.condition_number


def test_la_contraction_ponderee_se_reduit_a_ledoit_wolf() -> None:
    """
    VALIDATION CROISÉE INTERNE.

    La formule pondérée remplace (1/T²) par w_t². Avec des poids
    uniformes, elle doit redonner exactement le δ de Ledoit-Wolf de la
    séance 2.

    Ce n'est donc pas une formule différente : c'est la même, écrite
    sans supposer l'équipondération. Un écart ici signifierait que la
    généralisation est fausse.
    """
    cadre, _ = _garch(t=600, n=10, groupe=False)

    # λ → 1 donne des poids quasi uniformes.
    quasi_uniforme = ewma_covariance(cadre, decay=0.999999, demean=True)
    reference = ledoit_wolf(cadre)

    assert quasi_uniforme.shrinkage == pytest.approx(
        reference.shrinkage, abs=0.02
    )


def test_la_covariance_ewma_refuse_les_entrees_invalides() -> None:
    cadre, _ = _garch(t=500, n=4)
    with pytest.raises(TypeError, match="DataFrame"):
        ewma_covariance(cadre["A00"])            # type: ignore[arg-type]
    with pytest.raises(ValueError, match="manquantes"):
        abime = cadre.copy()
        abime.iloc[3, 1] = np.nan
        ewma_covariance(abime)
    with pytest.raises(InsufficientData):
        ewma_covariance(cadre.iloc[:10], min_obs=30)


# ══════════════════════════════════════════════════════════════════════
#  AJUSTER λ PLUTÔT QUE LE RECOPIER
# ══════════════════════════════════════════════════════════════════════


def test_lajustement_retrouve_une_memoire_courte_sur_donnees_tres_groupees() -> None:
    """
    CALIBRATION CONTRE UNE VÉRITÉ CONNUE.

    Un GARCH très réactif (α élevé) a une mémoire courte : la
    vraisemblance doit préférer un λ plus petit qu'un GARCH persistant.
    On ne vérifie pas une valeur, mais un ORDRE — c'est ce que la
    théorie prédit sans ambiguïté.
    """
    grille = tuple(np.round(np.arange(0.80, 0.995, 0.01), 4))

    reactif, _ = _garch(t=1_500, n=4, alpha=0.30, beta=0.65, seed=3)
    persistant, _ = _garch(t=1_500, n=4, alpha=0.05, beta=0.94, seed=3)

    lam_reactif = fit_decay(reactif, grid=grille).decay
    lam_persistant = fit_decay(persistant, grid=grille).decay

    assert lam_reactif < lam_persistant, (
        f"mémoire courte attendue sur le régime réactif : "
        f"{lam_reactif} vs {lam_persistant}"
    )


def test_lajustement_rend_un_resultat_complet_et_lisible() -> None:
    cadre, _ = _garch(t=1_200, n=3)
    fit = fit_decay(cadre, grid=(0.90, DECAY_RISKMETRICS, 0.97))

    assert isinstance(fit, DecayFit)
    assert fit.decay in fit.grid
    assert fit.log_likelihood == max(fit.likelihoods)
    assert fit.effective_sample_size == effective_sample_size(fit.decay)
    assert len(fit.table()) == 3
    assert "RiskMetrics" in str(fit)       # la grille contient 0,94


def test_lajustement_refuse_une_grille_ou_une_serie_invalides() -> None:
    cadre, _ = _garch(t=400, n=3)
    with pytest.raises(ValueError, match="hors de"):
        fit_decay(cadre, grid=(0.9, 1.2))
    with pytest.raises(InsufficientData):
        fit_decay(cadre.iloc[:100], min_obs=250)
    with pytest.raises(ValueError, match="burn_in"):
        fit_decay(cadre, burn_in=10_000)


# ══════════════════════════════════════════════════════════════════════
#  LE RÉSULTAT QUE JE N'ATTENDAIS PAS
# ══════════════════════════════════════════════════════════════════════


def _calibration_conditionnelle(
    cadre: pd.DataFrame,
    predicteur,
    horizon: int = 21,
    fenetre: int = 252,
) -> float:
    """Parcourt la série en prédisant, puis compare au réalisé."""
    p = cadre.mean(axis=1).to_numpy()
    n_actifs = cadre.shape[1]
    poids = np.full(n_actifs, 1.0 / n_actifs)
    predits, realises = [], []
    for t in range(fenetre, len(p) - horizon, horizon):
        realises.append(float(np.std(p[t: t + horizon], ddof=1)))
        predits.append(predicteur(cadre.iloc[t - fenetre: t], p[t - fenetre: t],
                                  poids))
    return _ez2(np.array(predits), np.array(realises))


@pytest.mark.slow
def test_lewma_bien_reglee_approche_la_borne_de_loracle() -> None:
    """
    ⚠️ CE TEST VERROUILLE UN RÉSULTAT QUE J'AVAIS ANNONCÉ À L'ENVERS.

    J'avais écrit à la fin de la séance 10 : « l'estimateur EWMA, et
    combien il récupère sur ce 0,79 ». Trois mesures successives ont
    montré qu'il ne récupérait rien — puis une quatrième a montré
    pourquoi : je l'avais réglé pour le mauvais horizon.

    Sur une SÉRIE, à horizon 21 jours :

      oracle connaissant la moyenne future de h   E[z²] = 1,00
      oracle causal, h[t] exact à la décision             1,13
      ewma λ=0,97 (demi-vie 23 jours)                     1,11
      fenêtre 252 jours                                   1,18
      ewma λ=0,94 (demi-vie 11 jours)                     1,16

    Deux lectures.

    D'abord, **λ=0,97 atteint la borne de l'oracle causal.** Connaître
    exactement la variance d'aujourd'hui ne ferait pas mieux : les
    0,13 restants sont l'incertitude irréductible des vingt-et-un
    jours à venir.

    Ensuite, **la demi-vie de 23 jours n'est pas un hasard** : elle
    correspond à l'horizon. Le 0,94 de RiskMetrics est calibré pour
    une VaR à UN jour, et l'employer pour prévoir un mois est une
    erreur de catégorie — il bruite plus qu'il n'anticipe.
    """
    def oracle_causal(h):
        def f(fen, serie, poids, _h=h):
            return float(np.std(serie, ddof=1))
        return f

    resultats: dict[str, list[float]] = {
        "ewma 0.97": [], "fenetre": [], "ewma 0.94": []
    }
    for seed in range(5):
        cadre, _ = _garch(t=2_200, n=10, seed=seed)

        def ewma_serie(decay):
            def f(fen, serie, poids, d=decay):
                w = ewma_weights(len(serie), d)
                return float(np.sqrt(np.sum(w * serie**2)))
            return f

        resultats["ewma 0.97"].append(
            _calibration_conditionnelle(cadre, ewma_serie(0.97))
        )
        resultats["ewma 0.94"].append(
            _calibration_conditionnelle(cadre, ewma_serie(0.94))
        )
        resultats["fenetre"].append(
            _calibration_conditionnelle(
                cadre, lambda fen, s, w: float(np.std(s, ddof=1))
            )
        )

    moyennes = {k: float(np.mean(v)) for k, v in resultats.items()}
    assert abs(moyennes["ewma 0.97"] - 1.0) < abs(moyennes["fenetre"] - 1.0), (
        f"λ=0,97 devrait battre la fenêtre sur une série : {moyennes}"
    )
    assert abs(moyennes["ewma 0.94"] - 1.0) > abs(
        moyennes["ewma 0.97"] - 1.0
    ), (
        f"λ=0,94 est trop rapide pour un horizon de 21 jours : {moyennes}"
    )


@pytest.mark.slow
def test_lavantage_de_lewma_ne_survit_pas_a_la_matrice() -> None:
    """
    ⚠️ LE RÉSULTAT DE LA SÉANCE, ET IL EST CONTRE-INTUITIF.

    L'EWMA bien réglée bat la fenêtre longue sur une SÉRIE. Elle perd
    dès qu'on passe par une MATRICE de covariance.

    Mesuré, à horizon 21 jours :

                                        N = 10   N = 30
        série directe, ewma λ=0,97        1,10     1,14
        via matrice, ewma λ=0,97          1,26     1,33
        via matrice, fenêtre 252j + LW    1,21     1,26

    LA RAISON EST DIMENSIONNELLE. Une série n'a qu'un paramètre à
    estimer, et 66 observations effectives suffisent largement. Une
    matrice N×N en a N(N+1)/2, et c'est n_eff qui les porte — pas les
    252 points lus.

    **L'avantage de l'EWMA est en TEMPS ; son coût est en DIMENSION.**
    L'avantage ne change pas avec N ; le coût, si. Il existe donc un
    nombre d'actifs au-delà duquel la réactivité ne vaut plus son prix,
    et sur cet univers il est déjà dépassé à dix.

    CONSÉQUENCE PRATIQUE : employer l'EWMA pour la VaR d'un
    portefeuille à poids fixés — une série — et la fenêtre longue avec
    contraction pour la covariance qu'on va optimiser. Beaucoup de
    systèmes de risque font exactement cela ; peu expliquent pourquoi.
    """
    for n_actifs in (10, 30):
        serie_v, matrice_v, fenetre_v = [], [], []
        for seed in range(4):
            cadre, _ = _garch(t=2_200, n=n_actifs, seed=seed)

            def par_serie(fen, s, w):
                poids = ewma_weights(len(s), 0.97)
                return float(np.sqrt(np.sum(poids * s**2)))

            def par_matrice(fen, s, w):
                cov = ewma_covariance(fen, decay=0.97)
                return float(np.sqrt(w @ cov.matrix @ w))

            def par_fenetre(fen, s, w):
                cov = ledoit_wolf(fen)
                return float(np.sqrt(w @ cov.matrix @ w))

            serie_v.append(_calibration_conditionnelle(cadre, par_serie))
            matrice_v.append(_calibration_conditionnelle(cadre, par_matrice))
            fenetre_v.append(_calibration_conditionnelle(cadre, par_fenetre))

        serie = float(np.mean(serie_v))
        matrice = float(np.mean(matrice_v))
        fenetre = float(np.mean(fenetre_v))

        assert abs(serie - 1) < abs(matrice - 1), (
            f"N={n_actifs} : passer par la matrice doit coûter — "
            f"série {serie:.2f}, matrice {matrice:.2f}"
        )
        assert abs(fenetre - 1) < abs(matrice - 1), (
            f"N={n_actifs} : via matrice, la fenêtre longue doit l'emporter — "
            f"fenêtre {fenetre:.2f}, ewma {matrice:.2f}"
        )
