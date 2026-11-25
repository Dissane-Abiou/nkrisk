"""
Tests du module de validation (séance 12).

Quatre niveaux, comme partout dans ce projet :

  1. Propriétés mathématiques — identités qui tiennent quelles que
     soient les données.
  2. Calibration contre vérité connue — on fabrique une erreur de
     taille choisie et on vérifie que l'instrument la retrouve.
  3. Couverture Monte Carlo — on ne valide pas l'estimation, on valide
     l'INTERVALLE. Marqués `slow`.
  4. Cohérence avec l'API existante — la statistique de biais et
     `BacktestResult.calibration()` doivent s'accorder là où elles
     mesurent la même chose, et on verrouille le fait qu'elles
     divergent ailleurs.

S'y ajoute un niveau que la séance 8 a imposé à tout ce projet : un
test de CAUSALITÉ, qui vérifie qu'aucune décision ne regarde le futur.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import (
    prevision_garch_causale,
    trajectoire_garch,
    univers_facteurs,
)

from nkrisk.backtest import MinimumVariance, walk_forward
from nkrisk.covariance import CovarianceEstimate, sample_covariance
from nkrisk.types import Estimate, InsufficientData
from nkrisk.validation import (
    INAPPLICABLE,
    MIN_BLOCKS,
    REJETE,
    SURVEILLANCE,
    VALIDE,
    CovarianceModel,
    EwmaModel,
    FactorCovariance,
    LedoitWolfModel,
    ModelInventory,
    SampleModel,
    bias_statistic,
    build_inventory,
    validate_covariance_model,
    with_materiality,
)

VOL = 0.012


def rendements_gaussiens(n: int, graine: int = 0, vol: float = VOL) -> np.ndarray:
    return np.random.default_rng(graine).normal(0.0, vol, n)


def univers(
    n_actifs: int = 15,
    n_facteurs: int = 3,
    n_periodes: int = 800,
    graine: int = 0,
    loadings_positifs: bool = False,
) -> pd.DataFrame:
    """
    Alias local sur le générateur partagé de `conftest.py`.

    Les SIGNATURES diffèrent d'un fichier de tests à l'autre — ordre des
    arguments, valeurs par défaut — parce qu'elles ont divergé avant que
    le générateur ne soit mis en commun. On garde l'alias plutôt que de
    réécrire cent appels : ce qui comptait était de supprimer le CORPS
    dupliqué, pas d'uniformiser les habitudes d'appel.

    ⚠️ `loadings_positifs=False` ici, contre `True` ailleurs : ce fichier
    a été écrit sur des expositions à signes aléatoires, où la
    corrélation moyenne est voisine de zéro. Le sens de plusieurs
    résultats s'inverse entre les deux régimes — voir
    `test_le_sens_du_biais_dune_diagonale_suit_le_signe_des_correlations`.
    """
    return univers_facteurs(
        n_actifs,
        graine,
        n_facteurs=n_facteurs,
        n_periodes=n_periodes,
        loadings_positifs=loadings_positifs,
    )


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — propriétés mathématiques
# ══════════════════════════════════════════════════════════════════════


def test_une_prevision_exacte_donne_un_biais_de_un():
    r = rendements_gaussiens(3000, graine=1)
    test = bias_statistic(r, np.full(r.size, VOL))
    assert test.bias.value == pytest.approx(1.0, abs=0.05)
    assert test.verdict == VALIDE


def test_le_biais_est_exactement_inversement_proportionnel_a_la_prevision():
    """
    Propriété exacte, pas statistique : b(c·σ̂) = b(σ̂) / c.

    Elle découle de la définition et doit tenir au bit près, pour
    n'importe quelles données. C'est le test qui attrape une erreur de
    facteur d'échelle — la famille de bugs la plus silencieuse de ce
    projet depuis la séance 1.
    """
    r = rendements_gaussiens(500, graine=2)
    base = bias_statistic(r, np.full(r.size, VOL)).bias.value
    for c in (0.5, 0.8, 1.3, 2.0, 7.0):
        mis_echelle = bias_statistic(r, np.full(r.size, VOL * c)).bias.value
        assert mis_echelle == pytest.approx(base / c, rel=1e-12)


def test_le_biais_ne_depend_pas_de_lunite_des_rendements():
    """
    Multiplier réalisé ET prédit par la même constante ne change rien.

    Le rapport z est sans dimension ; si ce test échouait, c'est qu'une
    constante d'échelle serait entrée là où elle n'a rien à faire.
    """
    r = rendements_gaussiens(400, graine=3)
    s = np.full(r.size, VOL)
    reference = bias_statistic(r, s).bias.value
    for facteur in (1e-3, 100.0):
        assert bias_statistic(r * facteur, s * facteur).bias.value == pytest.approx(
            reference, rel=1e-12
        )


def test_a_un_bloc_lintervalle_redonne_la_formule_du_kurtosis():
    """
    L'affirmation du docstring, vérifiée : à `block_size=1`,
    l'estimateur par blocs égale

        écart-type(b) = b · √((κ̂ − 1)/T) / 2

    au facteur de Bessel √(T/(T−1)) près. Une docstring qui annonce une
    égalité sans test est une docstring qui dérivera.
    """
    r = rendements_gaussiens(1500, graine=4)
    s = np.full(r.size, VOL)
    test = bias_statistic(r, s, block_size=1, confidence=0.95)

    z2 = (r / s) ** 2
    t_obs = z2.size
    m2 = float(z2.mean())
    kurt = float((z2**2).mean() / m2**2)
    sd_formule = math.sqrt(m2) * math.sqrt((kurt - 1.0) / t_obs) / 2.0
    sd_bessel = sd_formule * math.sqrt(t_obs / (t_obs - 1))

    z_crit = 1.959963984540054
    sd_module = test.bias.half_width / z_crit
    assert sd_module == pytest.approx(sd_bessel, rel=1e-10)
    assert test.kurtosis == pytest.approx(kurt, rel=1e-12)


def test_la_dispersion_vaut_un_quand_les_z_sont_independants():
    """
    Sous l'hypothèse nulle gaussienne et indépendante, la dispersion
    entre blocs est exactement celle que le hasard impose.
    """
    r = rendements_gaussiens(4200, graine=5)
    test = bias_statistic(r, np.full(r.size, VOL), block_size=21)
    assert test.dispersion_ratio == pytest.approx(1.0, abs=0.25)


def test_la_dispersion_grossit_quand_la_prevision_est_constante_a_tort():
    """
    Une prévision constante sur une volatilité qui change par paliers
    reste juste EN MOYENNE mais fausse à chaque période. Le niveau ne
    doit pas le voir ; la dispersion doit le voir.
    """
    g = np.random.default_rng(6)
    bas, haut = VOL * 0.5, VOL * math.sqrt(2.0 * 1.0 - 0.25)
    bloc = 21
    morceaux = []
    for i in range(80):
        vol = bas if i % 2 == 0 else haut
        morceaux.append(g.normal(0.0, vol, bloc))
    r = np.concatenate(morceaux)
    vol_moyenne = math.sqrt(float(np.mean(r**2)))

    test = bias_statistic(r, np.full(r.size, vol_moyenne), block_size=bloc)
    assert test.bias.value == pytest.approx(1.0, abs=0.02)  # niveau juste
    assert test.dispersion_ratio > 2.0  # stabilité mauvaise


def test_les_trois_verdicts_sont_atteignables_et_dans_le_bon_ordre():
    r = rendements_gaussiens(4000, graine=7)
    n = r.size

    assert bias_statistic(r, np.full(n, VOL)).verdict == VALIDE
    # Erreur réelle mais petite : détectée, non matérielle.
    assert bias_statistic(r, np.full(n, VOL * 0.95)).verdict == SURVEILLANCE
    # Erreur réelle et grosse.
    assert bias_statistic(r, np.full(n, VOL * 0.6)).verdict == REJETE


def test_le_seuil_de_materialite_deplace_la_frontiere():
    r = rendements_gaussiens(4000, graine=8)
    annonce = np.full(r.size, VOL * 0.95)
    assert bias_statistic(r, annonce, materiality=0.10).verdict == SURVEILLANCE
    assert bias_statistic(r, annonce, materiality=0.01).verdict == REJETE
    assert bias_statistic(r, annonce, materiality=0.50).verdict == SURVEILLANCE


def test_with_materiality_rejuge_sans_recalculer():
    donnees = univers(graine=9)
    carte = validate_covariance_model(donnees, SampleModel(), lookback=252, step=21)
    plus_strict = with_materiality(carte, 1e-9)
    # Les mesures sont intactes ; seul le verdict peut changer.
    assert plus_strict.bias is not None and carte.bias is not None
    assert plus_strict.bias.bias.value == carte.bias.bias.value
    assert plus_strict.n_obs == carte.n_obs
    assert plus_strict.verdict in (VALIDE, SURVEILLANCE, REJETE)


def test_le_sens_du_danger_est_ecrit_en_mots():
    r = rendements_gaussiens(3000, graine=10)
    sous = bias_statistic(r, np.full(r.size, VOL * 0.7))
    sur = bias_statistic(r, np.full(r.size, VOL * 1.4))
    assert sous.bias.value > 1.0
    assert "SOUS-ESTIME" in sous.direction
    assert "dangereux" in sous.direction
    assert sur.bias.value < 1.0
    assert "surestime" in sur.direction


def test_lestimate_retourne_respecte_ses_propres_invariants():
    r = rendements_gaussiens(900, graine=11)
    test = bias_statistic(r, np.full(r.size, VOL), block_size=21)
    assert isinstance(test.bias, Estimate)
    assert test.bias.ci_low <= test.bias.value <= test.bias.ci_high
    assert test.bias.ci_low >= 0.0  # une volatilité relative négative n'existe pas
    assert test.bias.n_obs == r.size
    assert test.n_blocks == r.size // 21
    assert test.bias.assumptions  # les hypothèses ne sont jamais vides


@pytest.mark.parametrize(
    ("kwargs", "motif"),
    [
        ({"block_size": 0}, "block_size"),
        ({"block_size": -3}, "block_size"),
        ({"confidence": 0.0}, "confidence"),
        ({"confidence": 1.0}, "confidence"),
    ],
)
def test_les_parametres_absurdes_sont_refuses(kwargs, motif):
    r = rendements_gaussiens(500, graine=12)
    with pytest.raises(ValueError, match=motif):
        bias_statistic(r, np.full(r.size, VOL), **kwargs)


def test_des_longueurs_differentes_sont_refusees():
    with pytest.raises(ValueError, match="même longueur"):
        bias_statistic(np.zeros(10), np.ones(11))


def test_un_nan_est_refuse_et_non_ecarte_silencieusement():
    r = rendements_gaussiens(200, graine=13)
    s = np.full(r.size, VOL)
    s[42] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        bias_statistic(r, s)


def test_une_prevision_nulle_leve_une_erreur_plutot_que_de_diviser_par_zero():
    """
    Division par zéro : numpy retournerait `inf` avec un avertissement,
    et le biais sortirait infini sans que rien ne s'arrête. Une
    prévision de risque nulle est un bug en amont, pas une petite
    prévision.
    """
    r = rendements_gaussiens(200, graine=14)
    s = np.full(r.size, VOL)
    s[7] = 0.0
    with pytest.raises(ValueError, match="nulle"):
        bias_statistic(r, s)
    s[7] = -VOL
    with pytest.raises(ValueError, match="négative"):
        bias_statistic(r, s)


def test_trop_peu_de_blocs_leve_insufficient_data():
    """
    `InsufficientData` et non `ValueError` : il s'agit bien d'un MANQUE
    d'observations, et l'appelant doit pouvoir distinguer ce cas d'un
    défaut de données.
    """
    r = rendements_gaussiens(40, graine=15)
    with pytest.raises(InsufficientData):
        bias_statistic(r, np.full(r.size, VOL), block_size=21)

    # Juste en dessous du seuil, puis juste au-dessus.
    taille = 10
    r_court = rendements_gaussiens(taille * (MIN_BLOCKS - 1), graine=16)
    with pytest.raises(InsufficientData):
        bias_statistic(r_court, np.full(r_court.size, VOL), block_size=taille)
    r_juste = rendements_gaussiens(taille * MIN_BLOCKS, graine=16)
    bias_statistic(r_juste, np.full(r_juste.size, VOL), block_size=taille)


def test_le_bloc_incomplet_est_tronque_et_non_complete():
    """
    Un bloc partiel compterait autant qu'un bloc plein dans la
    dispersion entre blocs, ce qui gonflerait l'intervalle. On tronque,
    et `n_obs` le dit.
    """
    r = rendements_gaussiens(21 * 7 + 5, graine=17)
    test = bias_statistic(r, np.full(r.size, VOL), block_size=21)
    assert test.n_blocks == 7
    assert test.bias.n_obs == r.size  # on déclare ce qu'on a reçu


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — calibration contre vérité connue
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("facteur", [0.6, 0.8, 1.25, 1.6])
def test_une_erreur_de_taille_choisie_est_retrouvee(facteur):
    """Le cœur du niveau 2 : on fabrique l'erreur, l'instrument la lit."""
    r = rendements_gaussiens(8000, graine=18)
    test = bias_statistic(r, np.full(r.size, VOL * facteur))
    assert test.bias.value == pytest.approx(1.0 / facteur, rel=0.03)


def test_un_modele_juste_sur_donnees_stationnaires_est_valide():
    donnees = univers(n_actifs=12, n_periodes=1200, graine=19)
    for modele in (SampleModel(), LedoitWolfModel(), FactorCovariance()):
        carte = validate_covariance_model(donnees, modele, lookback=252, step=21)
        assert carte.verdict in (VALIDE, SURVEILLANCE), f"{modele.name} : {carte}"
        assert carte.bias is not None
        assert carte.bias.bias.value == pytest.approx(1.0, abs=0.15)


def test_les_deux_protocoles_rendent_des_verdicts_opposes_si_mal_conditionnee():
    """
    LE RÉSULTAT CENTRAL DE LA SÉANCE 12, verrouillé.

    Même estimateur, mêmes données, même fenêtre. Portefeuille fixe :
    la covariance empirique passe, parce que wᵀΣw n'inverse pas la
    matrice. Portefeuille de variance minimale du modèle : elle
    s'effondre, parce que l'inversion amplifie le bruit.

    L'angle mort est une propriété du protocole, pas du modèle.
    """
    donnees = univers(n_actifs=100, n_facteurs=5, n_periodes=800, graine=20)

    fixe = validate_covariance_model(
        donnees, SampleModel(), lookback=252, step=21, portfolio="fixed"
    )
    propre = validate_covariance_model(
        donnees, SampleModel(), lookback=252, step=21, portfolio="model"
    )

    assert fixe.bias is not None and propre.bias is not None
    assert fixe.verdict == VALIDE
    assert propre.verdict == REJETE
    # Le protocole qui inverse voit une sous-estimation massive.
    assert propre.bias.bias.value > 1.5
    # Et le protocole fixe, lui, ne voit rien du tout.
    assert fixe.bias.bias.value == pytest.approx(1.0, abs=0.10)


def test_le_protocole_du_modele_classe_les_estimateurs_comme_les_seances_2_et_7():
    """
    Un troisième instrument qui retrouve le même ordre.

    Les séances 2 et 7 avaient classé ces estimateurs par le levier du
    portefeuille optimisé et par la volatilité réalisée hors
    échantillon. Ici c'est l'honnêteté de la prévision, une grandeur
    sans rapport — et l'ordre est le même.
    """
    donnees = univers(n_actifs=100, n_facteurs=5, n_periodes=800, graine=21)
    biais = {}
    for modele in (SampleModel(), LedoitWolfModel(), FactorCovariance()):
        carte = validate_covariance_model(
            donnees, modele, lookback=252, step=21, portfolio="model"
        )
        assert carte.bias is not None
        biais[modele.name] = carte.bias.bias.value

    assert biais["empirique"] > biais["Ledoit-Wolf"]
    assert biais["Ledoit-Wolf"] > biais["facteurs (Marchenko-Pastur)"]


def test_un_modele_qui_refuse_de_sestimer_est_consigne_inapplicable():
    """
    Un refus du modèle est un RÉSULTAT de validation, pas une panne.
    La fiche doit le dire, et le motif doit être lisible.
    """

    class ModeleQuiRefuse:
        name = "refuse toujours"

        def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
            raise InsufficientData(
                n_obs=len(returns), n_required=10**9, what="un test"
            )

    donnees = univers(graine=22)
    carte = validate_covariance_model(donnees, ModeleQuiRefuse(), lookback=252)
    assert carte.verdict == INAPPLICABLE
    assert carte.bias is None
    assert carte.n_forecasts == 0
    assert "InsufficientData" in carte.reason
    assert not carte.usable


def test_un_bug_du_modele_remonte_au_lieu_detre_deguise_en_inapplicable():
    """
    La contrepartie du test précédent, et la leçon de la séance 8 :
    attraper `Exception` aurait transformé toute faute de frappe en
    « modèle inapplicable ». Seules les trois exceptions qui signifient
    « je refuse » sont attrapées.
    """

    class ModeleBogue:
        name = "bogué"

        def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
            raise AttributeError("faute de frappe dans le modèle")

    donnees = univers(graine=23)
    with pytest.raises(AttributeError, match="faute de frappe"):
        validate_covariance_model(donnees, ModeleBogue(), lookback=252)


# ══════════════════════════════════════════════════════════════════════
#  CAUSALITÉ — la discipline imposée par la séance 8
# ══════════════════════════════════════════════════════════════════════


def test_aucune_estimation_ne_voit_sa_propre_date_ni_le_futur():
    """
    Test direct, par espionnage : on note l'index maximal que chaque
    estimation a pu lire, et on vérifie qu'il est STRICTEMENT antérieur
    à la date de décision.

    C'est la propriété que `history = returns.iloc[debut:t]` garantit,
    et la séance 8 a montré qu'une avance d'un seul jour suffit à
    fabriquer une performance imaginaire.
    """
    donnees = univers(n_actifs=8, n_periodes=600, graine=24)
    positions = {d: i for i, d in enumerate(donnees.index)}
    vus: list[tuple[int, int]] = []

    class Espion:
        name = "espion"

        def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
            vus.append((positions[returns.index[0]], positions[returns.index[-1]]))
            return sample_covariance(returns)

    lookback, step = 200, 25
    validate_covariance_model(donnees, Espion(), lookback=lookback, step=step)

    assert vus, "aucune estimation n'a eu lieu — le test ne prouverait rien"
    for rang, (premier, dernier) in enumerate(vus):
        decision = lookback + rang * step
        assert dernier == decision - 1, (
            f"l'estimation du rang {rang} a lu jusqu'à l'indice {dernier} "
            f"alors que sa date de décision est {decision} : fuite de "
            f"{dernier - decision + 1} période(s)"
        )
        assert dernier - premier + 1 == lookback


def test_perturber_le_futur_ne_change_pas_les_previsions_passees():
    """
    La même propriété, vue de l'extérieur : on corrompt violemment la
    fin de l'historique et on vérifie que les prévisions antérieures
    sont identiques au bit près.
    """
    donnees = univers(n_actifs=8, n_periodes=600, graine=25)
    sigmas: dict[str, list[float]] = {"propre": [], "corrompu": []}

    def espion(cle):
        class Espion:
            name = "espion"

            def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
                est = sample_covariance(returns)
                poids = np.full(est.n_assets, 1.0 / est.n_assets)
                sigmas[cle].append(math.sqrt(float(poids @ est.matrix @ poids)))
                return est

        return Espion()

    corrompu = donnees.copy()
    corrompu.iloc[400:] *= 50.0

    validate_covariance_model(donnees, espion("propre"), lookback=200, step=25)
    validate_covariance_model(corrompu, espion("corrompu"), lookback=200, step=25)

    # Les prévisions dont l'historique se termine avant l'indice 400
    # ne peuvent pas avoir changé.
    n_intactes = (400 - 200) // 25
    assert n_intactes >= 3
    for i in range(n_intactes):
        assert sigmas["corrompu"][i] == pytest.approx(sigmas["propre"][i], rel=0.0)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 4 — cohérence avec l'API des séances précédentes
# ══════════════════════════════════════════════════════════════════════


def test_le_biais_est_linverse_de_la_calibration_quand_la_prevision_est_constante():
    """
    `calibration()` est prédit/réalisé, le biais est réalisé/prédit.
    Les deux agrègent différemment — moyenne des prédictions d'un côté,
    moyenne quadratique des rapports de l'autre — et ne coïncident donc
    que si la prédiction ne varie pas. On vérifie le cas où elles
    doivent s'accorder.
    """
    r = rendements_gaussiens(2000, graine=26)
    s = np.full(r.size, VOL)
    biais = bias_statistic(r, s).bias.value
    realisee = math.sqrt(float(np.mean(r**2)))
    calibration_equivalente = VOL / realisee
    assert biais == pytest.approx(1.0 / calibration_equivalente, rel=1e-12)


def test_le_biais_et_la_calibration_divergent_quand_la_prevision_varie():
    """
    L'autre moitié de l'énoncé, et c'est elle qui compte : dès que la
    prévision bouge, les deux mesures diffèrent, et c'est la même
    inégalité de Jensen qui avait cassé `calibration_diagnosis` à la
    séance 9. On verrouille la divergence pour que personne ne
    « corrige » l'une vers l'autre.
    """
    g = np.random.default_rng(27)
    bloc = 50
    vols = np.repeat([VOL * 0.5, VOL * 2.0] * 20, bloc)
    r = g.normal(0.0, 1.0, vols.size) * vols

    biais = bias_statistic(r, vols, block_size=bloc).bias.value
    agrege = float(vols.mean()) / math.sqrt(float(np.mean(r**2)))

    # Le modèle a raison : il annonce exactement la bonne volatilité à
    # chaque période, et la statistique de biais le voit.
    assert biais == pytest.approx(1.0, abs=0.1)

    # L'agrégat, lui, se trompe — et l'écart n'est pas un hasard : il
    # vaut le rapport entre la moyenne quadratique et la moyenne
    # arithmétique des prévisions, c'est-à-dire exactement l'écart de
    # Jensen. On le CALCULE plutôt que de le deviner.
    jensen = math.sqrt(float(np.mean(vols**2))) / float(vols.mean())
    assert 1.0 / agrege == pytest.approx(jensen, rel=0.12)
    assert jensen > 1.10, "le dispositif de test doit créer un écart matériel"

    # Et la conclusion qui compte : la statistique de biais est le plus
    # fidèle des deux instruments sur ces données.
    assert abs(biais - 1.0) < abs(1.0 / agrege - 1.0)


def test_le_biais_saccorde_avec_un_backtest_de_variance_minimale():
    """
    Pont avec la séance 8 : `walk_forward` et
    `validate_covariance_model` suivent la même discipline causale sur
    la même stratégie. Leurs mesures de calibration ne sont pas
    identiques (l'une agrège, l'autre standardise) mais elles doivent
    pointer dans le même sens.
    """
    donnees = univers(n_actifs=30, n_facteurs=3, n_periodes=900, graine=28)

    resultat = walk_forward(
        donnees,
        MinimumVariance(estimator=sample_covariance),
        lookback=252,
        step=21,
        mode="rolling",
    )
    calibration = resultat.calibration()
    assert calibration is not None

    carte = validate_covariance_model(
        donnees, SampleModel(), lookback=252, step=21, portfolio="model"
    )
    assert carte.bias is not None

    # calibration = prédit/réalisé ; biais = réalisé/prédit.
    cote_calibration = calibration.value < 1.0
    cote_biais = carte.bias.bias.value > 1.0
    assert cote_calibration == cote_biais, (
        f"les deux instruments se contredisent : calibration "
        f"{calibration.value:.3f}, biais {carte.bias.bias.value:.3f}"
    )


# ══════════════════════════════════════════════════════════════════════
#  Typage structurel et inventaire
# ══════════════════════════════════════════════════════════════════════


def test_un_modele_ecrit_ailleurs_satisfait_le_protocol_sans_rien_importer():
    """
    Cinquième usage du typage structurel dans ce projet. La classe
    ci-dessous n'hérite de rien et n'importe rien de `validation` ;
    elle est pourtant un `CovarianceModel`.
    """

    class MonModele:
        @property
        def name(self) -> str:
            return "diagonale seule"

        def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
            est = sample_covariance(returns)
            diag = np.diag(np.diag(est.matrix))
            return CovarianceEstimate(
                matrix=diag,
                assets=est.assets,
                n_obs=est.n_obs,
                method="diagonale",
            )

    mien = MonModele()
    assert isinstance(mien, CovarianceModel)

    donnees = univers(graine=29)
    carte = validate_covariance_model(donnees, mien, lookback=252, step=21)
    assert carte.model == "diagonale seule"
    assert carte.bias is not None
    assert carte.verdict == REJETE  # ignorer les corrélations ne pardonne pas


def test_le_sens_du_biais_dune_diagonale_suit_le_signe_des_correlations():
    """
    J'AVAIS ÉCRIT CE TEST À L'ENVERS, et la mesure m'a corrigé.

    J'avais asserté qu'un modèle qui ignore les corrélations
    SOUS-ESTIME toujours le risque d'un portefeuille équipondéré.
    Faux : il sous-estime quand la corrélation moyenne hors diagonale
    est positive, et il SURESTIME quand elle est négative.

    Et l'effet est brutal, parce que dans un portefeuille la
    corrélation moyenne est multipliée par N−1 :

        variance du portefeuille ≈ (1/N)·moyenne(σ²)·[1 + (N−1)·ρ̄]

    Avec N = 15, une corrélation moyenne de −0,02 — indistinguable de
    zéro à l'œil — déplace la variance du portefeuille de près d'un
    tiers. C'est la même leçon que la séance 5 avait tirée de la borne
    ρ ≥ −1/(N−1) : les petites corrélations ne sont pas petites une
    fois agrégées.
    """

    class Diagonale:
        name = "diagonale seule"

        def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
            est = sample_covariance(returns)
            return CovarianceEstimate(
                matrix=np.diag(np.diag(est.matrix)),
                assets=est.assets,
                n_obs=est.n_obs,
                method="diagonale",
            )

    n_actifs = 15
    for positives, sens in ((True, "sous-estime"), (False, "surestime")):
        donnees = univers(
            n_actifs=n_actifs, n_facteurs=3, graine=29, loadings_positifs=positives
        )
        carte = validate_covariance_model(
            donnees, Diagonale(), lookback=252, step=21
        )
        assert carte.bias is not None

        correlation = donnees.corr().to_numpy()
        hors_diagonale = float(
            correlation[~np.eye(n_actifs, dtype=bool)].mean()
        )
        # Le biais attendu se CALCULE depuis la corrélation réalisée.
        #   variance vraie   ≈ (1/N)·moyenne(σ²)·[1 + (N−1)·ρ̄]
        #   variance prédite  = (1/N)·moyenne(σ²)        (diagonale seule)
        #   biais = réalisé/prédit = √(1 + (N−1)·ρ̄)
        # La tolérance est large parce que ρ̄ exact est une moyenne
        # pondérée par les variances, et non la moyenne simple.
        attendu = math.sqrt(1.0 + (n_actifs - 1) * hors_diagonale)

        if positives:
            assert hors_diagonale > 0.1
            assert carte.bias.bias.value > 1.0, sens
        else:
            assert hors_diagonale < 0.0
            assert carte.bias.bias.value < 1.0, sens

        assert carte.bias.bias.value == pytest.approx(attendu, rel=0.25), (
            f"le mécanisme n'explique pas la mesure : attendu ≈ {attendu:.3f}, "
            f"mesuré {carte.bias.bias.value:.3f}"
        )


def test_les_quatre_adaptateurs_du_moteur_portent_un_nom_et_estiment():
    donnees = univers(n_actifs=10, n_periodes=400, graine=30)
    for modele in (
        SampleModel(),
        LedoitWolfModel(),
        EwmaModel(),
        EwmaModel(decay=0.97),
        FactorCovariance(),
        FactorCovariance(n_factors=2),
    ):
        assert isinstance(modele, CovarianceModel)
        assert isinstance(modele.name, str) and modele.name
        estimation = modele.estimate(donnees)
        assert estimation.n_assets == donnees.shape[1]


def test_linventaire_agrege_les_fiches_et_se_laisse_lire():
    donnees = univers(n_actifs=15, n_periodes=900, graine=31)
    inv = build_inventory(donnees, universe="test")
    assert isinstance(inv, ModelInventory)
    assert len(inv.cards) == 4

    table = inv.table()
    assert len(table) == 4
    for colonne in ("modèle", "biais", "verdict", "dispersion"):
        assert colonne in table.columns

    texte = str(inv)
    assert "INVENTAIRE DE MODÈLES" in texte
    for carte in inv.cards:
        assert carte.model in texte

    meilleur = inv.best()
    if meilleur is not None:
        ecarts = [abs(c.bias_value - 1.0) for c in inv.usable() if c.bias is not None]
        assert abs(meilleur.bias_value - 1.0) == pytest.approx(min(ecarts))


def test_une_fiche_declare_toujours_ses_angles_morts():
    donnees = univers(graine=32)
    fixe = validate_covariance_model(donnees, SampleModel(), lookback=252)
    propre = validate_covariance_model(
        donnees, SampleModel(), lookback=252, portfolio="model"
    )
    assert any("FIXE" in limite for limite in fixe.limits)
    assert any("variance minimale" in limite for limite in propre.limits)
    # Et la fiche imprimée les montre, sinon elles ne servent à rien.
    assert "⚠" in str(fixe)


@pytest.mark.parametrize(
    ("kwargs", "motif"),
    [
        ({"lookback": 1}, "lookback"),
        ({"step": 0}, "step"),
        ({"portfolio": "autre"}, "portfolio"),
    ],
)
def test_le_protocole_refuse_les_parametres_absurdes(kwargs, motif):
    donnees = univers(n_actifs=5, n_periodes=400, graine=33)
    with pytest.raises(ValueError, match=motif):
        validate_covariance_model(donnees, SampleModel(), **kwargs)


def test_des_poids_avec_un_portefeuille_calcule_est_contradictoire():
    donnees = univers(n_actifs=5, n_periodes=400, graine=34)
    poids = pd.Series(0.2, index=donnees.columns)
    with pytest.raises(ValueError, match="contradictoire"):
        validate_covariance_model(
            donnees, SampleModel(), weights=poids, portfolio="model"
        )


def test_des_poids_incomplets_sont_refuses_avec_les_noms_manquants():
    donnees = univers(n_actifs=5, n_periodes=400, graine=35)
    poids = pd.Series(0.5, index=donnees.columns[:2])
    with pytest.raises(ValueError, match="poids manquants"):
        validate_covariance_model(donnees, SampleModel(), weights=poids)


def test_un_univers_dun_seul_actif_est_refuse():
    donnees = univers(n_actifs=5, n_periodes=400, graine=36).iloc[:, :1]
    with pytest.raises(ValueError, match="au moins deux actifs"):
        validate_covariance_model(donnees, SampleModel())


def test_un_historique_plus_court_que_le_lookback_leve_insufficient_data():
    donnees = univers(n_actifs=5, n_periodes=100, graine=37)
    with pytest.raises(InsufficientData):
        validate_covariance_model(donnees, SampleModel(), lookback=252)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — couverture Monte Carlo de l'INTERVALLE
# ══════════════════════════════════════════════════════════════════════

def _prevision_bloc(h: np.ndarray, n: int, bloc: int) -> np.ndarray:
    """Chemin de prévisions GARCH causales, bloc par bloc."""
    s = np.empty(n)
    for d in range(0, n, bloc):
        f = min(d + bloc, n)
        s[d:f] = prevision_garch_causale(h, d, f - d)
    return s


@pytest.mark.slow
def test_loracle_garch_est_un_vrai_oracle():
    """
    LA VÉRIFICATION QUI DOIT PRÉCÉDER LES DEUX TESTS SUIVANTS.

    Un oracle correct doit donner E[m2] = 1. La séance 12 a commencé
    par un oracle qui valait 0,936 parce qu'il regardait le futur sans
    que ça se voie : dans un GARCH, h_{t+1} dépend de e_t, donc la
    variance moyenne d'un bloc est en partie causée par les rendements
    qu'elle sert à normaliser.

    Sans ce test, les deux suivants mesureraient le défaut du harnais
    en croyant mesurer celui de l'instrument. C'est arrivé quatre fois
    dans ce projet.
    """
    n, bloc, n_sim = 1008, 21, 300
    total = 0.0
    for i in range(n_sim):
        e, h = trajectoire_garch(3000 + i, n)
        test = bias_statistic(e, _prevision_bloc(h, n, bloc), block_size=bloc)
        total += test.bias.value**2
    assert total / n_sim == pytest.approx(1.0, abs=0.02)


@pytest.mark.slow
def test_lintervalle_par_blocs_couvre_son_niveau_nominal_sous_h0():
    """
    Le test de couverture : mille échantillons, mille intervalles,
    combien contiennent la vraie valeur ? Ici le modèle a raison par
    construction, donc tout rejet est un faux positif.
    """
    n, n_sim = 1008, 1000
    dedans = 0
    for i in range(n_sim):
        r = np.random.default_rng(1000 + i).normal(0.0, 1.0, n)
        test = bias_statistic(r, np.ones(n), block_size=1, confidence=0.95)
        if test.bias.ci_low <= 1.0 <= test.bias.ci_high:
            dedans += 1
    # ±3 points autour de 95 % : l'erreur d'échantillonnage sur 1000
    # tirages est d'environ 0,7 point, donc la marge est large.
    assert 0.92 <= dedans / n_sim <= 0.98


@pytest.mark.slow
def test_la_formule_classique_sous_couvre_sur_un_marche_realiste():
    """
    LE RÉSULTAT MESURÉ DE LA SÉANCE 12, verrouillé comme inégalité et
    non comme valeur ponctuelle.

    Sous volatilité persistante ET queues épaisses — c'est-à-dire sur
    n'importe quel marché — l'intervalle classique à 95 % couvre très
    en dessous de 95 %, et l'intervalle par blocs fait nettement mieux.

    On n'assère pas les valeurs exactes : elles dépendent des
    paramètres du générateur, et une égalité numérique écrite à la main
    serait la treizième dérive de narration de ce projet.
    """
    n, bloc, n_sim, nu = 1008, 21, 400, 5
    dans_robuste = dans_naif = 0
    for i in range(n_sim):
        e, h = trajectoire_garch(4000 + i, n, nu=nu)
        s = _prevision_bloc(h, n, bloc)
        test = bias_statistic(e, s, block_size=bloc, confidence=0.95)
        if test.bias.ci_low <= 1.0 <= test.bias.ci_high:
            dans_robuste += 1
        if test.naive_ci[0] <= 1.0 <= test.naive_ci[1]:
            dans_naif += 1

    couverture_naive = dans_naif / n_sim
    couverture_robuste = dans_robuste / n_sim

    # La formule classique échoue franchement.
    assert couverture_naive < 0.70
    # L'intervalle par blocs fait nettement mieux, sans atteindre 95 % :
    # il n'y a que n/bloc blocs, et le moment d'ordre quatre d'une série
    # à queues épaisses est mal estimé. Progrès mesuré, pas solution.
    assert couverture_robuste > couverture_naive + 0.20
    assert couverture_robuste < 0.95


@pytest.mark.slow
def test_lecart_type_reel_du_biais_suit_la_formule_du_kurtosis():
    """
    La raison théorique de l'échec précédent, vérifiée séparément.

    Pour des z indépendants, écart-type(b) = √((κ−1)/T)/2. Avec une loi
    de Student à ν degrés de liberté, κ = 3 + 6/(ν−4), donc le rapport
    à la formule gaussienne est √((κ−1)/2).
    """
    n, n_sim, nu = 1008, 1200, 5
    valeurs = np.empty(n_sim)
    for i in range(n_sim):
        g = np.random.default_rng(2000 + i)
        x = g.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
        valeurs[i] = bias_statistic(x, np.ones(n), block_size=1).bias.value

    kurt = 3.0 + 6.0 / (nu - 4)
    attendu = math.sqrt((kurt - 1.0) / 2.0)
    observe = float(valeurs.std(ddof=1)) / (1.0 / math.sqrt(2.0 * n))
    assert observe == pytest.approx(attendu, rel=0.15)
