"""
Tests du moteur de backtest — séance 8.

LE TEST QUI COMPTE EST CELUI DU DÉTECTEUR
=========================================

Un harnais de vérification qui n'attrape pas un bug qu'on y met exprès
ne prouve rien. On écrit donc ici des stratégies DÉLIBÉRÉMENT
TRICHEUSES, de trois sortes différentes, et on exige que l'audit les
démasque — tout en exigeant qu'il laisse tranquilles les stratégies
honnêtes.

C'est de la mutation testing appliquée au temps : on introduit la faute
qu'on prétend savoir détecter, et on mesure si on la détecte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nkrisk.backtest import (
    BacktestResult,
    DeflatedSharpe,
    EqualWeight,
    LookaheadAudit,
    MinimumVariance,
    PredictsRisk,
    Strategy,
    deflated_sharpe_ratio,
    detect_lookahead,
    expected_maximum_sharpe,
    minimum_track_record_length,
    walk_forward,
)
from nkrisk.covariance import ledoit_wolf, sample_covariance
from nkrisk.risk import minimum_variance_weights
from nkrisk.types import InsufficientData

JOURS = 252


# ══════════════════════════════════════════════════════════════════════
#  DONNÉES
# ══════════════════════════════════════════════════════════════════════


def _marche(
    n: int = 12, t: int = 900, k: int = 3, seed: int = 0, derive: float = 0.0
) -> pd.DataFrame:
    """Rendements à structure factorielle, avec dérive optionnelle."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 0.7, (n, k))
    b[:, 0] = np.abs(rng.normal(1.0, 0.3, n))
    f = rng.normal(0, 1, (t, k))
    e = rng.normal(0, 1, (t, n)) * rng.uniform(0.5, 1.8, n)
    vols = rng.uniform(0.008, 0.030, n)
    x = (f @ b.T + e) * vols + derive / JOURS
    return pd.DataFrame(
        x,
        index=pd.bdate_range("2015-01-01", periods=t),
        columns=[f"A{i:02d}" for i in range(n)],
    )


# ══════════════════════════════════════════════════════════════════════
#  STRATÉGIES TRICHEUSES — écrites pour être démasquées
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AjusteeSurToutLechantillon:
    """
    LA FUITE LA PLUS COURANTE DU MONDE RÉEL.

    La covariance est estimée UNE FOIS sur l'intégralité des données,
    puis « évaluée » période par période. C'est ce que produit tout
    enchaînement du type `scaler.fit(donnees_completes)` suivi d'une
    boucle d'évaluation.

    Remarque importante : cette stratégie respecte scrupuleusement la
    signature du contrat. Elle reçoit `history` et pourrait n'utiliser
    qu'elle. Aucun examen de type ne la distingue d'une stratégie
    honnête. Seule l'expérience la démasque.
    """

    donnees_completes: pd.DataFrame
    name: str = "ajustée sur tout l'échantillon"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        return minimum_variance_weights(ledoit_wolf(self.donnees_completes))


@dataclass(frozen=True)
class RegardeCinqJoursPlusLoin:
    """
    Fuite DISCRÈTE : la stratégie n'utilise que l'historique… plus cinq
    jours.

    C'est la version réaliste du décalage mal posé — un `shift(-5)` au
    lieu de `shift(5)`, une date de décision confondue avec une date
    d'exécution. Cinq jours sur neuf cents, soit un demi pour cent des
    données. Invisible à la lecture d'une courbe.
    """

    donnees_completes: pd.DataFrame
    avance: int = 5
    name: str = "regarde cinq jours plus loin"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        # Position réelle de la décision dans l'historique complet.
        # Écrire `len(history) + avance` serait faux en fenêtre
        # glissante, où len(history) est constant.
        ici = self.donnees_completes.index.get_loc(history.index[-1]) + 1
        fin = min(ici + self.avance, len(self.donnees_completes))
        return minimum_variance_weights(
            ledoit_wolf(self.donnees_completes.iloc[:fin])
        )


@dataclass(frozen=True)
class NormaliseeSurToutLechantillon:
    """
    Fuite SUBTILE : les poids ne viennent que de l'historique, mais ils
    sont normalisés par des écarts-types calculés sur tout l'échantillon.

    La décision « quoi acheter » est honnête ; la décision « combien »
    ne l'est pas. C'est la fuite qu'on trouve dans les chaînes de
    prétraitement, et celle qu'une relecture de code manque le plus
    souvent, parce que le coupable n'est pas dans la fonction qu'on
    relit.
    """

    donnees_completes: pd.DataFrame
    name: str = "normalisée sur tout l'échantillon"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        inverse_vol = 1.0 / self.donnees_completes.std(ddof=1)
        brut = inverse_vol.reindex(history.columns)
        return brut / brut.sum()


@dataclass(frozen=True)
class VolatiliteInverse:
    """Contrôle honnête : même idée, mais calculée sur l'historique seul."""

    name: str = "volatilité inverse"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        brut = 1.0 / history.std(ddof=1)
        return brut / brut.sum()


@dataclass(frozen=True)
class Espionne:
    """Enregistre le dernier indice vu, pour vérifier la borne du moteur."""

    vus: list = None  # type: ignore[assignment]
    name: str = "espionne"

    def __post_init__(self) -> None:
        if self.vus is None:
            object.__setattr__(self, "vus", [])

    def weights(self, history: pd.DataFrame) -> pd.Series:
        self.vus.append(history.index[-1])
        n = history.shape[1]
        return pd.Series(np.full(n, 1.0 / n), index=history.columns)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — LA CAUSALITÉ DU MOTEUR
# ══════════════════════════════════════════════════════════════════════


def test_la_decision_nutilise_aucune_donnee_de_la_periode_evaluee() -> None:
    """
    LA BORNE QUI DÉCIDE DE TOUT.

    La stratégie espionne enregistre la dernière date qu'elle a vue à
    chaque décision. Cette date doit être STRICTEMENT antérieure à la
    première période sur laquelle les poids sont appliqués.

    `iloc[debut:t]` au lieu de `iloc[debut:t+1]` : une unité, et c'est
    toute la différence entre un backtest et une illusion. Ce test est
    le seul endroit du projet où cette unité est vérifiée directement.
    """
    donnees = _marche(t=400)
    espionne = Espionne()
    lookback, step = 100, 20

    walk_forward(donnees, espionne, lookback=lookback, step=step)

    positions_decision = range(lookback, len(donnees), step)
    for vue, t in zip(espionne.vus, positions_decision, strict=True):
        assert vue == donnees.index[t - 1], (
            f"à la décision en position {t}, la stratégie a vu jusqu'à "
            f"{vue}, alors qu'elle ne devait pas dépasser "
            f"{donnees.index[t - 1]}"
        )
        assert vue < donnees.index[t]


def test_la_fenetre_glissante_a_bien_la_longueur_demandee() -> None:
    donnees = _marche(t=500)
    tailles: list[int] = []

    @dataclass(frozen=True)
    class Mesure:
        name: str = "mesure"

        def weights(self, history: pd.DataFrame) -> pd.Series:
            tailles.append(len(history))
            n = history.shape[1]
            return pd.Series(np.full(n, 1.0 / n), index=history.columns)

    walk_forward(donnees, Mesure(), lookback=120, step=30)
    assert set(tailles) == {120}


def test_la_fenetre_croissante_grandit() -> None:
    donnees = _marche(t=500)
    tailles: list[int] = []

    @dataclass(frozen=True)
    class Mesure:
        name: str = "mesure"

        def weights(self, history: pd.DataFrame) -> pd.Series:
            tailles.append(len(history))
            n = history.shape[1]
            return pd.Series(np.full(n, 1.0 / n), index=history.columns)

    walk_forward(donnees, Mesure(), lookback=120, step=30, mode="expanding")
    assert tailles == sorted(tailles)
    assert tailles[0] == 120
    assert tailles[-1] > tailles[0]


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — LE DÉTECTEUR, SUR DES FUITES CONNUES
# ══════════════════════════════════════════════════════════════════════


def test_une_strategie_honnete_nest_pas_accusee() -> None:
    """
    Le contrôle négatif, et il passe en premier.

    Un détecteur qui crie au loup sur du code correct est pire
    qu'inutile : on apprend à l'ignorer, et il ne sert plus à rien le
    jour où il a raison.
    """
    donnees = _marche(t=700)
    for fabrique, _nom in (
        (lambda _d: EqualWeight(), "équipondéré"),
        (lambda _d: VolatiliteInverse(), "volatilité inverse"),
        (lambda _d: MinimumVariance(ledoit_wolf, "var. min."), "var. min."),
    ):
        audit = detect_lookahead(donnees, fabrique, lookback=200, step=25)
        assert audit.clean, str(audit)
        assert audit.worst_difference == 0.0


def test_le_detecteur_attrape_une_strategie_ajustee_sur_tout() -> None:
    """La fuite la plus courante : un modèle calibré avant le découpage."""
    donnees = _marche(t=700)
    audit = detect_lookahead(
        donnees, AjusteeSurToutLechantillon, lookback=200, step=25
    )
    assert not audit.clean
    assert audit.worst_difference > 1e-6
    assert "FUITE DÉTECTÉE" in str(audit)


def test_le_detecteur_attrape_une_avance_de_cinq_jours() -> None:
    """
    Cinq jours d'avance sur sept cents, soit 0,7 % des données.

    C'est la fuite qu'aucune relecture ne trouve et qu'aucune courbe ne
    trahit. Le test de causalité, lui, ne la rate pas : il ne compare
    pas des performances, il compare des décisions.
    """
    donnees = _marche(t=700)
    audit = detect_lookahead(
        donnees,
        lambda d: RegardeCinqJoursPlusLoin(d, avance=5),
        lookback=200,
        step=25,
    )
    assert not audit.clean


def test_le_detecteur_attrape_une_avance_dun_seul_jour() -> None:
    """
    Un seul jour. La limite de sensibilité du détecteur.

    Un jour sur sept cents, c'est 0,14 % des données — et ça suffit à
    transformer un backtest en fiction. Le détecteur doit le voir, sans
    quoi il laisserait passer exactement le décalage d'indice le plus
    fréquent.
    """
    donnees = _marche(t=700)
    audit = detect_lookahead(
        donnees,
        lambda d: RegardeCinqJoursPlusLoin(d, avance=1),
        lookback=200,
        step=25,
    )
    assert not audit.clean


def test_le_detecteur_attrape_une_normalisation_fuyante() -> None:
    """
    La fuite subtile : le choix des actifs est honnête, leur
    pondération ne l'est pas.

    Le contrôle honnête correspondant — même calcul sur l'historique
    seul — doit passer. Les deux assertions ensemble montrent que le
    détecteur distingue la fuite de la méthode, et pas seulement une
    classe d'une autre.
    """
    donnees = _marche(t=700)

    sale = detect_lookahead(
        donnees, NormaliseeSurToutLechantillon, lookback=200, step=25
    )
    propre = detect_lookahead(
        donnees, lambda _d: VolatiliteInverse(), lookback=200, step=25
    )

    assert not sale.clean
    assert propre.clean


def test_laudit_nomme_la_premiere_coupure_fautive() -> None:
    """Un détecteur qui dit « il y a un problème » sans dire où est inutile."""
    donnees = _marche(t=700)
    audit = detect_lookahead(
        donnees, AjusteeSurToutLechantillon, lookback=200, step=25
    )
    fuites = [p for p in audit.probes if p.leaked]
    assert fuites
    assert all(p.n_decisions_before > 0 for p in fuites)
    assert isinstance(audit, LookaheadAudit)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — LA COMPTABILITÉ DU MOTEUR
# ══════════════════════════════════════════════════════════════════════


def test_sans_couts_le_net_egale_le_brut() -> None:
    donnees = _marche(t=500)
    r = walk_forward(donnees, EqualWeight(), lookback=150, step=20, cost_bps=0)
    assert np.allclose(r.returns.to_numpy(), r.gross_returns.to_numpy())
    assert r.total_cost() == 0.0


def test_les_couts_se_deduisent_exactement() -> None:
    donnees = _marche(t=500)
    r = walk_forward(donnees, EqualWeight(), lookback=150, step=20, cost_bps=25)
    ecart = r.gross_returns - r.returns
    assert np.allclose(ecart.to_numpy(), r.costs.to_numpy())
    assert r.total_cost() > 0


def test_les_couts_degradent_la_performance() -> None:
    donnees = _marche(t=800)
    sans = walk_forward(donnees, MinimumVariance(ledoit_wolf), lookback=250,
                        step=5, cost_bps=0)
    avec = walk_forward(donnees, MinimumVariance(ledoit_wolf), lookback=250,
                        step=5, cost_bps=50)
    assert avec.equity.iloc[-1] < sans.equity.iloc[-1]


def test_les_poids_cibles_somment_a_un() -> None:
    donnees = _marche(t=500)
    r = walk_forward(donnees, EqualWeight(), lookback=150, step=20)
    assert np.allclose(r.weights.sum(axis=1).to_numpy(), 1.0)


def test_lequipondere_a_pas_quotidien_reproduit_la_moyenne_des_actifs() -> None:
    """
    Vérification indépendante de l'arithmétique du moteur.

    À pas de 1, l'équipondéré est rebalancé chaque jour : son rendement
    doit être exactement la moyenne des rendements des actifs, sans
    dérive possible. Si cette identité élémentaire échoue, toute la
    comptabilité du moteur est à revoir.
    """
    donnees = _marche(t=400)
    r = walk_forward(donnees, EqualWeight(), lookback=100, step=1)
    attendu = donnees.iloc[100:].mean(axis=1)
    assert np.allclose(r.returns.to_numpy(), attendu.to_numpy())


def test_le_turnover_ignore_la_mise_en_place_initiale() -> None:
    """
    Acheter le portefeuille au premier jour n'est pas de la ROTATION.
    Compter ce 1,0 initial gonflerait le turnover annualisé de toute
    stratégie évaluée sur un historique court.
    """
    donnees = _marche(t=500)
    r = walk_forward(donnees, EqualWeight(), lookback=150, step=20)
    assert r.turnover.iloc[0] == 0.0
    assert r.turnover.iloc[1:].sum() > 0


def test_la_derive_des_poids_reduit_le_turnover_mesure() -> None:
    """
    Entre deux rebalancements, les poids suivent les prix. Le turnover
    se mesure donc contre les poids DÉRIVÉS.

    Pour un équipondéré, la dérive est faible mais non nulle : le
    turnover mesuré doit être strictement positif, et très inférieur au
    2,0 que donnerait une vente-rachat complète à chaque fois.
    """
    donnees = _marche(t=800)
    r = walk_forward(donnees, EqualWeight(), lookback=250, step=21)
    rotations = r.turnover.iloc[1:]
    assert (rotations > 0).all()
    assert rotations.max() < 0.5


def test_la_capacite_optionnelle_est_detectee_a_lexecution() -> None:
    """
    `PredictsRisk` est facultatif. Le moteur le découvre par isinstance
    et n'enregistre une prédiction que si elle existe.
    """
    donnees = _marche(t=600)

    assert isinstance(EqualWeight(), Strategy)
    assert not isinstance(EqualWeight(), PredictsRisk)
    assert isinstance(MinimumVariance(ledoit_wolf), PredictsRisk)

    sans = walk_forward(donnees, EqualWeight(), lookback=200, step=20)
    avec = walk_forward(donnees, MinimumVariance(ledoit_wolf), lookback=200,
                        step=20)

    assert sans.predicted_volatility is None
    assert sans.calibration() is None
    assert avec.predicted_volatility is not None
    assert len(avec.predicted_volatility) == avec.n_rebalances
    assert avec.calibration() is not None


def test_la_calibration_encadre_sa_valeur() -> None:
    """L'Estimate retourné doit respecter son propre invariant."""
    donnees = _marche(t=900)
    r = walk_forward(donnees, MinimumVariance(ledoit_wolf), lookback=250,
                     step=21)
    cal = r.calibration()
    assert cal is not None
    assert cal.ci_low <= cal.value <= cal.ci_high
    assert 0.5 < cal.value < 2.0


def test_le_resume_sexecute_sur_les_deux_familles() -> None:
    donnees = _marche(t=800)
    for strat in (EqualWeight(), MinimumVariance(sample_covariance)):
        r = walk_forward(donnees, strat, lookback=250, step=21, cost_bps=10)
        texte = r.summary()
        assert "volatilité réalisée" in texte
        assert "turnover annuel" in texte
        assert isinstance(r, BacktestResult)
        assert len(r.equity) == len(r.returns)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 4 — LE BIAIS DE SÉLECTION
# ══════════════════════════════════════════════════════════════════════


def test_le_sharpe_maximal_attendu_croit_avec_le_nombre_dessais() -> None:
    """
    Avec un seul essai, rien à corriger. Avec mille, le meilleur résultat
    d'un lot de stratégies TOUTES SANS VALEUR est déjà élevé.
    """
    dispersion = 0.05
    valeurs = [expected_maximum_sharpe(n, dispersion)
               for n in (1, 2, 10, 100, 1000)]
    assert valeurs[0] == 0.0
    assert valeurs == sorted(valeurs)
    assert valeurs[-1] > valeurs[1]


def test_le_sharpe_degonfle_punit_les_essais_multiples() -> None:
    """
    Mêmes rendements, même Sharpe observé. Seul change le nombre de fois
    où on a regardé — et la conclusion s'inverse.

    C'est tout le propos : un backtest ne mesure pas la qualité d'une
    stratégie, il mesure le nombre d'essais.
    """
    rng = np.random.default_rng(3)
    # Sharpe annualisé d'environ 1,5 sur quatre ans.
    r = pd.Series(rng.normal(1.5 / JOURS * 0.15, 0.15 / np.sqrt(JOURS), 1000))

    un = deflated_sharpe_ratio(r, n_trials=1)
    beaucoup = deflated_sharpe_ratio(r, n_trials=5_000)

    assert un.observed == pytest.approx(beaucoup.observed)
    assert un.expected_max_under_null == 0.0
    assert beaucoup.expected_max_under_null > 0.0
    assert beaucoup.probability < un.probability
    assert isinstance(un, DeflatedSharpe)


def test_du_bruit_pur_ne_survit_pas_a_la_correction() -> None:
    """Des rendements sans aucune dérive ne doivent convaincre personne."""
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0, 0.01, 1500))
    assert not deflated_sharpe_ratio(r, n_trials=50).survives


def test_une_vraie_dérive_survit_quand_les_essais_sont_peu_nombreux() -> None:
    """Le détecteur ne doit pas tout rejeter : contrôle positif."""
    rng = np.random.default_rng(5)
    # Sharpe annualisé d'environ 2 sur huit ans, un seul essai.
    r = pd.Series(rng.normal(2.0 * 0.01 / np.sqrt(JOURS), 0.01, 2000))
    assert deflated_sharpe_ratio(r, n_trials=1).survives


def test_la_longueur_minimale_dhistorique_decroit_avec_le_sharpe() -> None:
    """
    Plus le Sharpe est élevé, moins il faut d'historique pour le
    distinguer de zéro. C'est la réponse chiffrée à « ma stratégie fait
    1,5 depuis six mois ».
    """
    rng = np.random.default_rng(6)
    longueurs = []
    for sr_cible in (0.5, 1.0, 2.0):
        r = pd.Series(rng.normal(sr_cible * 0.01 / np.sqrt(JOURS), 0.01, 3000))
        longueurs.append(minimum_track_record_length(r))
    assert longueurs == sorted(longueurs, reverse=True)


def test_un_sharpe_negatif_demande_un_historique_infini() -> None:
    """On ne prouvera jamais qu'une stratégie perdante est gagnante."""
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(-0.0005, 0.01, 1000))
    assert minimum_track_record_length(r) == float("inf")


def test_le_sharpe_degonfle_refuse_un_historique_trop_court() -> None:
    rng = np.random.default_rng(8)
    with pytest.raises(InsufficientData):
        deflated_sharpe_ratio(pd.Series(rng.normal(0, 0.01, 30)), n_trials=1)


# ══════════════════════════════════════════════════════════════════════
#  REFUS EXPLICITES
# ══════════════════════════════════════════════════════════════════════


def test_le_moteur_refuse_un_historique_trop_court() -> None:
    donnees = _marche(t=100)
    with pytest.raises(InsufficientData):
        walk_forward(donnees, EqualWeight(), lookback=200, step=20)


def test_le_moteur_refuse_les_parametres_absurdes() -> None:
    donnees = _marche(t=500)
    with pytest.raises(ValueError, match="mode inconnu"):
        walk_forward(donnees, EqualWeight(), lookback=100, mode="magique")
    with pytest.raises(ValueError, match="lookback"):
        walk_forward(donnees, EqualWeight(), lookback=1)
    with pytest.raises(ValueError, match="step"):
        walk_forward(donnees, EqualWeight(), lookback=100, step=0)
    with pytest.raises(ValueError, match="cost_bps"):
        walk_forward(donnees, EqualWeight(), lookback=100, cost_bps=-1)


def test_le_moteur_refuse_les_valeurs_manquantes() -> None:
    donnees = _marche(t=500).copy()
    donnees.iloc[10, 2] = np.nan
    with pytest.raises(ValueError, match="manquantes"):
        walk_forward(donnees, EqualWeight(), lookback=100)


def test_le_moteur_refuse_une_strategie_aux_poids_incomplets() -> None:
    @dataclass(frozen=True)
    class Incomplete:
        name: str = "incomplète"

        def weights(self, history: pd.DataFrame) -> pd.Series:
            colonnes = list(history.columns)[:-1]
            return pd.Series(
                np.full(len(colonnes), 1.0 / len(colonnes)), index=colonnes
            )

    donnees = _marche(t=400)
    with pytest.raises(ValueError, match="n'a pas donné de poids"):
        walk_forward(donnees, Incomplete(), lookback=100, step=20)


def test_lestimateur_qui_echoue_arrete_le_backtest() -> None:
    """
    Pas de repli silencieux. Un moteur qui se rabattrait sur autre chose
    produirait une courbe ne correspondant à aucune stratégie
    exécutable.

    ⚠️ PREMIÈRE VERSION : `pytest.raises(Exception)`, avec un
    `# noqa: B017` pour faire taire ruff qui le signalait.

    Un `raises(Exception)` aveugle passe sur n'importe quoi — un
    ImportError, une faute de frappe dans un nom d'attribut, une
    signature changée. Le test serait resté vert en ne prouvant plus
    rien, et c'est le pire état possible pour un test.

    Le commentaire de `test_le_portefeuille_est_immuable`, écrit à la
    séance 5, dit exactement cela. Je l'ai écrit, puis violé trois
    séances plus tard — et j'ai fait taire l'outil qui me le signalait
    au lieu de l'écouter.

    **Un avertissement de linter qu'on supprime est un avertissement
    qu'on a lu et rejeté.** Celui-là avait raison.
    """
    # 30 actifs, fenêtre de 20 : la covariance empirique est singulière.
    donnees = _marche(n=30, t=200)
    with pytest.raises(ValueError, match="singulière"):
        walk_forward(donnees, MinimumVariance(sample_covariance), lookback=20,
                     step=10)


# ══════════════════════════════════════════════════════════════════════
#  POURQUOI LE DÉTECTEUR EST SENSIBLE : LE PLACEMENT DES SONDES
# ══════════════════════════════════════════════════════════════════════


def test_une_sonde_mal_placee_laisse_passer_la_fuite() -> None:
    """
    LA LEÇON DE CONCEPTION DE LA SÉANCE, ET ELLE EST CONTRE-INTUITIVE.

    Même détecteur, même fuite, même données. Seul change l'endroit où
    l'on coupe — et le verdict s'inverse.

    Une décision prise en position t avec une avance de `a` périodes lit
    jusqu'à l'indice t+a−1. Elle n'est démasquée que si la corruption
    commence à un indice qu'elle atteint, donc si cut ≤ t+a−1. Avec des
    coupures posées entre les dates de décision, la fenêtre utile fait
    `a` périodes sur `step` : 4 % de chances pour une avance d'un jour
    et un pas de 25.

    En posant la coupure SUR la date de décision, l'inégalité devient
    cut = t ≤ t+a−1, vraie dès que a ≥ 1.

    **La sensibilité d'un détecteur ne vient pas de sa finesse, mais de
    l'endroit où on le pose.** C'est vrai de la journalisation, des
    assertions, des points d'arrêt et de la métrologie en général ; ici
    c'est mesurable.
    """
    donnees = _marche(t=700)
    lookback, step = 200, 25
    def fabrique(donnees_vues: pd.DataFrame) -> Strategy:
        return RegardeCinqJoursPlusLoin(donnees_vues, avance=1)

    decisions = list(range(lookback, len(donnees), step))

    # Sondes posées à mi-chemin entre deux décisions : aveugles.
    entre_deux = [t + step // 2 for t in decisions[:-1]]
    aveugle = detect_lookahead(
        donnees, fabrique, lookback=lookback, step=step, cuts=entre_deux
    )

    # Sondes posées sur les décisions : c'est le défaut du module.
    alignee = detect_lookahead(
        donnees, fabrique, lookback=lookback, step=step, cuts=decisions
    )

    assert aveugle.clean, (
        "ce test documente un angle mort : si des sondes mal placées se "
        "mettent à voir la fuite, l'analyse ci-dessus est à refaire"
    )
    assert not alignee.clean


@pytest.mark.slow
def test_taux_de_detection_dune_avance_dun_jour() -> None:
    """
    Combien de fois sur vingt univers le détecteur voit-il une avance
    d'une seule période ?

    On mesure au lieu d'affirmer, comme pour le taux de faux positifs
    de Marchenko-Pastur à la séance 7. Un détecteur dont on ne connaît
    pas le taux de détection n'est pas un instrument, c'est une
    impression.
    """
    vus = 0
    essais = 20
    for seed in range(essais):
        donnees = _marche(t=600, seed=100 + seed)
        audit = detect_lookahead(
            donnees,
            lambda d: RegardeCinqJoursPlusLoin(d, avance=1),
            lookback=200,
            step=25,
        )
        vus += not audit.clean

    assert vus == essais, (
        f"avance d'un jour détectée {vus}/{essais} fois — le placement des "
        f"sondes sur les dates de décision devrait donner 100 %"
    )


@pytest.mark.slow
def test_aucun_faux_positif_sur_vingt_univers() -> None:
    """Le contrôle négatif, mesuré lui aussi."""
    faux = 0
    essais = 20
    for seed in range(essais):
        donnees = _marche(t=600, seed=200 + seed)
        audit = detect_lookahead(
            donnees,
            lambda _d: MinimumVariance(ledoit_wolf, "var. min."),
            lookback=200,
            step=25,
        )
        faux += not audit.clean

    assert faux == 0, f"{faux}/{essais} fausses alertes"


# ══════════════════════════════════════════════════════════════════════
#  LA CALIBRATION COMME SÉRIE, ET SON DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════


def _marche_groupe(
    n: int = 10, t: int = 2_400, k: int = 2, seed: int = 0, garch: bool = True
) -> pd.DataFrame:
    """
    Même générateur à facteurs, mais la volatilité peut être GROUPÉE :
    un GARCH(1,1) sur l'échelle commune reproduit le fait stylisé le
    mieux établi des marchés — les grosses séances arrivent en paquets.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 0.5, (n, k))
    b[:, 0] = np.abs(rng.normal(0.8, 0.2, n))
    if garch:
        a, be = 0.14, 0.85
        h = np.empty(t)
        h[0] = 1.0
        z = rng.normal(0, 1, t)
        for i in range(1, t):
            h[i] = (1 - a - be) + a * (z[i - 1] ** 2 * h[i - 1]) + be * h[i - 1]
        echelle = np.sqrt(h)[:, None]
    else:
        echelle = np.ones((t, 1))
    f = rng.normal(0, 1, (t, k)) * echelle
    e = rng.normal(0, 1, (t, n)) * echelle * rng.uniform(0.5, 1.5, n)
    return pd.DataFrame(
        (f @ b.T + e) * 0.20 / np.sqrt(JOURS),
        index=pd.bdate_range("2015-01-02", periods=t),
        columns=[f"A{i:02d}" for i in range(n)],
    )


def test_le_chemin_de_calibration_a_une_ligne_par_rebalancement() -> None:
    donnees = _marche_groupe(t=1_500)
    r = walk_forward(donnees, MinimumVariance(ledoit_wolf), lookback=252,
                     step=21)
    chemin = r.calibration_path()

    assert list(chemin.columns) == ["prédite", "réalisée", "rapport", "n_obs"]
    assert 0 < len(chemin) <= r.n_rebalances
    assert (chemin["prédite"] > 0).all()
    assert (chemin["réalisée"] > 0).all()
    assert chemin["n_obs"].max() <= 21


def test_le_diagnostic_saccorde_avec_la_calibration_agregee() -> None:
    """
    ⚠️ LE BUG QUE CE TEST VERROUILLE.

    La première version du diagnostic moyennait les VOLATILITÉS par
    fenêtre et annonçait « calibration globale 1,02 » là où
    `calibration()` disait 0,91, sous le même nom.

    La moyenne des écarts-types n'est pas l'écart-type de l'ensemble :
    par l'inégalité de Jensen, la seconde est toujours la plus grande,
    et d'autant plus que les régimes diffèrent. Il faut agréger les
    VARIANCES.

    Deux méthodes du même objet qui rendent des chiffres différents
    sous le même nom usent la confiance plus sûrement qu'une erreur
    franche.
    """
    donnees = _marche_groupe(t=2_400, garch=True)
    r = walk_forward(donnees, MinimumVariance(ledoit_wolf), lookback=252,
                     step=21)

    agregee = r.calibration()
    assert agregee is not None
    ligne = r.calibration_diagnosis().splitlines()[0]
    valeur = float(ligne.split()[-1])

    assert abs(valeur - agregee.value) < 0.05, (
        f"le diagnostic annonce {valeur:.2f} et calibration() "
        f"{agregee.value:.2f} — les deux doivent mesurer la même chose"
    )


def test_le_diagnostic_distingue_un_biais_uniforme_dun_biais_concentre() -> None:
    """
    CALIBRATION CONTRE UNE VÉRITÉ CONNUE, cette fois sur la FORME de
    l'erreur plutôt que sur sa taille.

    À volatilité constante, l'écart de calibration doit être réparti :
    une tranche de 10 % des périodes porte environ 10 % de la variance.

    À volatilité groupée, la même tranche en porte le triple — et la
    calibration hors de ces périodes remonte au-dessus de 1. Le modèle
    n'est alors pas biaisé : il est EN RETARD.

    Les deux diagnostics appellent des remèdes opposés, et c'est pour
    cela qu'il faut pouvoir les distinguer.
    """
    plat = walk_forward(
        _marche_groupe(t=2_400, garch=False),
        MinimumVariance(ledoit_wolf), lookback=252, step=21,
    ).calibration_diagnosis()

    groupe = walk_forward(
        _marche_groupe(t=2_400, garch=True),
        MinimumVariance(ledoit_wolf), lookback=252, step=21,
    ).calibration_diagnosis()

    assert "UNIFORME" in plat, plat
    assert "CONCENTRÉ" in groupe, groupe
    assert "EN RETARD" in groupe


def test_le_chemin_refuse_une_strategie_qui_ne_predit_rien() -> None:
    donnees = _marche_groupe(t=1_000)
    r = walk_forward(donnees, EqualWeight(), lookback=252, step=21)
    with pytest.raises(ValueError, match="n'annonce aucun risque"):
        r.calibration_path()
