"""
Moteur de backtest — et l'audit qui vérifie qu'il ne triche pas.

POURQUOI CE MODULE EXISTE, ET CE QU'IL CORRIGE
==============================================

À la séance 7, pour savoir si le modèle à facteurs battait Ledoit-Wolf,
j'ai écrit le protocole d'évaluation hors échantillon DEUX FOIS : une
dans `examples/seance7.py`, une dans `tests/test_factors.py`. Deux
copies de la même logique, portant sur le résultat le plus important du
projet.

C'est la même famille de problème que les huit narrations qui ont dérivé
depuis la séance 2 : deux sources pour une seule vérité. Ce module en
fait une.

LE BIAIS DE LOOK-AHEAD, ET POURQUOI C'EST UN PROBLÈME D'ARCHITECTURE
=====================================================================

Un backtest est le document le plus malhonnête de la finance. Il produit
un chiffre unique, sans marge d'erreur, après qu'on a essayé vingt
stratégies et gardé la meilleure.

Mais sa défaillance la plus courante est plus bête que cela : la
stratégie a vu des données postérieures à la décision qu'elle prenait.
Un décalage d'un seul jour suffit. Ça ne lève aucune exception, ça ne
produit aucun NaN — ça produit une courbe de performance magnifique.

Les formes habituelles :

- un indice recalculé a posteriori (composition d'aujourd'hui appliquée
  à hier) — biais du survivant
- une normalisation calculée sur tout l'échantillon avant le découpage
- un `shift()` dans le mauvais sens, ou absent
- un estimateur ajusté une fois sur tout l'historique, puis « évalué »
  période par période
- un rééquilibrage daté du jour de la décision, exécuté au cours de
  clôture du même jour

Aucune de ces erreurs n'est mathématique. Toutes sont des erreurs de
STRUCTURE DU PROGRAMME : quelque chose a eu accès à un objet qu'il
n'aurait pas dû pouvoir atteindre.

DEUX MÉCANISMES, ET LE SECOND EST CELUI QUI COMPTE
--------------------------------------------------

**1. Le confinement.** Le moteur ne passe à la stratégie qu'une TRANCHE
de l'historique, qui s'arrête strictement avant la date de décision.
C'est ce que fait tout backtest correct. C'est nécessaire et ce n'est
pas suffisant : ça repose sur la discipline de celui qui a écrit le
moteur, et ça ne dit rien d'une stratégie qui garderait par ailleurs une
référence vers les données complètes.

**2. L'audit différentiel.** On définit la propriété qu'on veut vraiment
garantir :

    Un backtest est CAUSAL si et seulement si perturber les données
    postérieures à une date t ne change AUCUNE décision antérieure à t.

Cette propriété est testable. On perturbe le futur, on rejoue tout, et
on compare les poids. Si quoi que ce soit bouge avant t, il y a une
fuite — quel que soit le chemin par lequel elle passe, et sans qu'on ait
besoin de lire le code de la stratégie.

C'est la différence entre vérifier qu'on a bien écrit le code et
vérifier que le code a bien la propriété voulue. Le premier se fait par
relecture, le second par expérience.

Voir `detect_lookahead`, et sa limite déclarée — elle existe.

CE QUE CE MODULE NE PRÉTEND PAS FAIRE
-------------------------------------
Il n'y a ici ni carnet d'ordres, ni impact de marché, ni modèle
d'exécution. Les coûts sont un nombre de points de base appliqué au
turnover. C'est grossier, et c'est déjà beaucoup plus honnête que zéro :
sans coûts, tout backtest favorise mécaniquement les stratégies qui
tournent le plus, et la comparaison ne veut plus rien dire.

Références : Bailey & López de Prado (2014), *The Deflated Sharpe
Ratio* · Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-
Mathematics and Financial Charlatanism* · Lo (2002), *The Statistics of
Sharpe Ratios*.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats

from .covariance import CovarianceEstimate
from .returns import TRADING_DAYS_PER_YEAR, annualized_volatility, sharpe_ratio
from .risk import minimum_variance_weights
from .types import Estimate, InsufficientData

EULER_MASCHERONI = 0.5772156649015329


# ══════════════════════════════════════════════════════════════════════
#  LE CONTRAT D'UNE STRATÉGIE
# ══════════════════════════════════════════════════════════════════════


@runtime_checkable
class Strategy(Protocol):
    """
    Une stratégie transforme un historique en poids de portefeuille.

    Typage structurel, comme `Shock` à la séance 5 : aucune classe à
    hériter, aucun module à importer. Une fonction enveloppée dans un
    objet qui porte `name` et `weights` est une stratégie.

    CONTRAT, ET IL EST STRICT
    -------------------------
    `history` contient TOUT ce que la stratégie a le droit de savoir, et
    rien d'autre. La dernière ligne est la dernière période observée
    AVANT la décision. La décision s'appliquera à partir de la période
    suivante, qui n'est pas dans `history`.

    Une stratégie qui consulte une autre source que `history` viole ce
    contrat. Rien dans le langage ne l'en empêche — c'est précisément
    pour cela que `detect_lookahead` existe.
    """

    @property
    def name(self) -> str: ...

    def weights(self, history: pd.DataFrame) -> pd.Series: ...


@runtime_checkable
class PredictsRisk(Protocol):
    """
    Capacité OPTIONNELLE : annoncer la volatilité qu'on attend.

    Une stratégie peut l'implémenter ou non. Le moteur la détecte à
    l'exécution par `isinstance` et, si elle est là, enregistre la
    prédiction pour la comparer plus tard au réalisé.

    POURQUOI UN SECOND PROTOCOLE PLUTÔT QU'UNE MÉTHODE DANS LE PREMIER
    ------------------------------------------------------------------
    Ajouter `predicted_volatility` à `Strategy` obligerait toute
    stratégie à l'implémenter, y compris celles qui n'estiment aucun
    risque — l'équipondéré, par exemple, ne prédit rien du tout. On leur
    imposerait un `return None` décoratif.

    Deux protocoles séparés expriment exactement ce qui est vrai : un
    contrat obligatoire, une capacité facultative. C'est l'équivalent
    structurel de la ségrégation d'interface, et le `isinstance` de
    `runtime_checkable` suffit à la découvrir.
    """

    def predicted_volatility(self, history: pd.DataFrame) -> float: ...


@dataclass(frozen=True)
class EqualWeight:
    """1/N. La référence que toute méthode doit battre (DeMiguel 2009)."""

    name: str = "équipondéré"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        n = history.shape[1]
        return pd.Series(np.full(n, 1.0 / n), index=history.columns)


@dataclass(frozen=True)
class MinimumVariance:
    """
    Variance minimale sur une covariance estimée à chaque rebalancement.

    `estimator` est n'importe quelle fonction DataFrame → CovarianceEstimate :
    `sample_covariance`, `ledoit_wolf`, ou une fermeture autour de
    `pca_factor_model`. C'est ce qui permet de comparer les trois
    estimateurs des séances 2 et 7 dans un protocole unique.

    ⚠️ Cette classe ne rattrape AUCUNE exception. Si l'estimateur échoue
    — matrice singulière, structure indétectable — le backtest s'arrête.
    Un moteur qui se rabattrait silencieusement sur autre chose
    produirait une courbe de performance qui ne correspondrait à aucune
    stratégie exécutable. Le repli, s'il en faut un, est une décision de
    l'utilisateur et doit être écrite dans SON estimateur.
    """

    estimator: Callable[[pd.DataFrame], CovarianceEstimate]
    name: str = "variance minimale"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        return minimum_variance_weights(self.estimator(history))

    def predicted_volatility(self, history: pd.DataFrame) -> float:
        cov = self.estimator(history)
        w = minimum_variance_weights(cov).reindex(cov.assets).to_numpy()
        return float(np.sqrt(w @ cov.matrix @ w) * np.sqrt(TRADING_DAYS_PER_YEAR))


# ══════════════════════════════════════════════════════════════════════
#  LE RÉSULTAT
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BacktestResult:
    """
    Tout ce qu'un backtest a produit, et de quoi juger si on peut y croire.

    Comme `Estimate` à la séance 1 et `CovarianceEstimate` à la séance 2 :
    on ne retourne pas un chiffre de performance nu. Les méthodes de
    lecture renvoient des `Estimate` avec leurs intervalles — parce que
    la performance d'un backtest est, elle aussi, un tirage.
    """

    strategy: str
    returns: pd.Series          # nets de coûts, par période
    gross_returns: pd.Series
    weights: pd.DataFrame       # poids CIBLES à chaque rebalancement
    turnover: pd.Series         # à chaque rebalancement
    costs: pd.Series            # par période
    predicted_volatility: pd.Series | None
    n_rebalances: int
    cost_bps: float
    lookback: int
    mode: str

    @property
    def equity(self) -> pd.Series:
        """Valeur cumulée d'un euro investi, nette de coûts."""
        return (1.0 + self.returns).cumprod()

    def realized_volatility(
        self, periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> Estimate:
        return annualized_volatility(
            self.returns, periods_per_year=periods_per_year
        )

    def sharpe(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> Estimate:
        return sharpe_ratio(
            self.returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def annual_turnover(
        self, periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> float:
        """
        Turnover annualisé : somme des |Δw| rapportée à une année.

        Un chiffre de 2,0 signifie qu'on remplace deux fois le
        portefeuille par an. Au-delà de 5, aucun gain de volatilité ne
        survit aux frais.
        """
        if len(self.returns) == 0:
            return 0.0
        annees = len(self.returns) / periods_per_year
        return float(self.turnover.sum() / annees)

    def total_cost(self) -> float:
        return float(self.costs.sum())

    def calibration(
        self, periods_per_year: int = TRADING_DAYS_PER_YEAR
    ) -> Estimate | None:
        """
        Rapport entre le risque ANNONCÉ et le risque SUBI.

        1,00 : la stratégie savait ce qu'elle faisait.
        < 1  : elle a sous-estimé son propre risque — le sens dangereux.
        > 1  : elle a été trop prudente — coûteux, mais pas dangereux.

        L'intervalle vient de celui de la volatilité réalisée : la
        prédiction moyenne est traitée comme connue, et c'est le
        dénominateur qui porte l'incertitude. C'est une approximation,
        déclarée comme telle dans `assumptions` — l'incertitude sur la
        prédiction elle-même n'est pas propagée.

        AUCUN BACKTEST PUBLIÉ NE MONTRE CE CHIFFRE. Ils montrent tous le
        rendement et la volatilité réalisés, jamais l'écart entre ce que
        le modèle promettait et ce qui est arrivé. C'est pourtant la
        seule mesure qui dise si le modèle de risque vaut quelque chose,
        indépendamment de la performance.
        """
        if self.predicted_volatility is None or self.predicted_volatility.empty:
            return None

        realisee = self.realized_volatility(periods_per_year)
        predite = float(self.predicted_volatility.mean())

        return Estimate(
            value=predite / realisee.value,
            ci_low=predite / realisee.ci_high,
            ci_high=predite / realisee.ci_low,
            confidence=realisee.confidence,
            n_obs=len(self.predicted_volatility),
            method="prédit/réalisé, incertitude du dénominateur seul",
            assumptions=(
                "la prédiction moyenne est traitée comme connue sans erreur",
                "l'incertitude n'est propagée que depuis la volatilité réalisée",
            ),
        )

    def calibration_path(
        self,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        """
        La calibration comme SÉRIE TEMPORELLE, et non comme un nombre.

        Pour chaque rebalancement : le risque annoncé, le risque
        effectivement subi jusqu'au rebalancement suivant, et leur
        rapport.

        POURQUOI UN SEUL CHIFFRE NE SUFFIT PAS
        --------------------------------------
        `calibration()` dit « le modèle a annoncé 0,79 fois le risque
        subi ». Cela ne dit pas LA MÊME CHOSE selon que :

          • le modèle sous-estime de 21 % à chaque période — c'est un
            défaut systématique, et on peut le corriger par un facteur ;
          • ou qu'il est juste 90 % du temps et complètement dépassé
            pendant trois mois de crise — c'est le regroupement de
            volatilité, et aucun facteur ne le corrige, parce que le
            problème est que le modèle est EN RETARD, pas qu'il est
            décalé.

        Les deux diagnostics appellent des remèdes opposés. Le chiffre
        agrégé ne permet pas de choisir ; cette série, si.

        Les volatilités réalisées par fenêtre sont très bruitées — une
        vingtaine d'observations chacune. On les lit en distribution,
        jamais une par une.

        `horizon` — SUR COMBIEN DE PÉRIODES ON JUGE LA PRÉVISION
        --------------------------------------------------------
        Par défaut, chaque prévision est jugée sur l'intervalle qui la
        sépare du rebalancement suivant. C'est le choix naturel quand
        on évalue une stratégie : la prévision sert jusqu'au prochain
        arbitrage.

        Ce n'est PAS le bon choix quand on compare des estimateurs
        entre eux. La qualité d'une prévision de volatilité dépend de
        l'horizon auquel on la juge, et un estimateur à mémoire courte
        n'a aucune raison de bien prédire trois mois à l'avance. Fixer
        `horizon` permet de séparer les deux questions : « quel
        estimateur », et « pour quel horizon ».

        ⚠️ Un horizon de 1 est refusé : on ne mesure pas une volatilité
        sur une seule observation. La première version rendait
        silencieusement un tableau VIDE dans ce cas, et le code
        appelant plantait trois lignes plus loin sur une colonne
        absente — un échec déplacé, donc un échec mal diagnostiqué.
        """
        if self.predicted_volatility is None or self.predicted_volatility.empty:
            raise ValueError(
                "cette stratégie n'annonce aucun risque : il n'y a rien à "
                "calibrer. Seules les stratégies qui satisfont `PredictsRisk` "
                "en produisent."
            )

        if horizon is not None and horizon < 2:
            raise ValueError(
                f"horizon={horizon} : une volatilité ne se mesure pas sur "
                f"moins de deux observations. Augmente l'horizon, ou juge "
                f"la prévision autrement (log-vraisemblance à un pas)."
            )

        dates = list(self.predicted_volatility.index)
        positions = {d: i for i, d in enumerate(self.returns.index)}
        lignes = []
        retenues = []
        for i, debut in enumerate(dates):
            if horizon is not None:
                depart = positions.get(debut)
                if depart is None:
                    continue
                fenetre = self.returns.iloc[depart: depart + horizon]
            else:
                fin = dates[i + 1] if i + 1 < len(dates) else None
                fenetre = (
                    self.returns.loc[debut:]
                    if fin is None
                    else self.returns.loc[debut:fin].iloc[:-1]
                )
            if len(fenetre) < 2:
                continue
            retenues.append(debut)
            realisee = float(
                np.std(fenetre.to_numpy(), ddof=1) * np.sqrt(periods_per_year)
            )
            predite = float(self.predicted_volatility.loc[debut])
            lignes.append(
                {
                    "prédite": predite,
                    "réalisée": realisee,
                    "rapport": predite / realisee if realisee > 0 else np.nan,
                    "n_obs": len(fenetre),
                }
            )
        if not lignes:
            raise ValueError(
                "aucune fenêtre exploitable : chaque prévision est suivie de "
                "moins de deux observations. Augmente `step`, ou fixe "
                "`horizon`."
            )
        return pd.DataFrame(
            lignes, index=pd.Index(retenues, name="rebalancement")
        )

    def calibration_diagnosis(
        self,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        quantile: float = 0.10,
    ) -> str:
        """
        Répond à une seule question : l'écart de calibration est-il
        UNIFORME ou CONCENTRÉ sur quelques périodes agitées ?

        Les deux diagnostics appellent des remèdes opposés. Un biais
        uniforme se corrige par un facteur ; un modèle en retard sur les
        changements de régime, non — il faut changer d'estimateur.

        ⚠️ L'AGRÉGATION SE FAIT EN VARIANCE, PAS EN VOLATILITÉ.

        La première version moyennait les volatilités par fenêtre et
        annonçait « calibration globale 1,02 » là où `calibration()`
        disait 0,91. Les deux ne mesuraient pas la même chose : la
        moyenne des écarts-types n'est pas l'écart-type de l'ensemble.
        Par Jensen, la seconde est toujours la plus grande, et d'autant
        plus que les régimes diffèrent.

        Deux méthodes du même objet, portant le même nom et rendant des
        chiffres différents : c'est le genre d'incohérence qui use la
        confiance plus sûrement qu'une erreur franche.

        On agrège donc les variances, pondérées par le nombre
        d'observations de chaque fenêtre — ce que fait `calibration()`.
        """
        chemin = self.calibration_path(periods_per_year)
        if len(chemin) < 10:
            return "pas assez de périodes pour diagnostiquer"

        n = chemin["n_obs"].to_numpy(dtype=float)
        var = chemin["réalisée"].to_numpy() ** 2
        pred = chemin["prédite"].to_numpy()

        seuil = float(np.quantile(chemin["réalisée"], 1.0 - quantile))
        agitee = chemin["réalisée"].to_numpy() > seuil
        calme = ~agitee

        def rapport(masque: np.ndarray) -> float:
            poids = n[masque]
            return float(
                pred[masque].mean()
                / np.sqrt(np.average(var[masque], weights=poids))
            )

        part_variance = float(
            np.sum(var[agitee] * n[agitee]) / np.sum(var * n)
        )
        part_periodes = float(agitee.mean())
        global_ = rapport(np.ones(len(chemin), dtype=bool))
        hors = rapport(calme)

        concentre = part_variance > 2.5 * part_periodes
        verdict = (
            "CONCENTRÉ. Le modèle n'est pas biaisé, il est EN RETARD sur "
            "les changements de régime : il voit la tempête une fois "
            "qu'elle est dans sa fenêtre. Aucun facteur correctif ne "
            "répare cela — il faut un estimateur qui réagit (EWMA, GARCH)."
            if concentre
            else "UNIFORME. Le biais ne dépend pas du régime. Cherche du "
            "côté de l'estimateur lui-même ou de l'optimiseur."
        )

        return (
            f"calibration globale                          {global_:.2f}\n"
            f"calibration hors des {part_periodes:.0%} périodes agitées     "
            f"{hors:.2f}\n"
            f"part de la variance totale dans ces {part_periodes:.0%}   "
            f"{part_variance:.0%}\n"
            f"→ écart {verdict}"
        )

    def summary(self, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> str:
        lignes = [f"{self.strategy}  ({self.mode}, fenêtre {self.lookback}, "
                  f"{self.n_rebalances} rebalancements, {self.cost_bps:.0f} pb)"]
        try:
            vol = self.realized_volatility(periods_per_year)
            lignes.append(f"  volatilité réalisée  {vol.format(2, percent=True)}")
        except InsufficientData:
            lignes.append("  volatilité réalisée  données insuffisantes")
        try:
            sr = self.sharpe(periods_per_year=periods_per_year)
            lignes.append(f"  ratio de Sharpe      {sr.format(2)}")
        except InsufficientData:
            lignes.append("  ratio de Sharpe      données insuffisantes")
        cal = self.calibration(periods_per_year)
        if cal is not None:
            lignes.append(f"  prédit / réalisé     {cal.format(2)}")
        lignes.append(f"  turnover annuel      {self.annual_turnover():.2f}×")
        lignes.append(f"  coûts cumulés        {self.total_cost():.2%}")
        return "\n".join(lignes)

    def __str__(self) -> str:
        return self.summary()


# ══════════════════════════════════════════════════════════════════════
#  LE MOTEUR
# ══════════════════════════════════════════════════════════════════════


def walk_forward(
    returns: pd.DataFrame,
    strategy: Strategy,
    *,
    lookback: int,
    step: int = 21,
    mode: str = "rolling",
    cost_bps: float = 0.0,
) -> BacktestResult:
    """
    Rejoue une stratégie période par période, sans jamais lui montrer le futur.

    PARAMÈTRES
    ----------
    lookback : nombre de périodes visibles à chaque décision.
    step : périodes entre deux rebalancements. 21 ≈ une fois par mois.
    mode : "rolling" (fenêtre glissante de longueur fixe) ou "expanding"
        (tout l'historique disponible). Voir plus bas.
    cost_bps : coût en points de base appliqué au turnover réalisé.

    LA LIGNE QUI DÉCIDE DE TOUT
    ---------------------------
        history = returns.iloc[debut:t]      # t EXCLU
        ...appliqué à returns.iloc[t:t+step]

    `iloc[debut:t]` exclut l'indice t. La stratégie voit donc jusqu'à
    t−1 inclus, et ses poids sont appliqués à partir de t. Aucun
    recouvrement d'une seule période.

    Écrire `iloc[debut:t+1]` serait la version fautive. Elle ne lève
    aucune erreur, elle ne change rien à la forme des tableaux, et elle
    suffit à produire une courbe de performance spectaculaire. Le test
    `test_la_decision_nutilise_aucune_donnee_de_la_periode_evaluee`
    verrouille cette borne, et l'audit de `detect_lookahead` la
    détecterait même si quelqu'un la contournait autrement.

    FENÊTRE GLISSANTE OU CROISSANTE : UN ARBITRAGE, PAS UNE BONNE RÉPONSE
    ---------------------------------------------------------------------
    La fenêtre croissante utilise toute l'information disponible, donc
    estime plus précisément — si le monde est stationnaire. La fenêtre
    glissante oublie le passé lointain, donc s'adapte aux changements de
    régime — au prix d'estimations plus bruitées.

    Les marchés ne sont pas stationnaires, et la séance 7 a montré que
    pour un modèle à facteurs, allonger l'historique ne sert de toute
    façon presque à rien passé un certain point. Les deux raisons
    penchent du même côté, et c'est pourquoi `rolling` est le défaut.
    Mais c'est un arbitrage, et le moteur offre les deux.

    LA DÉRIVE DES POIDS ENTRE DEUX REBALANCEMENTS
    ---------------------------------------------
    Entre deux dates de décision, on ne touche à rien : les poids
    dérivent avec les prix. Le turnover se mesure donc entre les poids
    DÉRIVÉS et les nouveaux poids cibles, jamais entre deux cibles
    successives.

    Ignorer la dérive est une erreur courante, et elle SURESTIME le
    turnover — donc les coûts — sur les stratégies peu actives. Comme
    elle va dans le sens pessimiste, personne ne la cherche.
    """
    _valider_entrees(returns, lookback, step, mode, cost_bps)

    n_periodes = len(returns)
    colonnes = list(returns.columns)
    valeurs = returns.to_numpy(dtype=float)

    predit_le_risque = isinstance(strategy, PredictsRisk)

    rendements_nets: list[float] = []
    rendements_bruts: list[float] = []
    couts_par_periode: list[float] = []
    index_sortie: list = []

    poids_cibles: dict = {}
    turnovers: dict = {}
    predictions: dict = {}

    poids_courants = np.zeros(len(colonnes))
    premier = True

    for t in range(lookback, n_periodes, step):
        debut = t - lookback if mode == "rolling" else 0
        history = returns.iloc[debut:t]          # ← t EXCLU. Toute la sécurité.

        cible = strategy.weights(history).reindex(colonnes)
        if cible.isna().any():
            manquants = [c for c, v in zip(colonnes, cible, strict=True)
                         if pd.isna(v)]
            raise ValueError(
                f"la stratégie '{strategy.name}' n'a pas donné de poids pour "
                f"{manquants}"
            )
        cible_np = cible.to_numpy(dtype=float)

        if predit_le_risque:
            predictions[returns.index[t]] = float(
                strategy.predicted_volatility(history)  # type: ignore[attr-defined]
            )

        rotation = float(np.abs(cible_np - poids_courants).sum())
        if premier:
            # La mise en place initiale est un coût réel, mais ce n'est
            # pas de la ROTATION. La compter fausserait le turnover
            # annualisé de toute stratégie sur un historique court.
            rotation = 0.0
            premier = False
        turnovers[returns.index[t]] = rotation
        poids_cibles[returns.index[t]] = cible_np

        cout = rotation * cost_bps / 10_000.0
        poids_courants = cible_np.copy()

        fin = min(t + step, n_periodes)
        for u in range(t, fin):
            r_actifs = valeurs[u]
            brut = float(poids_courants @ r_actifs)
            cout_u = cout if u == t else 0.0
            rendements_bruts.append(brut)
            couts_par_periode.append(cout_u)
            rendements_nets.append(brut - cout_u)
            index_sortie.append(returns.index[u])

            # Dérive : les poids suivent les prix jusqu'au prochain
            # rebalancement. Division par (1+brut) pour qu'ils resomment à 1.
            croissance = poids_courants * (1.0 + r_actifs)
            total = croissance.sum()
            poids_courants = (
                croissance / total if total > 0 else np.zeros_like(croissance)
            )

    if not index_sortie:
        raise InsufficientData(
            n_periodes, lookback + 1, f"backtest de '{strategy.name}'"
        )

    idx = pd.Index(index_sortie, name=returns.index.name)
    idx_reb = pd.Index(list(poids_cibles), name=returns.index.name)

    return BacktestResult(
        strategy=strategy.name,
        returns=pd.Series(rendements_nets, index=idx, name="net"),
        gross_returns=pd.Series(rendements_bruts, index=idx, name="brut"),
        weights=pd.DataFrame(
            list(poids_cibles.values()), index=idx_reb, columns=colonnes
        ),
        turnover=pd.Series(list(turnovers.values()), index=idx_reb,
                           name="turnover"),
        costs=pd.Series(couts_par_periode, index=idx, name="coûts"),
        predicted_volatility=(
            pd.Series(list(predictions.values()),
                      index=pd.Index(list(predictions), name=returns.index.name),
                      name="vol prédite")
            if predictions
            else None
        ),
        n_rebalances=len(poids_cibles),
        cost_bps=cost_bps,
        lookback=lookback,
        mode=mode,
    )


def _valider_entrees(
    returns: pd.DataFrame, lookback: int, step: int, mode: str, cost_bps: float
) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("attendu un DataFrame (une colonne par actif)")
    if returns.isna().to_numpy().any():
        raise ValueError(
            "valeurs manquantes dans les rendements — à traiter "
            "explicitement avant un backtest"
        )
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"mode inconnu : '{mode}' (rolling ou expanding)")
    if lookback < 2:
        raise ValueError("lookback doit valoir au moins 2 périodes")
    if step < 1:
        raise ValueError("step doit valoir au moins 1 période")
    if cost_bps < 0:
        raise ValueError("cost_bps ne peut pas être négatif")
    if len(returns) <= lookback:
        raise InsufficientData(len(returns), lookback + 1, "backtest")


# ══════════════════════════════════════════════════════════════════════
#  L'AUDIT DE CAUSALITÉ
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Probe:
    """Une sonde : on a corrompu le futur à partir d'une date, et voici l'effet."""

    cut_position: int
    cut_label: object
    n_decisions_before: int
    max_weight_difference: float
    leaked: bool


@dataclass(frozen=True)
class LookaheadAudit:
    """Verdict de l'audit de causalité."""

    strategy: str
    probes: tuple[Probe, ...]
    tolerance: float

    @property
    def clean(self) -> bool:
        return not any(p.leaked for p in self.probes)

    @property
    def worst_difference(self) -> float:
        return max((p.max_weight_difference for p in self.probes), default=0.0)

    def __str__(self) -> str:
        if self.clean:
            return (
                f"✓ '{self.strategy}' est causale sur {len(self.probes)} sondes.\n"
                f"  Écart maximal observé sur les décisions antérieures : "
                f"{self.worst_difference:.2e} (tolérance {self.tolerance:.0e})"
            )
        fuites = [p for p in self.probes if p.leaked]
        premiere = min(fuites, key=lambda p: p.cut_position)
        return (
            f"✗ FUITE DÉTECTÉE dans '{self.strategy}'.\n"
            f"  {len(fuites)} sonde(s) sur {len(self.probes)} ont fait bouger "
            f"des décisions ANTÉRIEURES à la corruption.\n"
            f"  Première occurrence : coupure en position "
            f"{premiere.cut_position}, "
            f"{premiere.n_decisions_before} décisions antérieures affectées "
            f"jusqu'à {premiere.max_weight_difference:.4f} de poids.\n"
            f"  La stratégie lit des données postérieures à ses décisions."
        )


def detect_lookahead(
    returns: pd.DataFrame,
    build: Callable[[pd.DataFrame], Strategy],
    *,
    lookback: int,
    step: int = 21,
    mode: str = "rolling",
    cuts: Sequence[int] | None = None,
    n_probes: int = 5,
    tolerance: float = 1e-10,
    seed: int = 0,
) -> LookaheadAudit:
    """
    Vérifie expérimentalement qu'un backtest ne lit pas le futur.

    LA PROPRIÉTÉ TESTÉE
    -------------------
        Un backtest est CAUSAL si perturber les données postérieures à
        une date t ne change aucune décision antérieure à t.

    C'est une définition, pas une heuristique. Elle ne dépend d'aucune
    hypothèse sur la stratégie, et elle se vérifie sans lire son code.

    LA PROCÉDURE
    ------------
    1. Jouer le backtest de référence.
    2. Choisir une date de coupure t.
    3. Remplacer toutes les lignes à partir de t par du bruit d'une tout
       autre échelle.
    4. Reconstruire la stratégie à partir de ces données corrompues, et
       rejouer.
    5. Comparer les poids décidés AVANT t. Le moindre écart est une fuite.

    POURQUOI UNE FABRIQUE `build` ET NON UNE STRATÉGIE DÉJÀ CONSTRUITE
    ------------------------------------------------------------------
    C'est le point de conception du module, et il n'est pas évident.

    La fuite la plus fréquente en pratique n'est pas dans l'appel, elle
    est dans la CONSTRUCTION : un objet ajusté une fois sur tout
    l'échantillon, puis « évalué » période par période. Un scaler
    entraîné sur l'ensemble des données, un modèle calibré avant le
    découpage, une liste de titres choisie en connaissant la suite.

    Si l'audit recevait un objet déjà construit, il ne pourrait
    corrompre que ce qui passe par les appels — et cette fuite-là
    resterait invisible. En recevant `build`, il reconstruit toute la
    chaîne sur les données corrompues, et la fuite se révèle.

    Pour une stratégie sans état, `build = lambda donnees: ma_strategie`.

    LA LIMITE, ET ELLE EST RÉELLE
    -----------------------------
    L'audit ne voit que ce qui passe par `returns` et par `build`. Une
    stratégie qui irait chercher le futur ailleurs — un fichier sur le
    disque, un appel réseau, une variable globale capturée avant
    l'audit — ne sera pas détectée.

    Autrement dit : l'audit prouve la causalité RELATIVEMENT aux
    données qu'on lui confie. C'est une garantie forte et bornée, et la
    borne doit être annoncée. Un outil de vérification qui laisse croire
    qu'il garantit plus qu'il ne garantit est pire qu'aucun outil.
    """
    _valider_entrees(returns, lookback, step, mode, 0.0)

    reference = walk_forward(
        returns, build(returns), lookback=lookback, step=step, mode=mode
    )

    n = len(returns)
    decisions = list(range(lookback, n, step))

    if cuts is None:
        # ⚠️ LES COUPURES SE PLACENT EXACTEMENT SUR DES DATES DE DÉCISION.
        #
        # Première version : des coupures réparties uniformément. Elle
        # attrapait une stratégie ajustée sur tout l'échantillon et
        # RATAIT une avance de un à cinq jours — la fuite la plus
        # fréquente, et la seule qu'on ne voit pas à la relecture.
        #
        # La raison tient en une inégalité. Une décision prise en
        # position t, en avance de `a` périodes, lit jusqu'à l'indice
        # t+a−1. Elle n'est démasquée que si la corruption commence à un
        # indice qu'elle atteint, donc si cut ≤ t+a−1. Avec des coupures
        # au hasard et un pas de 25 jours, la probabilité qu'une
        # coupure tombe dans cette fenêtre de `a` jours vaut environ
        # a/step — 4 % pour une avance d'un jour.
        #
        # En posant la coupure SUR la date de décision, l'inégalité
        # devient cut = t ≤ t+a−1, vraie dès que a ≥ 1. Toute avance,
        # même d'une seule période, est détectée.
        #
        # La sensibilité d'un détecteur ne vient pas de sa finesse mais
        # de l'endroit où on le pose.
        k = min(n_probes, len(decisions))
        choix = np.unique(np.linspace(0, len(decisions) - 1, num=k).astype(int))
        cuts = [decisions[i] for i in choix]

    if not cuts:
        raise ValueError(
            f"aucune coupure possible : {n} périodes pour un lookback de "
            f"{lookback} et un pas de {step}"
        )

    rng = np.random.default_rng(seed)
    echelle = float(np.nanstd(returns.to_numpy(dtype=float)))
    sondes: list[Probe] = []

    for cut in cuts:
        corrompu = returns.copy()
        bruit = rng.normal(0.0, max(echelle, 1e-8) * 5.0,
                           size=(n - cut, returns.shape[1]))
        corrompu.iloc[cut:] = bruit

        perturbe = walk_forward(
            corrompu, build(corrompu), lookback=lookback, step=step, mode=mode
        )

        label = returns.index[cut]
        avant_ref = reference.weights.iloc[: _n_avant(reference, cut, lookback,
                                                      step)]
        avant_per = perturbe.weights.iloc[: len(avant_ref)]

        if len(avant_ref) == 0:
            ecart = 0.0
        else:
            ecart = float(
                np.abs(avant_ref.to_numpy() - avant_per.to_numpy()).max()
            )

        sondes.append(
            Probe(
                cut_position=cut,
                cut_label=label,
                n_decisions_before=len(avant_ref),
                max_weight_difference=ecart,
                leaked=ecart > tolerance,
            )
        )

    return LookaheadAudit(
        strategy=reference.strategy, probes=tuple(sondes), tolerance=tolerance
    )


def _n_avant(result: BacktestResult, cut: int, lookback: int, step: int) -> int:
    """
    Combien de décisions DOIVENT être inchangées quand on corrompt à
    partir de l'indice `cut`.

    Critère : **t ≤ cut**, et non t < cut.

    Une décision en position t n'a le droit de lire que les indices
    strictement inférieurs à t. Si la corruption commence à l'indice
    cut ≥ t, aucun indice autorisé n'est touché : les poids doivent
    être rigoureusement identiques.

    La version stricte (t < cut) excluait de la comparaison la décision
    prise exactement à la coupure — c'est-à-dire la seule capable de
    révéler une avance d'une période. Elle rendait le détecteur aveugle
    au décalage d'indice le plus courant.

    Ici, l'égalité n'est pas un détail de bord : c'est le cas qui porte
    toute la sensibilité.
    """
    positions = range(lookback, lookback + step * result.n_rebalances, step)
    return sum(1 for t in positions if t <= cut)


# ══════════════════════════════════════════════════════════════════════
#  LE BIAIS DE SÉLECTION — PROMESSE FAITE À LA SÉANCE 1
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DeflatedSharpe:
    """
    Ratio de Sharpe corrigé du nombre d'essais.

    `probability` est la probabilité que le vrai Sharpe dépasse le seuil,
    compte tenu du nombre de stratégies essayées, de l'asymétrie et de
    l'épaisseur des queues des rendements.
    """

    observed: float
    expected_max_under_null: float
    probability: float
    n_trials: int
    n_obs: int
    skewness: float
    kurtosis: float

    @property
    def survives(self) -> bool:
        """Au seuil usuel de 95 %."""
        return self.probability > 0.95

    def __str__(self) -> str:
        verdict = (
            "le résultat survit à la correction"
            if self.survives
            else "INDISTINGUABLE DE LA CHANCE une fois le nombre d'essais pris "
                 "en compte"
        )
        return (
            f"Sharpe observé                  : {self.observed:.2f}\n"
            f"Attendu au mieux par pure chance : "
            f"{self.expected_max_under_null:.2f}  ({self.n_trials} essais)\n"
            f"Probabilité que le vrai Sharpe soit positif : "
            f"{self.probability:.1%}\n"
            f"→ {verdict}"
        )


def expected_maximum_sharpe(n_trials: int, sharpe_dispersion: float) -> float:
    """
    Sharpe maximal attendu quand on essaie N stratégies toutes SANS VALEUR.

    Résultat de Bailey & López de Prado (2014), par la théorie des
    valeurs extrêmes pour des variables gaussiennes :

        E[max] ≈ σ·[ (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

    où γ est la constante d'Euler-Mascheroni et σ l'écart-type des
    Sharpe entre essais.

    CE QUE CE CHIFFRE SIGNIFIE, ET POURQUOI IL FAIT MAL
    ---------------------------------------------------
    Il ne dit pas que les essais étaient mauvais. Il dit : **même si
    AUCUNE des stratégies n'avait la moindre valeur, voici le Sharpe que
    la meilleure afficherait.** Avec assez d'essais, ce nombre dépasse
    n'importe quel seuil qu'on voudrait exiger.

    C'est la base du « théorème des fausses stratégies » : il n'existe
    aucun Sharpe assez élevé pour rejeter l'hypothèse de nullité si on
    ne sait pas combien de fois on a regardé.

    Les Sharpe sont ici par période, non annualisés.
    """
    if n_trials < 1:
        raise ValueError("n_trials doit valoir au moins 1")
    if sharpe_dispersion < 0:
        raise ValueError("la dispersion ne peut pas être négative")
    if n_trials == 1:
        return 0.0

    g = EULER_MASCHERONI
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_dispersion * ((1.0 - g) * a + g * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sharpe_dispersion: float | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_obs: int = 60,
) -> DeflatedSharpe:
    """
    Ratio de Sharpe dégonflé (Bailey & López de Prado, 2014).

    Corrige le Sharpe observé de trois biais que le Sharpe classique
    ignore entièrement :

    1. **Le nombre d'essais.** Vingt stratégies testées, la meilleure
       retenue : son Sharpe est le maximum de vingt tirages, pas un
       tirage.
    2. **L'asymétrie.** Une stratégie qui gagne souvent un peu et perd
       rarement beaucoup — vendre des options, par exemple — affiche un
       excellent Sharpe jusqu'au jour où elle n'en affiche plus.
    3. **L'épaisseur des queues.** Des rendements leptokurtiques rendent
       l'erreur-type du Sharpe plus grande que la formule gaussienne.

    LA FORMULE
    ----------
        DSR = Φ( (SR − SR₀)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )

    où SR₀ est le seuil de référence — ici le Sharpe maximal attendu
    sous l'hypothèse nulle, donné par `expected_maximum_sharpe`.

    `sharpe_dispersion` est l'écart-type des Sharpe PAR PÉRIODE observés
    sur l'ensemble des essais. Si on ne le connaît pas, on le remplace
    par l'erreur-type du Sharpe observé — une approximation
    conservatrice, signalée dans le résultat.

    POURQUOI CETTE FONCTION EST DANS CE MODULE
    ------------------------------------------
    Parce qu'elle n'a de sens qu'avec `n_trials`, et que `n_trials` est
    un fait sur le PROCESSUS de recherche, pas sur la série de
    rendements. Aucune fonction qui ne reçoit qu'une série ne peut le
    deviner. C'est pour cela que le Sharpe de la séance 1 porte dans ses
    `assumptions` la mention « UN SEUL essai — non corrigé du biais de
    sélection » : il ne pouvait pas faire mieux, et il le disait.

    Ton backtest ne mesure pas la qualité de ta stratégie. Il mesure le
    nombre de fois où tu as regardé.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < min_obs:
        raise InsufficientData(len(r), min_obs, "Sharpe dégonflé")

    n = len(r)
    ecart = float(np.std(r, ddof=1))
    if ecart == 0:
        raise ValueError("volatilité nulle : le Sharpe n'est pas défini")

    sr = float(np.mean(r)) / ecart          # par période
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))  # kurtosis NON centrée

    if sharpe_dispersion is None:
        # Repli : l'erreur-type de Lo (2002), qui décrit la dispersion
        # qu'auraient des essais indépendants de même longueur.
        sharpe_dispersion = float(np.sqrt((1.0 + 0.5 * sr**2) / n))

    sr0 = expected_maximum_sharpe(n_trials, sharpe_dispersion)

    denominateur = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denominateur <= 0:
        # Peut arriver sur des rendements très asymétriques : la
        # correction de Lo cesse d'être définie. On le dit plutôt que
        # de renvoyer un nombre.
        raise ValueError(
            f"variance du Sharpe non positive (asymétrie {g3:.2f}, "
            f"kurtosis {g4:.2f}) : l'approximation de Lo ne s'applique pas "
            f"à ces rendements"
        )

    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(denominateur)
    probabilite = float(stats.norm.cdf(z))

    echelle = np.sqrt(periods_per_year)
    return DeflatedSharpe(
        observed=sr * echelle,
        expected_max_under_null=sr0 * echelle,
        probability=probabilite,
        n_trials=n_trials,
        n_obs=n,
        skewness=g3,
        kurtosis=g4,
    )


def minimum_track_record_length(
    returns: pd.Series,
    target_sharpe: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_obs: int = 60,
) -> float:
    """
    Combien de périodes faut-il pour que ce Sharpe soit statistiquement
    distinguable de `target_sharpe` ?

        MinTRL = 1 + [1 − γ₃·SR + (γ₄−1)/4·SR²] · (Z_α / (SR − SR₀))²

    Retourne un nombre de périodes, à comparer à ce qu'on possède.

    C'est le chiffre à opposer à « ma stratégie fait du 1,5 de Sharpe
    depuis six mois ». La réponse n'est pas « je n'y crois pas », elle
    est « il faudrait N mois pour que j'aie le droit d'y croire ».

    `target_sharpe` est annualisé, comme celui qu'on lit partout.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < min_obs:
        raise InsufficientData(len(r), min_obs, "longueur minimale d'historique")

    ecart = float(np.std(r, ddof=1))
    if ecart == 0:
        raise ValueError("volatilité nulle : le Sharpe n'est pas défini")

    sr = float(np.mean(r)) / ecart
    sr0 = target_sharpe / np.sqrt(periods_per_year)

    if sr <= sr0:
        return float("inf")

    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    denominateur = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denominateur <= 0:
        raise ValueError(
            "variance du Sharpe non positive : l'approximation de Lo ne "
            "s'applique pas à ces rendements"
        )

    z = stats.norm.ppf(confidence)
    return float(1.0 + denominateur * (z / (sr - sr0)) ** 2)
