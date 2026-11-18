"""
Drawdown — le risque tel qu'il est réellement vécu.

CE QUI CHANGE DANS CETTE SÉANCE
===============================

Volatilité, VaR, CVaR ont toutes un point commun : elles décrivent la
DISTRIBUTION des rendements. Elles ignorent complètement l'ORDRE dans
lequel ces rendements arrivent.

Or personne ne vit une distribution. On vit un chemin.

Le drawdown est la première mesure de ce projet qui dépend du chemin. Et
la différence n'est pas un détail théorique : on peut réordonner les mêmes
rendements et faire varier la perte maximale d'un facteur trois, sans
bouger d'un centième la volatilité, la VaR ou la CVaR.

LA DÉFINITION
-------------
À chaque instant t, le drawdown est la chute depuis le plus haut atteint
jusque-là :

    DD(t) = V(t) / max(V(s), s ≤ t) − 1

C'est toujours négatif ou nul. Le maximum drawdown est le minimum de
cette courbe sur toute la période — la pire perte qu'un investisseur
entré au pire moment aurait subie avant de ressortir au pire moment.

POURQUOI CETTE MESURE COMPTE PLUS QUE LES AUTRES EN PRATIQUE
------------------------------------------------------------
C'est elle qui provoque les ventes de panique. Un investisseur ne liquide
pas parce que sa VaR à 99 % a été dépassée ; il liquide parce que son
compte affiche −40 % depuis le sommet et qu'il n'en peut plus. Le
drawdown est le lien entre la statistique et le comportement.

C'est aussi, et c'est le paradoxe de la séance, la mesure la PLUS mal
estimée de tout ce projet. Le maximum drawdown est une statistique
d'ordre : un seul nombre extrait d'un seul chemin. Son intervalle de
confiance à 90 % s'étale sur un facteur trois.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import Estimate, InsufficientData

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class DrawdownEpisode:
    """
    Un épisode complet : du sommet au creux, puis jusqu'au retour au sommet.

    `recovery` vaut None quand la série se termine avant que le niveau
    précédent n'ait été retrouvé — cas fréquent et important, qu'il ne faut
    surtout pas remplacer par zéro ou par la dernière date.
    """

    depth: float
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp | None
    peak_idx: int
    trough_idx: int
    recovery_idx: int | None

    @property
    def decline_days(self) -> int:
        """Durée de la chute, du sommet au creux."""
        return self.trough_idx - self.peak_idx

    @property
    def recovery_days(self) -> int | None:
        """Durée du retour au sommet. None si jamais retrouvé."""
        if self.recovery_idx is None:
            return None
        return self.recovery_idx - self.trough_idx

    @property
    def total_days(self) -> int | None:
        """Durée totale sous l'eau."""
        if self.recovery_idx is None:
            return None
        return self.recovery_idx - self.peak_idx

    @property
    def recovered(self) -> bool:
        return self.recovery_idx is not None

    @property
    def gain_required(self) -> float:
        """
        Le gain nécessaire pour effacer la perte.

        L'ASYMÉTRIE QUE TOUT LE MONDE SOUS-ESTIME
        -----------------------------------------
        Perdre 50 % exige de gagner 100 % pour revenir au point de départ.
        La formule est g = 1/(1+d) − 1, et elle est convexe :

            perte −10 %  →  il faut +11,1 %
            perte −20 %  →  il faut +25,0 %
            perte −33 %  →  il faut +50,0 %
            perte −50 %  →  il faut +100,0 %
            perte −80 %  →  il faut +400,0 %

        C'est la raison arithmétique pour laquelle éviter les grosses
        pertes compte davantage que capter les gros gains.
        """
        return 1.0 / (1.0 + self.depth) - 1.0

    def format(self) -> str:
        rec = (f"{self.recovery_days} j pour revenir"
               if self.recovered else "jamais revenu au sommet")
        return (f"{self.depth:>7.2%}  du {self.peak.date()} au {self.trough.date()}  "
                f"({self.decline_days} j de chute, {rec})")


def drawdown_curve(prices: pd.Series) -> pd.Series:
    """
    La courbe de drawdown — « underwater plot ».

    Plus informative que le seul maximum : elle montre combien de temps on
    passe sous l'eau, ce que le chiffre unique ne dit pas. Deux
    portefeuilles peuvent avoir le même maximum drawdown et passer l'un
    trois mois, l'autre six ans en dessous de leur sommet.
    """
    _validate(prices)
    v = prices.to_numpy(dtype=float)
    return pd.Series(v / np.maximum.accumulate(v) - 1.0,
                     index=prices.index, name="drawdown")


def max_drawdown(prices: pd.Series) -> DrawdownEpisode:
    """Le pire épisode de la série, avec ses dates et ses durées."""
    dd = drawdown_curve(prices)
    v = prices.to_numpy(dtype=float)

    trough = int(np.argmin(dd.to_numpy()))
    peak = int(np.argmax(v[: trough + 1]))

    rec = None
    if trough + 1 < len(v):
        after = np.nonzero(v[trough + 1:] >= v[peak])[0]
        if len(after):
            rec = int(after[0]) + trough + 1

    return DrawdownEpisode(
        depth=float(dd.iloc[trough]),
        peak=prices.index[peak], trough=prices.index[trough],
        recovery=prices.index[rec] if rec is not None else None,
        peak_idx=peak, trough_idx=trough, recovery_idx=rec,
    )


def drawdown_episodes(prices: pd.Series, min_depth: float = 0.05
                      ) -> list[DrawdownEpisode]:
    """
    Tous les épisodes dépassant `min_depth`, du plus profond au plus léger.

    Le maximum drawdown seul est un point unique, donc fragile. Regarder
    les cinq pires épisodes donne une image bien plus robuste du
    comportement d'un actif en période de stress.
    """
    _validate(prices)
    v = prices.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(v)
    sous_eau = v < peaks

    episodes: list[DrawdownEpisode] = []
    i = 0
    n = len(v)
    while i < n:
        if not sous_eau[i]:
            i += 1
            continue
        debut = i
        while i < n and sous_eau[i]:
            i += 1
        fin = i  # premier indice revenu au sommet, ou n

        seg = slice(debut, fin)
        creux = debut + int(np.argmin(v[seg]))
        sommet = debut - 1 if debut > 0 else 0
        prof = float(v[creux] / v[sommet] - 1.0)
        if prof > -min_depth:
            continue
        episodes.append(DrawdownEpisode(
            depth=prof, peak=prices.index[sommet], trough=prices.index[creux],
            recovery=prices.index[fin] if fin < n else None,
            peak_idx=sommet, trough_idx=creux,
            recovery_idx=fin if fin < n else None))

    return sorted(episodes, key=lambda e: e.depth)


def ulcer_index(prices: pd.Series) -> float:
    """
    Indice d'ulcère (Martin & McCann, 1989) — racine de la moyenne des
    carrés du drawdown.

    Le maximum drawdown ne retient qu'un seul instant de toute l'histoire.
    L'indice d'ulcère intègre la courbe entière : il pénalise les baisses
    PROFONDES **et LONGUES**. Deux trajectoires de même maximum drawdown
    mais dont l'une passe trois mois sous l'eau et l'autre six ans ont le
    même maximum et des ulcères très différents.

    ⚠️ CE QUE J'AI ÉCRIT D'ABORD, ET QUI ÉTAIT FAUX ⚠️

    J'avais présenté cet indice comme « le pendant de ce que la CVaR est
    à la VaR : une moyenne plutôt qu'un extremum, donc plus stable ».
    L'analogie est séduisante et elle est fausse. Mesuré sur 200
    trajectoires de même loi :

        coefficient de variation — ulcère 0,47  /  maximum drawdown 0,30
        écart entre deux moitiés — ulcère 39 %  /  maximum drawdown 27 %

    L'ulcère est donc **plus** dispersé, pas moins. La raison : il mesure
    le NIVEAU moyen de la trajectoire sous son sommet, et hérite donc de
    toute la variabilité du chemin, pas seulement de celle d'un extremum
    borné par zéro.

    La conclusion correcte n'est pas « plus fiable » mais « autre chose » :
    l'ulcère mesure la douleur intégrée dans le temps, le maximum mesure
    son pic. Les deux sont complémentaires, aucun ne remplace l'autre.
    """
    dd = drawdown_curve(prices).to_numpy()
    return float(np.sqrt(np.mean(dd**2)))


def calmar_ratio(prices: pd.Series,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Rendement annualisé divisé par la profondeur du maximum drawdown.

    ⚠️ À MANIER AVEC PRÉCAUTION. Son dénominateur est le maximum drawdown,
    c'est-à-dire la statistique la plus bruitée du projet. Un ratio de
    Calmar hérite donc de toute cette instabilité : deux échantillons de
    la même loi peuvent donner des Calmar séparés d'un facteur trois.

    Il est populaire parce qu'il parle le langage des investisseurs
    (« combien je gagne par unité de douleur »). Il est fragile pour
    exactement la même raison.
    """
    _validate(prices)
    n = len(prices) - 1
    total = float(prices.iloc[-1] / prices.iloc[0])
    annuel = total ** (periods_per_year / n) - 1.0
    mdd = abs(max_drawdown(prices).depth)
    if mdd < 1e-12:
        raise ValueError("drawdown nul : le ratio de Calmar n'est pas défini")
    return annuel / mdd


def expected_max_drawdown(
    volatility: float,
    years: float,
    drift: float = 0.0,
    n_sim: int = 4000,
    seed: int = 0,
    confidence: float = 0.90,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Estimate:
    """
    Distribution du maximum drawdown par simulation, avec son intervalle.

    ★ LA FONCTION QUI PORTE LA LEÇON DE LA SÉANCE ★

    Elle existe pour répondre à une question qu'aucune mesure précédente
    ne pouvait traiter : « ce drawdown de −35 % est-il anormal ? »

    Sans référence, on ne peut pas le dire. Avec la distribution simulée
    sous l'hypothèse nulle (même volatilité, même dérive, aucun talent ni
    aucune malchance particulière), on obtient enfin un point de
    comparaison.

    DEUX RÉSULTATS À RETENIR
    ------------------------
    1. L'intervalle est ÉNORME. Pour cinq ans à 20 % de volatilité, les
       90 % centraux vont d'environ −20 % à −54 %. Le maximum drawdown
       est une statistique d'ordre tirée d'un seul chemin ; il n'a presque
       aucun pouvoir discriminant.

    2. Le drawdown espéré CROÎT avec l'horizon, même avec une dérive
       positive. Médiane d'environ −17 % sur un an, −33 % sur cinq ans,
       −49 % sur vingt ans. Plus on détient longtemps, plus la pire chute
       déjà traversée sera profonde — mécaniquement, parce qu'on multiplie
       les occasions d'en subir une.

    La conséquence pratique est contre-intuitive : promettre à un
    investisseur de long terme qu'il « encaissera les creux » revient à
    lui promettre qu'il en verra de plus en plus profonds.
    """
    if years <= 0:
        raise ValueError("years doit être strictement positif")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence doit être dans (0,1)")

    n_days = round(years * periods_per_year)
    if n_days < 2:
        raise InsufficientData(n_days, 2, "drawdown espéré")

    rng = np.random.default_rng(seed)
    sd = volatility / np.sqrt(periods_per_year)
    mu = (drift - 0.5 * volatility**2) / periods_per_year

    # Tirage vectorisé : (n_sim, n_days) de rendements log
    r = rng.normal(mu, sd, size=(n_sim, n_days))
    log_eq = np.cumsum(r, axis=1)
    sommets = np.maximum.accumulate(log_eq, axis=1)
    # En log, DD = exp(log_v - log_sommet) - 1 ; le minimum sur la ligne
    # correspond au minimum de (log_v - log_sommet), monotone croissante.
    pires = np.exp(np.min(log_eq - sommets, axis=1)) - 1.0

    alpha = 1.0 - confidence
    return Estimate(
        value=float(np.median(pires)),
        # Convention NUMÉRIQUE, imposée par le type Estimate : ci_low ≤ ci_high.
        # Les drawdowns étant négatifs, ci_low est le plus PROFOND.
        # J'avais d'abord inversé les deux pour encoder « bas = moins grave » ;
        # la validation écrite en séance 1 a refusé l'objet. Un type qui
        # attrape son auteur fait exactement son travail.
        ci_low=float(np.quantile(pires, alpha / 2)),       # le plus profond
        ci_high=float(np.quantile(pires, 1 - alpha / 2)),  # le moins profond
        confidence=confidence,
        n_obs=n_sim,
        method="monte-carlo",
        assumptions=(
            "rendements normaux, indépendants, de volatilité constante",
            "aucune autocorrélation ni regroupement de volatilité",
            "les vrais marchés produisent des drawdowns PLUS profonds",
        ),
    )


def _validate(prices: pd.Series) -> None:
    if not isinstance(prices, pd.Series):
        raise TypeError("attendu une pd.Series de prix indexée par date")
    if len(prices) < 2:
        raise InsufficientData(len(prices), 2, "drawdown")
    v = prices.to_numpy()
    if not np.all(np.isfinite(v)):
        raise ValueError("valeurs non finies dans la série de prix")
    if np.any(v <= 0):
        raise ValueError("prix nuls ou négatifs : drawdown non défini")
