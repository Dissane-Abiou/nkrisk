"""
Correction du biais de sélection — le Sharpe dégonflé.

POURQUOI CE N'EST PAS DANS `backtest.py`
-----------------------------------------
Le moteur de rejeu répond à « qu'aurait fait cette stratégie ? ».
Ce module répond à une question d'une autre nature : « combien de
stratégies ai-je essayées avant de garder celle-là ? »

La première est une question de simulation, la seconde une question
d'inférence. Elles partageaient un fichier de 1 152 lignes pour la
seule raison qu'elles sont arrivées à la même séance. Un relecteur qui
veut vérifier la discipline point-in-time n'a aucune raison de faire
défiler la théorie des valeurs extrêmes.

LE PROBLÈME QU'IL TRAITE
------------------------
Un Sharpe de 1,0 n'a pas le même sens selon qu'on a testé une
stratégie ou douze mille. En cherchant assez longtemps, on trouve
toujours une combinaison qui a bien marché dans le passé — et le
maximum d'un grand nombre de tirages bruités est grand par
construction, pas par talent.

Bailey et López de Prado (2014) corrigent le Sharpe observé par le
nombre d'essais, via l'espérance du maximum de N tirages d'une loi
normale. Mesuré dans `examples/seance8.py` : un Sharpe de 1,0 obtenu
après 12 650 essais voit sa probabilité d'être réel tomber de 99,6 % à
10,3 %, sur exactement les mêmes rendements.

Référence
---------
Bailey, D. & López de Prado, M. (2014). « The Deflated Sharpe Ratio ».
*Journal of Portfolio Management* 40(5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .types import InsufficientData

TRADING_DAYS_PER_YEAR = 252

#: Constante d'Euler-Mascheroni, qui apparaît dans l'espérance du
#: maximum de N tirages gaussiens (théorie des valeurs extrêmes).
EULER_MASCHERONI = 0.5772156649015329

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
