"""
Générateur de prix synthétiques à propriétés statistiques CONNUES.

POURQUOI C'EST LE FICHIER LE PLUS IMPORTANT POUR APPRENDRE
==========================================================

Question : comment sais-tu que ton estimateur de volatilité est correct ?

Avec des données réelles, tu ne le sais pas. Tu calcules 18,3 % et... c'est
tout. Tu n'as aucun moyen de vérifier, parce que la « vraie » volatilité du
marché est inobservable. Tu peux avoir un bug qui multiplie tout par 1,02 et
ne jamais t'en apercevoir.

Avec des données synthétiques, tu FIXES la vérité. Tu génères une série dont
tu sais qu'elle a une volatilité annualisée de 20 %, tu lances ton estimateur,
et tu vérifies qu'il retrouve 20 % à l'incertitude près. Si non, tu as un bug.

C'est la seule façon rigoureuse de valider un moteur de risque. C'est aussi
la raison pour laquelle je te fais commencer par là plutôt que par le
téléchargement de données réelles : télécharger des prix est facile et
n'apprend rien ; valider un estimateur est difficile et apprend tout.

LE MODÈLE : mouvement brownien géométrique
------------------------------------------
On suppose que le prix suit :

    S_t = S_0 · exp[ (μ - σ²/2)·t + σ·W_t ]

où W_t est un mouvement brownien standard. Conséquence directe : les
rendements logarithmiques sur un intervalle Δt sont normaux, indépendants,
d'espérance (μ - σ²/2)·Δt et de variance σ²·Δt.

Le terme -σ²/2 s'appelle la CORRECTION D'ITÔ, et c'est le premier vrai
concept de finance quantitative de ce projet. On y revient dans returns.py :
c'est la raison mathématique pour laquelle la volatilité te coûte du
rendement même quand elle est symétrique.

⚠️ Ce modèle est FAUX pour les marchés réels. Il n'a ni queues épaisses, ni
regroupement de volatilité, ni sauts. Les vrais rendements ont tout ça. On
l'utilise ici uniquement comme banc d'essai contrôlé — jamais comme
description du monde.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Nombre de jours de bourse dans une année. Convention quasi universelle.
# 252 = 365 jours - 104 jours de week-end - ~9 jours fériés.
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class GBMSpec:
    """
    Spécification d'une série synthétique. Tous les paramètres sont ANNUELS.

    mu : dérive annuelle attendue du prix (rendement arithmétique espéré).
         0.08 = 8 % par an.
    sigma : volatilité annualisée. 0.20 = 20 % par an.
    s0 : prix initial.
    """

    mu: float
    sigma: float
    s0: float = 100.0

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma ne peut pas être négatif")
        if self.s0 <= 0:
            raise ValueError("le prix initial doit être strictement positif")

    @property
    def daily_log_mean(self) -> float:
        """Espérance du rendement log quotidien — avec la correction d'Itô."""
        return (self.mu - 0.5 * self.sigma**2) / TRADING_DAYS_PER_YEAR

    @property
    def daily_log_std(self) -> float:
        """
        Écart-type du rendement log quotidien.

        Noter le √252 : la VARIANCE est additive dans le temps (pour des
        rendements indépendants), donc elle est proportionnelle à t, donc
        l'écart-type est proportionnel à √t. C'est toute l'origine de la
        « règle de la racine carrée du temps », qu'on utilisera partout.
        """
        return self.sigma / np.sqrt(TRADING_DAYS_PER_YEAR)


def generate_prices(
    spec: GBMSpec,
    n_days: int,
    seed: int,
    start: str = "2020-01-01",
    name: str = "SYNTH",
) -> pd.Series:
    """
    Génère une série de prix suivant exactement `spec`.

    `seed` est OBLIGATOIRE et non optionnel : un test qui utilise de
    l'aléatoire non reproductible est un test qui échouera un jour sans
    qu'on puisse comprendre pourquoi. Dans du code numérique, l'aléa
    non graine est un bug par construction.

    Retourne une Series pandas indexée par date ouvrable.
    """
    if n_days < 1:
        raise ValueError("n_days doit être >= 1")

    rng = np.random.default_rng(seed)

    # On tire les rendements LOG, parce que c'est eux qui sont normaux
    # dans ce modèle. Les rendements simples, eux, sont log-normaux.
    log_returns = rng.normal(
        loc=spec.daily_log_mean,
        scale=spec.daily_log_std,
        size=n_days,
    )

    # cumsum sur les log-rendements = produit sur les rendements simples.
    # C'est la propriété d'additivité temporelle du log : elle transforme
    # une multiplication coûteuse et instable en une addition.
    log_price = np.log(spec.s0) + np.cumsum(log_returns)
    prices = np.exp(log_price)

    # On préfixe le prix initial pour que la série contienne n_days + 1
    # points et donc exactement n_days rendements.
    prices = np.concatenate([[spec.s0], prices])

    dates = pd.bdate_range(start=start, periods=len(prices))
    return pd.Series(prices, index=dates, name=name)


def generate_correlated_prices(
    specs: dict[str, GBMSpec],
    correlation: np.ndarray,
    n_days: int,
    seed: int,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Génère plusieurs séries avec une matrice de corrélation IMPOSÉE.

    Indispensable pour la séance 2 (covariance et décomposition du risque) :
    on ne peut valider un estimateur de corrélation que si on connaît la
    corrélation vraie.

    Méthode : décomposition de Cholesky. Si Z est un vecteur de normales
    indépendantes et L·Lᵀ = C, alors L·Z a pour matrice de corrélation C.
    C'est le résultat qui rend possible toute simulation multivariée.
    """
    names = list(specs.keys())
    k = len(names)

    if correlation.shape != (k, k):
        raise ValueError(
            f"la matrice de corrélation doit être {k}×{k}, "
            f"reçu {correlation.shape}"
        )
    if not np.allclose(correlation, correlation.T):
        raise ValueError("la matrice de corrélation doit être symétrique")
    if not np.allclose(np.diag(correlation), 1.0):
        raise ValueError("la diagonale d'une matrice de corrélation vaut 1")

    # Une matrice de corrélation doit être semi-définie positive. Si elle ne
    # l'est pas, elle ne correspond à aucune réalité possible — et Cholesky
    # échouera. On veut un message clair plutôt qu'une erreur cryptique.
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues.min() < -1e-10:
        raise ValueError(
            f"matrice de corrélation non semi-définie positive "
            f"(valeur propre minimale = {eigenvalues.min():.2e}). "
            f"Aucun ensemble d'actifs ne peut avoir ces corrélations."
        )

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(correlation + 1e-12 * np.eye(k))

    # (n_days, k) de normales indépendantes → corrélées
    z = rng.normal(size=(n_days, k))
    correlated = z @ L.T

    columns = {}
    for i, nm in enumerate(names):
        spec = specs[nm]
        log_returns = spec.daily_log_mean + spec.daily_log_std * correlated[:, i]
        log_price = np.log(spec.s0) + np.cumsum(log_returns)
        columns[nm] = np.concatenate([[spec.s0], np.exp(log_price)])

    dates = pd.bdate_range(start=start, periods=n_days + 1)
    return pd.DataFrame(columns, index=dates)


# ══════════════════════════════════════════════════════════════════════
#  QUEUES ÉPAISSES — indispensable pour la séance 3
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StudentTSpec:
    """
    Prix dont les rendements suivent une loi de Student, pas une normale.

    POURQUOI C'EST NÉCESSAIRE
    -------------------------
    Le mouvement brownien géométrique produit des rendements normaux.
    Les marchés réels, non : ils ont des QUEUES ÉPAISSES. Les krachs sont
    beaucoup plus fréquents que la loi normale ne le prédit.

    Ordre de grandeur qui devrait te glacer : sous hypothèse normale, une
    baisse journalière de 5 écarts-types a une probabilité d'environ
    1 sur 3,5 millions — soit une occurrence tous les 14 000 ans de
    bourse. Le 19 octobre 1987, le S&P 500 a perdu plus de 20 écarts-types.
    Sous la normale, cet événement est proprement impossible : sa
    probabilité est de l'ordre de 10⁻⁸⁸.

    Il s'est pourtant produit. La loi normale ne décrit donc pas la
    réalité là où ça compte le plus — dans la queue.

    LE PARAMÈTRE df (degrés de liberté)
    -----------------------------------
      df → ∞   la Student tend vers la normale
      df = 5   queues nettement épaisses, kurtosis excédentaire = 6
      df = 4   kurtosis excédentaire = 6 ... ordre de grandeur observé
               sur les indices actions quotidiens
      df ≤ 4   la kurtosis n'existe plus (intégrale divergente)
      df ≤ 2   la VARIANCE elle-même n'existe plus

    On normalise la Student pour qu'elle ait exactement la volatilité
    demandée : ainsi la seule différence avec le GBM est la FORME des
    queues, pas leur échelle. C'est ce qui rend la comparaison honnête.
    """

    mu: float
    sigma: float
    df: float = 4.0
    s0: float = 100.0

    def __post_init__(self) -> None:
        if self.df <= 2:
            raise ValueError(
                f"df={self.df} : avec df ≤ 2 la variance est infinie et la "
                f"notion de volatilité n'a plus de sens."
            )
        if self.sigma < 0:
            raise ValueError("sigma ne peut pas être négatif")

    @property
    def excess_kurtosis(self) -> float:
        """Kurtosis excédentaire théorique : 6/(df-4), infinie si df ≤ 4."""
        return np.inf if self.df <= 4 else 6.0 / (self.df - 4.0)


def generate_student_t_prices(
    spec: StudentTSpec,
    n_days: int,
    seed: int,
    start: str = "2020-01-01",
    name: str = "FAT",
) -> pd.Series:
    """
    Série de prix à queues épaisses, de volatilité annualisée EXACTE.

    La Student brute a une variance de df/(df-2). On divise donc par
    √(df/(df-2)) pour la ramener à une variance unitaire avant de la
    mettre à l'échelle. Sans cette normalisation, on comparerait des
    distributions de largeurs différentes et l'expérience ne prouverait
    rien.
    """
    rng = np.random.default_rng(seed)

    daily_std = spec.sigma / np.sqrt(TRADING_DAYS_PER_YEAR)
    daily_mean = (spec.mu - 0.5 * spec.sigma**2) / TRADING_DAYS_PER_YEAR

    raw = rng.standard_t(df=spec.df, size=n_days)
    normalized = raw / np.sqrt(spec.df / (spec.df - 2.0))

    log_returns = daily_mean + daily_std * normalized
    prices = np.concatenate(
        [[spec.s0], spec.s0 * np.exp(np.cumsum(log_returns))]
    )
    dates = pd.bdate_range(start=start, periods=len(prices))
    return pd.Series(prices, index=dates, name=name)
