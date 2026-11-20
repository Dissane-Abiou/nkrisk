"""
Pondération exponentielle — un estimateur qui réagit, et ce qu'il coûte.

POURQUOI CE MODULE EXISTE
=========================

La séance 9 a confronté le moteur à de vraies actions. Le diagnostic est
tombé net :

    calibration globale                        0,79
    calibration hors des 11 % périodes agitées 1,11
    part de la variance totale dans ces 11 %   59 %

Le modèle de risque annonçait 79 % du risque subi. Et l'écart n'était
pas uniforme : **11 % des périodes portaient 59 % de la variance**. La
pire fenêtre, le 28 février 2020, annonçait 7,3 % de volatilité pour
98,4 % réalisée — un facteur treize.

Le verdict du diagnostic intégré : CONCENTRÉ. Le modèle n'est pas
biaisé, **il est en retard**. Une fenêtre glissante de 252 jours accorde
le même poids à la séance d'hier et à celle d'il y a un an. Quand la
tempête arrive, elle compte pour 1/252 — et il faut des mois pour
qu'elle pèse sur l'estimation.

Aucun facteur correctif ne répare cela. Multiplier toutes les
prédictions par 1,27 rendrait le modèle juste en moyenne et faux
partout : trop prudent les neuf dixièmes du temps, toujours dépassé le
reste.

LA RÉPONSE : OUBLIER PROGRESSIVEMENT
------------------------------------
Au lieu d'accorder 1/T à chacune des T dernières séances, on accorde un
poids qui décroît géométriquement avec l'ancienneté :

    σ²_t = (1−λ) · Σ_{i≥0} λ^i · r²_{t−1−i}

λ proche de 1 : longue mémoire, réaction lente.
λ plus petit : mémoire courte, réaction vive.

C'est la formule de RiskMetrics (J.P. Morgan, 1996), et elle est encore
le point de départ de la plupart des systèmes de risque en production.

CE QUE PERSONNE N'ÉCRIT À CÔTÉ DE CETTE FORMULE
===============================================

La réactivité n'est pas gratuite, et le prix se lit dans une seule
grandeur : **la taille d'échantillon effective**.

Avec des poids w_i normalisés, la taille effective au sens de Kish vaut
1/Σw²_i. Pour une pondération exponentielle, elle converge vers
(1+λ)/(1−λ) :

    λ = 0,80  →   9 observations effectives   (demi-vie   3 jours)
    λ = 0,90  →  19                           (demi-vie   7 jours)
    λ = 0,94  →  32                           (demi-vie  11 jours)
    λ = 0,97  →  66                           (demi-vie  23 jours)
    λ = 0,99  → 199                           (demi-vie  69 jours)

La fenêtre glissante du projet en utilisait 252.

**À λ = 0,94, l'estimateur voit l'équivalent de trente-deux
observations.** Trois conséquences, toutes mesurables, et toutes
ignorées par la présentation habituelle :

1. **L'intervalle de confiance s'élargit d'un facteur √(252/32) ≈ 2,8.**
   Un moteur qui transporte son incertitude ne peut pas faire semblant
   de ne pas le voir. C'est pourquoi `ewma_volatility` construit son
   intervalle sur n_eff et non sur n.

2. **Le problème de la séance 2 revient.** Pour N actifs et n_eff
   observations effectives, le rapport qui gouverne tout vaut N/n_eff.
   À N = 12 et λ = 0,94, cela fait 0,37 — la zone où la covariance
   empirique est déjà dangereusement mal conditionnée. À N = 50, elle
   est **singulière**.

3. **Le turnover augmente.** Une estimation qui bouge fait bouger les
   poids, et chaque mouvement se paie en frais.

La pondération exponentielle n'est donc pas « un meilleur
estimateur ». C'est un curseur entre deux erreurs : être en retard, ou
être imprécis. Ce module permet de le régler en connaissance de cause
plutôt que de recopier 0,94.

LE 0,94, JUSTEMENT
------------------
C'est la valeur publiée par RiskMetrics en 1996, ajustée sur un panier
de marchés de l'époque. Elle est devenue une constante par citation.

`fit_decay` l'estime par maximum de vraisemblance à un pas en avant, sur
tes données. La prédiction de chaque séance n'utilise que les
précédentes : le critère est causal par construction, au sens de la
séance 8.

Références : J.P. Morgan / Reuters (1996), *RiskMetrics Technical
Document*, 4ᵉ édition · Kish (1965), *Survey Sampling*, pour la taille
effective · Ledoit & Wolf (2004) pour la contraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .covariance import CovarianceEstimate
from .returns import TRADING_DAYS_PER_YEAR
from .types import Estimate, InsufficientData

DECAY_RISKMETRICS = 0.94


# ══════════════════════════════════════════════════════════════════════
#  LES POIDS, ET CE QU'ILS COÛTENT
# ══════════════════════════════════════════════════════════════════════


def ewma_weights(n_obs: int, decay: float) -> np.ndarray:
    """
    Poids exponentiels normalisés, du plus ancien au plus récent.

    w_i ∝ λ^(n−1−i), renormalisés pour sommer à 1.

    La renormalisation compte : sans elle, une série courte produit des
    poids qui somment à 1 − λ^n, donc une variance systématiquement
    sous-estimée. L'erreur est de 5 % à n = 50 pour λ = 0,94, et elle
    disparaît silencieusement quand n grandit — le genre de biais qu'on
    ne voit jamais parce qu'il ne se manifeste que sur les cas courts.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"λ doit être dans (0,1), reçu {decay}")
    if n_obs < 1:
        raise ValueError("au moins une observation")
    ages = np.arange(n_obs - 1, -1, -1, dtype=float)
    w = decay**ages
    return w / w.sum()


def effective_sample_size(decay: float, n_obs: int | None = None) -> float:
    """
    Taille d'échantillon effective au sens de Kish : 1 / Σ w²_i.

    C'est le nombre d'observations à poids égaux qui donnerait la même
    précision. Pour n_obs = None, on rend la limite asymptotique
    (1+λ)/(1−λ).

    POURQUOI CETTE GRANDEUR EST LE CŒUR DU MODULE
    ---------------------------------------------
    Elle transforme une intuition — « l'EWMA regarde moins loin » — en
    un nombre qu'on peut mettre dans un intervalle de confiance.

    Sans elle, on écrirait un intervalle calculé sur 2 500 observations
    pour une estimation qui en utilise réellement l'équivalent de 32.
    L'intervalle serait faux d'un facteur 8 en variance, et faux dans
    le sens dangereux : il annoncerait une précision qui n'existe pas.

    C'est exactement le mensonge que tout ce projet refuse depuis la
    séance 1 — et un moteur qui adopterait l'EWMA sans ajuster ses
    intervalles le commettrait sans s'en apercevoir.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"λ doit être dans (0,1), reçu {decay}")
    if n_obs is None:
        return float((1.0 + decay) / (1.0 - decay))
    w = ewma_weights(n_obs, decay)
    return float(1.0 / np.sum(w**2))


def half_life(decay: float) -> float:
    """Nombre de périodes au bout duquel le poids est divisé par deux."""
    if not 0.0 < decay < 1.0:
        raise ValueError(f"λ doit être dans (0,1), reçu {decay}")
    return float(np.log(0.5) / np.log(decay))


# ══════════════════════════════════════════════════════════════════════
#  VOLATILITÉ
# ══════════════════════════════════════════════════════════════════════


def ewma_volatility(
    returns: pd.Series,
    decay: float = DECAY_RISKMETRICS,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    demean: bool = False,
    min_obs: int = 30,
) -> Estimate:
    """
    Volatilité annualisée à pondération exponentielle, avec son
    intervalle construit sur la taille EFFECTIVE.

    L'intervalle suit la loi du khi-deux comme celui de la séance 1,
    mais avec n_eff − 1 degrés de liberté au lieu de n − 1. C'est une
    approximation du type Satterthwaite : la somme pondérée de carrés
    gaussiens n'est pas exactement un khi-deux, et on lui substitue
    celui qui a la même précision. Déclaré dans `assumptions`.

    POURQUOI `demean=False` PAR DÉFAUT
    ----------------------------------
    Convention RiskMetrics : on ne retranche pas la moyenne des
    rendements quotidiens.

    Ce n'est pas de la paresse, c'est le résultat de la séance 1. Sur
    dix ans de données, la volatilité s'estime à ±2,8 % d'imprécision
    relative et le rendement moyen à ±488 %, avec un intervalle qui
    contient zéro. Retrancher une moyenne qu'on ne sait pas estimer
    ajoute du bruit sans retirer de biais — et sur une fenêtre
    effective de trente-deux jours, c'est bien pire encore.

    On préfère donc l'hypothèse « moyenne nulle », qui est fausse mais
    de peu, à une estimation qui est juste mais de rien.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    r = returns.dropna().to_numpy(dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError("valeurs non finies dans les rendements")
    if len(r) < min_obs:
        raise InsufficientData(len(r), min_obs, "volatilité EWMA")

    w = ewma_weights(len(r), decay)
    centre = float(np.sum(w * r)) if demean else 0.0
    var = float(np.sum(w * (r - centre) ** 2))
    if var <= 0:
        raise ValueError("variance nulle : la série ne bouge pas")

    vol = np.sqrt(var * periods_per_year)
    n_eff = effective_sample_size(decay, len(r))
    df = max(n_eff - 1.0, 1.0)

    alpha = 1.0 - confidence
    bas = df / stats.chi2.ppf(1.0 - alpha / 2.0, df)
    haut = df / stats.chi2.ppf(alpha / 2.0, df)

    return Estimate(
        value=float(vol),
        ci_low=float(vol * np.sqrt(bas)),
        ci_high=float(vol * np.sqrt(haut)),
        confidence=confidence,
        n_obs=round(n_eff),
        method=f"ewma λ={decay:.3f} (n_eff={n_eff:.0f} sur {len(r)} points)",
        assumptions=(
            "rendements gaussiens indépendants conditionnellement à la variance",
            f"degrés de liberté effectifs ({df:.0f}) par approximation de "
            f"Satterthwaite — la somme pondérée n'est pas un khi-deux exact",
            "moyenne supposée nulle" if not demean else "moyenne estimée",
            f"n_obs rapporté est la taille EFFECTIVE, pas les {len(r)} "
            f"observations lues",
        ),
    )


# ══════════════════════════════════════════════════════════════════════
#  COVARIANCE
# ══════════════════════════════════════════════════════════════════════


def ewma_covariance(
    returns: pd.DataFrame,
    decay: float = DECAY_RISKMETRICS,
    demean: bool = False,
    shrinkage: float | None = None,
    min_obs: int = 30,
) -> CovarianceEstimate:
    """
    Matrice de covariance à pondération exponentielle.

        Σ = Σ_i w_i · (x_i − μ)(x_i − μ)ᵀ

    `shrinkage=None` (défaut) applique la contraction de Ledoit-Wolf
    généralisée aux poids — voir `_shrinkage_pondere`. Passer `0.0`
    désactive la contraction, un nombre entre 0 et 1 la force.

    ⚠️ POURQUOI LA CONTRACTION EST ACTIVE PAR DÉFAUT ICI, ALORS QU'ELLE
       NE L'EST PAS DANS `sample_covariance`

    Parce que la pondération exponentielle RAMÈNE le problème de la
    séance 2, et l'aggrave.

    Le rapport qui gouverne le conditionnement n'est pas N/T mais
    N/n_eff. À λ = 0,94, n_eff vaut 32 quelle que soit la longueur de
    l'historique — lire dix ans de données n'y change rien, puisque les
    poids lointains sont négligeables.

        N = 12  →  N/n_eff = 0,37   déjà mal conditionnée
        N = 32  →  N/n_eff = 1,00   singulière
        N = 50  →  N/n_eff = 1,55   sans espoir

    Un moteur qui proposerait une covariance EWMA nue inviterait
    l'utilisateur à inverser une matrice presque toujours dégénérée,
    sans le lui dire. Le défaut doit être utilisable ; l'utilisateur
    qui veut la matrice brute la demande explicitement.
    """
    x = _valider(returns)
    n_obs, n_assets = x.shape
    if n_obs < min_obs:
        raise InsufficientData(n_obs, min_obs, "covariance EWMA")

    w = ewma_weights(n_obs, decay)
    centre = (w[:, None] * x).sum(axis=0) if demean else np.zeros(n_assets)
    xc = x - centre

    # Σ = Xᵀ diag(w) X, écrit sans construire la matrice diagonale.
    pondere = xc * w[:, None]
    sigma = xc.T @ pondere
    sigma = (sigma + sigma.T) / 2.0        # symétrisation (cf. séance 7)

    n_eff = effective_sample_size(decay, n_obs)
    delta = (
        _shrinkage_pondere(xc, w, sigma, n_assets)
        if shrinkage is None
        else float(np.clip(shrinkage, 0.0, 1.0))
    )
    if delta > 0.0:
        mu = np.trace(sigma) / n_assets
        sigma = delta * mu * np.eye(n_assets) + (1.0 - delta) * sigma

    return CovarianceEstimate(
        matrix=sigma,
        assets=tuple(returns.columns),
        # On rapporte la taille EFFECTIVE : c'est elle qui gouverne
        # `ratio_n_over_t`, donc le diagnostic de fiabilité. Y mettre
        # n_obs annoncerait une matrice bien mieux déterminée qu'elle
        # ne l'est.
        n_obs=round(n_eff),
        method=f"ewma λ={decay:.3f}",
        shrinkage=delta,
    )


def _shrinkage_pondere(
    xc: np.ndarray, w: np.ndarray, sigma: np.ndarray, n_assets: int
) -> float:
    """
    Intensité de contraction de Ledoit-Wolf, généralisée à des poids.

    La formule d'origine (séance 2) suppose T observations de poids
    1/T :

        b̄² = (1/T²)·Σ_t ‖x_t x_tᵀ − S‖²  / N

    On remplace (1/T²) par w_t², qui s'y réduit exactement quand tous
    les w valent 1/T. Ce n'est donc pas une formule différente : c'est
    la même, écrite sans supposer l'équipondération.

    ⚠️ CE QUE CETTE GÉNÉRALISATION NE GARANTIT PAS

    Ledoit et Wolf démontrent que leur δ minimise l'erreur quadratique
    attendue SOUS ÉCHANTILLONNAGE IID. Rien de tel n'est démontré ici :
    des poids exponentiels servent précisément à traiter des données
    non identiquement distribuées.

    Le δ obtenu reste donc une heuristique raisonnable — il réagit dans
    le bon sens au rapport bruit/structure — et pas un optimum. Le
    dire est plus utile que de laisser croire qu'on hérite de la
    garantie de l'article.
    """
    d2 = float(np.sum((sigma - np.trace(sigma) / n_assets * np.eye(n_assets)) ** 2)
               / n_assets)
    if d2 <= 0:
        return 0.0

    # Σ_t w_t² ‖x_t x_tᵀ − Σ‖² / N, vectorisé.
    # ‖x x ᵀ − Σ‖² = (xᵀx)² − 2 xᵀΣx + ‖Σ‖²
    xx = np.einsum("ij,ij->i", xc, xc)
    xsx = np.einsum("ij,jk,ik->i", xc, sigma, xc)
    normes = xx**2 - 2.0 * xsx + float(np.sum(sigma**2))
    b_bar2 = float(np.sum(w**2 * normes) / n_assets)

    return float(np.clip(min(b_bar2, d2) / d2, 0.0, 1.0))


def _valider(returns: pd.DataFrame) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("attendu un DataFrame (une colonne par actif)")
    if returns.isna().to_numpy().any():
        raise ValueError("valeurs manquantes dans les rendements")
    x = returns.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("au moins 2 actifs requis")
    return x


# ══════════════════════════════════════════════════════════════════════
#  CHOISIR λ AU LIEU DE LE RECOPIER
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DecayFit:
    """Résultat d'un ajustement du facteur de décroissance."""

    decay: float
    log_likelihood: float
    grid: tuple[float, ...]
    likelihoods: tuple[float, ...]
    n_obs: int

    @property
    def effective_sample_size(self) -> float:
        return effective_sample_size(self.decay)

    @property
    def half_life(self) -> float:
        return half_life(self.decay)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "λ": self.grid,
                "log-vraisemblance": self.likelihoods,
                "n_eff": [effective_sample_size(d) for d in self.grid],
                "demi-vie": [half_life(d) for d in self.grid],
            }
        )

    def __str__(self) -> str:
        riskmetrics = next(
            (
                v
                for d, v in zip(self.grid, self.likelihoods, strict=True)
                if abs(d - DECAY_RISKMETRICS) < 1e-9
            ),
            None,
        )
        lignes = [
            f"λ ajusté            {self.decay:.3f}  "
            f"(n_eff {self.effective_sample_size:.0f}, "
            f"demi-vie {self.half_life:.0f} jours)",
            f"log-vraisemblance   {self.log_likelihood:,.1f}  "
            f"sur {self.n_obs:,} observations",
        ]
        if riskmetrics is not None:
            lignes.append(
                f"λ = 0,94 de RiskMetrics donnerait {riskmetrics:,.1f}, "
                f"soit {self.log_likelihood - riskmetrics:+,.1f}"
            )
        return "\n".join(lignes)


def fit_decay(
    returns: pd.Series | pd.DataFrame,
    grid: Sequence[float] | None = None,
    burn_in: int = 60,
    min_obs: int = 250,
) -> DecayFit:
    """
    Ajuste λ par maximum de vraisemblance à un pas en avant.

    Pour chaque λ candidat, on parcourt la série en prédisant la
    variance de chaque séance à partir des PRÉCÉDENTES seulement, et on
    somme la log-vraisemblance gaussienne :

        LL(λ) = −½ · Σ_t [ log σ²_t + r²_t / σ²_t ]

    où σ²_t est la récurrence σ²_t = λ·σ²_{t−1} + (1−λ)·r²_{t−1}.

    POURQUOI CE CRITÈRE PLUTÔT QU'UN AUTRE
    --------------------------------------
    Il est CAUSAL par construction. La prédiction de la séance t
    n'utilise aucune donnée de t ni d'après. C'est exactement la
    propriété que `detect_lookahead` vérifie expérimentalement sur les
    backtests à la séance 8, et ici elle est garantie par la forme de
    la récurrence plutôt que par un découpage.

    Un critère qui ajusterait λ sur toute la série d'un coup — en
    minimisant l'écart à la volatilité réalisée totale, par exemple —
    laisserait le futur entrer dans le choix, et donnerait un λ
    optimiste.

    POUR UN DATAFRAME
    -----------------
    On somme les log-vraisemblances actif par actif. Cela suppose
    l'indépendance entre actifs, ce qui est faux — mais le λ qui en
    sort est un compromis raisonnable pour tout l'univers, et c'est ce
    qu'on cherche. Un λ par actif serait plus juste et rendrait la
    matrice de covariance incohérente.
    """
    if grid is None:
        grid = tuple(np.round(np.arange(0.80, 0.995, 0.005), 4))
    grid = tuple(float(g) for g in grid)
    for g in grid:
        if not 0.0 < g < 1.0:
            raise ValueError(f"λ hors de (0,1) dans la grille : {g}")

    if isinstance(returns, pd.Series):
        colonnes = [returns.dropna().to_numpy(dtype=float)]
    else:
        if returns.isna().to_numpy().any():
            raise ValueError("valeurs manquantes dans les rendements")
        colonnes = [returns[c].to_numpy(dtype=float) for c in returns.columns]

    n = len(colonnes[0])
    if n < min_obs:
        raise InsufficientData(n, min_obs, "ajustement de λ")
    if burn_in >= n:
        raise ValueError(f"burn_in ({burn_in}) ≥ longueur de série ({n})")

    vraisemblances = [
        sum(_log_vraisemblance(serie, g, burn_in) for serie in colonnes)
        for g in grid
    ]
    meilleur = int(np.argmax(vraisemblances))

    return DecayFit(
        decay=grid[meilleur],
        log_likelihood=float(vraisemblances[meilleur]),
        grid=grid,
        likelihoods=tuple(float(v) for v in vraisemblances),
        n_obs=(n - burn_in) * len(colonnes),
    )


def _log_vraisemblance(r: np.ndarray, decay: float, burn_in: int) -> float:
    """
    Log-vraisemblance gaussienne à un pas en avant, aux constantes près.

    La récurrence est amorcée sur la variance des `burn_in` premières
    séances, qui sont ensuite exclues du score : elles ont servi à
    initialiser, les compter reviendrait à noter l'amorçage.
    """
    var = float(np.var(r[:burn_in], ddof=1))
    if var <= 0:
        return -np.inf

    total = 0.0
    for i in range(burn_in, len(r)):
        var = decay * var + (1.0 - decay) * r[i - 1] ** 2
        if var <= 0:
            return -np.inf
        total += -0.5 * (np.log(var) + r[i] ** 2 / var)
    return total
