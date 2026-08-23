"""
Estimation de la matrice de covariance.

LE PROBLÈME CENTRAL DE LA SÉANCE 2
==================================

Pour N actifs, la matrice de covariance contient N(N+1)/2 paramètres
distincts à estimer. Avec 100 actifs, cela fait **5 050 paramètres**.

Combien d'observations as-tu pour les estimer ? Un an de données
quotidiennes en donne 252.

Cinq mille paramètres estimés à partir de deux cent cinquante-deux
observations. Le problème est massivement sous-déterminé, et la matrice
empirique qui en sort n'est pas simplement « imprécise » : quand N > T
elle est **singulière**, donc non inversible, donc inutilisable pour
toute optimisation de portefeuille. Et même quand N < T, elle est si mal
conditionnée que l'inverser amplifie le bruit de façon catastrophique.

C'est le problème le plus important de la finance quantitative appliquée,
et c'est celui que la plupart des tutoriels passent sous silence en
appelant simplement `np.cov()`.

LA GÉOMÉTRIE DU DÉSASTRE
------------------------
Les valeurs propres de la matrice empirique sont systématiquement biaisées :
les grandes sont surestimées, les petites sous-estimées — et ce d'autant
plus que N/T est grand. C'est un résultat de théorie des matrices
aléatoires (loi de Marchenko-Pastur).

Or l'inversion d'une matrice inverse ses valeurs propres. Les petites
valeurs propres, celles qui sont le plus sous-estimées, deviennent
d'énormes valeurs propres dans l'inverse. **L'optimiseur va donc charger
massivement les directions les moins fiablement estimées.** Il ne
maximise pas le rendement : il maximise l'erreur d'estimation.

C'est la version rigoureuse du surnom de Michaud (1989) : « error
maximizer ».

LA SOLUTION : LA CONTRACTION
----------------------------
Ledoit et Wolf (2004) proposent de tirer l'estimation empirique vers une
cible structurée et bien conditionnée :

    Σ* = δ·F + (1-δ)·S

où S est la matrice empirique, F une cible rigide (ici un multiple de
l'identité) et δ ∈ [0,1] une intensité de contraction calculée de façon
optimale, en forme close, sans validation croisée.

On accepte volontairement un BIAIS pour réduire massivement la VARIANCE.
C'est le compromis biais-variance, appliqué à une matrice. L'estimateur
qui en résulte est toujours inversible, toujours défini positif, et
domine l'empirique hors échantillon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovarianceEstimate:
    """
    Une matrice de covariance, avec le diagnostic de sa fiabilité.

    Comme pour `Estimate` à la séance 1 : on refuse de retourner une
    matrice nue. Elle vient avec ce qu'il faut pour juger si on peut
    s'en servir.
    """

    matrix: np.ndarray
    assets: tuple[str, ...]
    n_obs: int
    method: str
    shrinkage: float = 0.0

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def condition_number(self) -> float:
        """
        Rapport entre la plus grande et la plus petite valeur propre.

        C'est le facteur d'amplification du bruit lors de l'inversion.
        Un conditionnement de 10⁶ signifie qu'une erreur relative de
        10⁻⁶ dans les données devient une erreur de 100 % dans l'inverse.

        Repères pratiques :
          < 100      excellent
          100–1 000  utilisable
          > 10 000   dangereux pour toute optimisation
          > 10¹²     numériquement singulier
        """
        eig = np.linalg.eigvalsh(self.matrix)
        if eig.min() <= 0:
            return np.inf
        return float(eig.max() / eig.min())

    @property
    def ratio_n_over_t(self) -> float:
        """
        N/T — le paramètre qui gouverne tout.

        > 1   la matrice empirique est SINGULIÈRE, mathématiquement
        > 0.5 zone rouge, l'inversion est du bruit pur
        > 0.1 la contraction apporte déjà beaucoup
        """
        return self.n_assets / self.n_obs

    @property
    def is_invertible(self) -> bool:
        return np.linalg.eigvalsh(self.matrix).min() > 1e-12

    def volatilities(self) -> pd.Series:
        """Volatilités individuelles = racines de la diagonale."""
        return pd.Series(np.sqrt(np.diag(self.matrix)), index=self.assets)

    def correlation(self) -> pd.DataFrame:
        """
        Matrice de corrélation : C = D⁻¹ Σ D⁻¹, avec D = diag(σ).

        Utile pour l'inspection humaine — une covariance est illisible,
        une corrélation se lit d'un coup d'œil.
        """
        sd = np.sqrt(np.diag(self.matrix))
        c = self.matrix / np.outer(sd, sd)
        np.fill_diagonal(c, 1.0)
        return pd.DataFrame(c, index=self.assets, columns=self.assets)

    def diagnostic(self) -> str:
        """Verdict lisible sur l'utilisabilité de cette matrice."""
        ratio = self.ratio_n_over_t
        cond = self.condition_number

        if not self.is_invertible:
            verdict = "SINGULIÈRE — inutilisable pour toute optimisation"
        elif cond > 1e4:
            verdict = "DANGEREUSE — l'inversion amplifiera le bruit"
        elif cond > 1e3:
            verdict = "fragile — à contracter"
        else:
            verdict = "utilisable"

        return (
            f"{self.method}  |  N={self.n_assets}  T={self.n_obs}  "
            f"N/T={ratio:.2f}  cond={cond:,.0f}"
            + (f"  δ={self.shrinkage:.1%}" if self.shrinkage else "")
            + f"\n  → {verdict}"
        )


def sample_covariance(returns: pd.DataFrame) -> CovarianceEstimate:
    """
    Matrice de covariance empirique — l'estimateur naïf.

    C'est `np.cov()`. Il est sans biais, et c'est à peu près sa seule
    qualité. Dès que N/T dépasse 0,1 il devient mal conditionné ; dès que
    N > T il est singulier.

    On l'implémente quand même, parce qu'on ne peut pas comprendre
    pourquoi la contraction est nécessaire sans mesurer d'abord à quel
    point l'alternative est cassée.
    """
    x = _validate(returns)
    n_obs, n_assets = x.shape
    # rowvar=False : les colonnes sont les variables, les lignes les
    # observations. Inverser les deux est une erreur classique qui
    # produit une matrice de la mauvaise taille — ou pire, de la bonne
    # taille mais fausse, si N == T.
    cov = np.cov(x, rowvar=False, ddof=1)
    return CovarianceEstimate(
        matrix=np.atleast_2d(cov),
        assets=tuple(returns.columns),
        n_obs=n_obs,
        method="empirique",
        shrinkage=0.0,
    )


def ledoit_wolf(returns: pd.DataFrame) -> CovarianceEstimate:
    """
    Contraction de Ledoit-Wolf vers un multiple de l'identité.

    Référence : Ledoit & Wolf (2004), « A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices », Journal of Multivariate
    Analysis 88(2).

    L'ESTIMATEUR
    ------------
        Σ* = δ·μ·I + (1-δ)·S

    où μ = trace(S)/N est la valeur propre moyenne (la cible est donc
    « toutes les variances égales à la moyenne, aucune corrélation »)
    et δ est calculé pour minimiser l'erreur quadratique attendue.

    LA FORMULE DE δ, ET SON INTERPRÉTATION
    --------------------------------------
    Avec le produit scalaire ⟨A,B⟩ = trace(ABᵀ)/N :

        d² = ‖S - μI‖²          dispersion de S autour de la cible
        b̄² = (1/T²)·Σ_t ‖x_t x_tᵀ - S‖²   erreur d'estimation de S
        b²  = min(b̄², d²)       borné pour garantir δ ∈ [0,1]
        δ   = b²/d²

    Lis δ comme un rapport bruit/signal : au numérateur, l'imprécision de
    l'estimation empirique ; au dénominateur, la structure réellement
    présente dans les données. Quand les données sont abondantes, b² est
    petit, δ tend vers 0, on garde S. Quand elles sont rares, δ tend
    vers 1, on abandonne S au profit de la cible rigide.

    Aucun paramètre à régler, aucune validation croisée. C'est ce qui a
    fait le succès de cette méthode en production.
    """
    x = _validate(returns)
    n_obs, n_assets = x.shape

    if n_obs < 2:
        raise ValueError("au moins 2 observations requises")

    # Centrage : la covariance porte sur les écarts à la moyenne.
    x_c = x - x.mean(axis=0)

    # S avec normalisation 1/T (et non 1/(T-1)) : c'est la convention de
    # l'article, et mélanger les deux fausse le calcul de δ.
    s = (x_c.T @ x_c) / n_obs

    # Produit scalaire normalisé de Ledoit-Wolf : ⟨A,B⟩ = tr(ABᵀ)/N
    mu = np.trace(s) / n_assets

    delta_mat = s - mu * np.eye(n_assets)
    d2 = np.trace(delta_mat @ delta_mat.T) / n_assets

    # b̄² : dispersion des matrices de covariance instantanées autour de S.
    # Écriture vectorisée du Σ_t ‖x_t x_tᵀ - S‖².
    # ‖x_t x_tᵀ‖² = (x_tᵀx_t)² , et le terme croisé se simplifie.
    x2 = np.einsum("ij,ij->i", x_c, x_c)  # ‖x_t‖² pour chaque t
    b_bar2 = (np.sum(x2**2) / n_obs - np.trace(s @ s.T)) / (n_obs * n_assets)

    b2 = min(b_bar2, d2)
    shrinkage = 0.0 if d2 <= 0 else float(b2 / d2)
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    sigma = shrinkage * mu * np.eye(n_assets) + (1.0 - shrinkage) * s

    return CovarianceEstimate(
        matrix=sigma,
        assets=tuple(returns.columns),
        n_obs=n_obs,
        method="ledoit-wolf",
        shrinkage=shrinkage,
    )


def _validate(returns: pd.DataFrame) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            "attendu un DataFrame (une colonne par actif). Pour un actif "
            "unique, utilise annualized_volatility du module returns."
        )
    if returns.isna().to_numpy().any():
        raise ValueError(
            "valeurs manquantes dans les rendements. Les dates où un actif "
            "ne cote pas doivent être traitées explicitement — supprimer la "
            "ligne entière, ou reporter le dernier prix. Le choix change le "
            "résultat, donc il t'appartient."
        )
    x = returns.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("au moins 2 actifs requis pour une covariance")
    return x
