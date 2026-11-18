"""
L'objet Portefeuille — la pièce centrale de l'architecture.

POURQUOI CE MODULE EXISTE
=========================

Jusqu'à la séance 4, chaque fonction prenait un couple (poids, covariance)
passé séparément :

    decompose_risk(w, cov)
    portfolio_volatility(w, cov)
    minimum_variance_weights(cov)

Ça fonctionne, et c'est un mauvais découpage. Trois raisons :

1. **L'alignement est revalidé partout.** Chaque fonction doit vérifier
   que les poids et la covariance portent sur les mêmes actifs, dans le
   bon ordre. La même vérification est écrite cinq fois, donc elle finit
   par être oubliée une fois.

2. **Un couple n'a pas d'identité.** On ne peut pas nommer un
   portefeuille, le comparer à un autre, ni le transformer en gardant
   trace de ce qu'il était.

3. **Rien n'exprime l'invariant.** « Ces poids vont avec cette
   covariance » est une règle du domaine. Tant qu'elle vit dans des
   conventions d'appel, elle n'est pas appliquée.

Un objet valeur résout les trois : la validation a lieu **une fois**, à
la construction, et tout ce qui existe ensuite est cohérent par
construction.

LE CHOIX DE L'IMMUTABILITÉ
--------------------------
`frozen=True`. Aucune méthode ne modifie un portefeuille : les
transformations retournent un nouvel objet. C'est ce qui rend les
scénarios de stress composables sans effet de bord, et ce qui rend le
moteur utilisable depuis plusieurs fils d'exécution sans verrou.

CE QUE J'AI DÉLIBÉRÉMENT PAS FAIT : mettre en cache
---------------------------------------------------
`decompose()` recalcule à chaque appel. On pourrait mémoriser le
résultat, et la tentation est forte puisque l'objet est gelé — donc le
cache ne peut jamais être invalide.

Je ne l'ai pas fait, pour deux raisons.

D'abord, c'est une optimisation prématurée : décomposer un portefeuille
de sept lignes coûte quelques microsecondes. Optimiser ce qui n'est pas
mesuré comme lent est du travail perdu.

Ensuite, et c'est le vrai motif : poser un cache sur un dataclass gelé
oblige à passer par `object.__setattr__`, c'est-à-dire à contourner
explicitement la garantie d'immutabilité. On troque une propriété
vérifiable par le langage contre une convention que seul le prochain
lecteur fera respecter. Mauvais échange tant que la performance n'est
pas un problème démontré.

**Si elle le devenait** : je mettrais un `functools.lru_cache` sur une
fonction libre `_decompose(weights_tuple, cov_id)` plutôt que sur la
méthode, pour garder l'objet intact et rendre le cache testable
séparément.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .covariance import CovarianceEstimate
from .risk import RiskDecomposition, decompose_risk

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Portfolio:
    """
    Un portefeuille : des poids, un modèle de risque, une valeur, un nom.

    weights : pd.Series indexée par nom d'actif. Doit sommer à 1.
    covariance : le modèle de risque, portant sur les mêmes actifs.
    value : la valeur de marché. Sert aux scénarios de prix, qui la font
            bouger ; vaut 1 par défaut quand seuls les poids importent.
    name : utile dès qu'on compare un avant et un après.
    """

    weights: pd.Series
    covariance: CovarianceEstimate
    value: float = 1.0
    name: str = "portefeuille"

    # ── validation : une fois, ici, et plus jamais ailleurs ──────────
    def __post_init__(self) -> None:
        actifs = list(self.covariance.assets)

        manquants = set(actifs) - set(self.weights.index)
        if manquants:
            raise ValueError(
                f"poids absents pour {sorted(manquants)}. La covariance et "
                f"les poids doivent porter sur les mêmes actifs."
            )
        inconnus = set(self.weights.index) - set(actifs)
        if inconnus:
            raise ValueError(
                f"poids fournis pour des actifs absents de la covariance : "
                f"{sorted(inconnus)}."
            )

        # On réindexe sur l'ordre de la covariance. Supposer que les deux
        # sont déjà dans le même ordre est l'erreur la plus insidieuse du
        # domaine : elle produit un résultat plausible et complètement faux.
        aligne = self.weights.reindex(actifs).astype(float)
        if aligne.isna().any():
            raise ValueError("valeurs manquantes dans les poids")
        if not np.isclose(aligne.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"les poids somment à {aligne.sum():.6f}, attendu 1. "
                f"Pour modéliser du levier ou du cash, ajoute-les comme "
                f"positions explicites."
            )
        if self.value <= 0:
            raise ValueError(f"valeur du portefeuille non positive : {self.value}")

        object.__setattr__(self, "weights", aligne)

    # ── accès ────────────────────────────────────────────────────────
    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.covariance.assets)

    @property
    def n_assets(self) -> int:
        return len(self.covariance.assets)

    def decompose(self, periods_per_year: int = TRADING_DAYS_PER_YEAR
                  ) -> RiskDecomposition:
        """Décomposition d'Euler. Recalculée à chaque appel — voir l'en-tête."""
        return decompose_risk(self.weights, self.covariance, periods_per_year)

    @property
    def volatility(self) -> float:
        """Volatilité annualisée. Raccourci de lecture."""
        return self.decompose().portfolio_volatility

    # ── transformations : toujours un nouvel objet ───────────────────
    def with_weights(self, weights: pd.Series, name: str | None = None
                     ) -> Portfolio:
        """Mêmes actifs, même modèle de risque, autres poids."""
        return replace(self, weights=weights,
                       name=name or f"{self.name} (repondéré)")

    def with_covariance(self, covariance: CovarianceEstimate,
                        name: str | None = None) -> Portfolio:
        """Mêmes poids, autre modèle de risque. C'est la porte des scénarios."""
        return replace(self, covariance=covariance,
                       name=name or f"{self.name} (risque modifié)")

    def rescaled(self, value: float, name: str | None = None) -> Portfolio:
        """Même composition, autre valeur de marché."""
        return replace(self, value=value, name=name or self.name)

    def __str__(self) -> str:
        d = self.decompose()
        return (f"{self.name} — {self.n_assets} positions, "
                f"volatilité {d.portfolio_volatility:.2%}, "
                f"{d.effective_bets:.1f} paris effectifs")


def equal_weighted(covariance: CovarianceEstimate,
                   name: str = "équipondéré") -> Portfolio:
    """
    Le portefeuille 1/N, référence de comparaison obligatoire.

    DeMiguel, Garlappi & Uppal (2009) l'ont comparé à quatorze modèles
    d'optimisation : il les bat hors échantillon. Toute méthode inventée
    ici doit d'abord battre celle-ci.
    """
    n = len(covariance.assets)
    w = pd.Series(np.full(n, 1.0 / n),
                  index=pd.Index(covariance.assets, name="actif"))
    return Portfolio(weights=w, covariance=covariance, name=name)
