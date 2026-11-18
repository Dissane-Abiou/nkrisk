"""
Tests de scénario — imposer ce qui n'est pas encore arrivé.

LA LIMITE QUE CE MODULE LÈVE
============================

Volatilité, covariance, VaR, drawdown : toutes ces mesures lisent le
passé. Aucune ne voit un risque absent de l'échantillon. Une VaR
calibrée sur 2003-2007 n'avait jamais vu 2008 ; elle ne pouvait pas le
prédire, et ce n'était pas un défaut d'implémentation mais la nature
même d'un estimateur empirique.

Un test de scénario fait l'inverse : il n'estime rien, il **impose**.
« Si les volatilités doublaient et que toutes les corrélations montaient
à 0,9, que deviendrait mon risque ? » La question ne demande aucune
donnée nouvelle, seulement un modèle qu'on sait transformer.

LA DÉCISION D'ARCHITECTURE
==========================

Tout choc est une fonction **Portefeuille → Portefeuille**.

C'est la seule décision de ce module, et tout en découle :

  • **La composition est gratuite.** Enchaîner deux chocs, c'est composer
    deux fonctions. `Combined` se contente de les appliquer en séquence ;
    aucune hiérarchie de classes, aucun cas particulier.

  • **Rien n'est muté.** Un choc ne touche pas au portefeuille d'origine,
    il en produit un autre. On peut donc comparer l'avant et l'après, et
    rejouer dix scénarios sur le même portefeuille sans les contaminer
    entre eux.

  • **L'extension ne demande rien.** `Shock` est un `Protocol`, pas une
    classe de base. Un utilisateur écrit sa propre classe avec un attribut
    `name` et une méthode `apply`, sans importer quoi que ce soit d'ici.
    Le typage structurel dit : un choc est défini par ce qu'il FAIT, pas
    par ce dont il hérite.

  • **Les tests sont triviaux.** Une fonction pure d'un objet gelé vers
    un objet gelé se teste sans montage ni doublure.

LA CONTRAINTE QUI A DICTÉ L'IMPLÉMENTATION
==========================================

Une matrice de corrélation doit rester semi-définie positive. Sinon elle
ne décrit aucun monde possible, et √(wᵀΣw) peut devenir imaginaire.

La façon intuitive de « monter les corrélations » — ajouter un delta à
tous les coefficients hors diagonale — casse cette propriété. Mesuré sur
la matrice de démonstration :

    +0,1  →  valeur propre minimale  +0,018   valide
    +0,3  →  valeur propre minimale  −0,005   INVALIDE
    +0,5  →  valeur propre minimale  −0,016   INVALIDE

La méthode retenue est une **combinaison convexe** vers la matrice de
uns : C(α) = (1−α)·C + α·J. Elle est sûre par construction, et la preuve
tient en deux lignes : J est semi-définie positive (rang 1, valeurs
propres n et 0), une combinaison convexe de matrices SDP est SDP, et la
diagonale reste exactement 1. Vérifié jusqu'à α = 1.

Ce n'est pas un détail d'implémentation : c'est la contrainte qui a
choisi l'algorithme.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .covariance import CovarianceEstimate
from .portfolio import Portfolio


@runtime_checkable
class Shock(Protocol):
    """
    Un choc transforme un portefeuille en un autre.

    Typage STRUCTUREL volontaire : n'importe quelle classe portant un
    `name` et un `apply` est un choc, sans hériter de quoi que ce soit.
    Une classe de base abstraite forcerait l'utilisateur à importer notre
    module pour étendre le moteur ; le protocole ne demande rien.

    ⚠️ POURQUOI `name` EST UNE PROPRIÉTÉ ET NON UN ATTRIBUT ⚠️

    La version d'origine déclarait simplement `name: str`. Tous les tests
    passaient, `isinstance(choc, Shock)` renvoyait True — et c'était
    néanmoins faux.

    Un `name: str` nu dans un Protocol exige un attribut **modifiable**.
    Or toutes les implémentations sont des dataclasses gelées, donc leur
    `name` est en lecture seule : aucune ne satisfaisait réellement le
    contrat. Seul mypy l'a vu. `runtime_checkable` ne vérifie que
    l'EXISTENCE de l'attribut, jamais sa mutabilité — d'où un
    `isinstance` trompeusement vert.

    Déclarer `name` en propriété lecture seule exprime ce que le
    protocole demande vraiment : savoir se nommer, pas être renommable.

    C'est le meilleur argument que je connaisse en faveur du typage
    statique : il a trouvé une erreur de contrat qu'aucun test
    d'exécution ne pouvait attraper.
    """

    @property
    def name(self) -> str: ...

    def apply(self, portfolio: Portfolio) -> Portfolio: ...


# ══════════════════════════════════════════════════════════════════════
#  CHOCS SUR LE MODÈLE DE RISQUE
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VolatilityShock:
    """
    Multiplie les volatilités, en laissant les corrélations intactes.

    `multiplier` vaut soit un nombre appliqué à tout le portefeuille,
    soit un dictionnaire actif → facteur, les actifs absents restant
    inchangés.

    MISE EN ŒUVRE : on décompose Σ = D·C·D avec D = diag(σ), on multiplie
    D, on recompose. Passer par la corrélation garantit que seules les
    volatilités bougent — appliquer le facteur directement à Σ reviendrait
    à l'appliquer au CARRÉ, erreur classique et silencieuse.
    """

    multiplier: float | Mapping[str, float]
    name: str = "choc de volatilité"

    def __post_init__(self) -> None:
        if isinstance(self.multiplier, (int, float)):
            if self.multiplier <= 0:
                raise ValueError("le multiplicateur doit être strictement positif")
        else:
            for a, m in self.multiplier.items():
                if m <= 0:
                    raise ValueError(f"multiplicateur non positif pour {a} : {m}")

    def apply(self, portfolio: Portfolio) -> Portfolio:
        cov = portfolio.covariance
        actifs = list(cov.assets)

        if isinstance(self.multiplier, (int, float)):
            facteurs = np.full(len(actifs), float(self.multiplier))
        else:
            inconnus = set(self.multiplier) - set(actifs)
            if inconnus:
                raise ValueError(f"actifs inconnus dans le choc : {sorted(inconnus)}")
            facteurs = np.array([float(self.multiplier.get(a, 1.0)) for a in actifs])

        sd = np.sqrt(np.diag(cov.matrix))
        corr = cov.matrix / np.outer(sd, sd)
        np.fill_diagonal(corr, 1.0)
        sd_new = sd * facteurs
        nouvelle = np.outer(sd_new, sd_new) * corr
        np.fill_diagonal(nouvelle, sd_new**2)

        return portfolio.with_covariance(
            CovarianceEstimate(matrix=nouvelle, assets=cov.assets,
                               n_obs=cov.n_obs, method=f"{cov.method} + {self.name}",
                               shrinkage=cov.shrinkage),
            name=f"{portfolio.name} | {self.name}")


@dataclass(frozen=True)
class CorrelationShock:
    """
    Pousse les corrélations vers une cible, par combinaison convexe.

        C(α) = (1−α)·C + α·J      avec J rempli de `toward`, diagonale 1

    `intensity` = α ∈ [0,1]. À 0 rien ne change ; à 1 toutes les
    corrélations valent `toward`.

    POURQUOI CE CHOC EST LE PLUS IMPORTANT DU MODULE
    ------------------------------------------------
    En crise, les corrélations montent. Les actifs qui se compensaient en
    temps normal tombent ensemble. La diversification disparaît
    exactement au moment où on en a besoin — et aucune mesure calibrée
    sur une période calme ne peut l'anticiper, parce que le phénomène
    n'est pas dans les données.

    C'est le scénario que la volatilité seule ne voit jamais.

    ⚠️ LA BORNE SUR `toward`, ET LE TROU QU'ELLE A RÉVÉLÉ DANS MA PREUVE

    J'avais démontré que la combinaison convexe préserve la semi-définie
    positivité : J est SDP, C est SDP, une combinaison convexe de matrices
    SDP est SDP. La preuve est juste — et incomplète. Elle suppose que la
    CIBLE est SDP, ce que je n'ai jamais vérifié.

    Une matrice uniforme (1 sur la diagonale, ρ ailleurs) a pour valeurs
    propres 1+(n−1)ρ une fois et 1−ρ (n−1) fois. Elle n'est donc SDP que si

        ρ ≥ −1/(n−1)

    Pour 7 actifs la borne vaut −0,167 ; pour 20 actifs, −0,053. Un test
    balayant toute la plage a trouvé le cas `toward = −0,5`, où l'assertion
    interne a sauté — exactement comme elle devait.

    Cette borne n'est pas un artefact du code : c'est un fait financier.
    **On ne peut pas avoir beaucoup d'actifs tous fortement anticorrélés
    entre eux.** La diversification par corrélation négative est
    mathématiquement limitée, et d'autant plus que le portefeuille compte
    de lignes.

    La validation ne peut pas vivre dans `__post_init__`, parce que n n'y
    est pas connu : elle a lieu dans `apply`, où le portefeuille l'apporte.
    """

    intensity: float
    toward: float = 1.0
    name: str = "montée des corrélations"

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity doit être dans [0,1], reçu {self.intensity}")
        if not -1.0 <= self.toward <= 1.0:
            raise ValueError(f"toward doit être dans [-1,1], reçu {self.toward}")
        # La borne réelle dépend du nombre d'actifs : voir apply().

    def apply(self, portfolio: Portfolio) -> Portfolio:
        cov = portfolio.covariance
        n = len(cov.assets)

        sd = np.sqrt(np.diag(cov.matrix))
        corr = cov.matrix / np.outer(sd, sd)
        np.fill_diagonal(corr, 1.0)

        borne = -1.0 / (n - 1)
        if self.toward < borne - 1e-12:
            raise ValueError(
                f"toward={self.toward:.3f} est impossible pour {n} actifs : "
                f"une corrélation uniforme doit valoir au moins {borne:.4f} "
                f"(= −1/(n−1)) pour que la matrice reste semi-définie "
                f"positive. Aucun ensemble de {n} actifs ne peut être "
                f"mutuellement anticorrélé à ce point.")

        cible = np.full((n, n), float(self.toward))
        np.fill_diagonal(cible, 1.0)
        a = float(self.intensity)
        corr_new = (1.0 - a) * corr + a * cible
        np.fill_diagonal(corr_new, 1.0)

        nouvelle = np.outer(sd, sd) * corr_new
        np.fill_diagonal(nouvelle, sd**2)

        # La preuve mathématique garantit la validité ; on la vérifie quand
        # même. Un invariant qu'on croit tenir et qu'on ne teste pas est un
        # invariant qui finira par ne plus tenir.
        if np.linalg.eigvalsh(nouvelle).min() < -1e-10:
            raise AssertionError(
                "le choc a produit une covariance non semi-définie positive — "
                "impossible par construction, donc il y a un bug ici")

        return portfolio.with_covariance(
            CovarianceEstimate(matrix=nouvelle, assets=cov.assets,
                               n_obs=cov.n_obs, method=f"{cov.method} + {self.name}",
                               shrinkage=cov.shrinkage),
            name=f"{portfolio.name} | {self.name}")


# ══════════════════════════════════════════════════════════════════════
#  CHOC SUR LES PRIX
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PriceShock:
    """
    Applique des mouvements instantanés de prix.

    Deux effets, et le second est celui qu'on oublie :

    1. La valeur du portefeuille bouge de Σ wᵢ·rᵢ.
    2. Les POIDS dérivent. Après une chute de 40 % sur la tech et de 2 %
       sur les obligations, le portefeuille n'a plus la composition qu'on
       avait choisie — il est mécaniquement plus obligataire. Ignorer
       cette dérive fausse toute mesure de risque faite après le choc.

    C'est pour cela qu'un choc de prix retourne un Portefeuille et pas
    seulement un montant de perte : l'état d'après est une information
    en soi.
    """

    moves: Mapping[str, float]
    name: str = "choc de prix"

    def __post_init__(self) -> None:
        for a, r in self.moves.items():
            if r <= -1.0:
                raise ValueError(
                    f"mouvement de {r:.0%} sur {a} : une perte de 100 % ou plus "
                    f"rend le prix nul ou négatif, hors du domaine du modèle")

    def apply(self, portfolio: Portfolio) -> Portfolio:
        actifs = list(portfolio.assets)
        inconnus = set(self.moves) - set(actifs)
        if inconnus:
            raise ValueError(f"actifs inconnus dans le choc : {sorted(inconnus)}")

        r = np.array([float(self.moves.get(a, 0.0)) for a in actifs])
        w = portfolio.weights.to_numpy()

        rendement = float(w @ r)
        croissance = w * (1.0 + r)
        total = croissance.sum()
        if total <= 0:
            raise ValueError("le choc annule la valeur du portefeuille")

        nouveaux = pd.Series(croissance / total, index=portfolio.weights.index)
        return Portfolio(weights=nouveaux, covariance=portfolio.covariance,
                         value=portfolio.value * (1.0 + rendement),
                         name=f"{portfolio.name} | {self.name}")


@dataclass(frozen=True)
class Combined:
    """
    Plusieurs chocs appliqués en séquence.

    COMPOSITION, PAS HÉRITAGE. Cette classe ne spécialise rien : elle
    détient une liste et la parcourt. Ajouter un type de choc ne demande
    aucune modification ici, et `Combined` est lui-même un `Shock`, donc
    il se compose avec d'autres sans cas particulier.

    SUR L'ORDRE : TOUS LES CHOCS ACTUELS COMMUTENT

    J'avais d'abord écrit l'inverse — « un choc de prix ne commute avec
    aucun, parce qu'il fait dériver les poids ». Le test l'a démenti :
    écart exactement nul.

    La raison est une propriété de la conception, pas un hasard. L'état
    d'un portefeuille se sépare en deux parts disjointes :

        poids et valeur  ←  touchés par PriceShock seul
        covariance       ←  touchée par les chocs de risque seuls

    Deux transformations qui agissent sur des parts disjointes commutent
    nécessairement. C'est une bonne propriété : l'ordre d'un scénario
    composé ne peut pas produire de surprise.

    ⚠️ ELLE CESSERAIT DE TENIR si on ajoutait un choc COUPLÉ — par exemple
    « baisse de 3 écarts-types », dont l'ampleur dépend de la volatilité
    courante, ou un choc de corrélation dont l'intensité dépendrait des
    poids. Un tel choc lirait une part que l'autre modifie. Le jour où
    on en ajoute un, ce raisonnement tombe et il faudra documenter
    l'ordre explicitement.
    """

    shocks: Sequence[Shock]
    name: str = "scénario composé"

    def apply(self, portfolio: Portfolio) -> Portfolio:
        p = portfolio
        for s in self.shocks:
            p = s.apply(p)
        return p.with_covariance(p.covariance, name=f"{portfolio.name} | {self.name}")


# ══════════════════════════════════════════════════════════════════════
#  RÉSULTAT
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StressResult:
    """Avant, après, et ce qui a changé entre les deux."""

    scenario: str
    before: Portfolio
    after: Portfolio

    @property
    def pnl(self) -> float:
        """Perte ou gain immédiat, en proportion de la valeur initiale."""
        return self.after.value / self.before.value - 1.0

    @property
    def vol_before(self) -> float:
        return self.before.decompose().portfolio_volatility

    @property
    def vol_after(self) -> float:
        return self.after.decompose().portfolio_volatility

    @property
    def vol_ratio(self) -> float:
        return self.vol_after / self.vol_before

    @property
    def bets_before(self) -> float:
        return self.before.decompose().effective_bets

    @property
    def bets_after(self) -> float:
        return self.after.decompose().effective_bets

    @property
    def diversification_lost(self) -> float:
        """
        Part de la diversification effective détruite par le scénario.

        0 = rien perdu, 1 = il ne reste plus qu'un seul pari.
        C'est le chiffre à retenir d'un test de corrélation.
        """
        avant, apres = self.bets_before, self.bets_after
        if avant <= 1.0:
            return 0.0
        return float(min(1.0, max(0.0, (avant - apres) / (avant - 1.0))))

    def format(self, width: int = 34) -> str:
        nom = (self.scenario if len(self.scenario) <= width
               else self.scenario[: width - 1] + "…")
        return (f"{nom:<{width}}"
                f"P&L {self.pnl:>7.1%}   "
                f"vol {self.vol_before:>6.1%} → {self.vol_after:>6.1%} "
                f"({self.vol_ratio:>4.1f}×)   "
                f"paris {self.bets_before:>4.1f} → {self.bets_after:>4.1f}"
                f"   {self._divers_label()}")

    def _divers_label(self) -> str:
        """Perte ou gain de diversification, avec le bon signe.

        `diversification_lost` est bornée à [0,1] par définition : elle
        mesure une perte. Quand un choc AUGMENTE le nombre de paris —
        une décorrélation, une liquidation d'une poche concentrée — elle
        vaut 0, ce qui est correct mais muet. On affiche alors le gain.
        """
        avant, apres = self.bets_before, self.bets_after
        if apres > avant + 1e-9:
            return f"paris +{apres / avant - 1:>4.0%}"
        return f"divers. −{self.diversification_lost:>4.0%}"


def stress(portfolio: Portfolio, shock: Shock) -> StressResult:
    """Applique un choc et retourne la comparaison avant/après."""
    return StressResult(scenario=shock.name, before=portfolio,
                        after=shock.apply(portfolio))


def stress_table(portfolio: Portfolio, shocks: Sequence[Shock]) -> pd.DataFrame:
    """Plusieurs scénarios sur le même portefeuille, en tableau."""
    lignes = []
    for s in shocks:
        r = stress(portfolio, s)
        lignes.append({
            "scénario": s.name,
            "P&L": r.pnl,
            "vol avant": r.vol_before,
            "vol après": r.vol_after,
            "facteur": r.vol_ratio,
            "paris avant": r.bets_before,
            "paris après": r.bets_after,
            "diversification perdue": r.diversification_lost,
        })
    return pd.DataFrame(lignes).set_index("scénario")


# ══════════════════════════════════════════════════════════════════════
#  SCÉNARIOS NOMMÉS
# ══════════════════════════════════════════════════════════════════════
#
# ⚠️ CES SCÉNARIOS SONT STYLISÉS, PAS DES REJEUX HISTORIQUES ⚠️
#
# Les paramètres reproduisent des ORDRES DE GRANDEUR largement documentés
# — ampleur de la baisse actions, pic de volatilité, montée des
# corrélations — et non les vecteurs de rendement réellement observés.
#
# Un vrai rejeu historique chargerait les rendements datés de la fenêtre
# concernée et les appliquerait tels quels. L'architecture le permet sans
# rien changer : `HistoricalReplay` serait un `Shock` de plus, avec la
# même signature. Ce qui manque, ce sont les données, pas le code.
#
# Le distinguer explicitement n'est pas un scrupule : présenter un
# scénario stylisé comme un rejeu historique est exactement le genre
# d'approximation qui discrédite un moteur de risque en revue.


CRISE_2008 = Combined(
    [VolatilityShock(2.5, name="volatilité ×2,5"),
     CorrelationShock(0.70, toward=0.90, name="corrélations → 0,90")],
    name="crise type 2008")

CHOC_LIQUIDITE = Combined(
    [VolatilityShock(3.0, name="volatilité ×3"),
     CorrelationShock(0.85, toward=0.95, name="corrélations → 0,95")],
    name="choc de liquidité brutal")

HAUSSE_TAUX = Combined(
    [VolatilityShock({"OBLIG": 2.0}, name="volatilité obligataire ×2"),
     CorrelationShock(0.40, toward=0.60, name="corrélations → 0,60")],
    name="remontée brutale des taux")

DECORRELATION = CorrelationShock(0.80, toward=0.0,
                                 name="décorrélation totale (contrefactuel)")

SCENARIOS_STANDARD: tuple[Shock, ...] = (
    VolatilityShock(1.5, name="volatilité ×1,5"),
    VolatilityShock(2.0, name="volatilité ×2"),
    CorrelationShock(0.50, toward=0.90, name="corrélations à mi-chemin de 0,90"),
    CorrelationShock(1.00, toward=0.90, name="corrélations toutes à 0,90"),
    CRISE_2008,
    CHOC_LIQUIDITE,
    DECORRELATION,
)
