"""
Validation de modèles : mesurer si un modèle de risque dit la vérité.

CE QUE CE MODULE AJOUTE, ET POURQUOI IL ARRIVE MAINTENANT
---------------------------------------------------------
Les onze séances précédentes ont construit des estimateurs. Celle-ci
construit l'instrument qui les juge — et, accessoirement, qui juge
l'instrument lui-même.

Jusqu'ici le moteur savait répondre à « quel est le risque de ce
portefeuille ? ». Il ne savait pas répondre à « est-ce que mon modèle
de risque mérite qu'on le croie ? ». Ce sont deux questions
différentes, et la seconde est celle que pose un validateur de
modèles.

LA STATISTIQUE DE BIAIS
-----------------------
L'outil standard de la profession. On note, pour chaque période :

    z_t = r_t / σ̂_t

où r_t est le rendement réalisé du portefeuille sur la période t et
σ̂_t la volatilité que le modèle avait annoncée AVANT de voir r_t.

Si le modèle a raison, z_t est de variance 1. On mesure donc :

    b = √( (1/T) Σ z_t² )

    b ≈ 1   le modèle dit la vérité
    b > 1   il SOUS-ESTIME le risque — le sens dangereux
    b < 1   il surestime — coûteux, pas dangereux

POURQUOI LA MOYENNE QUADRATIQUE ET NON L'ÉCART-TYPE
---------------------------------------------------
La profession utilise généralement l'écart-type de z, qui retire la
moyenne d'échantillon. Ce module utilise la moyenne quadratique, et
c'est un choix, pas une négligence.

L'hypothèse nulle d'un modèle de variance est E[r²] = σ̂², c'est-à-dire
E[z²] = 1. Elle porte sur le moment d'ordre deux non centré. Retirer la
moyenne d'échantillon de z revient à tester Var(z) = 1, donc à estimer
une moyenne que l'hypothèse nulle déclare nulle — on dépense un degré
de liberté pour corriger un biais qui n'existe pas sous H₀, et on teste
au passage une hypothèse légèrement différente de celle qu'on croit
tester. L'écart est minuscule en pratique ; la clarté de l'énoncé ne
l'est pas.

CE QUI DISTINGUE CE MODULE DE `backtest.calibration()`
------------------------------------------------------
`BacktestResult.calibration()` (séance 8) calcule un rapport AGRÉGÉ :
la moyenne des prédictions divisée par la volatilité réalisée globale.
La statistique de biais standardise PÉRIODE PAR PÉRIODE, puis agrège.

Les deux coïncident quand la prédiction est constante. Dès qu'elle
varie, elles divergent — et c'est la même inégalité de Jensen qui avait
cassé `calibration_diagnosis` à la séance 9. La statistique de biais
est la mesure la plus fine des deux : elle crédite le modèle d'avoir vu
venir une période agitée, ce que le rapport agrégé ne peut pas faire.

⚠️ ATTENTION AU SENS. `calibration()` est prédit/réalisé ; la
statistique de biais est réalisé/prédit. Les deux sont environ
réciproques, et un lecteur pressé lira le mauvais sens du danger. C'est
pourquoi `BiasTest.direction` écrit la conclusion en mots.

L'INTERVALLE DE CONFIANCE, ET LÀ EST LA VRAIE DIFFICULTÉ
--------------------------------------------------------
La formule que publie la profession est :

    b est acceptable si  |b − 1| < 1,96 / √(2T)

Elle vient d'un calcul exact mais sous deux hypothèses qu'aucune série
financière ne vérifie : les z_t sont indépendants, et ils sont
gaussiens. Si l'une des deux tombe, l'intervalle est TROP ÉTROIT et le
test rejette des modèles corrects.

Ce module construit l'intervalle autrement, par blocs :

    1. on groupe les z² par bloc de `block_size` observations
    2. on calcule la moyenne de z² dans chaque bloc
    3. l'incertitude vient de la dispersion ENTRE blocs

Un bloc correspond à la durée de vie d'une prévision : si le modèle est
réestimé tous les vingt et un jours, les vingt et un z qui suivent
partagent la même erreur de prévision et ne sont pas vingt et une
informations indépendantes. Ce découpage traite donc d'un coup les deux
problèmes — les queues épaisses et la dépendance — sans avoir à
modéliser ni l'une ni l'autre.

À `block_size=1`, l'estimateur par blocs se réduit à la formule
classique corrigée du kurtosis :

    écart-type(b) ≈ b · √((κ̂ − 1) / T) / 2

au facteur √(T/(T−1)) près, qui est la correction de Bessel — on divise
par T−1 et non par T. Cette formule redonne 1/√(2T) quand κ̂ = 3, la
valeur gaussienne. La formule de la profession est donc le cas
particulier gaussien ET indépendant d'une formule plus générale.
`examples/seance12.py` mesure ce que coûte la confusion entre les deux.

LE VERDICT, ET POURQUOI IL A TROIS NIVEAUX ET NON DEUX
-------------------------------------------------------
Un test d'hypothèse répond par oui ou non à « ce modèle est-il
distinguable d'un modèle correct ? ». Ce n'est pas la question d'un
validateur, qui doit répondre à « peut-on s'en servir ? ».

Avec assez d'observations, tout modèle devient statistiquement
distinguable d'un modèle parfait, parce qu'aucun modèle n'est parfait.
Un verdict binaire rejette donc tout, à condition d'attendre assez
longtemps. Il faut donc croiser deux critères de natures différentes :

    VALIDÉ        1 est dans l'intervalle — indistinguable d'un
                  modèle correct
    SURVEILLANCE  1 est hors de l'intervalle, mais l'écart reste
                  sous le seuil de matérialité — détectablement
                  faux, pas dangereusement
    REJETÉ        hors intervalle ET au-delà du seuil
    INAPPLICABLE  le modèle a refusé de s'estimer sur ces données

C'est la proportionnalité qu'exige la ligne directrice E-23 du BSIF :
l'intensité de la validation suit l'importance du risque, et un écart
statistiquement réel mais économiquement nul ne justifie pas de
retirer un modèle.

Références
----------
Ligne directrice E-23 du BSIF, *Gestion du risque de modélisation*, en
vigueur le 1er mai 2027 pour toutes les institutions financières
fédérales.
Sur la statistique de biais et son usage comme instrument de
surveillance d'un modèle de risque : la littérature des fournisseurs de
modèles à facteurs, où elle est l'indicateur de référence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from .covariance import CovarianceEstimate, ledoit_wolf, sample_covariance
from .ewma import DECAY_RISKMETRICS, ewma_covariance
from .factors import InsufficientStructure, pca_factor_model
from .types import Estimate, InsufficientData

try:  # pragma: no cover - dépend de la version de Python
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import (  # type: ignore[assignment]
        Protocol,
        runtime_checkable,
    )


TRADING_DAYS_PER_YEAR = 252

#: Seuil de matérialité par défaut : un modèle qui se trompe de moins de
#: dix pour cent sur l'échelle du risque est faux mais utilisable. Le
#: chiffre est une convention, pas un résultat — il est donc un paramètre.
MATERIALITY_DEFAULT = 0.10

#: Nombre minimal de blocs pour que la dispersion entre blocs ait un
#: sens. Avec un seul bloc il n'y a aucune dispersion à mesurer ; avec
#: deux, l'estimation est si bruitée qu'elle ne vaut rien. On exige
#: davantage, et on le dit plutôt que de retourner un intervalle faux.
MIN_BLOCKS = 5

VALIDE = "VALIDÉ"
SURVEILLANCE = "SURVEILLANCE"
REJETE = "REJETÉ"
INAPPLICABLE = "INAPPLICABLE"


# ══════════════════════════════════════════════════════════════════════
#  Le contrat : ce qu'est un modèle de covariance
# ══════════════════════════════════════════════════════════════════════


@runtime_checkable
class CovarianceModel(Protocol):
    """
    Un modèle de covariance est tout objet qui porte un nom et sait
    estimer une covariance à partir d'un historique.

    CINQUIÈME USAGE DU TYPAGE STRUCTUREL DANS CE PROJET, et le premier
    qui serve à comparer des implémentations entre elles plutôt qu'à
    les étendre.

    Les estimateurs du moteur sont des fonctions libres de signatures
    légèrement différentes : `sample_covariance(returns)`,
    `ewma_covariance(returns, decay=...)`,
    `pca_factor_model(returns).covariance()`. Aucune hiérarchie de
    classes ne les relie, et il n'y a aucune raison d'en créer une.

    Le Protocol les rend comparables sans les toucher : les
    adaptateurs ci-dessous sont des enveloppes de quelques lignes, et
    l'inventaire les traite indifféremment.
    """

    @property
    def name(self) -> str: ...

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate: ...


@dataclass(frozen=True)
class SampleModel:
    """Covariance empirique, sans contraction. Le témoin négatif."""

    name: str = "empirique"

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        return sample_covariance(returns)


@dataclass(frozen=True)
class LedoitWolfModel:
    """Contraction de Ledoit-Wolf (séance 2)."""

    name: str = "Ledoit-Wolf"

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        return ledoit_wolf(returns)


@dataclass(frozen=True)
class EwmaModel:
    """Pondération exponentielle (séance 11)."""

    decay: float = DECAY_RISKMETRICS
    shrinkage: float | None = None

    @property
    def name(self) -> str:
        return f"EWMA λ={self.decay:g}"

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        return ewma_covariance(returns, decay=self.decay, shrinkage=self.shrinkage)


@dataclass(frozen=True)
class FactorCovariance:
    """
    Modèle à facteurs statistiques (séance 7).

    `n_factors=None` laisse le critère de Marchenko-Pastur choisir.
    """

    n_factors: int | None = None

    @property
    def name(self) -> str:
        if self.n_factors is None:
            return "facteurs (Marchenko-Pastur)"
        return f"facteurs k={self.n_factors}"

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        return pca_factor_model(returns, n_factors=self.n_factors).covariance()


# ══════════════════════════════════════════════════════════════════════
#  La statistique de biais
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BiasTest:
    """
    Résultat complet d'un test de biais.

    Porte l'intervalle robuste ET l'intervalle naïf, parce que l'écart
    entre les deux est lui-même une information : il dit de combien la
    formule de la profession se trompe sur ces données-là.
    """

    bias: Estimate
    naive_ci: tuple[float, float]
    kurtosis: float
    n_blocks: int
    block_size: int
    materiality: float
    #: Écart-type observé des moyennes de z² entre blocs. Sert à
    #: `dispersion_ratio` ; stocké plutôt que recalculé parce que
    #: l'objet est gelé et ne garde pas les données.
    _block_sd: float = 0.0

    @property
    def verdict(self) -> str:
        dans_intervalle = self.bias.ci_low <= 1.0 <= self.bias.ci_high
        if dans_intervalle:
            return VALIDE
        if abs(self.bias.value - 1.0) < self.materiality:
            return SURVEILLANCE
        return REJETE

    @property
    def naive_verdict(self) -> str:
        """
        Le verdict qu'aurait rendu la formule classique.

        Sert uniquement à mesurer le désaccord entre les deux
        intervalles. Ce n'est pas le verdict du module.
        """
        bas, haut = self.naive_ci
        if bas <= 1.0 <= haut:
            return VALIDE
        if abs(self.bias.value - 1.0) < self.materiality:
            return SURVEILLANCE
        return REJETE

    @property
    def direction(self) -> str:
        """La conclusion en mots, pour qu'on ne lise pas le sens à l'envers."""
        if self.verdict == VALIDE:
            return "risque annoncé indistinguable du risque subi"
        if self.bias.value > 1.0:
            ecart = (self.bias.value - 1.0) * 100.0
            return f"SOUS-ESTIME le risque de {ecart:.1f} % — sens dangereux"
        ecart = (1.0 - self.bias.value) * 100.0
        return f"surestime le risque de {ecart:.1f} % — prudent, donc coûteux"

    @property
    def dispersion_ratio(self) -> float:
        """
        La STABILITÉ de la calibration, là où `bias` mesure son NIVEAU.

        Sous l'hypothèse nulle et pour des z gaussiens indépendants, la
        moyenne de z² sur un bloc de k observations a un écart-type de
        √(2/k). On rapporte la dispersion observée entre blocs à cette
        valeur théorique.

            ≈ 1  le modèle se trompe autant que le hasard l'impose
            > 1  le modèle est juste EN MOYENNE mais faux PÉRIODE PAR
                 PÉRIODE — il annonce trop peu quand ça bouge et trop
                 quand c'est calme, et les deux erreurs s'annulent

        POURQUOI CETTE SECONDE MESURE EST NÉCESSAIRE
        --------------------------------------------
        Un biais de 1,00 ne veut pas dire qu'un modèle est bon. Il veut
        dire que ses erreurs se compensent sur la fenêtre mesurée. Un
        modèle qui annonce 10 % quand le risque est de 20 % et 20 %
        quand il est de 10 % obtient un biais parfait et n'a jamais eu
        raison une seule fois.

        C'est la même distinction que `calibration_diagnosis` avait
        établie à la séance 9 en découvrant que l'écart de calibration
        sur données réelles était CONCENTRÉ : un chiffre agrégé peut
        être juste et cacher tout ce qui compte.
        """
        attendu = math.sqrt(2.0 / self.block_size)
        if attendu <= 0.0:
            return math.nan
        return self._block_sd / attendu

    @property
    def interval_ratio(self) -> float:
        """
        De combien l'intervalle robuste est plus large que le naïf.

        Supérieur à 1 : la formule classique est trop confiante, et
        elle rejette des modèles corrects.
        """
        largeur_naive = self.naive_ci[1] - self.naive_ci[0]
        if largeur_naive <= 0.0:
            return math.inf
        return self.bias.half_width * 2.0 / largeur_naive

    def __str__(self) -> str:
        return (
            f"biais {self.bias.value:.3f} "
            f"[{self.bias.ci_low:.3f} — {self.bias.ci_high:.3f}] "
            f"→ {self.verdict} ({self.direction})"
        )


def bias_statistic(
    realized: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    block_size: int = 1,
    confidence: float = 0.95,
    materiality: float = MATERIALITY_DEFAULT,
) -> BiasTest:
    """
    Teste si un modèle de volatilité dit la vérité sur son propre risque.

    Paramètres
    ----------
    realized : rendements réalisés, une valeur par période.
    predicted : volatilité annoncée AVANT chaque période, même longueur,
        même échelle temporelle que `realized`. Un mélange d'échelles —
        rendement quotidien contre volatilité annualisée — produit un
        biais d'un facteur √252 et aucune erreur : c'est la raison pour
        laquelle les deux séries sont passées ensemble et non
        séparément.
    block_size : nombre d'observations couvertes par une même prévision.
        Mettre la période de réestimation du modèle. À 1, l'intervalle
        se réduit à la formule classique corrigée du kurtosis.
    confidence : niveau de l'intervalle.
    materiality : seuil au-delà duquel un écart détecté devient un rejet.

    Lève
    ----
    InsufficientData
        S'il n'y a pas assez de blocs pour que la dispersion entre
        blocs soit mesurable, ou si une volatilité annoncée est nulle
        ou négative. Ce dernier cas mérite une exception et non un
        NaN : une prévision de risque nulle n'est pas une petite
        prévision, c'est un bug en amont.
    """
    r = np.asarray(realized, dtype=float).ravel()
    s = np.asarray(predicted, dtype=float).ravel()

    if r.shape != s.shape:
        raise ValueError(
            f"realized et predicted doivent avoir la même longueur : "
            f"{r.shape[0]} contre {s.shape[0]}"
        )
    if block_size < 1:
        raise ValueError(f"block_size doit être ≥ 1, reçu {block_size}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence doit être dans (0,1), reçu {confidence}")

    # Deux refus de natures différentes, donc deux exceptions différentes.
    # `InsufficientData` parle d'un MANQUE d'observations ; un NaN au
    # milieu d'une série complète, ou une prévision nulle, ne sont pas des
    # manques mais des défauts, et les confondre brouillerait le
    # diagnostic de l'appelant.
    fini = np.isfinite(r) & np.isfinite(s)
    if not fini.all():
        # On ne jette pas les points silencieusement : un trou dans la
        # série de prévisions est une information sur le modèle, pas un
        # détail de nettoyage.
        raise ValueError(
            f"{int((~fini).sum())} période(s) sur {r.size} portent un NaN ou "
            "un infini. Un trou dans la série de prévisions doit être "
            "expliqué, pas écarté."
        )
    if (s <= 0.0).any():
        raise ValueError(
            f"{int((s <= 0.0).sum())} volatilité(s) annoncée(s) nulle(s) ou "
            "négative(s). Une prévision de risque nulle n'est pas une petite "
            "prévision : c'est un bug en amont."
        )

    z2 = (r / s) ** 2
    n_total = z2.size
    n_blocks = n_total // block_size

    if n_blocks < MIN_BLOCKS:
        raise InsufficientData(
            n_obs=n_blocks,
            n_required=MIN_BLOCKS,
            what=(
                f"statistique de biais par blocs de {block_size} "
                f"(il faudrait {MIN_BLOCKS * block_size} observations, "
                f"ou un block_size plus petit)"
            ),
        )

    # On tronque au multiple inférieur plutôt que de compléter le dernier
    # bloc : un bloc incomplet porterait moins de poids mais compterait
    # autant dans la dispersion entre blocs, ce qui gonflerait
    # artificiellement l'intervalle.
    utilise = n_blocks * block_size
    blocs = z2[:utilise].reshape(n_blocks, block_size)
    moyennes_bloc = blocs.mean(axis=1)

    m2 = float(moyennes_bloc.mean())
    sd_m2 = float(moyennes_bloc.std(ddof=1) / math.sqrt(n_blocks))

    b = math.sqrt(m2)

    # Delta method : b = √m2, donc écart-type(b) = écart-type(m2) / (2√m2).
    sd_b = sd_m2 / (2.0 * b) if b > 0.0 else math.inf

    z_crit = float(stats.norm.ppf(0.5 + confidence / 2.0))
    demi = z_crit * sd_b

    # Une volatilité relative négative n'a pas de sens : on tronque la
    # borne basse à zéro et on le déclare, plutôt que d'afficher un
    # intervalle qui contient des valeurs impossibles.
    ci_low = max(0.0, b - demi)
    ci_high = b + demi

    # Kurtosis des z, utile au diagnostic et à la formule naïve.
    z2_centre = z2[:utilise]
    m2_brut = float(z2_centre.mean())
    kurt = float((z2_centre**2).mean() / m2_brut**2) if m2_brut > 0.0 else math.nan

    # La formule de la profession : gaussienne et indépendante.
    sd_naif = b * math.sqrt(2.0 / n_total) / 2.0
    naive_ci = (max(0.0, b - z_crit * sd_naif), b + z_crit * sd_naif)

    hypotheses = [
        f"un bloc de {block_size} observation(s) = une prévision",
        "l'incertitude vient de la dispersion ENTRE blocs, pas en leur sein",
        "aucune hypothèse de normalité ni d'indépendance en deçà du bloc",
    ]
    if block_size == 1:
        hypotheses.append(
            "block_size=1 : formule classique corrigée du kurtosis, "
            "à la correction de Bessel près"
        )

    estimation = Estimate(
        value=b,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_obs=n_total,
        method=f"statistique de biais, intervalle par {n_blocks} blocs",
        assumptions=tuple(hypotheses),
    )

    return BiasTest(
        bias=estimation,
        naive_ci=naive_ci,
        kurtosis=kurt,
        n_blocks=n_blocks,
        block_size=block_size,
        materiality=materiality,
        _block_sd=float(moyennes_bloc.std(ddof=1)),
    )


# ══════════════════════════════════════════════════════════════════════
#  La fiche de modèle
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelCard:
    """
    Le dossier de validation d'un modèle, en un objet.

    C'est la pièce qu'un inventaire de modèles doit contenir pour
    chaque modèle en service : ce qu'il est, sur quoi il a été mesuré,
    ce que la mesure a donné, et sous quelles conditions le résultat
    vaut quelque chose.

    POURQUOI UNE FICHE PLUTÔT QU'UN NOMBRE
    --------------------------------------
    Un validateur ne transmet pas « biais = 1,07 ». Il transmet le
    chiffre, la fenêtre sur laquelle il a été calculé, le portefeuille
    de test, le nombre de prévisions distinctes, l'état de
    conditionnement de la matrice, et la liste de ce qui n'a pas été
    vérifié. Sans ce contexte, le chiffre est inexploitable — et
    surtout, il est irréfutable, ce qui est pire.
    """

    model: str
    verdict: str
    n_obs: int
    n_forecasts: int
    window: tuple[str, str]
    bias: BiasTest | None = None
    condition_number: float | None = None
    ratio_n_over_t: float | None = None
    reason: str = ""
    limits: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        return self.verdict in (VALIDE, SURVEILLANCE)

    @property
    def bias_value(self) -> float:
        return math.nan if self.bias is None else self.bias.bias.value

    def row(self) -> dict[str, object]:
        """Une ligne de tableau, pour l'inventaire."""
        return {
            "modèle": self.model,
            "biais": self.bias_value,
            "IC bas": math.nan if self.bias is None else self.bias.bias.ci_low,
            "IC haut": math.nan if self.bias is None else self.bias.bias.ci_high,
            "dispersion": math.nan
            if self.bias is None
            else self.bias.dispersion_ratio,
            "kurtosis z": math.nan if self.bias is None else self.bias.kurtosis,
            "conditionnement": math.nan
            if self.condition_number is None
            else self.condition_number,
            "N/T": math.nan if self.ratio_n_over_t is None else self.ratio_n_over_t,
            "prévisions": self.n_forecasts,
            "verdict": self.verdict,
        }

    def __str__(self) -> str:
        lignes = [
            f"┌─ FICHE DE MODÈLE — {self.model}",
            f"│  verdict        {self.verdict}",
            f"│  fenêtre        {self.window[0]} → {self.window[1]}",
            f"│  observations   {self.n_obs} "
            f"({self.n_forecasts} prévisions distinctes)",
        ]
        if self.bias is not None:
            b = self.bias
            lignes += [
                f"│  biais          {b.bias.value:.3f} "
                f"[{b.bias.ci_low:.3f} — {b.bias.ci_high:.3f}]",
                f"│  conclusion     {b.direction}",
                f"│  dispersion     {b.dispersion_ratio:.2f} × le hasard "
                f"(1,00 = aussi stable que possible ; au-delà, juste en "
                f"moyenne et faux période par période)",
                f"│  kurtosis de z  {b.kurtosis:.2f} "
                f"(3,00 = gaussien ; au-delà, la formule classique ment)",
                f"│  IC classique   [{b.naive_ci[0]:.3f} — {b.naive_ci[1]:.3f}] "
                f"→ {b.naive_verdict}",
            ]
        if self.condition_number is not None:
            lignes.append(f"│  conditionnement {self.condition_number:.4g}")
        if self.reason:
            lignes.append(f"│  motif          {self.reason}")
        for limite in self.limits:
            lignes.append(f"│  ⚠ {limite}")
        lignes.append("└─")
        return "\n".join(lignes)


@dataclass(frozen=True)
class ModelInventory:
    """
    L'inventaire : tous les modèles en service, avec leur verdict.

    C'est la pièce centrale que demande E-23 — la liste des modèles
    porteurs de risque, chacun avec sa validation et sa cote. Le point
    important n'est pas la liste : c'est qu'elle soit PRODUITE par la
    mesure, et donc qu'elle ne puisse pas être flatteuse.
    """

    cards: tuple[ModelCard, ...]
    universe: str = ""

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([c.row() for c in self.cards])

    @property
    def n_rejected(self) -> int:
        return sum(1 for c in self.cards if c.verdict == REJETE)

    @property
    def n_inapplicable(self) -> int:
        return sum(1 for c in self.cards if c.verdict == INAPPLICABLE)

    def usable(self) -> tuple[ModelCard, ...]:
        return tuple(c for c in self.cards if c.usable)

    def best(self) -> ModelCard | None:
        """
        Le modèle utilisable dont le biais est le plus proche de 1.

        Volontairement pas « le meilleur modèle » : c'est le plus
        honnête sur ce portefeuille de test et cette fenêtre. Un autre
        portefeuille peut donner un autre gagnant, et la séance 12 le
        montre.
        """
        candidats = [c for c in self.usable() if c.bias is not None]
        if not candidats:
            return None
        return min(candidats, key=lambda c: abs(c.bias_value - 1.0))

    def __str__(self) -> str:
        entete = "INVENTAIRE DE MODÈLES"
        if self.universe:
            entete += f" — {self.universe}"
        lignes = [entete, "=" * len(entete), ""]
        lignes.append(
            f"{'modèle':<30}{'biais':>8}{'IC 95 %':>20}{'disp.':>8}{'verdict':>16}"
        )
        lignes.append("-" * 82)
        for c in self.cards:
            if c.bias is None:
                lignes.append(
                    f"{c.model:<30}{'—':>8}{'—':>20}{'—':>8}{c.verdict:>16}"
                )
                continue
            ic = f"[{c.bias.bias.ci_low:.2f} — {c.bias.bias.ci_high:.2f}]"
            lignes.append(
                f"{c.model:<30}{c.bias_value:>8.3f}{ic:>20}"
                f"{c.bias.dispersion_ratio:>8.2f}{c.verdict:>16}"
            )
        lignes.append("")
        meilleur = self.best()
        if meilleur is not None:
            lignes.append(f"Le plus honnête ici : {meilleur.model}")
        if self.n_rejected:
            lignes.append(f"⚠ {self.n_rejected} modèle(s) rejeté(s).")
        if self.n_inapplicable:
            lignes.append(
                f"⚠ {self.n_inapplicable} modèle(s) inapplicable(s) sur ces données."
            )
        return "\n".join(lignes)


# ══════════════════════════════════════════════════════════════════════
#  Le protocole de validation
# ══════════════════════════════════════════════════════════════════════


def validate_covariance_model(
    returns: pd.DataFrame,
    model: CovarianceModel,
    *,
    lookback: int = TRADING_DAYS_PER_YEAR,
    step: int = 21,
    weights: pd.Series | None = None,
    portfolio: str = "fixed",
    confidence: float = 0.95,
    materiality: float = MATERIALITY_DEFAULT,
) -> ModelCard:
    """
    Valide un modèle de covariance par rejeu causal, et rend sa fiche.

    LA PROCÉDURE
    ------------
    Pour chaque date de réestimation t :

        historique = returns.iloc[t - lookback : t]     ← t EXCLU
        Σ̂ = model.estimate(historique)
        σ̂ = √(wᵀ Σ̂ w)

    puis, pour chacune des `step` périodes suivantes, le rendement
    réalisé du portefeuille est comparé à cette même σ̂. On obtient
    `step` observations par prévision, et l'intervalle est construit
    par blocs de cette taille — exactement la dépendance que la formule
    classique ignore.

    LES DEUX PROTOCOLES, ET POURQUOI IL EN FAUT DEUX
    ------------------------------------------------
    `portfolio="fixed"` (défaut) : les poids sont constants
    (équipondérés sauf indication) et ne dépendent jamais du modèle.
    C'est ce qui rend les modèles COMPARABLES — si chaque modèle
    choisissait son propre portefeuille, l'écart mesuré mélangerait la
    qualité de la covariance et celle de l'optimisation.

    `portfolio="model"` : à chaque réestimation, les poids sont ceux de
    la variance minimale calculée sur la matrice DU MODÈLE,

        w = Σ̂⁻¹1 / (1ᵀΣ̂⁻¹1)

    et la prévision porte sur ce portefeuille-là. C'est la question que
    se pose réellement un utilisateur : « si je suis les conseils de ce
    modèle, me dit-il la vérité sur le risque qu'il me fait prendre ? »

    ⚠️ LE PROTOCOLE FIXE A UN ANGLE MORT, ET IL FAUT LE NOMMER

    Pour un portefeuille FIXE, la prévision est wᵀΣ̂w — une combinaison
    linéaire des entrées de la matrice. La covariance empirique étant
    sans biais entrée par entrée, cette combinaison est sans biais
    aussi, même quand la matrice est atrocement conditionnée. Le
    protocole fixe déclare donc « VALIDÉ » un estimateur dont la
    séance 2 avait démontré qu'il était inutilisable.

    Ce n'est pas une contradiction : l'inversion est ce qui amplifie le
    bruit, et un portefeuille fixe n'inverse rien. Le protocole fixe ne
    peut pas voir un défaut qui n'apparaît qu'à l'inversion.

    `portfolio="model"` inverse, et le défaut réapparaît.
    `examples/seance12.py` montre les deux verdicts côte à côte sur le
    même modèle et les mêmes données. La leçon porte au-delà de la
    finance : **un angle mort est une propriété du protocole de
    validation, pas du modèle validé.** Un rapport de validation qui
    ne dit pas ce qu'il ne peut pas voir est pire qu'absent, parce
    qu'il rassure.

    POURQUOI CERTAINES EXCEPTIONS SONT ATTRAPÉES ET PAS LES AUTRES
    --------------------------------------------------------------
    `InsufficientData`, `InsufficientStructure` et `LinAlgError`
    signifient « ce modèle refuse de s'estimer sur ces données », ce
    qui est un RÉSULTAT de validation : la fiche le consigne en
    INAPPLICABLE. Toute autre exception est un bug, et elle remonte.

    Attraper `Exception` ici aurait transformé n'importe quelle faute de
    frappe en « modèle inapplicable » — c'est la même erreur que le
    `pytest.raises(Exception)` de la séance 8, dont ce projet a déjà
    payé le prix une fois.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns doit être un DataFrame, reçu {type(returns)}")
    if returns.shape[1] < 2:
        raise ValueError(
            f"il faut au moins deux actifs pour valider une covariance, "
            f"reçu {returns.shape[1]}"
        )
    if lookback < 2:
        raise ValueError(f"lookback doit être ≥ 2, reçu {lookback}")
    if step < 1:
        raise ValueError(f"step doit être ≥ 1, reçu {step}")
    if portfolio not in ("fixed", "model"):
        raise ValueError(
            f"portfolio doit être 'fixed' ou 'model', reçu {portfolio!r}"
        )
    if portfolio == "model" and weights is not None:
        raise ValueError(
            "portfolio='model' calcule ses propres poids : passer `weights` "
            "en même temps est contradictoire."
        )

    n_periodes, n_actifs = returns.shape
    if n_periodes <= lookback:
        raise InsufficientData(
            n_obs=n_periodes,
            n_required=lookback + step,
            what=(
                f"validation du modèle '{model.name}' "
                f"(lookback {lookback} : rien ne reste après la première "
                "estimation)"
            ),
        )

    if weights is None:
        w = pd.Series(1.0 / n_actifs, index=returns.columns)
    else:
        w = weights.reindex(returns.columns)
        if w.isna().any():
            manquants = sorted(set(returns.columns) - set(weights.index))
            raise ValueError(
                f"poids manquants pour : {manquants}. Les poids doivent "
                "couvrir exactement les colonnes de returns."
            )

    matrice_rendements = returns.to_numpy()
    rendements_fixes = matrice_rendements @ w.to_numpy()

    fenetre = (
        str(returns.index[0].date())
        if isinstance(returns.index, pd.DatetimeIndex)
        else str(returns.index[0]),
        str(returns.index[-1].date())
        if isinstance(returns.index, pd.DatetimeIndex)
        else str(returns.index[-1]),
    )

    realises: list[float] = []
    predites: list[float] = []
    conditionnements: list[float] = []
    n_previsions = 0
    dernier_motif = ""

    for t in range(lookback, n_periodes, step):
        historique = returns.iloc[t - lookback : t]  # ← t EXCLU. Causalité.

        try:
            estimation = model.estimate(historique)
        except (InsufficientData, InsufficientStructure, np.linalg.LinAlgError) as exc:
            dernier_motif = f"{type(exc).__name__}: {exc}"
            continue

        if portfolio == "fixed":
            poids_t = w.to_numpy()
            rendements_t = rendements_fixes
        else:
            # Variance minimale sur la matrice DU MODÈLE. L'inversion est
            # ce qui révèle le mauvais conditionnement, et c'est tout
            # l'intérêt de ce protocole.
            try:
                inv_un = np.linalg.solve(
                    estimation.matrix, np.ones(estimation.n_assets)
                )
            except np.linalg.LinAlgError as exc:
                dernier_motif = f"matrice non inversible : {exc}"
                continue
            denominateur = float(inv_un.sum())
            if not np.isfinite(denominateur) or abs(denominateur) < 1e-12:
                dernier_motif = "1ᵀΣ⁻¹1 nul ou non fini — matrice dégénérée"
                continue
            poids_t = inv_un / denominateur
            rendements_t = matrice_rendements @ poids_t

        variance = float(poids_t @ estimation.matrix @ poids_t)
        if variance <= 0.0:
            dernier_motif = (
                "variance de portefeuille nulle ou négative — matrice non PSD"
            )
            continue

        sigma = math.sqrt(variance)
        conditionnements.append(estimation.condition_number)
        n_previsions += 1

        fin = min(t + step, n_periodes)
        for s in range(t, fin):
            realises.append(float(rendements_t[s]))
            predites.append(sigma)

    if portfolio == "fixed":
        angle_mort = (
            "portefeuille FIXE : la prévision wᵀΣw n'inverse pas la "
            "matrice, donc ce protocole ne peut pas voir le mauvais "
            "conditionnement — comparer avec portfolio='model'"
        )
    else:
        angle_mort = (
            "portefeuille de variance minimale DU MODÈLE : mélange "
            "délibérément qualité de la covariance et effet de l'inversion"
        )

    limites = (
        angle_mort,
        "un biais de 1,00 dit que les erreurs se compensent, pas qu'il "
        "n'y en a pas — lire aussi dispersion_ratio",
        "une seule fenêtre et un seul univers — un autre peut inverser le classement",
        f"horizon d'une période, réestimation tous les {step}",
    )

    if n_previsions == 0:
        return ModelCard(
            model=model.name,
            verdict=INAPPLICABLE,
            n_obs=0,
            n_forecasts=0,
            window=fenetre,
            reason=dernier_motif or "aucune estimation n'a abouti",
            limits=limites,
        )

    try:
        test = bias_statistic(
            np.asarray(realises),
            np.asarray(predites),
            block_size=step,
            confidence=confidence,
            materiality=materiality,
        )
    except InsufficientData as exc:
        return ModelCard(
            model=model.name,
            verdict=INAPPLICABLE,
            n_obs=len(realises),
            n_forecasts=n_previsions,
            window=fenetre,
            reason=str(exc),
            limits=limites,
        )

    conditionnement_median = float(np.median(conditionnements))

    return ModelCard(
        model=model.name,
        verdict=test.verdict,
        n_obs=len(realises),
        n_forecasts=n_previsions,
        window=fenetre,
        bias=test,
        condition_number=conditionnement_median,
        ratio_n_over_t=n_actifs / lookback,
        reason=dernier_motif,
        limits=limites,
    )


def build_inventory(
    returns: pd.DataFrame,
    models: list[CovarianceModel] | tuple[CovarianceModel, ...] | None = None,
    *,
    universe: str = "",
    **kwargs: object,
) -> ModelInventory:
    """
    Passe plusieurs modèles dans le MÊME protocole et rend l'inventaire.

    `models=None` prend les quatre estimateurs du moteur. Les mêmes
    données, la même fenêtre, le même portefeuille de test, le même
    intervalle : c'est la seule façon de comparer des modèles sans
    que le protocole favorise l'un d'eux.
    """
    if models is None:
        models = (
            SampleModel(),
            LedoitWolfModel(),
            EwmaModel(),
            FactorCovariance(),
        )

    cartes = tuple(
        validate_covariance_model(returns, m, **kwargs)  # type: ignore[arg-type]
        for m in models
    )
    return ModelInventory(cards=cartes, universe=universe)


def with_materiality(card: ModelCard, materiality: float) -> ModelCard:
    """
    Recalcule le verdict d'une fiche sous un autre seuil de matérialité.

    Utile pour répondre à « et si on tolérait cinq pour cent ? » sans
    relancer le rejeu, qui est la partie coûteuse. Le seuil est une
    décision de gouvernance, pas une propriété des données : il doit
    donc pouvoir changer sans recalculer la mesure.
    """
    if card.bias is None:
        return card
    nouveau_test = replace(card.bias, materiality=materiality)
    return replace(card, bias=nouveau_test, verdict=nouveau_test.verdict)
