"""
Backtest de l'Expected Shortfall — la mesure que Bâle impose depuis FRTB.

POURQUOI CE MODULE EXISTE, ET POURQUOI IL EST PLUS DIFFICILE QU'IL N'Y PARAÎT
----------------------------------------------------------------------------
La séance 3 a implémenté la VaR, sa CVaR, et le test de Kupiec. Elle a
aussi démontré que la VaR n'est pas une mesure de risque cohérente :
deux obligations indépendantes ont chacune une VaR nulle et leur
combinaison une VaR positive, donc la VaR affirme que diversifier *crée*
du risque.

Le régulateur en a tiré la conséquence. La révision du dispositif de
risque de marché (FRTB) a remplacé la VaR à 99 % par l'**Expected
Shortfall à 97,5 %** pour le calcul du capital. L'ES est sous-additive,
donc cohérente, et elle regarde la taille des pertes extrêmes et non
seulement leur fréquence.

Mais ce gain a un prix, et c'est le sujet de cette séance.

    Backtester une VaR est facile : on compte les dépassements, et
    sous l'hypothèse nulle leur nombre suit une loi binomiale(T, α).
    Cette loi ne dépend que de α. Le test de Kupiec est donc
    **indépendant de la distribution**.

    Backtester une ES ne peut pas l'être. L'ES est une moyenne
    conditionnelle : la loi de n'importe quelle statistique qui la
    teste dépend de la FORME de la queue au-delà de la VaR, et cette
    forme n'est pas contenue dans les deux nombres annoncés. Il n'existe
    donc pas de valeur critique universelle, et les valeurs critiques
    doivent être simulées sous une hypothèse explicite.

C'est la raison pour laquelle `acerbi_szekely` **exige** un objet
`StandardizedNull` et refuse d'en choisir un par défaut. Un défaut
silencieux serait une hypothèse de distribution cachée dans un test de
conformité réglementaire — exactement le genre de chose que ce projet
refuse depuis la séance 1.

LES DEUX STATISTIQUES D'ACERBI ET SZÉKELY (2014)
------------------------------------------------
Convention de ce module : les rendements `X` sont signés (une perte est
négative), la VaR et l'ES sont des **magnitudes positives**, et un
dépassement est `X_t < −VaR_t`.

**Test 1 — conditionnel aux dépassements.**

    Z₁ = (1/N) · Σ_{t ∈ dépassements} X_t / ES_t  +  1

où N est le nombre de dépassements. Sous H₀, E[X | dépassement] = −ES,
donc chaque terme vaut environ −1 et Z₁ ≈ 0.

**Test 2 — fréquence ET magnitude.**

    Z₂ = (1 / (T·α)) · Σ_{t ∈ dépassements} X_t / ES_t  +  1

Le dénominateur est le nombre de dépassements *attendu* et non le nombre
observé. C'est toute la différence : Z₂ paie à la fois les dépassements
trop nombreux et les dépassements trop profonds.

Pour les deux : **Z < 0 signifie que le modèle sous-estime le risque**,
donc le sens dangereux. Z > 0 signifie qu'il est trop prudent.

⚠️ L'ANGLE MORT DU TEST 1, ET IL EST BÉANT

Le test 1 ne demande que « sachant qu'il y a eu dépassement, la
magnitude était-elle juste ? ». Un modèle qui annonce les VaR et ES d'un
niveau de confiance **différent** de celui qu'il prétend tenir est
parfaitement cohérent avec lui-même : conditionnellement à un
dépassement, sa magnitude est correcte. Le test 1 ne voit rien, alors
que le modèle dépasse deux fois trop souvent.

`examples/seance13.py` le mesure : sur un tel modèle, Z₁ vaut +0,0006 et
Z₂ vaut −0,997. C'est la leçon de la séance 12 dans un domaine
entièrement différent — **un angle mort appartient au test, pas au
modèle testé** — et c'est pourquoi ce module impose `test=2` par défaut
et documente le test 1 comme un complément de diagnostic, jamais comme
une validation.

COMPARER N'EST PAS TESTER
-------------------------
`acerbi_szekely` répond à « ce modèle est-il réfuté ? ». Elle ne répond
pas à « lequel de ces deux modèles est le meilleur ? ». Pour cela il
faut une fonction de score, et l'ES pose là un problème de fond :

    **L'ES seule n'est pas élicitable** (Gneiting, 2011) : il n'existe
    aucune fonction de perte L(X, e) dont l'espérance soit minimisée à
    la vraie ES. On ne peut donc pas classer des prévisions d'ES par un
    score portant sur l'ES seule.

    **Le couple (VaR, ES) l'est** (Fissler & Ziegel, 2016). D'où
    `fz_loss`, qui score le couple.

Conséquence pratique, et elle est mesurée dans la démonstration : avec
une VaR fausse, l'ES qui minimise la perte n'est plus la vraie ES. On ne
peut pas évaluer une prévision d'ES sans s'engager aussi sur une
prévision de VaR.

Références
----------
Acerbi, C. & Székely, B. (2014). « Backtesting Expected Shortfall ».
    *Risk Magazine*.
Fissler, T. & Ziegel, J. (2016). « Higher order elicitability and
    Osband's principle ». *Annals of Statistics* 44(4).
Gneiting, T. (2011). « Making and evaluating point forecasts ».
    *JASA* 106(494).
Comité de Bâle, *Minimum capital requirements for market risk* (FRTB) :
    ES à 97,5 % en remplacement de la VaR à 99 %.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats

from .types import InsufficientData
from .var import TailRisk

#: Niveau de l'Expected Shortfall retenu par Bâle (FRTB) : queue de 2,5 %.
LEVEL_FRTB = 0.025

#: Nombre minimal de dépassements attendus pour que le test ait un sens.
#: Sous α = 2,5 %, cela correspond à environ 400 observations. En dessous,
#: la statistique repose sur une poignée de points et son intervalle est
#: dominé par le hasard d'échantillonnage.
MIN_EXPECTED_BREACHES = 10

#: Tolérance sur l'accord entre le rapport ES/VaR de l'hypothèse nulle et
#: celui des prévisions. Au-delà, la nulle contredit les prévisions
#: qu'elle est censée tester.
RATIO_TOLERANCE = 0.05


# ══════════════════════════════════════════════════════════════════════
#  Les prévisions de queue
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TailForecast:
    """
    Un chemin de prévisions (VaR, ES), validé à la construction.

    Les deux séries sont des **magnitudes positives** et doivent
    satisfaire ES ≥ VaR > 0. Cette contrainte n'est pas cosmétique :
    l'ES est une moyenne de pertes toutes supérieures à la VaR, donc
    ES < VaR est mathématiquement impossible et signale un bug en amont.
    La refuser ici évite de propager un non-sens dans un test de
    conformité.
    """

    var: np.ndarray
    es: np.ndarray
    alpha: float
    method: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.var, dtype=float).ravel()
        e = np.asarray(self.es, dtype=float).ravel()
        object.__setattr__(self, "var", v)
        object.__setattr__(self, "es", e)

        if not (0.0 < self.alpha < 0.5):
            raise ValueError(
                f"alpha doit être dans (0, 0.5) — c'est une probabilité de "
                f"queue, pas un niveau de confiance. Reçu {self.alpha}. "
                f"Pour une ES à 97,5 %, passer 0.025."
            )
        if v.shape != e.shape:
            raise ValueError(
                f"var et es doivent avoir la même longueur : "
                f"{v.size} contre {e.size}"
            )
        if v.size == 0:
            raise ValueError("aucune prévision fournie")
        if not (np.isfinite(v).all() and np.isfinite(e).all()):
            raise ValueError(
                "une prévision de VaR ou d'ES est NaN ou infinie. Un trou "
                "dans un chemin de prévisions doit être expliqué, pas écarté."
            )
        if (v <= 0.0).any():
            raise ValueError(
                f"{int((v <= 0.0).sum())} VaR nulle(s) ou négative(s). La "
                "convention de ce module est la MAGNITUDE POSITIVE : une VaR "
                "de 0.025 signifie « perte de 2,5 % ». Un signe négatif ici "
                "est presque toujours une convention mal traduite."
            )
        mauvais = e < v
        if mauvais.any():
            i = int(np.argmax(mauvais))
            raise ValueError(
                f"ES < VaR à l'indice {i} ({e[i]:.6g} < {v[i]:.6g}). L'ES "
                "est la moyenne des pertes qui dépassent la VaR, donc elle "
                "ne peut pas lui être inférieure : il y a un bug en amont."
            )

    @property
    def n_forecasts(self) -> int:
        return int(self.var.size)

    @property
    def ratio(self) -> float:
        """ES/VaR moyen — l'épaisseur de queue que le modèle annonce."""
        return float(np.mean(self.es / self.var))

    def __str__(self) -> str:
        return (
            f"{self.n_forecasts} prévisions à α={self.alpha:.1%} "
            f"({self.method or 'méthode non déclarée'}) — "
            f"ES/VaR moyen {self.ratio:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  L'hypothèse nulle, qui doit être explicite
# ══════════════════════════════════════════════════════════════════════


@runtime_checkable
class StandardizedNull(Protocol):
    """
    La loi des rendements standardisés sous l'hypothèse nulle.

    Sixième usage du typage structurel dans ce projet.

    Sous H₀, le rendement réalisé vaut `var_t · u_t` où `u` suit cette
    loi, normalisée pour que sa VaR au niveau α valonne exactement 1.
    C'est ce qui rend le tirage compatible avec les prévisions : par
    construction, la VaR simulée est la VaR annoncée.

    `ratio` est l'ES/VaR de la loi au niveau α. Il DOIT s'accorder avec
    celui des prévisions, sinon la nulle contredit ce qu'elle teste.
    """

    @property
    def name(self) -> str: ...

    def ratio(self, alpha: float) -> float: ...

    def draw(self, size: int, alpha: float, rng: np.random.Generator) -> np.ndarray: ...


@dataclass(frozen=True)
class GaussianNull:
    """
    Rendements standardisés gaussiens. À α = 2,5 %, ES/VaR ≈ 1,193.

    Presque jamais le bon choix sur données de marché, et conservé
    précisément pour mesurer ce que coûte de le choisir quand même.
    """

    name: str = "gaussienne"

    def ratio(self, alpha: float) -> float:
        q = float(stats.norm.ppf(alpha))
        es = float(stats.norm.expect(lambda x: -x, lb=-np.inf, ub=q)) / alpha
        return es / (-q)

    def draw(
        self, size: int, alpha: float, rng: np.random.Generator
    ) -> np.ndarray:
        q = float(stats.norm.ppf(alpha))
        return rng.standard_normal(size) / (-q)


@dataclass(frozen=True)
class StudentTNull:
    """
    Rendements standardisés de Student. ES/VaR croît quand ν décroît :
    environ 1,29 à ν = 8, 1,37 à ν = 5, 1,58 à ν = 3.
    """

    nu: float = 5.0

    def __post_init__(self) -> None:
        if self.nu <= 1.0:
            raise ValueError(
                f"nu doit être > 1 pour que l'espérance de queue existe, "
                f"reçu {self.nu}"
            )

    @property
    def name(self) -> str:
        return f"Student t({self.nu:g})"

    def ratio(self, alpha: float) -> float:
        loi = stats.t(self.nu)
        q = float(loi.ppf(alpha))
        es = float(loi.expect(lambda x: -x, lb=-np.inf, ub=q)) / alpha
        return es / (-q)

    def draw(
        self, size: int, alpha: float, rng: np.random.Generator
    ) -> np.ndarray:
        # `rng.standard_t` tire la MÊME loi que scipy et va environ
        # cinquante fois plus vite, ce qui compte quand on simule des
        # dizaines de milliers de chemins pour une valeur critique.
        q = float(stats.t(self.nu).ppf(alpha))
        return rng.standard_t(self.nu, size=size) / (-q)


@dataclass(frozen=True)
class EmpiricalNull:
    """
    Rééchantillonnage des résidus standardisés du modèle lui-même.

    **C'est le choix à privilégier en pratique.** Il ne suppose aucune
    forme de queue : il réutilise celle que les données ont montrée, et
    ramène le backtest d'ES près de l'indépendance distributionnelle que
    Kupiec possède gratuitement.

    `residuals` doit contenir les rendements divisés par la volatilité
    prévue à la même date — donc des quantités causales, produites par
    un rejeu point-in-time.

    LIMITE À ASSUMER : la queue rééchantillonnée est celle de
    l'échantillon. Si la fenêtre n'a pas vu de krach, la nulle n'en
    contient pas, et le test sera trop indulgent. C'est la limite de la
    VaR historique de la séance 3, revenue à un autre endroit.
    """

    residuals: np.ndarray
    name: str = "empirique (rééchantillonnage)"

    def __post_init__(self) -> None:
        r = np.asarray(self.residuals, dtype=float).ravel()
        r = r[np.isfinite(r)]
        if r.size < 100:
            raise InsufficientData(
                n_obs=int(r.size),
                n_required=100,
                what=(
                    "hypothèse nulle empirique (la queue à "
                    "rééchantillonner serait faite de trop peu de points)"
                ),
            )
        object.__setattr__(self, "residuals", r)

    def _standardise(self, alpha: float) -> np.ndarray:
        q = float(np.quantile(self.residuals, alpha))
        if q >= 0.0:
            raise ValueError(
                f"le quantile {alpha:.1%} des résidus vaut {q:.4g} ≥ 0 : ces "
                "résidus ne ressemblent pas à des rendements centrés."
            )
        return self.residuals / (-q)

    def ratio(self, alpha: float) -> float:
        u = self._standardise(alpha)
        seuil = float(np.quantile(u, alpha))
        queue = u[u <= seuil]
        return float(-queue.mean() / (-seuil))

    def draw(
        self, size: int, alpha: float, rng: np.random.Generator
    ) -> np.ndarray:
        u = self._standardise(alpha)
        return rng.choice(u, size=size, replace=True)


# ══════════════════════════════════════════════════════════════════════
#  Le test
# ══════════════════════════════════════════════════════════════════════

VALIDE = "ACCEPTÉ"
REJETE = "REJETÉ"

Z_CONDITIONAL = 1
Z_UNCONDITIONAL = 2


@dataclass(frozen=True)
class AcerbiSzekelyResult:
    """Résultat d'un test d'Acerbi-Székely, avec sa valeur critique simulée."""

    statistic: float
    p_value: float
    critical_value: float
    test: int
    alpha: float
    n_obs: int
    n_breaches: int
    expected_breaches: float
    null_name: str
    n_sim: int
    level: float

    @property
    def rejected(self) -> bool:
        """
        Rejet UNILATÉRAL à gauche.

        Le test est volontairement unilatéral : un modèle trop prudent
        immobilise du capital, un modèle trop optimiste cache une perte.
        Les deux erreurs ne sont pas symétriques, et le dispositif
        réglementaire ne les traite pas comme telles. `statistic` reste
        disponible pour juger l'excès de prudence.
        """
        return self.statistic <= self.critical_value

    @property
    def verdict(self) -> str:
        return REJETE if self.rejected else VALIDE

    @property
    def direction(self) -> str:
        if self.statistic < 0.0:
            return (
                f"SOUS-ESTIME la queue ({self.statistic:+.3f}) — sens dangereux"
            )
        if self.statistic > 0.0:
            return f"trop prudent ({self.statistic:+.3f}) — coûteux en capital"
        return "exactement calibré"

    @property
    def blind_to_frequency(self) -> bool:
        """
        Vrai pour le test 1, qui ne voit pas la fréquence des dépassements.

        Exposé comme propriété plutôt que laissé dans la documentation :
        un angle mort qu'on peut interroger par programme est un angle
        mort qu'on n'oublie pas.
        """
        return self.test == Z_CONDITIONAL

    def __str__(self) -> str:
        nom = "Z₁ (conditionnel)" if self.test == Z_CONDITIONAL else "Z₂"
        ligne = (
            f"{nom} = {self.statistic:+.4f}   "
            f"critique {self.critical_value:+.4f}   p={self.p_value:.4f}   "
            f"→ {self.verdict}"
        )
        detail = (
            f"\n    {self.n_breaches} dépassements sur {self.n_obs} "
            f"(attendu {self.expected_breaches:.1f}) · "
            f"nulle {self.null_name} · {self.n_sim} simulations"
        )
        avert = (
            "\n    ⚠ ce test ne voit PAS la fréquence des dépassements"
            if self.blind_to_frequency
            else ""
        )
        return ligne + detail + avert


def _statistic(
    x: np.ndarray, forecast: TailForecast, test: int
) -> tuple[float, int]:
    """Z₁ ou Z₂, et le nombre de dépassements."""
    depassement = x < -forecast.var
    n = int(depassement.sum())

    if n == 0:
        # Aucun dépassement : la statistique n'est pas définie pour le
        # test 1 (division par zéro) et vaut exactement +1 pour le
        # test 2. On retourne +1 dans les deux cas, ce qui est le sens
        # correct — « rien n'est arrivé » est l'extrême de la prudence —
        # et on laisse l'appelant lire n_breaches.
        return 1.0, 0

    somme = float(np.sum(x[depassement] / forecast.es[depassement]))
    if test == Z_CONDITIONAL:
        return somme / n + 1.0, n
    return somme / (x.size * forecast.alpha) + 1.0, n


def tail_statistic(
    returns: np.ndarray | pd.Series,
    forecast: TailForecast,
    *,
    test: int = Z_UNCONDITIONAL,
) -> tuple[float, int]:
    """
    La statistique Z seule, sans valeur critique, et le nombre de
    dépassements.

    `acerbi_szekely` simule des milliers de chemins pour obtenir sa
    valeur critique. Quand on veut seulement comparer des statistiques
    entre elles — une étude de puissance, une courbe de diagnostic — ce
    coût est inutile. D'où cette porte d'entrée.

    ⚠️ Une statistique sans sa valeur critique ne permet AUCUN verdict.
    Z₂ = −0,2 peut être banal sur mille observations et accablant sur
    dix mille. Pour juger un modèle, c'est `acerbi_szekely`.
    """
    x = np.asarray(returns, dtype=float).ravel()
    if test not in (Z_CONDITIONAL, Z_UNCONDITIONAL):
        raise ValueError(f"test doit valoir 1 ou 2, reçu {test}")
    if x.size != forecast.n_forecasts:
        raise ValueError(
            f"returns et forecast doivent avoir la même longueur : "
            f"{x.size} contre {forecast.n_forecasts}"
        )
    if not np.isfinite(x).all():
        raise ValueError("un rendement réalisé est NaN ou infini")
    return _statistic(x, forecast, test)


def acerbi_szekely(
    returns: np.ndarray | pd.Series,
    forecast: TailForecast,
    *,
    null: StandardizedNull,
    test: int = Z_UNCONDITIONAL,
    n_sim: int = 10_000,
    level: float = 0.05,
    seed: int = 0,
    check_ratio: bool = True,
) -> AcerbiSzekelyResult:
    """
    Backtest d'Expected Shortfall par la méthode d'Acerbi-Székely.

    Paramètres
    ----------
    returns : rendements réalisés, signés (une perte est négative).
    forecast : le chemin (VaR, ES), en magnitudes positives, de même
        longueur que `returns` et produit CAUSALEMENT — chaque prévision
        ne doit utiliser que l'information antérieure à sa date.
    null : la loi des rendements standardisés sous H₀. **Obligatoire.**
        Il n'existe pas de valeur critique universelle pour une
        statistique d'ES : la loi de la statistique dépend de la forme de
        la queue, qui n'est pas contenue dans les deux nombres annoncés.
        Un défaut silencieux cacherait une hypothèse de distribution dans
        un test de conformité. `EmpiricalNull` est le choix recommandé.
    test : 2 (défaut) pour Z₂, qui teste fréquence et magnitude ;
        1 pour Z₁, qui est conditionnel aux dépassements et **aveugle à
        leur fréquence** — à n'utiliser qu'en complément de diagnostic.
    n_sim : nombre de chemins simulés pour la valeur critique.
    level : niveau du test, unilatéral à gauche.

    Lève
    ----
    ValueError
        Si les longueurs ne concordent pas, si le test demandé n'existe
        pas, ou si le rapport ES/VaR de la nulle contredit celui des
        prévisions (voir `check_ratio`).
    InsufficientData
        Si le nombre de dépassements ATTENDU est trop faible pour que la
        statistique repose sur autre chose que du hasard.
    """
    x = np.asarray(returns, dtype=float).ravel()

    if test not in (Z_CONDITIONAL, Z_UNCONDITIONAL):
        raise ValueError(f"test doit valoir 1 ou 2, reçu {test}")
    if not (0.0 < level < 0.5):
        raise ValueError(f"level doit être dans (0, 0.5), reçu {level}")
    if n_sim < 200:
        raise ValueError(
            f"n_sim = {n_sim} est trop faible pour estimer un quantile de "
            f"queue à {level:.0%}. En dessous de 200, la valeur critique est "
            "elle-même du bruit."
        )
    if x.size != forecast.n_forecasts:
        raise ValueError(
            f"returns et forecast doivent avoir la même longueur : "
            f"{x.size} contre {forecast.n_forecasts}"
        )
    if not np.isfinite(x).all():
        raise ValueError(
            f"{int((~np.isfinite(x)).sum())} rendement(s) réalisé(s) NaN ou "
            "infini(s). Un trou dans la série réalisée doit être expliqué."
        )

    attendus = x.size * forecast.alpha
    if attendus < MIN_EXPECTED_BREACHES:
        raise InsufficientData(
            n_obs=x.size,
            n_required=math.ceil(MIN_EXPECTED_BREACHES / forecast.alpha),
            what=(
                f"backtest d'ES à α={forecast.alpha:.1%} "
                f"({attendus:.1f} dépassements attendus seulement)"
            ),
        )

    if check_ratio:
        attendu = null.ratio(forecast.alpha)
        observe = forecast.ratio
        if abs(attendu - observe) > RATIO_TOLERANCE:
            raise ValueError(
                f"l'hypothèse nulle « {null.name} » a un rapport ES/VaR de "
                f"{attendu:.3f} à α={forecast.alpha:.1%}, alors que les "
                f"prévisions annoncent {observe:.3f}. La nulle contredit donc "
                "les prévisions qu'elle doit tester, et sa valeur critique ne "
                "vaudrait rien.\n"
                "  → Le couple (VaR, ES) FIXE déjà le rapport de queue : "
                "l'hypothèse nulle n'est pas libre.\n"
                "  → Utiliser EmpiricalNull(résidus standardisés du modèle), "
                "ou StudentTNull avec le ν qui reproduit ce rapport, ou "
                "check_ratio=False si l'écart est assumé."
            )

    statistique, n_depassements = _statistic(x, forecast, test)

    # La valeur critique, par simulation. Il n'y a pas de forme close.
    rng = np.random.default_rng(seed)
    simulees = np.empty(n_sim)
    for i in range(n_sim):
        u = null.draw(x.size, forecast.alpha, rng)
        simulees[i] = _statistic(forecast.var * u, forecast, test)[0]

    critique = float(np.quantile(simulees, level))
    # p-value unilatérale à gauche. Le +1 au numérateur et au dénominateur
    # évite une p-value exactement nulle, qui n'aurait aucun sens avec un
    # nombre fini de simulations.
    p = float((np.sum(simulees <= statistique) + 1) / (n_sim + 1))

    return AcerbiSzekelyResult(
        statistic=statistique,
        p_value=p,
        critical_value=critique,
        test=test,
        alpha=forecast.alpha,
        n_obs=int(x.size),
        n_breaches=n_depassements,
        expected_breaches=float(attendus),
        null_name=null.name,
        n_sim=n_sim,
        level=level,
    )


# ══════════════════════════════════════════════════════════════════════
#  Comparer deux modèles : la fonction de perte de Fissler-Ziegel
# ══════════════════════════════════════════════════════════════════════


def fz_loss(
    returns: np.ndarray | pd.Series, forecast: TailForecast
) -> np.ndarray:
    """
    Perte de Fissler-Ziegel pour le couple (VaR, ES) — version 0-homogène.

    En magnitudes positives, avec X le rendement réalisé :

        L = −1/(α·E) · 1{X ≤ −V} · (X + V)  +  V/E  +  log(E)

    PROPRIÉTÉ QUI JUSTIFIE TOUT LE RESTE : son espérance est minimisée
    exactement au vrai couple (VaR, ES). C'est la définition de
    l'élicitabilité conjointe, et c'est **testable** — donc testé, dans
    `test_la_perte_fz_est_minimisee_a_la_verite`.

    POURQUOI UNE PERTE SUR LE COUPLE ET NON SUR L'ES SEULE
    ------------------------------------------------------
    Parce qu'aucune perte sur l'ES seule ne possède cette propriété :
    l'ES n'est pas élicitable (Gneiting, 2011). Le couple (VaR, ES) l'est
    (Fissler & Ziegel, 2016), et c'est le plus petit ensemble qui le soit.

    Conséquence qu'il faut avoir en tête avant de comparer deux modèles :
    on ne peut pas juger une prévision d'ES sans s'engager aussi sur une
    prévision de VaR. Avec une VaR fausse, l'ES qui minimise la perte
    n'est plus la vraie ES — la démonstration le mesure.

    ⚠️ CETTE PERTE SERT À COMPARER, PAS À TESTER. Sa valeur absolue ne
    veut rien dire ; seules les différences entre modèles sur les mêmes
    données en ont un. Pour savoir si UN modèle est réfuté, c'est
    `acerbi_szekely`.
    """
    x = np.asarray(returns, dtype=float).ravel()
    if x.size != forecast.n_forecasts:
        raise ValueError(
            f"returns et forecast doivent avoir la même longueur : "
            f"{x.size} contre {forecast.n_forecasts}"
        )
    if not np.isfinite(x).all():
        raise ValueError("un rendement réalisé est NaN ou infini")

    v, e, a = forecast.var, forecast.es, forecast.alpha
    depassement = x <= -v
    return -(1.0 / (a * e)) * depassement * (x + v) + v / e + np.log(e)


@dataclass(frozen=True)
class TailComparison:
    """Comparaison appariée de deux modèles de queue par la perte FZ."""

    name_a: str
    name_b: str
    mean_loss_a: float
    mean_loss_b: float
    difference: float
    t_statistic: float
    p_value: float
    n_obs: int

    @property
    def winner(self) -> str:
        """Le modèle de perte la plus faible. Pas forcément significatif."""
        return self.name_a if self.mean_loss_a < self.mean_loss_b else self.name_b

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:
        verdict = (
            f"{self.winner} est meilleur"
            if self.significant
            else "écart non significatif"
        )
        return (
            f"{self.name_a} {self.mean_loss_a:+.5f} contre "
            f"{self.name_b} {self.mean_loss_b:+.5f}   "
            f"écart {self.difference:+.5f}   p={self.p_value:.4f}   → {verdict}"
        )


def compare_tail_models(
    returns: np.ndarray | pd.Series,
    a: TailForecast,
    b: TailForecast,
    *,
    name_a: str = "A",
    name_b: str = "B",
) -> TailComparison:
    """
    Compare deux modèles de queue sur les mêmes données, perte FZ appariée.

    Le test est un t apparié sur la différence de pertes période par
    période. L'appariement est essentiel : les deux modèles voient les
    mêmes chocs, donc leurs pertes sont fortement corrélées, et un test
    non apparié jetterait l'essentiel de l'information.

    ⚠️ L'autocorrélation des différences de perte n'est PAS corrigée. En
    présence de volatilité groupée, la p-value est donc trop petite — le
    test est anticonservateur. Une correction HAC (Newey-West) serait le
    traitement correct ; elle n'est pas implémentée, et c'est déclaré ici
    plutôt que découvert par l'utilisateur.
    """
    if a.alpha != b.alpha:
        raise ValueError(
            f"les deux modèles doivent porter sur le même niveau de queue : "
            f"{a.alpha} contre {b.alpha}"
        )
    la = fz_loss(returns, a)
    lb = fz_loss(returns, b)
    d = la - lb

    n = int(d.size)
    moyenne = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd <= 0.0:
        t_stat, p = 0.0, 1.0
    else:
        t_stat = moyenne / (sd / math.sqrt(n))
        p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))

    return TailComparison(
        name_a=name_a,
        name_b=name_b,
        mean_loss_a=float(la.mean()),
        mean_loss_b=float(lb.mean()),
        difference=moyenne,
        t_statistic=float(t_stat),
        p_value=p,
        n_obs=n,
    )


# ══════════════════════════════════════════════════════════════════════
#  Produire les prévisions, causalement
# ══════════════════════════════════════════════════════════════════════


def rolling_tail_forecasts(
    returns: pd.Series,
    estimator: Callable[[pd.Series], TailRisk],
    *,
    lookback: int = 500,
    step: int = 1,
) -> tuple[np.ndarray, TailForecast]:
    """
    Rejeu point-in-time : produit un chemin (VaR, ES) sans voir le futur.

    La ligne qui porte toute la garantie est la même que celle de la
    séance 8 :

        historique = returns.iloc[t - lookback : t]     ← t EXCLU

    `estimator` est n'importe quelle fonction Série → `TailRisk` :
    `historical_var`, `parametric_var`, `cornish_fisher_var`, ou une
    fermeture autour de l'une d'elles. C'est le même procédé d'injection
    que la séance 9 a utilisé pour les sources de données et la séance 12
    pour les modèles de covariance.

    Retourne les rendements réalisés alignés et les prévisions, dans cet
    ordre — jamais l'un sans l'autre, pour qu'un désalignement d'un cran
    soit impossible à écrire par accident.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns doit être une Series, reçu {type(returns)}")
    if lookback < 2:
        raise ValueError(f"lookback doit être ≥ 2, reçu {lookback}")
    if step < 1:
        raise ValueError(f"step doit être ≥ 1, reçu {step}")

    n = len(returns)
    if n <= lookback:
        raise InsufficientData(
            n_obs=n,
            n_required=lookback + step,
            what=(
                f"rejeu de prévisions de queue (lookback {lookback} : rien "
                "ne reste après la première estimation)"
            ),
        )

    valeurs = returns.to_numpy(dtype=float)
    realises: list[float] = []
    vars_: list[float] = []
    ess: list[float] = []
    confiance: float | None = None
    methode = ""

    for t in range(lookback, n, step):
        historique = returns.iloc[t - lookback : t]  # ← t EXCLU. Causalité.
        risque = estimator(historique)

        if confiance is None:
            confiance = risque.confidence
            methode = risque.method
        elif risque.confidence != confiance:
            raise ValueError(
                f"l'estimateur a changé de niveau de confiance en cours de "
                f"rejeu : {confiance} puis {risque.confidence}"
            )

        fin = min(t + step, n)
        for s in range(t, fin):
            realises.append(float(valeurs[s]))
            vars_.append(risque.var)
            ess.append(risque.cvar)

    assert confiance is not None  # garanti par n > lookback
    return (
        np.asarray(realises),
        TailForecast(
            var=np.asarray(vars_),
            es=np.asarray(ess),
            alpha=1.0 - confiance,
            method=methode,
        ),
    )
