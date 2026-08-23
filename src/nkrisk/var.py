"""
Valeur à risque (VaR) et perte moyenne au-delà (CVaR).

CE QUE MESURE UNE VaR
=====================
« Avec 99 % de confiance, je ne perdrai pas plus de X en une journée. »

C'est la mesure de risque la plus répandue de la finance. Elle est aussi
la plus mal comprise et la plus dangereuse — au point que la
réglementation bancaire internationale l'a abandonnée au profit de la
CVaR dans le cadre FRTB.

Ce fichier construit les deux, et démontre les trois raisons pour
lesquelles la VaR peut te ruiner :

  1. La version paramétrique, de loin la plus utilisée, suppose des
     rendements normaux. Les marchés n'en ont pas. Elle sous-estime donc
     systématiquement le risque exactement là où il compte : dans la queue.

  2. La VaR ne dit RIEN sur ce qui se passe au-delà du seuil. Deux
     portefeuilles peuvent avoir la même VaR à 99 % et des pertes
     catastrophiques radicalement différentes le 1 % du temps restant.

  3. La VaR n'est PAS sous-additive. Elle peut affirmer que diversifier
     augmente le risque. Ce n'est pas une imprécision : c'est une
     incohérence mathématique démontrable, et on la démontre plus bas.

CONVENTION DE SIGNE, à fixer une fois pour toutes
-------------------------------------------------
Toutes les fonctions retournent la perte comme un nombre POSITIF.
Une VaR de 0,03 signifie « perte de 3 % ». Mélanger les conventions de
signe entre modules est une source de bug classique et coûteuse ; on
choisit ici, on documente, et on ne dévie jamais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .types import InsufficientData


@dataclass(frozen=True)
class TailRisk:
    """VaR et CVaR calculées par une méthode donnée, avec son diagnostic."""

    var: float
    cvar: float
    confidence: float
    method: str
    n_obs: int
    assumptions: tuple[str, ...] = ()

    @property
    def tail_ratio(self) -> float:
        """
        CVaR / VaR — mesure de l'épaisseur de la queue.

        Sous hypothèse normale à 99 %, ce rapport vaut environ 1,15.
        Plus il est élevé, plus la queue est lourde : quand la perte
        dépasse la VaR, elle la dépasse de beaucoup.

        C'est le chiffre qui te dit à quel point la VaR seule te ment.
        """
        return self.cvar / self.var if self.var > 0 else float("nan")

    def format(self) -> str:
        pct = int(round(self.confidence * 100))
        return (
            f"{self.method:<16} VaR {pct}% = {self.var:>7.2%}   "
            f"CVaR = {self.cvar:>7.2%}   ratio = {self.tail_ratio:.2f}"
        )


def historical_var(
    returns: pd.Series, confidence: float = 0.99, min_obs: int = 250
) -> TailRisk:
    """
    VaR historique — on lit directement le quantile empirique.

    Aucune hypothèse de distribution : on trie les rendements passés et on
    prend le percentile. Si tu as 1 000 jours et que tu veux la VaR à
    99 %, c'est le 10ᵉ pire jour.

    AVANTAGE : la seule méthode qui capture naturellement les queues
    épaisses, l'asymétrie et tout ce que la théorie ignore, parce qu'elle
    ne modélise rien du tout.

    LIMITE, et elle est sévère : elle ne peut pas voir un événement qui
    ne s'est pas produit dans ton échantillon. Une VaR historique calculée
    sur 2003-2007 n'avait jamais vu 2008. Le risque le plus dangereux
    est toujours celui qui n'est pas encore dans les données.

    LIMITE STATISTIQUE : à 99 %, seul 1 % de ton échantillon informe
    l'estimation. Sur 250 jours, cela fait **2,5 observations**. Tu
    estimes donc un quantile extrême avec deux ou trois points. C'est
    pourquoi `min_obs` est fixé à 250 et devrait idéalement être bien
    plus élevé.
    """
    r = _clean(returns, min_obs, "VaR historique")
    n = len(r)

    seuil = float(np.quantile(r, 1.0 - confidence))
    var = -seuil

    queue = r[r <= seuil]
    cvar = -float(np.mean(queue)) if len(queue) > 0 else var

    return TailRisk(
        var=var,
        cvar=cvar,
        confidence=confidence,
        method="historique",
        n_obs=n,
        assumptions=(
            "le passé contient les scénarios pertinents (souvent faux)",
            f"queue estimée sur ~{int(n * (1 - confidence))} observations",
        ),
    )


def parametric_var(
    returns: pd.Series, confidence: float = 0.99, min_obs: int = 60
) -> TailRisk:
    """
    VaR paramétrique gaussienne — la méthode la plus répandue, et la pire.

    On suppose les rendements normaux, on estime μ et σ, et on lit le
    quantile de la loi normale :

        VaR = -(μ + z_{1-α}·σ)

    La CVaR gaussienne a également une forme close :

        CVaR = σ·φ(z_p)/p - μ     avec p = 1-α et φ la densité normale

    ⚠️ POURQUOI ELLE MENT ⚠️

    Elle n'utilise que deux nombres : la moyenne et l'écart-type. Toute
    l'information sur la FORME de la distribution est jetée. Or c'est
    précisément la forme des queues qui détermine le risque extrême.

    Deux portefeuilles de volatilité identique — l'un normal, l'autre à
    queues épaisses — reçoivent exactement la même VaR paramétrique.
    Leur risque réel de perte extrême diffère pourtant d'un facteur trois
    ou plus. La démonstration de la séance 3 le mesure.

    Ordre de grandeur à retenir : sous la normale, une baisse de 5
    écarts-types a une probabilité de 1 sur 3,5 millions — un jour tous
    les 14 000 ans de bourse. Elles arrivent en pratique tous les
    quelques années.

    Ne l'utilise jamais seule. Elle est ici pour être comparée aux autres.
    """
    r = _clean(returns, min_obs, "VaR paramétrique")
    n = len(r)

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    p = 1.0 - confidence

    z = stats.norm.ppf(p)
    var = -(mu + z * sigma)

    # Espérance conditionnelle dans la queue gaussienne
    cvar = sigma * stats.norm.pdf(z) / p - mu

    return TailRisk(
        var=var,
        cvar=cvar,
        confidence=confidence,
        method="paramétrique",
        n_obs=n,
        assumptions=(
            "rendements NORMAUX — faux sur tous les marchés connus",
            "toute la forme de la distribution est ignorée",
            "sous-estime systématiquement le risque de queue",
        ),
    )


def cornish_fisher_var(
    returns: pd.Series, confidence: float = 0.99, min_obs: int = 250
) -> TailRisk:
    """
    VaR de Cornish-Fisher — la paramétrique corrigée de l'asymétrie et
    de l'aplatissement.

    On ne se contente plus de μ et σ : on ajuste le quantile normal en
    fonction des moments d'ordre 3 (asymétrie S) et 4 (kurtosis
    excédentaire K), via un développement asymptotique :

        z_cf = z + (z²-1)·S/6 + (z³-3z)·K/24 - (2z³-5z)·S²/36

    C'est un compromis intéressant : plus honnête que la gaussienne, plus
    stable que l'historique parce qu'elle utilise tout l'échantillon
    plutôt que la seule queue.

    LIMITE : c'est un développement en série, valable pour des écarts
    MODÉRÉS à la normalité. Sur des queues très épaisses, ou à des
    niveaux de confiance très élevés, il peut se comporter de façon
    erratique — jusqu'à produire des quantiles non monotones. On vérifie
    donc explicitement la monotonie et on prévient si elle est violée.
    """
    r = _clean(returns, min_obs, "VaR Cornish-Fisher")
    n = len(r)

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    s = float(stats.skew(r))
    k = float(stats.kurtosis(r))  # déjà excédentaire (normale → 0)

    p = 1.0 - confidence
    z = stats.norm.ppf(p)

    z_cf = (
        z
        + (z**2 - 1) * s / 6.0
        + (z**3 - 3 * z) * k / 24.0
        - (2 * z**3 - 5 * z) * s**2 / 36.0
    )

    var = -(mu + z_cf * sigma)

    # La CVaR n'a pas de forme close en Cornish-Fisher : on intègre
    # numériquement les quantiles ajustés sur la queue.
    ps = np.linspace(1e-6, p, 2000)
    zs = stats.norm.ppf(ps)
    zs_cf = (
        zs
        + (zs**2 - 1) * s / 6.0
        + (zs**3 - 3 * zs) * k / 24.0
        - (2 * zs**3 - 5 * zs) * s**2 / 36.0
    )
    cvar = -float(np.mean(mu + zs_cf * sigma))

    avertissements = []
    if np.any(np.diff(zs_cf) < 0):
        avertissements.append(
            "développement non monotone — hors de son domaine de validité"
        )

    return TailRisk(
        var=var,
        cvar=cvar,
        confidence=confidence,
        method="cornish-fisher",
        n_obs=n,
        assumptions=(
            "écart MODÉRÉ à la normalité (développement asymptotique)",
            f"asymétrie={s:.2f}, kurtosis excédentaire={k:.2f}",
            *avertissements,
        ),
    )


# ══════════════════════════════════════════════════════════════════════
#  BACKTEST : la VaR tient-elle ses promesses ?
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class KupiecResult:
    """Résultat du test de couverture non conditionnelle de Kupiec."""

    n_obs: int
    n_breaches: int
    expected_breaches: float
    breach_rate: float
    expected_rate: float
    lr_statistic: float
    p_value: float

    @property
    def rejected(self) -> bool:
        """Le modèle est-il rejeté au seuil de 5 % ?"""
        return self.p_value < 0.05

    def format(self) -> str:
        verdict = "REJETÉ" if self.rejected else "accepté"
        return (
            f"{self.n_breaches:>4} dépassements sur {self.n_obs} "
            f"(attendu {self.expected_breaches:.0f})   "
            f"taux {self.breach_rate:.2%} vs {self.expected_rate:.2%}   "
            f"p={self.p_value:.4f}  → modèle {verdict}"
        )


def kupiec_test(
    returns: pd.Series, var: float, confidence: float = 0.99
) -> KupiecResult:
    """
    Test de Kupiec (1995) — le backtest réglementaire de la VaR.

    C'EST LE TEST DE COUVERTURE DE LA SÉANCE 1, APPLIQUÉ À LA VaR
    -------------------------------------------------------------
    Si ta VaR à 99 % est correcte, alors les pertes doivent la dépasser
    exactement 1 % du temps. Ni plus — le modèle serait dangereux — ni
    moins — il serait inutilement conservateur et immobiliserait du
    capital pour rien.

    On compte donc les dépassements et on teste par rapport de
    vraisemblance si le taux observé est compatible avec le taux promis :

        LR = -2·ln[ ((1-p)^(T-x)·p^x) / ((1-x/T)^(T-x)·(x/T)^x) ]  ~ χ²(1)

    C'est exactement la même logique que le test de couverture de la
    séance 1 : on ne fait pas confiance à un intervalle, on VÉRIFIE
    empiriquement qu'il couvre ce qu'il prétend couvrir.

    LIMITE CONNUE : ce test ne regarde que le NOMBRE de dépassements, pas
    leur répartition dans le temps. Un modèle qui produit ses dix
    dépassements groupés sur une même semaine de krach passe le test
    aussi bien qu'un modèle qui les étale. C'est ce que le test de
    Christoffersen (indépendance) ajoute — hors périmètre de cette séance.
    """
    r = _clean(returns, 30, "backtest de Kupiec")
    t = len(r)

    x = int(np.sum(r < -var))
    p = 1.0 - confidence
    taux = x / t

    if x == 0:
        # log(0) diverge : on traite le cas séparément.
        lr = -2.0 * (t * np.log(1 - p))
    elif x == t:
        lr = -2.0 * (t * np.log(p))
    else:
        num = (t - x) * np.log(1 - p) + x * np.log(p)
        den = (t - x) * np.log(1 - taux) + x * np.log(taux)
        lr = -2.0 * (num - den)

    return KupiecResult(
        n_obs=t,
        n_breaches=x,
        expected_breaches=t * p,
        breach_rate=taux,
        expected_rate=p,
        lr_statistic=float(lr),
        p_value=float(1.0 - stats.chi2.cdf(lr, df=1)),
    )


# ══════════════════════════════════════════════════════════════════════
#  L'INCOHÉRENCE MATHÉMATIQUE DE LA VaR
# ══════════════════════════════════════════════════════════════════════


def subadditivity_counterexample(
    default_probability: float = 0.04,
    confidence: float = 0.95,
    loss_given_default: float = 1.0,
) -> dict[str, float]:
    """
    Contre-exemple constructif : la VaR punit la diversification.

    ★ LE RÉSULTAT LE PLUS IMPORTANT DE CETTE SÉANCE ★

    Artzner, Delbaen, Eber et Heath (1999) ont défini les quatre axiomes
    qu'une mesure de risque « cohérente » doit satisfaire. Le plus
    intuitif est la SOUS-ADDITIVITÉ :

        ρ(A + B) ≤ ρ(A) + ρ(B)

    « Un portefeuille combiné n'est jamais plus risqué que la somme de ses
    parties. » C'est la traduction mathématique de l'idée que diversifier
    ne peut pas nuire.

    **La VaR ne satisfait pas cet axiome.** Voici pourquoi, avec des
    nombres que tu peux vérifier de tête.

    Prends deux obligations indépendantes, chacune avec 4 % de
    probabilité de défaut. On mesure la VaR à 95 %.

      • Obligation seule : la probabilité de perte est 4 %, donc inférieure
        au seuil de 5 %. Le 95ᵉ percentile de la perte est donc ZÉRO.
        VaR(A) = VaR(B) = 0.

      • Portefeuille moitié-moitié : la probabilité qu'au moins une des
        deux fasse défaut vaut 1 - 0,96² = 7,84 %, supérieure à 5 %. Le
        95ᵉ percentile tombe donc dans la zone « au moins un défaut ».
        VaR(A+B) = 0,5 (on perd la moitié du capital).

    Résultat : VaR(A+B) = 0,5 > 0 = VaR(A) + VaR(B).

    **La VaR affirme que diversifier a créé du risque à partir de rien.**

    Ce n'est pas une approximation ni un artefact numérique : c'est une
    conséquence directe du fait que la VaR est un quantile, et qu'un
    quantile ignore tout de ce qui se passe au-delà de lui. La CVaR, elle,
    est sous-additive — c'est la raison de fond pour laquelle Bâle est
    passé de la VaR à l'expected shortfall dans le cadre FRTB.
    """
    p = default_probability
    lgd = loss_given_default

    # Une seule obligation : perte = lgd avec proba p, 0 sinon.
    var_seule = lgd if p > (1 - confidence) else 0.0

    # Deux obligations indépendantes, 50/50.
    p_aucun = (1 - p) ** 2
    p_au_moins_un = 1 - p_aucun

    if p_au_moins_un > (1 - confidence):
        # Le quantile tombe dans la zone « au moins un défaut ».
        # Deux défauts simultanés : proba p². Si elle reste sous le seuil,
        # le quantile correspond à exactement un défaut → perte lgd/2.
        var_combine = lgd if p**2 > (1 - confidence) else lgd / 2
    else:
        var_combine = 0.0

    return {
        "var_actif_seul": var_seule,
        "somme_des_var": 2 * var_seule / 2,  # deux demi-positions
        "var_portefeuille_combine": var_combine,
        "proba_au_moins_un_defaut": p_au_moins_un,
        "sous_additivite_violee": var_combine > var_seule,
    }


def _clean(returns: pd.Series, min_obs: int, quoi: str) -> np.ndarray:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    r = returns.dropna().to_numpy(dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError(f"valeurs non finies dans les rendements ({quoi})")
    if len(r) < min_obs:
        raise InsufficientData(len(r), min_obs, quoi)
    return r
