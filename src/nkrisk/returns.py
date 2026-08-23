"""
Rendements, volatilité, et leurs incertitudes.

C'est la fondation de tout le reste. Chaque mesure de risque que nous
construirons ensuite (VaR, contribution au risque, exposition factorielle,
stress test) part d'une série de rendements. Une erreur ici se propage
partout et est presque impossible à détecter en aval.

LES QUATRE CHOSES À COMPRENDRE DANS CE FICHIER
==============================================

1. Il existe DEUX définitions du rendement, et choisir la mauvaise est le
   bug silencieux le plus fréquent de la finance appliquée.

2. La volatilité s'annualise en √252, et ce n'est pas une convention
   arbitraire : c'est une conséquence de l'additivité de la variance.

3. Une volatilité estimée sur un an de données a une marge d'erreur
   d'environ ±9 % en relatif. Presque aucun logiciel grand public ne
   te le dit.

4. La volatilité te COÛTE du rendement, mécaniquement, même quand elle
   est parfaitement symétrique. C'est la « traînée de volatilité ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .types import Estimate, InsufficientData

TRADING_DAYS_PER_YEAR = 252


# ══════════════════════════════════════════════════════════════════════
#  1. LES DEUX RENDEMENTS
# ══════════════════════════════════════════════════════════════════════
#
# Rendement SIMPLE (ou arithmétique) :   r = P_t / P_{t-1} - 1
# Rendement LOGARITHMIQUE (ou continu) : l = ln(P_t / P_{t-1})
#
# Ils sont liés par : l = ln(1 + r)  et  r = e^l - 1
#
# Pour de petites variations ils sont presque égaux (ln(1+x) ≈ x quand x
# est petit) : +1 % simple = +0,995 % log. C'est précisément ce qui rend
# l'erreur si dangereuse — elle est invisible sur un jour, et énorme sur
# dix ans.
#
# QUAND UTILISER LEQUEL — la règle qui résout tout :
#
#   • Additivité dans le TEMPS      → rendements LOG
#     ln(P_2/P_0) = ln(P_1/P_0) + ln(P_2/P_1). On additionne.
#     Donc : cumul d'une performance, volatilité, simulation.
#
#   • Additivité entre ACTIFS       → rendements SIMPLES
#     Le rendement d'un portefeuille est la moyenne pondérée des
#     rendements simples de ses composantes. Ce n'est PAS vrai pour
#     les rendements log.
#     Donc : agrégation de portefeuille, contribution d'une position.
#
# Un moteur de risque a besoin des deux. Un moteur qui n'en implémente
# qu'un seul contient nécessairement un bug quelque part.
# ══════════════════════════════════════════════════════════════════════


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Rendements simples : r_t = P_t / P_{t-1} - 1.

    À utiliser pour agréger entre actifs (portefeuille).
    """
    _validate_prices(prices)
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Rendements logarithmiques : l_t = ln(P_t / P_{t-1}).

    À utiliser pour agréger dans le temps (cumul, volatilité, simulation).
    """
    _validate_prices(prices)
    return np.log(prices / prices.shift(1)).dropna()


def _validate_prices(prices: pd.Series | pd.DataFrame) -> None:
    """
    Un prix nul ou négatif casse le logarithme et n'a pas de sens
    économique pour une action. On échoue tôt, bruyamment, avec un
    message qui dit quoi faire.
    """
    if len(prices) < 2:
        raise InsufficientData(len(prices), 2, "rendements")
    values = prices.to_numpy()
    if np.any(~np.isfinite(values)):
        raise ValueError(
            "La série de prix contient des NaN ou des infinis. "
            "Nettoie les données avant : soit tu supprimes ces dates, "
            "soit tu interpoles — mais la décision t'appartient, ce n'est "
            "pas au moteur de la prendre à ta place en silence."
        )
    if np.any(values <= 0):
        raise ValueError(
            "La série contient des prix nuls ou négatifs. Le rendement "
            "logarithmique n'est pas défini. Vérifie ta source de données."
        )


# ══════════════════════════════════════════════════════════════════════
#  2. VOLATILITÉ, ET SA MARGE D'ERREUR
# ══════════════════════════════════════════════════════════════════════


def annualized_volatility(
    returns: pd.Series,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_obs: int = 20,
) -> Estimate:
    """
    Volatilité annualisée, avec intervalle de confiance exact.

    POURQUOI √252
    -------------
    Si les rendements sont indépendants, la variance d'une somme est la
    somme des variances :

        Var(r_1 + ... + r_252) = 252 · Var(r_quotidien)

    L'écart-type est la racine de la variance, donc :

        σ_annuel = √252 · σ_quotidien

    Toute la « règle de la racine du temps » sort de là. Et sa faiblesse
    aussi : elle suppose l'INDÉPENDANCE. Les vrais marchés ont du
    regroupement de volatilité (les grosses journées se suivent), ce qui
    viole cette hypothèse et fait généralement SOUS-ESTIMER le risque à
    long horizon. On corrigera ça dans une séance ultérieure.

    L'INTERVALLE DE CONFIANCE
    -------------------------
    Sous hypothèse de normalité, on a le résultat exact :

        (n-1)·s² / σ²  ~  χ²(n-1)

    d'où on inverse pour obtenir un intervalle sur σ². C'est un intervalle
    EXACT (pas asymptotique) — mais exact *conditionnellement à la
    normalité*, qui est fausse sur les vrais marchés. L'intervalle réel est
    donc plus large que celui qu'on affiche. On le note dans `assumptions`.

    L'ORDRE DE GRANDEUR À RETENIR
    -----------------------------
    Avec un an de données quotidiennes (n = 252), l'incertitude relative
    sur la volatilité est d'environ ±9 % à 95 %. Autrement dit : une vol
    estimée à 20 % est en réalité « quelque part entre 18,4 % et 21,9 % ».

    Avec un seul mois (n = 21), la fourchette explose à peu près à
    [15 %, 30 %]. C'est pourquoi estimer une volatilité sur un mois de
    données ne veut à peu près rien dire — et c'est exactement ce que font
    beaucoup d'applications de courtage.
    """
    r = _clean_returns(returns, min_obs, "volatilité")
    n = len(r)

    # ddof=1 : estimateur non biaisé de la variance (division par n-1, pas n).
    # La correction de Bessel. Sur 252 points l'écart est de 0,2 % — mais un
    # moteur de risque qui utilise ddof=0 est faux, point.
    s_daily = float(np.std(r, ddof=1))

    scale = np.sqrt(periods_per_year)
    point = s_daily * scale

    alpha = 1.0 - confidence
    df = n - 1

    # Var ∈ [(n-1)s²/χ²_{1-α/2}, (n-1)s²/χ²_{α/2}]
    # Attention à l'inversion : un grand quantile du chi2 au dénominateur
    # donne la borne BASSE. C'est contre-intuitif et c'est une source de
    # bug classique — d'où le test dédié dans tests/test_returns.py.
    chi2_hi = stats.chi2.ppf(1 - alpha / 2, df)
    chi2_lo = stats.chi2.ppf(alpha / 2, df)

    var_daily = s_daily**2
    var_low = df * var_daily / chi2_hi
    var_high = df * var_daily / chi2_lo

    return Estimate(
        value=point,
        ci_low=float(np.sqrt(var_low) * scale),
        ci_high=float(np.sqrt(var_high) * scale),
        confidence=confidence,
        n_obs=n,
        method="chi2-exact",
        assumptions=(
            "rendements indépendants et identiquement distribués",
            "rendements normaux (faux en pratique : queues épaisses)",
            "volatilité constante sur la période (faux : regroupement)",
        ),
    )


# ══════════════════════════════════════════════════════════════════════
#  3. RENDEMENT MOYEN — ET POURQUOI ON NE PEUT PAS L'ESTIMER
# ══════════════════════════════════════════════════════════════════════


def annualized_mean_return(
    returns: pd.Series,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_obs: int = 20,
) -> Estimate:
    """
    Rendement moyen annualisé (arithmétique), avec intervalle de confiance.

    ⚠️ LA LEÇON LA PLUS IMPORTANTE DE CE PROJET ⚠️

    Regarde la largeur de l'intervalle que cette fonction retourne, puis
    compare-la à celle de `annualized_volatility` sur les mêmes données.

    Tu vas constater quelque chose de dérangeant : la volatilité s'estime
    correctement avec un an de données, alors que le rendement espéré ne
    s'estime PAS, même avec vingt ans.

    Pourquoi, mathématiquement :
      • erreur-type de la moyenne     = σ / √n        → décroît en 1/√n
      • erreur-type de la volatilité  ≈ σ / √(2n)     → décroît aussi en 1/√n
    Ça a l'air similaire. La différence est dans l'ÉCHELLE du signal :
    une action a typiquement μ ≈ 8 % et σ ≈ 20 %. Le rapport signal/bruit
    de la moyenne est donc μ/σ ≈ 0,4, contre ≈ 1,41·√2 pour la vol.

    Concrètement, pour distinguer un rendement espéré de 8 % d'un rendement
    espéré de 4 % avec 95 % de confiance et σ = 20 %, il te faut de l'ordre
    de **quatre-vingt-dix ans** de données.

    CONSÉQUENCE DIRECTE, et c'est celle qui devrait tuer 90 % des projets
    de « logiciel qui dit quoi acheter » : toute méthode qui repose sur
    l'estimation des rendements futurs à partir du passé repose sur une
    quantité qu'on ne sait pas mesurer. C'est la raison profonde pour
    laquelle l'optimisation de portefeuille de Markowitz est surnommée un
    « maximiseur d'erreur » (Michaud, 1989), et pourquoi le portefeuille
    naïf 1/N bat quatorze modèles sophistiqués hors échantillon
    (DeMiguel, Garlappi & Uppal, 2009).

    Le risque se mesure. Le rendement, non. Un moteur honnête traite ces
    deux grandeurs très différemment.
    """
    r = _clean_returns(returns, min_obs, "rendement moyen")
    n = len(r)

    mean_daily = float(np.mean(r))
    s_daily = float(np.std(r, ddof=1))

    point = mean_daily * periods_per_year

    # Intervalle de Student sur la moyenne. On utilise t plutôt que la
    # normale parce que σ est estimé, pas connu.
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    se_daily = s_daily / np.sqrt(n)
    half = t_crit * se_daily * periods_per_year

    return Estimate(
        value=point,
        ci_low=point - half,
        ci_high=point + half,
        confidence=confidence,
        n_obs=n,
        method="student-t",
        assumptions=(
            "rendements indépendants et identiquement distribués",
            "espérance de rendement constante dans le temps (très douteux)",
            "l'annualisation linéaire suppose des rendements simples",
        ),
    )


def geometric_return(
    prices: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Rendement géométrique annualisé — le rendement RÉELLEMENT réalisé.

    C'est le taux constant qui, appliqué sur toute la période, aurait
    produit le prix final observé :

        (P_fin / P_début)^(252/n) - 1

    Pas d'intervalle de confiance ici, parce que ce n'est pas une
    estimation : c'est un fait comptable sur le passé observé. La nuance
    est essentielle et beaucoup de gens la manquent — « ce que j'ai gagné »
    et « ce que j'espère gagner » sont deux objets de nature différente.
    """
    _validate_prices(prices)
    n_periods = len(prices) - 1
    total = float(prices.iloc[-1] / prices.iloc[0])
    return total ** (periods_per_year / n_periods) - 1.0


def compound_annualize(
    mean_periodic_return: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualisation par COMPOSITION : (1 + r)^252 - 1.

    ⚠️ CE PETIT BOUT DE CODE VIENT D'UN VRAI BUG QUE J'AI COMMIS ⚠️

    Il existe deux façons d'annualiser un rendement moyen quotidien :

        linéaire      :  r_annuel = r_quotidien × 252
        composée      :  r_annuel = (1 + r_quotidien)^252 - 1

    Elles ne donnent PAS le même résultat. Pour r = 8 % annuel, l'écart
    est d'environ 0,33 point. C'est petit — et c'est précisément le
    problème.

    Ce qui s'est passé dans ma première version de la démonstration :
    je comparais un rendement arithmétique annualisé LINÉAIREMENT à un
    rendement géométrique annualisé par COMPOSITION. L'écart de
    convention (~0,33 pt) était du même ordre que l'effet que je voulais
    montrer (la traînée de volatilité, 0,13 pt à σ = 5 %). Résultat : la
    traînée apparaissait NÉGATIVE, c'est-à-dire que le graphique
    « démontrait » l'inverse de la réalité.

    Aucun test ne l'a attrapé, parce que mes tests portaient sur des
    volatilités élevées où le vrai effet écrase l'artefact. Je ne l'ai vu
    qu'en REGARDANT le tableau de sortie et en constatant qu'un signe
    n'allait pas.

    LA LEÇON, et elle vaut plus que le code :
      1. Un bug de convention ne plante pas. Il produit un nombre
         plausible, du bon ordre de grandeur, et faux.
      2. Les tests unitaires ne le voient pas si tu les écris dans le
         même régime que celui où l'artefact est masqué.
      3. La seule défense est de vérifier le SIGNE et l'ORDRE DE GRANDEUR
         contre une prédiction théorique indépendante — ici σ²/2.

    C'est pour ça qu'un moteur de risque a besoin de valeurs de référence
    calculées à la main, pas seulement de tests de non-régression.
    """
    return (1.0 + mean_periodic_return) ** periods_per_year - 1.0


def volatility_drag(
    arithmetic_mean: float,
    volatility: float,
) -> float:
    """
    La traînée de volatilité : combien la variabilité te coûte, mécaniquement.

    Approximation classique :

        rendement_géométrique  ≈  rendement_arithmétique  -  σ²/2

    LE CONCEPT, en une phrase : gagner 50 % puis perdre 50 % ne te ramène
    pas à zéro, ça te laisse à -25 %. La moyenne arithmétique de +50 % et
    -50 % vaut 0. Ta richesse, elle, a fondu d'un quart.

    C'est ce -σ²/2 qu'on a croisé dans synthetic.py sous le nom de
    correction d'Itô. Ce n'est pas une coïncidence : c'est le même terme.

    Ordre de grandeur : σ = 20 % → traînée = 0,20²/2 = 2 points de
    rendement annuel perdus. σ = 40 % → 8 points. La traînée croît en
    CARRÉ de la volatilité, ce qui rend les actifs très volatils bien plus
    coûteux qu'ils n'en ont l'air.

    C'est l'argument quantitatif le plus solide en faveur de la
    diversification : réduire σ augmente ton rendement composé même si
    le rendement espéré ne bouge pas d'un pouce.
    """
    return arithmetic_mean - 0.5 * volatility**2


# ══════════════════════════════════════════════════════════════════════
#  4. RATIO DE SHARPE — avec l'incertitude que personne n'affiche
# ══════════════════════════════════════════════════════════════════════


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_obs: int = 60,
) -> Estimate:
    """
    Ratio de Sharpe annualisé, avec son intervalle de confiance.

    SR = (rendement moyen - taux sans risque) / volatilité

    L'erreur-type suit le résultat de Lo (2002), « The Statistics of
    Sharpe Ratios », pour des rendements iid :

        SE(SR) ≈ √[ (1 + SR²/2) / n ]      (SR exprimé par période)

    POURQUOI TU DOIS CONNAÎTRE CE CHIFFRE
    -------------------------------------
    Prends une stratégie avec un Sharpe annualisé de 1,0 mesuré sur un an
    de données quotidiennes. L'intervalle de confiance à 95 % est d'environ
    [-0,24 ; 2,24]. Il CONTIENT ZÉRO.

    Traduction : après un an de résultats, tu ne peux pas distinguer
    statistiquement une stratégie excellente d'une stratégie sans aucune
    valeur. C'est la raison pour laquelle l'industrie exige trois ans
    d'historique minimum avant de prendre un gestionnaire au sérieux — et
    c'est le résultat du « minimum track record length » de Bailey et
    López de Prado.

    ET LE PIÈGE FATAL PAR-DESSUS
    ----------------------------
    Tout ce qui précède suppose UN SEUL essai. Si tu testes 200 variantes
    de stratégie et que tu retiens la meilleure, son Sharpe apparent est
    contaminé par le biais de sélection. Le « théorème des fausses
    stratégies » (Bailey & López de Prado) établit qu'avec assez d'essais,
    **il n'existe aucun Sharpe assez élevé pour rejeter l'hypothèse que la
    stratégie est fausse**.

    On implémentera le Deflated Sharpe Ratio, qui corrige de ce biais,
    dans une séance ultérieure. Retiens dès maintenant : ton backtest ne
    mesure pas la qualité de ta stratégie, il mesure le nombre de fois où
    tu as regardé.
    """
    r = _clean_returns(returns, min_obs, "ratio de Sharpe")
    n = len(r)

    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period

    mean_excess = float(np.mean(excess))
    std_excess = float(np.std(excess, ddof=1))

    if std_excess == 0:
        raise ValueError(
            "Volatilité nulle : le ratio de Sharpe n'est pas défini. "
            "Vérifie que ta série de prix bouge réellement."
        )

    sr_period = mean_excess / std_excess
    sr_annual = sr_period * np.sqrt(periods_per_year)

    # Lo (2002), rendements iid.
    se_period = np.sqrt((1.0 + 0.5 * sr_period**2) / n)
    se_annual = se_period * np.sqrt(periods_per_year)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    half = z * se_annual

    return Estimate(
        value=float(sr_annual),
        ci_low=float(sr_annual - half),
        ci_high=float(sr_annual + half),
        confidence=confidence,
        n_obs=n,
        method="lo-2002-iid",
        assumptions=(
            "rendements indépendants et identiquement distribués",
            "UN SEUL essai — non corrigé du biais de sélection",
            "approximation asymptotique (peu fiable si n < 60)",
        ),
    )


def _clean_returns(returns: pd.Series, min_obs: int, what: str) -> np.ndarray:
    """Validation commune. Échoue tôt, échoue bruyamment."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    r = returns.dropna().to_numpy(dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError(f"valeurs non finies dans les rendements ({what})")
    if len(r) < min_obs:
        raise InsufficientData(len(r), min_obs, what)
    return r
