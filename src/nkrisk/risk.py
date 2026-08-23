"""
Décomposition du risque d'un portefeuille.

C'EST ICI QUE LE MOTEUR DEVIENT UN ALADDIN
==========================================

Jusqu'à présent on décrivait des actifs isolés. À partir de maintenant on
décrit un PORTEFEUILLE — et la question centrale n'est plus « quel est le
risque de cette position ? » mais :

    « Combien cette position ajoute-t-elle au risque de l'ensemble,
      compte tenu de tout ce que je détiens déjà ? »

Ce n'est pas la même question, et les réponses diffèrent souvent du tout
au tout. Une position très volatile mais décorrélée du reste peut RÉDUIRE
le risque total. Une position peu volatile mais parfaitement corrélée à
ton exposition dominante peut l'augmenter bien plus que sa taille ne le
suggère.

C'est exactement ce que fournit Aladdin à un gérant, et c'est le calcul
que presque aucun outil grand public n'expose.

LA MATHÉMATIQUE, ET POURQUOI ELLE EST ÉLÉGANTE
----------------------------------------------
Le risque du portefeuille :

    σ_p = √(wᵀΣw)

Cette fonction est **homogène de degré 1** en w : doubler toutes les
positions double le risque. Le théorème d'Euler sur les fonctions
homogènes donne alors :

    σ_p = Σ_i  w_i · ∂σ_p/∂w_i

Autrement dit, le risque total se décompose EXACTEMENT en une somme de
contributions par position. Pas approximativement — exactement. C'est
une identité, et on en fait un test.

Les deux quantités qui en découlent :

  • Contribution MARGINALE  ∂σ_p/∂w_i = (Σw)_i / σ_p
    Le risque ajouté par un euro supplémentaire dans la position i.
    C'est une DÉRIVÉE : elle répond à « et si j'en achetais un peu plus ? »

  • Contribution COMPOSANTE  c_i = w_i · (Σw)_i / σ_p
    La part du risque total imputable à la position i.
    C'est une ATTRIBUTION : elle répond à « d'où vient mon risque ? »

Confondre les deux est l'erreur classique. La marginale sert à décider
d'un arbitrage ; la composante sert à comprendre une exposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .covariance import CovarianceEstimate

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class RiskDecomposition:
    """Résultat complet d'une décomposition. Toutes valeurs annualisées."""

    portfolio_volatility: float
    weights: pd.Series
    marginal: pd.Series
    component: pd.Series
    percent: pd.Series
    standalone: pd.Series
    diversification_ratio: float
    effective_bets: float

    def table(self) -> pd.DataFrame:
        """Vue synthétique, triée par contribution décroissante."""
        df = pd.DataFrame(
            {
                "poids": self.weights,
                "vol seule": self.standalone,
                "contrib marginale": self.marginal,
                "contrib composante": self.component,
                "% du risque": self.percent,
            }
        )
        return df.sort_values("% du risque", ascending=False)

    def __str__(self) -> str:
        lignes = [
            f"Volatilité du portefeuille : {self.portfolio_volatility:.2%}",
            f"Ratio de diversification   : {self.diversification_ratio:.2f}",
            f"Nombre effectif de paris   : {self.effective_bets:.1f} "
            f"(sur {len(self.weights)} positions)",
        ]
        return "\n".join(lignes)


def decompose_risk(
    weights: pd.Series,
    covariance: CovarianceEstimate,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> RiskDecomposition:
    """
    Décompose le risque du portefeuille position par position.

    `weights` doit être indexé par les mêmes noms d'actifs que la
    covariance. On réaligne explicitement plutôt que de supposer le même
    ordre — supposer l'ordre est une des erreurs les plus insidieuses de
    ce domaine, parce qu'elle produit un résultat parfaitement plausible
    et complètement faux.
    """
    actifs = list(covariance.assets)

    manquants = set(actifs) - set(weights.index)
    if manquants:
        raise ValueError(
            f"poids absents pour : {sorted(manquants)}. La covariance et les "
            f"poids doivent porter sur les mêmes actifs."
        )

    w = weights.reindex(actifs).to_numpy(dtype=float)
    sigma = covariance.matrix

    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"les poids somment à {w.sum():.6f}, attendu 1. Si tu veux "
            f"modéliser du levier ou du cash, ajoute-les comme positions "
            f"explicites plutôt que de laisser la somme dériver."
        )

    ann = periods_per_year

    # σ_p² = wᵀΣw  — la forme quadratique fondamentale
    var_p = float(w @ sigma @ w)
    if var_p <= 0:
        raise ValueError(
            "variance de portefeuille nulle ou négative. La matrice de "
            "covariance n'est pas définie positive — contracte-la."
        )
    vol_p = np.sqrt(var_p * ann)

    # Contribution marginale : ∂σ_p/∂w = Σw/σ_p
    sigma_w = sigma @ w
    marginal = sigma_w / np.sqrt(var_p) * np.sqrt(ann)

    # Contribution composante : c_i = w_i · marginale_i
    component = w * marginal

    # Vérification de l'identité d'Euler. Ce n'est pas de la paranoïa :
    # si cette égalité ne tient pas, il y a une erreur d'algèbre en amont
    # et tous les chiffres produits sont faux.
    if not np.isclose(component.sum(), vol_p, rtol=1e-9):
        raise AssertionError(
            f"identité d'Euler violée : Σc_i = {component.sum():.10f} "
            f"≠ σ_p = {vol_p:.10f}. Bug dans la décomposition."
        )

    percent = component / vol_p

    # Volatilités individuelles annualisées
    standalone = np.sqrt(np.diag(sigma) * ann)

    # RATIO DE DIVERSIFICATION (Choueifaty & Coignard, 2008)
    #   DR = (Σ w_i σ_i) / σ_p
    # Moyenne pondérée des volatilités individuelles, divisée par la
    # volatilité effectivement obtenue. Vaut 1 si tout est parfaitement
    # corrélé — c'est-à-dire si la diversification n'apporte rien. Plus
    # il est élevé, plus les corrélations travaillent pour toi.
    div_ratio = float((w @ standalone) / vol_p)

    enb = effective_number_of_bets(w, sigma)

    idx = pd.Index(actifs, name="actif")
    return RiskDecomposition(
        portfolio_volatility=float(vol_p),
        weights=pd.Series(w, index=idx),
        marginal=pd.Series(marginal, index=idx),
        component=pd.Series(component, index=idx),
        percent=pd.Series(percent, index=idx),
        standalone=pd.Series(standalone, index=idx),
        diversification_ratio=div_ratio,
        effective_bets=enb,
    )


def effective_number_of_bets(weights: np.ndarray, sigma: np.ndarray) -> float:
    """
    Nombre effectif de paris — entropie des composantes principales.

    Référence : Meucci (2009), « Managing Diversification », Risk 22(5).

    ⚠️ CETTE FONCTION EXISTE PARCE QUE MA PREMIÈRE VERSION ÉTAIT FAUSSE ⚠️

    J'avais d'abord écrit l'inverse de l'indice de Herfindahl appliqué aux
    contributions au risque :

        ENB = 1 / Σ (c_i / σ_p)²                    ← FAUX

    Un test l'a démoli. Prends huit actifs corrélés à 0,9999, pondérés
    également : par symétrie, chacun contribue exactement 1/8 du risque,
    donc cette formule retourne 8. Or il n'y a manifestement **qu'un seul
    pari**, détenu huit fois.

    L'erreur conceptuelle : cette formule mesure la CONCENTRATION du
    risque entre les lignes, pas la DIVERSIFICATION. Ce sont deux choses
    différentes, et le portefeuille parfaitement corrélé est précisément
    le contre-exemple qui les sépare — risque parfaitement réparti entre
    les positions, et pourtant aucune diversification.

    LA BONNE APPROCHE
    -----------------
    Il faut compter les sources de risque INDÉPENDANTES, pas les lignes.
    On diagonalise donc la covariance pour passer dans une base où les
    facteurs sont décorrélés par construction :

        Σ = E Λ Eᵀ                (décomposition en valeurs propres)
        w̃ = Eᵀ w                  (poids dans la base des composantes)
        p_i = w̃_i² λ_i / σ_p²     (part de variance du facteur i, Σp_i = 1)
        ENB = exp(−Σ p_i ln p_i)  (exponentielle de l'entropie de Shannon)

    Vérification sur les deux cas limites, faisable de tête :
      • corrélation parfaite → une seule valeur propre porte toute la
        variance → p = (1,0,…,0) → entropie 0 → ENB = 1 ✓
      • actifs indépendants, équipondérés → p_i = 1/N pour tout i →
        entropie = ln N → ENB = N ✓

    L'exponentielle de l'entropie est la mesure canonique du « nombre
    effectif » d'une distribution — la même qui donne la perplexité en
    traitement du langage. Ce n'est pas un choix arbitraire.

    ⚠️ LA LIMITE DE CET INDICATEUR — À CONNAÎTRE AVANT DE L'UTILISER ⚠️

    L'ENB dépend de la BASE de facteurs choisie, et l'analyse en
    composantes principales n'est qu'un choix parmi d'autres. Quand
    plusieurs valeurs propres sont égales, la décomposition n'est pas
    unique — n'importe quelle rotation du sous-espace convient — et le
    résultat devient arbitraire.

    Mesuré sur dix actifs de volatilité identique, équipondérés :

        ρ = 0      → ENB = 10
        ρ = 0,001  → ENB = 1
        ρ = 0,90   → ENB = 1

    Une discontinuité brutale entre ρ = 0 et ρ = 0,001. Ce n'est pas un
    bug : c'est mathématiquement correct. Le portefeuille équipondéré
    d'actifs parfaitement symétriques est exactement aligné sur la
    première composante principale, donc il charge un seul facteur —
    quelle que soit la valeur de ρ, dès qu'elle est non nulle.

    Sur des données réalistes — volatilités hétérogènes, poids inégaux —
    le comportement redevient lisse et monotone :

        ρ = 0 → 4,98   ρ = 0,3 → 1,48   ρ = 0,7 → 1,10   ρ = 0,99 → 1,00

    RETIENS DONC : cet indicateur est fiable sur des portefeuilles réels
    et dégénéré sur des cas parfaitement symétriques. Meucci et ses
    coauteurs ont publié en 2015 une variante dite « torsion minimale »
    précisément pour lever cette ambiguïté de base — je l'ai testée ici,
    elle règle le cas ρ = 0 mais en casse d'autres. Il n'existe pas de
    définition universellement correcte du nombre de paris.

    C'est une leçon plus générale que ce fichier : un indicateur publié,
    utilisé en production, peut avoir un domaine de validité limité que
    l'article d'origine ne met pas en avant. Le mesurer soi-même sur des
    cas extrêmes connus est la seule façon de le découvrir.
    """
    var_p = float(weights @ sigma @ weights)
    if var_p <= 0:
        return float("nan")

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)

    # Poids dans la base propre
    w_tilde = eigenvectors.T @ weights

    # Part de variance portée par chaque facteur indépendant
    contributions = (w_tilde**2) * eigenvalues
    p = contributions / contributions.sum()

    # Les valeurs propres numériquement nulles donnent des p négatifs
    # minuscules : on les écarte plutôt que de laisser ln() produire un NaN.
    p = p[p > 1e-15]
    if len(p) == 0:
        return float("nan")

    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def portfolio_volatility(
    weights: pd.Series,
    covariance: CovarianceEstimate,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Volatilité annualisée du portefeuille. σ_p = √(wᵀΣw · 252)."""
    w = weights.reindex(list(covariance.assets)).to_numpy(dtype=float)
    return float(np.sqrt(w @ covariance.matrix @ w * periods_per_year))


def minimum_variance_weights(covariance: CovarianceEstimate) -> pd.Series:
    """
    Portefeuille de variance minimale : w = Σ⁻¹1 / (1ᵀΣ⁻¹1).

    ⚠️ CETTE FONCTION EXISTE POUR MONTRER UN DANGER, PAS POUR ÊTRE UTILISÉE
    NAÏVEMENT ⚠️

    C'est le seul portefeuille « optimal » qui ne requiert PAS d'estimer
    les rendements espérés — donc le seul qui échappe au problème de la
    séance 2 (leçon 2 : le rendement ne s'estime pas). À ce titre, il est
    bien plus défendable que le Markowitz classique.

    Mais il inverse la matrice de covariance. Si celle-ci est mal
    conditionnée, l'inversion amplifie le bruit et produit des poids
    aberrants : des positions énormes, souvent vendeuses, sur les actifs
    dont la covariance est le plus mal estimée.

    La démonstration de la séance 2 mesure exactement cet effet, avec et
    sans contraction. Regarde la différence avant de te servir de cette
    fonction pour quoi que ce soit.
    """
    if not covariance.is_invertible:
        raise ValueError(
            f"matrice singulière (N={covariance.n_assets}, "
            f"T={covariance.n_obs}, N/T={covariance.ratio_n_over_t:.2f}). "
            f"Utilise ledoit_wolf() plutôt que sample_covariance()."
        )

    n = covariance.n_assets
    ones = np.ones(n)
    # np.linalg.solve plutôt que np.linalg.inv : résoudre un système est
    # numériquement plus stable qu'inverser puis multiplier. Règle
    # générale en calcul numérique — on n'inverse jamais une matrice si
    # on peut résoudre un système à la place.
    inv_ones = np.linalg.solve(covariance.matrix, ones)
    w = inv_ones / (ones @ inv_ones)
    return pd.Series(w, index=pd.Index(covariance.assets, name="actif"))


def equal_weights(covariance: CovarianceEstimate) -> pd.Series:
    """
    Le portefeuille 1/N — à ne surtout pas mépriser.

    DeMiguel, Garlappi & Uppal (2009), « Optimal Versus Naive
    Diversification », comparent 1/N à quatorze modèles d'optimisation
    sophistiqués sur plusieurs jeux de données. **1/N les bat hors
    échantillon**, sur le ratio de Sharpe comme sur l'équivalent-certitude.

    La raison est celle de la séance 1 : les modèles sophistiqués ont
    besoin d'estimer des rendements espérés et des covariances, et
    l'erreur d'estimation coûte plus cher que ce que l'optimisation
    rapporte. 1/N n'estime rien, donc ne se trompe sur rien.

    C'est ta référence de comparaison obligatoire. Toute méthode que tu
    inventeras devra battre 1/N hors échantillon — et ce sera plus dur
    que tu ne le crois.
    """
    n = covariance.n_assets
    return pd.Series(
        np.full(n, 1.0 / n), index=pd.Index(covariance.assets, name="actif")
    )
