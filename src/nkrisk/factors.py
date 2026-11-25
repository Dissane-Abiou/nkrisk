"""
Modèles à facteurs — la réponse structurelle au problème de la séance 2.

CE QUI N'ALLAIT TOUJOURS PAS APRÈS LEDOIT-WOLF
==============================================

La séance 2 a mesuré le désastre : pour N actifs, la covariance compte
N(N+1)/2 paramètres, estimés sur T observations. À N=100 et T=252, cela
fait 5 050 paramètres pour 252 points. La matrice empirique est mal
conditionnée bien avant d'être singulière, et l'inverser amplifie le
bruit.

La contraction de Ledoit-Wolf **régularise** ce problème. Elle ne le
résout pas. Elle tire l'estimation vers une cible rigide pour réduire la
variance, en acceptant un biais. C'est un pansement statistique
excellent, et c'est tout ce qu'il est : le nombre de paramètres reste
N(N+1)/2. On a simplement décidé de moins croire ce qu'on a estimé.

Un modèle à facteurs fait quelque chose de différent en nature. Il
**réduit le nombre de paramètres** en imposant une structure :

    Σ  ≈  B Bᵀ + D

où B est une matrice N×k d'expositions à k facteurs, et D une matrice
diagonale de variances spécifiques. Le décompte :

    covariance libre   N(N+1)/2        → 500 actifs : 125 250
    modèle à k=5       N·k + N         → 500 actifs :   3 000

Quarante fois moins. Et cette réduction n'est pas un artifice de calcul :
elle encode une hypothèse sur le monde — **les actifs bougent ensemble
parce qu'ils partagent un petit nombre de causes communes.** Si cette
hypothèse est juste, on ne perd presque rien et on gagne énormément. Si
elle est fausse, le modèle est faux, et aucune astuce numérique ne le
sauvera.

C'est la différence entre régulariser et modéliser. Ledoit-Wolf ne dit
rien sur le monde. Un modèle à facteurs fait une affirmation réfutable.

CE QU'ON RÉGULARISE, ET CE QU'ON NE RÉGULARISE PAS
--------------------------------------------------
Décision de conception importante, et facile à manquer.

Le modèle travaille sur la matrice de **corrélation**, pas sur la
covariance. Les volatilités individuelles sont reprises telles quelles
et la diagonale de la matrice reconstruite est exactement préservée.

Pourquoi : estimer une variance, c'est estimer UN paramètre avec T
observations — c'est facile, et la séance 1 l'a mesuré (±2,8 %
d'imprécision relative sur dix ans). Estimer une corrélation, c'est
estimer N(N−1)/2 paramètres avec les mêmes T observations — c'est là
qu'est tout le problème.

**On ne régularise que ce qui est réellement difficile.** Appliquer la
structure à facteurs aux variances aussi déformerait des grandeurs qu'on
sait déjà bien estimer, pour aucun gain.

COMBIEN DE FACTEURS ? LE SEUL VRAI CHOIX DU MODÈLE
--------------------------------------------------
Tout le reste découle de k. Trop peu : on jette de la structure réelle.
Trop : on modélise du bruit, et on retrouve le problème qu'on fuyait.

La réponse paresseuse est « assez pour expliquer 80 % de la variance » —
un seuil arbitraire déguisé en critère. La réponse de ce module vient de
la théorie des matrices aléatoires.

Pour une matrice de T×N observations de **bruit pur** standardisé, les
valeurs propres de la matrice de corrélation empirique ne sont pas
toutes égales à 1. Elles s'étalent, et la loi de Marchenko-Pastur dit
exactement sur quel intervalle : [(1−√q)², (1+√q)²] avec q = N/T.

C'est un résultat remarquable, et son usage ici est direct : **toute
valeur propre au-dessus de (1+√q)² ne peut pas être produite par du
bruit.** Elle signale une structure réelle. On compte ces valeurs
propres, et on a k.

Pas de seuil arbitraire. Pas de validation croisée. Une borne calculée
à partir des dimensions du problème, et une comparaison.

Référence : Laloux, Cizeau, Bouchaud & Potters (1999), « Noise Dressing
of Financial Correlation Matrices », Physical Review Letters 83(7).

LA LIMITE QUE J'AI DÉCOUVERTE EN ÉCRIVANT LE TEST INVERSE
---------------------------------------------------------
J'avais écrit un test qui affirmait l'évidence : « l'erreur doit
décroître quand l'historique s'allonge ». Il a échoué, et c'était lui
qui avait raison.

À univers fixé, l'erreur de reconstruction PLAFONNE quand on allonge
l'historique. Le dernier quadruplement de T ne gagne plus rien de
mesurable. Ce n'est pas du bruit résiduel, c'est un BIAIS.

**L'ACP n'est pas de l'analyse factorielle.** Elle suppose
implicitement que toutes les variances spécifiques sont égales — c'est
sous cette hypothèse qu'elle coïncide avec l'estimateur du maximum de
vraisemblance d'un modèle factoriel. Quand elles diffèrent, et elles
diffèrent toujours, les directions principales sont inclinées vers les
actifs les plus bruyants. Cette inclinaison ne dépend pas de T.

Elle dépend de N. À historique fixé, élargir l'univers fait baisser
l'erreur bien plus loin — et c'est la comparaison croisée qui tranche :
**quatre cents actions sur huit ans battent quinze actions sur deux
siècles.**

Chaque plafond est posé par l'autre dimension : allonger l'historique
bute sur le biais d'un univers étroit, élargir l'univers bute sur
l'erreur d'échantillonnage d'un historique court. Mais seule la
seconde dimension est accessible dans la vraie vie.

C'est la consistance des modèles factoriels approchés de Chamberlain &
Rothschild (1983) : elle s'obtient en élargissant l'univers.

**Conséquence pratique, et elle est contre-intuitive.** Le régime
N > T — celui où la covariance empirique est singulière et où tout le
reste échoue — est précisément celui où cette méthode travaille le
mieux. Le problème de la séance 2 est la condition de validité de la
solution de la séance 7.

POURQUOI CETTE SECTION NE CONTIENT PLUS DE TABLEAU
--------------------------------------------------
Elle en contenait un. Les chiffres venaient d'une exécution antérieure
du script de démonstration, et quand j'ai porté le nombre de tirages de
8 à 12 pour stabiliser la mesure, le tableau de la démonstration a
changé et celui-ci n'a pas suivi. 0,0255 contre 0,0268 : assez proche
pour ne pas sauter aux yeux, assez différent pour être faux.

Huitième occurrence de la même erreur dans ce projet, et la première
dans une docstring plutôt que dans une narration de démonstration.
La règle était pourtant écrite en tête de `examples/seance7.py`.

La correction n'est donc pas de remettre les bons chiffres — ils
dériveront à nouveau. C'est de ne plus en écrire ici. Les conclusions
ci-dessus sont verrouillées par des tests qui les recalculent
(`test_allonger_lhistorique_ne_fait_pas_converger_le_modele`,
`test_lerreur_decroit_quand_lunivers_sagrandit`,
`test_un_grand_univers_bat_un_long_historique`). Les chiffres exacts
s'obtiennent en exécutant `python examples/seance7.py`, leçon 7.

**Une prose qui n'affirme aucun nombre ne peut pas dériver.** Quand un
nombre doit être affirmé, il faut un test qui le recalcule — pas une
relecture attentive.

Si l'on voulait vraiment estimer un modèle factoriel sur un univers
étroit, il faudrait passer à une estimation par maximum de
vraisemblance, qui modélise les variances spécifiques au lieu de les
supposer égales (`sklearn.decomposition.FactorAnalysis`). Au prix d'un
algorithme itératif, sans forme close, et sans critère de sélection de k
aussi propre que Marchenko-Pastur.

CE QUE CE MODULE NE FAIT PAS, ET POURQUOI
-----------------------------------------
Les facteurs extraits ici sont **statistiques** : ce sont des directions
de variance commune, sans nom et sans interprétation garantie. Le
premier est presque toujours « le marché », les suivants ressemblent
souvent à des secteurs — mais rien ne l'impose, et la lecture peut
changer d'une période à l'autre.

Un modèle à facteurs **fondamentaux** (style Barra, et c'est ce
qu'Aladdin utilise) part au contraire de facteurs nommés d'avance —
taille, valeur, momentum, secteur, pays — et estime les expositions par
régression. Les facteurs sont alors interprétables et stables, au prix
d'une hypothèse bien plus forte : il faut avoir choisi les bons facteurs
à l'avance.

Le choix statistique est fait ici parce qu'il est vérifiable sans
données fondamentales. Il faut savoir énoncer sa limite : **un facteur
statistique explique sans expliquer.** Il dit qu'il y a une cause
commune, jamais laquelle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .covariance import CovarianceEstimate
from .portfolio import Portfolio
from .types import value_equal

TRADING_DAYS_PER_YEAR = 252


# ══════════════════════════════════════════════════════════════════════
#  LA BORNE DE MARCHENKO-PASTUR
# ══════════════════════════════════════════════════════════════════════


def marchenko_pastur_bounds(n_assets: int, n_obs: int) -> tuple[float, float]:
    """
    Bornes du spectre d'une matrice de corrélation de bruit pur.

    Pour T observations de N variables indépendantes standardisées, les
    valeurs propres de la corrélation empirique se répartissent, quand
    N et T deviennent grands à q = N/T fixé, sur :

        [ (1 − √q)² , (1 + √q)² ]

    Retourne (borne_basse, borne_haute).

    LIRE CES BORNES CORRECTEMENT
    ----------------------------
    À N=100 et T=252, q = 0,40 et l'intervalle est [0,13 ; 2,61]. Du
    bruit pur produit donc des valeurs propres allant de 0,13 à 2,61,
    c'est-à-dire un rapport de VINGT entre la plus grande et la plus
    petite — alors que la vraie matrice est l'identité, dont toutes les
    valeurs propres valent 1.

    C'est la mesure la plus directe du problème de la séance 2 : le
    conditionnement apparent d'une matrice de corrélation est presque
    entièrement un artefact d'échantillonnage.

    LIMITE CONNUE DE CETTE VERSION
    ------------------------------
    Laloux et al. proposent un raffinement : le premier facteur (« le
    marché ») capte une part énorme de la variance, et il faudrait
    réajuster la borne sur la variance résiduelle σ² = 1 − λ₁/N plutôt
    que sur 1. Sans cet ajustement, la borne est trop haute et on
    sous-estime k.

    Ce n'est pas implémenté ici, et c'est un choix assumé : l'ajustement
    rend la borne dépendante du résultat qu'elle sert à calculer, ce qui
    demande une itération et un critère d'arrêt. La version simple est
    donc **conservatrice** — elle rate des facteurs faibles plutôt que
    d'en inventer. Pour un moteur de risque, se tromper dans ce
    sens-là est le bon côté.
    """
    if n_assets < 1 or n_obs < 1:
        raise ValueError("n_assets et n_obs doivent être strictement positifs")
    q = n_assets / n_obs
    racine = np.sqrt(q)
    return (float((1.0 - racine) ** 2), float((1.0 + racine) ** 2))


def n_significant_factors(
    eigenvalues: np.ndarray, n_assets: int, n_obs: int
) -> int:
    """
    Nombre de valeurs propres au-dessus de la borne haute de
    Marchenko-Pastur — donc inexplicables par le bruit seul.

    `eigenvalues` doit provenir d'une matrice de CORRÉLATION (trace = N).
    Appliquer ce critère au spectre d'une covariance n'a aucun sens : les
    bornes sont calibrées sur des variables de variance unitaire.
    """
    _, haute = marchenko_pastur_bounds(n_assets, n_obs)
    return int(np.sum(np.asarray(eigenvalues) > haute))


# ══════════════════════════════════════════════════════════════════════
#  LE MODÈLE
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class FactorModel:
    """
    Un modèle à facteurs statistiques estimé par décomposition spectrale.

    CONVENTION DE NORMALISATION — à lire avant d'utiliser `exposures`
    ----------------------------------------------------------------
    Tout est exprimé sur des rendements **standardisés** (divisés par
    leur écart-type). Les facteurs sont orthonormés et de variance 1 :

        z_i  =  Σ_j B_ij · f_j  +  e_i      avec  Var(f_j) = 1

    Conséquences directes, et c'est ce qui rend cette convention
    pratique :

    - `exposures[i, j]` est le nombre d'écarts-types dont bouge l'actif i
      quand le facteur j bouge d'un écart-type. Directement lisible.
    - la somme des carrés d'une ligne de B est le R² de l'actif : la part
      de sa variance expliquée par les facteurs.
    - `specific_variances[i]` vaut 1 − R²_i, donc toujours dans [0, 1].
    - la covariance des facteurs est l'identité, donc on ne la stocke
      pas. Un modèle à facteurs FONDAMENTAUX aurait des facteurs
      corrélés et devrait la porter ; on la retrouverait alors dans la
      reconstruction sous la forme B Σ_f Bᵀ.
    """

    exposures: np.ndarray          # B, (N, k), sur rendements standardisés
    specific_variances: np.ndarray  # d, (N,), dans [0, 1]
    volatilities: np.ndarray       # σ, (N,), par période
    eigenvalues: np.ndarray        # spectre complet de la corrélation, (N,)
    assets: tuple[str, ...]
    n_obs: int
    method: str

    def __post_init__(self) -> None:
        n = len(self.assets)
        if self.exposures.shape[0] != n:
            raise ValueError(
                f"expositions de forme {self.exposures.shape}, "
                f"attendu {n} lignes"
            )
        if self.specific_variances.shape != (n,):
            raise ValueError("variances spécifiques de mauvaise dimension")
        if self.volatilities.shape != (n,):
            raise ValueError("volatilités de mauvaise dimension")
        if np.any(self.specific_variances < -1e-10):
            raise ValueError(
                "variance spécifique négative — la part systématique "
                "dépasse la variance totale, ce qui est impossible et "
                "signale un bug dans l'estimation"
            )
        if np.any(self.volatilities <= 0):
            raise ValueError("volatilité nulle ou négative")

    # ── dimensions ───────────────────────────────────────────────────
    def __eq__(self, autre: object) -> bool:
        if autre.__class__ is not self.__class__:
            return NotImplemented
        return value_equal(self, autre)

    @property
    def n_factors(self) -> int:
        return int(self.exposures.shape[1])

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_parameters(self) -> int:
        """
        Nombre de paramètres libres du modèle : N·k expositions + N
        variances spécifiques.

        À comparer à N(N+1)/2 pour une covariance libre. C'est le chiffre
        qui justifie toute la démarche.
        """
        return self.n_assets * self.n_factors + self.n_assets

    @property
    def n_parameters_free_covariance(self) -> int:
        return self.n_assets * (self.n_assets + 1) // 2

    @property
    def compression_ratio(self) -> float:
        """Facteur de réduction du nombre de paramètres."""
        return self.n_parameters_free_covariance / self.n_parameters

    # ── lecture ──────────────────────────────────────────────────────
    def exposures_frame(self) -> pd.DataFrame:
        """Expositions, lisibles : une ligne par actif, une colonne par facteur."""
        return pd.DataFrame(
            self.exposures,
            index=pd.Index(self.assets, name="actif"),
            columns=[f"F{j + 1}" for j in range(self.n_factors)],
        )

    def systematic_share(self) -> pd.Series:
        """
        Part de la variance de chaque actif expliquée par les facteurs — le R².

        C'est le diagnostic le plus utile du modèle. Un actif à R² = 0,85
        est presque entièrement porté par les facteurs communs ; un actif
        à R² = 0,15 vit sa propre vie, et le modèle n'a presque rien à
        dire sur lui.
        """
        return pd.Series(
            1.0 - self.specific_variances,
            index=pd.Index(self.assets, name="actif"),
            name="R²",
        )

    def explained_variance_ratio(self) -> np.ndarray:
        """
        Part de la variance TOTALE captée par chaque facteur retenu.

        Les valeurs propres d'une corrélation somment à N, d'où la
        division. Attention à ne pas confondre avec le R² par actif :
        ceci est agrégé sur tout l'univers.
        """
        return np.asarray(self.eigenvalues[: self.n_factors]) / self.n_assets

    @property
    def noise_bounds(self) -> tuple[float, float]:
        return marchenko_pastur_bounds(self.n_assets, self.n_obs)

    # ── le point d'intégration avec tout le reste du moteur ──────────
    def covariance(self) -> CovarianceEstimate:
        """
        Reconstruit une matrice de covariance utilisable partout ailleurs.

            C = B Bᵀ + diag(d)          (corrélation implicite)
            Σ = diag(σ) · C · diag(σ)

        C'EST ICI QUE LA CONCEPTION DES SÉANCES PRÉCÉDENTES PAIE
        --------------------------------------------------------
        Cette méthode retourne un `CovarianceEstimate`, le même type que
        `sample_covariance` et `ledoit_wolf`. Donc `Portfolio`,
        `decompose_risk`, `minimum_variance_weights`, tout le moteur de
        scénarios de stress — rien de tout cela n'a besoin de savoir
        qu'un modèle à facteurs existe. Ça marche, sans une ligne
        modifiée ailleurs.

        Ce n'est pas de la chance : c'est le bénéfice d'avoir fait passer
        le moteur par un type de covariance plutôt que par des tableaux
        nus. La facture avait été payée à la séance 2.

        DEUX GARANTIES QUE LA COVARIANCE EMPIRIQUE N'OFFRE PAS
        ------------------------------------------------------
        1. **Toujours définie positive**, même quand N > T. B Bᵀ est
           semi-définie positive par construction, diag(d) est positive
           dès que d > 0, et la somme de deux matrices SDP est SDP. La
           covariance empirique, elle, est singulière dès que N > T.

        2. **Diagonale exactement préservée.** Les volatilités
           individuelles sortent du modèle identiques à celles mesurées.
           Seule la structure de corrélation est restructurée. C'est
           voulu — voir l'en-tête du module.
        """
        corr = self.exposures @ self.exposures.T + np.diag(self.specific_variances)
        # La diagonale analytique vaut exactement 1 (R² + (1−R²)), mais
        # l'accumulation en virgule flottante laisse un résidu de l'ordre
        # de 1e-16. On la force : une corrélation dont la diagonale vaut
        # 0,9999999999 produit des avertissements en cascade plus loin.
        np.fill_diagonal(corr, 1.0)
        sigma = np.outer(self.volatilities, self.volatilities) * corr
        return CovarianceEstimate(
            matrix=sigma,
            assets=self.assets,
            n_obs=self.n_obs,
            method=f"facteurs (k={self.n_factors})",
            shrinkage=0.0,
        )

    def diagnostic(self) -> str:
        basse, haute = self.noise_bounds
        retenu = self.eigenvalues[: self.n_factors]
        premier_rejete = (
            self.eigenvalues[self.n_factors]
            if self.n_factors < len(self.eigenvalues)
            else float("nan")
        )
        r2 = self.systematic_share()
        return (
            f"{self.method}  |  N={self.n_assets}  T={self.n_obs}  "
            f"k={self.n_factors}\n"
            f"  bruit attendu sur [{basse:.2f} ; {haute:.2f}]  "
            f"— valeurs propres retenues : "
            f"{', '.join(f'{v:.2f}' for v in retenu[:6])}"
            + ("…" if self.n_factors > 6 else "")
            + f"\n  première rejetée : {premier_rejete:.2f}"
            f"\n  R² médian {r2.median():.0%}  "
            f"(min {r2.min():.0%}, max {r2.max():.0%})"
            f"\n  {self.n_parameters} paramètres contre "
            f"{self.n_parameters_free_covariance} pour une covariance libre "
            f"— compression ×{self.compression_ratio:.0f}"
        )

    def __str__(self) -> str:
        return self.diagnostic()


# ══════════════════════════════════════════════════════════════════════
#  L'ESTIMATION
# ══════════════════════════════════════════════════════════════════════


def pca_factor_model(
    returns: pd.DataFrame,
    n_factors: int | None = None,
) -> FactorModel:
    """
    Estime un modèle à facteurs statistiques par décomposition spectrale
    de la matrice de corrélation.

    `n_factors=None` (défaut) : k est choisi par le critère de
    Marchenko-Pastur. Passer un entier force k, ce qui sert surtout à
    tester le critère contre une vérité connue.

    LA PROCÉDURE, ET POURQUOI CHAQUE ÉTAPE EST DANS CET ORDRE
    ---------------------------------------------------------
    1. Standardiser les rendements. Sans cela, la décomposition est
       dominée par les actifs les plus volatils au lieu de la structure
       commune — et surtout, la borne de Marchenko-Pastur ne
       s'appliquerait plus, puisqu'elle suppose des variables de
       variance unitaire.
    2. Décomposer la corrélation en valeurs propres. `eigh` et non
       `eig` : la matrice est symétrique, l'algorithme dédié est plus
       rapide et garantit des valeurs propres réelles. Avec `eig`, une
       corrélation parfaitement symétrique peut sortir des valeurs
       propres à partie imaginaire de 1e-17, et tout le code aval doit
       alors s'encombrer de `.real`.
    3. Choisir k par comparaison à la borne de bruit.
    4. Construire B = V·√Λ sur les k premières composantes.
    5. Déduire d par différence, pour que la diagonale soit exacte.

    POURQUOI d EST TOUJOURS POSITIF — LA PREUVE TIENT EN TROIS LIGNES
    -----------------------------------------------------------------
    La diagonale de la corrélation vaut 1 pour tout actif i :

        1 = C_ii = Σ_{j=1..N} λ_j · v_ji²

    La part systématique n'en prend que les k premiers termes, tous
    positifs puisque λ_j ≥ 0 pour une matrice SDP. Donc

        Σ_{j=1..k} λ_j · v_ji²  ≤  1     et     d_i = 1 − (…) ≥ 0

    Ce n'est pas une contrainte qu'on impose ni un `max(0, ·)` qu'on
    applique en espérant : c'est une conséquence. Le test
    correspondant le vérifie sur mille matrices tirées au hasard, non
    pas parce que la preuve serait douteuse, mais parce qu'un bug
    d'implémentation la contredirait et qu'il faut le savoir.
    """
    x = _validate(returns)
    n_obs, n_assets = x.shape

    if n_obs < 3:
        raise ValueError("au moins 3 observations requises")

    vols = x.std(axis=0, ddof=1)
    if np.any(vols <= 0):
        morts = [a for a, v in zip(returns.columns, vols, strict=True) if v <= 0]
        raise ValueError(
            f"actif(s) de variance nulle : {morts}. Une série constante n'a "
            f"ni corrélation ni exposition définie — retire-la de l'univers "
            f"plutôt que de laisser le moteur diviser par zéro."
        )

    z = (x - x.mean(axis=0)) / vols
    corr = (z.T @ z) / (n_obs - 1)
    # Symétrisation : corr est symétrique en théorie, mais l'ordre des
    # opérations flottantes laisse un écart de ~1e-16 entre corr[i,j] et
    # corr[j,i]. `eigh` ne lit qu'un triangle, donc le résultat
    # dépendrait silencieusement duquel.
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    # eigh trie par valeur propre CROISSANTE. On inverse : la convention
    # universelle en analyse factorielle est décroissante, et un lecteur
    # qui voit `eigenvalues[0]` attend le facteur dominant.
    valeurs, vecteurs = np.linalg.eigh(corr)
    ordre = np.argsort(valeurs)[::-1]
    valeurs = valeurs[ordre]
    vecteurs = vecteurs[:, ordre]

    # Une corrélation est SDP en théorie ; l'arithmétique flottante peut
    # rendre -1e-17. On écrase, sinon √λ produit un NaN.
    valeurs = np.maximum(valeurs, 0.0)

    if n_factors is None:
        k = n_significant_factors(valeurs, n_assets, n_obs)
        methode = "facteurs PCA (k par Marchenko-Pastur)"
    else:
        if not 1 <= n_factors <= n_assets:
            raise ValueError(
                f"n_factors doit être entre 1 et {n_assets}, reçu {n_factors}"
            )
        k = n_factors
        methode = "facteurs PCA (k imposé)"

    if k == 0:
        raise InsufficientStructure(n_assets, n_obs, float(valeurs[0]))

    vecteurs_k = _fix_signs(vecteurs[:, :k])
    b = vecteurs_k * np.sqrt(valeurs[:k])

    systematique = np.einsum("ij,ij->i", b, b)
    d = np.clip(1.0 - systematique, 0.0, 1.0)

    return FactorModel(
        exposures=b,
        specific_variances=d,
        volatilities=vols,
        eigenvalues=valeurs,
        assets=tuple(returns.columns),
        n_obs=n_obs,
        method=methode,
    )


@dataclass(frozen=True)
class InsufficientStructure(Exception):
    """
    Levée quand aucune valeur propre ne dépasse le seuil de bruit.

    Autrement dit : rien dans ces données ne distingue leur matrice de
    corrélation de celle de variables indépendantes. Le moteur refuse
    d'extraire un facteur, exactement comme `InsufficientData` refuse
    d'estimer une volatilité sur trois points.

    La tentation serait de retourner quand même le premier vecteur
    propre — il existe toujours, et il a l'air d'un résultat. Ce serait
    présenter du bruit comme une structure, ce que ce moteur ne fait
    pas.
    """

    n_assets: int
    n_obs: int
    largest_eigenvalue: float

    def __str__(self) -> str:
        _, haute = marchenko_pastur_bounds(self.n_assets, self.n_obs)
        return (
            f"Aucun facteur détectable : la plus grande valeur propre vaut "
            f"{self.largest_eigenvalue:.3f}, sous le seuil de bruit "
            f"{haute:.3f} pour N={self.n_assets}, T={self.n_obs}. "
            f"Ces rendements sont indistinguables de variables indépendantes."
        )


def _fix_signs(vectors: np.ndarray) -> np.ndarray:
    """
    Fixe l'indétermination de signe des vecteurs propres.

    LE PIÈGE QUE CETTE FONCTION DÉSAMORCE
    -------------------------------------
    Si v est vecteur propre, −v l'est aussi, avec la même valeur propre.
    LAPACK choisit l'un ou l'autre selon des détails internes :
    l'architecture du processeur, la version de la bibliothèque, parfois
    l'ordre des lignes en mémoire.

    Conséquence : deux exécutions du même code sur deux machines peuvent
    produire des expositions de signes opposés. La covariance
    reconstruite est identique — B Bᵀ est invariant par changement de
    signe d'une colonne — mais toute lecture directe des expositions,
    tout test d'égalité, toute comparaison à scikit-learn devient
    instable.

    C'est exactement le genre de non-déterminisme qui produit un test
    rouge une fois sur dix en intégration continue et qu'on finit par
    marquer `flaky` au lieu de comprendre.

    CONVENTION RETENUE : la composante de plus grande valeur absolue de
    chaque facteur est rendue positive. C'est celle de scikit-learn, ce
    qui rend la validation croisée possible sans contorsion.
    """
    sorti = vectors.copy()
    for j in range(sorti.shape[1]):
        i_max = int(np.argmax(np.abs(sorti[:, j])))
        if sorti[i_max, j] < 0:
            sorti[:, j] *= -1.0
    return sorti


def _validate(returns: pd.DataFrame) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("attendu un DataFrame (une colonne par actif)")
    if returns.isna().to_numpy().any():
        raise ValueError(
            "valeurs manquantes dans les rendements — à traiter "
            "explicitement avant d'estimer un modèle"
        )
    x = returns.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("au moins 2 actifs requis")
    return np.asarray(x, dtype=float)


# ══════════════════════════════════════════════════════════════════════
#  ATTRIBUTION DU RISQUE AUX FACTEURS
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FactorAttribution:
    """
    Décomposition du risque d'un portefeuille entre facteurs et spécifique.

    Toutes les volatilités sont annualisées. Les contributions somment
    EXACTEMENT à la volatilité totale — c'est l'identité d'Euler de la
    séance 2, appliquée cette fois à des facteurs plutôt qu'à des actifs.
    """

    total_volatility: float
    systematic_volatility: float
    specific_volatility: float
    factor_exposures: pd.Series      # bᵀ = Bᵀw, l'exposition du portefeuille
    factor_contributions: pd.Series  # contribution de chaque facteur à σ_p
    specific_contribution: float

    @property
    def systematic_share(self) -> float:
        """Part de la VARIANCE portée par les facteurs communs."""
        return (self.systematic_volatility / self.total_volatility) ** 2

    def table(self) -> pd.DataFrame:
        lignes = pd.DataFrame(
            {
                "exposition": self.factor_exposures,
                "contrib. à σ": self.factor_contributions,
                "% du risque": self.factor_contributions / self.total_volatility,
            }
        )
        lignes.loc["spécifique"] = [
            np.nan,
            self.specific_contribution,
            self.specific_contribution / self.total_volatility,
        ]
        return lignes

    def __str__(self) -> str:
        return (
            f"Volatilité totale      : {self.total_volatility:.2%}\n"
            f"  dont systématique    : {self.systematic_volatility:.2%} "
            f"({self.systematic_share:.0%} de la variance)\n"
            f"  dont spécifique      : {self.specific_volatility:.2%}"
        )


def attribute_risk(
    weights: pd.Series,
    model: FactorModel,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> FactorAttribution:
    """
    Attribue le risque d'un portefeuille à ses facteurs.

    LE CALCUL
    ---------
    Soit w les poids, σ les volatilités, et u_i = w_i·σ_i les poids en
    unités de risque. L'exposition du portefeuille au facteur j est

        b_j = Σ_i u_i · B_ij

    et la variance se sépare exactement en deux morceaux :

        σ_p²  =  Σ_j b_j²   +   Σ_i u_i² · d_i
                 ─────────       ──────────────
                 systématique     spécifique

    La première somme est le risque qu'on partage avec le marché ; la
    seconde celui qu'on s'est choisi. La diversification agit sur la
    seconde et **pas du tout** sur la première : détenir mille actions
    ne réduit pas l'exposition au facteur marché, cela ne fait que la
    moyenner.

    POURQUOI C'EST LA DÉCOMPOSITION QUI COMPTE EN PRATIQUE
    ------------------------------------------------------
    La décomposition par actif de la séance 2 répond à « d'où vient mon
    risque ? » en nommant des lignes du portefeuille. Celle-ci répond à
    « de quoi suis-je dépendant ? » en nommant des causes.

    Deux portefeuilles sans un seul titre en commun peuvent avoir des
    expositions factorielles quasi identiques. La première décomposition
    les dit différents ; la seconde dit la vérité. C'est précisément le
    service que rend Aladdin à un gérant : pas « tu détiens trop de X »,
    mais « ta position nette sur ce facteur est celle-ci, à travers des
    titres qui n'ont l'air de rien avoir en commun ».

    CONTRIBUTIONS AU SENS D'EULER
    -----------------------------
    σ_p est homogène de degré 1 en b, donc σ_p = Σ_j b_j·(∂σ_p/∂b_j),
    et la contribution du facteur j vaut b_j²/σ_p. Les contributions
    somment exactement à σ_p, spécifique inclus — ce n'est pas une
    normalisation après coup, c'est le théorème.
    """
    actifs = list(model.assets)
    manquants = set(actifs) - set(weights.index)
    if manquants:
        raise ValueError(f"poids absents pour {sorted(manquants)}")

    w = weights.reindex(actifs).to_numpy(dtype=float)
    if np.isnan(w).any():
        raise ValueError("valeurs manquantes dans les poids")

    echelle = np.sqrt(periods_per_year)
    # Poids en unités de risque : u_i = w_i·σ_i. C'est le changement de
    # variable qui rend la formule lisible, puisque B porte sur des
    # rendements standardisés et w sur des rendements bruts.
    u = w * model.volatilities * echelle

    b = model.exposures.T @ u                       # (k,)
    var_systematique = float(b @ b)
    var_specifique = float(np.sum(u**2 * model.specific_variances))
    var_totale = var_systematique + var_specifique

    if var_totale <= 0:
        raise ValueError("variance de portefeuille nulle ou négative")

    sigma_p = np.sqrt(var_totale)
    noms = [f"F{j + 1}" for j in range(model.n_factors)]

    return FactorAttribution(
        total_volatility=sigma_p,
        systematic_volatility=float(np.sqrt(var_systematique)),
        specific_volatility=float(np.sqrt(var_specifique)),
        factor_exposures=pd.Series(b, index=noms, name="exposition"),
        factor_contributions=pd.Series(b**2 / sigma_p, index=noms,
                                       name="contribution"),
        specific_contribution=var_specifique / sigma_p,
    )


# ══════════════════════════════════════════════════════════════════════
#  SCÉNARIO DE FACTEUR
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FactorShock:
    """
    Fait bouger UN facteur, et laisse le modèle en déduire ce qui arrive
    à chaque actif.

    POURQUOI CETTE CLASSE EST DANS `factors` ET NON DANS `stress`
    -------------------------------------------------------------
    Elle n'importe rien de `nkrisk.stress`, n'hérite de rien, et
    satisfait pourtant le protocole `Shock` : elle expose un `name` et un
    `apply(portfolio) -> portfolio`. `isinstance(choc, Shock)` renvoie
    True, et elle se compose avec `Combined` comme n'importe quel autre
    choc.

    C'était l'argument avancé à la séance 5 pour préférer un `Protocol` à
    une classe de base abstraite. Ceci en est la vérification en
    conditions réelles, et pas sur une classe jouet de douze lignes : un
    module entier étend le moteur de scénarios sans que `stress.py`
    apprenne un seul mot sur les facteurs.

    CE QUE CE SCÉNARIO APPORTE PAR RAPPORT À UN CHOC DE PRIX
    --------------------------------------------------------
    Avec `PriceShock`, il faut inventer N mouvements — un par actif. Et
    les inventer de façon COHÉRENTE est précisément ce que les humains
    font mal : on écrit « −30 % sur la tech, −5 % sur les services aux
    collectivités » en oubliant que ces deux chiffres doivent respecter
    la structure de corrélation mesurée, sans quoi le scénario est
    impossible au sens du modèle de risque qu'on utilise par ailleurs.

    Ici on fournit UN nombre — l'ampleur du mouvement du facteur — et les
    N mouvements d'actifs sont DÉDUITS des expositions estimées. Les
    actifs les plus exposés tombent le plus, automatiquement, dans les
    proportions que les données ont montrées.

    Le scénario devient une hypothèse sur une cause, pas une liste de
    souhaits sur des conséquences.

    PARAMÈTRES
    ----------
    model : le modèle d'où viennent les expositions.
    factor : numéro du facteur, à partir de 1 (comme les étiquettes F1, F2…).
    sigmas : ampleur du mouvement, en écarts-types du facteur. Négatif pour
        une baisse. −3 est un mauvais jour, −5 un événement de crise.
    horizon : nombre de périodes sur lesquelles le mouvement se produit.
        L'ampleur est mise à l'échelle en √horizon.

        ⚠️ Cette racine suppose des rendements indépendants d'une période
        à l'autre. C'est faux en crise, où les baisses s'enchaînent :
        la vraie dispersion à 10 jours dépasse √10 fois la dispersion
        quotidienne. Le scénario est donc OPTIMISTE à horizon long, et
        il vaut mieux le savoir que de l'ignorer. La règle des √t est
        d'ailleurs celle que Bâle impose, avec la même faiblesse.
    """

    model: FactorModel
    factor: int
    sigmas: float
    horizon: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.factor <= self.model.n_factors:
            raise ValueError(
                f"facteur {self.factor} inexistant : le modèle en compte "
                f"{self.model.n_factors}"
            )
        if self.horizon < 1:
            raise ValueError("l'horizon doit valoir au moins une période")
        if not self.name:
            signe = "+" if self.sigmas >= 0 else "−"
            jours = "" if self.horizon == 1 else f" sur {self.horizon} j"
            object.__setattr__(
                self,
                "name",
                f"F{self.factor} à {signe}{abs(self.sigmas):.1f}σ{jours}",
            )

    def asset_moves(self) -> pd.Series:
        """
        Les mouvements d'actifs impliqués par le scénario.

        r_i = σ_i · √h · B_{i,j} · s

        Lecture : B_{i,j} est le nombre d'écarts-types dont bouge l'actif
        i quand le facteur j bouge d'un écart-type ; on multiplie par
        l'ampleur voulue, puis par la volatilité propre de l'actif pour
        revenir en rendement.

        Exposer ce calcul en méthode publique n'est pas cosmétique : un
        scénario qu'on ne peut pas inspecter est un scénario qu'on ne
        peut pas contester. Cette série est ce qu'on met sous les yeux
        d'un comité de risque.
        """
        j = self.factor - 1
        moves = (
            self.model.volatilities
            * np.sqrt(self.horizon)
            * self.model.exposures[:, j]
            * self.sigmas
        )
        return pd.Series(
            moves, index=pd.Index(self.model.assets, name="actif"), name="choc"
        )

    def apply(self, portfolio: Portfolio) -> Portfolio:
        actifs = list(portfolio.assets)
        if set(actifs) != set(self.model.assets):
            raise ValueError(
                "le modèle à facteurs et le portefeuille ne portent pas sur "
                "les mêmes actifs"
            )

        r = self.asset_moves().reindex(actifs).to_numpy()
        if np.any(r <= -1.0):
            pires = [
                a for a, v in zip(actifs, r, strict=True) if v <= -1.0
            ]
            raise ValueError(
                f"le scénario implique une perte de 100 % ou plus sur "
                f"{pires} — hors du domaine du modèle. Réduis `sigmas` ou "
                f"l'horizon."
            )

        w = portfolio.weights.to_numpy()
        rendement = float(w @ r)
        croissance = w * (1.0 + r)
        total = croissance.sum()
        if total <= 0:
            raise ValueError("le scénario annule la valeur du portefeuille")

        return Portfolio(
            weights=pd.Series(croissance / total, index=portfolio.weights.index),
            covariance=portfolio.covariance,
            value=portfolio.value * (1.0 + rendement),
            name=f"{portfolio.name} | {self.name}",
        )
