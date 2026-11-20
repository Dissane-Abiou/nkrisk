"""
Tests du modèle à facteurs — séance 7.

Même architecture à quatre niveaux que les séances précédentes :

  1. propriétés garanties par construction, vérifiées sur des tirages
  2. calibration contre une structure factorielle dont on connaît k
  3. comportement sur du bruit pur — le test qui compte vraiment ici
  4. validation croisée contre scikit-learn

Le niveau 3 est le plus important de ce module. Un extracteur de
facteurs trouve TOUJOURS quelque chose : le premier vecteur propre
d'une matrice existe quelles que soient les données. La question n'est
donc jamais « est-ce que ça marche », mais « est-ce que ça sait se
taire ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nkrisk.covariance import sample_covariance
from nkrisk.factors import (
    FactorModel,
    FactorShock,
    InsufficientStructure,
    attribute_risk,
    marchenko_pastur_bounds,
    n_significant_factors,
    pca_factor_model,
)
from nkrisk.portfolio import Portfolio
from nkrisk.risk import decompose_risk

# ══════════════════════════════════════════════════════════════════════
#  GÉNÉRATEURS À VÉRITÉ CONNUE
# ══════════════════════════════════════════════════════════════════════


def _rendements_a_facteurs(
    n_assets: int,
    n_obs: int,
    n_factors: int,
    seed: int,
    force: float = 0.8,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rendements dont la structure factorielle est CONNUE par construction.

    `force` règle l'écart-type des expositions vraies, donc le rapport
    signal/bruit. Le premier facteur reçoit des expositions toutes
    positives : c'est le « marché », et aucun univers d'actions réel ne
    s'en écarte.

    Retourne les rendements et la matrice d'expositions vraie, sur
    rendements standardisés.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, force, (n_assets, n_factors))
    b[:, 0] = np.abs(b[:, 0]) + force  # facteur marché : exposition positive
    f = rng.normal(0.0, 1.0, (n_obs, n_factors))
    e = rng.normal(0.0, 1.0, (n_obs, n_assets))
    z = f @ b.T + e

    vols = rng.uniform(0.008, 0.030, n_assets)
    noms = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(z * vols, columns=noms), b


def _bruit_pur(n_assets: int, n_obs: int, seed: int) -> pd.DataFrame:
    """Actifs rigoureusement indépendants. Aucune structure à trouver."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.008, 0.030, n_assets)
    x = rng.normal(0.0, 1.0, (n_obs, n_assets)) * vols
    return pd.DataFrame(x, columns=[f"A{i:02d}" for i in range(n_assets)])


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — PROPRIÉTÉS GARANTIES PAR CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════


def test_diagonale_exactement_preservee() -> None:
    """
    Les volatilités individuelles sortent du modèle identiques à celles
    mesurées. C'est le choix de conception central : on ne régularise que
    la corrélation, jamais les variances.
    """
    r, _ = _rendements_a_facteurs(25, 600, 3, seed=1)
    modele = pca_factor_model(r)

    reconstruite = np.diag(modele.covariance().matrix)
    empirique = np.diag(sample_covariance(r).matrix)

    assert np.allclose(reconstruite, empirique, rtol=1e-12)


def test_variances_specifiques_toujours_dans_zero_un() -> None:
    """
    d_i = 1 − R²_i, donc nécessairement dans [0, 1].

    La preuve est dans la docstring de `pca_factor_model` et tient en
    trois lignes. On la vérifie quand même sur deux cents tirages : ce
    n'est pas la preuve qu'on teste, c'est l'implémentation.
    """
    for seed in range(200):
        n = int(np.random.default_rng(seed).integers(4, 30))
        r, _ = _rendements_a_facteurs(n, 250, 2, seed=seed)
        try:
            modele = pca_factor_model(r)
        except InsufficientStructure:
            continue
        d = modele.specific_variances
        assert d.min() >= 0.0, f"variance spécifique négative au tirage {seed}"
        assert d.max() <= 1.0, f"variance spécifique > 1 au tirage {seed}"


def test_definie_positive_meme_quand_le_nombre_dactifs_depasse_lhistorique() -> None:
    """
    LE RÉSULTAT QUI JUSTIFIE TOUT LE MODULE.

    Avec N = 60 actifs et T = 40 observations, la covariance empirique
    est mathématiquement singulière : son rang ne peut pas dépasser 39.
    Elle est donc non inversible, et toute optimisation de portefeuille
    construite dessus est impossible — pas imprécise : impossible.

    Le modèle à facteurs, lui, produit une matrice strictement définie
    positive, parce que B Bᵀ est SDP et que diag(d) est strictement
    positive. Ce n'est pas une amélioration de précision. C'est la
    différence entre un objet utilisable et un objet qui n'existe pas.
    """
    r, _ = _rendements_a_facteurs(60, 40, 3, seed=11, force=1.2)

    vp_empirique = np.linalg.eigvalsh(sample_covariance(r).matrix)
    assert vp_empirique.min() < 1e-12, (
        "la covariance empirique devrait être singulière à N > T — "
        "si ce n'est pas le cas, le générateur de test est en cause"
    )

    modele = pca_factor_model(r)
    vp_facteurs = np.linalg.eigvalsh(modele.covariance().matrix)
    assert vp_facteurs.min() > 0.0
    assert modele.covariance().is_invertible


def test_contributions_deuler_somment_a_la_volatilite() -> None:
    """
    σ_p est homogène de degré 1 : la somme des contributions des facteurs
    et du spécifique vaut exactement la volatilité totale. Pas à 1 % près :
    exactement, à la précision machine.
    """
    r, _ = _rendements_a_facteurs(20, 500, 3, seed=2)
    modele = pca_factor_model(r)

    rng = np.random.default_rng(99)
    poids = rng.uniform(0.1, 1.0, modele.n_assets)
    poids /= poids.sum()
    w = pd.Series(poids, index=list(modele.assets))

    a = attribute_risk(w, modele)
    total = a.factor_contributions.sum() + a.specific_contribution

    assert total == pytest.approx(a.total_volatility, rel=1e-12)


def test_variance_systematique_plus_specifique_egale_le_total() -> None:
    """Les deux morceaux de la variance se recomposent exactement."""
    r, _ = _rendements_a_facteurs(15, 400, 2, seed=3)
    modele = pca_factor_model(r)
    w = pd.Series(np.full(15, 1 / 15), index=list(modele.assets))

    a = attribute_risk(w, modele)
    somme = a.systematic_volatility**2 + a.specific_volatility**2

    assert somme == pytest.approx(a.total_volatility**2, rel=1e-12)


def test_tous_les_facteurs_reconstruisent_exactement_la_correlation() -> None:
    """
    Cas limite : avec k = N, le modèle ne comprime plus rien et doit
    redonner la corrélation empirique au bit près. Si ce test échoue, la
    décomposition spectrale elle-même est fausse, et tout le reste du
    module repose sur du sable.
    """
    r, _ = _rendements_a_facteurs(10, 300, 3, seed=4)
    modele = pca_factor_model(r, n_factors=10)

    assert np.allclose(modele.specific_variances, 0.0, atol=1e-10)

    attendue = sample_covariance(r).correlation().to_numpy()
    obtenue = modele.covariance().correlation().to_numpy()
    assert np.allclose(obtenue, attendue, atol=1e-10)


def test_convention_de_signe_deterministe() -> None:
    """
    La composante de plus grande valeur absolue de chaque facteur est
    positive.

    Sans cette convention, deux machines peuvent produire des expositions
    opposées pour les mêmes données — LAPACK ne garantit rien sur le
    signe d'un vecteur propre.
    """
    r, _ = _rendements_a_facteurs(20, 400, 4, seed=5)
    modele = pca_factor_model(r)

    for j in range(modele.n_factors):
        colonne = modele.exposures[:, j]
        assert colonne[np.argmax(np.abs(colonne))] > 0


def test_la_covariance_reconstruite_est_insensible_au_signe_des_facteurs() -> None:
    """
    Le corollaire rassurant de l'indétermination de signe : B Bᵀ est
    invariant quand on retourne une colonne de B.

    C'est pour cela qu'un bug de signe ne se voit JAMAIS dans la
    covariance — seulement dans la lecture des expositions. Un test qui
    ne vérifierait que la covariance passerait sur un code faux.
    """
    r, _ = _rendements_a_facteurs(12, 300, 3, seed=6)
    modele = pca_factor_model(r)

    retourne = modele.exposures.copy()
    retourne[:, 1] *= -1.0
    jumeau = FactorModel(
        exposures=retourne,
        specific_variances=modele.specific_variances,
        volatilities=modele.volatilities,
        eigenvalues=modele.eigenvalues,
        assets=modele.assets,
        n_obs=modele.n_obs,
        method=modele.method,
    )

    assert np.allclose(jumeau.covariance().matrix, modele.covariance().matrix)
    # …et pourtant les expositions diffèrent : c'est tout le piège.
    assert not np.allclose(jumeau.exposures, modele.exposures)


def test_conditionnement_borne_par_la_plus_petite_variance_specifique() -> None:
    """
    La plus petite valeur propre de B Bᵀ + diag(d) est au moins min(d),
    puisque B Bᵀ est SDP. Le conditionnement est donc borné d'avance —
    garantie qu'aucune covariance empirique n'offre.
    """
    r, _ = _rendements_a_facteurs(40, 120, 3, seed=7)
    modele = pca_factor_model(r)

    corr = modele.exposures @ modele.exposures.T + np.diag(
        modele.specific_variances
    )
    assert np.linalg.eigvalsh(corr).min() >= modele.specific_variances.min() - 1e-12


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — CALIBRATION CONTRE UNE VÉRITÉ CONNUE
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("k_vrai", [1, 2, 3, 5])
def test_retrouve_le_nombre_de_facteurs_injectes(k_vrai: int) -> None:
    """
    On fabrique des données avec exactement k facteurs, et on vérifie que
    Marchenko-Pastur en trouve k.

    Impossible sur données réelles : le vrai nombre de facteurs y est
    inobservable. C'est tout l'intérêt des données synthétiques — elles
    sont le seul endroit où « juste » a un sens.
    """
    r, _ = _rendements_a_facteurs(40, 800, k_vrai, seed=100 + k_vrai, force=1.0)
    modele = pca_factor_model(r)
    assert modele.n_factors == k_vrai


def _erreur_de_correlation(
    n_assets: int, n_obs: int, n_factors: int, seeds: range
) -> float:
    """
    Erreur absolue moyenne hors diagonale entre la corrélation estimée et
    la vraie, moyennée sur plusieurs tirages.

    On compare les corrélations et non les expositions : la base
    factorielle n'est identifiable qu'à une rotation près, alors que la
    corrélation qu'elle engendre est unique.
    """
    erreurs = []
    for seed in seeds:
        r, b_vrai = _rendements_a_facteurs(
            n_assets, n_obs, n_factors, seed=seed, force=1.0
        )
        modele = pca_factor_model(r, n_factors=n_factors)

        vraie = b_vrai @ b_vrai.T + np.eye(n_assets)
        echelle = 1.0 / np.sqrt(np.diag(vraie))
        vraie = vraie * np.outer(echelle, echelle)

        ecart = np.abs(modele.covariance().correlation().to_numpy() - vraie)
        erreurs.append(ecart[~np.eye(n_assets, dtype=bool)].mean())
    return float(np.mean(erreurs))


def test_allonger_lhistorique_ne_fait_pas_converger_le_modele() -> None:
    """
    ⚠️ CE TEST VÉRIFIE UNE LIMITE, PAS UNE QUALITÉ. ⚠️

    J'avais d'abord écrit l'inverse : « l'erreur doit décroître quand
    l'historique s'allonge ». Ça paraît évident — c'est ce que fait tout
    estimateur. Le test a échoué, et c'est le test qui avait raison.

    À univers fixé, l'erreur de l'ACP PLAFONNE. Multiplier l'historique
    par seize la réduit de moins d'un quart, puis plus rien. Ce n'est pas
    du bruit résiduel : c'est un biais, et il ne part jamais.

    LA RAISON
    ---------
    L'ACP n'est pas de l'analyse factorielle. Elle suppose implicitement
    que toutes les variances spécifiques sont ÉGALES — c'est sous cette
    hypothèse qu'elle coïncide avec l'estimateur du maximum de
    vraisemblance. Quand elles diffèrent, les directions principales sont
    inclinées vers les actifs les plus bruyants, et cette inclinaison ne
    dépend pas du tout de T.

    Elle dépend de N. Voir le test suivant.

    C'est le résultat de Chamberlain & Rothschild (1983) sur les modèles
    factoriels approchés, retrouvé ici par la mesure.
    """
    courte = _erreur_de_correlation(15, 2_000, 3, range(8))
    longue = _erreur_de_correlation(15, 32_000, 3, range(8))

    assert longue < courte, "l'erreur devrait tout de même baisser un peu"
    assert longue > 0.5 * courte, (
        f"seize fois plus d'historique a divisé l'erreur par "
        f"{courte / longue:.1f} — si ce facteur devient grand, c'est que "
        f"le biais a disparu et que l'analyse de ce test est à refaire"
    )


def test_lerreur_decroit_quand_lunivers_sagrandit() -> None:
    """
    Le pendant du test précédent, et la bonne nouvelle.

    À historique FIXÉ, passer de 10 à 120 actifs divise l'erreur par plus
    de deux. C'est la consistance du modèle factoriel approché : elle
    s'obtient en N, pas en T.

    CONSÉQUENCE PRATIQUE, ET ELLE EST CONTRE-INTUITIVE
    ---------------------------------------------------
    Pour estimer un modèle à facteurs, mieux vaut un univers LARGE qu'un
    historique LONG. Trois ans sur cinq cents actions valent mieux que
    trente ans sur quinze.

    Et c'est précisément le régime N > T, celui où la covariance
    empirique est singulière et où tout le reste échoue, qui convient le
    mieux à cette méthode. Le problème de la séance 2 est aussi la
    condition de validité de la solution de la séance 7.
    """
    etroit = _erreur_de_correlation(10, 2_000, 3, range(8))
    large = _erreur_de_correlation(120, 2_000, 3, range(8))

    assert large < 0.6 * etroit, (
        f"élargir l'univers de 10 à 120 actifs n'a réduit l'erreur que de "
        f"{1 - large / etroit:.0%}"
    )


def test_le_r2_moyen_reflete_la_force_du_signal() -> None:
    """Des expositions plus fortes doivent produire un R² plus élevé."""
    faible, _ = _rendements_a_facteurs(25, 1000, 2, seed=8, force=0.4)
    fort, _ = _rendements_a_facteurs(25, 1000, 2, seed=8, force=1.5)

    r2_faible = pca_factor_model(faible, n_factors=2).systematic_share().mean()
    r2_fort = pca_factor_model(fort, n_factors=2).systematic_share().mean()

    assert r2_fort > r2_faible + 0.2


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — SAVOIR SE TAIRE
# ══════════════════════════════════════════════════════════════════════


def test_bruit_pur_ne_produit_aucun_facteur() -> None:
    """
    Sur des actifs rigoureusement indépendants, le moteur doit REFUSER
    d'extraire un facteur.

    Le premier vecteur propre existe pourtant, et il a parfaitement
    l'air d'un résultat : des expositions, une valeur propre, un R². Le
    retourner serait présenter du bruit comme une structure. C'est
    l'erreur que ce test interdit.
    """
    r = _bruit_pur(40, 600, seed=9)
    with pytest.raises(InsufficientStructure):
        pca_factor_model(r)


def test_les_valeurs_propres_du_bruit_restent_dans_les_bornes() -> None:
    """
    Vérification directe de la loi de Marchenko-Pastur : sur du bruit
    pur, le spectre tient dans [(1−√q)², (1+√q)²].

    On tolère un léger dépassement : la loi est asymptotique et les
    effets de taille finie font légèrement déborder les extrêmes.
    """
    n_assets, n_obs = 80, 400
    r = _bruit_pur(n_assets, n_obs, seed=10)
    corr = r.corr().to_numpy()
    valeurs = np.linalg.eigvalsh(corr)
    basse, haute = marchenko_pastur_bounds(n_assets, n_obs)

    assert valeurs.max() < haute * 1.15
    assert valeurs.min() > basse * 0.5


def test_un_vrai_facteur_depasse_nettement_le_seuil_de_bruit() -> None:
    """Le critère doit séparer franchement, pas de justesse."""
    r, _ = _rendements_a_facteurs(50, 500, 1, seed=12, force=1.0)
    corr = r.corr().to_numpy()
    valeurs = np.sort(np.linalg.eigvalsh(corr))[::-1]
    _, haute = marchenko_pastur_bounds(50, 500)

    assert valeurs[0] > 3 * haute
    assert valeurs[1] < haute


@pytest.mark.slow
def test_taux_de_faux_positifs_du_critere_de_marchenko_pastur() -> None:
    """
    Sur 300 jeux de bruit pur, combien de fois le critère invente-t-il un
    facteur ?

    Ce test MESURE une propriété statistique au lieu de l'affirmer. La
    borne est asymptotique : à taille finie, la plus grande valeur propre
    la dépasse parfois. On veut savoir à quelle fréquence, parce que
    c'est le taux auquel le moteur se trompera en production.

    Seuil à 10 % : au-delà, le critère ne serait plus assez conservateur
    pour qu'on lui fasse confiance sans surveillance.
    """
    faux_positifs = 0
    essais = 1_000
    for seed in range(essais):
        r = _bruit_pur(50, 300, seed=1000 + seed)
        corr = r.corr().to_numpy()
        valeurs = np.linalg.eigvalsh(corr)
        if n_significant_factors(valeurs, 50, 300) > 0:
            faux_positifs += 1

    taux = faux_positifs / essais
    assert taux < 0.10, f"taux de faux positifs trop élevé : {taux:.1%}"


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 4 — VALIDATION CROISÉE INDÉPENDANTE
# ══════════════════════════════════════════════════════════════════════


def test_concordance_avec_scikit_learn() -> None:
    """
    Comparaison à une implémentation écrite par d'autres personnes.

    DEUX ASSERTIONS, ET LA SECONDE EST LA PLUS IMPORTANTE
    -----------------------------------------------------
    La première compare les expositions directement, ce qui n'est
    possible que parce que scikit-learn applique la même convention de
    signe (`svd_flip(..., u_based_decision=False)` : la composante
    maximale de chaque vecteur de charge est rendue positive).

    C'est un détail d'implémentation de scikit-learn, pas une garantie
    de son API — et il a déjà changé d'une version à l'autre. La seconde
    assertion compare donc B Bᵀ, qui est invariant par changement de
    signe. **Si scikit-learn change de convention demain, la première
    assertion tombera et la seconde tiendra** : on saura immédiatement
    qu'il s'agit d'un désaccord de convention et non d'un désaccord de
    résultat.

    Un test ne doit jamais dépendre d'une convention non documentée d'une
    bibliothèque tierce sans le dire.
    """
    sklearn_decomposition = pytest.importorskip("sklearn.decomposition")

    r, _ = _rendements_a_facteurs(18, 500, 3, seed=13)
    modele = pca_factor_model(r, n_factors=3)

    x = r.to_numpy()
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    acp = sklearn_decomposition.PCA(n_components=3).fit(z)

    assert np.allclose(acp.explained_variance_, modele.eigenvalues[:3], rtol=1e-10)

    b_sklearn = acp.components_.T * np.sqrt(acp.explained_variance_)
    assert np.allclose(b_sklearn, modele.exposures, atol=1e-10)

    # Invariante au signe : celle-ci doit tenir quoi qu'il arrive.
    assert np.allclose(
        b_sklearn @ b_sklearn.T,
        modele.exposures @ modele.exposures.T,
        atol=1e-10,
    )


# ══════════════════════════════════════════════════════════════════════
#  INTÉGRATION AVEC LE RESTE DU MOTEUR
# ══════════════════════════════════════════════════════════════════════


def test_le_modele_sutilise_partout_sans_modification() -> None:
    """
    Un `Portfolio` construit sur une covariance issue du modèle à
    facteurs fonctionne avec tout le moteur des séances 2 à 5 — aucune
    ligne n'a été modifiée ailleurs pour que ce soit vrai.

    C'est le test qui vérifie une propriété d'ARCHITECTURE plutôt qu'un
    calcul.
    """
    r, _ = _rendements_a_facteurs(12, 400, 3, seed=14)
    modele = pca_factor_model(r)

    w = pd.Series(np.full(12, 1 / 12), index=list(modele.assets))
    p = Portfolio(weights=w, covariance=modele.covariance(), name="sur facteurs")

    d = p.decompose()
    assert d.portfolio_volatility > 0
    assert d.percent.sum() == pytest.approx(1.0, rel=1e-10)
    assert 1.0 <= d.effective_bets <= 12.0


def test_les_deux_decompositions_donnent_la_meme_volatilite() -> None:
    """
    L'attribution par facteurs et la décomposition par actifs de la
    séance 2 partent du même portefeuille et doivent trouver la même
    volatilité totale. Deux chemins de calcul entièrement différents,
    un seul résultat.

    C'est le genre de test croisé qui attrape les erreurs d'échelle —
    celles que la séance 1 a appris à craindre.
    """
    r, _ = _rendements_a_facteurs(16, 500, 3, seed=15)
    modele = pca_factor_model(r)
    w = pd.Series(np.full(16, 1 / 16), index=list(modele.assets))

    par_facteurs = attribute_risk(w, modele).total_volatility
    par_actifs = decompose_risk(w, modele.covariance()).portfolio_volatility

    assert par_facteurs == pytest.approx(par_actifs, rel=1e-10)


# ══════════════════════════════════════════════════════════════════════
#  REFUS EXPLICITES
# ══════════════════════════════════════════════════════════════════════


def test_refuse_un_actif_de_variance_nulle() -> None:
    r, _ = _rendements_a_facteurs(8, 200, 2, seed=16)
    r = r.copy()
    r["A03"] = 0.0
    with pytest.raises(ValueError, match="variance nulle"):
        pca_factor_model(r)


def test_refuse_les_valeurs_manquantes() -> None:
    r, _ = _rendements_a_facteurs(8, 200, 2, seed=17)
    r = r.copy()
    r.iloc[5, 2] = np.nan
    with pytest.raises(ValueError, match="manquantes"):
        pca_factor_model(r)


def test_refuse_un_k_hors_bornes() -> None:
    r, _ = _rendements_a_facteurs(8, 200, 2, seed=18)
    with pytest.raises(ValueError, match="n_factors"):
        pca_factor_model(r, n_factors=0)
    with pytest.raises(ValueError, match="n_factors"):
        pca_factor_model(r, n_factors=9)


def test_lattribution_refuse_un_poids_manquant() -> None:
    r, _ = _rendements_a_facteurs(8, 200, 2, seed=19)
    modele = pca_factor_model(r)
    w = pd.Series(np.full(7, 1 / 7), index=list(modele.assets)[:7])
    with pytest.raises(ValueError, match="poids absents"):
        attribute_risk(w, modele)


def test_les_bornes_de_bruit_refusent_des_dimensions_absurdes() -> None:
    with pytest.raises(ValueError):
        marchenko_pastur_bounds(0, 100)
    with pytest.raises(ValueError):
        marchenko_pastur_bounds(10, 0)


# ══════════════════════════════════════════════════════════════════════
#  LE SCÉNARIO DE FACTEUR
# ══════════════════════════════════════════════════════════════════════


def _modele_et_portefeuille(
    n_assets: int = 14, n_obs: int = 600, seed: int = 20
) -> tuple[FactorModel, Portfolio]:
    r, _ = _rendements_a_facteurs(n_assets, n_obs, 3, seed=seed, force=1.0)
    modele = pca_factor_model(r)
    w = pd.Series(np.full(n_assets, 1 / n_assets), index=list(modele.assets))
    return modele, Portfolio(
        weights=w, covariance=modele.covariance(), value=100_000.0, name="test"
    )


def test_le_choc_de_facteur_satisfait_le_protocole_shock() -> None:
    """
    `FactorShock` vit dans `factors.py` et n'importe rien de `stress.py`.
    Il est pourtant un `Shock` au sens du protocole, et se compose avec
    les chocs existants.

    C'est la vérification en conditions réelles de l'argument avancé à la
    séance 5 pour préférer un Protocol à une classe de base abstraite :
    un module entier étend le moteur de scénarios sans que `stress.py`
    apprenne un mot sur les facteurs.
    """
    from nkrisk.stress import Combined, Shock, VolatilityShock

    modele, p = _modele_et_portefeuille()
    choc = FactorShock(modele, factor=1, sigmas=-3.0)

    assert isinstance(choc, Shock)

    compose = Combined([choc, VolatilityShock(2.0)], name="enchaîné")
    assert isinstance(compose, Shock)
    apres = compose.apply(p)
    assert apres.value < p.value
    assert apres.volatility > p.volatility


def test_la_perte_du_portefeuille_egale_son_exposition_au_facteur() -> None:
    """
    IDENTITÉ CROISÉE ENTRE DEUX MODULES.

    Sous un mouvement de s écarts-types du facteur j, le rendement du
    portefeuille vaut exactement s · b_j, où b_j est l'exposition
    calculée par `attribute_risk`.

    Deux chemins de calcul entièrement séparés — l'un passe par les
    poids en unités de risque, l'autre par des mouvements de prix actif
    par actif — qui doivent tomber sur le même nombre. C'est le genre de
    test qui attrape une erreur d'échelle que ni l'un ni l'autre ne
    révélerait seul.
    """
    modele, p = _modele_et_portefeuille()
    attribution = attribute_risk(p.weights, modele, periods_per_year=1)

    for j in (1, 2):
        for s in (-3.0, -1.0, 2.5):
            choc = FactorShock(modele, factor=j, sigmas=s)
            apres = choc.apply(p)
            rendement = apres.value / p.value - 1.0
            attendu = s * attribution.factor_exposures[f"F{j}"]
            assert rendement == pytest.approx(attendu, rel=1e-10)


def test_les_actifs_les_plus_exposes_bougent_le_plus() -> None:
    """
    C'est tout l'intérêt du scénario : on ne choisit pas qui souffre, les
    données l'ont déjà dit. Le classement des mouvements suit donc
    exactement celui des expositions en rendement.

    ⚠️ J'avais d'abord écrit l'assertion dans le mauvais sens : « les plus
    exposés tombent le plus », en comparant les rangs tels quels. Sous un
    choc NÉGATIF, l'actif le plus exposé a le mouvement le plus négatif,
    donc le rang le plus BAS. Les deux classements sont inversés, pas
    égaux.

    Petite erreur, mais exactement du type que la séance 1 a appris à
    craindre : un signe qui ne fait rien planter et retourne la
    conclusion.
    """
    modele, _ = _modele_et_portefeuille()
    exposition = pd.Series(
        modele.exposures[:, 0] * modele.volatilities,
        index=pd.Index(modele.assets, name="actif"),
    )

    baisse = FactorShock(modele, factor=1, sigmas=-3.0).asset_moves()
    hausse = FactorShock(modele, factor=1, sigmas=+3.0).asset_moves()

    # Choc positif : même classement que les expositions.
    assert (hausse.rank() == exposition.rank()).all()
    # Choc négatif : classement exactement inversé.
    assert (baisse.rank() == exposition.rank(ascending=False)).all()
    # Et le plus exposé est bien celui qui perd le plus.
    assert baisse.idxmin() == exposition.idxmax()


def test_les_poids_derivent_apres_le_choc() -> None:
    """
    L'effet qu'on oublie : après le choc, le portefeuille n'a plus la
    composition choisie. L'actif qui a le plus baissé pèse moins.
    """
    modele, p = _modele_et_portefeuille()
    choc = FactorShock(modele, factor=1, sigmas=-3.0)
    apres = choc.apply(p)

    pire = choc.asset_moves().idxmin()
    assert apres.weights[pire] < p.weights[pire]
    assert apres.weights.sum() == pytest.approx(1.0)


def test_lhorizon_met_a_lechelle_en_racine_du_temps() -> None:
    """Dix jours doivent donner √10 fois le mouvement d'un jour."""
    modele, _ = _modele_et_portefeuille()
    un_jour = FactorShock(modele, factor=1, sigmas=-2.0, horizon=1)
    dix_jours = FactorShock(modele, factor=1, sigmas=-2.0, horizon=10)

    rapport = dix_jours.asset_moves() / un_jour.asset_moves()
    assert np.allclose(rapport.to_numpy(), np.sqrt(10.0))


def test_le_choc_de_facteur_ne_touche_pas_le_modele_de_risque() -> None:
    """
    Un scénario de facteur agit sur (poids, valeur) et laisse la
    covariance intacte — la séparation en parts disjointes découverte à
    la séance 5. C'est ce qui fait qu'il commute avec les chocs de
    risque.
    """
    modele, p = _modele_et_portefeuille()
    apres = FactorShock(modele, factor=2, sigmas=-2.0).apply(p)
    assert np.allclose(apres.covariance.matrix, p.covariance.matrix)


def test_les_chocs_de_facteur_et_de_volatilite_commutent() -> None:
    from nkrisk.stress import Combined, VolatilityShock

    modele, p = _modele_et_portefeuille()
    f = FactorShock(modele, factor=1, sigmas=-3.0)
    v = VolatilityShock(1.8)

    a = Combined([f, v], name="a").apply(p)
    b = Combined([v, f], name="b").apply(p)

    assert a.value == pytest.approx(b.value, rel=1e-12)
    assert np.allclose(a.weights.to_numpy(), b.weights.to_numpy())
    assert np.allclose(a.covariance.matrix, b.covariance.matrix)


def test_refuse_un_facteur_inexistant() -> None:
    modele, _ = _modele_et_portefeuille()
    with pytest.raises(ValueError, match="inexistant"):
        FactorShock(modele, factor=modele.n_factors + 1, sigmas=-1.0)


def test_refuse_un_scenario_qui_efface_un_actif() -> None:
    modele, p = _modele_et_portefeuille()
    with pytest.raises(ValueError, match="100 %"):
        FactorShock(modele, factor=1, sigmas=-50.0, horizon=250).apply(p)


def test_refuse_un_univers_qui_ne_correspond_pas() -> None:
    modele, _ = _modele_et_portefeuille(seed=21)
    autre_modele, autre_p = _modele_et_portefeuille(n_assets=9, seed=22)
    with pytest.raises(ValueError, match="mêmes actifs"):
        FactorShock(modele, factor=1, sigmas=-1.0).apply(autre_p)
    assert autre_modele.n_assets == 9


# ══════════════════════════════════════════════════════════════════════
#  LA MESURE QUI DÉCIDE : HORS ÉCHANTILLON
# ══════════════════════════════════════════════════════════════════════


def _monde_a_facteurs(n: int, t: int, seed: int, k: int = 3) -> pd.DataFrame:
    """Univers dont la structure à facteurs est réelle, bruit hétérogène."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 0.7, (n, k))
    b[:, 0] = np.abs(rng.normal(1.0, 0.3, n))
    f = rng.normal(0, 1, (t, k))
    e = rng.normal(0, 1, (t, n)) * rng.uniform(0.5, 1.8, n)
    vols = rng.uniform(0.008, 0.030, n)
    return pd.DataFrame(
        (f @ b.T + e) * vols, columns=[f"A{i:03d}" for i in range(n)]
    )


def _volatilite_realisee_hors_echantillon(
    estimateur: str, dedans: pd.DataFrame, dehors: pd.DataFrame
) -> float:
    """
    Construit le portefeuille de variance minimale sur l'échantillon
    d'estimation, puis mesure sa volatilité RÉALISÉE sur des données
    qu'aucun estimateur n'a vues. Plus bas vaut mieux.

    C'est le protocole d'évaluation de Ledoit & Wolf eux-mêmes, et celui
    de DeMiguel, Garlappi & Uppal (2009). Mesurer la performance dans
    l'échantillon qui a servi à estimer ne dit rien : l'optimiseur y
    exploite son propre bruit.
    """
    from nkrisk.covariance import ledoit_wolf, sample_covariance
    from nkrisk.risk import minimum_variance_weights

    if estimateur == "empirique":
        cov = sample_covariance(dedans)
    elif estimateur == "ledoit-wolf":
        cov = ledoit_wolf(dedans)
    else:
        cov = pca_factor_model(dedans).covariance()

    w = minimum_variance_weights(cov).reindex(dehors.columns).to_numpy()
    serie = dehors.to_numpy() @ w
    return float(np.std(serie, ddof=1) * np.sqrt(252))


@pytest.mark.slow
def test_le_modele_a_facteurs_bat_ledoit_wolf_quand_la_structure_existe() -> None:
    """
    Dans un univers qui A une structure à facteurs, à N/T = 0,4, le
    modèle à facteurs doit produire un portefeuille de variance minimale
    moins volatil hors échantillon que celui construit sur Ledoit-Wolf.

    C'est la seule comparaison qui tranche. Tout le reste — nombre de
    paramètres, conditionnement, élégance — ne sont que des arguments.
    """
    gains = []
    for seed in range(40):
        tout = _monde_a_facteurs(100, 1250, seed)
        dedans, dehors = tout.iloc[:250], tout.iloc[250:]
        lw = _volatilite_realisee_hors_echantillon("ledoit-wolf", dedans, dehors)
        fa = _volatilite_realisee_hors_echantillon("facteurs", dedans, dehors)
        gains.append(lw - fa)

    assert np.mean(gains) > 0, (
        f"le modèle à facteurs ne bat pas Ledoit-Wolf : gain moyen "
        f"{np.mean(gains):.4f}"
    )


@pytest.mark.slow
def test_le_modele_a_facteurs_fonctionne_la_ou_lempirique_nexiste_pas() -> None:
    """
    À N=150 et T=100, la covariance empirique est singulière : il n'y a
    pas de portefeuille de variance minimale à construire. Le modèle à
    facteurs en produit un, et bat largement Ledoit-Wolf.

    Ce régime n'est pas un cas d'école. Un gérant actions qui suit le
    S&P 500 a N=500 ; deux ans d'historique quotidien donnent T=500. Le
    régime N ≈ T est la situation NORMALE de son métier.
    """
    from nkrisk.covariance import sample_covariance

    tout = _monde_a_facteurs(150, 1100, seed=77)
    dedans, dehors = tout.iloc[:100], tout.iloc[100:]

    assert not sample_covariance(dedans).is_invertible

    lw = _volatilite_realisee_hors_echantillon("ledoit-wolf", dedans, dehors)
    fa = _volatilite_realisee_hors_echantillon("facteurs", dedans, dehors)
    assert fa < lw


# ══════════════════════════════════════════════════════════════════════
#  SURFACE DE LECTURE ET GARDE-FOUS DU TYPE
# ══════════════════════════════════════════════════════════════════════


def test_toutes_les_methodes_de_lecture_sexecutent() -> None:
    """
    Test de fumée sur l'affichage.

    Sans intérêt mathématique, et pourtant utile : une erreur de format
    dans une f-string ne se voit qu'à l'exécution, et le moment où on
    découvre que `diagnostic()` plante est en général celui où on
    cherche à comprendre un résultat surprenant.
    """
    modele, p = _modele_et_portefeuille()

    assert modele.n_parameters < modele.n_parameters_free_covariance
    assert modele.compression_ratio > 1.0
    assert modele.exposures_frame().shape == (
        modele.n_assets,
        modele.n_factors,
    )
    assert modele.explained_variance_ratio().shape == (modele.n_factors,)
    assert len(modele.noise_bounds) == 2
    assert "k=" in modele.diagnostic()
    assert str(modele) == modele.diagnostic()

    a = attribute_risk(p.weights, modele)
    table = a.table()
    assert "spécifique" in table.index
    assert 0.0 < a.systematic_share <= 1.0
    assert "systématique" in str(a)
    assert table["% du risque"].sum() == pytest.approx(1.0, rel=1e-10)


def test_le_diagnostic_tient_quand_aucun_facteur_nest_rejete() -> None:
    """Cas limite k = N : il n'y a pas de « première valeur propre rejetée »."""
    r, _ = _rendements_a_facteurs(6, 300, 2, seed=30)
    modele = pca_factor_model(r, n_factors=6)
    assert "nan" in modele.diagnostic().lower()


def test_le_type_refuse_un_modele_incoherent() -> None:
    """
    `FactorModel` valide à la construction, comme `Estimate` et
    `Portfolio`. Un modèle incohérent ne doit pas pouvoir exister.
    """
    r, _ = _rendements_a_facteurs(8, 300, 2, seed=31)
    bon = pca_factor_model(r)

    def variante(**remplace: object) -> FactorModel:
        champs = {
            "exposures": bon.exposures,
            "specific_variances": bon.specific_variances,
            "volatilities": bon.volatilities,
            "eigenvalues": bon.eigenvalues,
            "assets": bon.assets,
            "n_obs": bon.n_obs,
            "method": bon.method,
        }
        champs.update(remplace)
        return FactorModel(**champs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lignes"):
        variante(exposures=bon.exposures[:-1])
    with pytest.raises(ValueError, match="variances spécifiques"):
        variante(specific_variances=bon.specific_variances[:-1])
    with pytest.raises(ValueError, match="volatilités"):
        variante(volatilities=bon.volatilities[:-1])
    with pytest.raises(ValueError, match="négative"):
        variante(specific_variances=bon.specific_variances - 2.0)
    with pytest.raises(ValueError, match="négative"):
        variante(volatilities=bon.volatilities * -1.0)


def test_le_message_dinsufficient_structure_est_exploitable() -> None:
    """
    Une exception dont le message ne dit pas quoi faire est une
    exception inutile. Celle-ci doit nommer la valeur observée, le
    seuil, et les dimensions.
    """
    with pytest.raises(InsufficientStructure) as capture:
        pca_factor_model(_bruit_pur(30, 500, seed=32))

    message = str(capture.value)
    assert "valeur propre" in message
    assert "seuil de bruit" in message
    assert "N=30" in message and "T=500" in message


@pytest.mark.slow
def test_un_grand_univers_bat_un_long_historique() -> None:
    """
    LA COMPARAISON QUI TRANCHE ENTRE LES DEUX DIMENSIONS.

    Quinze actifs avec 51 200 jours d'historique — deux cent trois ans,
    un historique que personne n'aura jamais — contre quatre cents
    actifs avec 2 000 jours, soit huit ans.

    Le grand univers gagne. C'est la forme la plus frappante de la
    consistance en N : à effort de données totalement déséquilibré,
    élargir bat allonger.

    Ce test existe parce que la docstring du module affirme cette
    phrase. Une affirmation chiffrée sans test qui la recalcule finit
    toujours par dériver — c'est arrivé huit fois dans ce projet.
    """
    long_historique = _erreur_de_correlation(15, 51_200, 3, range(12))
    grand_univers = _erreur_de_correlation(400, 2_000, 3, range(8))

    assert grand_univers < long_historique, (
        f"400 actifs sur 8 ans : {grand_univers:.4f} ; "
        f"15 actifs sur 203 ans : {long_historique:.4f}"
    )
