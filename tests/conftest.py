"""
Fixtures et générateurs partagés par toute la suite de tests.

POURQUOI CE FICHIER EXISTE, ET POURQUOI IL ARRIVE TROP TARD
------------------------------------------------------------
Avant la séance 15, le générateur d'univers à facteurs était recopié
dans `test_validation.py`, `test_report.py`, `seance7.py`, `seance12.py`
et `seance14.py`. Le générateur GARCH l'était dans cinq fichiers aussi.
Chacune des copies avait dérivé : signatures différentes, graines
différentes, et dans deux cas une convention d'échelle différente.

C'est le défaut le plus banal d'une suite de tests qui grossit, et
personne ne le voit venir parce que chaque copie, prise seule, est
correcte. Le coût n'apparaît qu'au moment où il faut changer la
génération — il faut alors trouver les cinq, et on en oublie une.

Les générateurs sont ici sous forme de FONCTIONS plutôt que de fixtures
pytest, et c'est délibéré : un test qui a besoin d'un univers de 110
actifs avec des queues de Student n'est pas servi par une fixture
paramétrée au niveau du module. Les fixtures (`univers_propre`,
`univers_agite`…) couvrent les cas courants ; la fonction reste
disponible pour les autres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

#: Paramètres du GARCH(1,1) utilisé partout dans le projet. Variance de
#: long terme égale à 1, persistance α+β = 0,99 — quasi intégrée, ce qui
#: est le régime réaliste et celui qui a cassé deux oracles.
GARCH_ALPHA = 0.09
GARCH_BETA = 0.90
GARCH_OMEGA = 0.01
GARCH_PERSISTANCE = GARCH_ALPHA + GARCH_BETA

#: Longueur par défaut : assez pour que les tests de queue à 2,5 % aient
#: plus de dix dépassements attendus après la fenêtre d'estimation.
N_PERIODES = 1100


def univers_facteurs(
    n_actifs: int = 15,
    graine: int = 0,
    *,
    n_facteurs: int = 3,
    nu: int | None = None,
    garch: bool = False,
    loadings_positifs: bool = True,
    n_periodes: int = N_PERIODES,
    echelle: float = 0.01,
) -> pd.DataFrame:
    """
    Rendements à structure de facteurs connue.

    Paramètres
    ----------
    nu : degrés de liberté d'une loi de Student pour les innovations.
        `None` donne des innovations gaussiennes. Les tirages sont
        toujours remis à variance unité, pour que `nu` change la FORME
        de la queue sans changer le niveau de risque.
    garch : ajoute une volatilité de marché commune qui bouge.
    loadings_positifs : expositions toutes de même signe — un facteur de
        marché, donc des corrélations positives comme sur un vrai
        marché. Mettre `False` donne des signes aléatoires, où la
        corrélation moyenne est voisine de zéro ; la séance 12 a montré
        que le sens de plusieurs résultats s'inverse entre les deux.
    """
    g = np.random.default_rng(graine)
    expositions = g.normal(0.0, 0.5, (n_actifs, n_facteurs))
    if loadings_positifs:
        expositions = np.abs(expositions)

    def tirer(forme: tuple[int, int]) -> np.ndarray:
        if nu is None:
            return g.normal(0.0, 1.0, forme)
        return g.standard_t(nu, forme) * np.sqrt((nu - 2) / nu)

    brut = tirer((n_periodes, n_facteurs)) @ expositions.T + tirer(
        (n_periodes, n_actifs)
    )

    if garch:
        h = np.empty(n_periodes)
        h[0] = 1.0
        choc = g.normal(0.0, 1.0, n_periodes)
        for t in range(n_periodes - 1):
            h[t + 1] = (
                GARCH_OMEGA + GARCH_ALPHA * (choc[t] ** 2) * h[t] + GARCH_BETA * h[t]
            )
        brut = brut * np.sqrt(h)[:, None]

    return pd.DataFrame(
        brut * echelle,
        index=pd.bdate_range("2019-01-02", periods=n_periodes),
        columns=[f"A{i:02d}" for i in range(n_actifs)],
    )


def trajectoire_garch(
    graine: int, n: int = N_PERIODES, nu: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Un GARCH(1,1) de variance longue 1, et sa vraie variance
    conditionnelle.

    Retourne `(innovations, h)`. Le second tableau est ce qui permet de
    construire un ORACLE — une prévision correcte par construction — et
    c'est pour lui que cette fonction existe.

    ⚠️ `h[t]` est connu à la fin de `t-1`, donc utilisable pour prévoir
    `t`. En revanche `h[t+1]` dépend de l'innovation de `t` : s'en
    servir pour normaliser `t` revient à regarder le futur. Ce piège a
    cassé l'oracle de la séance 13, et la remarque est ici pour qu'il ne
    recommence pas ailleurs.
    """
    g = np.random.default_rng(graine)
    h = np.empty(n)
    e = np.empty(n)
    h[0] = 1.0
    echelle = 1.0 if nu is None else np.sqrt((nu - 2) / nu)
    for t in range(n):
        u = g.normal() if nu is None else g.standard_t(nu) * echelle
        e[t] = u * np.sqrt(h[t])
        if t + 1 < n:
            h[t + 1] = GARCH_OMEGA + GARCH_ALPHA * e[t] ** 2 + GARCH_BETA * h[t]
    return e, h


def prevision_garch_causale(h: np.ndarray, debut: int, k: int) -> float:
    """
    La prévision GARCH à k pas, connue en `debut` — le seul oracle
    correct pour un bloc.

        E[h_{t+j} | F_t] = 1 + (α+β)^j · (h_t − 1)

    moyennée sur les k pas. N'utilise que `h[debut]`.
    """
    poids = float(np.mean([GARCH_PERSISTANCE**j for j in range(1, k + 1)]))
    return float(np.sqrt(1.0 + poids * (h[debut] - 1.0)))


def prix_depuis_rendements(
    returns: pd.DataFrame, *, defauts: bool = False
) -> dict[str, pd.Series]:
    """
    Reconstitue des séries de prix, avec ou sans défauts volontaires.

    `defauts=True` glisse les trois anomalies que la séance 9 sait
    reconnaître : une introduction en bourse tardive, un calendrier à
    moitié manquant, et une série figée pendant dix séances.
    """
    prix: dict[str, pd.Series] = {}
    for i, col in enumerate(returns.columns):
        serie = 100.0 * (1.0 + returns[col]).cumprod()
        if defauts:
            if i == 0:
                serie = serie.iloc[400:]
            elif i == 1:
                serie = serie.iloc[::2]
            elif i == 2:
                serie = serie.copy()
                serie.iloc[200:210] = serie.iloc[200]
        prix[col] = serie
    return prix


# ══════════════════════════════════════════════════════════════════════
#  Fixtures pour les cas courants
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def univers_propre() -> pd.DataFrame:
    """Gaussien, volatilité constante, N/T faible. Le témoin."""
    return univers_facteurs(15, graine=1)


@pytest.fixture(scope="session")
def univers_agite() -> pd.DataFrame:
    """Volatilité groupée — le régime où aucun modèle à fenêtre fixe ne tient."""
    return univers_facteurs(15, graine=4, garch=True)


@pytest.fixture(scope="session")
def univers_queues_epaisses() -> pd.DataFrame:
    """Innovations de Student : le corps est bien décrit, la queue non."""
    return univers_facteurs(18, graine=2, nu=4)


@pytest.fixture(scope="session")
def univers_encombre() -> pd.DataFrame:
    """N/T élevé — le régime où l'inversion de la matrice amplifie le bruit."""
    return univers_facteurs(110, graine=3, n_facteurs=5)
