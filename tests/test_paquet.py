"""
Tests du PAQUET, pas du calcul (séance 15).

Ces tests ne vérifient aucune statistique. Ils vérifient les propriétés
qu'une bibliothèque doit tenir envers ceux qui l'installent, et qui
sont restées fausses pendant quatorze séances sans que rien ne le
signale :

  — les annotations de type servent-elles à quelqu'un ? (PEP 561)
  — l'égalité des objets valeur répond-elle, ou lève-t-elle ?
  — la version est-elle la même partout ?
  — ce qu'on documente comme public l'est-il réellement ?
  — la correction de performance tient-elle ?

Un défaut d'emballage ne casse pas un test de calcul. C'est pour cela
qu'il survit si longtemps, et c'est pour cela que ce fichier existe.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import time
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

import numpy as np
import pandas as pd
import pytest

import nkrisk
from nkrisk.covariance import ledoit_wolf, sample_covariance
from nkrisk.factors import pca_factor_model
from nkrisk.shortfall import EmpiricalNull, TailForecast
from nkrisk.types import value_equal

RACINE = pathlib.Path(__file__).resolve().parent.parent


def _rendements(n: int = 400, k: int = 5, graine: int = 0) -> pd.DataFrame:
    g = np.random.default_rng(graine)
    return pd.DataFrame(g.standard_normal((n, k)) * 0.01)


# ══════════════════════════════════════════════════════════════════════
#  Contrat de typage (PEP 561)
# ══════════════════════════════════════════════════════════════════════


def test_le_marqueur_py_typed_est_livre_avec_le_paquet():
    """
    Sans ce fichier, mypy et les éditeurs traitent `nkrisk` comme du
    code NON TYPÉ chez celui qui l'installe. Les dix mille lignes
    d'annotations ne servent alors qu'à leur auteur.

    Le fichier est vide : c'est sa présence dans la distribution qui
    fait le contrat. D'où ce test, qui vérifie l'emplacement INSTALLÉ
    et non celui du dépôt.
    """
    installe = pathlib.Path(nkrisk.__file__).parent / "py.typed"
    assert installe.exists(), (
        "py.typed absent du paquet installé : les annotations sont "
        "invisibles pour les consommateurs"
    )


def test_pyproject_embarque_le_marqueur():
    config = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    donnees = config["tool"]["setuptools"]["package-data"]
    assert "py.typed" in donnees["nkrisk"]


def test_mypy_est_configure_en_strict():
    """
    Avant la séance 15, « mypy sans erreur » reposait sur deux options.
    Ce test empêche que la configuration soit assouplie en silence pour
    faire passer un ajout pressé.
    """
    config = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["mypy"].get("strict") is True


# ══════════════════════════════════════════════════════════════════════
#  Égalité des objets valeur portant des tableaux
# ══════════════════════════════════════════════════════════════════════


def test_comparer_deux_estimations_identiques_rend_vrai():
    """
    Avant la séance 15, cette ligne levait :

        ValueError: The truth value of an array with more than one
        element is ambiguous.

    Le `__eq__` généré par `dataclass` comparait les champs avec `==`,
    ce qui sur un `ndarray` rend un tableau et non un booléen. Le
    défaut vivait depuis la séance 2 parce qu'aucun test ne comparait
    deux estimations.
    """
    r = _rendements()
    assert sample_covariance(r) == sample_covariance(r)


def test_comparer_deux_estimations_differentes_rend_faux():
    assert sample_covariance(_rendements(graine=1)) != sample_covariance(
        _rendements(graine=2)
    )
    # Même données, méthode différente : le champ `method` les sépare.
    r = _rendements()
    assert sample_covariance(r) != ledoit_wolf(r)


def test_comparer_a_un_objet_dun_autre_type_rend_faux_sans_lever():
    estimation = sample_covariance(_rendements())
    assert estimation != 42
    assert estimation is not None
    assert (estimation == "autre chose") is False


@pytest.mark.parametrize(
    "fabrique",
    [
        lambda: sample_covariance(_rendements()),
        lambda: pca_factor_model(_rendements(n=600, k=8), n_factors=2),
        lambda: TailForecast(
            var=np.full(10, 2.0), es=np.full(10, 2.5), alpha=0.025
        ),
        lambda: EmpiricalNull(np.random.default_rng(0).standard_normal(300)),
    ],
)
def test_les_objets_porteurs_de_tableaux_ne_sont_pas_hachables(fabrique):
    """
    Conséquence VOULUE de la correction : un objet qui porte un tableau
    mutable ne peut pas honorer le contrat du hachage, puisque sa valeur
    peut changer sous lui. Mieux vaut un refus net qu'un dictionnaire
    qui perd ses clés.
    """
    objet = fabrique()
    assert objet == fabrique()
    with pytest.raises(TypeError):
        hash(objet)


def test_value_equal_respecte_le_drapeau_compare_des_champs():
    """
    Un champ marqué `compare=False` ne doit pas entrer dans l'égalité —
    c'est ce qui permet au cache de `EmpiricalNull` de ne pas changer
    l'identité de l'objet.
    """
    residus = np.random.default_rng(3).standard_normal(300)
    a = EmpiricalNull(residus)
    b = EmpiricalNull(residus)
    a.draw(10, 0.025, np.random.default_rng(0))  # remplit le cache de `a`
    assert a._cache and not b._cache
    assert a == b, "le cache ne doit pas entrer dans l'égalité"
    assert value_equal(a, b)


# ══════════════════════════════════════════════════════════════════════
#  Performance : la correction doit tenir
# ══════════════════════════════════════════════════════════════════════


def test_le_cache_ne_change_aucun_resultat():
    """
    Une optimisation qui change un résultat est un bogue. On vérifie
    que la version mise en cache rend exactement ce que rendait la
    version qui recalculait — au bit près.
    """
    residus = np.random.default_rng(4).standard_normal(2000)
    nulle = EmpiricalNull(residus)
    alpha = 0.025

    # Référence : le calcul tel qu'il était avant la mise en cache.
    q = float(np.quantile(residus, alpha))
    attendu = residus / (-q)

    assert np.array_equal(nulle._standardise(alpha), attendu)
    assert np.array_equal(nulle._standardise(alpha), attendu)  # second appel
    # Et sur un autre niveau de queue, le cache ne doit pas se tromper.
    q2 = float(np.quantile(residus, 0.05))
    assert np.array_equal(nulle._standardise(0.05), residus / (-q2))
    assert np.array_equal(nulle._standardise(alpha), attendu)


def test_la_standardisation_nest_calculee_quune_fois_par_niveau():
    """
    LE TEST DE NON-RÉGRESSION DE LA SÉANCE 15.

    `draw` est appelé des milliers de fois pour une seule valeur
    critique. S'il recalcule la standardisation à chaque appel — ce
    qu'il faisait — plus de la moitié du temps d'un rapport part en
    recalcul d'un invariant.

    On compte les appels à `np.quantile` plutôt que de mesurer un
    temps : un compteur est déterministe, une durée dépend de la
    machine qui exécute la CI.
    """
    residus = np.random.default_rng(5).standard_normal(2000)
    nulle = EmpiricalNull(residus)
    rng = np.random.default_rng(0)

    appels = {"n": 0}
    vrai_quantile = np.quantile

    def compte(*args, **kwargs):
        appels["n"] += 1
        return vrai_quantile(*args, **kwargs)

    np.quantile = compte  # type: ignore[assignment]
    try:
        for _ in range(500):
            nulle.draw(100, 0.025, rng)
    finally:
        np.quantile = vrai_quantile  # type: ignore[assignment]

    assert appels["n"] == 1, (
        f"{appels['n']} appels à np.quantile pour 500 tirages — la "
        "standardisation est recalculée dans la boucle"
    )


@pytest.mark.slow
def test_un_rapport_complet_reste_sous_une_borne_de_temps():
    """
    Borne LARGE et volontairement grossière : on attrape une régression
    d'un facteur deux, pas une variation de 10 %. Un seuil serré
    deviendrait instable sur une machine partagée, et un test de
    performance instable finit désactivé.
    """
    from nkrisk.report import build_report

    g = np.random.default_rng(1)
    expositions = np.abs(g.normal(0.0, 0.5, (12, 3)))
    donnees = pd.DataFrame(
        (g.normal(0.0, 1.0, (2834, 3)) @ expositions.T + g.normal(0.0, 1.0, (2834, 12)))
        * 0.01,
        index=pd.bdate_range("2015-05-22", periods=2834),
        columns=[f"A{i:02d}" for i in range(12)],
    )

    debut = time.perf_counter()
    build_report(donnees, universe="performance", n_sim=6000, check_causality=False)
    ecoule = time.perf_counter() - debut

    # Mesuré à 2,0 s après correction, contre 3,8 s avant. La borne à
    # 15 s laisse une marge de sept fois pour une CI lente.
    assert ecoule < 15.0, f"rapport complet en {ecoule:.1f} s"


# ══════════════════════════════════════════════════════════════════════
#  Version et contrat public
# ══════════════════════════════════════════════════════════════════════


def test_pyproject_lit_la_version_depuis_la_source():
    """
    LE GARDE-FOU QUI REMPLACE UN TEST MENSONGER.

    Première version : `__init__.py` portait deux `__version__`, le
    second écrasant le premier.

    Deuxième version : `__version__` lisait
    `importlib.metadata.version("nkrisk")`. L'intention était bonne,
    le résultat dépendait de `sys.path` — sur une machine où une
    installation éditable avait laissé d'anciennes métadonnées,
    `python -c` annonçait 0.1.0 pendant que `pytest`, qui ajoute
    `src/` au chemin, voyait 0.2.0. **Et le test de cohérence passait
    au vert**, parce qu'il s'exécutait précisément dans le contexte où
    les deux concordaient.

    Troisième version, celle-ci : un littéral dans la source, et
    `pyproject` le lit. Les deux ne peuvent plus diverger — non pas
    parce qu'un test les compare, mais parce qu'il n'y a qu'une
    valeur. Ce test vérifie donc la CONFIGURATION, qui est statique,
    et non deux valeurs d'exécution.
    """
    config = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in config["project"].get("dynamic", []), (
        "la version doit être dynamique, sinon elle est recopiée"
    )
    assert "version" not in config["project"], (
        "version littérale dans pyproject : elle peut diverger de la source"
    )
    assert (
        config["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "nkrisk.__version__"
    )


def test_la_version_est_un_litteral_et_non_un_mecanisme():
    """
    Remplacer une duplication par un mécanisme introduit un mode de
    panne que la duplication n'avait pas. Ici, la valeur doit rester
    lisible sans exécuter quoi que ce soit.
    """
    source = (RACINE / "src" / "nkrisk" / "__init__.py").read_text(encoding="utf-8")
    affectations = [
        ligne
        for ligne in source.splitlines()
        if ligne.startswith("__version__")
    ]
    assert len(affectations) == 1, f"{len(affectations)} affectations de version"
    arbre = ast.parse(source)

    # On interroge les IMPORTS RÉELS, pas le texte : une recherche
    # textuelle attrape aussi le commentaire qui explique pourquoi on
    # n'importe pas `importlib.metadata`. Premier jet de ce test, et il
    # échouait sur sa propre justification.
    importes = {
        noeud.module
        for noeud in ast.walk(arbre)
        if isinstance(noeud, ast.ImportFrom) and noeud.module
    } | {
        alias.name
        for noeud in ast.walk(arbre)
        if isinstance(noeud, ast.Import)
        for alias in noeud.names
    }
    assert "importlib.metadata" not in importes, (
        "la version ne doit pas dépendre des métadonnées d'installation : "
        "elles varient avec sys.path"
    )
    valeurs = [
        noeud.value.value
        for noeud in ast.walk(arbre)
        if isinstance(noeud, ast.Assign)
        and isinstance(noeud.value, ast.Constant)
        and any(
            isinstance(c, ast.Name) and c.id == "__version__" for c in noeud.targets
        )
    ]
    assert valeurs == [nkrisk.__version__]


def test_linstallation_nest_pas_perimee():
    """
    Compare le littéral de la source aux métadonnées effectivement
    installées. Ce n'est pas un test du code mais de l'ENVIRONNEMENT :
    une installation éditable périmée fait mentir `pip show`, les
    éditeurs et tout outil qui interroge les métadonnées.

    Il a sa place ici parce que c'est exactement ce qui est arrivé, et
    que le message doit porter le remède.
    """
    try:
        installee = metadata_version("nkrisk")
    except PackageNotFoundError:
        pytest.skip("nkrisk n'est pas installé — exécution depuis les sources")
    assert installee == nkrisk.__version__, (
        f"métadonnées périmées : installé {installee}, source "
        f"{nkrisk.__version__}.\n"
        f"  → pip uninstall -y nkrisk && pip install -e \".[dev]\""
    )


def test_la_version_courante_figure_au_changelog():
    journal = (RACINE / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"[{nkrisk.__version__}]" in journal, (
        f"la version {nkrisk.__version__} n'a pas d'entrée dans CHANGELOG.md"
    )


def test_le_changelog_declare_une_politique_de_stabilite():
    journal = (RACINE / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Politique de stabilité" in journal
    assert "aucune API n'est garantie" in journal


def test_tout_ce_qui_est_declare_public_est_importable():
    """
    Un `__all__` qui mentionne un nom absent casse `from nkrisk import *`
    et trompe les éditeurs. Il faut donc le vérifier, pas le supposer.
    """
    manquants = [nom for nom in nkrisk.__all__ if not hasattr(nkrisk, nom)]
    assert not manquants, f"déclarés dans __all__ mais absents : {manquants}"


def test_aucun_nom_prive_nest_exporte():
    prives = [nom for nom in nkrisk.__all__ if nom.startswith("_")]
    assert not prives, f"noms privés exportés : {prives}"


def test_le_module_racine_documente_ce_qui_est_public():
    doc = nkrisk.__doc__ or ""
    assert "API\npublique" in doc or "API publique" in doc
    assert "CHANGELOG" in doc


def test_les_sous_modules_sont_tous_importables():
    """
    Un module qui n'est importé par aucun autre peut casser sans que la
    suite le voie. On les charge tous explicitement.
    """
    modules = [
        "backtest", "covariance", "drawdown", "ewma", "factors",
        "portfolio", "rendering", "report", "returns", "risk",
        "shortfall", "stress", "types", "validation",
        "data.market", "data.quality", "data.synthetic",
    ]
    for nom in modules:
        assert importlib.import_module(f"nkrisk.{nom}") is not None


def test_la_presentation_ne_contamine_pas_le_domaine():
    """
    La séparation de la séance 15 n'a de valeur que si elle tient :
    `report` ne doit pas importer `rendering` au chargement, sinon le
    domaine redevient dépendant de son format de sortie.
    """
    source = (RACINE / "src" / "nkrisk" / "report.py").read_text(encoding="utf-8")
    entete = source[: source.index("TRADING_DAYS_PER_YEAR")]
    assert "rendering" not in entete, (
        "report.py importe rendering au niveau module : le domaine "
        "dépend de sa présentation"
    )
    assert "<style>" not in source and "font-size" not in source


def test_le_rendu_html_reste_autonome():
    """
    Propriété de la SORTIE, testée depuis le module qui la produit.
    Un rapport archivé pendant dix ans ne doit dépendre de rien.
    """
    from nkrisk.rendering import render_html
    from nkrisk.report import ValidationReport

    page = render_html(
        ValidationReport(
            universe="<b>test</b>",
            window=("2020-01-01", "2024-01-01"),
            n_obs=1000,
            n_assets=10,
        )
    )
    for interdit in ("<script", "http://", "https://", "@import", "<link"):
        assert interdit not in page, f"ressource externe : {interdit}"
    assert "&lt;b&gt;test&lt;/b&gt;" in page
