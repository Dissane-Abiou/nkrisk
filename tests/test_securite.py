"""
Tests de sécurité — séance 10.

CE QUI DISTINGUE CES TESTS DES AUTRES
=====================================

Les deux cents tests précédents vérifient que le moteur fait ce qu'il
doit quand on l'utilise comme prévu. Ceux-ci vérifient ce qu'il fait
quand on l'utilise comme il ne faut pas.

C'est une inversion complète de la posture. Un test fonctionnel demande
« est-ce que ça marche ? ». Un test de sécurité demande « qu'est-ce que
je peux lui faire faire qu'il n'est pas censé faire ? », et il n'a de
valeur que si l'attaque qu'il tente **fonctionnait réellement** avant
le correctif.

Les trois premières classes ci-dessous sont des attaques que j'ai
exécutées sur mon propre code de la séance 9, et qui ont réussi.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nkrisk.data.market import (
    CsvSource,
    InMemorySource,
    YahooSource,
    load_prices,
    valider_symbole,
)


@pytest.fixture
def arborescence(tmp_path: Path) -> tuple[Path, Path]:
    """
    Un dossier de données autorisé, et un dossier privé à côté.

    Reproduit la situation réelle : l'application sert des fichiers
    d'un répertoire, et tout le reste du disque est hors limites.
    """
    donnees = tmp_path / "donnees"
    prive = tmp_path / "prive"
    donnees.mkdir()
    prive.mkdir()

    normal = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=5),
            "Close": [10.0, 11.0, 12.0, 11.5, 12.5],
        }
    )
    normal.to_csv(donnees / "AAPL.csv", index=False)

    secret = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=5),
            "Close": [999.0, 999.0, 999.0, 999.0, 999.0],
        }
    )
    secret.to_csv(prive / "secrets.csv", index=False)
    return donnees, prive


# ══════════════════════════════════════════════════════════════════════
#  CWE-22 — TRAVERSÉE DE RÉPERTOIRE
# ══════════════════════════════════════════════════════════════════════


def test_le_cas_normal_fonctionne_toujours(arborescence) -> None:
    """Le contrôle négatif d'abord : durcir ne doit rien casser."""
    donnees, _ = arborescence
    s = CsvSource(donnees).fetch("AAPL")
    assert len(s) == 5
    assert s.iloc[0] == 10.0


@pytest.mark.parametrize(
    "attaque",
    [
        "../prive/secrets",
        "../../prive/secrets",
        "./../prive/secrets",
        "..\\prive\\secrets",
        "sous/../../prive/secrets",
    ],
)
def test_la_remontee_de_repertoire_est_refusee(arborescence, attaque) -> None:
    """
    ⚠️ CETTE ATTAQUE RÉUSSISSAIT SUR LE CODE DE LA SÉANCE 9.

        chemin = Path(self.folder) / f"{ticker}.csv"

    `fetch("../prive/secrets")` lisait le fichier voisin et en rendait
    le contenu comme une série de prix parfaitement ordinaire.
    """
    donnees, _ = arborescence
    with pytest.raises((ValueError, FileNotFoundError)):
        CsvSource(donnees).fetch(attaque)


def test_un_chemin_absolu_ne_remplace_pas_le_dossier(arborescence) -> None:
    """
    ⚠️ LE PIÈGE LE PLUS TRAÎTRE, ET IL EST DANS LA BIBLIOTHÈQUE STANDARD.

    L'opérateur `/` de `pathlib` ABANDONNE son membre de gauche quand
    celui de droite est absolu :

        Path("/data") / "/etc/passwd"   →   Path("/etc/passwd")

    Le dossier autorisé disparaît sans un mot, sans exception, sans
    avertissement. Rien dans la lecture de `a / b` ne suggère que le
    résultat puisse ne pas commencer par `a`.

    C'est documenté, c'est cohérent avec la sémantique POSIX, et c'est
    une source inépuisable de vulnérabilités — précisément parce que le
    code a l'air correct.
    """
    donnees, prive = arborescence
    assert Path("/data") / "/etc/passwd" == Path("/etc/passwd")  # la cause

    with pytest.raises((ValueError, FileNotFoundError)):
        CsvSource(donnees).fetch(str(prive / "secrets"))


def test_un_lien_symbolique_vers_lexterieur_est_refuse(
    arborescence, tmp_path
) -> None:
    """
    LE CAS QUI JUSTIFIE LA DÉFENSE EN PROFONDEUR, ET LE SEUL.

    Le nom « PIEGE » est un symbole parfaitement valide : la validation
    d'entrée ne peut rien y redire, et elle a raison. C'est le système
    de fichiers qui redirige.

    Seul le confinement du chemin RÉSOLU attrape ce cas. Validation et
    confinement ne font donc pas double emploi : chacun couvre
    exactement ce que l'autre laisse passer, et ce test est la preuve
    qu'il en faut deux.
    """
    donnees, prive = arborescence
    lien = donnees / "PIEGE.csv"
    try:
        lien.symlink_to(prive / "secrets.csv")
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("liens symboliques indisponibles sur ce système")

    assert valider_symbole("PIEGE") == "PIEGE"  # la validation laisse passer
    with pytest.raises(ValueError, match="hors du dossier"):
        CsvSource(donnees).fetch("PIEGE")


# ══════════════════════════════════════════════════════════════════════
#  VALIDATION DES SYMBOLES
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "symbole", ["AAPL", "RY.TO", "BRK-B", "^GSPC", "BTC-USD", "A", "X" * 24]
)
def test_les_symboles_legitimes_passent(symbole) -> None:
    """
    Une liste blanche trop étroite est un bogue, pas une protection.
    Ces formes existent toutes sur de vrais marchés.
    """
    assert valider_symbole(symbole) == symbole


@pytest.mark.parametrize(
    "symbole",
    [
        "",
        "..",
        "../etc",
        "/etc/passwd",
        "a/b",
        "a\\b",
        "A" * 25,
        "AAPL\x00",
        "\x1b[2JAAPL",
        ".AAPL",
        "-AAPL",
        "AA PL",
        "AAPL;rm -rf /",
        "AAPL$(whoami)",
        "AAPL`id`",
        "AAPL'--",
    ],
)
def test_les_symboles_hostiles_sont_refuses(symbole) -> None:
    with pytest.raises(ValueError, match="symbole invalide"):
        valider_symbole(symbole)


def test_un_symbole_non_textuel_est_refuse() -> None:
    with pytest.raises(TypeError):
        valider_symbole(42)  # type: ignore[arg-type]


def test_le_message_derreur_ne_renvoie_pas_la_valeur_recue() -> None:
    """
    CWE-117 — INJECTION DANS LES JOURNAUX.

    Un message d'erreur finit dans un journal, et un journal se lit dans
    un terminal. Un symbole contenant des séquences d'échappement ANSI
    peut effacer l'écran, déplacer le curseur, ou réécrire les lignes
    précédentes pour masquer une attaque en cours.

    On signale donc la LONGUEUR et la règle attendue, jamais la valeur.
    Le refus reste diagnosticable sans rendre l'attaquant maître de ce
    qui s'affiche.
    """
    hostile = "\x1b[2J\x1b[1;1H FAUX MESSAGE"
    with pytest.raises(ValueError) as capture:
        valider_symbole(hostile)

    message = str(capture.value)
    assert "\x1b" not in message
    assert "FAUX MESSAGE" not in message
    assert str(len(hostile)) in message       # reste utilisable pour déboguer


def test_la_source_yahoo_valide_avant_dappeler_le_reseau() -> None:
    """
    Le symbole part dans une URL construite par yfinance. On ignore
    comment elle l'échappe, et ce n'est pas à nous de le savoir : ce
    qu'on contrôle, c'est ce qu'on lui transmet.

    Le refus doit survenir AVANT tout appel réseau — d'où l'absence de
    `yfinance` dans ce test, qui passe malgré tout.
    """
    with pytest.raises(ValueError, match="symbole invalide"):
        YahooSource().fetch("../../etc/passwd")


# ══════════════════════════════════════════════════════════════════════
#  ÉPUISEMENT DE RESSOURCES
# ══════════════════════════════════════════════════════════════════════


def test_un_fichier_trop_gros_est_refuse_avant_lecture(arborescence) -> None:
    """
    `read_csv` charge tout en mémoire. Un fichier de plusieurs
    gigaoctets — volontaire, ou simplement mal exporté — fait tomber le
    processus.

    La borne est vérifiée sur la TAILLE du fichier, donc avant d'en
    lire le moindre octet. Vérifier après lecture ne protégerait de
    rien.
    """
    donnees, _ = arborescence
    with pytest.raises(ValueError, match="limite"):
        CsvSource(donnees, max_bytes=10).fetch("AAPL")


def test_trop_de_lignes_interrompt_plutot_que_de_tronquer(
    arborescence,
) -> None:
    """
    Tronquer en silence serait pire que refuser : l'appelant croirait
    analyser la série entière. Même principe que `InsufficientData` à
    la séance 1 — échouer bruyamment est une vertu.
    """
    donnees, _ = arborescence
    with pytest.raises(ValueError, match="lecture interrompue"):
        CsvSource(donnees, max_rows=3).fetch("AAPL")


# ══════════════════════════════════════════════════════════════════════
#  CE QUI N'EST DÉLIBÉRÉMENT PAS EXPOSÉ
# ══════════════════════════════════════════════════════════════════════


def test_aucune_deserialisation_ni_evaluation_dans_le_paquet() -> None:
    """
    DÉCISION DE CONCEPTION, VÉRIFIÉE PLUTÔT QU'AFFIRMÉE.

    `pickle` exécute du code arbitraire à la lecture. C'est la
    vulnérabilité la plus classique des bibliothèques scientifiques
    Python, parce que `pickle` est commode et que le format a l'air
    d'un format de données. Ce n'en est pas un : c'est un langage de
    programmation sérialisé.

    Même raisonnement pour `eval`, `exec`, `os.system` et
    `yaml.load` sans `Loader`.

    Un commentaire disant « on n'utilise pas pickle » vieillit mal. Ce
    test, non : il échouera le jour où quelqu'un l'ajoutera.
    """
    interdits = (
        "import pickle",
        "from pickle",
        "read_pickle",
        "to_pickle",
        "import marshal",
        "eval(",
        "exec(",
        "os.system",
        "subprocess.",
        "__import__(",
        "yaml.load(",
    )
    racine = Path(__file__).resolve().parent.parent / "src" / "nkrisk"
    fautifs: list[str] = []
    for fichier in racine.rglob("*.py"):
        texte = fichier.read_text(encoding="utf-8")
        for motif in interdits:
            if motif in texte:
                fautifs.append(f"{fichier.name} : {motif}")

    assert not fautifs, f"primitives dangereuses introduites : {fautifs}"


def test_le_chargement_groupe_valide_chaque_symbole(arborescence) -> None:
    """
    La validation ne doit pas être contournable en passant par la
    fonction de plus haut niveau — une erreur d'architecture courante,
    où le contrôle vit dans une couche qu'on peut sauter.
    """
    donnees, prive = arborescence
    with pytest.raises(ValueError, match="symbole invalide"):
        load_prices(["AAPL", "../prive/secrets"], CsvSource(donnees))


def test_le_mode_souple_ne_desactive_pas_la_validation(arborescence) -> None:
    """
    `strict=False` sert à tolérer un symbole introuvable, pas à tolérer
    une tentative de traversée. L'échec est consigné, et rien n'est lu.
    """
    donnees, _ = arborescence
    res = load_prices(
        ["AAPL", "../prive/secrets"], CsvSource(donnees), strict=False
    )
    assert set(res.tickers) == {"AAPL"}
    assert "../prive/secrets" in res.failures


def test_la_source_en_memoire_na_pas_besoin_de_validation() -> None:
    """
    Elle ne touche ni au disque ni au réseau, et ses clés viennent du
    code appelant. Y ajouter une validation serait du rite : une
    protection sans menace correspondante coûte de la lisibilité et
    donne l'illusion d'une couverture.
    """
    s = pd.Series(
        [1.0, 2.0], index=pd.bdate_range("2020-01-01", periods=2)
    )
    source = InMemorySource({"n'importe quoi/../": s})
    assert len(source.fetch("n'importe quoi/../")) == 2
