"""
Séance 10 — La sécurité, et elle commence par mon propre code.

    python examples/seance10.py

Cette démonstration EXÉCUTE RÉELLEMENT les attaques, d'abord contre une
reproduction du code vulnérable livré à la séance 9, puis contre le code
corrigé. Rien n'est raconté qui ne soit montré.

Tout se passe dans un dossier temporaire, créé et détruit par le script.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nkrisk.data.market import CsvSource, valider_symbole

LARGE = 76


def titre(n: int, texte: str) -> None:
    print()
    print("═" * LARGE)
    print(f"  LEÇON {n} — {texte}")
    print("═" * LARGE)
    print()


# ══════════════════════════════════════════════════════════════════════
#  LA VERSION VULNÉRABLE, REPRODUITE À L'IDENTIQUE
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CsvSourceSeance9:
    """
    Le code tel qu'il a été livré hier. Une seule ligne le distingue du
    code corrigé, et elle a l'air parfaitement normale.
    """

    folder: str | Path
    name: str = "csv (version vulnérable)"

    def fetch(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.Series:
        chemin = Path(self.folder) / f"{ticker}.csv"      # ← la faille
        if not chemin.exists():
            raise FileNotFoundError(f"fichier absent : {chemin}")
        df = pd.read_csv(chemin, parse_dates=["Date"])
        return pd.Series(
            df["Close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(df["Date"]),
            name=ticker,
        )


def preparer(racine: Path) -> tuple[Path, Path]:
    """Un dossier de données autorisé, et un dossier privé à côté."""
    donnees, prive = racine / "donnees", racine / "prive"
    donnees.mkdir()
    prive.mkdir()

    pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=4),
            "Close": [10.0, 10.5, 10.2, 10.8],
        }
    ).to_csv(donnees / "AAPL.csv", index=False)

    pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=4),
            "Close": [111.0, 222.0, 333.0, 444.0],
        }
    ).to_csv(prive / "positions_du_client.csv", index=False)

    return donnees, prive


def essayer(source, symbole: str, legitime: bool = False) -> str:
    """
    Tente une lecture et rend une ligne de verdict.

    `legitime` dit si cette lecture DEVAIT réussir. Sans ce drapeau, le
    même symbole « ⚠️ » marquait un accès normal et une évasion réussie,
    ce qui rend la démonstration illisible — et c'est exactement le
    défaut qu'on reproche à un rapport d'anomalies qui ne hiérarchise
    pas.
    """
    try:
        s = source.fetch(symbole)
        valeurs = ", ".join(f"{v:.0f}" for v in s.to_numpy()[:3])
        marque = "lu" if legitime else "⚠️  LU — ÉVASION RÉUSSIE"
        return f"{marque} — premières valeurs : {valeurs}…"
    except Exception as exc:
        prefixe = "⚠️  REFUS INATTENDU" if legitime else "refusé"
        return f"{prefixe} — {type(exc).__name__} : {str(exc)[:44]}"


with tempfile.TemporaryDirectory() as temporaire:
    RACINE = Path(temporaire)
    DONNEES, PRIVE = preparer(RACINE)

    vulnerable = CsvSourceSeance9(DONNEES)
    corrige = CsvSource(DONNEES)

    # ══════════════════════════════════════════════════════════════════
    titre(1, "UNE LIGNE QUI A L'AIR NORMALE")
    # ══════════════════════════════════════════════════════════════════

    print("  Le chargeur CSV de la séance 9 tenait en une ligne :")
    print()
    print('      chemin = Path(self.folder) / f"{ticker}.csv"')
    print()
    print("  Rien à y redire à la lecture. Le dossier est fixé par la")
    print("  configuration, le symbole vient de l'appelant, et on")
    print("  concatène. C'est ce qu'on écrirait tous.")
    print()
    print("  Dossier autorisé : donnees/   (contient AAPL.csv)")
    print("  Dossier privé    : prive/     (contient "
          "positions_du_client.csv)")
    print()
    print("  Usage normal :")
    print(f"    fetch('AAPL')   {essayer(vulnerable, 'AAPL', legitime=True)}")

    # ══════════════════════════════════════════════════════════════════
    titre(2, "DEUX ÉVASIONS, EXÉCUTÉES POUR DE VRAI")
    # ══════════════════════════════════════════════════════════════════

    attaques = [
        ("../prive/positions_du_client", "remontée de répertoire"),
        (str(PRIVE / "positions_du_client"), "chemin absolu"),
    ]

    print("  Contre le code de la séance 9 :")
    print()
    for charge, description in attaques:
        apercu = charge if len(charge) <= 32 else "…" + charge[-31:]
        print(f"    {description:<24} {apercu:<34}")
        print(f"      → {essayer(vulnerable, charge)}")
    print()
    print("  Les deux réussissent. Le contenu d'un fichier qui n'a rien à")
    print("  faire là ressort comme une série de prix parfaitement")
    print("  ordinaire, et rien dans le programme ne proteste.")
    print()
    print("  ⚠️ LE SECOND CAS EST LE PLUS TRAÎTRE, ET IL VIENT DE LA")
    print("     BIBLIOTHÈQUE STANDARD :")
    print()
    print(f"      Path('/data') / '/etc/passwd'  ==  "
          f"{Path('/data') / '/etc/passwd'}")
    print()
    print("     L'opérateur `/` de pathlib ABANDONNE son membre de gauche")
    print("     quand celui de droite est absolu. Le dossier autorisé")
    print("     disparaît sans exception et sans avertissement.")
    print()
    print("     C'est documenté, c'est cohérent avec POSIX, et c'est une")
    print("     source inépuisable de failles — précisément parce que le")
    print("     code a l'air correct. Rien dans `a / b` ne suggère que le")
    print("     résultat puisse ne pas commencer par `a`.")

    # ══════════════════════════════════════════════════════════════════
    titre(3, "LE CORRECTIF, ET POURQUOI IL A DEUX COUCHES")
    # ══════════════════════════════════════════════════════════════════

    print("  1. LISTE BLANCHE sur le symbole — et non liste noire.")
    print()
    print("     On pourrait interdire `..`, puis `/`, puis `%2e%2e`, puis")
    print("     les équivalents Unicode. Chaque interdiction en oublie")
    print("     une, et l'histoire des traversées de répertoire est")
    print("     l'histoire de ces oublis.")
    print()
    print("     Une liste blanche inverse la charge : tout est refusé")
    print("     sauf ce qu'on a décrit. L'oubli devient un faux refus —")
    print("     visible et corrigeable — au lieu d'un faux accès,")
    print("     silencieux.")
    print()
    print("  2. CONFINEMENT du chemin résolu sous la racine.")
    print()
    print("  Mêmes attaques, code corrigé :")
    print()
    for charge, description in attaques:
        print(f"    {description:<24} → {essayer(corrige, charge)}")
    print(f"    {'usage normal':<24} → "
          f"{essayer(corrige, 'AAPL', legitime=True)}")

    # ══════════════════════════════════════════════════════════════════
    titre(4, "LE CAS QUI PROUVE QU'IL EN FAUT DEUX")
    # ══════════════════════════════════════════════════════════════════

    print("  Objection légitime : si la liste blanche interdit `/`, `\\`")
    print("  et `..`, le confinement ne fait-il pas double emploi ?")
    print()
    print("  Non, et voici le contre-exemple. On place dans le dossier")
    print("  autorisé un lien symbolique qui pointe ailleurs :")
    print()

    lien = DONNEES / "PIEGE.csv"
    lien_possible = True
    try:
        lien.symlink_to(PRIVE / "positions_du_client.csv")
    except (OSError, NotImplementedError):
        lien_possible = False

    if not lien_possible:
        print("    (liens symboliques indisponibles sur ce système —")
        print("     sous Windows ils exigent des droits particuliers)")
        print()
        print("  → Le test `test_un_lien_symbolique_vers_lexterieur_est_refuse`")
        print("    couvre ce cas et se saute automatiquement ici.")
    else:
        print("    donnees/PIEGE.csv  →  prive/positions_du_client.csv")
        print()
        print(f"    validation du nom 'PIEGE' : "
              f"{valider_symbole('PIEGE')!r} — ACCEPTÉ, et c'est correct :")
        print("      « PIEGE » est un nom de symbole parfaitement valide.")
        print("      C'est le système de fichiers qui redirige, pas le nom.")
        print()
        print(f"    version vulnérable  → {essayer(vulnerable, 'PIEGE')}")
        print(f"    version corrigée    → {essayer(corrige, 'PIEGE')}")
        print()
        print("  → La validation laisse passer, le confinement attrape.")
        print("    Et inversement, la validation protège l'adaptateur")
        print("    réseau, que le confinement ne concerne pas du tout.")
        print()
        print("  → **Chaque couche couvre exactement ce que l'autre laisse")
        print("    passer.** La défense en profondeur n'est pas de la")
        print("    redondance par prudence : c'est deux mécanismes aux")
        print("    angles morts différents, et il faut pouvoir nommer")
        print("    lequel attrape quoi.")

    # ══════════════════════════════════════════════════════════════════
    titre(5, "CE QU'ON NE RENVOIE PAS DANS UN MESSAGE D'ERREUR")
    # ══════════════════════════════════════════════════════════════════

    print("  Un message d'erreur finit dans un journal, et un journal se")
    print("  lit dans un terminal.")
    print()
    hostile = "\x1b[2J\x1b[1;1H"
    print(f"  Symbole hostile : {len(hostile)} octets de séquences")
    print("  d'échappement ANSI — « efface l'écran, curseur en haut à")
    print("  gauche ». Un message d'erreur qui les recopierait donnerait")
    print("  à l'attaquant le contrôle de ce que voit l'opérateur.")
    print()
    try:
        valider_symbole(hostile + "AAPL")
    except ValueError as exc:
        message = str(exc)
        print(f"    message rendu : {message[:62]}…")
        print()
        print(f"    contient des octets d'échappement ? "
              f"{'OUI' if chr(27) in message else 'non'}")
        print(f"    contient la valeur reçue ?         "
              f"{'OUI' if hostile in message else 'non'}")
        print(f"    reste diagnosticable ?             "
              f"{'oui — la longueur y est' if str(len(hostile) + 4) in message else 'non'}")
    print()
    print("  → On signale la LONGUEUR et la règle attendue, jamais la")
    print("    valeur. CWE-117, et c'est la raison pour laquelle les")
    print("    messages d'erreur de ce module paraissent avares.")

    # ══════════════════════════════════════════════════════════════════
    titre(6, "ÉPUISEMENT DE RESSOURCES")
    # ══════════════════════════════════════════════════════════════════

    print("  `read_csv` charge tout en mémoire. Un fichier de plusieurs")
    print("  gigaoctets — malveillant, ou simplement mal exporté — fait")
    print("  tomber le processus.")
    print()
    print(f"    taille limitée à 10 octets  → "
          f"{essayer(CsvSource(DONNEES, max_bytes=10), 'AAPL')}")
    print(f"    lignes limitées à 2         → "
          f"{essayer(CsvSource(DONNEES, max_rows=2), 'AAPL')}")
    print(f"    sans limite abaissée        → "
          f"{essayer(corrige, 'AAPL', legitime=True)}")
    print()
    print("  → La taille est vérifiée AVANT d'ouvrir le fichier. La")
    print("    vérifier après lecture ne protégerait de rien.")
    print()
    print("  → Et le dépassement de lignes lève une erreur au lieu de")
    print("    tronquer. Tronquer en silence serait pire que refuser :")
    print("    l'appelant croirait analyser la série entière. Même")
    print("    principe qu'`InsufficientData` à la séance 1.")


# ══════════════════════════════════════════════════════════════════════
titre(7, "LA MENACE QU'AUCUNE DE CES PROTECTIONS NE COUVRE")
# ══════════════════════════════════════════════════════════════════════

print("  Tout ce qui précède relève de catégories classiques : lecture")
print("  hors périmètre, exécution, déni de service, journaux.")
print()
print("  Il en existe une qui ne l'est pas, et c'est la plus sérieuse")
print("  pour un moteur de risque.")
print()
print("  **Un moteur de risque échoue surtout en produisant un chiffre")
print("  FAUX et CRÉDIBLE.** Pas d'intrusion, pas d'exécution, pas")
print("  d'exfiltration — juste une décision prise sur une base fausse.")
print()
print("  Aucune des protections de cette séance n'empêche cela. Les")
print("  contre-mesures du projet contre ce mode de défaillance ne sont")
print("  pas dans un module de sécurité, elles sont dans l'architecture")
print("  entière :")
print()
for ligne in (
    "toute estimation transporte son intervalle, et refuse de",
    "  répondre sous un seuil de fiabilité            (séance 1)",
    "les hypothèses de validité voyagent avec le résultat (séance 1)",
    "le rapport de données signale sans réparer          (séance 9)",
    "l'audit de causalité prouve qu'aucun backtest ne lit",
    "  le futur                                          (séance 8)",
    "le Sharpe dégonflé corrige du nombre d'essais       (séance 8)",
    "la calibration compare l'annoncé au subi            (séance 8)",
):
    print(f"    • {ligne}" if not ligne.startswith("  ") else f"     {ligne}")
print()
print("  → **La sécurité d'un outil de décision, c'est d'abord qu'il ne")
print("    mente pas sur ce qu'il sait.** Le reste est nécessaire, et")
print("    vient après.")
print()
print("  → Le modèle de menace complet, avec ce qui est explicitement")
print("    HORS périmètre, est dans SECURITE.md. Un périmètre non")
print("    déclaré est un périmètre que le lecteur suppose plus large")
print("    qu'il ne l'est — et c'est ainsi qu'on finit par s'appuyer sur")
print("    une protection qui n'existe pas.")

print()
print("═" * LARGE)
print("  Fin de la séance 10.   python -m pytest tests/ -q -m \"\"")
print("═" * LARGE)
