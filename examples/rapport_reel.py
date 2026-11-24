"""
Le rapport de validation sur de VRAIES actions.

À LANCER SUR UNE MACHINE QUI A ACCÈS AU RÉSEAU :

    pip install yfinance
    python examples/rapport_reel.py

Il produit `docs/rapport-validation.html`, qui est le fichier à publier.

POURQUOI CE SCRIPT EST SÉPARÉ DE LA DÉMONSTRATION
-------------------------------------------------
`examples/seance14.py` tourne partout, sans réseau, sur des univers
synthétiques fabriqués pour exercer la détection de conflits. Il prouve
que le rapport FONCTIONNE.

Celui-ci produit un rapport qui dit quelque chose de VRAI sur un vrai
marché. Les deux sont nécessaires et ne se remplacent pas : un rapport
sur données synthétiques ne vaut rien comme verdict, et un rapport sur
données réelles ne prouve rien sur le code.

CE QU'IL FAUT ATTENDRE
----------------------
La séance 9 avait trouvé une calibration de 0,79 sur douze titres de
Toronto et New York — le moteur annonçait 79 % du risque qu'il subissait,
et l'écart était CONCENTRÉ sur 11 % des périodes. Ce rapport reconfronte
ce résultat à tous les instruments construits depuis, et il est probable
qu'il rejette plusieurs des modèles du moteur. C'est l'objectif : un
rapport de validation qui valide tout n'a rien validé.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from nkrisk.data.market import YahooSource, load_prices
from nkrisk.report import build_report
from nkrisk.returns import simple_returns

RACINE = Path(__file__).resolve().parent.parent
CIBLE = RACINE / "docs" / "rapport-validation.html"

#: Un univers volontairement hétérogène : deux places boursières, donc
#: des calendriers différents, et des secteurs qui ne bougent pas
#: ensemble. C'est là que le module de qualité de la séance 9 sert.
TITRES = [
    "XIU.TO",  # indice TSX 60
    "RY.TO",   # banque
    "ENB.TO",  # pipelines
    "SHOP.TO", # technologie, introduction tardive
    "BCE.TO",  # télécoms
    "CNR.TO",  # rail
    "SPY",     # S&P 500
    "QQQ",     # Nasdaq 100
    "XLE",     # énergie américaine
    "TLT",     # obligations longues
    "GLD",     # or
    "VNQ",     # immobilier coté
]

DEBUT = "2015-01-01"


def main() -> int:
    print(f"Chargement de {len(TITRES)} titres depuis {DEBUT}…")
    try:
        resultat = load_prices(
            TITRES, source=YahooSource(), start=DEBUT, strict=False
        )
    except Exception as exc:
        print(f"\n✗ Chargement impossible : {type(exc).__name__}: {exc}")
        print("\n  Vérifier l'accès réseau et que yfinance est installé :")
        print("      pip install yfinance")
        return 1

    prix = resultat.series
    if resultat.failures:
        for ticker, raison in resultat.failures.items():
            print(f"  ⚠ {ticker} non chargé : {raison}")
    if len(prix) < 2:
        print("\n✗ Moins de deux titres chargés : rien à valider.")
        return 1

    print(f"  {len(prix)} titres chargés.")

    # Alignement sur les dates communes. La séance 9 a mesuré que
    # reporter le dernier prix sous-estime la corrélation de 10 % à
    # 10 % de séances manquantes — on prend donc l'intersection, et le
    # rapport signalera ce que cela a coûté.
    cadre = pd.DataFrame(prix).dropna()
    rendements = pd.DataFrame(
        {col: simple_returns(cadre[col]) for col in cadre.columns}
    ).dropna()

    print(
        f"  {len(rendements)} séances communes, "
        f"{rendements.index[0].date()} → {rendements.index[-1].date()}"
    )
    print("\nConstruction du rapport (quelques minutes)…\n")

    rapport = build_report(
        rendements,
        prices=prix,
        universe=f"{len(prix)} titres TSX / NYSE",
        n_sim=6000,
        check_causality=True,
    )

    print(rapport)

    CIBLE.parent.mkdir(exist_ok=True)
    CIBLE.write_text(rapport.to_html(), encoding="utf-8")
    print(f"\n✓ {CIBLE.relative_to(RACINE)} écrit.")
    print("\n  Pour le publier :")
    print("      git add docs/rapport-validation.html")
    print('      git commit -m "Rapport de validation sur donnees reelles"')
    print("      git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
