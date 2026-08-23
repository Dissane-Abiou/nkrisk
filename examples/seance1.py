"""
Séance 1 — Démonstration.

Lance-moi :  PYTHONPATH=src python3 examples/seance1.py

Chaque section démontre une leçon avec des chiffres réels. Lis la sortie
en même temps que le code.
"""

from __future__ import annotations

import numpy as np

from nkrisk import (
    annualized_mean_return,
    compound_annualize,
    annualized_volatility,
    geometric_return,
    log_returns,
    sharpe_ratio,
    simple_returns,
    volatility_drag,
)
from nkrisk.data import GBMSpec, generate_prices


def titre(n: int, texte: str) -> None:
    print(f"\n{'═' * 72}\n  LEÇON {n} — {texte}\n{'═' * 72}")


# ═══════════════════════════════════════════════════════════════════════
titre(1, "LES DEUX RENDEMENTS NE SONT PAS INTERCHANGEABLES")

prix = generate_prices(GBMSpec(mu=0.08, sigma=0.20), n_days=252 * 10, seed=7)
r_simple = simple_returns(prix)
r_log = log_returns(prix)

print(f"\n  Sur un jour, l'écart est invisible :")
print(f"    rendement simple du jour 1 : {r_simple.iloc[0]:+.6f}")
print(f"    rendement log    du jour 1 : {r_log.iloc[0]:+.6f}")
print(f"    écart                      : {abs(r_simple.iloc[0]-r_log.iloc[0]):.2e}")

print(f"\n  Sur 10 ans, l'écart devient énorme :")
print(f"    somme des rendements simples : {r_simple.sum():+.2%}  ← FAUX")
print(f"    somme des rendements log     : {r_log.sum():+.2%}")
print(f"    performance réellement subie : {prix.iloc[-1]/prix.iloc[0]-1:+.2%}")
print(f"\n  → Additionner des rendements simples dans le temps est un BUG.")
print(f"    exp(somme des log) = {np.exp(r_log.sum())-1:+.2%}  ← correct")


# ═══════════════════════════════════════════════════════════════════════
titre(2, "LA VOLATILITÉ S'ESTIME. LE RENDEMENT, NON.")

print("\n  Mêmes données, mêmes 10 ans. Comparons les deux intervalles :\n")

vol = annualized_volatility(r_log)
moy = annualized_mean_return(r_simple)

print(f"    Volatilité annualisée : {vol.format(2, percent=True)}")
print(f"      → imprécision relative : {vol.relative_uncertainty:.1%}")
print(f"\n    Rendement moyen annuel : {moy.format(2, percent=True)}")
print(f"      → imprécision relative : {moy.relative_uncertainty:.1%}")

print(f"\n  → Avec DIX ANS de données, l'intervalle sur le rendement espéré")
print(f"    est {moy.relative_uncertainty/vol.relative_uncertainty:.0f}× plus large,")
print(f"    en relatif, que celui sur la volatilité.")
print(f"    Et il contient probablement zéro. Vérifions :")
print(f"      borne basse du rendement : {moy.ci_low:+.2%}")
print(f"      contient zéro ? {'OUI' if moy.ci_low < 0 < moy.ci_high else 'non'}")
print(f"\n  C'est LA raison pour laquelle un logiciel qui prédit quoi acheter")
print(f"  s'appuie sur une quantité que personne ne sait mesurer.")


# ═══════════════════════════════════════════════════════════════════════
titre(3, "COMBIEN DE DONNÉES POUR SAVOIR QUELQUE CHOSE ?")

print("\n  Volatilité vraie = 20 %. Que voit-on selon la taille d'échantillon ?\n")
print(f"    {'Historique':<16}{'n':>6}   {'Estimation et intervalle à 95 %':<38}")
print(f"    {'-'*16}{'-'*6}   {'-'*38}")

for libelle, jours in [
    ("1 mois", 21), ("3 mois", 63), ("1 an", 252),
    ("3 ans", 756), ("10 ans", 2520), ("30 ans", 7560),
]:
    p = generate_prices(GBMSpec(mu=0.08, sigma=0.20), n_days=jours, seed=2024)
    e = annualized_volatility(log_returns(p))
    barre = "±" + f"{e.relative_uncertainty:>5.1%}"
    print(f"    {libelle:<16}{jours:>6}   "
          f"{e.value:>6.1%}  [{e.ci_low:>5.1%} — {e.ci_high:>5.1%}]  {barre}")

print(f"\n  → Un mois de données ne permet PAS de distinguer une action")
print(f"    à 15 % de volatilité d'une action à 30 %. Beaucoup")
print(f"    d'applications de courtage affichent pourtant ce chiffre")
print(f"    sans le moindre avertissement.")


# ═══════════════════════════════════════════════════════════════════════
titre(4, "LA TRAÎNÉE DE VOLATILITÉ — CE QUE LE RISQUE TE COÛTE")

print("\n  Cinq actifs avec le MÊME rendement espéré de 8 %,")
print("  mais des volatilités différentes.\n")
print("  ⚠️ Deux pièges méthodologiques évités ici — lis-les, ils valent")
print("     plus que le tableau :")
print("     1. Sur UNE seule simulation, le bruit d'échantillonnage écrase")
print("        l'effet quand σ est petit. On moyenne sur 300 trajectoires.")
print("     2. Le rendement arithmétique doit être annualisé par COMPOSITION,")
print("        comme le géométrique. Ma première version annualisait l'un")
print("        linéairement et l'autre par composition : l'écart de convention")
print("        (0,33 pt) dépassait l'effet mesuré (0,13 pt) et le tableau")
print("        montrait une traînée NÉGATIVE. Un bug qui ne plante pas.\n")
print(f"    {'Volatilité':<12}{'Rdt arithm.':>13}{'Rdt géom.':>12}"
      f"{'Perdu':>9}{'σ²/2':>9}")
print(f"    {'-'*12}{'-'*13}{'-'*12}{'-'*9}{'-'*9}")

for sigma in [0.05, 0.10, 0.20, 0.30, 0.50]:
    geos, aris = [], []
    for s in range(300):
        p = generate_prices(
            GBMSpec(mu=0.08, sigma=sigma), n_days=252 * 30, seed=s + 900
        )
        geos.append(geometric_return(p))
        moy_quot = float(np.mean(simple_returns(p)))
        aris.append(compound_annualize(moy_quot))
    geo, ari = float(np.mean(geos)), float(np.mean(aris))
    print(f"    {sigma:>9.0%}   {ari:>12.2%}{geo:>12.2%}"
          f"{ari-geo:>9.2%}{sigma**2/2:>9.2%}")

print(f"\n  → La colonne 'Perdu' suit la colonne σ²/2. Ce n'est pas une")
print(f"    coïncidence : c'est la formule. La traînée croît en CARRÉ de")
print(f"    la volatilité. Doubler le risque quadruple ce qu'il te coûte.")
print(f"\n  → C'est l'argument quantitatif en faveur de la diversification :")
print(f"    réduire σ augmente ta richesse finale même si le rendement")
print(f"    espéré ne bouge pas d'un millimètre.")


# ═══════════════════════════════════════════════════════════════════════
titre(5, "UN AN DE BONS RÉSULTATS NE PROUVE RIEN")

print("\n  Une stratégie avec un VRAI ratio de Sharpe de 0,5")
print("  (c'est bon — le marché actions fait environ 0,4).\n")
print(f"    {'Historique':<14}{'Sharpe mesuré':>15}"
      f"{'Intervalle à 95 %':>26}{'Prouvé ?':>11}")
print(f"    {'-'*14}{'-'*15}{'-'*26}{'-'*11}")

for libelle, jours in [
    ("1 an", 252), ("3 ans", 756), ("5 ans", 1260),
    ("10 ans", 2520), ("20 ans", 5040),
]:
    p = generate_prices(GBMSpec(mu=0.10, sigma=0.20), n_days=jours, seed=31337)
    e = sharpe_ratio(simple_returns(p))
    prouve = "OUI" if e.ci_low > 0 else "non"
    print(f"    {libelle:<14}{e.value:>15.2f}"
          f"{f'[{e.ci_low:+.2f} ; {e.ci_high:+.2f}]':>26}{prouve:>11}")

print(f"\n  → Il faut des ANNÉES pour prouver qu'une stratégie correcte")
print(f"    vaut mieux que rien. Et tout ceci suppose UN SEUL essai.")
print(f"\n  → Si tu testes 200 variantes et gardes la meilleure, son Sharpe")
print(f"    apparent est contaminé par le biais de sélection. Le théorème")
print(f"    des fausses stratégies (Bailey & López de Prado) dit qu'avec")
print(f"    assez d'essais, AUCUN Sharpe n'est assez élevé pour prouver")
print(f"    quoi que ce soit.")


# ═══════════════════════════════════════════════════════════════════════
titre(6, "LE MOTEUR REFUSE DE RÉPONDRE QUAND IL NE SAIT PAS")

for libelle, jours in [("1 mois", 21), ("3 mois", 63), ("1 an", 252)]:
    p = generate_prices(GBMSpec(mu=0.08, sigma=0.20), n_days=jours, seed=8)
    e = annualized_volatility(log_returns(p))
    verdict = "utilisable" if e.is_meaningful() else "REFUSÉ — trop imprécis"
    print(f"\n    Sur {libelle:<8} {e.format(2, percent=True)}")
    print(f"    {'':<13}→ {verdict}")

print(f"\n  Et chaque estimation transporte ses hypothèses de validité :")
for h in e.assumptions:
    print(f"      • {h}")

print(f"\n  → Aucune fonction de ce moteur ne retourne un nombre nu.")
print(f"    L'incertitude est dans le TYPE. Impossible de l'oublier.")

print(f"\n{'═' * 72}")
print("  Fin de la séance 1. Lance les tests :  python3 -m pytest tests/ -v")
print(f"{'═' * 72}\n")
