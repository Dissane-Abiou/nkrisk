"""
Séance 5 — Portefeuille et tests de scénario.

    python examples/seance5.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from nkrisk.covariance import ledoit_wolf
from nkrisk.data import GBMSpec, generate_correlated_prices
from nkrisk.portfolio import Portfolio, equal_weighted
from nkrisk.returns import simple_returns
from nkrisk.stress import (
    CHOC_LIQUIDITE,
    CRISE_2008,
    Combined,
    CorrelationShock,
    PriceShock,
    Shock,
    VolatilityShock,
    stress,
)


def titre(n: int, texte: str) -> None:
    print(f"\n{'═' * 76}\n  LEÇON {n} — {texte}\n{'═' * 76}")


NOMS = ["TECH_A", "TECH_B", "TECH_C", "TECH_D", "BANQUE", "OR", "OBLIG"]
VOLS = np.array([0.35, 0.38, 0.32, 0.40, 0.25, 0.18, 0.06])
CORR = np.array([
    [1.00, 0.85, 0.82, 0.88, 0.45, 0.05, -0.10],
    [0.85, 1.00, 0.80, 0.86, 0.42, 0.03, -0.08],
    [0.82, 0.80, 1.00, 0.83, 0.40, 0.06, -0.12],
    [0.88, 0.86, 0.83, 1.00, 0.44, 0.04, -0.09],
    [0.45, 0.42, 0.40, 0.44, 1.00, 0.10, -0.15],
    [0.05, 0.03, 0.06, 0.04, 0.10, 1.00, 0.20],
    [-0.10, -0.08, -0.12, -0.09, -0.15, 0.20, 1.00]])

COV = ledoit_wolf(simple_returns(generate_correlated_prices(
    {n: GBMSpec(0.08, float(v)) for n, v in zip(NOMS, VOLS, strict=True)},
    CORR, 252 * 8, seed=2024)))


def parite_de_risque() -> Portfolio:
    C, n = COV.matrix, len(NOMS)
    w = np.full(n, 1 / n)
    for _ in range(500):
        for i in range(n):
            b = sum(C[i][j] * w[j] for j in range(n) if j != i)
            w[i] = (-b + np.sqrt(b * b + 4 * C[i][i] / n)) / (2 * C[i][i])
    return Portfolio(pd.Series(w / w.sum(), index=NOMS), COV,
                     name="parité de risque")


PLAT = equal_weighted(COV, name="équipondéré")
PARITE = parite_de_risque()


# ═══════════════════════════════════════════════════════════════════════
titre(1, "LA LIMITE QUE CE MODULE LÈVE")

print("""
  Tout ce qu'on a construit jusqu'ici lit le passé. Une VaR calibrée sur
  2003-2007 n'avait jamais vu 2008 — ce n'était pas un défaut
  d'implémentation, c'est la nature d'un estimateur empirique.

  Un test de scénario n'estime rien. Il impose. « Si les volatilités
  doublaient et que les corrélations montaient à 0,9, que deviendrait
  mon risque ? » La question ne demande aucune donnée nouvelle.

  LA DÉCISION D'ARCHITECTURE, et c'est la seule du module :

      tout choc est une fonction   Portefeuille → Portefeuille

  Trois conséquences gratuites :
    • la composition est de la composition de fonctions
    • rien n'est muté, donc on compare l'avant et l'après
    • `Shock` est un Protocol, donc étendre n'exige aucun héritage
""")


# ═══════════════════════════════════════════════════════════════════════
titre(2, "LA CONTRAINTE QUI A CHOISI L'ALGORITHME")

print("\n  « Monter les corrélations » semble trivial : on ajoute un delta.")
print("  Mais une matrice de corrélation doit rester semi-définie positive,")
print("  sinon elle ne décrit aucun monde possible.\n")
print(f"    {'méthode':<34}{'λ minimale':>14}{'verdict':>12}")
print(f"    {'-'*34}{'-'*14}{'-'*12}")
for d in (0.1, 0.3, 0.5):
    M = np.clip(CORR + d, -1, 1)
    np.fill_diagonal(M, 1.0)
    lam = np.linalg.eigvalsh(M).min()
    print(f"    {'ajout naïf de +' + f'{d:.1f}':<34}{lam:>+14.4f}"
          f"{'valide' if lam >= -1e-10 else 'INVALIDE':>12}")
J = np.ones((7, 7))
for a in (0.3, 0.7, 1.0):
    M = (1 - a) * CORR + a * J
    lam = np.linalg.eigvalsh(M).min()
    print(f"    {'combinaison convexe α=' + f'{a:.1f}':<34}{lam:>+14.6f}"
          f"{'valide' if lam >= -1e-10 else 'INVALIDE':>12}")

print("\n  → La combinaison convexe est sûre par construction : J est SDP,")
print("    C est SDP, une combinaison convexe de matrices SDP est SDP, et")
print("    la diagonale reste 1.")

print("\n\n  ⚠️  ET VOICI LE TROU QUE MA PREUVE AVAIT")
print("\n  Elle supposait la CIBLE semi-définie positive, sans le vérifier.")
print("  Une matrice uniforme (1 sur la diagonale, ρ ailleurs) a pour")
print("  valeurs propres 1+(n−1)ρ et 1−ρ. Elle n'est donc valide que si :\n")
print("      ρ ≥ −1/(n−1)\n")
entete = "nombre d'actifs"
print(f"    {entete:>18}{'corrélation minimale':>24}")
print(f"    {'-'*18}{'-'*24}")
for n in (2, 3, 7, 20, 50):
    print(f"    {n:>18}{-1/(n-1):>24.4f}")
print("\n  → Un test balayant toute la plage a trouvé le cas `toward = −0,5`,")
print("    où l'assertion interne a sauté. Exactement comme elle devait.")
print("\n  → Et ce n'est pas un artefact de code : on ne peut pas avoir")
print("    beaucoup d'actifs tous fortement anticorrélés entre eux. La")
print("    diversification par corrélation négative est bornée, et d'autant")
print("    plus que le portefeuille compte de lignes.")


# ═══════════════════════════════════════════════════════════════════════
titre(3, "CE QU'UN CHOC DE PRIX FAIT, ET QU'ON OUBLIE")

krach = PriceShock({"TECH_A": -0.45, "TECH_B": -0.48, "TECH_C": -0.42,
                    "TECH_D": -0.50, "BANQUE": -0.35, "OR": 0.08,
                    "OBLIG": -0.04}, name="krach actions")
r = stress(PLAT, krach)

print(f"\n  Perte immédiate : {r.pnl:.1%}\n")
print(f"    {'position':<10}{'avant':>10}{'après':>10}{'écart':>10}")
print(f"    {'-'*10}{'-'*10}{'-'*10}{'-'*10}")
for a in NOMS:
    av, ap = PLAT.weights[a], r.after.weights[a]
    print(f"    {a:<10}{av:>10.1%}{ap:>10.1%}{ap-av:>+10.1%}")

print("\n  → Le portefeuille n'a plus la composition qu'on avait choisie.")
print(f"    L'or est passé de {PLAT.weights['OR']:.1%} à "
      f"{r.after.weights['OR']:.1%}, les obligations de "
      f"{PLAT.weights['OBLIG']:.1%} à {r.after.weights['OBLIG']:.1%}.")
print("    Personne n'a décidé ça : c'est la dérive mécanique.")
print("\n  → C'est pour cette raison qu'un choc de prix retourne un")
print("    PORTEFEUILLE et pas seulement un montant de perte. Mesurer le")
print("    risque après le choc sur les poids d'avant donnerait un chiffre")
print("    faux.")


# ═══════════════════════════════════════════════════════════════════════
titre(4, "LE RÉSULTAT CONTRE-INTUITIF")

print("\n  Deux portefeuilles, même univers d'actifs :\n")
for p in (PLAT, PARITE):
    print(f"    {p}")

print("\n  Et maintenant on casse les corrélations :\n")
print(f"    {'portefeuille':<20}{'scénario':<26}{'vol avant':>11}"
      f"{'vol après':>11}{'facteur':>10}")
print(f"    {'-'*20}{'-'*26}{'-'*11}{'-'*11}{'-'*10}")
for p in (PLAT, PARITE):
    for s in (CorrelationShock(1.0, toward=0.90, name="corrélations → 0,90"),
              CRISE_2008, CHOC_LIQUIDITE):
        res = stress(p, s)
        print(f"    {p.name:<20}{s.name:<26}{res.vol_before:>11.1%}"
              f"{res.vol_after:>11.1%}{res.vol_ratio:>10.1f}×")

rp = stress(PARITE, CorrelationShock(1.0, toward=0.90))
rl = stress(PLAT, CorrelationShock(1.0, toward=0.90))
print("\n  → Le portefeuille le MIEUX diversifié encaisse le PIRE facteur :")
print(f"    ×{rp.vol_ratio:.1f} contre ×{rl.vol_ratio:.1f}.")
print(f"\n  → La raison : l'équipondéré n'avait déjà que "
      f"{PLAT.decompose().effective_bets:.1f} pari effectif. Il n'avait")
print(f"    presque rien à perdre. La parité de risque en avait "
      f"{PARITE.decompose().effective_bets:.1f} et en")
print(f"    perd l'essentiel — il lui en reste {rp.bets_after:.1f}.")
print("\n  → **La diversification est précisément ce qu'une crise détruit.**")
print(f"    Son niveau absolu reste meilleur ({rp.vol_after:.1%} contre "
      f"{rl.vol_after:.1%}), et")
print("    c'est ce qui compte au bout du compte. Mais la protection qu'on")
print("    croyait avoir s'évapore au moment exact où on en a besoin.")


# ═══════════════════════════════════════════════════════════════════════
titre(5, "LA PROPRIÉTÉ QUE J'AI DÉCOUVERTE EN TESTANT L'INVERSE")

v = VolatilityShock(2.0)
c = CorrelationShock(0.5, toward=0.9)
pc = PriceShock({"TECH_A": -0.5, "OR": 0.2})

print("\n  J'avais documenté qu'un choc de prix ne commute avec aucun autre,")
print("  puisqu'il fait dériver les poids. Les six ordres possibles :\n")
print(f"    {'ordre':<34}{'volatilité':>13}{'valeur':>11}")
print(f"    {'-'*34}{'-'*13}{'-'*11}")
noms = {id(v): "vol", id(c): "corr", id(pc): "prix"}
for ordre in itertools.permutations([v, c, pc]):
    res = Combined(list(ordre)).apply(PLAT)
    lbl = " → ".join(noms[id(s)] for s in ordre)
    print(f"    {lbl:<34}{res.volatility:>13.6%}{res.value:>11.4f}")

print("\n  → Identiques à la précision machine. Ma documentation était fausse.")
print("\n  → La raison est une propriété de la conception : l'état se sépare")
print("    en deux parts disjointes. Un choc de prix touche (poids, valeur),")
print("    un choc de risque touche (covariance). Deux transformations qui")
print("    agissent sur des parts disjointes commutent nécessairement.")
print("\n  → Elle cesserait de tenir avec un choc COUPLÉ — « baisse de trois")
print("    écarts-types », dont l'ampleur dépend de la volatilité courante.")
print("    Le jour où on en ajoute un, il faudra documenter l'ordre.")


# ═══════════════════════════════════════════════════════════════════════
titre(6, "ÉTENDRE LE MOTEUR SANS RIEN IMPORTER")


class ChocConcentration:
    """Un choc défini hors de la bibliothèque, qui n'hérite de rien."""

    name = "liquidation forcée de la tech"

    def apply(self, p: Portfolio) -> Portfolio:
        tech = [a for a in p.assets if a.startswith("TECH")]
        w = p.weights.copy()
        libere = w[tech].sum()
        w[tech] = 0.0
        autres = [a for a in p.assets if a not in tech]
        w[autres] = w[autres] + libere * w[autres] / w[autres].sum()
        return p.with_weights(w, name=f"{p.name} | {self.name}")


perso = ChocConcentration()
print("\n  Cette classe de douze lignes n'importe rien de nkrisk.stress")
print("  et n'hérite de rien. Elle est pourtant un Shock valide :\n")
print(f"    isinstance(perso, Shock)  →  {isinstance(perso, Shock)}\n")
res = stress(PLAT, perso)
print(f"    {res.format()}")
print("\n  → C'est ce que le typage structurel achète. Une classe de base")
print("    abstraite aurait forcé l'utilisateur à importer notre module")
print("    pour étendre le moteur. Un Protocol ne demande rien : un choc")
print("    est défini par ce qu'il FAIT, pas par ce dont il hérite.")

print("\n  → Et il se compose immédiatement avec les chocs existants :\n")
print(f"    {stress(PLAT, Combined([perso, CRISE_2008], name='liquidation puis crise')).format()}")

print(f"\n{'═' * 76}")
print("  Fin de la séance 5.   python -m pytest tests/ -q -m \"\"")
print(f"{'═' * 76}\n")
