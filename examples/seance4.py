"""
Séance 4 — Le drawdown, ou le risque tel qu'il est vécu.

    python examples/seance4.py
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from nkrisk.data import GBMSpec, generate_prices
from nkrisk.drawdown import (
    calmar_ratio,
    drawdown_curve,
    drawdown_episodes,
    expected_max_drawdown,
    max_drawdown,
    ulcer_index,
)
from nkrisk.returns import annualized_volatility, log_returns
from nkrisk.var import historical_var


def titre(n: int, texte: str) -> None:
    print(f"\n{'═' * 74}\n  LEÇON {n} — {texte}\n{'═' * 74}")


def prix(lr: np.ndarray) -> pd.Series:
    v = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(lr))])
    return pd.Series(v, index=pd.bdate_range("2020-01-01", periods=len(v)))


# ═══════════════════════════════════════════════════════════════════════
titre(1, "LES MÊMES RENDEMENTS, DANS UN AUTRE ORDRE")

base = np.random.default_rng(479).normal(0.0002, 0.0126, 1260)
rng = np.random.default_rng(1)
lignes = [("ordre d'origine", base),
          ("ordre inversé", base[::-1]),
          ("gains d'abord", np.sort(base)[::-1]),
          ("pertes d'abord", np.sort(base))]
for i in range(3):
    lignes.append((f"mélange {i+1}", rng.permutation(base)))

print("\n  Cinq ans de rendements quotidiens. Le MULTIENSEMBLE ne change")
print("  jamais — seul l'ordre varie.\n")
print(f"    {'ordre':<20}{'volatilité':>12}{'VaR 99 %':>11}"
      f"{'CVaR 99 %':>12}{'max drawdown':>15}")
print(f"    {'-'*20}{'-'*12}{'-'*11}{'-'*12}{'-'*15}")

mdds = []
for nom, r in lignes:
    p = prix(r)
    lr = log_returns(p)
    v = annualized_volatility(lr).value
    t = historical_var(lr, confidence=0.99)
    d = max_drawdown(p).depth
    mdds.append(d)
    print(f"    {nom:<20}{v:>12.2%}{t.var:>11.2%}{t.cvar:>12.2%}{d:>15.2%}")

naturels = [mdds[0], mdds[1], *mdds[4:]]
print("\n  → Volatilité, VaR et CVaR sont identiques sur les sept lignes.")
print("    Elles décrivent une DISTRIBUTION, et la distribution ne bouge pas.")
print(f"\n  → Entre ordres plausibles, le drawdown va déjà de "
      f"{max(naturels):.1%} à {min(naturels):.1%}.")
print(f"    Et les deux ordres adversaires atteignent {mdds[2]:.1%} : avec")
print("    1 260 rendements, enchaîner toutes les pertes suffit à effacer")
print("    quasiment tout le capital. Les deux tris monotones saturent donc")
print("    ensemble ici — la leçon 2 les sépare sur un échantillon court.")
print("\n    Personne ne vit une distribution. On vit un chemin.")


# ═══════════════════════════════════════════════════════════════════════
titre(2, "DEUX IDENTITÉS QUI SE DÉMONTRENT")

a, b = max_drawdown(prix(base)).depth, max_drawdown(prix(base[::-1])).depth
print("\n  INVERSER LA SÉRIE NE CHANGE RIEN")
print(f"    ordre d'origine : {a:.6%}")
print(f"    ordre inversé   : {b:.6%}")
print(f"    écart           : {abs(a-b):.2e}")
print("\n    En rendements log, le drawdown maximal est le maximum, sur toutes")
print("    les fenêtres [i,j], de −(somme des rendements sur la fenêtre).")
print("    Inverser envoie [i,j] sur [n−j, n−i] : même somme. L'ensemble des")
print("    sommes est donc identique, donc son maximum aussi.")
print("\n    → Le drawdown dépend du chemin, mais pas de son SENS.")

print("\n\n  TRIER LES GAINS EN PREMIER EST LE PIRE ORDRE POSSIBLE")
v8 = np.random.default_rng(11).normal(0, 0.03, 8)
tous = [max_drawdown(prix(np.array(p))).depth for p in permutations(v8)]
tri = max_drawdown(prix(np.sort(v8)[::-1])).depth
print(f"\n    Énumération exhaustive des 8! = "
      f"{len(tous):,}".replace(",", " ") + " ordres de 8 rendements :")
print(f"      drawdown le plus faible possible : {max(tous):.2%}")
print(f"      drawdown le plus fort possible   : {min(tous):.2%}")
print(f"      tri décroissant (gains d'abord)  : {tri:.2%}"
      f"{'   ← égal au pire' if abs(tri-min(tous))<1e-12 else ''}")
print("\n    → Aucune permutation ne fait pire. On porte le sommet aussi haut")
print("      que possible, puis on enchaîne toutes les pertes.")


# ═══════════════════════════════════════════════════════════════════════
titre(3, "L'ASYMÉTRIE QUE TOUT LE MONDE SOUS-ESTIME")

print("\n  Pour effacer une perte, il faut un gain plus grand qu'elle :\n")
print(f"    {'perte subie':>14}{'gain nécessaire':>20}{'rapport':>12}")
print(f"    {'-'*14}{'-'*20}{'-'*12}")
for perte in (-0.10, -0.20, -1/3, -0.50, -0.65, -0.80, -0.90):
    p = pd.Series([100.0, 100*(1+perte)],
                  index=pd.bdate_range("2020-01-01", periods=2))
    g = max_drawdown(p).gain_required
    print(f"    {perte:>14.0%}{g:>20.1%}{g/abs(perte):>12.1f}×")

print("\n  → La fonction g = 1/(1+d) − 1 est convexe : le rapport explose.")
print("    C'est la raison arithmétique pour laquelle éviter les grosses")
print("    pertes compte davantage que capter les gros gains.")


# ═══════════════════════════════════════════════════════════════════════
titre(4, "LE DRAWDOWN EST LA STATISTIQUE LA PLUS BRUITÉE DU PROJET")

print("\n  Distribution simulée du maximum drawdown — volatilité 20 %,")
print("  dérive +8 %, 4 000 trajectoires par ligne.\n")
print(f"    {'horizon':>10}{'médiane':>12}{'5 % des cas':>15}"
      f"{'95 % des cas':>15}{'rapport':>11}")
print(f"    {'-'*10}{'-'*12}{'-'*15}{'-'*15}{'-'*11}")

for ans in (1, 2, 3, 5, 10, 20, 30):
    e = expected_max_drawdown(0.20, ans, drift=0.08, n_sim=4000, seed=42)
    print(f"    {str(ans)+' an'+('s' if ans>1 else ''):>10}{e.value:>12.1%}"
          f"{e.ci_high:>15.1%}{e.ci_low:>15.1%}{e.ci_low/e.ci_high:>11.1f}×")

e5 = expected_max_drawdown(0.20, 5, drift=0.08, n_sim=4000, seed=42)
print(f"\n  → À cinq ans, l'intervalle à 90 % s'étend de {e5.ci_low:.0%} "
      f"à {e5.ci_high:.0%}.")
print(f"    Un gérant dont le portefeuille a chuté de {e5.ci_low:.0%} et un autre")
print(f"    de {e5.ci_high:.0%} peuvent avoir exactement le même talent et la même")
print("    stratégie. Le maximum drawdown ne discrimine presque rien.")
print(f"\n  → Le moteur le signale lui-même : "
      f"{'estimation jugée fiable' if e5.is_meaningful() else 'estimation marquée PEU FIABLE'}.")

print("\n\n  ET IL S'APPROFONDIT AVEC LA DURÉE DE DÉTENTION")
m1 = expected_max_drawdown(0.20, 1, drift=0.08, n_sim=4000, seed=42).value
m30 = expected_max_drawdown(0.20, 30, drift=0.08, n_sim=4000, seed=42).value
print(f"\n    Médiane à 1 an : {m1:.1%}     Médiane à 30 ans : {m30:.1%}")
print("\n    Et ce avec une dérive POSITIVE de +8 % par an. Plus on détient")
print("    longtemps, plus on multiplie les occasions de traverser une forte")
print("    baisse, donc plus la pire déjà vécue sera profonde.")
print("\n    → Promettre à un investisseur de long terme qu'il « encaissera")
print("      les creux » revient à lui promettre qu'il en verra de plus en")
print("      plus profonds. C'est le lien entre la statistique et la panique.")


# ═══════════════════════════════════════════════════════════════════════
titre(5, "CE QUE LE MAXIMUM NE VOIT PAS : LE TEMPS PASSÉ SOUS L'EAU")

def idx(n):
    return pd.bdate_range("2020-01-01", periods=n)
rapide = pd.Series([100.0, 70.0] + [100.0]*40, index=idx(42))
lente = pd.Series([100.0] + [70.0]*40 + [100.0], index=idx(42))

print("\n  Deux trajectoires, même chute de 30 % :\n")
sous_eau = "jours sous l'eau"
print(f"    {'trajectoire':<22}{'max drawdown':>15}{sous_eau:>20}{'ulcère':>11}")
print(f"    {'-'*22}{'-'*15}{'-'*20}{'-'*11}")
for nom, s in [("remontée immédiate", rapide), ("40 jours au fond", lente)]:
    dd = drawdown_curve(s)
    print(f"    {nom:<22}{max_drawdown(s).depth:>15.1%}"
          f"{int((dd<0).sum()):>20}{ulcer_index(s):>11.4f}")

print(f"\n  → Maximum drawdown identique. Ulcère "
      f"{ulcer_index(lente)/ulcer_index(rapide):.0f} fois plus élevé.")
print("    L'indice d'ulcère intègre toute la courbe ; le maximum ne lit")
print("    qu'un seul instant.")

print("\n\n  ⚠️  CE QUE J'AVAIS ÉCRIT D'ABORD, ET QUI ÉTAIT FAUX")
u, m = [], []
for seed in range(200):
    p = generate_prices(GBMSpec(0.08, 0.20), 1260, seed=seed)
    u.append(ulcer_index(p))
    m.append(abs(max_drawdown(p).depth))
u, m = np.array(u), np.array(m)
def cv(x):
    return x.std() / x.mean()
print("\n    J'avais présenté l'ulcère comme « le pendant de ce que la CVaR")
print("    est à la VaR : une moyenne plutôt qu'un extremum, donc plus")
print("    stable ». Mesuré sur 200 trajectoires de même loi :\n")
print(f"      coefficient de variation — ulcère {cv(u):.2f} / maximum {cv(m):.2f}")
print(f"\n    L'ulcère est donc {cv(u)/cv(m):.1f} fois PLUS dispersé, pas moins.")
print("    L'analogie était séduisante et fausse : l'ulcère mesure le niveau")
print("    moyen de la trajectoire sous son sommet, et hérite de toute la")
print("    variabilité du chemin — pas seulement de celle d'un extremum")
print("    borné par zéro.")
print("\n    → La conclusion correcte n'est pas « plus fiable » mais")
print("      « autre chose ». Sixième fois dans ce projet qu'une mesure")
print("      contredit ma narration.")


# ═══════════════════════════════════════════════════════════════════════
titre(6, "UN PORTEFEUILLE RÉEL, ÉPISODE PAR ÉPISODE")

p = generate_prices(GBMSpec(0.09, 0.22), 252*12, seed=363)
eps = drawdown_episodes(p, min_depth=0.10)
print("\n  Douze ans, volatilité 22 %. Épisodes dépassant 10 % :\n")
for e in eps[:6]:
    print(f"    {e.format()}")

dd = drawdown_curve(p)
print(f"\n    maximum drawdown        : {max_drawdown(p).depth:>8.1%}")
print(f"    gain nécessaire ensuite : {max_drawdown(p).gain_required:>8.1%}")
print(f"    temps sous le sommet    : {(dd<-0.001).mean():>8.1%} des jours")
print(f"    temps à plus de 10 %    : {(dd<-0.10).mean():>8.1%} des jours")
print(f"    temps à plus de 20 %    : {(dd<-0.20).mean():>8.1%} des jours")
print(f"    indice d'ulcère         : {ulcer_index(p):>8.4f}")
print(f"    ratio de Calmar         : {calmar_ratio(p):>8.2f}")

ref = expected_max_drawdown(annualized_volatility(log_returns(p)).value,
                            12, drift=0.09, n_sim=4000, seed=1)
reel = max_drawdown(p).depth
print("\n  Ce drawdown est-il anormal ? Référence simulée à 12 ans :")
print(f"    médiane attendue : {ref.value:.1%}   intervalle 90 % : "
      f"[{ref.ci_low:.1%} , {ref.ci_high:.1%}]")
verdict = ("dans la zone attendue" if ref.ci_low <= reel <= ref.ci_high
           else "hors de l'intervalle")
print(f"    observé          : {reel:.1%}  →  {verdict}")
print(f"\n  → Sans cette référence, « −{abs(reel):.0%} » ne veut rien dire.")
print("    C'est toute la différence entre un chiffre et une mesure.")
print(f"\n  → Et retiens la ligne « temps sous le sommet » : "
      f"{(dd<-0.001).mean():.0%} des jours.")
print("    Un nouveau plus-haut est rare par construction. L'investisseur")
print("    voit donc presque toujours un chiffre inférieur à son meilleur")
print("    souvenir. C'est la condition normale, pas l'anomalie.")

print(f"\n{'═' * 74}")
print("  Fin de la séance 4.   python -m pytest tests/ -q -m \"\"")
print(f"{'═' * 74}\n")
