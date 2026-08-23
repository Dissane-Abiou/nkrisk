"""
Séance 2 — Covariance et décomposition du risque.

    python examples/seance2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nkrisk.covariance import ledoit_wolf, sample_covariance
from nkrisk.data import GBMSpec, generate_correlated_prices
from nkrisk.returns import simple_returns
from nkrisk.risk import decompose_risk, equal_weights, minimum_variance_weights


def titre(n: int, texte: str) -> None:
    print(f"\n{'═' * 74}\n  LEÇON {n} — {texte}\n{'═' * 74}")


def marche(n_assets: int, n_days: int, seed: int, rho: float = 0.35) -> pd.DataFrame:
    specs = {f"A{i:02d}": GBMSpec(mu=0.08, sigma=0.20) for i in range(n_assets)}
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    return simple_returns(generate_correlated_prices(specs, corr, n_days, seed=seed))


# ═══════════════════════════════════════════════════════════════════════
titre(1, "5 050 PARAMÈTRES, 252 OBSERVATIONS")

print("\n  Pour N actifs, la covariance a N(N+1)/2 paramètres distincts.")
print("  Combien d'observations pour les estimer ? T, la longueur de")
print("  l'historique. Regarde le rapport :\n")
print(f"    {'N actifs':>10}{'paramètres':>13}{'T = 252':>12}{'obs/param':>12}")
print(f"    {'-'*10}{'-'*13}{'-'*12}{'-'*12}")
for n in (5, 20, 50, 100, 500):
    params = n * (n + 1) // 2
    print(f"    {n:>10}{params:>13,}{252:>12}{252/params:>12.3f}")

print("\n  → À 100 actifs, tu as 0,05 observation par paramètre.")
print("    C'est vingt fois moins que le minimum nécessaire pour espérer")
print("    estimer quoi que ce soit.")


# ═══════════════════════════════════════════════════════════════════════
titre(2, "LE CONDITIONNEMENT EXPLOSE QUAND N/T APPROCHE 1")

print("\n  Le conditionnement est le facteur d'amplification du bruit")
print("  lors de l'inversion. T fixé à 120 jours :\n")
print(f"    {'N':>5}{'N/T':>8}{'conditionnement':>20}{'verdict':>22}")
print(f"    {'-'*5}{'-'*8}{'-'*20}{'-'*22}")

for n in (40, 70, 100, 110, 118):
    cov = sample_covariance(marche(n, 120, seed=700))
    c = cov.condition_number
    v = ("excellent" if c < 100 else "utilisable" if c < 1e3
         else "fragile" if c < 1e4 else "DANGEREUX")
    print(f"    {n:>5}{n/120:>8.2f}{c:>20,.0f}{v:>22}")

print("\n  → De 75 à plus d'un million. La dégradation n'est pas linéaire :")
print("    elle explose quand N/T franchit ~0,8.")
print("\n  → Et si N > T, la matrice est SINGULIÈRE. Pas imprécise :")
print("    mathématiquement non inversible. Son rang ne peut pas dépasser T.")


# ═══════════════════════════════════════════════════════════════════════
titre(3, "LE « MAXIMISEUR D'ERREUR », MESURÉ")

print("\n  Le portefeuille de variance minimale inverse la covariance.")
print("  On le calcule sur 15 échantillons INDÉPENDANTS tirés de la MÊME")
print("  distribution. La vérité sous-jacente ne change jamais — les poids")
print("  devraient donc être identiques. Mesurons.\n")
print("  (levier = somme des valeurs absolues des poids ; 1,0 = pas de")
print("   position vendeuse, 20 = 950 % acheteur et 850 % vendeur)\n")
print(f"    {'N':>5}{'N/T':>7}{'levier empirique':>20}{'levier Ledoit-Wolf':>21}")
print(f"    {'-'*5}{'-'*7}{'-'*20}{'-'*21}")

_ratios = []
for n in (40, 70, 100, 110):
    lev_e, lev_l = [], []
    for s in range(15):
        r = marche(n, 120, seed=s + 800)
        lev_e.append(np.abs(minimum_variance_weights(sample_covariance(r))).sum())
        lev_l.append(np.abs(minimum_variance_weights(ledoit_wolf(r))).sum())
    me, ml = float(np.mean(lev_e)), float(np.mean(lev_l))
    _ratios.append(me / ml)
    print(f"    {n:>5}{n/120:>7.2f}{me:>20.2f}{ml:>21.2f}")

print("\n  → L'optimiseur ne trouve pas le portefeuille le moins risqué.")
print("    Il trouve les directions où la covariance est le plus mal")
print("    estimée, et s'y engouffre. Les petites valeurs propres, les plus")
print("    biaisées, deviennent d'énormes valeurs propres après inversion.")
print("\n  → C'est la version rigoureuse du surnom de Michaud (1989) :")
print(f"    « error maximizer ». Ici la contraction divise le levier par")
print(f"    {_ratios[0]:.1f} à {_ratios[-1]:.1f} selon N/T — et le facteur")
print(f"    continue de croître au-delà de N/T = 0,92.")


# ═══════════════════════════════════════════════════════════════════════
titre(4, "LA CONTRACTION DE LEDOIT-WOLF")

print("\n  Σ* = δ·μI + (1-δ)·S")
print("\n  δ se lit comme un rapport bruit/signal, calculé en forme close.")
print("  Aucun paramètre à régler, aucune validation croisée :\n")
print(f"    {'N':>5}{'T':>7}{'N/T':>7}{'δ':>9}{'cond. emp.':>14}{'cond. L-W':>13}")
print(f"    {'-'*5}{'-'*7}{'-'*7}{'-'*9}{'-'*14}{'-'*13}")

for n, t in [(20, 2000), (20, 250), (50, 250), (100, 120), (60, 40)]:
    r = marche(n, t, seed=42)
    e, l = sample_covariance(r), ledoit_wolf(r)
    ce = "singulière" if not e.is_invertible else f"{e.condition_number:,.0f}"
    print(f"    {n:>5}{t:>7}{n/t:>7.2f}{l.shrinkage:>9.1%}{ce:>14}"
          f"{l.condition_number:>13,.0f}")

print("\n  → Quand les données abondent, δ est faible : on garde l'empirique.")
print("    Quand elles se raréfient, δ monte : on se réfugie sur la cible.")
print("\n  → Dernière ligne : 60 actifs pour 40 observations. L'empirique est")
print("    singulière, donc inutilisable. La contractée reste inversible.")
print("\n  → On accepte un BIAIS pour effondrer la VARIANCE. Compromis")
print("    biais-variance appliqué à une matrice.")


# ═══════════════════════════════════════════════════════════════════════
titre(5, "D'OÙ VIENT VRAIMENT TON RISQUE")

# Portefeuille réaliste : une grappe corrélée + des diversifiants
noms = ["TECH_A", "TECH_B", "TECH_C", "TECH_D", "BANQUE", "OR", "OBLIG"]
vols = np.array([0.35, 0.38, 0.32, 0.40, 0.25, 0.18, 0.06])
corr = np.array([
    [1.00, 0.85, 0.82, 0.88, 0.45, 0.05, -0.10],
    [0.85, 1.00, 0.80, 0.86, 0.42, 0.03, -0.08],
    [0.82, 0.80, 1.00, 0.83, 0.40, 0.06, -0.12],
    [0.88, 0.86, 0.83, 1.00, 0.44, 0.04, -0.09],
    [0.45, 0.42, 0.40, 0.44, 1.00, 0.10, -0.15],
    [0.05, 0.03, 0.06, 0.04, 0.10, 1.00, 0.20],
    [-0.10, -0.08, -0.12, -0.09, -0.15, 0.20, 1.00],
])
specs = {n: GBMSpec(mu=0.08, sigma=float(v)) for n, v in zip(noms, vols)}
prix = generate_correlated_prices(specs, corr, 252 * 8, seed=2024)
cov = ledoit_wolf(simple_returns(prix))

w = pd.Series([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10], index=noms)
d = decompose_risk(w, cov)

print("\n  Sept positions, presque équipondérées en CAPITAL.")
print("  Voici la répartition du RISQUE :\n")
tbl = d.table()
print(f"    {'actif':<10}{'poids':>8}{'vol seule':>12}"
      f"{'contrib marg.':>15}{'% du risque':>14}")
print(f"    {'-'*10}{'-'*8}{'-'*12}{'-'*15}{'-'*14}")
for actif, row in tbl.iterrows():
    print(f"    {actif:<10}{row['poids']:>8.1%}{row['vol seule']:>12.1%}"
          f"{row['contrib marginale']:>15.1%}{row['% du risque']:>14.1%}")

print(f"\n    {'TOTAL':<10}{tbl['poids'].sum():>8.1%}{'':>12}{'':>15}"
      f"{tbl['% du risque'].sum():>14.1%}")

for ligne in str(d).split("\n"):
    print(f"    {ligne}")

tech = tbl.loc[[n for n in noms if n.startswith("TECH")], "% du risque"].sum()
cap_tech = tbl.loc[[n for n in noms if n.startswith("TECH")], "poids"].sum()
print(f"\n  → Les quatre lignes tech pèsent {cap_tech:.0%} du capital")
print(f"    mais {tech:.0%} du risque.")
print(f"\n  → L'obligation pèse 10 % du capital et une contribution")
print(f"    marginale NÉGATIVE : en acheter davantage RÉDUIRAIT le risque")
print(f"    total, malgré sa volatilité propre non nulle. C'est l'effet de")
print(f"    corrélation négative, invisible sans cette décomposition.")
print(f"\n  → Nombre effectif de paris : {d.effective_bets:.1f} pour 7 lignes.")
print(f"    Tu crois détenir sept positions. Tu détiens à peu près")
print(f"    {round(d.effective_bets)} pari indépendant.")

print("\n  ⚠️  La somme des contributions vaut EXACTEMENT la volatilité du")
print("     portefeuille. Ce n'est pas une approximation : c'est le théorème")
print("     d'Euler sur les fonctions homogènes de degré 1. Le moteur lève")
print("     une exception si l'identité est violée, parce que ce serait la")
print("     preuve d'un bug en amont.")


# ═══════════════════════════════════════════════════════════════════════
titre(6, "AJOUTER DES LIGNES N'EST PAS DIVERSIFIER")

print("\n  Vingt actifs de volatilité identique, équipondérés.")
print("  Seule la corrélation change :\n")
print(f"    {'corrélation':>13}{'vol du ptf':>14}{'ratio de div.':>16}")
print(f"    {'-'*13}{'-'*14}{'-'*16}")

_vols = []
for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
    r = marche(20, 252 * 10, seed=99, rho=rho)
    c = sample_covariance(r)
    dd = decompose_risk(equal_weights(c), c)
    _vols.append(dd.portfolio_volatility)
    print(f"    {rho:>13.2f}{dd.portfolio_volatility:>14.1%}"
          f"{dd.diversification_ratio:>16.2f}")

print("\n  → Même nombre de lignes, même volatilité individuelle. La")
print(f"    volatilité du portefeuille passe de {_vols[0]:.1%} à {_vols[-1]:.1%}.")
print("\n  → La diversification n'est pas une affaire de NOMBRE de positions.")
print("    C'est une affaire de CORRÉLATION. Vingt actions technologiques")
print("    corrélées à 0,85 forment un portefeuille moins diversifié que")
print("    trois actifs réellement indépendants.")

print(f"\n{'═' * 74}")
print("  Fin de la séance 2.   python -m pytest tests/ -q -m \"\"")
print(f"{'═' * 74}\n")
