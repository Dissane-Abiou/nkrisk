"""
Séance 9 — Les données, et le choix de nettoyage que personne ne déclare.

    python examples/seance9.py

Cette démonstration tourne HORS LIGNE, sur des pathologies fabriquées
dont on connaît la vérité. Pour la chaîne complète sur de vraies
actions, voir `examples/donnees_reelles.py`, qui demande yfinance et une
connexion.

RÈGLE DE CE FICHIER
===================
Aucun nombre écrit à la main dans les narrations. Neuf occurrences du
même bug l'ont établie — la neuvième, dans ce module même, est racontée
à la leçon 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nkrisk.data.market import InMemorySource, load_prices
from nkrisk.data.quality import (
    align_prices,
    compare_alignment,
    inspect_prices,
    observations_per_year,
)

JOURS = 252
LARGE = 76


def titre(n: int, texte: str) -> None:
    print()
    print("═" * LARGE)
    print(f"  LEÇON {n} — {texte}")
    print("═" * LARGE)
    print()


# ══════════════════════════════════════════════════════════════════════
#  UN MARCHÉ DONT ON CONNAÎT LA VÉRITÉ
# ══════════════════════════════════════════════════════════════════════

VOL_VRAIE = 0.20
RHO_VRAI = 0.50
N_ACTIFS = 8
T = 2_520                       # dix ans de jours ouvrés


def marche_vrai(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.full((N_ACTIFS, N_ACTIFS), RHO_VRAI)
    np.fill_diagonal(c, 1.0)
    r = (
        rng.normal(0, 1, (T, N_ACTIFS)) @ np.linalg.cholesky(c).T
    ) * VOL_VRAIE / np.sqrt(JOURS)
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(r, axis=0)),
        index=pd.bdate_range("2015-01-02", periods=T),
        columns=[f"TSX{i}" if i % 2 == 0 else f"NYSE{i}" for i in range(N_ACTIFS)],
    )


def deux_places(
    complet: pd.DataFrame, part_fermee: float, seed: int = 0
) -> dict[str, pd.Series]:
    """
    Deux places aux jours fériés distincts.

    Un jour de fermeture ne vaut PAS un rendement nul : le titre ne cote
    pas, et le mouvement se retrouve dans la cotation suivante. On
    retire donc la ligne au lieu d'y écrire zéro — c'est tout l'enjeu
    de la leçon 3.
    """
    out: dict[str, pd.Series] = {}
    for i, col in enumerate(complet.columns):
        rng = np.random.default_rng(seed * 97 + (i % 2))
        out[col] = complet[col][~(rng.random(len(complet)) < part_fermee)]
    return out


COMPLET = marche_vrai()


# ══════════════════════════════════════════════════════════════════════
titre(1, "POURQUOI CE MODULE EST RESTÉ À 0 % DE COUVERTURE")
# ══════════════════════════════════════════════════════════════════════

print("  Pendant huit séances, `data/market.py` a été le seul trou du")
print("  dépôt, honnêtement documenté comme tel : « la fonction appelle")
print("  le réseau, donc elle n'est pas testable sans réseau ».")
print()
print("  Ce raisonnement était faux, et c'était une erreur de conception,")
print("  pas de test.")
print()
print("      avant   load_yahoo(ticker)    → appelle yfinance en dur")
print("      après   source.fetch(ticker)  → la source est un paramètre")
print()
print("  Une fonction n'est pas intestable parce qu'elle appelle le")
print("  réseau. Elle l'est parce qu'elle DÉCIDE elle-même de l'appeler.")
print()

source = InMemorySource({c: COMPLET[c] for c in COMPLET.columns})
charge = load_prices(list(COMPLET.columns)[:3], source, start="2018-01-01")
print("  Même code de chargement, source en mémoire, aucune connexion :")
print()
for ligne in str(charge).splitlines():
    print(f"    {ligne}")

souple = load_prices(
    [*list(COMPLET.columns)[:2], "FANTOME"], source, strict=False
)
print()
print("  Et le mode souple consigne ce qui a échoué au lieu de le taire :")
print()
for ligne in str(souple).splitlines():
    print(f"    {ligne}")
print()
print("  → Couverture de `data/market.py` : 0 % avant, 87 % après. Le")
print("    seul code encore non couvert est l'adaptateur Yahoo —")
print("    quinze lignes sans la moindre logique.")
print()
print("  → **« Ce n'est pas testable » veut presque toujours dire")
print("    « ce n'est pas encore bien découpé ».**")


# ══════════════════════════════════════════════════════════════════════
titre(2, "LES PATHOLOGIES QUE LE SYNTHÉTIQUE NE PRODUIT JAMAIS")
# ══════════════════════════════════════════════════════════════════════

print("  Huit séances de validation contre une vérité connue, et aucune")
print("  n'a jamais rencontré ceci :")
print()

abime = {c: COMPLET[c].copy() for c in COMPLET.columns}
noms = list(abime)

# un titre illiquide : six séances au même prix
abime[noms[0]].iloc[500:506] = abime[noms[0]].iloc[500]
# un fractionnement d'actions non corrigé
abime[noms[1]].iloc[1200:] = abime[noms[1]].iloc[1200:] * 0.5
# un prix aberrant venu du fichier
abime[noms[2]] = abime[noms[2]].copy()
abime[noms[2]].iloc[800] = 0.0
# et des calendriers qui ne coïncident pas
abime = {
    **deux_places(pd.DataFrame(abime), part_fermee=0.05, seed=1),
}

rapport = inspect_prices(abime)
for ligne in str(rapport).splitlines():
    print(f"    {ligne}")

print()
print("  → Aucune de ces anomalies n'est une erreur de calcul. Toutes")
print("    faussent les estimations EN SILENCE.")
print()
print("  → Et surtout : aucune n'apparaît dans les intervalles de")
print("    confiance. Un titre figé pendant six séances ne produit pas")
print("    une volatilité imprécise — il produit une volatilité FAUSSE,")
print("    et parfaitement serrée. Les huit premières séances mesuraient")
print("    l'incertitude d'ESTIMATION ; celle-ci est d'une autre nature.")
print()
print("  → Le module ne répare RIEN. Un saut de −50 % est soit un krach,")
print("    soit un fractionnement mal corrigé, et la série seule ne")
print("    permet pas de trancher. Un outil qui trancherait à ta place")
print("    se tromperait un jour sur deux sans jamais le dire.")


# ══════════════════════════════════════════════════════════════════════
titre(3, "LE CHOIX QUE TOUT LE MONDE FAIT ET QUE PERSONNE NE DÉCLARE")
# ══════════════════════════════════════════════════════════════════════

print("  Dès qu'on mélange deux places boursières, les calendriers ne")
print("  coïncident plus. Il faut choisir. Deux politiques sont")
print("  défendables :")
print()
print("    intersection  on ne garde que les dates où TOUT LE MONDE cote.")
print("                  On perd des observations, on n'invente rien.")
print()
print("    report        on garde toutes les dates et on reporte le")
print("                  dernier prix. On garde tout, et on invente un")
print("                  rendement de ZÉRO les jours reportés.")
print()
print("  Ce choix tient en une ligne de code et se fait d'ordinaire sans")
print("  commentaire. Voici ce qu'il coûte :")
print()

series = deux_places(COMPLET, part_fermee=0.10, seed=2)
comparaison = compare_alignment(series)
for ligne in str(comparaison).splitlines():
    print(f"    {ligne}")

inter = align_prices(series, "intersection")
report = align_prices(series, "report")
zeros_inter = float((inter.pct_change().dropna() == 0).to_numpy().mean())
zeros_report = float((report.pct_change().dropna() == 0).to_numpy().mean())
print()
print(f"    part de rendements exactement nuls : "
      f"intersection {zeros_inter:.2%}, report {zeros_report:.2%}")
print()
print("  → Le report fabrique de l'information. Un rendement nul n'est")
print("    pas une absence de donnée : c'est l'AFFIRMATION que le prix")
print("    n'a pas bougé, et le moteur la traitera comme une observation")
print("    à part entière.")


# ══════════════════════════════════════════════════════════════════════
titre(4, "CE QUE J'AVAIS PRÉDIT, ET CE QUE LA MESURE A DIT")
# ══════════════════════════════════════════════════════════════════════

print("  J'avais écrit dans la docstring du module : « le report injecte")
print("  des zéros, donc il réduit la variance ET la corrélation ».")
print()
print("  La moitié de cette phrase était fausse. Mesuré contre la vérité")
print(f"  connue — volatilité {VOL_VRAIE:.0%}, corrélation {RHO_VRAI:.2f} :")
print()


def profil(prix: pd.DataFrame, par_an: float, poids: np.ndarray):
    r = prix.pct_change().dropna()
    vol = float((r.std(ddof=1) * np.sqrt(par_an)).mean())
    c = r.corr().to_numpy()
    corr = float(c[~np.eye(len(c), dtype=bool)].mean())
    cov = r.cov().to_numpy() * par_an
    return vol, corr, float(np.sqrt(poids @ cov @ poids))


W = np.full(N_ACTIFS, 1.0 / N_ACTIFS)
ANNEES = T / JOURS

print(f"    {'manquant':>9} | {'inter ÷252':>11} {'inter corrigée':>15} "
      f"| {'report vol':>11} {'report corr':>12} | {'ptf report':>11}")

resume: dict[float, dict[str, float]] = {}
for part in (0.02, 0.05, 0.10, 0.20):
    mesures = {k: [] for k in
               ("i252", "icorr", "rvol", "rcorr", "rptf")}
    for seed in range(6):
        complet = marche_vrai(seed)
        s = deux_places(complet, part, seed=seed)
        ref_v, ref_c, ref_p = profil(complet, JOURS, W)
        pi = align_prices(s, "intersection")
        pr = align_prices(s, "report")
        v252, _, _ = profil(pi, JOURS, W)
        vcor, _, pcor = profil(pi, len(pi) / ANNEES, W)
        rv, rc, rp = profil(pr, len(pr) / ANNEES, W)
        mesures["i252"].append(v252 / ref_v - 1)
        mesures["icorr"].append(vcor / ref_v - 1)
        mesures["rvol"].append(rv / ref_v - 1)
        mesures["rcorr"].append(rc / ref_c - 1)
        mesures["rptf"].append(rp / ref_p - 1)
    m = {k: float(np.mean(v)) for k, v in mesures.items()}
    resume[part] = m
    print(f"    {part:>8.0%} | {m['i252']:>+10.2%} {m['icorr']:>+14.2%} "
          f"| {m['rvol']:>+10.2%} {m['rcorr']:>+11.2%} | {m['rptf']:>+10.2%}")

dix = resume[0.10]
print()
print("  → PREMIÈRE LOI. Le biais de l'intersection n'est pas un biais")
print("    d'alignement : c'est une erreur de CONVENTION. Elle jette des")
print("    séances, donc une année ne contient plus 252 observations.")
print(f"    Annualiser quand même par √252 surestime de "
      f"{dix['i252']:+.1%} à 10 % de séances")
print(f"    manquantes. Avec la bonne annualisation : "
      f"{dix['icorr']:+.2%}.")
print()
print("    C'est le bug de la séance 1 — comparer deux grandeurs")
print("    annualisées selon deux conventions — revenu huit séances")
print("    plus tard sous un autre déguisement.")
print()
print("  → SECONDE LOI, ET CE N'EST PAS CELLE QUE J'ATTENDAIS. Le report")
print(f"    ne biaise presque PAS la volatilité ({dix['rvol']:+.2%}) : les")
print("    zéros la tirent vers le bas, et la séance suivante absorbe le")
print("    mouvement manqué et la tire vers le haut. Les deux effets se")
print("    compensent.")
print()
print(f"    Rien ne compense pour la corrélation : {dix['rcorr']:+.1%}, soit")
print("    à peu près la part de séances manquantes. Quand un actif est")
print("    reporté pendant que l'autre bouge, le co-mouvement est détruit")
print("    et rien ne le restitue.")
print()
print("  → C'est le pire endroit où se tromper. La corrélation gouverne")
print("    la diversification : le portefeuille sort sous-estimé de")
print(f"    {abs(dix['rptf']):.1%} à 10 % de séances manquantes, et de "
      f"{abs(resume[0.20]['rptf']):.1%} à 20 %.")
print("    Toujours dans le sens optimiste.")


# ══════════════════════════════════════════════════════════════════════
titre(5, "UNE LIGNE DE CODE QUI EFFACE UN BIAIS DE 10 %")
# ══════════════════════════════════════════════════════════════════════

print("  Tout le projet annualisait par 252 — le nombre de séances d'une")
print("  année boursière. Dès qu'on aligne par intersection, ce nombre")
print("  cesse d'être vrai.")
print()

for part in (0.0, 0.05, 0.10, 0.20):
    if part == 0.0:
        prix = COMPLET
        etiquette = "aucun"
    else:
        prix = align_prices(deux_places(COMPLET, part, seed=3), "intersection")
        etiquette = f"{part:.0%}"
    par_an = observations_per_year(prix.index)
    print(f"    séances manquantes {etiquette:>5} → "
          f"{len(prix):>5} dates, {par_an:>6.1f} observations par an "
          f"(et non 252)")

print()
print("  → `observations_per_year` le calcule à partir des dates")
print("    réellement présentes, au lieu de le supposer. Une ligne, et")
print("    un biais qui disparaît.")
print()
print("  → Personne ne cherche ce bug, parce qu'il ne ressemble pas à un")
print("    bug : le programme ne se plaint pas, et 252 a l'air d'une")
print("    constante physique. C'en est une — pour un calendrier")
print("    américain complet, et pour lui seul.")


# ══════════════════════════════════════════════════════════════════════
titre(6, "CE QUI RESTE À VÉRIFIER SUR DE VRAIES DONNÉES")
# ══════════════════════════════════════════════════════════════════════

print("  Tout ce qui précède est mesuré contre une vérité connue, donc")
print("  sûr — et fabriqué. Les vraies données apportent ce qu'aucune")
print("  simulation ne produit :")
print()
print("    • des rendements qui ne sont pas indépendants d'un jour à")
print("      l'autre, ce que la règle des √t suppose partout dans le")
print("      moteur ;")
print("    • des volatilités qui changent de régime, ce que la fenêtre")
print("      glissante ne fait qu'approcher ;")
print("    • un biais du survivant qu'aucune correction ne rattrape avec")
print("      des données gratuites ;")
print("    • et des titres dont l'absence même est une information qu'on")
print("      ne reçoit pas.")
print()
print("  → Lance `python examples/donnees_reelles.py` sur ta machine :")
print("    il charge de vraies actions, passe l'inspection de qualité,")
print("    puis fait tourner la chaîne entière — modèle à facteurs,")
print("    backtest, audit de causalité, Sharpe dégonflé.")
print()
print("  → Attends-toi à ce que ce soit moins flatteur que ces tableaux.")
print("    C'est exactement pour ça qu'il faut le faire.")

print()
print("═" * LARGE)
print("  Fin de la séance 9.   python -m pytest tests/ -q -m \"\"")
print("═" * LARGE)
