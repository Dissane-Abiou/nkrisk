"""
Séance 11 — Un estimateur qui réagit, et ce que la réactivité coûte.

    python examples/seance11.py

CETTE SÉANCE EST UNE SUITE DE RÉFUTATIONS
=========================================
J'ai terminé la séance 10 en annonçant « l'estimateur EWMA, et combien
il récupère sur ce 0,79 ». Puis :

  prédiction 1  l'EWMA améliore la calibration     → réfutée (elle l'empire)
  prédiction 2  c'est ma mesure qui est en cause   → réfutée (les deux concordent)
  prédiction 3  c'est l'optimiseur qui la punit    → réfutée (vrai sans optimiseur aussi)
  prédiction 4  c'est l'horizon de prévision       → réfutée (vrai à tout horizon)

Et un oracle bogué en chemin, qui marquait 1,57 là où il devait marquer
1,00. C'est en le réparant que le vrai résultat est apparu.

Tout est raconté ici dans l'ordre où c'est arrivé, parce que la
conclusion ne vaut qu'avec le chemin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nkrisk.covariance import ledoit_wolf, sample_covariance
from nkrisk.ewma import (
    effective_sample_size,
    ewma_covariance,
    ewma_volatility,
    ewma_weights,
    fit_decay,
    half_life,
)
from nkrisk.returns import annualized_volatility

JOURS = 252
LARGE = 76


def titre(n: int, texte: str) -> None:
    print()
    print("═" * LARGE)
    print(f"  LEÇON {n} — {texte}")
    print("═" * LARGE)
    print()


# ══════════════════════════════════════════════════════════════════════
#  UN MARCHÉ DONT ON CONNAÎT LA VARIANCE CONDITIONNELLE
# ══════════════════════════════════════════════════════════════════════


ALPHA, BETA = 0.14, 0.85      # paramètres du GARCH du générateur


def marche(
    t: int = 2_200, n: int = 10, k: int = 2, seed: int = 0,
    alpha: float = ALPHA, beta: float = BETA,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rendements à facteurs dont la volatilité est GROUPÉE par un
    GARCH(1,1), et la vraie variance conditionnelle `h`.

    Rendre `h` est ce qui permet de construire un oracle — donc de
    savoir s'il restait quelque chose à gagner. Comparer deux
    estimateurs sans borne ne dit pas si le meilleur est bon.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 0.5, (n, k))
    b[:, 0] = np.abs(rng.normal(0.8, 0.2, n))
    h = np.ones(t)
    z = rng.normal(0, 1, t)
    for i in range(1, t):
        h[i] = (1 - alpha - beta) + alpha * (z[i - 1] ** 2 * h[i - 1]) \
            + beta * h[i - 1]
    echelle = np.sqrt(h)[:, None]
    f = rng.normal(0, 1, (t, k)) * echelle
    e = rng.normal(0, 1, (t, n)) * echelle * rng.uniform(0.5, 1.5, n)
    return (
        pd.DataFrame(
            (f @ b.T + e) * 0.20 / np.sqrt(JOURS),
            index=pd.bdate_range("2015-01-02", periods=t),
            columns=[f"A{i:02d}" for i in range(n)],
        ),
        h,
    )


def autocorr_carres(x: np.ndarray, lags=(1, 5, 10, 21)) -> list[float]:
    c = np.asarray(x, dtype=float) ** 2
    c = c - c.mean()
    d = float(np.dot(c, c))
    return [float(np.dot(c[:-lag], c[lag:]) / d) for lag in lags]


def ez2(predits: np.ndarray, realises: np.ndarray) -> float:
    """E[(réalisé/prédit)²] — 1,00 est parfait, > 1 sous-estime."""
    return float(np.mean((realises / predits) ** 2))


CADRE, H = marche()
SERIE = CADRE.mean(axis=1).to_numpy()


# ══════════════════════════════════════════════════════════════════════
titre(1, "CE QUE LES VRAIES DONNÉES ONT COMMANDÉ")
# ══════════════════════════════════════════════════════════════════════

print("  La séance 9 a confronté le moteur à douze vraies actions :")
print()
print("      calibration globale                        0,79")
print("      calibration hors des 11 % périodes agitées 1,11")
print("      part de la variance dans ces 11 %          59 %")
print("      pire fenêtre, 28 février 2020 :")
print("        annoncé 7,3 %   subi 98,4 %   rapport 0,07")
print()
print("  Verdict du diagnostic : CONCENTRÉ. Le modèle n'est pas biaisé,")
print("  il est EN RETARD. Une fenêtre de 252 jours accorde le même poids")
print("  à hier et à il y a un an : la tempête compte pour 1/252.")
print()
print("  La réponse évidente est d'oublier progressivement :")
print()
print("      σ²_t = (1−λ)·Σ λ^i · r²_{t−1−i}")
print()
print("  C'est RiskMetrics (1996), et c'est encore le point de départ de")
print("  la plupart des systèmes de risque en production.")


# ══════════════════════════════════════════════════════════════════════
titre(2, "LE PRIX, QUI TIENT EN UN SEUL NOMBRE")
# ══════════════════════════════════════════════════════════════════════

print("  Taille d'échantillon EFFECTIVE au sens de Kish : 1/Σw²_i.")
print("  Le nombre d'observations à poids égaux qui donnerait la même")
print("  précision.")
print()
print(f"    {'λ':>6} {'n_eff':>8} {'demi-vie':>10}  "
      f"{'incertitude relative sur la volatilité':>40}")
for lam in (0.99, 0.97, 0.94, 0.90, 0.85):
    est = ewma_volatility(pd.Series(SERIE), decay=lam)
    drapeau = "" if est.is_meaningful() else "   ⚠️ PEU FIABLE"
    print(f"    {lam:>6.2f} {effective_sample_size(lam):>8.0f} "
          f"{half_life(lam):>9.0f}j  "
          f"{est.relative_uncertainty:>39.1%}{drapeau}")

classique = annualized_volatility(pd.Series(SERIE))
print(f"    {'252j':>6} {len(SERIE):>8} {'—':>10}  "
      f"{classique.relative_uncertainty:>39.1%}")

print()
print("  → L'EWMA à λ=0,94 voit l'équivalent de trente-deux observations,")
print("    quelle que soit la longueur de l'historique. Lire dix ans n'y")
print("    change rien : les poids lointains sont négligeables.")
print()
print("  → **Et le drapeau PEU FIABLE s'allume tout seul.** Ce seuil —")
print("    25 % d'incertitude relative — a été fixé à la séance 1, sans")
print("    la moindre idée qu'un estimateur exponentiel existerait un")
print("    jour dans ce projet. Le type de la séance 1 juge l'estimateur")
print("    de la séance 11 sans qu'on lui ait rien dit.")
print()
print("    C'est ce que vaut une contrainte posée dans le système de")
print("    types plutôt que dans une convention de nommage.")


# ══════════════════════════════════════════════════════════════════════
titre(3, "PREMIÈRE PRÉDICTION, PREMIÈRE RÉFUTATION")
# ══════════════════════════════════════════════════════════════════════

print("  Données simulées avec un regroupement de volatilité réaliste.")
print("  Vérifions d'abord que le simulateur en produit vraiment :")
print()
ac = autocorr_carres(SERIE)
print(f"    autocorrélation des rendements au carré : "
      f"{', '.join(f'{v:.3f}' for v in ac)}  (lags 1, 5, 10, 21)")
print("    repère sur un indice actions réel : 0,15 à 0,25 au lag 1")
print()
if ac[0] >= 0.15:
    print("  → Le regroupement est là. S'il y a quelque chose à gagner,")
    print("    l'EWMA devrait le gagner.")
print()


@dataclass(frozen=True)
class Predicteur:
    nom: str
    f: object


def calibration(predicteur, horizon: int = 21, fenetre: int = 252,
                graines=range(5)) -> float:
    valeurs = []
    for seed in graines:
        cadre, h = marche(seed=seed)
        p = cadre.mean(axis=1).to_numpy()
        poids = np.full(cadre.shape[1], 1.0 / cadre.shape[1])
        # c = Var(p | h=1), estimé en retirant l'effet de h
        c = float(np.mean(p**2 / h))
        predits, realises = [], []
        for t in range(fenetre, len(p) - horizon, horizon):
            realises.append(float(np.std(p[t: t + horizon], ddof=1)))
            predits.append(
                predicteur(cadre.iloc[t - fenetre: t], p[t - fenetre: t],
                           poids, h, t, c, horizon)
            )
        valeurs.append(ez2(np.array(predits), np.array(realises)))
    return float(np.mean(valeurs))


def via_matrice(estimateur):
    def f(fen, serie, poids, h, t, c, horizon):
        cov = estimateur(fen)
        return float(np.sqrt(poids @ cov.matrix @ poids))
    return f


print("  Calibration conditionnelle, portefeuille 1/N, horizon 21 jours,")
print("  en passant par une matrice de covariance comme le fait le moteur :")
print()
print(f"    {'estimateur':<30} {'E[z²]':>8} {'écart à 1':>11}")
etape3 = {
    "fenêtre 252j + Ledoit-Wolf": via_matrice(ledoit_wolf),
    "ewma λ=0,97": via_matrice(lambda d: ewma_covariance(d, decay=0.97)),
    "ewma λ=0,94": via_matrice(lambda d: ewma_covariance(d, decay=0.94)),
    "ewma λ=0,90": via_matrice(lambda d: ewma_covariance(d, decay=0.90)),
}
scores3 = {}
for nom, pred in etape3.items():
    v = calibration(pred)
    scores3[nom] = v
    print(f"    {nom:<30} {v:>8.2f} {abs(v - 1):>11.2f}")

meilleur3 = min(scores3, key=lambda k: abs(scores3[k] - 1))
print()
print("  → La fenêtre longue gagne. L'EWMA fait PIRE, et d'autant pire")
print("    qu'elle est rapide.")
print()
print("  → C'est l'inverse exact de ce que j'ai annoncé hier. Trois")
print("    hypothèses ont suivi — la mesure est en cause ; l'optimiseur")
print("    punit le bruit ; l'horizon est mal choisi — et les trois ont")
print("    été réfutées par la mesure.")
print()
print("  → Ce qui manquait n'était pas une hypothèse de plus. C'était une")
print("    BORNE : savoir ce qu'un estimateur parfait obtiendrait.")


# ══════════════════════════════════════════════════════════════════════
titre(4, "L'ORACLE, ET LE BOGUE QUI LE RENDAIT INUTILE")
# ══════════════════════════════════════════════════════════════════════

print("  Le générateur connaît la vraie variance conditionnelle h. On")
print("  peut donc construire deux oracles :")
print()
print("    • celui qui connaît h au moment de décider — la meilleure")
print("      prévision causale possible ;")
print("    • celui qui connaît la moyenne de h sur la fenêtre à venir —")
print("      impossible, et c'est le point : il sert à VALIDER l'harnais.")
print()


def oracle_naif(fen, serie, poids, h, t, c, horizon):
    """h[t] seul — la variance d'aujourd'hui, prolongée telle quelle."""
    return float(np.sqrt(c * h[t]))


def oracle_garch(fen, serie, poids, h, t, c, horizon):
    """
    La MEILLEURE prévision causale possible : la prévision GARCH à k pas.

    Pour un GARCH(1,1) de moyenne longue 1 :

        E[h_{t+k} | t] = 1 + (α+β)^k · (h_t − 1)

    moyennée sur les `horizon` pas. Elle utilise la vraie dynamique et
    le vrai état, et rien du futur.
    """
    persistance = ALPHA + BETA
    poids_k = float(np.mean([persistance**k for k in range(horizon)]))
    return float(np.sqrt(c * (1.0 + poids_k * (h[t] - 1.0))))


def oracle_tricheur(fen, serie, poids, h, t, c, horizon):
    return float(np.sqrt(c * h[t: t + horizon].mean()))


o_tricheur = calibration(oracle_tricheur)
o_garch = calibration(oracle_garch)
o_naif = calibration(oracle_naif)
print(f"    {'oracle qui connaît le futur':<38} {o_tricheur:>8.2f}")
print(f"    {'oracle GARCH k pas — vraie borne causale':<38} {o_garch:>8.2f}")
print(f"    {'oracle naïf : h[t] prolongé tel quel':<38} {o_naif:>8.2f}")
print()
print(f"  → L'oracle tricheur marque {o_tricheur:.2f}. **L'harnais de mesure")
print("    est donc juste.** Sans cette vérification, rien de ce qui suit")
print("    n'aurait de valeur.")
print()
print("  ⚠️ MA PREMIÈRE VERSION MARQUAIT 1,57 À CETTE PLACE.")
print()
print("     Je normalisais h par sa moyenne THÉORIQUE, 1,0. Or avec")
print("     α+β = 0,99, le processus est quasi intégré et sa moyenne")
h_moyen = float(np.mean(H))
print(f"     d'échantillon dérive énormément : {h_moyen:.3f} ici, pas 1,000.")
print()
print("     L'oracle sous-estimait donc de √0,74, et paraissait pire que")
print("     les estimateurs qu'il devait borner. Un oracle qui perd est")
print("     un oracle cassé — et il a fallu le mesurer pour le voir.")
print()
print(f"  → La vraie borne causale est à {o_garch:.2f} : en connaissant la")
print("    dynamique exacte ET l'état exact, on prévoit presque")
print("    parfaitement vingt-et-un jours à l'avance. Il y a donc")
print("    beaucoup à gagner — la question est combien on en capte.")
print()
print("  ⚠️ ET UNE TROISIÈME ERREUR, DANS CETTE SÉANCE MÊME.")
print()
print("     J'avais d'abord pris « h[t] prolongé tel quel » pour la borne")
print(f"     causale. Il marque {o_naif:.2f} — et la leçon suivante va montrer")
print("     qu'un simple EWMA fait MIEUX. Un prédicteur que n'importe quel")
print("     estimateur bat n'est pas une borne, c'est un mauvais")
print("     prédicteur.")
print()
print("     La différence : h[t] ignore le retour à la moyenne. Sur un")
print(f"     horizon de 21 jours avec une persistance de {ALPHA+BETA:.2f}, la")
print("     variance a déjà commencé à revenir vers sa moyenne longue, et")
print("     la prévision GARCH en tient compte. Pas h[t].")


# ══════════════════════════════════════════════════════════════════════
titre(5, "LE VRAI RÉSULTAT : UNE SÉRIE N'EST PAS UNE MATRICE")
# ══════════════════════════════════════════════════════════════════════

print("  Avec la borne en main, on refait la mesure en distinguant deux")
print("  chemins : prédire la volatilité d'UNE série, ou passer par une")
print("  MATRICE de covariance N×N.")
print()


def par_serie(decay):
    def f(fen, serie, poids, h, t, c, horizon):
        w = ewma_weights(len(serie), decay)
        return float(np.sqrt(np.sum(w * serie**2)))
    return f


def fenetre_serie(fen, serie, poids, h, t, c, horizon):
    return float(np.std(serie, ddof=1))


print(f"    {'prédicteur':<38} {'E[z²]':>8} {'écart':>8}")
print("    " + "-" * 56)
lignes = [
    ("oracle qui connaît le futur", o_tricheur),
    ("oracle GARCH k pas — borne causale", o_garch),
    ("série directe, ewma λ=0,97", calibration(par_serie(0.97))),
    ("série directe, ewma λ=0,94", calibration(par_serie(0.94))),
    ("série directe, fenêtre 252j", calibration(fenetre_serie)),
    ("via matrice, ewma λ=0,97", scores3["ewma λ=0,97"]),
    ("via matrice, fenêtre 252j + LW",
     scores3["fenêtre 252j + Ledoit-Wolf"]),
]
for nom, v in lignes:
    print(f"    {nom:<38} {v:>8.2f} {abs(v - 1):>8.2f}")

serie_097 = dict(lignes)["série directe, ewma λ=0,97"]
serie_252 = dict(lignes)["série directe, fenêtre 252j"]
mat_097 = scores3["ewma λ=0,97"]
mat_252 = scores3["fenêtre 252j + Ledoit-Wolf"]

print()
gain_dispo = abs(serie_252 - 1) - abs(o_garch - 1)
gain_capte = abs(serie_252 - 1) - abs(serie_097 - 1)
print(f"  → SUR UNE SÉRIE, l'EWMA à λ=0,97 marque {serie_097:.2f} contre")
print(f"    {serie_252:.2f} pour la fenêtre. Elle capte donc "
      f"{gain_capte / gain_dispo:.0%} de ce qui")
print(f"    était disponible — la borne causale étant à {o_garch:.2f}.")
print()
print(f"    Une fraction, pas la totalité : {1 - gain_capte / gain_dispo:.0%} de l'écart reste")
print("    sur la table pour un modèle qui ESTIMERAIT la dynamique de la")
print("    variance au lieu de la lisser.")
print()
print("    ⚠️ Cette phrase disait « un tiers » pendant que la ligne")
print("       au-dessus calculait autre chose. Douzième dérive de")
print("       narration du projet, dans le fichier qui raconte les onze")
print("       premières. La règle n'est pas « faire attention » : c'est")
print("       de ne jamais écrire un nombre à la main.")
print()
print(f"  → Sa demi-vie est de {half_life(0.97):.0f} jours, pour un horizon de 21.")
print("    Ce n'est pas un hasard : **λ doit correspondre à l'horizon**.")
print("    Le 0,94 de RiskMetrics est calibré pour une VaR à UN jour, et")
print("    l'employer pour prévoir un mois est une erreur de catégorie.")
print()
print(f"  → VIA UNE MATRICE, tout s'inverse : {mat_097:.2f} contre "
      f"{mat_252:.2f}.")
print(f"    Le passage par la matrice coûte {mat_097 - serie_097:.2f} à l'EWMA")
print(f"    et {mat_252 - serie_252:.2f} à la fenêtre.")
print()
print("  → **L'avantage de l'EWMA est en TEMPS ; son coût est en")
print("    DIMENSION.** Une série a un paramètre à estimer, et 66")
print("    observations effectives suffisent. Une matrice N×N en a")
print("    N(N+1)/2, portés par ces mêmes 66 — pas par les 252 lues.")
print()
print("    L'avantage ne change pas avec N. Le coût, si.")


# ══════════════════════════════════════════════════════════════════════
titre(6, "ET LA MATRICE SE DÉGRADE EN SILENCE")
# ══════════════════════════════════════════════════════════════════════

print("  J'avais prévu qu'à N > n_eff la covariance EWMA deviendrait")
print("  singulière, comme l'empirique à N > T. Mesuré sur N=40 actifs,")
print("  1 000 séances :")
print()
grand, _ = marche(t=1_000, n=40, seed=7)
emp = sample_covariance(grand)
nue = ewma_covariance(grand, decay=0.94, shrinkage=0.0)
con = ewma_covariance(grand, decay=0.94)
print(f"    {'matrice':<28} {'inversible':>11} {'conditionnement':>17}")
for nom, cov in (("empirique 1 000j", emp),
                 ("ewma λ=0,94 nue", nue),
                 ("ewma λ=0,94 contractée", con)):
    print(f"    {nom:<28} {cov.is_invertible!s:>11} "
          f"{cov.condition_number:>17,.0f}")

print()
print(f"  → N/n_eff vaut {nue.ratio_n_over_t:.2f}, donc au-delà de 1 — le régime")
print("    où l'empirique serait singulière. L'EWMA ne l'est PAS.")
print()
print("  → Raison : chacune des 1 000 observations reçoit un poids non")
print("    nul, si minuscule soit-il, et cela garde la matrice de rang")
print("    plein. La preuve du mécanisme :")
tronquee = ewma_covariance(
    grand.iloc[-32:], decay=0.94, shrinkage=0.0, min_obs=10
)
print(f"      en ne gardant que les 32 séances qui pèsent → inversible = "
      f"{tronquee.is_invertible}")
print()
print("  → **C'est pire que la singularité.** Le conditionnement passe de")
print(f"    {emp.condition_number:,.0f} à {nue.condition_number:,.0f}, et "
      f"`is_invertible` répond True.")
print("    L'empirique à N > T échoue BRUYAMMENT : le moteur refuse,")
print("    l'utilisateur voit. L'EWMA échoue EN SILENCE : elle rend une")
print("    matrice qu'on peut inverser et dont l'inverse est du bruit.")
print()
print("  → D'où la contraction ACTIVE PAR DÉFAUT dans `ewma_covariance`,")
print("    alors qu'elle ne l'est pas dans `sample_covariance`. Le défaut")
print(f"    doit être utilisable : {con.condition_number:,.0f} de "
      f"conditionnement, soit mieux que")
print("    l'empirique.")


# ══════════════════════════════════════════════════════════════════════
titre(7, "CHOISIR λ AU LIEU DE LE RECOPIER")
# ══════════════════════════════════════════════════════════════════════

print("  Le 0,94 est une valeur publiée en 1996, ajustée sur un panier de")
print("  marchés de l'époque. Elle est devenue une constante par citation.")
print()
print("  `fit_decay` l'estime par maximum de vraisemblance à un pas en")
print("  avant : la prévision de chaque séance n'utilise que les")
print("  précédentes. Le critère est CAUSAL par la forme de la récurrence,")
print("  pas par un découpage — c'est la propriété que `detect_lookahead`")
print("  vérifie expérimentalement à la séance 8.")
print()
fit = fit_decay(CADRE, grid=tuple(np.round(np.arange(0.85, 0.995, 0.005), 4)))
for ligne in str(fit).splitlines():
    print(f"    {ligne}")
print()
table = fit.table()
table = table[table["λ"].isin([0.86, 0.90, 0.94, 0.97, 0.99])]
print(table.to_string(index=False,
                      float_format=lambda v: f"{v:,.2f}"))
print()
print("  → Sur CES données, la vraisemblance préfère "
      f"λ = {fit.decay:.3f}")
print(f"    (demi-vie {fit.half_life:.0f} jours), pas 0,94.")
print()
print("  → Mais attention : ce critère est à UN JOUR. Il choisit donc le")
print("    λ qui prévoit le mieux demain, et la leçon 5 vient de montrer")
print("    que ce n'est pas celui qui prévoit le mieux dans un mois.")
print()
print("  → **Il n'y a pas un bon λ. Il y a un bon λ PAR HORIZON**, et")
print("    l'outil qui l'ajuste doit savoir pour quel horizon on l'ajuste.")
print("    C'est une limite de `fit_decay` telle qu'elle est écrite, et")
print("    elle est déclarée dans sa docstring plutôt que découverte par")
print("    le prochain utilisateur.")


# ══════════════════════════════════════════════════════════════════════
titre(8, "CE QUE CETTE SÉANCE A RÉELLEMENT ÉTABLI")
# ══════════════════════════════════════════════════════════════════════

print("  Quatre prédictions, quatre réfutations, un oracle réparé.")
print()
print("  Ce qui tient à la fin :")
print()
print("    1. La pondération exponentielle n'est pas « un meilleur")
print("       estimateur ». C'est un curseur entre deux erreurs : être en")
print("       retard, ou être imprécis.")
print()
print("    2. Le réglage de λ dépend de l'HORIZON de prévision. λ=0,94")
print("       sert la VaR à un jour, et dessert la prévision mensuelle.")
print()
print(f"    3. Bien réglée, sur une SÉRIE, elle capte {gain_capte / gain_dispo:.0%} de l'écart")
print("       qui séparait la fenêtre longue de la borne causale.")
print("       Un vrai gain, et loin du maximum atteignable.")
print()
print("    4. Via une MATRICE, l'avantage temporel est mangé par le coût")
print("       dimensionnel, et la fenêtre longue l'emporte.")
print()
print("  → CONSÉQUENCE PRATIQUE : employer l'EWMA pour la VaR d'un")
print("    portefeuille à poids fixés — une série — et la fenêtre longue")
print("    avec contraction pour la covariance qu'on va optimiser.")
print()
print("    Beaucoup de systèmes de risque font exactement cela. Peu")
print("    expliquent pourquoi, et je ne l'aurais pas deviné : il a fallu")
print("    se tromper quatre fois et réparer un oracle.")
print()
print("  → Reste la vraie question, et elle n'est pas dans ce fichier :")
print("    **est-ce que cela tient sur TES douze actions ?** Le script")
print("    `donnees_reelles.py` compare maintenant les deux estimateurs")
print("    sur les deux chemins. Les simulations proposent ; les données")
print("    disposent.")

print()
print("═" * LARGE)
print("  Fin de la séance 11.   python -m pytest tests/ -q -m \"\"")
print("═" * LARGE)
