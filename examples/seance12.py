"""
Séance 12 — Validation de modèles : juger le juge.

Les onze séances précédentes ont construit des estimateurs. Celle-ci
construit l'instrument qui les juge, puis découvre que l'instrument
standard de la profession est faux — et de combien.

Trois résultats, dans l'ordre où ils sont apparus :

  1. Un oracle cassé, le quatrième de ce projet. Ma première
     construction d'un « modèle correct par construction » regardait
     discrètement le futur, et m'aurait fait publier que la
     statistique de biais est biaisée.

  2. L'intervalle de confiance que publie la profession a, sur un
     marché réaliste, une couverture très inférieure à son niveau
     nominal. Il déclare faux des modèles corrects.

  3. L'inventaire de modèles : les quatre estimateurs de covariance du
     moteur, passés dans le même protocole, sur trois univers. Le
     classement change d'un univers à l'autre — ce qui est la raison
     pour laquelle E-23 exige une revalidation PÉRIODIQUE et non une
     validation.

Aucun nombre de ce fichier n'est écrit à la main. Douze dérives de
narration ont été documentées dans ce projet ; la règle qui les arrête
n'est pas la vigilance, c'est l'interdiction d'écrire un nombre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nkrisk.validation import (
    EwmaModel,
    FactorCovariance,
    LedoitWolfModel,
    SampleModel,
    bias_statistic,
    build_inventory,
)

SEP = "=" * 70
rng_global = np.random.default_rng(12)

# Paramètres du générateur GARCH(1,1) de moyenne longue 1.
ALPHA, BETA, OMEGA = 0.09, 0.90, 0.01
PERSISTANCE = ALPHA + BETA
NU = 5  # degrés de liberté de la loi de Student pour les queues épaisses

N_SIM = 1500
NOMINAL = 0.95  # niveau de confiance nominal de tous les intervalles
T_SIM = 1008  # quatre ans de séances
STEP = 21  # réestimation mensuelle


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 1 — LA STATISTIQUE DE BIAIS SUR UNE VÉRITÉ CONNUE")
print(SEP)
print()
print("On fabrique des rendements de volatilité FIXÉE, puis on donne au")
print("test des prévisions délibérément fausses d'un facteur connu.")
print("Si l'instrument est juste, il doit retrouver ce facteur.")
print()

VRAIE_VOL = 0.012
r_test = rng_global.normal(0.0, VRAIE_VOL, 2000)

print(f"  {'prévision annoncée':<34}{'biais mesuré':>14}{'attendu':>10}   verdict")
print("  " + "-" * 68)
for facteur in (1.0, 0.80, 1.25, 0.95):
    annonce = np.full(r_test.size, VRAIE_VOL * facteur)
    test = bias_statistic(r_test, annonce)
    libelle = f"{facteur:.0%} de la vraie volatilité"
    print(
        f"  {libelle:<34}{test.bias.value:>14.3f}{1.0 / facteur:>10.3f}"
        f"   {test.verdict}"
    )

print()
print("  → L'instrument retrouve l'inverse du facteur d'erreur, ce qui est")
print("    bien ce que mesure réalisé/prédit. Il est donc juste.")
print()
test_exact = bias_statistic(r_test, np.full(r_test.size, VRAIE_VOL))
print(f"  Sur la prévision exacte : {test_exact}")
print()
print("  ⚠️ LE SENS DU DANGER. `backtest.calibration()` de la séance 8")
print("     est prédit/réalisé ; la statistique de biais est")
print("     réalisé/prédit. Les deux sont environ réciproques, et un")
print("     lecteur pressé lira le mauvais sens. D'où `direction`, qui")
print("     écrit la conclusion en mots :")
print()
sous = bias_statistic(r_test, np.full(r_test.size, VRAIE_VOL * 0.8))
print(f"     {sous.direction}")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 2 — L'ORACLE CASSÉ (le quatrième de ce projet)")
print(SEP)
print()
print("Pour mesurer si l'intervalle de confiance est honnête, il faut un")
print("modèle CORRECT PAR CONSTRUCTION : on connaît la vraie variance")
print("conditionnelle, on la donne au test, et on compte combien de fois")
print("l'intervalle contient 1. Si la couverture n'est pas le niveau")
print("nominal, l'intervalle est faux.")
print()
print("Tout repose donc sur la correction de l'oracle. Voici ma première")
print("version, et pourquoi elle était fausse.")
print()


def trajectoire_garch(graine: int, nu: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Un GARCH(1,1) de variance longue 1, et sa vraie variance conditionnelle.

    Retourne (rendements standardisés, variances conditionnelles h).
    """
    g = np.random.default_rng(graine)
    h = np.empty(T_SIM)
    e = np.empty(T_SIM)
    h[0] = 1.0
    echelle = 1.0 if nu is None else np.sqrt((nu - 2) / nu)
    for t in range(T_SIM):
        u = g.normal() if nu is None else g.standard_t(nu) * echelle
        e[t] = u * np.sqrt(h[t])
        if t + 1 < T_SIM:
            h[t + 1] = OMEGA + ALPHA * e[t] ** 2 + BETA * h[t]
    return e, h


def oracle_anticipant(h: np.ndarray, debut: int, k: int) -> float:
    """
    MA PREMIÈRE VERSION, ET ELLE EST FAUSSE.

    « La prévision correcte pour un bloc, c'est la racine de la vraie
    variance moyenne du bloc. » L'énoncé paraît imparable : la moyenne
    des h_t sur le bloc divisée par elle-même vaut exactement 1.
    """
    return float(np.sqrt(h[debut : debut + k].mean()))


def oracle_causal(h: np.ndarray, debut: int, k: int) -> float:
    """
    LA VERSION CORRECTE : la prévision GARCH à k pas, connue en `debut`.

        E[h_{t+j} | F_t] = 1 + (α+β)^j · (h_t − 1)

    moyennée sur les k pas du bloc. N'utilise que h[debut].
    """
    poids = float(np.mean([PERSISTANCE**j for j in range(1, k + 1)]))
    return float(np.sqrt(1.0 + poids * (h[debut] - 1.0)))


def serie_prevue(h: np.ndarray, oracle) -> np.ndarray:
    s = np.empty(T_SIM)
    for d in range(0, T_SIM, STEP):
        f = min(d + STEP, T_SIM)
        s[d:f] = oracle(h, d, f - d)
    return s


def moment_moyen(oracle, nu: int | None, n_sim: int) -> float:
    """E[m2] sur n_sim trajectoires. Doit valoir 1 pour un oracle correct."""
    total = 0.0
    for i in range(n_sim):
        e, h = trajectoire_garch(3000 + i, nu)
        test = bias_statistic(e, serie_prevue(h, oracle), block_size=STEP)
        total += test.bias.value**2
    return total / n_sim


N_VERIF = 400
m2_anticipant = moment_moyen(oracle_anticipant, None, N_VERIF)
m2_causal = moment_moyen(oracle_causal, None, N_VERIF)

print(f"  Vérification de l'oracle sur {N_VERIF} trajectoires.")
print("  Un oracle correct doit donner E[m2] = 1,000 exactement.")
print()
print(f"    {'oracle « variance moyenne du bloc »':<40}{m2_anticipant:>10.4f}")
print(f"    {'oracle GARCH à k pas (causal)':<40}{m2_causal:>10.4f}")
print()
derive = (1.0 - m2_anticipant) * 100.0
print(f"  → Le premier manque la cible de {derive:.1f} %. Il n'est donc pas")
print("    un oracle : il est un modèle légèrement anticipant.")
print()
print("  LE MÉCANISME, et il est instructif.")
print()
print("    Dans un GARCH, h_{t+1} dépend de e_t. La moyenne des h sur le")
print("    bloc est donc en partie CAUSÉE par les rendements qu'elle sert")
print("    à normaliser. Un gros e_t gonfle les h qui le suivent, donc")
print("    gonfle le dénominateur, donc rapetisse le rapport. Numérateur")
print("    et dénominateur sont négativement dépendants, et")
print("    E[X/Y] < E[X]/E[Y].")
print()
print("    Mon raisonnement conditionnait sur « toute la trajectoire de h »")
print("    comme si h était exogène. Il ne l'est pas : h est fabriqué par")
print("    les e. L'identité que je croyais exacte ne tient que pour un")
print("    processus à volatilité exogène, ce qu'un GARCH n'est pas.")
print()
print("  CE QUE J'AURAIS PUBLIÉ SANS CETTE VÉRIFICATION :")
print(f"    « la statistique de biais sous-estime de {derive:.1f} % quand la")
print("    volatilité est persistante » — un résultat faux, sur un défaut")
print("    qui était dans mon harnais de mesure et non dans l'instrument.")
print()
print("  C'est le QUATRIÈME oracle cassé de ce projet (séances 6, 11 ×2,")
print("  et celui-ci). La leçon ne change pas : un oracle qui n'est pas")
print("  vérifié n'est pas un oracle, c'est une hypothèse déguisée en")
print("  référence.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 3 — CE QUE VAUT L'INTERVALLE DE LA PROFESSION")
print(SEP)
print()
print("La formule publiée est : le biais est acceptable si")
print()
print("      |b − 1| < 1,96 / √(2T)")
print()
print("Elle est exacte sous deux hypothèses : les z sont indépendants, et")
print("ils sont gaussiens. On va retirer ces hypothèses une par une et")
print(f"mesurer la couverture réelle d'un intervalle nominalement à {NOMINAL:.0%}.")
print()
print("Dans les quatre régimes, LE MODÈLE A RAISON PAR CONSTRUCTION. Tout")
print("rejet est donc un faux positif.")
print()


def couverture(tirage, block_size: int) -> tuple[float, float, float]:
    """Couverture de l'intervalle robuste, du naïf, et vrai écart-type de b."""
    dans_robuste = dans_naif = 0
    valeurs = np.empty(N_SIM)
    for i in range(N_SIM):
        r, s = tirage(i)
        test = bias_statistic(r, s, block_size=block_size, confidence=NOMINAL)
        valeurs[i] = test.bias.value
        if test.bias.ci_low <= 1.0 <= test.bias.ci_high:
            dans_robuste += 1
        if test.naive_ci[0] <= 1.0 <= test.naive_ci[1]:
            dans_naif += 1
    return dans_robuste / N_SIM, dans_naif / N_SIM, float(valeurs.std(ddof=1))


def tirage_iid(i: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(1000 + i)
    return g.normal(0.0, 1.0, T_SIM), np.ones(T_SIM)


def tirage_queues(i: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(2000 + i)
    x = g.standard_t(NU, T_SIM) * np.sqrt((NU - 2) / NU)
    return x, np.ones(T_SIM)


def tirage_garch(i: int) -> tuple[np.ndarray, np.ndarray]:
    e, h = trajectoire_garch(3000 + i, None)
    return e, serie_prevue(h, oracle_causal)


def tirage_garch_queues(i: int) -> tuple[np.ndarray, np.ndarray]:
    e, h = trajectoire_garch(4000 + i, NU)
    return e, serie_prevue(h, oracle_causal)


regimes = [
    ("z indépendants et gaussiens", tirage_iid, 1),
    (f"queues épaisses — Student t({NU})", tirage_queues, 1),
    ("volatilité persistante (GARCH)", tirage_garch, STEP),
    (f"GARCH + queues t({NU}) — un marché", tirage_garch_queues, STEP),
]

sd_naif_theorique = 1.0 / np.sqrt(2.0 * T_SIM)

print(f"  {N_SIM} simulations, T = {T_SIM}, bloc = {STEP}.")
print()
print(
    f"  {'régime':<36}{'par blocs':>11}{'classique':>11}{'sd réel/naïf':>14}"
)
print("  " + "-" * 72)

resultats = []
for libelle, tirage, bloc in regimes:
    rob, naif, sd_reel = couverture(tirage, bloc)
    resultats.append((libelle, rob, naif, sd_reel))
    print(
        f"  {libelle:<36}{rob:>10.1%}{naif:>11.1%}"
        f"{sd_reel / sd_naif_theorique:>14.2f}"
    )

print()
pire = min(resultats, key=lambda x: x[2])
print(f"  → Dans le régime « {pire[0]} » — c'est-à-dire sur")
print(f"    n'importe quel marché réel — l'intervalle classique à {NOMINAL:.0%}")
print(f"    a une couverture de {pire[2]:.1%}.")
print()
faux_positifs = 1.0 - pire[2]
print("    Autrement dit : il déclare BIAISÉ un modèle parfaitement")
print(f"    correct dans {faux_positifs:.0%} des cas. Un validateur qui")
print("    applique cette formule retire des modèles sains, et il le fait")
print("    avec la conviction d'être rigoureux.")
print()
print(f"    L'intervalle par blocs monte à {pire[1]:.1%} dans le même régime.")
print(f"    Mieux, mais pas {NOMINAL:.0%} — et il faut le dire :")
print()
print(f"      • il n'y a que {T_SIM // STEP} blocs, donc la dispersion entre")
print("        blocs est elle-même mal estimée ;")
print("      • le moment d'ordre quatre d'une série à queues épaisses est")
print("        la grandeur la plus difficile à estimer de la statistique.")
print()
print("    La correction réduit l'erreur d'un facteur important sans la")
print("    supprimer. C'est un progrès mesuré, pas une solution.")
print()
print("  POURQUOI LA FORMULE CLASSIQUE ÉCHOUE, EN UNE PHRASE PAR CAUSE :")
print()
_, rob_q, naif_q, sd_q = resultats[1]
print(f"    Queues épaisses — l'écart-type réel de b vaut {sd_q / sd_naif_theorique:.2f} fois")
print("    la formule gaussienne, parce que l'incertitude sur un moment")
print("    d'ordre deux dépend du moment d'ordre quatre : √((κ−1)/T)/2 au")
print(f"    lieu de √(2/T)/2. Avec κ = {3 + 6 / (NU - 4):.0f} pour t({NU}), le rapport")
print(f"    théorique est √(({3 + 6 / (NU - 4):.0f}−1)/2) = {np.sqrt((3 + 6 / (NU - 4) - 1) / 2):.2f}.")
print()
_, rob_g, naif_g, sd_g = resultats[2]
print(f"    Dépendance — l'écart-type réel vaut {sd_g / sd_naif_theorique:.2f} fois la formule,")
print(f"    parce que les {STEP} observations qui suivent une prévision")
print("    partagent la même erreur de prévision. Ce ne sont pas")
print(f"    {STEP} informations, et la formule les compte comme telles.")
print()
print("    C'est le NOMBRE EFFECTIF D'OBSERVATIONS de la séance 11, revenu")
print("    dans un endroit entièrement différent. La pondération")
print("    exponentielle l'avait introduit pour mesurer sa propre mémoire ;")
print("    il gouverne ici la largeur d'un intervalle de confiance.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 4 — L'INVENTAIRE DE MODÈLES")
print(SEP)
print()
print("Les quatre estimateurs de covariance du moteur, passés dans le même")
print("protocole : mêmes données, même fenêtre, même portefeuille de test")
print("équipondéré, même intervalle. C'est la seule façon de les comparer")
print("sans que le protocole favorise l'un d'eux.")
print()


def univers_facteurs(
    n_actifs: int, n_facteurs: int, graine: int, garch: bool = False
) -> pd.DataFrame:
    """Rendements à structure de facteurs connue, volatilité fixe ou GARCH."""
    g = np.random.default_rng(graine)
    expositions = g.normal(0.0, 0.5, (n_actifs, n_facteurs))
    facteurs = g.normal(0.0, 1.0, (T_SIM, n_facteurs))
    specifique = g.normal(0.0, 1.0, (T_SIM, n_actifs))
    brut = facteurs @ expositions.T + specifique

    if garch:
        # Une volatilité de marché commune qui bouge : le régime où un
        # estimateur réactif devrait enfin servir à quelque chose.
        h = np.empty(T_SIM)
        h[0] = 1.0
        choc = g.normal(0.0, 1.0, T_SIM)
        for t in range(T_SIM - 1):
            h[t + 1] = OMEGA + ALPHA * (choc[t] ** 2) * h[t] + BETA * h[t]
        brut = brut * np.sqrt(h)[:, None]

    dates = pd.bdate_range("2018-01-02", periods=T_SIM)
    colonnes = [f"A{i:02d}" for i in range(n_actifs)]
    return pd.DataFrame(brut * 0.01, index=dates, columns=colonnes)


modeles = (SampleModel(), LedoitWolfModel(), EwmaModel(), FactorCovariance())

univers = [
    ("calme — 20 actifs, volatilité fixe", univers_facteurs(20, 3, 7)),
    ("encombré — 120 actifs, volatilité fixe", univers_facteurs(120, 5, 8)),
    ("agité — 20 actifs, volatilité GARCH", univers_facteurs(20, 3, 9, garch=True)),
]

inventaires = []
for nom, donnees in univers:
    inv = build_inventory(donnees, modeles, universe=nom)
    inventaires.append((nom, inv))
    print(inv)
    print()

print("  PREMIÈRE SURPRISE, ET C'EST MOI QUI AVAIS TORT.")
print()
nom_encombre, inv_encombre = inventaires[1]
carte_fixe = next(c for c in inv_encombre.cards if c.model == "empirique")
print(f"    Dans l'univers encombré (N/T = {carte_fixe.ratio_n_over_t:.2f}, conditionnement")
print(f"    médian {carte_fixe.condition_number:.4g}), la covariance empirique obtient un")
print(f"    biais de {carte_fixe.bias_value:.3f} et le verdict {carte_fixe.verdict}.")
print()
print("    J'avais écrit, avant de lancer la mesure, que ce serait le")
print("    régime où l'empirique s'effondrerait : c'est dans cette zone")
print("    que la séance 2 avait trouvé un levier et un conditionnement")
print("    catastrophiques. La mesure dit le contraire, et elle a raison.")
print()
print("    (Les chiffres de la séance 2 ne sont pas recopiés ici. Les")
print("     recopier serait la treizième dérive de narration de ce")
print("     projet : un nombre juste aujourd'hui, que rien ne recalcule,")
print("     est un nombre qui dérivera. `examples/seance2.py` les produit.)")
print()
print("    POURQUOI. La prévision d'un portefeuille FIXE est wᵀΣ̂w : une")
print("    combinaison linéaire des entrées de la matrice. La covariance")
print("    empirique est sans biais entrée par entrée, donc cette")
print("    combinaison est sans biais aussi — et elle le reste quel que")
print("    soit le conditionnement. Ce qui amplifie le bruit, c'est")
print("    l'INVERSION. Un portefeuille fixe n'inverse rien.")
print()
print("    Autrement dit : ce protocole ne peut pas voir le défaut. Pas")
print("    « ne le voit pas sur ces données » — ne PEUT pas, par")
print("    construction. Et la limite était déjà écrite dans la fiche,")
print("    que j'avais produite sans la lire.")
print()

print(SEP)
print("PARTIE 4b — LE MÊME MODÈLE, DEUX PROTOCOLES, DEUX VERDICTS")
print(SEP)
print()
print("On rejoue l'univers encombré en laissant chaque modèle choisir son")
print("propre portefeuille de variance minimale, w = Σ̂⁻¹1 / (1ᵀΣ̂⁻¹1).")
print("C'est la question du vrai utilisateur : « si je suis les conseils")
print("de ce modèle, me dit-il la vérité sur le risque qu'il me fait")
print("prendre ? » Et cette fois, on inverse.")
print()

inv_modele = build_inventory(
    univers[1][1],
    modeles,
    universe=f"{nom_encombre} — portefeuille DU MODÈLE",
    portfolio="model",
)
print(inv_modele)
print()

carte_modele = next(c for c in inv_modele.cards if c.model == "empirique")
print("  LE MÊME ESTIMATEUR, LES MÊMES DONNÉES, LA MÊME FENÊTRE :")
print()
print(f"    portefeuille fixe        biais {carte_fixe.bias_value:.3f}  → {carte_fixe.verdict}")
print(f"    portefeuille du modèle   biais {carte_modele.bias_value:.3f}  → {carte_modele.verdict}")
print()
sous_estimation = (carte_modele.bias_value - 1.0) * 100.0
print("    Sur le portefeuille qu'il recommande lui-même, l'empirique")
print(f"    sous-estime le risque de {sous_estimation:.0f} %.")
print()
print("  → UN ANGLE MORT EST UNE PROPRIÉTÉ DU PROTOCOLE DE VALIDATION,")
print("    PAS DU MODÈLE VALIDÉ.")
print()
print("    C'est la leçon la plus transposable de la séance, et elle ne")
print("    parle pas de finance. Un rapport de validation qui ne dit pas")
print("    ce qu'il ne peut pas voir est pire qu'absent : il rassure.")
print()
print("    Et l'ordre du classement, sur le protocole qui inverse :")
for c in inv_modele.cards:
    if c.bias is not None:
        print(f"      {c.model:<32}{c.bias_value:>8.3f}   {c.verdict}")
print()
print("    C'est l'ordre des séances 2 et 7, retrouvé par une mesure")
print("    entièrement différente — non plus le levier d'un portefeuille")
print("    ni sa volatilité réalisée, mais l'honnêteté d'une prévision.")
print("    Trois instruments indépendants qui concordent, c'est ce qui")
print("    distingue un résultat d'une coïncidence.")
print()

print("  DEUXIÈME SURPRISE — LÀ OÙ LA PONDÉRATION EXPONENTIELLE GAGNE.")
print()
nom_agite, inv_agite = inventaires[2]
ecart_max = max(
    abs(c.bias_value - 1.0) for c in inv_agite.cards if c.bias is not None
)
print(f"    Dans l'univers agité, les quatre modèles sont {inv_agite.cards[0].verdict} :")
print(f"    leurs biais sont tous à moins de {ecart_max:.1%} de 1. Sur le")
print("    NIVEAU, aucun ne se distingue.")
print()
print("    La colonne de dispersion, elle, les sépare :")
print()
for c in inv_agite.cards:
    if c.bias is not None:
        print(
            f"      {c.model:<32} niveau {c.bias_value:.3f}   "
            f"dispersion {c.bias.dispersion_ratio:.2f} × le hasard"
        )
print()
disp_ewma = next(
    c.bias.dispersion_ratio for c in inv_agite.cards if c.model.startswith("EWMA")
)
disp_autres = [
    c.bias.dispersion_ratio
    for c in inv_agite.cards
    if c.bias is not None and not c.model.startswith("EWMA")
]
print(f"    La pondération exponentielle est à {disp_ewma:.2f} là où les trois")
print(f"    autres sont entre {min(disp_autres):.2f} et {max(disp_autres):.2f}. Son avantage n'est pas")
print("    d'être mieux calibrée en moyenne : c'est de l'être plus")
print("    SOUVENT. Un biais agrégé ne pouvait pas le montrer.")
print()
print("    C'est « l'avantage est en temps » de la séance 11, retrouvé")
print("    par un troisième instrument. La séance 11 l'avait mesuré sur")
print("    une série et sur une matrice ; on le retrouve ici sur la")
print("    stabilité d'une calibration.")
print()

print("  CE QUE DIT L'ENSEMBLE DES TABLEAUX :")
print()
for nom, inv in inventaires:
    meilleur = inv.best()
    nom_meilleur = "aucun" if meilleur is None else meilleur.model
    print(
        f"    {nom:<42} → {nom_meilleur}"
        f"  ({inv.n_rejected} rejeté(s))"
    )
meilleur_modele = inv_modele.best()
print(
    f"    {'encombré, portefeuille du modèle':<42} → "
    f"{'aucun' if meilleur_modele is None else meilleur_modele.model}"
    f"  ({inv_modele.n_rejected} rejeté(s))"
)
print()
gagnants = {inv.best().model for _, inv in inventaires if inv.best() is not None}
if meilleur_modele is not None:
    gagnants.add(meilleur_modele.model)
print(f"  {len(gagnants)} gagnant(s) différent(s) sur {len(inventaires) + 1} configurations.")
print()
print("  → Un inventaire de modèles n'est pas un classement. C'est un")
print("    INSTANTANÉ : ce protocole, cette fenêtre, ce portefeuille, cet")
print("    univers. Changer l'un des quatre peut inverser le résultat, et")
print("    on vient de voir que changer le PROTOCOLE seul suffit à faire")
print("    passer un modèle de VALIDÉ à REJETÉ.")
print()
print("    C'est exactement pourquoi la ligne directrice E-23 exige une")
print("    revalidation PÉRIODIQUE et non une validation, et pourquoi elle")
print("    exige que la méthode de validation soit documentée. Un modèle")
print("    validé une fois est un modèle validé sur le régime d'alors, par")
print("    le protocole d'alors.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 5 — CE QUE LA SÉANCE 12 A AJOUTÉ AU MOTEUR")
print(SEP)
print()
print("  1. Une statistique de biais qui standardise période par période,")
print("     là où `calibration()` agrège. Les deux coïncident quand la")
print("     prévision est constante et divergent sinon — même inégalité")
print("     de Jensen qui avait cassé `calibration_diagnosis` à la")
print("     séance 9.")
print()
print("  2. Un intervalle de confiance par blocs, qui traite d'un coup les")
print("     queues épaisses et la dépendance, et qui se réduit à la")
print("     formule classique corrigée du kurtosis quand le bloc vaut 1.")
print()
print("  3. La mesure de ce que coûte la formule de la profession :")
for libelle, _, naif, _ in resultats:
    print(f"       {libelle:<38} couverture réelle {naif:>6.1%}")
print()
print("  4. Une mesure de STABILITÉ à côté de la mesure de NIVEAU. Un")
print("     biais de 1,00 dit que les erreurs se compensent sur la")
print("     fenêtre, pas qu'il n'y en a pas. C'est la colonne qui a")
print("     révélé où la pondération exponentielle gagne réellement.")
print()
print("  5. Deux protocoles — portefeuille fixe et portefeuille du")
print("     modèle — et la démonstration qu'ils rendent des verdicts")
print("     OPPOSÉS sur le même modèle et les mêmes données. Un angle")
print("     mort est une propriété du protocole, pas du modèle.")
print()
print("  6. Un cinquième usage du typage structurel, et le premier qui")
print("     serve à COMPARER des implémentations plutôt qu'à les étendre.")
print("     Les quatre estimateurs sont des fonctions libres de signatures")
print("     différentes ; quatre adaptateurs de six lignes les rendent")
print("     comparables sans toucher à aucun d'eux.")
print()
print("  7. Une fiche de modèle et un inventaire — la pièce que demande")
print("     E-23, produite par la mesure et non par la déclaration.")
print()
print("  8. Un verdict à trois niveaux plutôt que deux. Avec assez")
print("     d'observations, tout modèle devient statistiquement")
print("     distinguable d'un modèle parfait, puisque aucun ne l'est. Un")
print("     verdict binaire finit donc par tout rejeter. Il faut croiser")
print("     la significativité et la matérialité — c'est la")
print("     proportionnalité qu'exige E-23.")
print()
print("  ET LE RÉSULTAT QUE JE N'AVAIS PAS PRÉVU : le quatrième oracle")
print("  cassé de ce projet. J'ai construit un « modèle correct par")
print("  construction » qui regardait le futur sans que ça se voie, et")
print(f"  qui manquait sa cible de {derive:.1f} %. Sans la vérification de la")
print("  partie 2, la partie 3 aurait mesuré le défaut de mon harnais en")
print("  croyant mesurer celui de l'instrument.")
print()
print(SEP)
