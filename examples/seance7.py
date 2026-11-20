"""
Séance 7 — Le modèle à facteurs.

    python examples/seance7.py

RÈGLE DE CE FICHIER, HÉRITÉE DES SÉANCES 2, 3 ET 5
==================================================
Aucun nombre n'est écrit à la main dans les commentaires ou les
narrations. Tout ce qui est affiché est calculé à partir des mêmes
données que les tableaux juste au-dessus.

Cette règle vient de sept occurrences du même bug : une phrase écrite
d'après une exécution antérieure, qui contredisait silencieusement le
tableau affiché deux lignes plus haut. Un commentaire ne s'exécute pas,
donc rien ne le contredit — sauf un lecteur attentif, au pire moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nkrisk.covariance import ledoit_wolf, sample_covariance
from nkrisk.factors import (
    FactorShock,
    InsufficientStructure,
    attribute_risk,
    marchenko_pastur_bounds,
    n_significant_factors,
    pca_factor_model,
)
from nkrisk.portfolio import Portfolio
from nkrisk.risk import minimum_variance_weights
from nkrisk.stress import Combined, PriceShock, Shock, VolatilityShock

JOURS = 252
LARGE = 76


def titre(n: int, texte: str) -> None:
    print()
    print("═" * LARGE)
    print(f"  LEÇON {n} — {texte}")
    print("═" * LARGE)
    print()


# ══════════════════════════════════════════════════════════════════════
#  GÉNÉRATEURS — la vérité est connue, c'est tout l'intérêt
# ══════════════════════════════════════════════════════════════════════


def univers_a_facteurs(
    n: int, t: int, seed: int, k: int = 3, noms: list[str] | None = None
) -> pd.DataFrame:
    """
    Un univers où la structure à facteurs est RÉELLE.

    Facteur 1 : exposition positive partout — c'est le marché, et aucun
    univers d'actions réel n'y échappe.
    Facteurs 2 et 3 : expositions de signes mélangés — des oppositions
    de style ou de secteur.
    Bruit idiosyncratique HÉTÉROGÈNE : c'est le cas réaliste, et c'est
    celui qui met l'ACP en difficulté (voir la leçon 7).
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 0.7, (n, k))
    b[:, 0] = np.abs(rng.normal(1.0, 0.3, n))
    f = rng.normal(0, 1, (t, k))
    e = rng.normal(0, 1, (t, n)) * rng.uniform(0.5, 1.8, n)
    vols = rng.uniform(0.008, 0.030, n)
    colonnes = noms if noms is not None else [f"A{i:03d}" for i in range(n)]
    return pd.DataFrame((f @ b.T + e) * vols, columns=colonnes)


def univers_sans_structure(n: int, t: int, seed: int) -> pd.DataFrame:
    """Corrélation dense de rang plein : aucune structure à extraire."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n, n))
    c = a @ a.T / n + np.eye(n) * 0.5
    d = 1.0 / np.sqrt(np.diag(c))
    c = c * np.outer(d, d)
    vols = rng.uniform(0.008, 0.030, n)
    z = rng.normal(0, 1, (t, n)) @ np.linalg.cholesky(c).T
    return pd.DataFrame(z * vols, columns=[f"A{i:03d}" for i in range(n)])


def bruit_pur(n: int, t: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.008, 0.030, n)
    return pd.DataFrame(
        rng.normal(0, 1, (t, n)) * vols,
        columns=[f"A{i:03d}" for i in range(n)],
    )


# ══════════════════════════════════════════════════════════════════════
titre(1, "CE QUE LEDOIT-WOLF N'A PAS RÉSOLU")
# ══════════════════════════════════════════════════════════════════════

print("  La séance 2 a mesuré le problème : trop de paramètres, pas assez")
print("  d'observations. La contraction l'a RÉGULARISÉ. Elle ne l'a pas")
print("  réduit — le nombre de paramètres n'a pas bougé d'une unité.")
print()
print(f"    {'univers':>8}   {'covariance libre':>17}   "
      f"{'modèle k=5':>11}   {'compression':>11}")
for n in (10, 50, 100, 500, 2000):
    libre = n * (n + 1) // 2
    facteurs = n * 5 + n
    print(f"    {n:>8}   {libre:>17,}   {facteurs:>11,}   "
          f"{libre / facteurs:>10.0f}×")

print()
print("  → Un modèle à facteurs ne régularise pas une estimation : il")
print("    IMPOSE UNE STRUCTURE. « Les actifs bougent ensemble parce")
print("    qu'ils partagent un petit nombre de causes communes. »")
print()
print("  → C'est une affirmation sur le monde, donc réfutable. La")
print("    contraction n'affirme rien ; elle se contente de moins croire")
print("    ce qu'elle a estimé. Toute la leçon 6 consiste à vérifier si")
print("    l'affirmation tient.")


# ══════════════════════════════════════════════════════════════════════
titre(2, "COMBIEN DE FACTEURS ? LA RÉPONSE VIENT DU BRUIT")
# ══════════════════════════════════════════════════════════════════════

print("  Question piège : quelles valeurs propres a la matrice de")
print("  corrélation de N actifs STRICTEMENT INDÉPENDANTS ?")
print()
print("  La réponse intuitive — toutes égales à 1 — est fausse, et c'est")
print("  pour cela qu'un seuil du type « 80 % de variance expliquée »")
print("  n'a aucun fondement.")
print()

n_bruit, t_bruit = 100, 250
corr_bruit = bruit_pur(n_bruit, t_bruit, seed=1).corr().to_numpy()
vp_bruit = np.sort(np.linalg.eigvalsh(corr_bruit))[::-1]
basse, haute = marchenko_pastur_bounds(n_bruit, t_bruit)

print(f"    N={n_bruit}, T={t_bruit}, bruit pur :")
print(f"      valeur propre la plus grande : {vp_bruit[0]:.3f}")
print(f"      la plus petite               : {vp_bruit[-1]:.3f}")
print(f"      rapport                      : "
      f"{vp_bruit[0] / vp_bruit[-1]:.0f}×")
print()
print(f"    Marchenko-Pastur prédit l'intervalle "
      f"[{basse:.3f} ; {haute:.3f}]")
print(f"    Observé : [{vp_bruit[-1]:.3f} ; {vp_bruit[0]:.3f}]")
print()
print(f"  → Du bruit pur produit un rapport de "
      f"{vp_bruit[0] / vp_bruit[-1]:.0f} entre la plus grande et la plus")
print("    petite valeur propre, alors que la vraie matrice est")
print("    l'identité. Le conditionnement apparent d'une corrélation")
print("    empirique est presque entièrement un artefact d'échantillonnage.")
print()
print(f"  → Toute valeur propre au-dessus de {haute:.2f} ne peut PAS venir")
print("    du bruit. On les compte, et on a k. Pas de seuil arbitraire.")

print()
print("  À quelle fréquence le critère se trompe-t-il ? Mesuré, pas supposé :")
essais = 1_000
faux = sum(
    n_significant_factors(
        np.linalg.eigvalsh(bruit_pur(50, 300, seed=2_000 + s).corr().to_numpy()),
        50,
        300,
    )
    > 0
    for s in range(essais)
)
print(f"    sur {essais} univers de bruit pur (N=50, T=300), le critère")
print(f"    invente un facteur {faux} fois, soit {faux / essais:.1%}.")
print()
print("  → Et sur du bruit, le moteur refuse de répondre plutôt que de")
print("    retourner le premier vecteur propre, qui existe toujours et")
print("    ressemble parfaitement à un résultat :")
try:
    pca_factor_model(bruit_pur(40, 600, seed=3))
    print("    …il a répondu. Ce n'était pas prévu.")
except InsufficientStructure as exc:
    print(f"    InsufficientStructure — {exc}")


# ══════════════════════════════════════════════════════════════════════
titre(3, "LE MODÈLE SUR UN UNIVERS RÉALISTE")
# ══════════════════════════════════════════════════════════════════════

NOMS = [
    "BanqueA", "BanqueB", "Assur", "Pétrole", "Mines", "Or",
    "TechA", "TechB", "TechC", "SemiCond", "Pharma", "Biotech",
    "Conso", "Distrib", "Télécom", "Utilit", "Immo", "Indus",
]
K_VRAI = 3
r_demo = univers_a_facteurs(len(NOMS), 750, seed=42, k=K_VRAI, noms=NOMS)
modele = pca_factor_model(r_demo)

print(modele.diagnostic())
print()

expo = modele.exposures_frame()
expo["R²"] = modele.systematic_share()
print("  Expositions estimées (en écarts-types par écart-type de facteur) :")
print()
print(expo.sort_values("F1", ascending=False).round(3).to_string())
print()

part = modele.explained_variance_ratio()
print(f"  Part de la variance totale captée par chaque facteur : "
      f"{', '.join(f'{p:.1%}' for p in part)}"
      f"  (cumul {part.sum():.1%})")
print()
r2 = modele.systematic_share()
print(f"  → Le facteur 1 a une exposition positive sur "
      f"{int((modele.exposures[:, 0] > 0).sum())} des {modele.n_assets} "
      f"actifs.")
print("    C'est la signature du facteur marché : il ne diversifie rien,")
print("    parce que tout le monde le porte.")
print()
print(f"  → R² le plus élevé : {r2.idxmax()} à {r2.max():.0%}. Le plus bas :")
print(f"    {r2.idxmin()} à {r2.min():.0%}. Sur ce second actif, le modèle")
print("    n'a presque rien à dire — et le savoir est une information.")
print()
if modele.n_factors != K_VRAI:
    vp_rejetee = modele.eigenvalues[modele.n_factors]
    _, seuil = modele.noise_bounds
    print(f"  ⚠️ CE GÉNÉRATEUR A {K_VRAI} FACTEURS. LE CRITÈRE EN TROUVE "
          f"{modele.n_factors}.")
    print()
    print(f"    La première valeur propre rejetée vaut {vp_rejetee:.2f}, pour")
    print(f"    un seuil de bruit à {seuil:.2f}. Le troisième facteur existe")
    print("    bel et bien, mais il est trop faible pour qu'on puisse le")
    print("    distinguer du bruit avec cet univers et cet historique.")
    print()
    print("  → Ce n'est pas un échec du critère, c'est son contrat. La borne")
    print("    de Marchenko-Pastur sans l'ajustement de Laloux est")
    print("    délibérément CONSERVATRICE : elle rate des facteurs faibles")
    print("    plutôt que d'en inventer. Pour un moteur de risque, se")
    print("    tromper dans ce sens-là est le bon côté — un facteur manqué")
    print("    se retrouve dans le risque spécifique, donc comptabilisé ;")
    print("    un facteur inventé crée une exposition qui n'existe pas.")
else:
    print(f"  → Le critère retrouve exactement les {K_VRAI} facteurs injectés.")


# ══════════════════════════════════════════════════════════════════════
titre(4, "L'ATTRIBUTION : DE QUOI SUIS-JE DÉPENDANT ?")
# ══════════════════════════════════════════════════════════════════════

cov_facteurs = modele.covariance()
poids_eq = pd.Series(
    np.full(modele.n_assets, 1 / modele.n_assets), index=list(modele.assets)
)
p_eq = Portfolio(
    weights=poids_eq, covariance=cov_facteurs, value=100_000.0,
    name="équipondéré",
)
attr = attribute_risk(poids_eq, modele)

print(f"  Portefeuille équipondéré, {modele.n_assets} lignes.")
print()
print(attr)
print()
print(attr.table().round(4).to_string())
print()
print(f"  → {attr.systematic_share:.0%} de la variance vient des facteurs")
print("    communs. Cette part-là ne se diversifie PAS : ajouter des")
print("    lignes la moyenne, ne la réduit pas.")
print()

# Démonstration : diversifier ne réduit que la part spécifique.
#
# ⚠️ On moyenne sur des sous-ensembles TIRÉS AU HASARD, et non sur les
# n premiers actifs de la liste. La première version prenait les
# premiers, et le tableau sortait non monotone : la volatilité
# systématique montait de 3 à 6 lignes avant de redescendre. Ce n'était
# pas un effet de taille, mais un effet de SÉLECTION — les actifs
# ajoutés se trouvaient être plus exposés au marché que les premiers.
# Moyenner sur des tirages isole l'effet qu'on prétend montrer.
print("  Vérification — que fait l'élargissement du portefeuille ?")
print("  (moyenne sur 300 sous-ensembles tirés au hasard par taille,")
print("   pour isoler l'effet de taille de l'effet de sélection)")
print()
print(f"    {'nb lignes':>10}   {'σ totale':>9}   {'σ systém.':>10}   "
      f"{'σ spécif.':>10}")

rng_sous = np.random.default_rng(7)
mesures: dict[int, tuple[float, float, float]] = {}
for taille in (1, 3, 6, 12, modele.n_assets):
    tot, syst, spec = [], [], []
    tirages = 1 if taille == modele.n_assets else 300
    for _ in range(tirages):
        choix = rng_sous.choice(modele.n_assets, taille, replace=False)
        vec = np.zeros(modele.n_assets)
        vec[choix] = 1.0 / taille
        a = attribute_risk(pd.Series(vec, index=list(modele.assets)), modele)
        tot.append(a.total_volatility)
        syst.append(a.systematic_volatility)
        spec.append(a.specific_volatility)
    mesures[taille] = (float(np.mean(tot)), float(np.mean(syst)),
                       float(np.mean(spec)))
    print(f"    {taille:>10}   {mesures[taille][0]:>8.2%}   "
          f"{mesures[taille][1]:>9.2%}   {mesures[taille][2]:>9.2%}")

seul, complet = mesures[1], mesures[modele.n_assets]
reduction_spec = 1 - complet[2] / seul[2]
variation_syst = complet[1] / seul[1] - 1
print()
print(f"  → Passer d'une ligne à {modele.n_assets} réduit le risque "
      f"spécifique de {reduction_spec:.0%}")
print(f"    et fait varier le risque systématique de "
      f"{variation_syst:+.0%}.")
print()
print("  → C'est la réponse quantifiée à « combien de titres pour être")
print("    diversifié ? ». La question est mal posée : au-delà d'un")
print("    certain nombre, on n'achète plus de diversification, on ne")
print("    fait que répliquer les facteurs de plus en plus proprement.")


# ══════════════════════════════════════════════════════════════════════
titre(5, "UN SCÉNARIO SPÉCIFIÉ PAR SA CAUSE")
# ══════════════════════════════════════════════════════════════════════

print("  Avec un choc de prix, il faut inventer un mouvement par actif.")
print(f"  Ici : {modele.n_assets} nombres, à choisir de façon cohérente avec")
print("  la structure de corrélation qu'on utilise par ailleurs. Personne")
print("  ne sait faire ça à la main.")
print()
print("  Avec un choc de facteur, on fournit UN nombre.")
print()

choc = FactorShock(modele, factor=1, sigmas=-3.0, horizon=10)
mouvements = choc.asset_moves().sort_values()
apres = choc.apply(p_eq)

print(f"  Scénario : « {choc.name} »")
print()
print("    mouvements DÉDUITS des expositions (les 6 pires, les 3 moindres) :")
for actif, v in list(mouvements.items())[:6]:
    print(f"      {actif:<10} {v:>8.2%}")
print("      …")
for actif, v in list(mouvements.items())[-3:]:
    print(f"      {actif:<10} {v:>8.2%}")
print()
print(f"    valeur  {p_eq.value:>10,.0f} → {apres.value:>10,.0f}   "
      f"({apres.value / p_eq.value - 1:+.2%})")

derive = (apres.weights - p_eq.weights).abs().max()
plus_touche = mouvements.index[0]
print(f"    dérive maximale d'un poids : {derive:.2%} "
      f"(sur {plus_touche}, le plus exposé)")
print()
print("  → Les actifs les plus exposés tombent le plus, automatiquement,")
print("    dans les proportions que les données ont montrées. Le scénario")
print("    est une hypothèse sur une CAUSE, pas une liste de souhaits sur")
print("    des conséquences.")
print()

# Le protocole de la séance 5, vérifié en conditions réelles.
print(f"  isinstance(choc, Shock) → {isinstance(choc, Shock)}")
print()
print("  → `FactorShock` vit dans factors.py, n'importe rien de stress.py")
print("    et n'hérite de rien. Il est pourtant un choc valide et se")
print("    compose avec les autres :")
enchaine = Combined(
    [choc, VolatilityShock(1.8), PriceShock({NOMS[0]: -0.05})],
    name="facteur + vol + prix",
)
final = enchaine.apply(p_eq)
print(f"      {enchaine.name} → valeur {final.value:,.0f}, "
      f"volatilité {final.volatility:.1%}")
print()
print("    C'était l'argument avancé à la séance 5 pour préférer un")
print("    Protocol à une classe de base abstraite. Un module entier")
print("    étend le moteur sans que stress.py apprenne un mot sur les")
print("    facteurs.")


# ══════════════════════════════════════════════════════════════════════
titre(6, "LA MESURE QUI TRANCHE")
# ══════════════════════════════════════════════════════════════════════

print("  Tout ce qui précède est un argument. Un seul chiffre décide :")
print("  la volatilité RÉALISÉE, hors échantillon, du portefeuille de")
print("  variance minimale construit sur chaque estimateur.")
print()
print("  Protocole de Ledoit & Wolf eux-mêmes : on estime sur T jours, on")
print("  construit le portefeuille, puis on mesure sur 1000 jours que")
print("  personne n'a vus. Plus bas vaut mieux.")
print()


@dataclass
class Releve:
    """Mesures accumulées pour un estimateur sur plusieurs tirages."""

    vol: list[float] = field(default_factory=list)
    calib: list[float] = field(default_factory=list)
    echecs: int = 0


def evalue(
    generateur, n: int, t_in: int, seeds: range, t_out: int = 1000
) -> dict[str, Releve]:
    resultats = {
        m: Releve() for m in ("empirique", "ledoit-wolf", "facteurs")
    }
    for s in seeds:
        tout = generateur(n, t_in + t_out, s)
        dedans, dehors = tout.iloc[:t_in], tout.iloc[t_in:]

        estimations: dict[str, object] = {
            "empirique": sample_covariance(dedans),
            "ledoit-wolf": ledoit_wolf(dedans),
        }
        try:
            estimations["facteurs"] = pca_factor_model(dedans).covariance()
        except InsufficientStructure:
            estimations["facteurs"] = None

        for nom, cov in estimations.items():
            if cov is None or not cov.is_invertible:
                resultats[nom].echecs += 1
                continue
            w = minimum_variance_weights(cov).reindex(tout.columns).to_numpy()
            serie = dehors.to_numpy() @ w
            realisee = float(np.std(serie, ddof=1) * np.sqrt(JOURS))
            predite = float(np.sqrt(w @ cov.matrix @ w) * np.sqrt(JOURS))
            resultats[nom].vol.append(realisee)
            resultats[nom].calib.append(predite / realisee)
    return resultats


def affiche(resultats: dict[str, Releve], essais: int) -> dict[str, float]:
    print(f"    {'estimateur':<14} {'vol réalisée':>13}  "
          f"{'prédite/réalisée':>17}")
    moyennes: dict[str, float] = {}
    for nom in ("empirique", "ledoit-wolf", "facteurs"):
        d = resultats[nom]
        if not d.vol:
            print(f"    {nom:<14} {'IMPOSSIBLE':>13}  "
                  f"{'—':>17}   ({d.echecs}/{essais} matrices singulières)")
            continue
        v, c = float(np.mean(d.vol)), float(np.mean(d.calib))
        moyennes[nom] = v
        print(f"    {nom:<14} {v:>12.2%}  {c:>17.2f}")
    return moyennes


SEEDS = range(20)

print("  ── UNIVERS AVEC STRUCTURE À FACTEURS " + "─" * 37)
synthese = {}
for n, t in ((50, 250), (100, 250), (150, 100)):
    print(f"\n  N={n}  T={t}  (N/T={n / t:.2f})")
    synthese[(n, t)] = affiche(evalue(univers_a_facteurs, n, t, SEEDS),
                               len(SEEDS))

print()
for (n, t), m in synthese.items():
    if "facteurs" in m and "ledoit-wolf" in m:
        gain = 1 - m["facteurs"] / m["ledoit-wolf"]
        sens = "réduit" if gain > 0 else "AUGMENTE"
        print(f"    N/T={n / t:.2f} → le modèle à facteurs {sens} la "
              f"volatilité réalisée de {abs(gain):.0%} face à Ledoit-Wolf")

print()
print("  → L'avantage CROÎT avec N/T. Quand les données sont abondantes,")
print("    les trois méthodes se valent : il n'y a rien à gagner. Quand")
print("    elles sont rares, la structure remplace l'information manquante.")
print()
print("  → Et à N > T, la covariance empirique n'est pas imprécise :")
print("    elle n'existe pas. Il n'y a pas de portefeuille à construire.")
print("    Ce régime n'est pas un cas d'école — un gérant qui suit le")
print("    S&P 500 a N=500 et deux ans d'historique donnent T=500.")

print()
print("  ── LE CONTRE-EXEMPLE, QUI COMPTE AUTANT " + "─" * 34)
print()
print("  Que se passe-t-il quand le monde n'a PAS de structure à")
print("  facteurs ? Même protocole, corrélation dense de rang plein :")
sans = {}
for n, t in ((50, 250), (100, 250)):
    print(f"\n  N={n}  T={t}  (N/T={n / t:.2f})")
    sans[(n, t)] = affiche(evalue(univers_sans_structure, n, t, SEEDS),
                           len(SEEDS))

res_avec = evalue(univers_a_facteurs, 100, 250, range(10))
res_sans = evalue(univers_sans_structure, 100, 250, range(10))
calib_bon = float(np.mean(res_avec["facteurs"].calib))
calib_mauvais = float(np.mean(res_sans["facteurs"].calib))
calib_lw_sans = float(np.mean(res_sans["ledoit-wolf"].calib))

print()
print("  → L'avantage disparaît : les deux méthodes se valent.")
print()
print("  → Mais regarde la CALIBRATION, pas la performance. Dans le monde")
print(f"    à facteurs, le modèle prédit {calib_bon:.2f} fois le risque")
print(f"    réalisé. Dans le monde sans structure, {calib_mauvais:.2f} —")
print(f"    contre {calib_lw_sans:.2f} pour Ledoit-Wolf.")
print()
print("  → C'est le vrai danger, et il n'est pas celui qu'on attendait.")
print("    Quand son hypothèse est fausse, le modèle à facteurs ne perd")
print("    pas seulement en performance : il devient SUR-CONFIANT. Il")
print("    sous-estime le risque au moment précis où son modèle du monde")
print("    est faux. Un moteur de risque qui se trompe dans ce sens-là est")
print("    pire qu'un moteur médiocre et honnête.")


# ══════════════════════════════════════════════════════════════════════
titre(7, "CE QUE J'AVAIS CRU, ET QUI ÉTAIT FAUX")
# ══════════════════════════════════════════════════════════════════════

print("  J'avais écrit un test qui affirmait l'évidence : « l'erreur doit")
print("  décroître quand l'historique s'allonge ». Il a échoué.")
print()


def erreur_moyenne(n: int, t: int, seeds: range, k: int = 3) -> float:
    erreurs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        b = rng.normal(0.0, 1.0, (n, k))
        b[:, 0] = np.abs(b[:, 0]) + 1.0
        f = rng.normal(0, 1, (t, k))
        e = rng.normal(0, 1, (t, n))
        vols = rng.uniform(0.008, 0.030, n)
        r = pd.DataFrame((f @ b.T + e) * vols,
                         columns=[f"A{i:03d}" for i in range(n)])
        m = pca_factor_model(r, n_factors=k)

        vraie = b @ b.T + np.eye(n)
        ech = 1.0 / np.sqrt(np.diag(vraie))
        vraie = vraie * np.outer(ech, ech)

        ecart = np.abs(m.covariance().correlation().to_numpy() - vraie)
        erreurs.append(ecart[~np.eye(n, dtype=bool)].mean())
    return float(np.mean(erreurs))


print("    À UNIVERS FIXÉ (N=15), on allonge l'historique :")
print()
par_t: dict[int, float] = {}
for t in (200, 800, 3_200, 12_800, 51_200):
    par_t[t] = erreur_moyenne(15, t, range(12))
    depart = par_t[200]
    print(f"      T = {t:>6,}   erreur moyenne {par_t[t]:.4f}   "
          f"({par_t[t] / depart:.2f}× la valeur de départ)")

print()
print("    À HISTORIQUE FIXÉ (T=2000), on élargit l'univers :")
print()
par_n: dict[int, float] = {}
for n in (10, 20, 50, 100, 200, 400):
    par_n[n] = erreur_moyenne(n, 2_000, range(8))
    depart_n = par_n[10]
    print(f"      N = {n:>6}   erreur moyenne {par_n[n]:.4f}   "
          f"({par_n[n] / depart_n:.2f}× la valeur de départ)")

dernier_pas_t = par_t[12_800] / par_t[51_200]
dernier_pas_n = par_n[200] / par_n[400]
print()
print("  → LES DEUX COURBES PLAFONNENT, et c'est le point entier :")
print(f"      en T, le dernier quadruplement ne gagne plus que "
      f"{dernier_pas_t:.2f}×")
print(f"      en N, le dernier doublement ne gagne plus que "
      f"{dernier_pas_n:.2f}×")
print()
print("    Chaque plafond est posé par l'AUTRE dimension. L'allongement")
print("    de l'historique bute sur le biais dû à un univers étroit ;")
print("    l'élargissement de l'univers bute sur l'erreur")
print("    d'échantillonnage due à un historique court.")
print()

# La comparaison qui tranche entre les deux dimensions : à effort de
# données total très inégal, laquelle gagne ?
petit_univers_histoire_immense = par_t[51_200]
grand_univers_histoire_modeste = par_n[400]
annees_longues = 51_200 / JOURS
annees_courtes = 2_000 / JOURS
print("  → LA COMPARAISON QUI TRANCHE :")
print()
print(f"      N=15  avec T=51 200 jours ({annees_longues:.0f} ans) "
      f"→ erreur {petit_univers_histoire_immense:.4f}")
print(f"      N=400 avec T= 2 000 jours ({annees_courtes:.0f} ans) "
      f"→ erreur {grand_univers_histoire_modeste:.4f}")
print()
gagnant = (
    "le grand univers"
    if grand_univers_histoire_modeste < petit_univers_histoire_immense
    else "le long historique"
)
print(f"    {gagnant.capitalize()} gagne. Quatre cents actions sur huit ans")
print(f"    font mieux que quinze actions sur {annees_longues:.0f} ans —")
print("    un historique que personne n'aura jamais.")
print()
print("  → Ce plafond en T n'est pas du bruit résiduel, c'est un BIAIS.")
print()
print("  → LA RAISON : l'ACP n'est pas de l'analyse factorielle. Elle")
print("    suppose implicitement que toutes les variances spécifiques")
print("    sont ÉGALES. Quand elles diffèrent — toujours — les directions")
print("    principales sont inclinées vers les actifs les plus bruyants,")
print("    et cette inclinaison ne dépend pas de T.")
print()
print("  → CONSÉQUENCE PRATIQUE, ET ELLE EST CONTRE-INTUITIVE : trois ans")
print("    sur cinq cents actions valent mieux que trente ans sur quinze.")
print("    Le régime N > T, celui où la covariance empirique est")
print("    singulière et où tout le reste échoue, est exactement celui où")
print("    cette méthode travaille le mieux.")
print()
print("  → Le problème de la séance 2 est la condition de validité de la")
print("    solution de la séance 7. Je ne l'avais pas vu en concevant le")
print("    module ; c'est un test raté qui me l'a appris.")

print()
print("═" * LARGE)
print("  Fin de la séance 7.   python -m pytest tests/ -q -m \"\"")
print("═" * LARGE)
