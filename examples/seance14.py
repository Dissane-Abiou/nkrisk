"""
Séance 14 — Le rapport de validation : le moteur forme une opinion.

Treize séances ont construit des instruments. Celle-ci ne mesure rien de
neuf : elle fait parler les instruments d'une seule voix, et surtout
elle refuse de le faire silencieusement.

POURQUOI C'EST UN TRAVAIL, ET PAS DE L'EMBALLAGE
-------------------------------------------------
Parce que les instruments se contredisent. L'inventaire de la séance 12
valide un modèle que le backtest d'ES de la séance 13 rejette. Le
module de qualité de la séance 9 signale des données que les autres ont
utilisées sans broncher. Le protocole fixe valide ce que le protocole
du modèle rejette.

Un rapport qui choisirait un verdict et tairait les autres mentirait
par omission. Un rapport qui les empilerait serait illisible. Il faut
une règle, et elle doit être défendable.

Cette démonstration construit cinq univers, chacun fabriqué pour
déclencher une famille de conflit précise, et vérifie que le rapport la
voit. Puis elle produit le fichier HTML qui ira sur les GitHub Pages.

Aucun nombre de ce fichier n'est écrit à la main.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nkrisk.report import (
    CORPS_QUEUE,
    NIVEAU_STABILITE,
    PERIMETRE,
    PROTOCOLE,
    REGIME,
    build_report,
)

SEP = "=" * 70
T = 1300
RACINE = Path(__file__).resolve().parent.parent


def univers(
    n_actifs: int,
    graine: int,
    *,
    n_facteurs: int = 3,
    nu: int | None = None,
    garch: bool = False,
    n_periodes: int = T,
) -> pd.DataFrame:
    """
    Rendements à structure de facteurs connue.

    `nu` donne des queues de Student, `garch` une volatilité de marché
    qui bouge. Les expositions sont toutes positives — un facteur de
    marché — pour que les corrélations soient réalistes.
    """
    g = np.random.default_rng(graine)
    expositions = np.abs(g.normal(0.0, 0.5, (n_actifs, n_facteurs)))

    def tirer(forme: tuple[int, int]) -> np.ndarray:
        if nu is None:
            return g.normal(0.0, 1.0, forme)
        return g.standard_t(nu, forme) * np.sqrt((nu - 2) / nu)

    brut = tirer((n_periodes, n_facteurs)) @ expositions.T + tirer(
        (n_periodes, n_actifs)
    )

    if garch:
        h = np.empty(n_periodes)
        h[0] = 1.0
        choc = g.normal(0.0, 1.0, n_periodes)
        for t in range(n_periodes - 1):
            h[t + 1] = 0.01 + 0.09 * (choc[t] ** 2) * h[t] + 0.90 * h[t]
        brut = brut * np.sqrt(h)[:, None]

    return pd.DataFrame(
        brut * 0.01,
        index=pd.bdate_range("2019-01-02", periods=n_periodes),
        columns=[f"A{i:02d}" for i in range(n_actifs)],
    )


def prix_avec_defauts(returns: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Reconstitue des prix et y glisse trois défauts que la séance 9 sait
    reconnaître : une introduction en bourse tardive, un calendrier
    désaligné, et une série figée.
    """
    prix: dict[str, pd.Series] = {}
    for i, col in enumerate(returns.columns):
        serie = 100.0 * (1.0 + returns[col]).cumprod()
        if i == 0:
            serie = serie.iloc[400:]  # démarrage tardif
        elif i == 1:
            serie = serie.iloc[::2]  # calendrier à moitié
        elif i == 2:
            serie = serie.copy()
            serie.iloc[200:210] = serie.iloc[200]  # prix figé dix jours
        prix[col] = serie
    return prix


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 1 — CINQ UNIVERS, CINQ FAMILLES DE CONFLIT")
print(SEP)
print()
print("Chaque univers est fabriqué pour déclencher une famille précise.")
print("Si le rapport ne la voit pas, la détection ne vaut rien.")
print()

cas = [
    (
        "propre — 15 actifs, gaussien, N/T faible",
        univers(15, 1),
        None,
        "aucune famille en particulier (c'est le témoin)",
    ),
    (
        "queues épaisses — Student t(4)",
        univers(18, 2, nu=4),
        None,
        CORPS_QUEUE,
    ),
    (
        "encombré — 110 actifs pour 252 jours de fenêtre",
        univers(110, 3, n_facteurs=5),
        None,
        PROTOCOLE,
    ),
    (
        "agité — volatilité groupée",
        univers(15, 4, garch=True),
        None,
        REGIME,
    ),
]
donnees_sales = univers(12, 5)
cas.append(
    (
        "données suspectes — IPO tardive, calendrier, prix figé",
        donnees_sales,
        prix_avec_defauts(donnees_sales),
        PERIMETRE,
    )
)

rapports = []
print(f"  {'univers':<46}{'conflits':>9}   {'familles vues'}")
print("  " + "-" * 92)
for nom, donnees, prix, attendu in cas:
    rapport = build_report(
        donnees,
        prices=prix,
        universe=nom,
        n_sim=1500,
        check_causality=False,
    )
    rapports.append((nom, rapport, attendu))
    familles = sorted({c.family for c in rapport.conflicts})
    print(
        f"  {nom[:45]:<46}{len(rapport.conflicts):>9}   "
        f"{', '.join(familles) if familles else '—'}"
    )

print()
print("  Vérification famille par famille :")
print()
for nom, rapport, attendu in rapports:
    if attendu is None or not attendu.isupper():
        continue
    familles = {c.family for c in rapport.conflicts}
    marque = "✓" if attendu in familles else "✗ MANQUÉE"
    print(f"    {attendu:<22} sur « {nom[:38]:<38} » : {marque}")
print()
manquees = [
    a
    for _, r, a in rapports
    if a is not None and a.isupper() and a not in {c.family for c in r.conflicts}
]
if manquees:
    print(f"  ⚠ {len(manquees)} famille(s) non détectée(s). La détection est "
          "incomplète.")
else:
    print("  → Les cinq familles sont détectées sur les univers qui les portent.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 2 — LA RÈGLE DE CONJONCTION, ET POURQUOI ELLE EST SÉVÈRE")
print(SEP)
print()
print("Le verdict d'un modèle est LE PIRE des verdicts rendus sur lui.")
print("Pas une moyenne, pas un vote. La raison est asymétrique :")
print()
print("    un instrument qui valide dit « je n'ai pas trouvé de défaut")
print("    dans ce que je sais regarder » ;")
print("    un instrument qui rejette dit « j'ai trouvé un défaut ».")
print()
print("Une absence de preuve ne compense pas une preuve.")
print()

nom_encombre, rapport_encombre, _ = rapports[2]
print(f"  Sur « {nom_encombre} » :")
print()
print(rapport_encombre.table().to_string(index=False))
print()

desaccords = [
    s
    for s in rapport_encombre.subjects
    if len({f.verdict for f in rapport_encombre.findings_for(s)}) > 1
]
print(f"  {len(desaccords)} modèle(s) sur {len(rapport_encombre.subjects)} reçoivent")
print("  des verdicts DIFFÉRENTS selon l'instrument. Sans règle, le rapport")
print("  n'aurait rien pu conclure sur eux.")
print()
print("  Ce qui rend le verdict actionnable n'est pas le verdict : c'est la")
print("  colonne « contraint par ». Elle nomme l'instrument qui décide,")
print("  donc ce qu'il faut réparer.")
print()
for s in rapport_encombre.subjects:
    print(
        f"    {s:<32}{rapport_encombre.verdict_for(s):<14}"
        f"réparer d'abord : {rapport_encombre.binding_instrument(s)}"
    )
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 3 — DEUX DÉFAUTS DE MA PREMIÈRE VERSION")
print(SEP)
print()
print("  PREMIER DÉFAUT — UN TEST QUE J'AVAIS RENDU AVEUGLE.")
print()
print("  Ma première écriture du verdict de queue passait les résidus")
print("  standardisés du modèle à l'hypothèse nulle empirique, et coupait")
print("  la vérification de cohérence de la séance 13. L'hypothèse nulle")
print("  était donc la forme de queue RÉALISÉE : le test ne pouvait pas")
print("  rejeter une forme de queue mal annoncée, puisqu'il l'avait")
print("  absorbée dans sa propre référence. Il validait tout.")
print()
print("  C'est exactement le piège des séances 12 et 13, commis une")
print("  troisième fois par leur auteur. Le verdict de queue a donc été")
print("  séparé en deux composantes indépendantes :")
print()
_, rapport_queues, _ = rapports[1]
for f in rapport_queues.findings:
    if f.instrument == "queue":
        print(f"    {f.subject:<30}{f.verdict}")
        print(f"      {f.detail}")
print()
print("  La FORME se juge sans aucune simulation, par le rapport ES/VaR :")
print("  le modèle annonce une queue gaussienne, les résidus en ont une")
print("  autre. Le garde-fou de la séance 13 fait tout le travail.")
print()
print("  SECOND DÉFAUT — UN CONFLIT QUI VISE TOUT LE MONDE NE DÉSIGNE")
print("  PERSONNE.")
print()
nom_agite, rapport_agite, _ = rapports[3]
print(f"  Sur « {nom_agite} », ma première version affichait quatre fois")
print("  le même conflit NIVEAU / STABILITÉ, un par modèle. Quatre cris")
print("  identiques ne portent pas quatre informations : ils en portent")
print("  une seule, et elle ne parle pas des modèles.")
print()
for c in rapport_agite.conflicts:
    if c.family == REGIME:
        print(f"    {c.subject}")
        print(f"    → {c.interpretation}")
print()
print(f"  La famille {NIVEAU_STABILITE} est désormais repliée en un constat")
print("  de RÉGIME dès qu'elle vise tous les modèles jugés. Un rapport qui")
print("  crie à chaque ligne ne se lit plus.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 4 — LE FICHIER QUI SORT")
print(SEP)
print()

docs = RACINE / "docs"
docs.mkdir(exist_ok=True)
cible = docs / "rapport-validation.html"

nom_demo, rapport_demo, _ = rapports[1]
html = rapport_demo.to_html()
cible.write_text(html, encoding="utf-8")

print(f"  {cible.relative_to(RACINE)} — {len(html):,} octets")
print()
print("  Autonome : aucune feuille de style externe, aucun script, aucune")
print("  police chargée depuis ailleurs. Ce n'est pas de la coquetterie —")
print("  un rapport de validation doit pouvoir être joint à un courriel,")
print("  archivé, et rouvert dans dix ans. Tout ce qui est chargé depuis")
print("  ailleurs aura disparu.")
print()
print("  Lisible en mode clair et en mode sombre, et au format téléphone.")
print()
print("  ⚠️ CE RAPPORT EST SYNTHÉTIQUE. Les chiffres ci-dessus viennent de")
print("     données fabriquées pour exercer la détection de conflits, pas")
print("     d'un marché. `examples/rapport_reel.py` produit le vrai, et il")
print("     demande un accès réseau à Yahoo Finance.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 5 — CE QUE LA SÉANCE 14 A AJOUTÉ")
print(SEP)
print()
print("  Aucune mesure nouvelle. C'est assumé, et c'est écrit en tête du")
print("  module. Ce qui est nouveau :")
print()
print("  1. Une règle de conjonction défendable — le pire verdict l'emporte,")
print("     parce qu'une absence de preuve ne compense pas une preuve — et")
print("     `binding_instrument`, qui dit quoi réparer.")
print()
print("  2. Cinq familles de conflits, chacune avec son INTERPRÉTATION.")
print("     « Deux instruments divergent » n'est pas exploitable ; « le")
print("     modèle décrit les jours normaux et se trompe sur ceux qui")
print("     comptent » l'est.")
print()
print("  3. Le périmètre en premier : quand la qualité des données signale")
print("     des anomalies que les autres instruments ont ignorées, tous les")
print("     verdicts deviennent conditionnels, et le rapport le dit avant")
print("     de les énoncer.")
print()
print("  4. Le repliement des conflits universels en constat de régime.")
print()
print("  5. Un fichier HTML autonome, lisible en soixante secondes par")
print("     quelqu'un qui ne lira jamais le code.")
print()
print("  ET LA LIMITE, écrite dans chaque rapport produit : il ne cherche")
print("  que les conflits qu'on lui a appris à reconnaître. Deux")
print("  instruments peuvent diverger d'une façon qu'il ne voit pas —")
print("  exactement comme un test ne voit que ce que sa statistique")
print("  regarde. Les séances 12 et 13 l'ont établi pour les tests ; il n'y")
print("  avait aucune raison que le rapport y échappe.")
print()
print(SEP)
