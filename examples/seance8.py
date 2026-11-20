"""
Séance 8 — Le backtest, et l'audit qui vérifie qu'il ne triche pas.

    python examples/seance8.py

RÈGLE DE CE FICHIER, HÉRITÉE DES SÉANCES 2, 3, 5 ET 7
=====================================================
Aucun nombre n'est écrit à la main dans les narrations. Tout ce qui est
affiché est calculé à partir des mêmes données que les tableaux.

Huit occurrences du même bug ont établi cette règle. La huitième, à la
séance 7, était dans une docstring — qui, elle, ne s'exécute jamais.
Depuis, les affirmations chiffrées qui survivent dans la prose sont
celles qu'un test recalcule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from nkrisk.backtest import (
    EqualWeight,
    MinimumVariance,
    deflated_sharpe_ratio,
    detect_lookahead,
    expected_maximum_sharpe,
    minimum_track_record_length,
    walk_forward,
)
from nkrisk.covariance import ledoit_wolf, sample_covariance

JOURS = 252
LARGE = 76


def titre(n: int, texte: str) -> None:
    print()
    print("═" * LARGE)
    print(f"  LEÇON {n} — {texte}")
    print("═" * LARGE)
    print()


# ══════════════════════════════════════════════════════════════════════
#  UN MARCHÉ OÙ PERSONNE N'A D'AVANTAGE
# ══════════════════════════════════════════════════════════════════════


def marche(n: int = 25, t: int = 2016, k: int = 3, seed: int = 11
           ) -> pd.DataFrame:
    """
    Rendements à structure factorielle, et DÉRIVE RIGOUREUSEMENT NULLE
    pour tous les actifs.

    C'est le point du fichier : dans ce monde, aucune sélection d'actifs
    ne peut avoir de valeur, par construction. Tout ce qu'une stratégie
    y gagnera est de la chance — et on va mesurer à quel point cette
    chance peut être convaincante.
    """
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 0.7, (n, k))
    b[:, 0] = np.abs(rng.normal(1.0, 0.3, n))
    f = rng.normal(0, 1, (t, k))
    e = rng.normal(0, 1, (t, n)) * rng.uniform(0.5, 1.8, n)
    vols = rng.uniform(0.008, 0.030, n)
    return pd.DataFrame(
        (f @ b.T + e) * vols,
        index=pd.bdate_range("2016-01-04", periods=t),
        columns=[f"A{i:02d}" for i in range(n)],
    )


DONNEES = marche()
LOOKBACK, PAS = 252, 21
N_RETENUS = 4


# ══════════════════════════════════════════════════════════════════════
#  DEUX STRATÉGIES QUI SE RESSEMBLENT BEAUCOUP
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChoisitLesGagnants:
    """
    Garde les quatre actifs les plus performants… sur TOUT l'échantillon.

    Personne n'écrit ça volontairement. On l'écrit sans le voir : en
    filtrant un univers avant la boucle d'évaluation, en reprenant une
    liste de titres construite plus tard, en partant d'un indice dont la
    composition d'aujourd'hui est appliquée à hier.

    Noter qu'elle respecte parfaitement la signature du contrat. Elle
    reçoit `history` et pourrait n'utiliser qu'elle. Aucune vérification
    de type ne la distingue de la suivante.
    """

    donnees_completes: pd.DataFrame
    name: str = "choisit les gagnants (FUITE)"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        meilleurs = self.donnees_completes.mean().nlargest(N_RETENUS).index
        w = pd.Series(0.0, index=history.columns)
        w[meilleurs] = 1.0 / N_RETENUS
        return w


@dataclass(frozen=True)
class MomentumHonnete:
    """Exactement la même idée, sur l'historique visible seulement."""

    name: str = "momentum honnête"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        meilleurs = history.mean().nlargest(N_RETENUS).index
        w = pd.Series(0.0, index=history.columns)
        w[meilleurs] = 1.0 / N_RETENUS
        return w


# ══════════════════════════════════════════════════════════════════════
titre(1, "DEUX COURBES, DONT UNE EST UNE FICTION")
# ══════════════════════════════════════════════════════════════════════

derive_annuelle = float(DONNEES.mean(axis=1).mean() * JOURS)
dispersion_actifs = float(DONNEES.mean().std(ddof=1) * JOURS)

print("  Marché simulé : 25 actifs, 8 ans, et une ESPÉRANCE de rendement")
print("  rigoureusement nulle pour chacun. Aucune sélection d'actifs ne")
print("  peut y avoir la moindre valeur — c'est vrai par construction.")
print()
print("  ⚠️ Espérance nulle ne veut pas dire réalisation nulle. Sur CE")
print("     tirage précis :")
print()
print(f"       dérive réalisée du panier équipondéré : "
      f"{derive_annuelle:+.2%} par an")
print(f"       dispersion des dérives entre actifs   : "
      f"{dispersion_actifs:.2%} par an")
print()
print("     Huit ans de données ne suffisent pas à ramener une moyenne à")
print("     zéro — la séance 1 l'avait chiffré : il faudrait de l'ordre de")
print("     quatre-vingt-dix ans. Tout ce qui suit est donc de la chance,")
print("     y compris les résultats des stratégies HONNÊTES.")
print()

fuite = walk_forward(DONNEES, ChoisitLesGagnants(DONNEES),
                     lookback=LOOKBACK, step=PAS)
honnete = walk_forward(DONNEES, MomentumHonnete(),
                       lookback=LOOKBACK, step=PAS)
reference = walk_forward(DONNEES, EqualWeight(), lookback=LOOKBACK, step=PAS)

print(f"    {'stratégie':<30} {'capital final':>14} {'Sharpe':>18}")
for res in (fuite, honnete, reference):
    sr = res.sharpe()
    print(f"    {res.strategy:<30} {res.equity.iloc[-1]:>13.2f}× "
          f"{sr.value:>10.2f} [{sr.ci_low:.2f};{sr.ci_high:.2f}]")

ecart = fuite.equity.iloc[-1] / honnete.equity.iloc[-1]
print()
print(f"  → La première multiplie le capital par "
      f"{fuite.equity.iloc[-1]:.2f}, la seconde par "
      f"{honnete.equity.iloc[-1]:.2f}.")
print(f"    Un rapport de {ecart:.1f} entre deux stratégies qui font LA MÊME")
print("    CHOSE — garder les quatre meilleurs actifs — dans un marché où")
print("    aucun actif n'est meilleur qu'un autre.")
print()
print("  → La différence tient en un mot : la première les choisit en")
print("    connaissant la suite. Elle ne plante pas, ne produit aucun NaN,")
print("    et respecte scrupuleusement la signature du contrat.")
print()
print("  → C'est le mode de défaillance le plus coûteux de la finance")
print("    quantitative, et ce n'est pas une erreur de mathématiques :")
print("    c'est une erreur de STRUCTURE DU PROGRAMME. Quelque chose a")
print("    atteint un objet qu'il n'aurait pas dû pouvoir atteindre.")


# ══════════════════════════════════════════════════════════════════════
titre(2, "LA BORNE QUI DÉCIDE DE TOUT")
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Espionne:
    """Note la dernière date qu'on lui a montrée."""

    vues: list
    name: str = "espionne"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        self.vues.append(history.index[-1])
        n = history.shape[1]
        return pd.Series(np.full(n, 1.0 / n), index=history.columns)


vues: list = []
walk_forward(DONNEES, Espionne(vues), lookback=LOOKBACK, step=PAS)
positions = list(range(LOOKBACK, len(DONNEES), PAS))

print("  Le moteur tient en une ligne :")
print()
print("      history = returns.iloc[debut:t]      # t EXCLU")
print("      ...appliqué à returns.iloc[t:t+step]")
print()
print("  Une stratégie espionne enregistre ce qu'elle a réellement vu :")
print()
print(f"    {'décision':<14} {'dernière date vue':>19} "
      f"{'1re date échangée':>19}")
for t, vue in list(zip(positions, vues, strict=True))[:4]:
    print(f"    position {t:<5} {vue.date()!s:>19} "
          f"{DONNEES.index[t].date()!s:>19}")
print("    …")

avances = {
    (t - DONNEES.index.get_loc(v)) for t, v in zip(positions, vues, strict=True)
}
print()
assert len(avances) == 1, f"écarts hétérogènes : {sorted(avances)}"
print(f"  → Sur les {len(positions)} décisions, la première période échangée")
print(f"    vient toujours {avances.pop()} période APRÈS la dernière donnée")
print("    vue. Jamais le même jour, jamais avant, sans exception.")
print()
print("  → Écrire `iloc[debut:t+1]` donnerait un écart de 0 : la stratégie")
print("    déciderait en connaissant le jour qu'elle va échanger. Aucune")
print("    exception, aucun changement de forme des tableaux, et une")
print("    courbe de performance spectaculaire.")
print()
print("  → Une unité. C'est toute la différence entre un backtest et une")
print("    illusion, et c'est pour ça qu'un test la verrouille.")


# ══════════════════════════════════════════════════════════════════════
titre(3, "L'AUDIT : TESTER UNE PROPRIÉTÉ, PAS RELIRE DU CODE")
# ══════════════════════════════════════════════════════════════════════

print("  Le confinement par découpage ne protège que de ce qui passe par")
print("  l'appel. Il ne dit rien d'une stratégie qui garderait par ailleurs")
print("  une référence vers les données complètes — ce qui est exactement")
print("  le cas de la première.")
print()
print("  On définit donc la propriété qu'on veut vraiment garantir :")
print()
print("      Un backtest est CAUSAL si perturber les données postérieures")
print("      à une date t ne change AUCUNE décision antérieure à t.")
print()
print("  C'est une définition, pas une heuristique. Elle se vérifie par")
print("  l'expérience : on corrompt le futur, on rejoue tout, on compare.")
print()


@dataclass(frozen=True)
class NormaliseeSurTout:
    """Fuite subtile : le QUOI est honnête, le COMBIEN ne l'est pas."""

    donnees_completes: pd.DataFrame
    name: str = "normalisée sur tout l'échantillon (FUITE)"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        brut = (1.0 / self.donnees_completes.std(ddof=1)).reindex(
            history.columns
        )
        return brut / brut.sum()


@dataclass(frozen=True)
class AvanceDunJour:
    """Fuite minuscule : un seul jour d'avance. Un `shift` mal orienté."""

    donnees_completes: pd.DataFrame
    name: str = "un seul jour d'avance (FUITE)"

    def weights(self, history: pd.DataFrame) -> pd.Series:
        ici = self.donnees_completes.index.get_loc(history.index[-1]) + 1
        fin = min(ici + 1, len(self.donnees_completes))
        return pd.Series(
            minimum_variance_poids(self.donnees_completes.iloc[:fin]),
            index=history.columns,
        )


def minimum_variance_poids(donnees: pd.DataFrame) -> np.ndarray:
    from nkrisk.risk import minimum_variance_weights

    return minimum_variance_weights(ledoit_wolf(donnees)).to_numpy()


candidats = [
    ("équipondéré", lambda _d: EqualWeight()),
    ("momentum honnête", lambda _d: MomentumHonnete()),
    ("variance minimale", lambda _d: MinimumVariance(ledoit_wolf, "var. min.")),
    ("choisit les gagnants", ChoisitLesGagnants),
    ("normalisée sur tout", NormaliseeSurTout),
    ("un seul jour d'avance", AvanceDunJour),
]

print(f"    {'stratégie':<26} {'verdict':<10} {'écart max sur les poids':>24}")
audits = {}
for nom, fabrique in candidats:
    audit = detect_lookahead(DONNEES, fabrique, lookback=LOOKBACK, step=PAS)
    audits[nom] = audit
    verdict = "✓ causale" if audit.clean else "✗ FUITE"
    print(f"    {nom:<26} {verdict:<10} "
          f"{audit.worst_difference:>24.2e}")

propres = [n for n, a in audits.items() if a.clean]
sales = [n for n, a in audits.items() if not a.clean]
print()
print(f"  → {len(propres)} stratégies déclarées causales, {len(sales)} fuites")
print("    détectées. Aucune fausse alerte, et aucune fuite manquée.")
print()
print("  → L'audit n'a lu le code d'aucune d'entre elles. Il n'a mesuré ni")
print("    performance, ni volatilité : il a comparé des DÉCISIONS.")
print()
print("  → Et il a vu la fuite d'UN SEUL JOUR sur huit ans de données,")
print("    soit 0,05 % de l'historique — celle qu'aucune relecture ne")
print("    trouve et qu'aucune courbe ne trahit.")
print()
print("  La limite, et elle doit être dite :")
print("  l'audit ne voit que ce qui passe par les données et par la")
print("  fabrique de la stratégie. Une fuite qui viendrait d'un fichier,")
print("  du réseau ou d'une variable globale capturée avant l'audit lui")
print("  échapperait. Il prouve la causalité RELATIVEMENT à ce qu'on lui")
print("  confie. Un vérificateur qui laisse croire qu'il garantit plus")
print("  qu'il ne garantit est pire que pas de vérificateur du tout.")


# ══════════════════════════════════════════════════════════════════════
titre(4, "LA SENSIBILITÉ NE VIENT PAS DE LA FINESSE, MAIS DU PLACEMENT")
# ══════════════════════════════════════════════════════════════════════

print("  Première version du détecteur : des sondes réparties")
print("  uniformément sur l'historique. Elle attrapait la stratégie")
print("  ajustée sur tout l'échantillon, et RATAIT l'avance d'un jour.")
print()
print("  La raison tient en une inégalité. Une décision prise en position")
print("  t avec une avance de a périodes lit jusqu'à l'indice t+a−1. Elle")
print("  n'est démasquée que si la corruption commence à un indice")
print("  qu'elle atteint, donc si cut ≤ t+a−1.")
print()

decisions = list(range(LOOKBACK, len(DONNEES), PAS))
entre_deux = [t + PAS // 2 for t in decisions[:-1]]

aveugle = detect_lookahead(DONNEES, AvanceDunJour, lookback=LOOKBACK,
                           step=PAS, cuts=entre_deux)
alignee = detect_lookahead(DONNEES, AvanceDunJour, lookback=LOOKBACK,
                           step=PAS, cuts=decisions)

print(f"    {'placement des sondes':<34} {'nb sondes':>10} {'verdict':>12}")
print(f"    {'entre les dates de décision':<34} {len(entre_deux):>10} "
      f"{'✓ causale' if aveugle.clean else '✗ FUITE':>12}")
print(f"    {'SUR les dates de décision':<34} {len(decisions):>10} "
      f"{'✓ causale' if alignee.clean else '✗ FUITE':>12}")
print()
print("  → Même détecteur, même fuite, mêmes données. Seul change")
print("    l'endroit où l'on coupe, et le verdict s'inverse.")
print()
print(f"  → Avec des sondes au hasard et un pas de {PAS} jours, la fenêtre")
print(f"    utile fait 1 jour sur {PAS} : environ "
      f"{100 / PAS:.0f} % de chances de voir la fuite.")
print("    En posant la sonde SUR la date de décision, l'inégalité devient")
print("    cut = t ≤ t+a−1, vraie dès que a ≥ 1. Toute avance est vue.")
print()
print("  → **La sensibilité d'un détecteur ne vient pas de sa finesse,")
print("    mais de l'endroit où on le pose.** C'est vrai des journaux,")
print("    des assertions, des points d'arrêt et de la métrologie en")
print("    général. Ici, c'est mesurable.")


# ══════════════════════════════════════════════════════════════════════
titre(5, "LES COÛTS, OU POURQUOI LE MEILLEUR MODÈLE PERD QUAND MÊME")
# ══════════════════════════════════════════════════════════════════════

print("  Sans coûts de transaction, tout backtest favorise")
print("  mécaniquement les stratégies qui tournent le plus. La séance 7")
print("  comparait des volatilités réalisées sans jamais regarder ce que")
print("  coûtait de les obtenir.")
print()

strategies = [
    ("équipondéré", EqualWeight()),
    ("var. min. empirique", MinimumVariance(sample_covariance, "empirique")),
    ("var. min. Ledoit-Wolf", MinimumVariance(ledoit_wolf, "ledoit-wolf")),
]

print(f"    {'stratégie':<24} {'vol réalisée':>13} {'turnover/an':>12} "
      f"{'capital 0 pb':>13} {'capital 20 pb':>14}")
resultats = {}
for nom, strat in strategies:
    sans = walk_forward(DONNEES, strat, lookback=LOOKBACK, step=PAS,
                        cost_bps=0)
    avec = walk_forward(DONNEES, strat, lookback=LOOKBACK, step=PAS,
                        cost_bps=20)
    resultats[nom] = (sans, avec)
    print(f"    {nom:<24} {sans.realized_volatility().value:>12.2%} "
          f"{sans.annual_turnover():>11.2f}× "
          f"{sans.equity.iloc[-1]:>12.3f} {avec.equity.iloc[-1]:>13.3f}")

vol_eq = resultats["équipondéré"][0].realized_volatility().value
vol_lw = resultats["var. min. Ledoit-Wolf"][0].realized_volatility().value
rot_eq = resultats["équipondéré"][0].annual_turnover()
rot_lw = resultats["var. min. Ledoit-Wolf"][0].annual_turnover()
cout_lw = resultats["var. min. Ledoit-Wolf"][1].total_cost()
cout_eq = resultats["équipondéré"][1].total_cost()

print()
print("  ⚠️ NE COMPARE PAS LES COLONNES « capital » ENTRE LIGNES.")
print()
print("     L'équipondéré finit beaucoup plus haut — mais uniquement")
print(f"     parce que ce tirage a une dérive réalisée de "
      f"{derive_annuelle:+.2%} par an,")
print("     que la variance minimale couvre délibérément en neutralisant")
print("     le facteur marché. La colonne capital mesure ici de la chance,")
print("     pas de la compétence. Les colonnes qui comparent vraiment")
print("     quelque chose sont la volatilité et le turnover.")
print()
print(f"  → La variance minimale réduit la volatilité de "
      f"{1 - vol_lw / vol_eq:.0%} face à")
print("    l'équipondéré — c'est son métier, et elle le fait.")
print()
print(f"  → Elle le paie par un turnover de {rot_lw:.2f}× l'an contre "
      f"{rot_eq:.2f}×,")
print(f"    soit {rot_lw / rot_eq:.1f} fois plus de rotation. À 20 points de")
print(f"    base, cela fait {cout_lw:.2%} de frais cumulés contre "
      f"{cout_eq:.2%}.")
print()
print("  → Le coût propre à chaque stratégie, lui, se lit en comparant")
print("    ses deux colonnes à elle :")
print()
for nom in ("équipondéré", "var. min. empirique", "var. min. Ledoit-Wolf"):
    sans, avec = resultats[nom]
    perte = 1 - avec.equity.iloc[-1] / sans.equity.iloc[-1]
    print(f"      {nom:<24} les frais coûtent {perte:>6.2%} du capital final")
print()
print("  → Un backtest sans coûts n'est pas « une première approximation ».")
print("    C'est une mesure dont on a retiré le terme qui départage les")
print("    stratégies qu'on compare.")


# ══════════════════════════════════════════════════════════════════════
titre(6, "LE CHIFFRE QU'AUCUN BACKTEST PUBLIÉ NE MONTRE")
# ══════════════════════════════════════════════════════════════════════

print("  Tous les rapports de backtest donnent le rendement et la")
print("  volatilité RÉALISÉS. Aucun ne donne l'écart entre le risque que")
print("  le modèle ANNONÇAIT et celui qui est arrivé.")
print()
print("  C'est pourtant la seule mesure qui juge le modèle de risque")
print("  indépendamment de la performance — et c'est le principe du")
print("  projet depuis la séance 1, appliqué au backtest lui-même.")
print()

print(f"    {'estimateur':<24} {'annoncé':>10} {'subi':>10} "
      f"{'rapport':>26}")
for nom in ("var. min. empirique", "var. min. Ledoit-Wolf"):
    res = resultats[nom][0]
    cal = res.calibration()
    assert cal is not None
    print(f"    {nom:<24} "
          f"{res.predicted_volatility.mean():>9.2%} "  # type: ignore[union-attr]
          f"{res.realized_volatility().value:>9.2%} "
          f"{cal.format(2):>26}")

cal_emp = resultats["var. min. empirique"][0].calibration()
cal_lw = resultats["var. min. Ledoit-Wolf"][0].calibration()
assert cal_emp is not None and cal_lw is not None
print()
print(f"  → La covariance empirique annonce "
      f"{cal_emp.value:.2f} fois le risque subi.")
print(f"    Ledoit-Wolf annonce {cal_lw.value:.2f} fois.")
print()
if cal_emp.value < 1.0:
    print("  → Sous 1,00, l'estimateur SOUS-ESTIME son propre risque. Pour")
    print("    un moteur de risque, c'est le sens dangereux de l'erreur :")
    print("    on découvre la vraie volatilité le jour où elle se produit.")
print()
print("  → L'optimiseur de variance minimale choisit les poids qui")
print("    MINIMISENT la variance estimée. Il va donc chercher les")
print("    directions où l'estimation est la plus basse — qui sont")
print("    souvent celles où elle se trompe le plus vers le bas. La")
print("    sous-estimation n'est pas un accident, elle est le produit de")
print("    l'optimisation elle-même.")


# ══════════════════════════════════════════════════════════════════════
titre(7, "TON BACKTEST MESURE LE NOMBRE DE FOIS OÙ TU AS REGARDÉ")
# ══════════════════════════════════════════════════════════════════════

print("  Promesse faite dans la docstring de `sharpe_ratio`, séance 1 :")
print("  « On implémentera le Deflated Sharpe Ratio dans une séance")
print("  ultérieure. » La voici.")
print()
print("  Combien vaut le meilleur Sharpe d'un lot de stratégies TOUTES")
print("  SANS VALEUR ? La théorie des valeurs extrêmes le dit exactement :")
print()
dispersion = 0.05
print(f"    (dispersion des Sharpe entre essais : {dispersion:.2f} par période)")
print()
print(f"    {'essais':>10} {'meilleur Sharpe attendu par pure chance':>44}")
for n_essais in (1, 10, 100, 1_000, 10_000):
    attendu = expected_maximum_sharpe(n_essais, dispersion) * np.sqrt(JOURS)
    print(f"    {n_essais:>10} {attendu:>43.2f}")

print()
print("  → Avec assez d'essais, aucun Sharpe n'est assez élevé pour")
print("    rejeter l'hypothèse que la stratégie ne vaut rien. C'est le")
print("    « théorème des fausses stratégies » de Bailey et López de Prado.")
print()
print("  Appliquons-le à notre stratégie fuyante :")
print()

sr_fuite = fuite.sharpe()
print(f"    Sharpe affiché par « choisit les gagnants » : "
      f"{sr_fuite.format(2)}")
print()
combinaisons = comb(len(DONNEES.columns), N_RETENUS)
for n_essais in (1, 25, combinaisons):
    d = deflated_sharpe_ratio(fuite.returns, n_trials=n_essais)
    etat = "survit" if d.survives else "ne survit pas"
    print(f"      {n_essais:>6} essai(s) → probabilité "
          f"{d.probability:>6.1%}   {etat}")

print()
print("  → Le dernier chiffre n'est pas arbitraire : choisir "
      f"{N_RETENUS} actifs")
print(f"    parmi {len(DONNEES.columns)} offre "
      f"{combinaisons:,}".replace(",", " ") + " combinaisons. Une stratégie")
print("    qui « garde les quatre meilleurs » les a TOUTES essayées, même")
print("    si celui qui l'a écrite n'en a tapé qu'une.")
print()
print("  → Et rappel : dans ce marché, l'espérance de rendement de chaque")
print("    actif est rigoureusement nulle. Il n'y avait RIEN à trouver.")
print(f"    Le Sharpe de {sr_fuite.value:.2f} vient entièrement de deux")
print(f"    sources : la dérive de chance du tirage ({derive_annuelle:+.2%}")
print("    par an) et le fait d'avoir choisi en connaissant la suite.")
print()

print("  Le contrôle positif — le test ne doit pas tout rejeter :")
print()
rng = np.random.default_rng(99)
vraie = pd.Series(rng.normal(1.8 * 0.01 / np.sqrt(JOURS), 0.01, 2_000))
d_vraie = deflated_sharpe_ratio(vraie, n_trials=1)
print("    Stratégie à vraie dérive, un seul essai, 8 ans :")
print(f"      Sharpe {d_vraie.observed:.2f} → probabilité "
      f"{d_vraie.probability:.1%} → "
      f"{'survit' if d_vraie.survives else 'ne survit pas'}")
d_vraie_20 = deflated_sharpe_ratio(vraie, n_trials=20)
print("    La même, après 20 essais :")
print(f"      probabilité {d_vraie_20.probability:.1%} → "
      f"{'survit' if d_vraie_20.survives else 'ne survit pas'}")

print()
print("  Et la question qu'il faut poser à quiconque montre un backtest :")
print("  combien de temps faudrait-il pour y croire ?")
print()
for sr_vise in (0.5, 1.0, 1.5, 2.0):
    brut = rng.normal(0.0, 0.01, 3_000)
    # On recentre pour que le Sharpe RÉALISÉ vaille exactement la cible :
    # sinon chaque ligne du tableau mesurerait un Sharpe légèrement
    # différent de son étiquette, et la comparaison ne voudrait rien dire.
    serie = pd.Series(
        brut - brut.mean() + sr_vise * brut.std(ddof=1) / np.sqrt(JOURS)
    )
    jours = minimum_track_record_length(serie)
    print(f"    Sharpe annualisé {sr_vise:.1f} → il faut "
          f"{jours:>7.0f} jours, soit {jours / JOURS:>4.1f} ans "
          f"d'historique")

print()
print("  → C'est la réponse chiffrée à « ma stratégie fait 1,5 depuis six")
print("    mois ». Ce n'est pas « je n'y crois pas », c'est « il faudrait")
print("    tant d'années pour que j'aie le droit d'y croire ».")

print()
print("═" * LARGE)
print("  Fin de la séance 8.   python -m pytest tests/ -q -m \"\"")
print("═" * LARGE)
