"""
Qualité des données — et le choix de nettoyage que personne ne déclare.

LA PLACE DE CE MODULE DANS LE PROJET
====================================

Les huit premières séances ont porté sur l'incertitude d'ESTIMATION :
on dispose des bonnes données, et on ne sait pas exactement ce qu'elles
impliquent. Chaque fonction rend un intervalle de confiance qui mesure
précisément cette ignorance-là.

Il existe une seconde source d'erreur, antérieure à toute estimation, et
qu'aucun intervalle de confiance ne capture : **on ne dispose pas des
bonnes données.** Un prix figé parce que le titre ne s'échange pas. Un
calendrier qui ne coïncide pas avec celui du voisin. Un saut de 40 % qui
est soit un krach, soit un fractionnement d'actions mal corrigé.

Ces erreurs-là ne se voient pas dans les intervalles, parce qu'elles
changent la grandeur estimée au lieu de l'imprécision. Un titre figé
pendant six jours ne produit pas une volatilité imprécise : il produit
une volatilité FAUSSE, et parfaitement serrée.

LE RÉSULTAT CENTRAL DE CE MODULE
================================

Quand deux actifs ne cotent pas les mêmes jours — et c'est la règle dès
qu'on mélange deux places boursières — il faut choisir une politique
d'alignement. Deux sont défendables :

    intersection   on ne garde que les dates où TOUT LE MONDE cote.
                   On perd des observations, on n'invente rien.

    report         on garde toutes les dates et on reporte le dernier
                   prix connu. On garde les observations, et on invente
                   un rendement de ZÉRO les jours reportés.

Ce choix se fait d'ordinaire en une ligne, sans commentaire, souvent
sans que l'auteur sache qu'il a choisi. `compare_alignment` mesure ce
qu'il coûte.

CE QUE J'AVAIS PRÉDIT, ET CE QUE LA MESURE A DIT
------------------------------------------------
J'avais écrit ici : « le report injecte des zéros, donc il réduit la
variance ET la corrélation ». La moitié de cette phrase était fausse.

Mesuré contre une vérité connue — vraie volatilité 20 %, vraie
corrélation 0,50, dix ans, en faisant varier la part de séances
manquantes :

    séances      intersection    intersection    report      report
    manquantes      ÷252          corrigée       volatilité  corrélation
         2 %       +2,09 %        −0,01 %        +0,01 %      −2,31 %
         5 %       +5,15 %        −0,14 %        +0,02 %      −5,67 %
        10 %      +10,98 %        −0,02 %        +0,49 %     −10,54 %
        20 %      +24,41 %        −0,12 %        +2,01 %     −19,13 %

Deux lois, toutes deux linéaires en la part de séances manquantes, et
aucune des deux n'est celle que j'attendais.

**1. Le report ne biaise PAS la volatilité. Il biaise la corrélation.**
Les zéros injectés tirent la variance vers le bas, mais la séance
suivante absorbe le mouvement manqué et la tire vers le haut : les deux
effets se compensent presque exactement. Rien de tel pour la
corrélation — quand un actif est reporté pendant que l'autre bouge, le
co-mouvement est détruit et rien ne le restitue. Le biais y est donc
entier, et il vaut à peu près la part de séances manquantes, au signe
près.

C'est le pire endroit où se tromper. La corrélation est ce qui
gouverne la diversification. Au niveau du portefeuille équipondéré, la
volatilité sort sous-estimée de 0,9 % à 10 % de séances manquantes, et
de 5,9 % à 20 % — toujours dans le sens optimiste.

**2. Le biais de l'intersection n'est pas un biais d'alignement, c'est
une erreur de CONVENTION.** L'intersection jette des séances, donc une
année ne contient plus 252 observations. Annualiser quand même par
√252 surestime la volatilité d'exactement la part jetée. En annualisant
par le nombre d'observations réellement présentes — ce que calcule
`observations_per_year` — le biais disparaît : ±0,14 % dans tous les
régimes testés.

C'est le bug de la séance 1, à l'identique : comparer deux grandeurs
annualisées selon deux conventions différentes. Il a fallu huit séances
pour qu'il revienne sous un autre déguisement.

CE QUE CE MODULE NE FAIT JAMAIS
===============================
Il ne répare rien. Il décrit, il classe en bloquant ou en avertissement,
et il laisse décider. Une réparation silencieuse est un mensonge
différé : l'utilisateur ne saura jamais que la série qu'il analyse n'est
pas celle qu'il croit avoir chargée.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

BLOQUANT = "bloquant"
AVERTISSEMENT = "avertissement"


@dataclass(frozen=True)
class Anomaly:
    """Une anomalie constatée. Jamais corrigée, toujours nommée."""

    kind: str
    asset: str
    severity: str
    count: int
    detail: str
    first: pd.Timestamp | None = None

    def __str__(self) -> str:
        marque = "✗" if self.severity == BLOQUANT else "⚠"
        date = f"  dès {self.first.date()}" if self.first is not None else ""
        return f"  {marque} {self.asset:<12} {self.kind:<22} {self.detail}{date}"


@dataclass(frozen=True)
class QualityReport:
    """Verdict sur un jeu de séries de prix."""

    anomalies: tuple[Anomaly, ...]
    n_assets: int
    n_dates_union: int
    n_dates_intersection: int

    @property
    def blocking(self) -> tuple[Anomaly, ...]:
        return tuple(a for a in self.anomalies if a.severity == BLOQUANT)

    @property
    def warnings(self) -> tuple[Anomaly, ...]:
        return tuple(a for a in self.anomalies if a.severity == AVERTISSEMENT)

    @property
    def usable(self) -> bool:
        return not self.blocking

    @property
    def calendar_overlap(self) -> float:
        """Part des dates de l'union où TOUS les actifs cotent."""
        if self.n_dates_union == 0:
            return 0.0
        return self.n_dates_intersection / self.n_dates_union

    def raise_if_blocking(self) -> None:
        """
        Échoue bruyamment plutôt que de laisser passer.

        Même principe que `InsufficientData` à la séance 1 : une donnée
        inutilisable doit ARRÊTER le programme, pas se propager jusqu'à
        s'afficher comme un détail dans un tableau.
        """
        if self.blocking:
            details = "\n".join(str(a) for a in self.blocking)
            raise ValueError(
                f"{len(self.blocking)} anomalie(s) bloquante(s) :\n{details}"
            )

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "actif": a.asset,
                    "type": a.kind,
                    "gravité": a.severity,
                    "occurrences": a.count,
                    "détail": a.detail,
                }
                for a in self.anomalies
            ]
        )

    def __str__(self) -> str:
        lignes = [
            f"{self.n_assets} actifs  |  "
            f"{self.n_dates_union} dates au total, "
            f"{self.n_dates_intersection} communes à tous "
            f"({self.calendar_overlap:.1%})"
        ]
        if not self.anomalies:
            lignes.append("  ✓ aucune anomalie détectée")
        else:
            lignes.extend(str(a) for a in self.anomalies)
            lignes.append(
                f"  → {len(self.blocking)} bloquante(s), "
                f"{len(self.warnings)} avertissement(s)"
            )
        return "\n".join(lignes)


# ══════════════════════════════════════════════════════════════════════
#  INSPECTION
# ══════════════════════════════════════════════════════════════════════


def inspect_prices(
    series: Mapping[str, pd.Series],
    min_stale_run: int = 4,
    jump_threshold: float = 0.25,
    min_points: int = 60,
) -> QualityReport:
    """
    Passe un jeu de séries de prix au crible, sans rien modifier.

    `min_stale_run` : nombre de prix identiques consécutifs à partir
        duquel on signale. 4 jours d'affilée sans le moindre mouvement
        n'arrive pas par hasard sur un titre liquide.
    `jump_threshold` : variation en valeur absolue au-delà de laquelle
        on signale un saut. 25 % en une séance.
    `min_points` : en deçà, on avertit que l'historique est trop court
        pour estimer quoi que ce soit (séance 1 : la volatilité demande
        environ trois mois pour passer le seuil de fiabilité).

    CE QUI EST BLOQUANT, ET CE QUI NE L'EST PAS
    -------------------------------------------
    Bloquant : prix négatif ou nul, dates en double. Ces cas-là ne
    correspondent à aucune réalité de marché — ce sont des défauts de
    fichier, et continuer n'a aucun sens.

    Avertissement : tout le reste. Un prix figé pendant six jours peut
    être une erreur de fichier ou un titre réellement illiquide. Un
    saut de 40 % peut être un krach ou un fractionnement mal corrigé.
    **La série seule ne permet pas de trancher**, et un outil qui
    trancherait à la place de l'utilisateur se tromperait un jour sur
    deux sans jamais le dire.

    C'est la raison pour laquelle rien n'est corrigé ici.
    """
    anomalies: list[Anomaly] = []
    index_par_actif: dict[str, pd.DatetimeIndex] = {}

    for nom, s in series.items():
        if not isinstance(s, pd.Series):
            raise TypeError(f"'{nom}' n'est pas une pd.Series")
        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(f"'{nom}' n'est pas indexée par des dates")

        index_par_actif[nom] = pd.DatetimeIndex(s.index)
        anomalies.extend(
            _anomalies_une_serie(nom, s, min_stale_run, jump_threshold,
                                 min_points)
        )

    union = _union(index_par_actif)
    inter = _intersection(index_par_actif)

    # ⚠️ DEUX CAUSES TRÈS DIFFÉRENTES, QUE LA PREMIÈRE VERSION CONFONDAIT
    #
    # Sur le premier jeu de vraies données, le rapport annonçait « 7 %
    # de dates non communes » en mélangeant deux choses sans rapport :
    #
    #   • SHOP.TO est entré en Bourse en mai 2015, donc il n'existe pas
    #     avant — 96 dates ;
    #   • Toronto et New York n'ont pas les mêmes jours fériés — une
    #     centaine de dates réparties sur onze ans.
    #
    # Le premier cas n'est pas un désalignement de calendrier, c'est un
    # historique plus court. Et il coûte beaucoup plus cher : prendre
    # l'intersection jette alors les cinq premiers mois POUR LES DOUZE
    # titres, à cause d'un seul. L'utilisateur doit pouvoir décider
    # d'exclure ce titre plutôt que d'amputer tout l'univers, et il ne
    # peut pas décider ce qu'on ne lui montre pas.
    #
    # Les deux sont donc signalés séparément, avec leur coût propre.
    if index_par_actif and len(union) > 0:
        debuts = {n: idx.min() for n, idx in index_par_actif.items()}
        fins = {n: idx.max() for n, idx in index_par_actif.items()}
        debut_commun, fin_commune = max(debuts.values()), min(fins.values())

        for nom, debut in debuts.items():
            perdues = int((union < debut).sum()) + int((union > fins[nom]).sum())
            if perdues > 0:
                anomalies.append(
                    Anomaly(
                        kind="historique tronqué",
                        asset=nom,
                        severity=AVERTISSEMENT,
                        count=perdues,
                        detail=(
                            f"commence le {debut.date()} et finit le "
                            f"{fins[nom].date()} — en intersection, ce seul "
                            f"titre ampute {perdues} date(s) pour TOUT "
                            f"l'univers"
                        ),
                        first=pd.Timestamp(debut),
                    )
                )

        fenetre = union[(union >= debut_commun) & (union <= fin_commune)]
        manquantes = len(fenetre) - len(inter)
        if manquantes > 0:
            anomalies.append(
                Anomaly(
                    kind="calendriers désalignés",
                    asset="(univers)",
                    severity=AVERTISSEMENT,
                    count=manquantes,
                    detail=(
                        f"sur la fenêtre commune ({debut_commun.date()} → "
                        f"{fin_commune.date()}), {manquantes} date(s) sur "
                        f"{len(fenetre)} ne sont pas communes à tous "
                        f"({manquantes / len(fenetre):.1%}) — le choix "
                        f"d'alignement va changer tes chiffres"
                    ),
                )
            )

    return QualityReport(
        anomalies=tuple(anomalies),
        n_assets=len(series),
        n_dates_union=len(union),
        n_dates_intersection=len(inter),
    )


def _anomalies_une_serie(
    nom: str,
    s: pd.Series,
    min_stale_run: int,
    jump_threshold: float,
    min_points: int,
) -> list[Anomaly]:
    out: list[Anomaly] = []

    doublons = s.index.duplicated()
    if doublons.any():
        out.append(
            Anomaly(
                kind="dates en double",
                asset=nom,
                severity=BLOQUANT,
                count=int(doublons.sum()),
                detail=f"{int(doublons.sum())} date(s) répétée(s)",
                first=pd.Timestamp(s.index[doublons][0]),
            )
        )

    valeurs = s.to_numpy(dtype=float)
    non_positifs = valeurs <= 0
    if non_positifs.any():
        out.append(
            Anomaly(
                kind="prix non positif",
                asset=nom,
                severity=BLOQUANT,
                count=int(non_positifs.sum()),
                detail=f"{int(non_positifs.sum())} prix ≤ 0",
                first=pd.Timestamp(s.index[non_positifs][0]),
            )
        )

    if len(s) < min_points:
        out.append(
            Anomaly(
                kind="historique trop court",
                asset=nom,
                severity=AVERTISSEMENT,
                count=len(s),
                detail=(
                    f"{len(s)} points pour {min_points} requis — toute "
                    f"estimation sortira marquée PEU FIABLE"
                ),
            )
        )

    if not s.index.is_monotonic_increasing:
        out.append(
            Anomaly(
                kind="dates non triées",
                asset=nom,
                severity=BLOQUANT,
                count=1,
                detail="l'index n'est pas croissant",
            )
        )
        return out  # les détections suivantes n'auraient plus de sens

    # Les prix figés se détectent par simple égalité de valeurs
    # consécutives : aucun rendement n'intervient, donc la présence d'un
    # prix non positif ailleurs dans la série ne l'empêche pas.
    #
    # La première version court-circuitait les deux détections dès qu'un
    # prix ≤ 0 apparaissait. Un test l'a relevé : une série à la fois
    # figée et corrompue ne signalait que la corruption. Un rapport de
    # qualité doit dire TOUT ce qu'il a vu, pas s'arrêter au premier
    # défaut bloquant — sinon on corrige, on relance, et on découvre le
    # suivant.
    if len(valeurs) >= 2:
        debut, longueur = _plus_longue_serie_figee(valeurs)
        n_runs = _nombre_series_figees(valeurs, min_stale_run)
        if longueur >= min_stale_run:
            out.append(
                Anomaly(
                    kind="prix figés",
                    asset=nom,
                    severity=AVERTISSEMENT,
                    count=n_runs,
                    detail=(
                        f"{n_runs} épisode(s) de ≥ {min_stale_run} prix "
                        f"identiques, le plus long de {longueur} séances — "
                        f"la volatilité estimée sera trop basse"
                    ),
                    first=pd.Timestamp(s.index[debut]),
                )
            )

    # Les sauts, eux, SUPPOSENT des rendements définis. Avec un prix
    # nul ou négatif dans la série, les rapports n'ont plus de sens et
    # produiraient des signalements fantaisistes par-dessus un défaut
    # déjà bloquant.
    if len(valeurs) >= 2 and not non_positifs.any():
        rendements = valeurs[1:] / valeurs[:-1] - 1.0
        sauts = np.abs(rendements) > jump_threshold
        if sauts.any():
            pire = int(np.argmax(np.abs(rendements)))
            out.append(
                Anomaly(
                    kind="saut invraisemblable",
                    asset=nom,
                    severity=AVERTISSEMENT,
                    count=int(sauts.sum()),
                    detail=(
                        f"{int(sauts.sum())} variation(s) > "
                        f"{jump_threshold:.0%}, la pire à "
                        f"{rendements[pire]:+.1%} — krach réel ou "
                        f"fractionnement mal corrigé, la série seule ne "
                        f"permet pas de trancher"
                    ),
                    first=pd.Timestamp(s.index[1:][sauts][0]),
                )
            )

    return out


def _plus_longue_serie_figee(valeurs: np.ndarray) -> tuple[int, int]:
    """Position de départ et longueur du plus long palier de prix identiques."""
    if len(valeurs) == 0:
        return 0, 0
    identique = np.concatenate([[False], valeurs[1:] == valeurs[:-1]])
    meilleur_debut, meilleure_longueur = 0, 1
    debut, longueur = 0, 1
    for i in range(1, len(valeurs)):
        if identique[i]:
            longueur += 1
        else:
            debut, longueur = i, 1
        if longueur > meilleure_longueur:
            meilleure_longueur, meilleur_debut = longueur, debut
    return meilleur_debut, meilleure_longueur


def _nombre_series_figees(valeurs: np.ndarray, minimum: int) -> int:
    """Combien de paliers distincts atteignent la longueur minimale."""
    if len(valeurs) < minimum:
        return 0
    n, longueur = 0, 1
    for i in range(1, len(valeurs)):
        if valeurs[i] == valeurs[i - 1]:
            longueur += 1
            if longueur == minimum:
                n += 1
        else:
            longueur = 1
    return n


def _union(index: Mapping[str, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    if not index:
        return pd.DatetimeIndex([])
    out = next(iter(index.values()))
    for autre in index.values():
        out = out.union(autre)
    return pd.DatetimeIndex(out)


def _intersection(index: Mapping[str, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    if not index:
        return pd.DatetimeIndex([])
    out = next(iter(index.values()))
    for autre in index.values():
        out = out.intersection(autre)
    return pd.DatetimeIndex(out)


# ══════════════════════════════════════════════════════════════════════
#  ALIGNEMENT — LE CHOIX QUE PERSONNE NE DÉCLARE
# ══════════════════════════════════════════════════════════════════════

POLITIQUES = ("intersection", "report")


def align_prices(
    series: Mapping[str, pd.Series], policy: str = "intersection"
) -> pd.DataFrame:
    """
    Met plusieurs séries de prix sur un calendrier commun.

    `policy="intersection"` (défaut) : ne garde que les dates où tous
        les actifs cotent. Aucune valeur inventée, moins d'observations.

    `policy="report"` : garde l'union des dates et reporte le dernier
        prix connu. Toutes les observations conservées, et un rendement
        de ZÉRO inventé chaque jour reporté.

    POURQUOI L'INTERSECTION EST LE DÉFAUT
    -------------------------------------
    Parce que le report fabrique de l'information. Un rendement nul
    n'est pas une absence de donnée : c'est l'affirmation que le prix
    n'a pas bougé, et le moteur la traitera comme une observation à
    part entière.

    Mesuré contre une vérité connue (voir l'en-tête du module), cela ne
    fausse presque pas les volatilités individuelles — les zéros et les
    séances doubles se compensent — mais cela détruit du co-mouvement
    et sous-estime la corrélation à hauteur de la part de séances
    manquantes. La diversification apparente augmente, le risque du
    portefeuille baisse, **et c'est le sens dangereux de l'erreur**.

    L'intersection, elle, n'invente rien. Son seul piège est
    l'annualisation : elle laisse moins de 252 observations par an, et
    il faut l'annualiser par `observations_per_year` plutôt que par
    252. À cette condition, elle est non biaisée.

    Entre un estimateur un peu plus bruité et un estimateur optimiste,
    un moteur de risque prend le bruité.

    `compare_alignment` mesure l'écart sur TES données plutôt que de te
    demander de croire ce paragraphe.
    """
    if policy not in POLITIQUES:
        raise ValueError(
            f"politique inconnue : '{policy}'. Choix : {POLITIQUES}"
        )
    if not series:
        raise ValueError("aucune série à aligner")

    cadre = pd.DataFrame(dict(series)).sort_index()

    if policy == "intersection":
        aligne = cadre.dropna(how="any")
    else:
        aligne = cadre.ffill().dropna(how="any")

    if len(aligne) < 2:
        raise ValueError(
            f"l'alignement '{policy}' ne laisse que {len(aligne)} date(s) "
            f"exploitable(s). Les calendriers de ces actifs ne se "
            f"recoupent pratiquement pas."
        )
    return aligne


def observations_per_year(index: pd.DatetimeIndex) -> float:
    """
    Nombre d'observations réellement présentes par an dans cet index.

    LA FONCTION QUI CORRIGE LE BIAIS DE L'INTERSECTION
    ---------------------------------------------------
    Annualiser une volatilité, c'est multiplier par la racine du nombre
    d'observations par an. Tout le projet utilisait 252 — le nombre de
    séances d'une année boursière américaine.

    Dès qu'on aligne plusieurs calendriers par intersection, ce nombre
    cesse d'être vrai : on a jeté les séances non communes. Continuer à
    écrire 252 surestime la volatilité d'exactement la part jetée, soit
    jusqu'à +24 % dans les cas mesurés.

    Cette fonction le calcule à partir des dates réellement présentes,
    au lieu de le supposer. C'est une ligne de code, et elle efface un
    biais que personne ne cherche parce qu'il ne ressemble pas à un
    bug : le programme ne se plaint pas, et 252 a l'air d'une constante
    physique.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        raise ValueError("au moins deux dates sont nécessaires")
    annees = (idx[-1] - idx[0]).days / 365.25
    if annees <= 0:
        raise ValueError("l'index couvre une durée nulle")
    return float((len(idx) - 1) / annees)


@dataclass(frozen=True)
class AlignmentComparison:
    """Ce que le choix d'alignement change, chiffré."""

    volatilities: pd.DataFrame      # index : actif, colonnes : politique
    mean_correlation: dict[str, float]
    n_dates: dict[str, int]
    observations_per_year: dict[str, float]

    @property
    def volatility_shift(self) -> pd.Series:
        """Écart relatif de volatilité, report par rapport à intersection."""
        return (
            self.volatilities["report"] / self.volatilities["intersection"]
            - 1.0
        )

    def __str__(self) -> str:
        decalage = self.volatility_shift
        # Écart ABSOLU et non relatif : sur un univers peu corrélé, un
        # rapport entre deux corrélations proches de zéro produit des
        # pourcentages spectaculaires et vides de sens (0,010 → 0,003 se
        # lit « −66 % » alors que l'écart vaut sept millièmes).
        ecart_corr = (
            self.mean_correlation["report"]
            - self.mean_correlation["intersection"]
        )
        return (
            f"dates retenues       intersection "
            f"{self.n_dates['intersection']}, "
            f"report {self.n_dates['report']}\n"
            f"observations par an  intersection "
            f"{self.observations_per_year['intersection']:.1f}, "
            f"report {self.observations_per_year['report']:.1f}\n"
            f"corrélation moyenne  intersection "
            f"{self.mean_correlation['intersection']:.3f}, "
            f"report {self.mean_correlation['report']:.3f}  "
            f"({ecart_corr:+.3f})\n"
            f"volatilité           le report donne en moyenne "
            f"{decalage.mean():+.2%} "
            f"(de {decalage.min():+.2%} à {decalage.max():+.2%})"
        )


def compare_alignment(series: Mapping[str, pd.Series]) -> AlignmentComparison:
    """
    Mesure ce que le choix d'alignement coûte SUR TES DONNÉES.

    Deux politiques défendables, deux jeux de chiffres. L'écart n'est
    pas une curiosité : sur des univers multi-places, il dépasse
    couramment l'effet des raffinements de modélisation dont on débat
    par ailleurs.

    C'est le genre de sensibilité qu'un rapport de risque devrait
    porter et ne porte jamais. On débat du choix de l'estimateur de
    covariance pendant des heures, et on aligne les calendriers en une
    ligne sans commentaire.

    Il n'y a volontairement PAS de paramètre `periods_per_year` : chaque
    politique a son propre nombre d'observations par an, et le laisser
    choisir à l'appelant rouvrirait exactement le biais de convention
    que cette comparaison sert à mesurer.
    """
    resultats: dict[str, pd.Series] = {}
    correlations: dict[str, float] = {}
    comptes: dict[str, int] = {}
    annualisations: dict[str, float] = {}

    for politique in POLITIQUES:
        prix = align_prices(series, policy=politique)
        rendements = prix.pct_change().dropna()
        # ⚠️ Chaque politique a SON propre nombre d'observations par an.
        # Utiliser 252 pour les deux comparerait deux conventions
        # différentes et attribuerait à l'alignement un écart qui vient
        # de l'annualisation. C'est précisément le bug de la séance 1.
        par_an = observations_per_year(prix.index)
        annualisations[politique] = par_an
        resultats[politique] = rendements.std(ddof=1) * np.sqrt(par_an)
        comptes[politique] = len(prix)

        corr = rendements.corr().to_numpy()
        hors_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        correlations[politique] = float(np.mean(hors_diag))

    return AlignmentComparison(
        volatilities=pd.DataFrame(resultats),
        mean_correlation=correlations,
        n_dates=comptes,
        observations_per_year=annualisations,
    )
