"""
Le rapport de validation : faire parler les instruments d'une seule voix.

CE QUE CE MODULE N'APPORTE PAS, ET IL FAUT LE DIRE EN PREMIER
-------------------------------------------------------------
Aucune mesure nouvelle. Pas un estimateur, pas un test, pas une
statistique. Tout ce qu'il calcule a été construit dans les treize
séances précédentes.

Sa contribution est ailleurs : jusqu'ici le moteur savait répondre à
treize questions séparément, et n'avait **jamais formé une opinion**.
Un utilisateur devait lancer cinq scripts, lire cinq sorties, et
réconcilier lui-même des verdicts qui se contredisent. Ce module fait
cette réconciliation, et surtout il refuse de la faire silencieusement :
quand deux instruments divergent, il le dit, et il dit ce que la
divergence SIGNIFIE.

LA RÈGLE DE CONJONCTION, ET POURQUOI ELLE EST SÉVÈRE
-----------------------------------------------------
Le verdict global d'un modèle est **le pire** des verdicts rendus par
les instruments qui s'appliquent à lui. Pas une moyenne, pas un vote.

La raison est asymétrique et vaut d'être énoncée : un instrument qui
valide dit « je n'ai pas trouvé de défaut dans ce que je sais
regarder ». Un instrument qui rejette dit « j'ai trouvé un défaut ».
Une absence de preuve ne compense pas une preuve. Trois validations
n'annulent pas un rejet — elles disent seulement que le défaut est hors
du champ de trois instruments sur quatre.

Ce qui rend le verdict ACTIONNABLE n'est donc pas le verdict lui-même
mais `binding_instrument` : lequel des instruments contraint. C'est lui
qui dit quoi réparer.

LES QUATRE FAMILLES DE CONFLITS
-------------------------------
Un conflit n'est pas une erreur du moteur. C'est une information, et
chaque famille a une interprétation précise.

**CORPS / QUEUE.** L'inventaire valide le modèle, le backtest d'ES le
rejette. Ce n'est pas une contradiction : le modèle décrit correctement
les jours normaux et se trompe sur ceux qui comptent. Un modèle de
risque dont le corps est juste et la queue fausse est précisément le
modèle qui rassure jusqu'au krach.

**NIVEAU / STABILITÉ.** Le biais vaut 1,00 et la dispersion est élevée.
Le modèle est juste en moyenne parce que ses erreurs se compensent, et
il n'a eu raison aucune période. Découvert à la séance 12 ; ici il est
levé automatiquement.

**PROTOCOLE.** Le verdict change selon qu'on mesure sur un portefeuille
fixe ou sur celui que le modèle recommande. Le défaut n'apparaît qu'à
l'inversion de la matrice. C'était le résultat central de la séance 12 ;
ce module le cherche désormais systématiquement.

**PÉRIMÈTRE.** Celui-ci n'est pas un désaccord entre deux verdicts :
c'est une dépendance qui fragilise tous les autres. Le module de
qualité a signalé des anomalies sur des actifs que les autres
instruments ont ensuite utilisés sans restriction. Chaque verdict du
rapport est donc conditionnel à des données que le moteur lui-même tient
pour suspectes. D'où sa place : **en premier**, avant tout le reste.

**RÉGIME.** Pas une famille de plus mais une règle sur les quatre
autres : quand un conflit se déclenche pour TOUS les modèles jugés, il
ne parle plus des modèles, il parle des données. Quatre « ce modèle est
instable » ne sont pas quatre constats, c'est un seul : le régime est
instable, et aucun modèle à fenêtre fixe n'y tient. Le rapport replie
alors les quatre en un, parce qu'un conflit qui vise tout le monde ne
désigne personne — et un rapport qui crie à chaque ligne ne se lit plus.

CE QUE LE RAPPORT NE SAIT PAS FAIRE
-----------------------------------
Il ne cherche que les conflits qu'on lui a appris à reconnaître. Deux
instruments peuvent diverger d'une façon qui n'est dans aucune des
quatre familles, et le rapport ne le verra pas — exactement comme un
test ne voit que ce que sa statistique regarde (séances 12 et 13). Cette
limite est écrite dans chaque rapport produit, et non seulement ici.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import MinimumVariance, detect_lookahead, walk_forward
from .covariance import ledoit_wolf, sample_covariance
from .data.quality import QualityReport, inspect_prices
from .factors import InsufficientStructure
from .shortfall import (
    LEVEL_FRTB,
    EmpiricalNull,
    TailForecast,
    acerbi_szekely,
)
from .types import InsufficientData
from .validation import (
    INAPPLICABLE,
    REJETE,
    SURVEILLANCE,
    VALIDE,
    CovarianceModel,
    EwmaModel,
    FactorCovariance,
    LedoitWolfModel,
    ModelInventory,
    SampleModel,
    build_inventory,
)

TRADING_DAYS_PER_YEAR = 252

#: Ordre de sévérité. Le verdict global est le MAXIMUM de cet indice
#: sur les instruments applicables — la conjonction, pas la moyenne.
SEVERITE = {
    VALIDE: 0,
    SURVEILLANCE: 1,
    REJETE: 2,
    INAPPLICABLE: 3,
}

#: Au-delà de ce rapport, la calibration est juste en moyenne mais
#: instable période par période. Le seuil est une convention, donc un
#: paramètre, et il est déclaré dans chaque rapport.
DISPERSION_MAX = 1.50

#: Écart MATÉRIEL sur le rapport ES/VaR. Un écart statistiquement réel
#: mais inférieur à ce seuil est détecté sans être rejeté — c'est la
#: proportionnalité d'E-23, déjà appliquée à la statistique de biais.
SHAPE_TOLERANCE = 0.05

#: Nombre de tirages pour la loi du rapport de queue EMPIRIQUE sous
#: l'hypothèse gaussienne. On ne compare pas l'estimation à la valeur
#: théorique — voir `_verdict_queue` — mais à la loi du même estimateur
#: appliqué à des données qui vérifient l'hypothèse nulle.
SHAPE_BOOTSTRAP = 600

CORPS_QUEUE = "CORPS / QUEUE"
NIVEAU_STABILITE = "NIVEAU / STABILITÉ"
PROTOCOLE = "PROTOCOLE"
PERIMETRE = "PÉRIMÈTRE"
REGIME = "RÉGIME"


# ══════════════════════════════════════════════════════════════════════
#  Les pièces
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Finding:
    """Un constat d'un instrument sur un sujet."""

    instrument: str
    subject: str
    verdict: str
    detail: str

    @property
    def severity(self) -> int:
        return SEVERITE.get(self.verdict, 0)

    def __str__(self) -> str:
        return f"[{self.instrument}] {self.subject} → {self.verdict} · {self.detail}"


@dataclass(frozen=True)
class Conflict:
    """
    Un désaccord entre instruments, avec ce qu'il signifie.

    `interpretation` n'est pas décoratif. « Deux instruments
    divergent » n'est pas exploitable ; « le modèle décrit les jours
    normaux et se trompe sur ceux qui comptent » l'est.
    """

    family: str
    subject: str
    instruments: tuple[str, ...]
    verdicts: tuple[str, ...]
    interpretation: str

    def __str__(self) -> str:
        paires = ", ".join(
            f"{i} dit {v}"
            for i, v in zip(self.instruments, self.verdicts, strict=False)
        )
        return (
            f"⚠ {self.family} — {self.subject} : {paires}\n"
            f"   → {self.interpretation}"
        )


@dataclass(frozen=True)
class ValidationReport:
    """
    Le verdict réconcilié du moteur sur lui-même.

    `findings` porte les constats bruts, `conflicts` les désaccords
    interprétés, `data_caveats` ce qui conditionne tout le reste.
    """

    universe: str
    window: tuple[str, str]
    n_obs: int
    n_assets: int
    findings: tuple[Finding, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    data_caveats: tuple[str, ...] = ()
    causality: str = ""
    concentration: str = ""
    alpha: float = LEVEL_FRTB
    dispersion_max: float = DISPERSION_MAX
    generated: str = field(default_factory=lambda: date.today().isoformat())

    # ---- lecture ----

    @property
    def subjects(self) -> tuple[str, ...]:
        """Les modèles jugés, dans l'ordre d'apparition."""
        vus: list[str] = []
        for f in self.findings:
            if f.subject not in vus:
                vus.append(f.subject)
        return tuple(vus)

    def findings_for(self, subject: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.subject == subject)

    def verdict_for(self, subject: str) -> str:
        """
        LA RÈGLE DE CONJONCTION : le pire verdict l'emporte.

        Une absence de preuve ne compense pas une preuve.
        """
        constats = self.findings_for(subject)
        if not constats:
            return INAPPLICABLE
        return max(constats, key=lambda f: f.severity).verdict

    def binding_instrument(self, subject: str) -> str:
        """
        Lequel des instruments contraint le verdict — donc quoi réparer.

        C'est l'information actionnable du rapport. Le verdict dit s'il
        y a un problème ; celui-ci dit lequel.
        """
        constats = self.findings_for(subject)
        if not constats:
            return "aucun"
        return max(constats, key=lambda f: f.severity).instrument

    @property
    def usable_models(self) -> tuple[str, ...]:
        return tuple(
            s for s in self.subjects if self.verdict_for(s) in (VALIDE, SURVEILLANCE)
        )

    @property
    def conditional(self) -> bool:
        """
        Vrai si des anomalies de données conditionnent tous les verdicts.

        Un rapport conditionnel n'est pas un rapport faux : c'est un
        rapport dont la validité dépend d'une hypothèse qu'il déclare.
        """
        return any(c.family == PERIMETRE for c in self.conflicts)

    def table(self) -> pd.DataFrame:
        lignes = []
        for s in self.subjects:
            ligne: dict[str, object] = {"modèle": s}
            for f in self.findings_for(s):
                ligne[f.instrument] = f.verdict
            ligne["VERDICT"] = self.verdict_for(s)
            ligne["contraint par"] = self.binding_instrument(s)
            lignes.append(ligne)
        return pd.DataFrame(lignes)

    def __str__(self) -> str:
        titre = f"RAPPORT DE VALIDATION — {self.universe}"
        lignes = [titre, "=" * len(titre), ""]
        lignes.append(
            f"{self.n_assets} actifs · {self.n_obs} observations · "
            f"{self.window[0]} → {self.window[1]}"
        )
        lignes.append("")

        if self.data_caveats:
            lignes.append("DONNÉES — ce qui conditionne tout le reste")
            lignes.append("-" * 46)
            for avertissement in self.data_caveats:
                lignes.append(f"  ⚠ {avertissement}")
            lignes.append("")

        conflits_perimetre = [c for c in self.conflicts if c.family == PERIMETRE]
        autres = [c for c in self.conflicts if c.family != PERIMETRE]

        for conflit in conflits_perimetre + autres:
            lignes.append(str(conflit))
            lignes.append("")

        lignes.append("VERDICTS")
        lignes.append("-" * 46)
        table = self.table()
        if not table.empty:
            lignes.append(table.to_string(index=False))
        lignes.append("")

        if self.causality:
            lignes.append(f"CAUSALITÉ : {self.causality}")
        if self.concentration:
            lignes.append(f"CONCENTRATION : {self.concentration}")
        lignes.append("")
        lignes.append(
            "Ce rapport ne cherche que les conflits qu'on lui a appris à "
            "reconnaître. Deux instruments peuvent diverger d'une façon "
            "qu'il ne voit pas."
        )
        return "\n".join(lignes)

    # ---- sortie HTML ----

    def to_html(self) -> str:
        """
        Un fichier autonome, lisible en soixante secondes, sans dépendance.

        Le choix d'un fichier sans CSS externe ni script n'est pas de la
        coquetterie : un rapport de validation doit pouvoir être joint à
        un courriel, archivé, et rouvert dans dix ans. Tout ce qui est
        chargé depuis ailleurs aura disparu.
        """
        from .rendering import render_html

        # Import local : il casse le cycle entre le domaine et sa
        # présentation. `rendering` a besoin des constantes de ce
        # module ; ce module n'a besoin de `rendering` qu'au moment
        # d'écrire un fichier, donc jamais à l'import.
        return render_html(self)


# ══════════════════════════════════════════════════════════════════════
#  La construction
# ══════════════════════════════════════════════════════════════════════


def _ratio_gaussien(alpha: float) -> float:
    """ES/VaR THÉORIQUE d'une gaussienne au niveau de queue alpha."""
    q = float(stats.norm.ppf(alpha))
    es = float(stats.norm.expect(lambda x: -x, lb=-np.inf, ub=q)) / alpha
    return es / (-q)


def _ratio_empirique(u: np.ndarray, alpha: float) -> float:
    """ES/VaR estimé sur un échantillon, exactement comme `EmpiricalNull`."""
    seuil = float(np.quantile(u, alpha))
    if seuil >= 0.0:
        return math.nan
    queue = u[u <= seuil]
    if queue.size == 0:
        return math.nan
    return float(-queue.mean() / (-seuil))


def _loi_du_ratio_sous_h0(
    n: int, alpha: float, n_sim: int, seed: int = 0
) -> np.ndarray:
    """
    La loi de l'ESTIMATEUR du rapport de queue sous normalité, à n fixé.

    ⚠️ POURQUOI ON NE COMPARE PAS À LA VALEUR THÉORIQUE

    Ma première version comparait le rapport réalisé au 1,193 théorique
    de la gaussienne, avec une erreur-type obtenue par bootstrap. Elle
    déclarait « queue trop mince » sur des données parfaitement
    gaussiennes.

    J'ai d'abord attribué cela à un biais de petit échantillon de
    l'estimateur. **La mesure m'a contredit** : à n = 1 000, la médiane
    de l'estimateur vaut 1,194 pour une théorie à 1,193 — il est
    pratiquement sans biais. Le vrai coupable était la LARGEUR. La loi
    de l'estimateur à cette taille s'étale sur [1,13 ; 1,27], parce que
    le rapport est calculé sur les quelques dizaines de points qui
    dépassent le quantile ; l'erreur-type du bootstrap, elle, donnait un
    intervalle deux fois plus étroit.

    La correction est celle que ce projet applique depuis la séance 1 :
    on ne compare pas un estimateur à une vérité, on le compare à la loi
    du MÊME estimateur appliqué à des données qui vérifient
    l'hypothèse. Taux de déclenchement mesuré sur des données correctes,
    structure de rejeu par blocs comprise : 4,8 % pour un niveau nominal
    de 5 % (voir `test_le_test_de_forme_ne_se_declenche_pas_a_tort`).
    """
    rng = np.random.default_rng(seed)
    tirages = rng.standard_normal((n_sim, n))
    seuils = np.quantile(tirages, alpha, axis=1, keepdims=True)
    masque = tirages <= seuils
    sommes = (tirages * masque).sum(axis=1)
    comptes = masque.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        es = -sommes / np.maximum(comptes, 1)
        ratios = es / (-seuils.ravel())
    return np.asarray(ratios[np.isfinite(ratios)], dtype=float)


@dataclass(frozen=True)
class _Queue:
    """Les deux composantes du verdict de queue : la FORME et le NIVEAU."""

    verdict: str
    detail: str
    shape_announced: float = math.nan
    shape_realised: float = math.nan
    shape_verdict: str = ""
    level_verdict: str = ""


def _verdict_queue(
    returns: pd.DataFrame,
    modele: CovarianceModel,
    *,
    lookback: int,
    step: int,
    alpha: float,
    n_sim: int,
    shape_tolerance: float,
) -> _Queue:
    """
    Verdict du backtest d'ES, en DEUX composantes séparées.

    ⚠️ POURQUOI DEUX, ET C'EST UN DÉFAUT DE MA PREMIÈRE VERSION

    Ma première écriture passait les résidus standardisés du modèle à
    `EmpiricalNull` et coupait la vérification de cohérence. Résultat :
    l'hypothèse nulle était la forme de queue RÉALISÉE, donc le test ne
    pouvait pas rejeter une forme de queue mal annoncée — il l'avait
    absorbée dans sa propre nulle. Il validait tout.

    C'est le même piège que les séances 12 et 13 ont documenté : un test
    ne voit que ce que sa statistique regarde, et j'avais construit une
    statistique aveugle à la moitié de la question.

    La version correcte sépare :

    **LA FORME** — le modèle annonce un rapport ES/VaR gaussien
    (environ 1,19 à 2,5 %) ; les résidus réalisés en ont un autre. Si
    les deux divergent, le modèle se trompe sur l'épaisseur de sa queue,
    et le garde-fou de la séance 13 le détecte sans aucune simulation.

    **LE NIVEAU** — Z₂ sous la nulle empirique, qui teste la fréquence
    et la profondeur des dépassements à forme donnée.

    Le verdict retourné est la conjonction des deux, pour la même raison
    que partout dans ce module : une absence de preuve ne compense pas
    une preuve.
    """
    n_periodes, n_actifs = returns.shape
    w = np.full(n_actifs, 1.0 / n_actifs)
    rendements = returns.to_numpy() @ w

    q = float(stats.norm.ppf(alpha))
    ratio_annonce = _ratio_gaussien(alpha)

    realises: list[float] = []
    vars_: list[float] = []
    ess: list[float] = []
    sigmas: list[float] = []

    for t in range(lookback, n_periodes, step):
        historique = returns.iloc[t - lookback : t]  # ← t EXCLU. Causalité.
        try:
            est = modele.estimate(historique)
        except (InsufficientData, InsufficientStructure, np.linalg.LinAlgError) as exc:
            return _Queue(
                INAPPLICABLE,
                f"le modèle refuse de s'estimer : {type(exc).__name__}",
            )

        variance = float(w @ est.matrix @ w)
        if variance <= 0.0:
            return _Queue(INAPPLICABLE, "variance de portefeuille non positive")
        sigma = math.sqrt(variance)

        fin = min(t + step, n_periodes)
        for idx in range(t, fin):
            realises.append(float(rendements[idx]))
            vars_.append(sigma * (-q))
            ess.append(sigma * (-q) * ratio_annonce)
            sigmas.append(sigma)

    if not realises:
        return _Queue(INAPPLICABLE, "aucune prévision produite")

    x = np.asarray(realises)
    residus = x / np.asarray(sigmas)

    try:
        nulle = EmpiricalNull(residus)
        ratio_reel = nulle.ratio(alpha)
    except (InsufficientData, ValueError) as exc:
        return _Queue(INAPPLICABLE, f"nulle empirique impossible : {exc}")

    # --- composante 1 : la FORME ---
    #
    # On compare le rapport réalisé à la LOI DU MÊME ESTIMATEUR sous
    # normalité, à taille d'échantillon égale — pas à la valeur
    # théorique. Voir `_loi_du_ratio_sous_h0` : comparer à la théorie
    # avec un intervalle trop étroit faisait rejeter des données
    # gaussiennes, et c'était ma première version.
    loi_h0 = _loi_du_ratio_sous_h0(residus.size, alpha, SHAPE_BOOTSTRAP)
    if loi_h0.size < 50:
        return _Queue(INAPPLICABLE, "loi de référence du rapport non calculable")

    bas, haut = (float(x) for x in np.quantile(loi_h0, (0.025, 0.975)))
    attendu_a_cette_taille = float(np.median(loi_h0))
    ecart = ratio_reel - attendu_a_cette_taille

    significatif = not (bas <= ratio_reel <= haut)
    materiel = abs(ecart) > shape_tolerance

    if not significatif:
        v_forme = VALIDE
    elif not materiel:
        v_forme = SURVEILLANCE
    elif ecart > 0.0:
        v_forme = REJETE  # queue réalisée plus épaisse : le sens dangereux
    else:
        v_forme = SURVEILLANCE  # queue annoncée plus épaisse : prudent, coûteux

    # --- composante 2 : le NIVEAU ---
    try:
        prev = TailForecast(
            var=np.asarray(vars_),
            es=np.asarray(ess),
            alpha=alpha,
            method=f"{modele.name}, forme gaussienne annoncée",
        )
        resultat = acerbi_szekely(
            x, prev, null=nulle, n_sim=n_sim, check_ratio=False
        )
    except (InsufficientData, ValueError) as exc:
        return _Queue(INAPPLICABLE, str(exc))

    v_niveau = REJETE if resultat.rejected else VALIDE

    verdict = max((v_forme, v_niveau), key=lambda v: SEVERITE.get(v, 0))
    detail = (
        f"forme {ratio_reel:.3f} contre [{bas:.3f}—{haut:.3f}] attendu à "
        f"n={residus.size} ({v_forme}) · niveau Z₂ = "
        f"{resultat.statistic:+.3f} "
        f"vs {resultat.critical_value:+.3f} ({v_niveau}) · "
        f"{resultat.n_breaches} dépassements pour "
        f"{resultat.expected_breaches:.0f} attendus"
    )
    return _Queue(
        verdict=verdict,
        detail=detail,
        shape_announced=attendu_a_cette_taille,
        shape_realised=ratio_reel,
        shape_verdict=v_forme,
        level_verdict=v_niveau,
    )


def build_report(
    returns: pd.DataFrame,
    *,
    prices: dict[str, pd.Series] | None = None,
    universe: str = "",
    models: tuple[CovarianceModel, ...] | None = None,
    lookback: int = TRADING_DAYS_PER_YEAR,
    step: int = 21,
    alpha: float = LEVEL_FRTB,
    dispersion_max: float = DISPERSION_MAX,
    shape_tolerance: float = SHAPE_TOLERANCE,
    n_sim: int = 4000,
    check_causality: bool = True,
) -> ValidationReport:
    """
    Fait tourner tous les instruments sur les mêmes données et réconcilie.

    `prices` est facultatif : fourni, il déclenche l'inspection de
    qualité de la séance 9, qui conditionne tous les autres verdicts.
    Absent, le rapport le dit plutôt que de laisser croire que les
    données ont été vérifiées.

    `check_causality` peut être coupé : l'audit rejoue le backtest
    plusieurs fois et coûte cher. Quand il est coupé, le rapport
    l'écrit, parce qu'un rapport qui tait ce qu'il n'a pas vérifié est
    pire qu'un rapport incomplet.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns doit être un DataFrame, reçu {type(returns)}")
    if returns.shape[1] < 2:
        raise ValueError(
            f"il faut au moins deux actifs, reçu {returns.shape[1]}"
        )
    if len(returns) <= lookback:
        raise InsufficientData(
            n_obs=len(returns),
            n_required=lookback + step,
            what="rapport de validation (rien ne reste après la première fenêtre)",
        )

    if models is None:
        models = (SampleModel(), LedoitWolfModel(), EwmaModel(), FactorCovariance())

    index = returns.index
    fenetre = (
        str(index[0].date()) if isinstance(index, pd.DatetimeIndex) else str(index[0]),
        str(index[-1].date())
        if isinstance(index, pd.DatetimeIndex)
        else str(index[-1]),
    )

    findings: list[Finding] = []
    conflicts: list[Conflict] = []
    caveats: list[str] = []

    # ---- 1. Qualité des données (séance 9) : conditionne tout ----
    qualite: QualityReport | None = None
    if prices is not None:
        qualite = inspect_prices(prices)
        for a in qualite.blocking:
            caveats.append(f"{a.asset} — {a.kind} : {a.detail}")
        for a in qualite.warnings[:5]:
            caveats.append(f"{a.asset} — {a.kind} : {a.detail}")
        if qualite.anomalies:
            actifs = sorted({a.asset for a in qualite.anomalies})
            conflicts.append(
                Conflict(
                    family=PERIMETRE,
                    subject=f"{len(actifs)} actif(s) : {', '.join(actifs[:6])}"
                    + (" …" if len(actifs) > 6 else ""),
                    instruments=("qualité", "tous les autres"),
                    verdicts=(
                        f"{len(qualite.anomalies)} anomalie(s)",
                        "ont utilisé ces données",
                    ),
                    interpretation=(
                        "Tous les verdicts de ce rapport sont CONDITIONNELS à "
                        "des données que le moteur lui-même tient pour "
                        "suspectes. Les instruments en aval n'ont pas été "
                        "restreints, et ne savent pas qu'ils auraient dû "
                        "l'être."
                    ),
                )
            )
    else:
        caveats.append(
            "aucune série de prix fournie : la qualité des données n'a PAS "
            "été vérifiée, et les verdicts ci-dessous la supposent bonne"
        )

    # ---- 2. Inventaire de covariance, deux protocoles (séance 12) ----
    inv_fixe: ModelInventory = build_inventory(
        returns, list(models), universe=universe, lookback=lookback, step=step
    )
    inv_modele: ModelInventory = build_inventory(
        returns,
        list(models),
        universe=universe,
        lookback=lookback,
        step=step,
        portfolio="model",
    )

    for carte in inv_fixe.cards:
        detail = (
            f"biais {carte.bias_value:.3f}"
            if carte.bias is not None
            else carte.reason or "non mesurable"
        )
        findings.append(
            Finding("inventaire", carte.model, carte.verdict, detail)
        )

        if carte.bias is not None and carte.bias.dispersion_ratio > dispersion_max:
            conflicts.append(
                Conflict(
                    family=NIVEAU_STABILITE,
                    subject=carte.model,
                    instruments=("niveau", "stabilité"),
                    verdicts=(
                        f"biais {carte.bias_value:.3f}",
                        f"dispersion {carte.bias.dispersion_ratio:.2f} × le hasard",
                    ),
                    interpretation=(
                        "Le modèle est juste EN MOYENNE parce que ses erreurs "
                        "se compensent sur la fenêtre. Il n'a eu raison "
                        "aucune période prise isolément."
                    ),
                )
            )

    par_nom = {c.model: c for c in inv_modele.cards}
    for carte in inv_fixe.cards:
        autre = par_nom.get(carte.model)
        if autre is None:
            continue
        findings.append(
            Finding(
                "protocole modèle",
                carte.model,
                autre.verdict,
                f"biais {autre.bias_value:.3f} sur son propre portefeuille"
                if autre.bias is not None
                else autre.reason or "non mesurable",
            )
        )
        if SEVERITE.get(autre.verdict, 0) > SEVERITE.get(carte.verdict, 0):
            conflicts.append(
                Conflict(
                    family=PROTOCOLE,
                    subject=carte.model,
                    instruments=("portefeuille fixe", "portefeuille du modèle"),
                    verdicts=(carte.verdict, autre.verdict),
                    interpretation=(
                        "Le défaut n'apparaît qu'à l'INVERSION de la matrice. "
                        "Un portefeuille fixe n'inverse rien, donc ce "
                        "protocole-là ne peut pas le voir. L'angle mort "
                        "appartient au protocole, pas au modèle."
                    ),
                )
            )

    # ---- 3. Backtest d'Expected Shortfall (séance 13) ----
    queues: dict[str, _Queue] = {}
    for modele in models:
        q = _verdict_queue(
            returns,
            modele,
            lookback=lookback,
            step=step,
            alpha=alpha,
            n_sim=n_sim,
            shape_tolerance=shape_tolerance,
        )
        queues[modele.name] = q
        findings.append(Finding("queue", modele.name, q.verdict, q.detail))

        reference = next(
            (c for c in inv_fixe.cards if c.model == modele.name), None
        )
        if reference is not None and SEVERITE.get(q.verdict, 0) > SEVERITE.get(
            reference.verdict, 0
        ):
            conflicts.append(
                Conflict(
                    family=CORPS_QUEUE,
                    subject=modele.name,
                    instruments=("inventaire", "queue"),
                    verdicts=(reference.verdict, q.verdict),
                    interpretation=(
                        "Le modèle décrit correctement les jours normaux et "
                        "se trompe sur ceux qui comptent. Ce n'est pas une "
                        "contradiction mais une décomposition — et c'est le "
                        "profil exact d'un modèle qui rassure jusqu'au krach."
                    ),
                )
            )

    # ---- 4. Audit de causalité (séance 8) ----
    causalite = ""
    if check_causality:
        try:
            audit = detect_lookahead(
                returns,
                lambda _: MinimumVariance(estimator=ledoit_wolf),
                lookback=lookback,
                step=step,
                n_probes=3,
            )
            causalite = (
                f"✓ propre sur {len(audit.probes)} sondes "
                f"(écart max {audit.worst_difference:.1e})"
                if audit.clean
                else f"✗ FUITE DÉTECTÉE — écart {audit.worst_difference:.2e}"
            )
        except (InsufficientData, ValueError) as exc:
            causalite = f"non vérifiable : {exc}"
    else:
        causalite = "NON VÉRIFIÉE (audit désactivé pour cette exécution)"

    # ---- 5. Concentration de l'écart de calibration (séance 9) ----
    concentration = ""
    try:
        resultat = walk_forward(
            returns,
            MinimumVariance(estimator=sample_covariance),
            lookback=lookback,
            step=step,
            mode="rolling",
        )
        concentration = resultat.calibration_diagnosis()
    except (InsufficientData, ValueError, np.linalg.LinAlgError) as exc:
        concentration = f"non mesurable : {type(exc).__name__}"

    conflicts = _replier_les_conflits_universels(conflicts, len(models))

    return ValidationReport(
        universe=universe or "univers non nommé",
        window=fenetre,
        n_obs=len(returns),
        n_assets=returns.shape[1],
        findings=tuple(findings),
        conflicts=tuple(conflicts),
        data_caveats=tuple(caveats),
        causality=causalite,
        concentration=concentration,
        alpha=alpha,
        dispersion_max=dispersion_max,
    )


def _replier_les_conflits_universels(
    conflits: list[Conflict], n_modeles: int
) -> list[Conflict]:
    """
    Un conflit qui vise TOUS les modèles ne désigne aucun modèle.

    Il décrit alors une propriété des DONNÉES, et le dire quatre fois
    noie l'information au lieu de la porter. On replie donc la famille
    en un constat unique de régime, en gardant la liste des sujets.

    Le périmètre est exclu du repliement : il vise déjà l'ensemble par
    construction, et c'est précisément son objet.
    """
    if n_modeles < 2:
        return conflits

    par_famille: dict[str, list[Conflict]] = {}
    for c in conflits:
        if c.family != PERIMETRE:
            par_famille.setdefault(c.family, []).append(c)

    garde = [c for c in conflits if c.family == PERIMETRE]
    for famille, groupe in par_famille.items():
        sujets = {c.subject for c in groupe}
        if len(sujets) < n_modeles:
            garde.extend(groupe)
            continue
        garde.append(
            Conflict(
                family=REGIME,
                subject=f"les {len(sujets)} modèles, sans exception",
                instruments=(famille,),
                verdicts=(f"déclenché sur {len(sujets)}/{n_modeles} modèles",),
                interpretation=(
                    f"Le conflit « {famille} » vise TOUS les modèles jugés. "
                    "Il ne parle donc plus d'eux mais des données : c'est le "
                    "régime qui a cette propriété, et aucun modèle de cette "
                    "famille n'y tient. Les quatre constats ont été repliés "
                    "en un seul, parce qu'un conflit qui vise tout le monde "
                    "ne désigne personne."
                ),
            )
        )
    return garde
