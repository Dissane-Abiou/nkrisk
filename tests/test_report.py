"""
Tests du rapport de validation (séance 14).

Ce module ne mesure rien de neuf : il réconcilie. Les tests portent donc
sur la RÈGLE — la conjonction, la détection de conflits, le repliement —
et non sur des estimations.

Un test par famille de conflit, chacun sur un univers construit pour la
produire : une détection qu'on n'a jamais vue se déclencher ne vaut rien,
et c'est la leçon des quarante tests de sécurité de la séance 10.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import prix_depuis_rendements, univers_facteurs

from nkrisk.report import (
    CORPS_QUEUE,
    DISPERSION_MAX,
    NIVEAU_STABILITE,
    PERIMETRE,
    PROTOCOLE,
    REGIME,
    SEVERITE,
    Conflict,
    Finding,
    ValidationReport,
    _loi_du_ratio_sous_h0,
    _ratio_empirique,
    _ratio_gaussien,
    _replier_les_conflits_universels,
    build_report,
)
from nkrisk.shortfall import LEVEL_FRTB
from nkrisk.types import InsufficientData
from nkrisk.validation import INAPPLICABLE, REJETE, SURVEILLANCE, VALIDE

ALPHA = LEVEL_FRTB
T = 1100


def univers(
    n_actifs: int = 15,
    graine: int = 0,
    *,
    n_facteurs: int = 3,
    nu: int | None = None,
    garch: bool = False,
    n_periodes: int = T,
) -> pd.DataFrame:
    """Alias local sur le générateur partagé de `conftest.py`."""
    return univers_facteurs(
        n_actifs,
        graine,
        n_facteurs=n_facteurs,
        nu=nu,
        garch=garch,
        n_periodes=n_periodes,
    )


def rapport(donnees: pd.DataFrame, **kwargs) -> ValidationReport:
    kwargs.setdefault("n_sim", 600)
    kwargs.setdefault("check_causality", False)
    return build_report(donnees, **kwargs)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — la règle de conjonction
# ══════════════════════════════════════════════════════════════════════


def _rapport_artificiel(*verdicts: str) -> ValidationReport:
    return ValidationReport(
        universe="artificiel",
        window=("2020-01-01", "2024-01-01"),
        n_obs=1000,
        n_assets=10,
        findings=tuple(
            Finding(f"instrument{i}", "M", v, "") for i, v in enumerate(verdicts)
        ),
    )


@pytest.mark.parametrize(
    ("verdicts", "attendu"),
    [
        ((VALIDE, VALIDE, VALIDE), VALIDE),
        ((VALIDE, SURVEILLANCE, VALIDE), SURVEILLANCE),
        ((VALIDE, VALIDE, REJETE), REJETE),
        ((SURVEILLANCE, REJETE, VALIDE), REJETE),
        ((VALIDE, REJETE, INAPPLICABLE), INAPPLICABLE),
        ((VALIDE,), VALIDE),
    ],
)
def test_le_verdict_est_le_pire_des_verdicts(verdicts, attendu):
    """
    LA RÈGLE CENTRALE DU MODULE.

    Pas une moyenne, pas un vote : le maximum de sévérité. Une absence
    de preuve ne compense pas une preuve.
    """
    assert _rapport_artificiel(*verdicts).verdict_for("M") == attendu


def test_trois_validations_nannulent_pas_un_rejet():
    """L'énoncé en clair de la règle, pour qu'on ne l'assouplisse pas."""
    r = _rapport_artificiel(VALIDE, VALIDE, VALIDE, REJETE)
    assert r.verdict_for("M") == REJETE
    assert "M" not in r.usable_models


def test_linstrument_contraignant_est_celui_du_pire_verdict():
    """
    C'est l'information actionnable : le verdict dit qu'il y a un
    problème, celui-ci dit lequel réparer.
    """
    r = ValidationReport(
        universe="x",
        window=("a", "b"),
        n_obs=100,
        n_assets=5,
        findings=(
            Finding("inventaire", "M", VALIDE, ""),
            Finding("queue", "M", REJETE, ""),
            Finding("protocole modèle", "M", SURVEILLANCE, ""),
        ),
    )
    assert r.verdict_for("M") == REJETE
    assert r.binding_instrument("M") == "queue"


def test_un_modele_sans_constat_est_inapplicable():
    r = _rapport_artificiel()
    assert r.verdict_for("inconnu") == INAPPLICABLE
    assert r.binding_instrument("inconnu") == "aucun"


def test_lordre_de_severite_est_total_et_sans_surprise():
    assert SEVERITE[VALIDE] < SEVERITE[SURVEILLANCE] < SEVERITE[REJETE]
    assert SEVERITE[REJETE] < SEVERITE[INAPPLICABLE]


# ══════════════════════════════════════════════════════════════════════
#  Le repliement des conflits universels
# ══════════════════════════════════════════════════════════════════════


def _conflit(famille: str, sujet: str) -> Conflict:
    return Conflict(famille, sujet, ("a", "b"), ("x", "y"), "peu importe")


def test_un_conflit_qui_vise_tous_les_modeles_est_replie_en_regime():
    conflits = [_conflit(NIVEAU_STABILITE, f"M{i}") for i in range(4)]
    replies = _replier_les_conflits_universels(conflits, 4)
    assert len(replies) == 1
    assert replies[0].family == REGIME
    assert "4" in replies[0].subject


def test_un_conflit_qui_vise_une_partie_des_modeles_est_conserve():
    conflits = [_conflit(PROTOCOLE, f"M{i}") for i in range(3)]
    replies = _replier_les_conflits_universels(conflits, 4)
    assert len(replies) == 3
    assert all(c.family == PROTOCOLE for c in replies)


def test_le_perimetre_nest_jamais_replie():
    """
    Il vise l'ensemble par construction — c'est précisément son objet,
    pas une coïncidence à replier.
    """
    conflits = [_conflit(PERIMETRE, "tous les actifs")]
    assert _replier_les_conflits_universels(conflits, 4) == conflits


def test_deux_familles_sont_repliees_independamment():
    conflits = [_conflit(NIVEAU_STABILITE, f"M{i}") for i in range(3)]
    conflits += [_conflit(PROTOCOLE, "M0")]
    replies = _replier_les_conflits_universels(conflits, 3)
    familles = sorted(c.family for c in replies)
    assert familles == [PROTOCOLE, REGIME]


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — chaque famille est détectée sur un univers qui la porte
# ══════════════════════════════════════════════════════════════════════


def test_le_perimetre_se_declenche_quand_les_donnees_sont_suspectes():
    donnees = univers(10, graine=5)
    prix = prix_depuis_rendements(donnees, defauts=True)

    r = rapport(donnees, prices=prix)
    assert r.conditional
    familles = {c.family for c in r.conflicts}
    assert PERIMETRE in familles
    assert r.data_caveats


def test_sans_prix_le_rapport_dit_quil_na_pas_verifie_les_donnees():
    """
    Un rapport qui tait ce qu'il n'a pas vérifié est pire qu'un rapport
    incomplet : il laisse croire que la vérification a eu lieu.
    """
    r = rapport(univers(8, graine=6))
    assert any("n'a PAS été vérifiée" in c for c in r.data_caveats)
    assert not r.conditional  # aucune anomalie constatée, donc pas conditionnel


def test_le_protocole_se_declenche_quand_la_matrice_est_mal_conditionnee():
    """
    Le résultat central de la séance 12, désormais cherché
    automatiquement : le défaut n'apparaît qu'à l'inversion.
    """
    r = rapport(univers(100, graine=7, n_facteurs=5))
    assert PROTOCOLE in {c.family for c in r.conflicts}


def test_le_regime_se_declenche_quand_la_volatilite_est_groupee():
    r = rapport(univers(14, graine=8, garch=True))
    assert REGIME in {c.family for c in r.conflicts}


def test_le_corps_queue_se_declenche_sur_des_queues_epaisses():
    """
    Univers à queues de Student : le corps est bien décrit, la queue
    non. Un seul univers suffirait à passer par chance, donc on en
    essaie trois et on exige que la famille apparaisse au moins une
    fois — la détection doit être réelle, pas fortuite.
    """
    vu = []
    for graine in (2, 9, 12):
        r = rapport(univers(18, graine=graine, nu=4))
        familles = {c.family for c in r.conflicts}
        vu.append(CORPS_QUEUE in familles)
    assert any(vu), "CORPS / QUEUE jamais détecté sur trois univers à queues épaisses"


def test_le_corps_queue_suit_exactement_la_regle_de_severite():
    """
    Le MÉCANISME plutôt que le tirage : dès que la queue est
    strictement plus sévère que l'inventaire pour un modèle, le conflit
    doit exister pour ce modèle — et réciproquement.

    Un test qui dépend d'un tirage particulier ne vérifie pas une
    règle ; celui-ci la vérifie sur tous les modèles de deux univers.
    """
    for graine, nu in ((2, 4), (13, None)):
        r = rapport(univers(18, graine=graine, nu=nu))
        avec_conflit = {c.subject for c in r.conflicts if c.family == CORPS_QUEUE}
        for s in r.subjects:
            inv = next(
                (f for f in r.findings_for(s) if f.instrument == "inventaire"), None
            )
            queue = next(
                (f for f in r.findings_for(s) if f.instrument == "queue"), None
            )
            if inv is None or queue is None:
                continue
            attendu = SEVERITE[queue.verdict] > SEVERITE[inv.verdict]
            assert (s in avec_conflit) == attendu, (
                f"{s} : queue {queue.verdict}, inventaire {inv.verdict}, "
                f"conflit {'présent' if s in avec_conflit else 'absent'}"
            )


def test_un_conflit_porte_toujours_une_interpretation():
    """
    « Deux instruments divergent » n'est pas exploitable. Un conflit
    sans interprétation est un conflit inutile, donc on l'interdit.
    """
    for donnees in (univers(14, graine=10, garch=True), univers(90, graine=11)):
        for c in rapport(donnees).conflicts:
            assert c.interpretation
            assert len(c.interpretation) > 40
            assert c.instruments and c.verdicts


# ══════════════════════════════════════════════════════════════════════
#  Le test de forme, et sa calibration
# ══════════════════════════════════════════════════════════════════════


def test_le_ratio_empirique_retrouve_la_valeur_theorique_en_grand_echantillon():
    u = np.random.default_rng(20).standard_normal(400_000)
    assert _ratio_empirique(u, ALPHA) == pytest.approx(
        _ratio_gaussien(ALPHA), rel=0.02
    )


def test_la_loi_du_ratio_est_centree_sur_la_theorie_et_setale():
    """
    J'avais supposé un biais de petit échantillon. LA MESURE M'A
    CONTREDIT : l'estimateur est pratiquement sans biais, et c'est sa
    LARGEUR qui m'avait trompé. Ce test enregistre la correction.
    """
    theorie = _ratio_gaussien(ALPHA)
    for n in (1000, 4000):
        loi = _loi_du_ratio_sous_h0(n, ALPHA, 800, seed=21)
        assert float(np.median(loi)) == pytest.approx(theorie, abs=0.02)
    etroite = _loi_du_ratio_sous_h0(4000, ALPHA, 800, seed=22)
    large = _loi_du_ratio_sous_h0(1000, ALPHA, 800, seed=22)
    assert float(large.std(ddof=1)) > float(etroite.std(ddof=1))


@pytest.mark.slow
def test_le_test_de_forme_ne_se_declenche_pas_a_tort():
    """
    LE TEST DE COUVERTURE DU TEST DE FORME.

    Sur des résidus corrects — y compris avec la structure par blocs du
    rejeu et l'erreur d'estimation de la volatilité — il doit se
    déclencher à son niveau nominal de 5 %. Ma première version, qui
    comparait à la valeur théorique avec une erreur-type de bootstrap,
    se déclenchait bien plus souvent.
    """
    n, step, k = 1048, 21, 400
    loi = _loi_du_ratio_sous_h0(n, ALPHA, 2000, seed=23)
    bas, haut = (float(x) for x in np.quantile(loi, (0.025, 0.975)))

    dehors = 0
    for i in range(k):
        g = np.random.default_rng(3000 + i)
        x = g.standard_normal(n)
        residus = np.empty(n)
        for d in range(0, n, step):
            f = min(d + step, n)
            # volatilité estimée sur 252 observations : son erreur
            # relative est portée par un chi-deux, et elle est partagée
            # par tout le bloc.
            residus[d:f] = x[d:f] / math.sqrt(g.chisquare(251) / 251)
        r = _ratio_empirique(residus, ALPHA)
        if not (bas <= r <= haut):
            dehors += 1

    taux = dehors / k
    assert taux < 0.12, f"le test de forme se déclenche à tort {taux:.1%} du temps"


# ══════════════════════════════════════════════════════════════════════
#  Le rapport comme objet
# ══════════════════════════════════════════════════════════════════════


def test_le_rapport_juge_les_quatre_modeles_du_moteur():
    r = rapport(univers(12, graine=30))
    assert len(r.subjects) == 4
    for s in r.subjects:
        assert r.verdict_for(s) in SEVERITE
        assert r.binding_instrument(s) != "aucun"


def test_la_table_porte_une_colonne_par_instrument_plus_le_verdict():
    table = rapport(univers(12, graine=31)).table()
    assert "modèle" in table.columns
    assert "VERDICT" in table.columns
    assert "contraint par" in table.columns
    assert len(table) == 4


def test_le_texte_montre_les_conflits_et_la_limite():
    texte = str(rapport(univers(90, graine=32)))
    assert "RAPPORT DE VALIDATION" in texte
    assert "VERDICTS" in texte
    assert "ne cherche que les conflits qu'on lui a appris" in texte


def test_le_html_est_autonome():
    """
    Aucune ressource externe. Un rapport de validation doit pouvoir être
    archivé et rouvert dans dix ans ; tout ce qui est chargé depuis
    ailleurs aura disparu.
    """
    page = rapport(univers(12, graine=33)).to_html()
    assert page.startswith("<!doctype html>")
    assert "</html>" in page
    for interdit in ("<script", "http://", "https://", "@import"):
        assert interdit not in page, f"ressource externe : {interdit}"
    assert "prefers-color-scheme" in page  # lisible en sombre
    assert "width=device-width" in page  # lisible au téléphone


def test_le_html_echappe_ce_qui_vient_des_donnees():
    """
    Les noms d'actifs viennent de l'extérieur. Un ticker contenant du
    HTML ne doit pas se retrouver interprété — c'est la discipline de
    la séance 10 appliquée à une sortie plutôt qu'à une entrée.
    """
    donnees = univers(6, graine=34)
    prix = prix_depuis_rendements(donnees, defauts=True)
    # Le nom de l'actif remonte dans les avertissements de qualité.
    nouveaux = ["<script>pirate()</script>", *list(donnees.columns[1:])]
    donnees.columns = nouveaux
    prix = dict(zip(nouveaux, prix.values(), strict=True))

    page = rapport(donnees, prices=prix, universe="<b>gras</b>").to_html()
    assert "<script>pirate()</script>" not in page
    assert "<b>gras</b>" not in page
    assert "&lt;b&gt;gras&lt;/b&gt;" in page
    # Et le seul <script> de la page doit être… aucun.
    assert "<script" not in page


def test_le_seuil_de_dispersion_est_un_parametre_et_il_est_declare():
    donnees = univers(14, graine=35, garch=True)
    strict = rapport(donnees, dispersion_max=1.05)
    laxe = rapport(donnees, dispersion_max=9.0)
    assert len(strict.conflicts) >= len(laxe.conflicts)
    assert f"{DISPERSION_MAX:.2f}" in rapport(donnees).to_html()


def test_la_causalite_desactivee_est_ecrite_dans_le_rapport():
    r = rapport(univers(10, graine=36), check_causality=False)
    assert "NON VÉRIFIÉE" in r.causality


def test_la_causalite_activee_rend_un_verdict():
    r = build_report(
        univers(10, graine=37), n_sim=400, check_causality=True
    )
    assert r.causality
    assert "NON VÉRIFIÉE" not in r.causality


# ══════════════════════════════════════════════════════════════════════
#  Ce que le rapport refuse
# ══════════════════════════════════════════════════════════════════════


def test_un_seul_actif_est_refuse():
    with pytest.raises(ValueError, match="au moins deux actifs"):
        build_report(univers(5, graine=40).iloc[:, :1])


def test_un_historique_plus_court_que_la_fenetre_leve_insufficient_data():
    with pytest.raises(InsufficientData):
        build_report(univers(5, graine=41, n_periodes=100), lookback=252)


def test_un_type_errone_est_refuse():
    with pytest.raises(TypeError, match="DataFrame"):
        build_report(pd.Series([1.0, 2.0, 3.0]))


# ══════════════════════════════════════════════════════════════════════
#  Cohérence avec les séances précédentes
# ══════════════════════════════════════════════════════════════════════


def test_le_rapport_saccorde_avec_linventaire_quil_appelle():
    """
    Le rapport ne recalcule pas l'inventaire : il l'appelle. Les
    verdicts doivent donc coïncider exactement, sinon une divergence
    silencieuse s'installerait entre deux chemins de code.
    """
    from nkrisk.validation import build_inventory

    donnees = univers(14, graine=50)
    r = rapport(donnees)
    inv = build_inventory(donnees, lookback=252, step=21)

    for carte in inv.cards:
        constats = [
            f for f in r.findings_for(carte.model) if f.instrument == "inventaire"
        ]
        assert len(constats) == 1
        assert constats[0].verdict == carte.verdict


def test_le_verdict_de_queue_separe_la_forme_et_le_niveau():
    """
    Les deux composantes doivent être visibles dans le détail. Ma
    première version n'en avait qu'une, et elle était aveugle à la
    forme parce que sa nulle était la forme réalisée.
    """
    r = rapport(univers(14, graine=51, nu=4))
    queues = [f for f in r.findings if f.instrument == "queue"]
    assert len(queues) == 4
    for f in queues:
        if f.verdict != INAPPLICABLE:
            assert "forme" in f.detail
            assert "niveau" in f.detail
