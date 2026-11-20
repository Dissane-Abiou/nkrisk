"""
Tests de la couche de données — séance 9.

CE QUE CES TESTS PROUVENT, ET QUI N'ÉTAIT PAS PROUVABLE AVANT
=============================================================

`data/market.py` est resté à 0 % de couverture pendant huit séances,
avec cette justification : « la fonction appelle le réseau, donc elle
n'est pas testable sans réseau ».

Le raisonnement était faux. Une fonction n'est pas intestable parce
qu'elle appelle le réseau — elle l'est parce qu'elle DÉCIDE elle-même
d'appeler le réseau. Une fois la source passée en paramètre, tout se
teste hors ligne, et il ne reste non couvert que l'adaptateur Yahoo,
quinze lignes sans aucune logique.

**« Ce n'est pas testable » veut presque toujours dire « ce n'est pas
encore bien découpé ».**

Les tests de calibration de la fin sont les plus importants : ils
mesurent, contre une vérité connue, le biais de chaque politique
d'alignement de calendriers. C'est un choix que tout le monde fait en
une ligne et que personne ne chiffre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nkrisk.data.market import (
    CsvSource,
    InMemorySource,
    LoadResult,
    PriceSource,
    YahooSource,
    load_prices,
)
from nkrisk.data.quality import (
    AVERTISSEMENT,
    BLOQUANT,
    QualityReport,
    align_prices,
    compare_alignment,
    inspect_prices,
    observations_per_year,
)

JOURS = 252


# ══════════════════════════════════════════════════════════════════════
#  FABRIQUES
# ══════════════════════════════════════════════════════════════════════


def _prix(
    n: int = 6,
    t: int = 2_520,
    seed: int = 0,
    vol: float = 0.20,
    rho: float = 0.5,
) -> pd.DataFrame:
    """Prix dont la volatilité et la corrélation sont CONNUES."""
    rng = np.random.default_rng(seed)
    c = np.full((n, n), rho)
    np.fill_diagonal(c, 1.0)
    r = (rng.normal(0, 1, (t, n)) @ np.linalg.cholesky(c).T) * vol / np.sqrt(JOURS)
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(r, axis=0)),
        index=pd.bdate_range("2015-01-02", periods=t),
        columns=[f"A{i}" for i in range(n)],
    )


def _deux_places(
    complet: pd.DataFrame, part_fermee: float, seed: int = 0
) -> dict[str, pd.Series]:
    """
    Répartit les actifs sur deux places boursières aux fériés distincts.

    Modèle réaliste : un jour de fermeture ne vaut PAS un rendement nul.
    Le titre ne cote pas, et le mouvement du marché se retrouve dans la
    cotation suivante. C'est pour cela qu'on retire les lignes de la
    série observée au lieu d'y mettre des zéros.
    """
    series: dict[str, pd.Series] = {}
    for i, col in enumerate(complet.columns):
        rng = np.random.default_rng(seed * 97 + (i % 2))
        ferme = rng.random(len(complet)) < part_fermee
        series[col] = complet[col][~ferme]
    return series


# ══════════════════════════════════════════════════════════════════════
#  LES SOURCES
# ══════════════════════════════════════════════════════════════════════


def test_les_trois_sources_satisfont_le_protocole() -> None:
    """
    Quatrième usage du typage structurel dans le projet, après `Shock`,
    `Strategy` et les capacités optionnelles du backtest.
    """
    assert isinstance(InMemorySource({}), PriceSource)
    assert isinstance(CsvSource("."), PriceSource)
    assert isinstance(YahooSource(), PriceSource)


def test_la_source_en_memoire_filtre_les_dates() -> None:
    complet = _prix(n=2, t=500)
    source = InMemorySource({c: complet[c] for c in complet.columns})

    tout = source.fetch("A0")
    coupe = source.fetch("A0", start="2016-01-01", end="2016-12-31")

    assert len(tout) == 500
    assert coupe.index.min() >= pd.Timestamp("2016-01-01")
    assert coupe.index.max() <= pd.Timestamp("2016-12-31")
    assert 0 < len(coupe) < len(tout)
    assert coupe.name == "A0"


def test_la_source_en_memoire_refuse_un_symbole_inconnu() -> None:
    source = InMemorySource({"A0": _prix(n=1, t=100)["A0"]})
    with pytest.raises(KeyError, match="symbole inconnu"):
        source.fetch("INEXISTANT")


def test_la_source_csv_lit_un_fichier(tmp_path) -> None:
    complet = _prix(n=1, t=300)
    df = pd.DataFrame(
        {"Date": complet.index, "Close": complet["A0"].to_numpy()}
    )
    df.to_csv(tmp_path / "A0.csv", index=False)

    s = CsvSource(tmp_path).fetch("A0")
    assert len(s) == 300
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.name == "A0"


def test_la_source_csv_refuse_des_dates_en_double(tmp_path) -> None:
    """
    Pas de réparation silencieuse. Dédupliquer tout seul changerait la
    série sans que l'utilisateur le sache — un mensonge différé.
    """
    dates = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame(
        {"Date": [*list(dates), dates[3]], "Close": list(range(1, 12))}
    )
    df.to_csv(tmp_path / "DOUBLE.csv", index=False)

    with pytest.raises(ValueError, match="en double"):
        CsvSource(tmp_path).fetch("DOUBLE")


def test_la_source_csv_signale_un_fichier_absent(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        CsvSource(tmp_path).fetch("FANTOME")


def test_le_chargement_strict_sarrete_a_la_premiere_erreur() -> None:
    complet = _prix(n=3, t=200)
    source = InMemorySource({c: complet[c] for c in complet.columns})

    with pytest.raises(KeyError):
        load_prices(["A0", "INCONNU", "A1"], source)


def test_le_chargement_souple_consigne_les_echecs() -> None:
    complet = _prix(n=3, t=200)
    source = InMemorySource({c: complet[c] for c in complet.columns})

    res = load_prices(["A0", "INCONNU", "A1"], source, strict=False)

    assert isinstance(res, LoadResult)
    assert set(res.tickers) == {"A0", "A1"}
    assert "INCONNU" in res.failures
    assert "KeyError" in res.failures["INCONNU"]
    assert "ÉCHEC" in str(res)


def test_le_chargement_refuse_une_serie_trop_courte() -> None:
    complet = _prix(n=1, t=200)
    source = InMemorySource({"A0": complet["A0"].iloc[:1]})
    with pytest.raises(ValueError, match="trop courte"):
        load_prices(["A0"], source)


def test_le_chargement_refuse_ce_qui_nest_pas_une_serie_datee() -> None:
    class Bancale:
        name = "bancale"

        def fetch(self, ticker, start=None, end=None):  # type: ignore[no-untyped-def]
            return pd.Series([1.0, 2.0, 3.0])  # index entier, pas des dates

    with pytest.raises(TypeError, match="indexée par des dates"):
        load_prices(["A0"], Bancale())


# ══════════════════════════════════════════════════════════════════════
#  L'INSPECTION
# ══════════════════════════════════════════════════════════════════════


def test_des_donnees_propres_ne_declenchent_rien() -> None:
    """Le contrôle négatif d'abord : pas de cri au loup."""
    complet = _prix(n=4, t=600)
    rapport = inspect_prices({c: complet[c] for c in complet.columns})

    assert isinstance(rapport, QualityReport)
    assert rapport.usable
    assert rapport.anomalies == ()
    assert rapport.calendar_overlap == 1.0
    assert "aucune anomalie" in str(rapport)


def test_detecte_les_prix_figes() -> None:
    """
    Six séances au même prix, ce qui arrive sur un titre illiquide ou
    quand un jour férié a été reporté. La volatilité estimée sort trop
    basse, et parfaitement serrée — l'intervalle de confiance ne voit
    rien, parce que ce n'est pas un problème d'imprécision.
    """
    complet = _prix(n=1, t=400)
    s = complet["A0"].copy()
    s.iloc[100:106] = s.iloc[100]

    rapport = inspect_prices({"A0": s})
    figes = [a for a in rapport.anomalies if a.kind == "prix figés"]

    assert len(figes) == 1
    assert figes[0].severity == AVERTISSEMENT
    assert figes[0].first == s.index[100]
    assert rapport.usable  # avertissement, pas blocage


def test_detecte_un_saut_invraisemblable() -> None:
    complet = _prix(n=1, t=400)
    s = complet["A0"].copy()
    s.iloc[200:] = s.iloc[200:] * 0.5      # fractionnement 2:1 non corrigé

    rapport = inspect_prices({"A0": s})
    sauts = [a for a in rapport.anomalies if a.kind == "saut invraisemblable"]

    assert len(sauts) == 1
    assert "trancher" in sauts[0].detail
    assert rapport.usable


def test_un_prix_non_positif_est_bloquant() -> None:
    complet = _prix(n=1, t=300)
    s = complet["A0"].copy()
    s.iloc[42] = 0.0

    rapport = inspect_prices({"A0": s})
    assert not rapport.usable
    assert rapport.blocking[0].kind == "prix non positif"
    with pytest.raises(ValueError, match="bloquante"):
        rapport.raise_if_blocking()


def test_des_dates_en_double_sont_bloquantes() -> None:
    complet = _prix(n=1, t=300)
    s = pd.concat([complet["A0"], complet["A0"].iloc[[10]]]).sort_index()

    rapport = inspect_prices({"A0": s})
    assert not rapport.usable
    assert any(a.kind == "dates en double" for a in rapport.blocking)


def test_un_index_non_trie_est_bloquant() -> None:
    complet = _prix(n=1, t=300)
    s = complet["A0"].iloc[::-1]

    rapport = inspect_prices({"A0": s})
    assert any(a.kind == "dates non triées" for a in rapport.blocking)


def test_un_historique_court_est_signale() -> None:
    complet = _prix(n=1, t=30)
    rapport = inspect_prices({"A0": complet["A0"]}, min_points=60)
    courts = [a for a in rapport.anomalies if a.kind == "historique trop court"]
    assert len(courts) == 1
    assert courts[0].severity == AVERTISSEMENT


def test_detecte_le_desalignement_des_calendriers() -> None:
    complet = _prix(n=4, t=800)
    series = _deux_places(complet, part_fermee=0.05, seed=1)

    rapport = inspect_prices(series)
    desalignes = [
        a for a in rapport.anomalies if a.kind == "calendriers désalignés"
    ]

    assert len(desalignes) == 1
    assert rapport.calendar_overlap < 1.0
    assert "changer tes chiffres" in desalignes[0].detail
    assert not rapport.table().empty


def test_linspection_refuse_ce_qui_nest_pas_une_serie_datee() -> None:
    with pytest.raises(TypeError, match=r"pd\.Series"):
        inspect_prices({"A0": [1.0, 2.0]})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="indexée par des dates"):
        inspect_prices({"A0": pd.Series([1.0, 2.0, 3.0])})


def test_les_niveaux_de_gravite_sont_les_deux_attendus() -> None:
    complet = _prix(n=1, t=300)
    s = complet["A0"].copy()
    s.iloc[50:56] = s.iloc[50]
    s.iloc[100] = -1.0
    rapport = inspect_prices({"A0": s})
    assert {a.severity for a in rapport.anomalies} <= {BLOQUANT, AVERTISSEMENT}
    assert rapport.blocking and rapport.warnings


# ══════════════════════════════════════════════════════════════════════
#  L'ALIGNEMENT
# ══════════════════════════════════════════════════════════════════════


def test_les_deux_politiques_donnent_des_calendriers_differents() -> None:
    complet = _prix(n=4, t=800)
    series = _deux_places(complet, part_fermee=0.06, seed=2)

    inter = align_prices(series, "intersection")
    report = align_prices(series, "report")

    assert len(inter) < len(report) <= len(complet)
    assert not inter.isna().to_numpy().any()
    assert not report.isna().to_numpy().any()


def test_le_report_fabrique_des_rendements_nuls() -> None:
    """
    C'est le mécanisme exact du biais : un jour reporté n'est pas une
    donnée manquante, c'est l'AFFIRMATION que le prix n'a pas bougé.
    """
    complet = _prix(n=4, t=800)
    series = _deux_places(complet, part_fermee=0.08, seed=3)

    r_inter = align_prices(series, "intersection").pct_change().dropna()
    r_report = align_prices(series, "report").pct_change().dropna()

    zeros_inter = float((r_inter == 0.0).to_numpy().mean())
    zeros_report = float((r_report == 0.0).to_numpy().mean())

    assert zeros_inter < 0.001
    assert zeros_report > 0.05


def test_lalignement_refuse_une_politique_inconnue() -> None:
    complet = _prix(n=2, t=100)
    with pytest.raises(ValueError, match="politique inconnue"):
        align_prices({c: complet[c] for c in complet.columns}, policy="magie")


def test_lalignement_refuse_un_univers_vide() -> None:
    with pytest.raises(ValueError, match="aucune série"):
        align_prices({})


def test_lalignement_refuse_des_calendriers_qui_ne_se_recoupent_pas() -> None:
    a = pd.Series(
        [1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3)
    )
    b = pd.Series(
        [1.0, 2.0, 3.0], index=pd.bdate_range("2021-01-01", periods=3)
    )
    with pytest.raises(ValueError, match="ne se recoupent"):
        align_prices({"A": a, "B": b}, policy="intersection")


def test_les_observations_par_an_reflettent_le_calendrier_reel() -> None:
    complet = _prix(n=3, t=1_260)        # cinq ans de jours ouvrés
    plein = observations_per_year(complet.index)

    series = _deux_places(complet, part_fermee=0.20, seed=4)
    reduit = observations_per_year(align_prices(series, "intersection").index)

    assert 250 < plein < 265
    assert reduit < plein * 0.85

    with pytest.raises(ValueError, match="deux dates"):
        observations_per_year(complet.index[:1])


# ══════════════════════════════════════════════════════════════════════
#  CALIBRATION — LES DEUX BIAIS, MESURÉS CONTRE UNE VÉRITÉ CONNUE
# ══════════════════════════════════════════════════════════════════════


def _biais(part_fermee: float, graines: range) -> dict[str, float]:
    """
    Mesure les deux politiques contre un marché dont on connaît la
    volatilité (20 %) et la corrélation (0,50).
    """
    mesures: dict[str, list[float]] = {
        k: [] for k in ("inter_252", "inter_corr", "report_vol",
                        "report_corr", "ptf_report", "ptf_inter_corr")
    }
    for seed in graines:
        complet = _prix(n=8, t=2_520, seed=seed, vol=0.20, rho=0.5)
        series = _deux_places(complet, part_fermee, seed=seed)
        annees = len(complet) / JOURS
        w = np.full(complet.shape[1], 1.0 / complet.shape[1])

        # `w` passé en argument et non capturé : une fermeture sur une
        # variable de boucle est un piège classique — elle lit la
        # DERNIÈRE valeur, pas celle de l'itération où elle est définie.
        # Ici l'appel est immédiat, donc inoffensif ; ruff signale le
        # motif, et le motif est effectivement fragile.
        def profil(
            prix: pd.DataFrame, par_an: float, poids: np.ndarray
        ) -> tuple[float, float, float]:
            r = prix.pct_change().dropna()
            vol = float((r.std(ddof=1) * np.sqrt(par_an)).mean())
            c = r.corr().to_numpy()
            corr = float(c[~np.eye(len(c), dtype=bool)].mean())
            cov = r.cov().to_numpy() * par_an
            return vol, corr, float(np.sqrt(poids @ cov @ poids))

        ref_vol, ref_corr, ref_ptf = profil(complet, JOURS, w)
        inter = align_prices(series, "intersection")
        report = align_prices(series, "report")

        i252 = profil(inter, JOURS, w)
        icorr = profil(inter, len(inter) / annees, w)
        rep = profil(report, len(report) / annees, w)

        mesures["inter_252"].append(i252[0] / ref_vol - 1.0)
        mesures["inter_corr"].append(icorr[0] / ref_vol - 1.0)
        mesures["report_vol"].append(rep[0] / ref_vol - 1.0)
        mesures["report_corr"].append(rep[1] / ref_corr - 1.0)
        mesures["ptf_report"].append(rep[2] / ref_ptf - 1.0)
        mesures["ptf_inter_corr"].append(icorr[2] / ref_ptf - 1.0)

    return {k: float(np.mean(v)) for k, v in mesures.items()}


def test_le_biais_de_lintersection_est_une_erreur_de_convention() -> None:
    """
    PREMIÈRE LOI MESURÉE.

    L'intersection jette les séances non communes. Une année ne contient
    alors plus 252 observations, et annualiser quand même par √252
    surestime la volatilité d'à peu près la part jetée.

    Corriger l'annualisation par le nombre d'observations réellement
    présentes efface le biais entièrement.

    Ce n'est donc PAS un défaut de l'alignement : c'est le bug de la
    séance 1 — comparer deux grandeurs annualisées selon deux
    conventions — revenu huit séances plus tard sous un autre
    déguisement.
    """
    b = _biais(part_fermee=0.10, graines=range(10))

    assert 0.07 < b["inter_252"] < 0.16, (
        f"biais attendu de l'ordre de la part jetée (10 %), "
        f"mesuré {b['inter_252']:+.2%}"
    )
    assert abs(b["inter_corr"]) < 0.02, (
        f"l'annualisation corrigée devrait être non biaisée, "
        f"mesuré {b['inter_corr']:+.2%}"
    )
    assert abs(b["ptf_inter_corr"]) < 0.02


def test_le_report_ne_biaise_pas_la_volatilite_mais_la_correlation() -> None:
    """
    SECONDE LOI MESURÉE, ET CE N'EST PAS CELLE QUE J'ATTENDAIS.

    J'avais écrit dans la docstring du module que le report, en
    injectant des rendements nuls, réduisait la variance ET la
    corrélation. La moitié était fausse.

    La volatilité ne bouge quasiment pas : les zéros la tirent vers le
    bas, mais la séance suivante absorbe le mouvement manqué et la tire
    vers le haut. Les deux effets se compensent.

    Rien ne compense pour la corrélation. Quand un actif est reporté
    pendant que l'autre bouge, le co-mouvement est détruit et rien ne le
    restitue. Le biais y est entier et vaut à peu près la part de
    séances manquantes.

    C'est le pire endroit où se tromper : la corrélation gouverne la
    diversification, donc le portefeuille paraît moins risqué qu'il ne
    l'est.
    """
    b = _biais(part_fermee=0.10, graines=range(10))

    assert abs(b["report_vol"]) < 0.03, (
        f"la volatilité du report devrait être presque non biaisée, "
        f"mesuré {b['report_vol']:+.2%}"
    )
    assert -0.16 < b["report_corr"] < -0.05, (
        f"la corrélation devrait être sous-estimée d'environ la part "
        f"manquante (10 %), mesuré {b['report_corr']:+.2%}"
    )
    assert -0.08 < b["ptf_report"] < -0.01, (
        f"le risque du portefeuille devrait être sous-estimé, "
        f"mesuré {b['ptf_report']:+.2%}"
    )


@pytest.mark.slow
def test_les_deux_biais_croissent_avec_la_part_de_seances_manquantes() -> None:
    """
    Les deux lois sont linéaires en la part de séances manquantes. On
    le vérifie sur quatre régimes plutôt que de l'affirmer sur un seul.
    """
    mesures = {p: _biais(p, range(6)) for p in (0.02, 0.05, 0.10, 0.20)}

    inter = [mesures[p]["inter_252"] for p in (0.02, 0.05, 0.10, 0.20)]
    corr = [mesures[p]["report_corr"] for p in (0.02, 0.05, 0.10, 0.20)]

    assert inter == sorted(inter), f"le biais devrait croître : {inter}"
    assert corr == sorted(corr, reverse=True), (
        f"la sous-estimation devrait s'aggraver : {corr}"
    )
    # Ordre de grandeur : le biais vaut à peu près la part manquante.
    for p in (0.05, 0.10, 0.20):
        assert 0.5 * p < mesures[p]["inter_252"] < 1.8 * p
        assert 0.5 * p < -mesures[p]["report_corr"] < 1.8 * p


def test_la_comparaison_dalignement_expose_les_deux_conventions() -> None:
    """
    `compare_alignment` annualise CHAQUE politique par son propre nombre
    d'observations. Utiliser 252 pour les deux attribuerait à
    l'alignement un écart qui vient de la convention.
    """
    complet = _prix(n=6, t=1_500, rho=0.6)
    series = _deux_places(complet, part_fermee=0.08, seed=5)

    c = compare_alignment(series)

    assert c.n_dates["intersection"] < c.n_dates["report"]
    assert (
        c.observations_per_year["intersection"]
        < c.observations_per_year["report"]
    )
    assert c.mean_correlation["report"] < c.mean_correlation["intersection"]
    assert abs(c.volatility_shift.mean()) < 0.05
    assert "observations par an" in str(c)


# ══════════════════════════════════════════════════════════════════════
#  CE QUE LES VRAIES DONNÉES ONT RÉVÉLÉ
# ══════════════════════════════════════════════════════════════════════


def test_un_historique_tronque_nest_pas_un_desalignement_de_calendrier() -> None:
    """
    ⚠️ DÉFAUT TROUVÉ PAR LE PREMIER JEU DE VRAIES DONNÉES.

    Le rapport annonçait « 7 % de dates non communes » en additionnant
    deux causes sans rapport :

      • un titre entré en Bourse cinq mois après les autres ;
      • les jours fériés qui diffèrent entre Toronto et New York.

    Le premier n'est pas un désalignement de calendrier, c'est un
    historique plus court — et il coûte infiniment plus cher. Prendre
    l'intersection jette alors les cinq premiers mois POUR TOUT
    L'UNIVERS à cause d'un seul titre, alors que l'exclure de l'univers
    serait souvent le meilleur choix.

    Un rapport qui additionne deux causes aux remèdes opposés empêche
    de choisir le bon.
    """
    complet = _prix(n=4, t=1_000)
    series = {c: complet[c] for c in complet.columns}
    # un titre qui n'existe pas encore sur les 200 premières séances
    tardif = list(series)[-1]
    series[tardif] = series[tardif].iloc[200:]

    rapport = inspect_prices(series)

    tronques = [a for a in rapport.anomalies if a.kind == "démarrage tardif"]
    desalignes = [
        a for a in rapport.anomalies if a.kind == "calendriers désalignés"
    ]

    assert len(tronques) == 1
    assert tronques[0].asset == tardif
    assert tronques[0].count == 200
    assert "TOUT l'univers" in tronques[0].detail
    # Aucun jour férié décalé ici : rien ne doit être signalé à ce titre.
    assert desalignes == []
    # Les deux mesures doivent diverger, et c'est tout le propos.
    assert rapport.calendar_overlap < 1.0
    assert rapport.holiday_overlap == 1.0


def test_les_deux_causes_sont_rapportees_separement() -> None:
    """Quand les deux coexistent, chacune est chiffrée pour ce qu'elle est."""
    complet = _prix(n=4, t=1_000)
    series = _deux_places(complet, part_fermee=0.05, seed=7)
    tardif = list(series)[-1]
    series[tardif] = series[tardif].iloc[150:]

    rapport = inspect_prices(series)
    kinds = [a.kind for a in rapport.anomalies]

    assert "démarrage tardif" in kinds
    assert "calendriers désalignés" in kinds

    desaligne = next(
        a for a in rapport.anomalies if a.kind == "calendriers désalignés"
    )
    # Le désalignement ne doit compter QUE la fenêtre commune : son
    # décompte doit rester largement sous le total des dates perdues.
    assert desaligne.count < rapport.n_dates_union - rapport.n_dates_intersection
    assert "fenêtre commune" in desaligne.detail


def test_une_troncature_dun_jour_nest_pas_signalee() -> None:
    """
    ⚠️ SECOND DÉFAUT TROUVÉ PAR LES VRAIES DONNÉES.

    Les titres américains s'arrêtaient la veille des canadiens — un
    simple décalage de publication. Le rapport produisait cinq
    avertissements identiques pour un jour, au-dessus du seul qui
    comptait : une entrée en Bourse à 97 jours.

    Un détecteur qui signale le trivial au même rang que le grave n'est
    pas plus prudent, il est moins lu. **Le bruit d'un outil de
    détection est un coût, pas une précaution gratuite** — c'est la même
    leçon que le contrôle négatif de l'audit de causalité à la séance 8.
    """
    complet = _prix(n=5, t=800)
    series = {c: complet[c] for c in complet.columns}
    for nom in list(series)[:3]:
        series[nom] = series[nom].iloc[:-1]      # un jour de moins
    tardif = list(series)[-1]
    series[tardif] = series[tardif].iloc[40:]    # quarante de moins

    rapport = inspect_prices(series, min_truncation=5)
    kinds = [(a.kind, a.asset) for a in rapport.anomalies]

    assert ("démarrage tardif", tardif) in kinds
    assert not any(k == "fin anticipée" for k, _ in kinds), (
        "un décalage d'un jour ne doit pas produire d'avertissement"
    )
    assert len([k for k, _ in kinds if k == "démarrage tardif"]) == 1


def test_les_deux_mesures_de_recouvrement_ne_disent_pas_la_meme_chose() -> None:
    """
    `calendar_overlap` mélange historiques inégaux et fériés décalés.
    `holiday_overlap` ne mesure que les seconds, sur la fenêtre commune.

    La narration de `donnees_reelles.py` annonçait 7 % en citant le
    premier là où le rapport affichait 3,9 % du second, deux lignes
    plus haut. Dixième dérive de narration du projet.
    """
    complet = _prix(n=4, t=1_000)
    series = _deux_places(complet, part_fermee=0.05, seed=11)
    tardif = list(series)[-1]
    series[tardif] = series[tardif].iloc[300:]

    rapport = inspect_prices(series)

    assert rapport.calendar_overlap < rapport.holiday_overlap < 1.0
    assert rapport.n_dates_common_window < rapport.n_dates_union
