"""
Tests du backtest d'Expected Shortfall (séance 13).

Quatre niveaux, comme partout dans ce projet, plus un test de causalité.

Le test le plus intéressant du fichier est
`test_le_modele_au_mauvais_niveau_donne_un_z2_predit_exactement` : on
peut DÉRIVER la valeur que Z₂ doit prendre pour un modèle qui annonce
les chiffres d'un niveau p′ en prétendant tenir α, et c'est 1 − p′/α.
Un test qui compare une mesure à une prédiction théorique indépendante
vaut dix tests qui comparent une mesure à elle-même.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import prevision_garch_causale
from scipy import stats

from nkrisk.shortfall import (
    LEVEL_FRTB,
    MIN_EXPECTED_BREACHES,
    REJETE,
    AcerbiSzekelyResult,
    EmpiricalNull,
    GaussianNull,
    StandardizedNull,
    StudentTNull,
    TailForecast,
    acerbi_szekely,
    compare_tail_models,
    fz_loss,
    rolling_tail_forecasts,
    tail_statistic,
)
from nkrisk.types import InsufficientData
from nkrisk.var import TailRisk, historical_var, kupiec_test, parametric_var


def _prevision_bloc(h: np.ndarray, n: int, bloc: int) -> np.ndarray:
    """Chemin de prévisions GARCH causales, bloc par bloc."""
    sortie = np.empty(n)
    for d in range(0, n, bloc):
        f = min(d + bloc, n)
        sortie[d:f] = prevision_garch_causale(h, d, f - d)
    return sortie


ALPHA = LEVEL_FRTB


def var_es_normale(p: float) -> tuple[float, float]:
    """VaR et ES exactes d'une N(0,1) au niveau de queue p, en magnitudes."""
    q = float(stats.norm.ppf(p))
    es = float(stats.norm.expect(lambda x: -x, lb=-np.inf, ub=q)) / p
    return -q, es


V_VRAI, E_VRAI = var_es_normale(ALPHA)


def plat(v: float, e: float, n: int, alpha: float = ALPHA) -> TailForecast:
    return TailForecast(var=np.full(n, v), es=np.full(n, e), alpha=alpha)


def normaux(n: int, graine: int) -> np.ndarray:
    return np.random.default_rng(graine).standard_normal(n)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 1 — propriétés et refus
# ══════════════════════════════════════════════════════════════════════


def test_les_valeurs_exactes_de_la_normale_sont_celles_quon_croit():
    """
    Ancre du fichier : si ces deux nombres sont faux, tout le reste est
    faux sans qu'aucun autre test ne le voie.
    """
    assert pytest.approx(1.959964, rel=1e-6) == V_VRAI
    assert pytest.approx(2.337803, rel=1e-6) == E_VRAI
    assert pytest.approx(1.192778, rel=1e-6) == E_VRAI / V_VRAI


def test_une_prevision_exacte_donne_un_z_proche_de_zero():
    x = normaux(8000, 1)
    f = plat(V_VRAI, E_VRAI, x.size)
    for test in (1, 2):
        z, _ = tail_statistic(x, f, test=test)
        assert z == pytest.approx(0.0, abs=0.12), f"test {test}"


def test_le_signe_de_z_dit_le_sens_du_danger():
    x = normaux(8000, 2)
    for test in (1, 2):
        sous, _ = tail_statistic(x, plat(V_VRAI * 0.8, E_VRAI * 0.8, x.size), test=test)
        sur, _ = tail_statistic(x, plat(V_VRAI * 1.3, E_VRAI * 1.3, x.size), test=test)
        assert sous < 0.0, f"sous-estimation, test {test}"
        assert sur > 0.0, f"excès de prudence, test {test}"


def test_z_est_invariant_dechelle():
    """
    Multiplier rendements, VaR et ES par la même constante ne change
    rien : z est un rapport sans dimension. Si ce test échouait, une
    constante d'échelle serait entrée là où elle n'a rien à faire —
    la famille de bugs que la séance 1 a payée.
    """
    x = normaux(2000, 3)
    f = plat(V_VRAI, E_VRAI, x.size)
    for test in (1, 2):
        ref, n_ref = tail_statistic(x, f, test=test)
        for c in (1e-4, 0.5, 250.0):
            mis = plat(V_VRAI * c, E_VRAI * c, x.size)
            z, n = tail_statistic(x * c, mis, test=test)
            assert z == pytest.approx(ref, rel=1e-12)
            assert n == n_ref


def test_z1_et_z2_coincident_quand_les_depassements_sont_exactement_attendus():
    """
    Propriété exacte : Z₁ normalise par le nombre OBSERVÉ de
    dépassements, Z₂ par le nombre ATTENDU. Les deux sont donc égaux —
    au bit près — quand les deux coïncident. C'est le test qui prouve
    que la seule différence entre les deux formules est bien celle-là.
    """
    n = 400
    attendus = round(n * ALPHA)
    assert attendus == 10
    # On fabrique exactement 10 dépassements.
    x = np.full(n, 0.5)
    x[:attendus] = -3.0
    f = plat(V_VRAI, E_VRAI, n)

    z1, n1 = tail_statistic(x, f, test=1)
    z2, n2 = tail_statistic(x, f, test=2)
    assert n1 == n2 == attendus
    assert z1 == pytest.approx(z2, rel=1e-15)


def test_sans_aucun_depassement_la_statistique_vaut_un():
    """
    Cas limite : aucun dépassement. Z₂ vaut exactement +1 par la
    formule, et Z₁ n'est pas défini (division par zéro). On retourne +1
    dans les deux cas — « rien n'est arrivé » est l'extrême de la
    prudence — et `n_breaches` permet de le détecter.
    """
    x = np.full(500, 0.1)
    f = plat(V_VRAI, E_VRAI, x.size)
    for test in (1, 2):
        z, n = tail_statistic(x, f, test=test)
        assert n == 0
        assert z == 1.0


def test_tail_statistic_saccorde_avec_acerbi_szekely():
    x = normaux(1000, 4)
    f = plat(V_VRAI, E_VRAI, x.size)
    for test in (1, 2):
        z, n = tail_statistic(x, f, test=test)
        r = acerbi_szekely(x, f, null=GaussianNull(), test=test, n_sim=300)
        assert r.statistic == pytest.approx(z, rel=1e-15)
        assert r.n_breaches == n


def test_le_rejet_est_unilateral_a_gauche():
    """
    Un modèle absurdement prudent ne doit JAMAIS être rejeté par ce
    test. C'est un choix, pas un oubli : un modèle trop prudent
    immobilise du capital, un modèle trop optimiste cache une perte.
    """
    x = normaux(2000, 5)
    r = acerbi_szekely(
        x, plat(V_VRAI * 5, E_VRAI * 5, x.size), null=GaussianNull(), n_sim=500
    )
    assert r.statistic > 0.0
    assert not r.rejected
    assert "prudent" in r.direction


def test_la_p_value_nest_jamais_exactement_nulle():
    """
    Avec un nombre fini de simulations, une p-value de 0 n'a pas de
    sens : elle affirmerait une certitude que les tirages ne portent
    pas. D'où le +1 au numérateur et au dénominateur.
    """
    x = normaux(2000, 6)
    r = acerbi_szekely(
        x, plat(V_VRAI * 0.3, E_VRAI * 0.3, x.size), null=GaussianNull(), n_sim=500
    )
    assert r.rejected
    assert 0.0 < r.p_value <= 1.0


def test_seul_le_test_1_se_declare_aveugle_a_la_frequence():
    x = normaux(1000, 7)
    f = plat(V_VRAI, E_VRAI, x.size)
    r1 = acerbi_szekely(x, f, null=GaussianNull(), test=1, n_sim=300)
    r2 = acerbi_szekely(x, f, null=GaussianNull(), test=2, n_sim=300)
    assert r1.blind_to_frequency
    assert not r2.blind_to_frequency
    assert "ne voit PAS la fréquence" in str(r1)
    assert "ne voit PAS la fréquence" not in str(r2)
    assert isinstance(r1, AcerbiSzekelyResult)


# ---- TailForecast : ce qu'il refuse ----


def test_es_inferieure_a_la_var_est_refusee_avec_lindice_fautif():
    v = np.full(10, 2.0)
    e = np.full(10, 2.5)
    e[4] = 1.5
    with pytest.raises(ValueError, match="ES < VaR à l'indice 4"):
        TailForecast(var=v, es=e, alpha=ALPHA)


def test_une_var_negative_est_refusee_en_expliquant_la_convention():
    """
    L'erreur la plus probable d'un utilisateur : passer la VaR comme un
    quantile négatif. Le message doit nommer la convention, pas
    seulement refuser.
    """
    with pytest.raises(ValueError, match="MAGNITUDE POSITIVE"):
        TailForecast(var=np.full(10, -2.0), es=np.full(10, 2.5), alpha=ALPHA)


@pytest.mark.parametrize("mauvais", [0.0, -0.1, 0.5, 0.97, 1.0])
def test_un_alpha_hors_de_zero_un_demi_est_refuse(mauvais):
    with pytest.raises(ValueError, match="probabilité de"):
        TailForecast(var=np.full(10, 2.0), es=np.full(10, 2.5), alpha=mauvais)


def test_un_nan_dans_les_previsions_est_refuse():
    e = np.full(10, 2.5)
    e[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        TailForecast(var=np.full(10, 2.0), es=e, alpha=ALPHA)


def test_des_longueurs_differentes_sont_refusees():
    with pytest.raises(ValueError, match="même longueur"):
        TailForecast(var=np.full(10, 2.0), es=np.full(11, 2.5), alpha=ALPHA)


def test_le_rapport_de_queue_est_invariant_dechelle():
    """
    Propriété qui fonde la contrainte de cohérence : le rapport ES/VaR
    décrit la FORME de la queue, jamais son niveau.
    """
    base = plat(V_VRAI, E_VRAI, 50).ratio
    for c in (0.01, 3.0, 1000.0):
        assert plat(V_VRAI * c, E_VRAI * c, 50).ratio == pytest.approx(base, rel=1e-14)


# ---- validation des paramètres du test ----


@pytest.mark.parametrize(
    ("kwargs", "motif"),
    [
        ({"test": 3}, "test doit valoir"),
        ({"test": 0}, "test doit valoir"),
        ({"level": 0.0}, "level"),
        ({"level": 0.8}, "level"),
        ({"n_sim": 10}, "n_sim"),
    ],
)
def test_les_parametres_absurdes_du_test_sont_refuses(kwargs, motif):
    x = normaux(1000, 8)
    f = plat(V_VRAI, E_VRAI, x.size)
    kwargs.setdefault("n_sim", 300)
    with pytest.raises(ValueError, match=motif):
        acerbi_szekely(x, f, null=GaussianNull(), **kwargs)


def test_un_historique_trop_court_leve_insufficient_data():
    """
    À α = 2,5 %, il faut 400 observations pour espérer 10 dépassements.
    En dessous, la statistique repose sur une poignée de points, et le
    moteur refuse plutôt que de retourner un chiffre sans valeur.
    """
    n = int(MIN_EXPECTED_BREACHES / ALPHA) - 50
    x = normaux(n, 9)
    with pytest.raises(InsufficientData):
        acerbi_szekely(x, plat(V_VRAI, E_VRAI, n), null=GaussianNull(), n_sim=300)


def test_des_longueurs_discordantes_entre_realise_et_prevu_sont_refusees():
    x = normaux(1000, 10)
    with pytest.raises(ValueError, match="même longueur"):
        acerbi_szekely(x, plat(V_VRAI, E_VRAI, 999), null=GaussianNull(), n_sim=300)


# ══════════════════════════════════════════════════════════════════════
#  La loi nulle : explicite, et contrainte
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("nulle", "attendu"),
    [
        (GaussianNull(), 1.192778),
        (StudentTNull(8), 1.287902),
        (StudentTNull(5), 1.369953),
        (StudentTNull(3), 1.583556),
    ],
)
def test_les_rapports_de_queue_theoriques_sont_exacts(nulle, attendu):
    assert nulle.ratio(ALPHA) == pytest.approx(attendu, rel=1e-5)


def test_le_rapport_de_queue_croit_quand_la_queue_sepaissit():
    ratios = [StudentTNull(nu).ratio(ALPHA) for nu in (30, 10, 5, 3)]
    assert ratios == sorted(ratios)
    assert GaussianNull().ratio(ALPHA) < ratios[0]


def test_les_quatre_nulles_satisfont_le_protocol_sans_heriter_de_rien():
    """Sixième usage du typage structurel dans ce projet."""
    residus = np.random.default_rng(11).standard_normal(500)
    for nulle in (
        GaussianNull(),
        StudentTNull(5),
        EmpiricalNull(residus),
    ):
        assert isinstance(nulle, StandardizedNull)
        assert isinstance(nulle.name, str) and nulle.name
        tires = nulle.draw(1000, ALPHA, np.random.default_rng(0))
        assert tires.shape == (1000,)
        assert np.isfinite(tires).all()


def test_une_nulle_tiree_a_une_var_de_un_par_construction():
    """
    C'est la propriété qui rend le tirage compatible avec H₀ : une fois
    multiplié par la VaR annoncée, le chemin simulé a EXACTEMENT cette
    VaR. Sans elle, la « nulle » ne serait pas la nulle.
    """
    for nulle in (GaussianNull(), StudentTNull(5)):
        u = nulle.draw(2_000_000, ALPHA, np.random.default_rng(12))
        assert float(np.quantile(u, ALPHA)) == pytest.approx(-1.0, abs=0.01)


def test_une_nulle_incoherente_avec_les_previsions_est_refusee():
    """
    LE GARDE-FOU CENTRAL DE LA SÉANCE 13.

    Le couple (VaR, ES) fixe déjà le rapport de queue. Une nulle dont le
    rapport diffère contredit les prévisions qu'elle est censée tester,
    et sa valeur critique ne vaut rien. Le message doit l'expliquer et
    proposer la sortie.
    """
    x = normaux(1000, 13)
    f = plat(V_VRAI, E_VRAI, x.size)  # rapport gaussien ≈ 1,193
    with pytest.raises(ValueError, match="contredit donc"):
        acerbi_szekely(x, f, null=StudentTNull(3), n_sim=300)
    with pytest.raises(ValueError, match="EmpiricalNull"):
        acerbi_szekely(x, f, null=StudentTNull(3), n_sim=300)


def test_la_contrainte_de_coherence_peut_etre_levee_explicitement():
    x = normaux(1000, 14)
    f = plat(V_VRAI, E_VRAI, x.size)
    r = acerbi_szekely(
        x, f, null=StudentTNull(3), n_sim=300, check_ratio=False
    )
    assert r.null_name == "Student t(3)"


def test_la_nulle_empirique_retrouve_le_rapport_de_la_loi_quon_lui_donne():
    """
    Niveau 2 appliqué à la nulle elle-même : on lui donne des résidus
    de Student, elle doit retrouver le rapport de queue de la Student.
    La tolérance est large parce qu'un rapport de queue estimé sur
    N points n'en utilise que α·N.
    """
    for nu in (5, 8):
        r = np.random.default_rng(100 + nu).standard_t(nu, 60_000)
        empirique = EmpiricalNull(r).ratio(ALPHA)
        theorique = StudentTNull(nu).ratio(ALPHA)
        assert empirique == pytest.approx(theorique, rel=0.08), f"nu={nu}"


def test_la_nulle_empirique_refuse_trop_peu_de_points():
    with pytest.raises(InsufficientData):
        EmpiricalNull(np.random.default_rng(15).standard_normal(50))


def test_la_nulle_empirique_refuse_des_residus_qui_ne_sont_pas_centres():
    with pytest.raises(ValueError, match="ne ressemblent pas"):
        EmpiricalNull(np.full(500, 3.0) + 0.01).ratio(ALPHA)


def test_un_nu_trop_petit_est_refuse_car_lesperance_de_queue_nexiste_pas():
    with pytest.raises(ValueError, match="nu doit être"):
        StudentTNull(0.9)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 2 — calibration contre une vérité DÉRIVÉE
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("niveau_annonce", [0.0125, 0.025, 0.05, 0.10])
def test_le_modele_au_mauvais_niveau_donne_un_z2_predit_exactement(niveau_annonce):
    """
    LE TEST LE PLUS FORT DU FICHIER, parce que la cible est DÉRIVÉE et
    non mesurée.

    Un modèle qui annonce les VaR/ES du niveau p′ en prétendant tenir α
    dépasse avec probabilité p′. L'espérance de la somme des
    dépassements divisée par l'ES annoncée vaut donc −T·p′, et

        E[Z₂] = 1 − p′/α

    C'est une prédiction théorique indépendante du code. Si le code la
    retrouve pour quatre valeurs de p′, les conventions de signe, de
    normalisation et de seuil sont toutes justes à la fois.
    """
    v, e = var_es_normale(niveau_annonce)
    x = normaux(200_000, 16)
    z2, _ = tail_statistic(x, plat(v, e, x.size), test=2)
    attendu = 1.0 - niveau_annonce / ALPHA
    assert z2 == pytest.approx(attendu, abs=0.05)


@pytest.mark.parametrize("niveau_annonce", [0.0125, 0.05, 0.10])
def test_le_test_1_reste_aveugle_quel_que_soit_le_mauvais_niveau(niveau_annonce):
    """
    L'ANGLE MORT, VERROUILLÉ.

    Le même modèle que ci-dessus : son niveau est faux d'un facteur
    allant de deux à quatre, et Z₁ reste collé à zéro. Ce test existe
    pour qu'on ne « corrige » jamais Z₁ en croyant réparer un bug : ce
    comportement est sa définition, et c'est pour cela qu'il ne doit pas
    servir de validation.
    """
    v, e = var_es_normale(niveau_annonce)
    x = normaux(200_000, 17)
    z1, _ = tail_statistic(x, plat(v, e, x.size), test=1)
    assert z1 == pytest.approx(0.0, abs=0.03)


def test_un_modele_exact_nest_pas_rejete_et_un_modele_faux_lest():
    x = normaux(4000, 18)
    exact = acerbi_szekely(
        x, plat(V_VRAI, E_VRAI, x.size), null=GaussianNull(), n_sim=4000
    )
    faux = acerbi_szekely(
        x, plat(V_VRAI * 0.75, E_VRAI * 0.75, x.size), null=GaussianNull(), n_sim=4000
    )
    assert not exact.rejected
    assert faux.rejected
    assert faux.verdict == REJETE
    assert "dangereux" in faux.direction


# ══════════════════════════════════════════════════════════════════════
#  Fissler-Ziegel : élicitabilité conjointe
# ══════════════════════════════════════════════════════════════════════


def test_la_perte_fz_est_minimisee_a_la_verite():
    """
    L'AFFIRMATION DU DOCSTRING, VÉRIFIÉE.

    L'élicitabilité conjointe se définit exactement par « l'espérance de
    la perte est minimisée au vrai couple ». C'est donc testable, et ma
    première écriture de cette formule ne passait PAS ce test — trois
    variantes de mémoire ont été réfutées avant de dériver la bonne
    depuis la forme générale de Fissler-Ziegel.
    """
    x = np.random.default_rng(19).standard_normal(400_000)
    reference = float(fz_loss(x, plat(V_VRAI, E_VRAI, x.size)).mean())

    for fv, fe in ((0.92, 1.0), (1.08, 1.0), (1.0, 0.92), (1.0, 1.08), (0.95, 1.05)):
        v, e = V_VRAI * fv, E_VRAI * fe
        if e <= v:
            continue
        perte = float(fz_loss(x, plat(v, e, x.size)).mean())
        assert perte > reference, f"VaR×{fv}, ES×{fe} ne doit pas battre la vérité"


def test_une_var_fausse_deplace_les_es_optimales():
    """
    Le contenu opérationnel de « conjointement élicitable » : avec une
    VaR fausse, l'ES qui minimise la perte n'est plus la vraie ES. Donc
    on ne peut pas évaluer une prévision d'ES sans s'engager sur une VaR.
    """
    x = np.random.default_rng(20).standard_normal(300_000)

    def es_optimale(facteur_var: float) -> float:
        meilleur, arg = math.inf, math.nan
        for fe in np.linspace(0.85, 1.35, 51):
            v, e = V_VRAI * facteur_var, E_VRAI * fe
            if e <= v:
                continue
            m = float(fz_loss(x, plat(v, e, x.size)).mean())
            if m < meilleur:
                meilleur, arg = m, fe
        return arg

    assert es_optimale(1.00) == pytest.approx(1.0, abs=0.02)
    assert es_optimale(0.75) > 1.05
    assert es_optimale(1.25) > 1.02


def test_la_comparaison_prefere_le_modele_exact():
    x = np.random.default_rng(21).standard_normal(6000)
    exact = plat(V_VRAI, E_VRAI, x.size)
    faux = plat(V_VRAI * 0.85, E_VRAI * 0.85, x.size)
    c = compare_tail_models(x, exact, faux, name_a="exact", name_b="faux")
    assert c.winner == "exact"
    assert c.significant
    assert c.n_obs == x.size
    assert "exact est meilleur" in str(c)


def test_comparer_un_modele_a_lui_meme_ne_donne_aucun_ecart():
    x = np.random.default_rng(22).standard_normal(1000)
    f = plat(V_VRAI, E_VRAI, x.size)
    c = compare_tail_models(x, f, f)
    assert c.difference == pytest.approx(0.0, abs=1e-15)
    assert not c.significant
    assert "non significatif" in str(c)


def test_comparer_deux_niveaux_differents_est_refuse():
    x = np.random.default_rng(23).standard_normal(1000)
    a = plat(V_VRAI, E_VRAI, x.size, alpha=0.025)
    b = plat(V_VRAI, E_VRAI, x.size, alpha=0.01)
    with pytest.raises(ValueError, match="même niveau"):
        compare_tail_models(x, a, b)


# ══════════════════════════════════════════════════════════════════════
#  CAUSALITÉ — la discipline imposée par la séance 8
# ══════════════════════════════════════════════════════════════════════


def serie(n: int, graine: int) -> pd.Series:
    return pd.Series(
        np.random.default_rng(graine).standard_normal(n) * 0.012,
        index=pd.bdate_range("2018-01-02", periods=n),
    )


def test_aucune_prevision_de_queue_ne_voit_sa_propre_date_ni_le_futur():
    s = serie(900, 30)
    positions = {d: i for i, d in enumerate(s.index)}
    vus: list[tuple[int, int]] = []

    def espion(historique: pd.Series) -> TailRisk:
        vus.append((positions[historique.index[0]], positions[historique.index[-1]]))
        return parametric_var(historique, confidence=1.0 - ALPHA)

    lookback, step = 400, 20
    realises, prev = rolling_tail_forecasts(
        s, espion, lookback=lookback, step=step
    )

    assert vus
    for rang, (premier, dernier) in enumerate(vus):
        decision = lookback + rang * step
        assert dernier == decision - 1, (
            f"l'estimation du rang {rang} a lu jusqu'à {dernier} alors que sa "
            f"date de décision est {decision}"
        )
        assert dernier - premier + 1 == lookback

    assert realises.size == prev.n_forecasts
    assert prev.alpha == pytest.approx(ALPHA)


def test_perturber_le_futur_ne_change_pas_les_previsions_passees():
    s = serie(900, 31)
    corrompu = s.copy()
    corrompu.iloc[600:] *= 80.0

    def rejeu(donnees: pd.Series) -> np.ndarray:
        _, prev = rolling_tail_forecasts(
            donnees,
            lambda h: historical_var(h, confidence=1.0 - ALPHA, min_obs=100),
            lookback=400,
            step=20,
        )
        return prev.var

    propre = rejeu(s)
    sale = rejeu(corrompu)

    n_intactes = (600 - 400) // 20
    assert n_intactes >= 3
    for i in range(n_intactes):
        # Une prévision dont l'historique se termine avant l'indice 600
        # ne PEUT pas avoir changé. Égalité au bit près, pas approchée.
        assert sale[i * 20] == propre[i * 20]


def test_un_estimateur_qui_change_de_niveau_en_cours_de_route_est_refuse():
    """
    Un chemin de prévisions à niveau de confiance variable donnerait un
    `alpha` arbitraire pour tout le test. Mieux vaut échouer bruyamment.
    """
    s = serie(700, 32)
    compteur = {"n": 0}

    def instable(historique: pd.Series) -> TailRisk:
        compteur["n"] += 1
        conf = 0.975 if compteur["n"] < 3 else 0.99
        return parametric_var(historique, confidence=conf)

    with pytest.raises(ValueError, match="changé de niveau"):
        rolling_tail_forecasts(s, instable, lookback=300, step=20)


@pytest.mark.parametrize(
    ("kwargs", "motif"),
    [({"lookback": 1}, "lookback"), ({"step": 0}, "step")],
)
def test_le_rejeu_refuse_les_parametres_absurdes(kwargs, motif):
    s = serie(500, 33)
    with pytest.raises(ValueError, match=motif):
        rolling_tail_forecasts(s, parametric_var, **kwargs)


def test_un_historique_plus_court_que_le_lookback_leve_insufficient_data():
    s = serie(100, 34)
    with pytest.raises(InsufficientData):
        rolling_tail_forecasts(s, parametric_var, lookback=400)


def test_le_rejeu_produit_un_chemin_utilisable_par_le_test():
    """
    Intégration de bout en bout : des prix synthétiques jusqu'au verdict,
    en passant par var.py et shortfall.py. C'est le test qui attraperait
    une incompatibilité de conventions entre les deux modules.
    """
    s = serie(2000, 35)
    realises, prev = rolling_tail_forecasts(
        s,
        lambda h: parametric_var(h, confidence=1.0 - ALPHA),
        lookback=500,
        step=20,
    )
    residus = realises / (prev.var / V_VRAI)
    r = acerbi_szekely(
        realises,
        prev,
        null=EmpiricalNull(residus),
        n_sim=2000,
        check_ratio=False,
    )
    assert r.n_obs == realises.size
    assert math.isfinite(r.statistic)
    assert math.isfinite(r.critical_value)
    # Données gaussiennes, modèle gaussien : il ne doit pas être rejeté.
    assert not r.rejected, str(r)


# ══════════════════════════════════════════════════════════════════════
#  Cohérence avec la démonstration
# ══════════════════════════════════════════════════════════════════════


def test_la_version_vectorisee_du_demo_egale_le_module():
    """
    `examples/seance13.py` recalcule Z₂ en une opération matricielle pour
    tenir en vingt secondes au lieu de vingt minutes. Une seconde
    implémentation est une occasion de divergence silencieuse — exactement
    ce que le patron de l'oracle vérifié de la séance 6 a appris à ce
    projet. Donc on la verrouille.
    """
    n_traj, n_jours = 50, 600
    matrice = np.random.default_rng(36).standard_normal((n_traj, n_jours))

    depassement = matrice < -V_VRAI
    vectorise = (matrice * depassement).sum(axis=1) / E_VRAI / (n_jours * ALPHA) + 1.0

    f = plat(V_VRAI, E_VRAI, n_jours)
    for i in range(n_traj):
        z, _ = tail_statistic(matrice[i], f, test=2)
        assert vectorise[i] == pytest.approx(z, rel=1e-12)


# ══════════════════════════════════════════════════════════════════════
#  NIVEAU 3 — couverture Monte Carlo du TEST lui-même
# ══════════════════════════════════════════════════════════════════════


def _z2_vectorise(matrice: np.ndarray, v: float, e: float, alpha: float) -> np.ndarray:
    depassement = matrice < -v
    return (matrice * depassement).sum(axis=1) / e / (matrice.shape[1] * alpha) + 1.0


@pytest.mark.slow
def test_la_taille_du_test_est_son_niveau_nominal():
    """
    On ne valide pas la statistique, on valide le TEST : sous H₀, il doit
    rejeter exactement au niveau annoncé.
    """
    t, n_sim, niveau = 1000, 20_000, 0.05
    g = np.random.default_rng(40)
    nulle = _z2_vectorise(
        V_VRAI * GaussianNull().draw(n_sim * t, ALPHA, g).reshape(n_sim, t),
        V_VRAI,
        E_VRAI,
        ALPHA,
    )
    critique = float(np.quantile(nulle, niveau))

    echantillons = np.random.default_rng(41).standard_normal((20_000, t))
    z = _z2_vectorise(echantillons, V_VRAI, E_VRAI, ALPHA)
    taux = float(np.mean(z <= critique))
    assert taux == pytest.approx(niveau, abs=0.01)


@pytest.mark.slow
def test_z2_est_plus_puissant_que_kupiec():
    """
    LE RÉSULTAT MESURÉ DE LA SÉANCE 13, verrouillé comme inégalité.

    Même α, même niveau, mêmes trajectoires. On n'assère pas les valeurs
    exactes — elles dépendent du générateur — mais la domination et son
    ordre de grandeur.
    """
    t, n_rep, niveau, c = 1000, 1500, 0.05, 0.90
    g = np.random.default_rng(42)
    nulle = _z2_vectorise(
        V_VRAI * GaussianNull().draw(20_000 * t, ALPHA, g).reshape(20_000, t),
        V_VRAI,
        E_VRAI,
        ALPHA,
    )
    critique = float(np.quantile(nulle, niveau))

    echantillons = np.random.default_rng(43).standard_normal((n_rep, t))
    v, e = V_VRAI * c, E_VRAI * c

    puissance_z2 = float(np.mean(_z2_vectorise(echantillons, v, e, ALPHA) <= critique))
    puissance_kupiec = float(
        np.mean(
            [
                kupiec_test(
                    pd.Series(echantillons[i]), v, confidence=1.0 - ALPHA
                ).rejected
                for i in range(n_rep)
            ]
        )
    )

    assert puissance_z2 > puissance_kupiec + 0.05, (
        f"Z₂ {puissance_z2:.1%} contre Kupiec {puissance_kupiec:.1%}"
    )
    assert puissance_z2 > 0.70


@pytest.mark.slow
def test_la_loi_nulle_depend_de_la_queue_mais_faiblement():
    """
    MA PRÉDICTION ÉTAIT FAUSSE, et ce test l'enregistre.

    J'attendais que la dépendance distributionnelle démolisse le test.
    Elle est réelle — supposer la normalité quand la vérité est une t(3)
    fait monter les faux positifs au-dessus du niveau — mais modérée.
    On verrouille les deux moitiés : l'effet existe, ET il reste petit.
    Si une version future le rendait dramatique, ce serait une
    régression à examiner, pas un progrès.
    """
    t, n_sim, niveau = 1000, 30_000, 0.05

    def critique_et_taux(nulle, crit_reference=None):
        g = np.random.default_rng(44)
        ratio = nulle.ratio(ALPHA)
        e_loi = V_VRAI * ratio
        u = nulle.draw(n_sim * t, ALPHA, g).reshape(n_sim, t)
        tires = _z2_vectorise(V_VRAI * u, V_VRAI, e_loi, ALPHA)
        c = float(np.quantile(tires, niveau))
        taux = (
            None
            if crit_reference is None
            else float(np.mean(tires <= crit_reference))
        )
        return c, taux

    crit_gauss, _ = critique_et_taux(GaussianNull())
    crit_t3, taux_t3 = critique_et_taux(StudentTNull(3), crit_gauss)

    # L'effet existe et va dans le mauvais sens : on rejette trop.
    assert taux_t3 > niveau + 0.01
    # Mais il reste modéré : pas de doublement du taux de faux positifs.
    assert taux_t3 < 2.0 * niveau
    # La valeur critique correcte est plus large, pas plus étroite.
    assert crit_t3 < crit_gauss
