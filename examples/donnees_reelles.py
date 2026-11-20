"""
La chaîne complète, sur de vraies actions.

    pip install yfinance
    python examples/donnees_reelles.py

CE QUE FAIT CE SCRIPT, ET POURQUOI IL EST SÉPARÉ DES AUTRES
===========================================================

Les neuf démonstrations précédentes tournent hors ligne, sur des
données dont la vérité est connue. C'est une décision méthodologique
tenue depuis la séance 1 : **on ne valide pas un estimateur contre des
données dont on ignore la bonne réponse.**

Celui-ci fait autre chose. Il ne valide rien — il EXPOSE le moteur à ce
qu'il rencontrera en production, et montre ce qui se passe quand les
hypothèses lâchent. C'est pour cela qu'il vit dans un fichier à part,
n'est couvert par aucun test, et dépend du réseau.

L'univers mélange délibérément la Bourse de Toronto et celle de New
York. Leurs jours fériés diffèrent — fête de Victoria, fête du Canada
et congé civique d'un côté, Memorial Day, 4 juillet et Thanksgiving de
novembre de l'autre. Le désalignement de calendriers de la séance 9
n'est donc pas une hypothèse d'école : il est dans les données dès
qu'on ouvre le fichier.

⚠️ AVERTISSEMENTS QUI VALENT POUR TOUT CE QUE CE SCRIPT AFFICHE
---------------------------------------------------------------
1. BIAIS DU SURVIVANT. Yahoo ne sert que les titres encore cotés. Les
   faillites ont disparu de la base. Tout résultat historique calculé
   ici surestime donc le passé, et aucune correction n'est possible
   avec des données gratuites.
2. UN SEUL ÉCHANTILLON. Une période, un univers. La séance 8 a montré
   ce que vaut un backtest unique : la question n'est pas « combien
   a-t-il rapporté » mais « combien de fois ai-je regardé ».
3. LICENCE. Les données Yahoo sont à usage personnel.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from nkrisk.backtest import (
    EqualWeight,
    MinimumVariance,
    deflated_sharpe_ratio,
    detect_lookahead,
    walk_forward,
)
from nkrisk.covariance import ledoit_wolf
from nkrisk.data.market import YahooSource, load_prices
from nkrisk.data.quality import (
    align_prices,
    compare_alignment,
    inspect_prices,
    observations_per_year,
)
from nkrisk.ewma import ewma_covariance, ewma_weights
from nkrisk.factors import InsufficientStructure, attribute_risk, pca_factor_model
from nkrisk.returns import annualized_volatility, simple_returns

LARGE = 76

# Toronto et New York, volontairement mélangées : leurs calendriers
# diffèrent, et c'est précisément ce qu'on veut voir.
UNIVERS = [
    "RY.TO", "TD.TO", "BNS.TO", "ENB.TO", "CNR.TO", "SHOP.TO", "ABX.TO",
    "AAPL", "MSFT", "JPM", "XOM", "JNJ",
]
DEBUT = "2015-01-01"
LOOKBACK, PAS = 252, 21


def titre(texte: str) -> None:
    print()
    print("═" * LARGE)
    print(f"  {texte}")
    print("═" * LARGE)
    print()


def main() -> int:
    titre("1. CHARGEMENT")

    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("  yfinance n'est pas installé.")
        print("  Lance : pip install yfinance")
        return 1

    print(f"  {len(UNIVERS)} titres, depuis {DEBUT}, prix ajustés.")
    print("  (téléchargement en cours, quelques secondes)")
    print()

    charge = load_prices(UNIVERS, YahooSource(), start=DEBUT, strict=False)
    for ligne in str(charge).splitlines():
        print(f"    {ligne}")

    if len(charge.series) < 4:
        print()
        print("  Trop peu de séries chargées pour continuer. Vérifie ta")
        print("  connexion et les symboles.")
        return 1

    series = charge.series

    # ── 2. QUALITÉ ───────────────────────────────────────────────────
    titre("2. INSPECTION DE QUALITÉ")

    rapport = inspect_prices(series)
    for ligne in str(rapport).splitlines():
        print(f"    {ligne}")

    print()
    print("  ⚠️ CETTE NARRATION DISAIT N'IMPORTE QUOI JUSQU'À LA SÉANCE 10.")
    print("     Elle annonçait « 7 % de calendriers désalignés » en citant")
    print("     un chiffre qui additionne deux causes, deux lignes sous un")
    print("     rapport qui affichait 3,9 % pour la bonne. Dixième dérive")
    print("     de narration du projet, et la plus bête : les deux nombres")
    print("     étaient à l'écran en même temps.")
    print()
    manque_total = 1 - rapport.calendar_overlap
    manque_feries = 1 - rapport.holiday_overlap
    print(f"  → {manque_total:.1%} des dates disponibles ne sont pas communes")
    print("    à tous les titres. Mais ce chiffre mélange deux causes.")
    print()
    print(f"  → Désalignement de calendriers SEUL : {manque_feries:.1%} de la")
    print("    fenêtre commune. C'est l'écart Toronto / New York, et il n'a")
    print("    rien d'une hypothèse : il est dans les données.")
    print()
    print(f"  → Le reste — {manque_total - manque_feries:.1%} — vient de titres dont")
    print("    l'historique est plus court. Remède opposé : on n'aligne pas")
    print("    un calendrier, on décide d'exclure le titre ou d'accepter de")
    print("    perdre le début pour tout le monde.")
    rapport.raise_if_blocking()

    # ── 3. LE CHOIX D'ALIGNEMENT, SUR DE VRAIES DONNÉES ──────────────
    titre("3. CE QUE LE CHOIX D'ALIGNEMENT COÛTE ICI")

    comparaison = compare_alignment(series)
    for ligne in str(comparaison).splitlines():
        print(f"    {ligne}")

    decalage = comparaison.volatility_shift
    ecart_corr = (
        comparaison.mean_correlation["report"]
        - comparaison.mean_correlation["intersection"]
    )
    print()
    print(f"  → Volatilités : le report s'écarte de "
          f"{decalage.mean():+.2%} en moyenne.")
    print(f"  → Corrélation moyenne : "
          f"{ecart_corr:+.3f}, dans le sens attendu par la séance 9.")
    print()
    print("  → On retient l'intersection, annualisée par le nombre")
    print("    d'observations réellement présentes.")

    prix = align_prices(series, "intersection")
    par_an = observations_per_year(prix.index)
    print()
    print(f"    {len(prix)} dates retenues, {par_an:.1f} observations par an")
    print(f"    (le code naïf aurait écrit 252, soit "
          f"{252 / par_an - 1:+.1%} d'erreur d'annualisation)")

    rendements = simple_returns(prix).dropna()

    # ── 4. VOLATILITÉS ───────────────────────────────────────────────
    titre("4. VOLATILITÉS, AVEC LEUR INCERTITUDE")

    annuel = round(par_an)
    print(f"    {'titre':<10} {'volatilité annualisée':>45}")
    for col in rendements.columns:
        v = annualized_volatility(rendements[col], periods_per_year=annuel)
        print(f"    {col:<10} {v.format(2, percent=True):>45}")

    # ── 5. MODÈLE À FACTEURS ─────────────────────────────────────────
    titre("5. MODÈLE À FACTEURS SUR DE VRAIES ACTIONS")

    try:
        modele = pca_factor_model(rendements)
    except InsufficientStructure as exc:
        print(f"    {exc}")
        print()
        print("  → Le critère de Marchenko-Pastur refuse d'extraire un")
        print("    facteur. Sur un univers d'actions, ce serait très")
        print("    surprenant — vérifie la période et les titres.")
        return 1

    for ligne in modele.diagnostic().splitlines():
        print(f"    {ligne}")

    print()
    expo = modele.exposures_frame()
    expo["R²"] = modele.systematic_share()
    print(expo.sort_values("F1", ascending=False).round(3).to_string())

    positifs = int((modele.exposures[:, 0] > 0).sum())
    print()
    print(f"  → Le premier facteur a une exposition positive sur "
          f"{positifs} des {modele.n_assets}")
    print("    titres. C'est la signature du facteur marché, et elle")
    print("    apparaît ici sans qu'on l'ait demandée — contrairement aux")
    print("    univers simulés, où je la fabriquais exprès.")

    poids = pd.Series(
        np.full(modele.n_assets, 1.0 / modele.n_assets),
        index=list(modele.assets),
    )
    attribution = attribute_risk(poids, modele, periods_per_year=annuel)
    print()
    for ligne in str(attribution).splitlines():
        print(f"    {ligne}")
    print()
    print(f"  → {attribution.systematic_share:.0%} du risque vient des")
    print("    facteurs communs. Cette part ne se diversifie pas en")
    print("    ajoutant des lignes.")

    # ── 6. BACKTEST ──────────────────────────────────────────────────
    titre("6. BACKTEST, COÛTS COMPRIS")

    def facteurs_ou_contraction(fenetre: pd.DataFrame):
        """
        Modèle à facteurs, avec repli EXPLICITE sur la contraction.

        Le repli est écrit ici, dans la stratégie, et non dans le
        moteur — c'est une décision d'investissement, pas une commodité
        d'implémentation. Le moteur, lui, n'a pas le droit de se
        rabattre en silence : il produirait une courbe ne correspondant
        à aucune stratégie exécutable.
        """
        try:
            return pca_factor_model(fenetre).covariance()
        except InsufficientStructure:
            return ledoit_wolf(fenetre)

    strategies = [
        EqualWeight(),
        MinimumVariance(ledoit_wolf, "var. min. Ledoit-Wolf"),
        MinimumVariance(facteurs_ou_contraction, "var. min. facteurs"),
    ]

    print(f"    {'stratégie':<24} {'vol réalisée':>13} {'Sharpe':>9} "
          f"{'turnover':>10} {'capital':>9}")
    resultats = {}
    for strat in strategies:
        res = walk_forward(
            rendements, strat, lookback=LOOKBACK, step=PAS, cost_bps=20
        )
        resultats[strat.name] = res
        vol = res.realized_volatility(periods_per_year=annuel)
        sr = res.sharpe(periods_per_year=annuel)
        print(f"    {strat.name:<24} {vol.value:>12.2%} {sr.value:>9.2f} "
              f"{res.annual_turnover(annuel):>9.2f}× "
              f"{res.equity.iloc[-1]:>8.2f}")

    print()
    print("  Calibration — le risque annoncé contre le risque subi :")
    print()
    for nom, res in resultats.items():
        cal = res.calibration(periods_per_year=annuel)
        if cal is None:
            print(f"    {nom:<24} (ne prédit aucun risque)")
        else:
            print(f"    {nom:<24} {cal.format(2)}")

    print()
    print("  → Compare ces rapports à ceux de la séance 8, mesurés sur")
    print("    données simulées. Sur de vraies données, les rendements ne")
    print("    sont pas indépendants d'un jour à l'autre et la volatilité")
    print("    change de régime. Les deux violent les hypothèses du")
    print("    moteur, et la calibration s'en ressent.")

    # ── 6b. D'OÙ VIENT L'ÉCART ? ─────────────────────────────────────
    titre("6b. L'ÉCART DE CALIBRATION EST-IL UNIFORME OU CONCENTRÉ ?")

    print("  Un chiffre agrégé ne dit pas la même chose selon que le")
    print("  modèle se trompe un peu tout le temps, ou énormément")
    print("  pendant quelques mois. Les deux appellent des remèdes")
    print("  opposés, et seule la répartition permet de choisir.")
    print()

    for nom in ("var. min. Ledoit-Wolf", "var. min. facteurs"):
        res = resultats[nom]
        print(f"    ── {nom} " + "─" * (56 - len(nom)))
        for ligne in res.calibration_diagnosis(
            periods_per_year=annuel
        ).splitlines():
            print(f"      {ligne}")
        chemin = res.calibration_path(periods_per_year=annuel)
        pires = chemin.nsmallest(4, "rapport")
        print()
        print("      les quatre fenêtres les plus mal anticipées :")
        for date, ligne in pires.iterrows():
            print(f"        {date.date()}  annoncé {ligne['prédite']:>6.1%}  "
                  f"subi {ligne['réalisée']:>6.1%}  "
                  f"rapport {ligne['rapport']:.2f}")
        print()

    print("  → Si les pires fenêtres tombent sur des dates que tu")
    print("    reconnais — mars 2020, septembre 2022 — le diagnostic est")
    print("    établi : le modèle ne se trompe pas de niveau, il arrive")
    print("    en retard. Une fenêtre glissante de 252 jours ne voit la")
    print("    tempête qu'une fois qu'elle est entrée dedans.")

    # ── 6c. SÉRIE CONTRE MATRICE, SUR TES DONNÉES ────────────────────
    titre("6c. L'EWMA TIENT-ELLE SES PROMESSES ICI ?")

    print("  La séance 11 a établi en simulation que la pondération")
    print("  exponentielle gagne sur une SÉRIE et perd via une MATRICE :")
    print("  son avantage est en temps, son coût est en dimension.")
    print()
    print("  Les simulations proposent. Voici ce que disent tes données.")
    print()

    horizon = 21
    fenetre = LOOKBACK
    serie_ptf = rendements.mean(axis=1).to_numpy()
    w_eq = np.full(rendements.shape[1], 1.0 / rendements.shape[1])

    predicteurs = {
        "série, ewma λ=0,97": ("serie", 0.97),
        "série, ewma λ=0,94": ("serie", 0.94),
        "série, fenêtre 252j": ("serie", None),
        "matrice, ewma λ=0,97": ("matrice", 0.97),
        "matrice, fenêtre + LW": ("matrice", None),
    }
    scores: dict[str, float] = {}
    for nom, (chemin, lam) in predicteurs.items():
        predits, realises = [], []
        for t in range(fenetre, len(serie_ptf) - horizon, horizon):
            realises.append(
                float(np.std(serie_ptf[t: t + horizon], ddof=1))
            )
            if chemin == "serie":
                bloc = serie_ptf[t - fenetre: t]
                if lam is None:
                    predits.append(float(np.std(bloc, ddof=1)))
                else:
                    poids = ewma_weights(len(bloc), lam)
                    predits.append(float(np.sqrt(np.sum(poids * bloc**2))))
            else:
                fen = rendements.iloc[t - fenetre: t]
                cov = (
                    ledoit_wolf(fen)
                    if lam is None
                    else ewma_covariance(fen, decay=lam)
                )
                predits.append(float(np.sqrt(w_eq @ cov.matrix @ w_eq)))
        z = np.array(realises) / np.array(predits)
        scores[nom] = float(np.mean(z**2))

    print(f"    {'prédicteur':<26} {'E[z²]':>8} {'écart à 1':>11}")
    print("    " + "-" * 47)
    for nom, v in scores.items():
        print(f"    {nom:<26} {v:>8.2f} {abs(v - 1):>11.2f}")

    def meilleur(prefixe: str) -> tuple[str, float]:
        candidats = {k: v for k, v in scores.items() if k.startswith(prefixe)}
        gagnant = min(candidats, key=lambda k: abs(candidats[k] - 1))
        return gagnant, candidats[gagnant]

    g_serie, v_serie = meilleur("série")
    g_matrice, v_matrice = meilleur("matrice")
    print()
    print(f"  → Meilleur sur une série  : {g_serie} ({v_serie:.2f})")
    print(f"  → Meilleur via une matrice : {g_matrice} ({v_matrice:.2f})")
    print()
    serie_ewma = scores["série, ewma λ=0,97"]
    serie_fen = scores["série, fenêtre 252j"]
    if abs(serie_ewma - 1) < abs(serie_fen - 1):
        print("  → Sur une série, l'EWMA l'emporte : la simulation tient.")
    else:
        print("  → Sur une série, l'EWMA NE l'emporte PAS ici. La")
        print("    simulation ne tient pas sur ces données, et c'est le")
        print("    genre de démenti qui vaut plus qu'une confirmation.")
    mat_ewma = scores["matrice, ewma λ=0,97"]
    mat_fen = scores["matrice, fenêtre + LW"]
    if abs(mat_fen - 1) < abs(mat_ewma - 1):
        print("  → Via une matrice, la fenêtre longue reprend l'avantage :")
        print("    le coût dimensionnel est bien là.")
    else:
        print("  → Via une matrice, l'EWMA tient aussi. Avec douze actifs")
        print("    seulement, le coût dimensionnel peut rester sous le")
        print("    gain temporel — la simulation le montrait croissant")
        print("    avec N.")

    # ── 7. AUDIT DE CAUSALITÉ ────────────────────────────────────────
    titre("7. AUDIT DE CAUSALITÉ")

    audit = detect_lookahead(
        rendements,
        lambda _d: MinimumVariance(ledoit_wolf, "var. min."),
        lookback=LOOKBACK,
        step=PAS,
    )
    for ligne in str(audit).splitlines():
        print(f"    {ligne}")
    print()
    print("  → Même vérification que sur données simulées. Le moteur n'a")
    print("    aucune raison de se comporter autrement ici, et c'est")
    print("    précisément ce qu'on veut constater plutôt que supposer.")

    # ── 8. BIAIS DE SÉLECTION ────────────────────────────────────────
    titre("8. ET MAINTENANT, LA QUESTION QUI FÂCHE")

    meilleur = max(
        resultats.items(),
        key=lambda kv: kv[1].sharpe(periods_per_year=annuel).value,
    )
    nom, res = meilleur
    sr = res.sharpe(periods_per_year=annuel)
    print(f"  La meilleure des trois est « {nom} », "
          f"Sharpe {sr.value:.2f}.")
    print()
    try:
        for essais in (1, 3, 100):
            d = deflated_sharpe_ratio(
                res.returns, n_trials=essais, periods_per_year=annuel
            )
            etat = "survit" if d.survives else "ne survit pas"
            print(f"    {essais:>4} essai(s) → probabilité "
                  f"{d.probability:>6.1%}   {etat}")
    except ValueError as exc:
        print(f"    Sharpe dégonflé non calculable : {exc}")

    print()
    print("  → « 3 essais » est le nombre honnête pour ce script : trois")
    print("    stratégies comparées. Mais si tu as choisi ces douze")
    print("    titres parce que tu les connais, ou cette période parce")
    print("    qu'elle t'arrangeait, le vrai nombre d'essais est bien")
    print("    plus grand — et il n'est écrit nulle part.")
    print()
    print("  → C'est la limite que ce script ne peut pas franchir, et")
    print("    aucun outil ne le peut : le nombre d'essais est un fait")
    print("    sur TOI, pas sur les données.")

    print()
    print("═" * LARGE)
    print("  Fin. Rien ici ne constitue un conseil en investissement.")
    print("═" * LARGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
