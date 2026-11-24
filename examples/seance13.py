"""
Séance 13 — Backtest d'Expected Shortfall : ce que coûte la cohérence.

La séance 3 a démontré que la VaR n'est pas une mesure de risque
cohérente. Bâle en a tiré la conséquence et l'a remplacée, dans FRTB,
par l'Expected Shortfall à 97,5 %. L'ES est sous-additive, elle regarde
la taille des pertes extrêmes et pas seulement leur fréquence.

Cette séance mesure le prix de ce changement. Quatre résultats, dont
deux que je n'avais pas prévus et un où ma prédiction était fausse :

  1. Le test 1 d'Acerbi-Székely a un angle mort béant : un modèle qui
     dépasse DEUX FOIS trop souvent le passe sans broncher.

  2. Le test 2 est nettement plus puissant que Kupiec, et récupère une
     partie de la puissance que la séance 3 avait trouvée manquante.

  3. La loi nulle d'un test d'ES n'est pas libre : le couple (VaR, ES)
     fixe déjà le rapport de queue. C'est ce qui rend le test possible,
     et c'est ce qui le rend dépendant d'une hypothèse que Kupiec n'a
     pas besoin de faire.

  4. J'avais prédit que cette dépendance serait dramatique. **Elle est
     modérée**, et le chiffre est dans la partie 4.

Aucun nombre de ce fichier n'est écrit à la main.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from nkrisk.shortfall import (
    LEVEL_FRTB,
    EmpiricalNull,
    GaussianNull,
    StudentTNull,
    TailForecast,
    acerbi_szekely,
    compare_tail_models,
    fz_loss,
    tail_statistic,
)
from nkrisk.var import kupiec_test

SEP = "=" * 70
ALPHA = LEVEL_FRTB
T = 1000
NIVEAU = 0.05
N_SIM_NULLE = 6000
N_SIM_PARTIE4 = 40_000
N_REP = 2000


def var_es_normale(p: float) -> tuple[float, float]:
    """VaR et ES exactes d'une N(0,1) au niveau de queue p, en magnitudes."""
    q = float(stats.norm.ppf(p))
    es = float(stats.norm.expect(lambda x: -x, lb=-np.inf, ub=q)) / p
    return -q, es


def plat(valeur_var: float, valeur_es: float, n: int = T, nom: str = "") -> TailForecast:
    return TailForecast(
        var=np.full(n, valeur_var),
        es=np.full(n, valeur_es),
        alpha=ALPHA,
        method=nom,
    )


def z2_vectorise(matrice: np.ndarray, v: float, e: float) -> np.ndarray:
    """
    Z₂ pour chaque LIGNE d'une matrice de trajectoires.

    Les boucles Python sur des dizaines de milliers de trajectoires
    coûtent des minutes ; la même chose en une opération matricielle
    coûte une seconde. La formule est identique à celle du module, et
    `test_la_version_vectorisee_du_demo_egale_le_module` le verrouille.
    """
    depassement = matrice < -v
    somme = (matrice * depassement).sum(axis=1) / e
    return somme / (matrice.shape[1] * ALPHA) + 1.0


V_VRAI, E_VRAI = var_es_normale(ALPHA)


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 1 — L'INSTRUMENT, CALIBRÉ CONTRE UNE VÉRITÉ CONNUE")
print(SEP)
print()
print(f"Niveau de Bâle : ES à {1 - ALPHA:.1%}, donc une queue de {ALPHA:.1%}.")
print(f"Sur une N(0,1) : VaR = {V_VRAI:.4f}, ES = {E_VRAI:.4f}, "
      f"rapport ES/VaR = {E_VRAI / V_VRAI:.4f}.")
print()
print("On donne au test des prévisions fausses d'un facteur connu. Si")
print("l'instrument est juste, Z doit être négatif quand le modèle")
print("sous-estime, positif quand il est trop prudent, nul au centre.")
print()

rng = np.random.default_rng(13)
x_test = rng.standard_normal(4000)

print(f"  {'prévision annoncée':<26}{'Z₂':>10}{'dépass.':>10}{'attendu':>9}   verdict")
print("  " + "-" * 66)
for facteur in (1.00, 0.85, 0.70, 1.20):
    f = plat(V_VRAI * facteur, E_VRAI * facteur, n=x_test.size)
    r = acerbi_szekely(x_test, f, null=GaussianNull(), n_sim=N_SIM_NULLE)
    libelle = f"{facteur:.0%} de la vérité"
    print(
        f"  {libelle:<26}{r.statistic:>10.4f}{r.n_breaches:>10}"
        f"{r.expected_breaches:>9.0f}   {r.verdict}"
    )

print()
print("  → Signes corrects, et le modèle exact n'est pas rejeté. L'instrument")
print("    est juste.")
print()
print("  UNE PROPRIÉTÉ UTILE, VÉRIFIÉE AU PASSAGE : le rapport ES/VaR est")
print("  invariant d'échelle. Un modèle qui divise tout par deux garde le")
print("  même rapport de queue — donc l'hypothèse nulle contraint la FORME")
print("  de la queue, jamais son niveau. C'est ce qui rend la partie 4")
print("  possible.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 2 — L'ANGLE MORT DU TEST 1")
print(SEP)
print()
print("Acerbi et Székely proposent deux statistiques. La première est")
print("conditionnelle aux dépassements :")
print()
print("    Z₁ = moyenne sur les dépassements de (X_t / ES_t)  +  1")
print()
print("Elle demande : « sachant qu'il y a eu dépassement, la magnitude")
print("était-elle juste ? » C'est une question légitime. Ce n'est pas la")
print("question du régulateur.")
print()

V_FAUX, E_FAUX = var_es_normale(0.05)
print("Voici un modèle qui annonce les VaR/ES du niveau 5 % en prétendant")
print(f"tenir le niveau {ALPHA:.1%}. Il dépasse donc deux fois trop souvent,")
print("mais ses deux nombres sont parfaitement cohérents ENTRE EUX :")
print()
print(f"    vérité à {ALPHA:.1%}   VaR = {V_VRAI:.4f}   ES = {E_VRAI:.4f}"
      f"   rapport {E_VRAI / V_VRAI:.4f}")
print(f"    annoncé        VaR = {V_FAUX:.4f}   ES = {E_FAUX:.4f}"
      f"   rapport {E_FAUX / V_FAUX:.4f}")
print()

resultats_angle = []
for nom, v, e in (
    ("modèle exact", V_VRAI, E_VRAI),
    ("modèle au mauvais niveau", V_FAUX, E_FAUX),
):
    f = plat(v, e)
    z1s, z2s, nbs = [], [], []
    for i in range(N_REP):
        xi = np.random.default_rng(500 + i).standard_normal(T)
        s1, n1 = tail_statistic(xi, f, test=1)
        s2, _ = tail_statistic(xi, f, test=2)
        z1s.append(s1)
        z2s.append(s2)
        nbs.append(n1)
    resultats_angle.append((nom, np.mean(z1s), np.mean(z2s), np.mean(nbs)))

print(f"  {'':<28}{'Z₁':>10}{'Z₂':>10}{'dépassements':>15}")
print("  " + "-" * 63)
for nom, m1, m2, mn in resultats_angle:
    print(f"  {nom:<28}{m1:>10.4f}{m2:>10.4f}{mn:>11.1f} / {T * ALPHA:.0f}")

print()
ecart_z1 = abs(resultats_angle[1][1] - resultats_angle[0][1])
print(f"  → Le test 1 bouge de {ecart_z1:.4f} entre un modèle exact et un modèle")
print(f"    qui dépasse {resultats_angle[1][3] / resultats_angle[0][3]:.1f} fois trop souvent. Il ne voit RIEN.")
print(f"    Le test 2, lui, passe à {resultats_angle[1][2]:.3f}.")
print()
print("  POURQUOI. Le test 1 normalise par le nombre de dépassements")
print("  OBSERVÉS ; le test 2 par le nombre ATTENDU. Un seul caractère de")
print("  différence dans la formule, et c'est toute la fréquence qui entre")
print("  ou sort du test.")
print()
print("  C'est la leçon de la séance 12 dans un domaine entièrement")
print("  différent : **un angle mort appartient au test, pas au modèle")
print("  testé.** D'où `AcerbiSzekelyResult.blind_to_frequency`, qui est")
print("  une propriété interrogeable et non une phrase de documentation.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 3 — PUISSANCE : COMPTER NE SUFFIT PAS")
print(SEP)
print()
print("La séance 3 avait mesuré que le test de Kupiec à 99 % sur 1 500")
print("jours ne rejette un modèle faux que 42 % du temps. Kupiec ne")
print("regarde que le NOMBRE de dépassements ; Z₂ regarde aussi leur")
print("PROFONDEUR. La question est de savoir ce que cette information")
print("supplémentaire achète réellement.")
print()
print(f"Même α = {ALPHA:.1%} pour les deux tests, même T = {T}, même niveau")
print(f"de {NIVEAU:.0%}, mêmes trajectoires. {N_REP} répétitions.")
print()

gen = np.random.default_rng(99)
tirages_h0 = GaussianNull().draw(N_SIM_NULLE * T, ALPHA, gen).reshape(N_SIM_NULLE, T)
nulle_h0 = z2_vectorise(V_VRAI * tirages_h0, V_VRAI, E_VRAI)
CRITIQUE = float(np.quantile(nulle_h0, NIVEAU))
print(f"Valeur critique de Z₂ à {NIVEAU:.0%}, par simulation : {CRITIQUE:+.4f}")
print("(Il n'y a pas de forme close — c'est le sujet de la partie 4.)")
print()

print(f"  {'σ annoncé':<20}{'Kupiec':>10}{'Z₂':>10}{'gain':>10}")
print("  " + "-" * 50)

# Les mêmes trajectoires pour tous les niveaux d'erreur : la comparaison
# est appariée, donc l'écart mesuré ne vient pas du hasard du tirage.
echantillons = np.random.default_rng(700).standard_normal((N_REP, T))

puissances = []
for c in (1.00, 0.95, 0.90, 0.85, 0.80):
    v, e = V_VRAI * c, E_VRAI * c
    z = z2_vectorise(echantillons, v, e)
    rz = int(np.sum(z <= CRITIQUE))
    rk = sum(
        kupiec_test(pd.Series(echantillons[i]), v, confidence=1.0 - ALPHA).rejected
        for i in range(N_REP)
    )
    pk, pz = rk / N_REP, rz / N_REP
    puissances.append((c, pk, pz))
    etiq = f"{c:.2f} × vrai" + ("   (H₀)" if c == 1.0 else "")
    gain = "—" if c == 1.0 else f"{pz - pk:+.1%}"
    print(f"  {etiq:<20}{pk:>10.1%}{pz:>10.1%}{gain:>10}")

print()
mieux = [(c, pk, pz) for c, pk, pz in puissances if c != 1.0]
meilleur_gain = max(mieux, key=lambda r: r[2] - r[1])
print("  → Z₂ domine Kupiec à chaque niveau d'erreur. Le gain est maximal")
print(f"    pour une sous-estimation de {(1 - meilleur_gain[0]) * 100:.0f} % : "
      f"{meilleur_gain[1]:.1%} contre {meilleur_gain[2]:.1%},")
print(f"    soit {meilleur_gain[2] / meilleur_gain[1]:.2f} fois plus de détections.")
print()
taille_k = puissances[0][1]
taille_z = puissances[0][2]
print(f"  Et la comparaison est honnête : sous H₀, Kupiec rejette {taille_k:.1%}")
print(f"  et Z₂ {taille_z:.1%}, tous deux proches du {NIVEAU:.0%} nominal. Z₂ n'achète")
print("  pas sa puissance en trichant sur sa taille.")
print()
print("  LA RAISON, et elle est simple : une sous-estimation modérée de la")
print("  volatilité déplace peu le NOMBRE de dépassements mais beaucoup")
print("  leur PROFONDEUR. Compter jette l'information qui bouge le plus.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 4 — LA LOI NULLE N'EST PAS LIBRE (et ma prédiction était fausse)")
print(SEP)
print()
print("Kupiec possède gratuitement une propriété précieuse : sa loi nulle")
print("est la binomiale(T, α), qui ne dépend que de α. Le test est donc")
print("INDÉPENDANT DE LA DISTRIBUTION.")
print()
print("Un test d'ES ne peut pas l'être. L'ES est une moyenne")
print("conditionnelle : la loi de toute statistique qui la teste dépend de")
print("la forme de la queue au-delà de la VaR.")
print()
print("Mais cette liberté est partiellement fermée, et c'est le premier")
print("résultat de cette partie. Le couple (VaR, ES) FIXE le rapport de")
print("queue, qui est invariant d'échelle. Une loi nulle dont le rapport")
print("ES/VaR ne correspond pas aux prévisions contredit ce qu'elle teste,")
print("et le module la refuse :")
print()
for nulle in (GaussianNull(), StudentTNull(8), StudentTNull(5), StudentTNull(3)):
    print(f"    {nulle.name:<22} ES/VaR = {nulle.ratio(ALPHA):.4f}")
print()
try:
    acerbi_szekely(x_test, plat(V_VRAI, E_VRAI, n=x_test.size),
                   null=StudentTNull(3), n_sim=300)
except ValueError:
    print("    Nulle t(3) sur des prévisions gaussiennes → refusée. Bien.")
print()
print("Reste la liberté sur la forme AU-DELÀ du rapport. J'avais prédit")
print("qu'elle suffirait à démolir le test. Mesurons.")
print()
print("Pour chaque loi, le modèle annonce les VaR/ES EXACTES de la loi qui")
print("génère vraiment les données — donc il a raison. On compare la vraie")
print("valeur critique à celle d'un validateur qui suppose la normalité.")
print()
erreur_mc = math.sqrt(NIVEAU * (1 - NIVEAU) / N_SIM_PARTIE4)
print(f"  {N_SIM_PARTIE4:,} trajectoires par loi. L'erreur-type de Monte Carlo sur")
print(f"  un taux de rejet est de {erreur_mc:.2%} — tout écart plus petit que")
print("  cela n'est pas interprétable, et il a fallu monter jusque-là pour")
print("  que la tendance cesse d'être du bruit.")
print()
print(f"  {'vraie loi':<16}{'ES/VaR':>9}{'critique':>11}{'rapport':>9}"
      f"{'rejet si normalité supposée':>30}")
print("  " + "-" * 75)

lignes_nulle = []
for nulle in (GaussianNull(), StudentTNull(8), StudentTNull(5), StudentTNull(3)):
    ratio = nulle.ratio(ALPHA)
    e_loi = V_VRAI * ratio
    g = np.random.default_rng(4242)
    u = nulle.draw(N_SIM_PARTIE4 * T, ALPHA, g).reshape(N_SIM_PARTIE4, T)
    tires = z2_vectorise(V_VRAI * u, V_VRAI, e_loi)
    crit = float(np.quantile(tires, NIVEAU))
    lignes_nulle.append((nulle.name, ratio, crit, tires))

crit_gauss = lignes_nulle[0][2]
for nom, ratio, crit, tires in lignes_nulle:
    faux_positifs = float(np.mean(tires <= crit_gauss))
    print(
        f"  {nom:<16}{ratio:>9.3f}{crit:>11.4f}{crit / crit_gauss:>9.2f}"
        f"{faux_positifs:>29.1%}"
    )

print()
pire = max(lignes_nulle, key=lambda r: float(np.mean(r[3] <= crit_gauss)))
taux_pire = float(np.mean(pire[3] <= crit_gauss))
print("  → MA PRÉDICTION ÉTAIT FAUSSE. J'attendais un effet massif ;")
print(f"    l'erreur va de {NIVEAU:.0%} à {taux_pire:.1%} dans le pire cas mesuré")
print(f"    ({pire[0]}). C'est réel, c'est le mauvais sens — on rejette trop —")
print("    mais ce n'est pas un scandale.")
print()
print("  POURQUOI SI PEU. Parce que Z₂ divise chaque dépassement par l'ES")
print("  annoncée, et que l'ES encode déjà l'essentiel de l'épaisseur de")
print("  queue. Une fois le rapport ES/VaR imposé, ce qui reste libre dans")
print("  la forme pèse peu. La contrainte de cohérence que le module")
print("  impose fait donc plus que refuser des nulles absurdes : elle")
print("  retire au problème la majeure partie de sa gravité.")
print()
print("  CE QU'IL FAUT QUAND MÊME FAIRE. `EmpiricalNull` rééchantillonne")
print("  les résidus standardisés du modèle lui-même : aucune forme")
print("  supposée, et on retrouve presque l'indépendance distributionnelle")
print("  de Kupiec. Sa limite est celle de la VaR historique de la")
print("  séance 3 — une fenêtre sans krach produit une nulle sans krach.")
print()
residus = np.random.default_rng(55).standard_t(5, 3000) * np.sqrt(3 / 5)
nulle_emp = EmpiricalNull(residus)
print(f"    Exemple : résidus t(5) → ES/VaR empirique {nulle_emp.ratio(ALPHA):.3f}")
ecart_emp = nulle_emp.ratio(ALPHA) - StudentTNull(5).ratio(ALPHA)
print(f"    (la valeur théorique de la t(5) est {StudentTNull(5).ratio(ALPHA):.3f} ;")
print(f"     l'écart de {ecart_emp:+.3f} est de l'erreur d'échantillonnage — un")
print(f"     rapport de queue estimé sur {residus.size} points en utilise {int(residus.size * ALPHA)})")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 5 — COMPARER N'EST PAS TESTER")
print(SEP)
print()
print("Tout ce qui précède répond à « ce modèle est-il réfuté ? ». Rien")
print("n'y répond à « lequel de ces deux modèles est le meilleur ? ».")
print()
print("Et là il y a un obstacle de fond : **l'ES seule n'est pas")
print("élicitable** (Gneiting, 2011). Il n'existe aucune fonction de perte")
print("L(X, e) dont l'espérance soit minimisée à la vraie ES. On ne peut")
print("donc pas classer des prévisions d'ES par un score portant sur l'ES.")
print()
print("Le couple (VaR, ES), lui, est élicitable (Fissler & Ziegel, 2016) :")
print()
print("    L = −1/(α·E) · 1{X ≤ −V} · (X + V)  +  V/E  +  log(E)")
print()
print("C'est une affirmation testable, donc on la teste.")
print()

x_long = np.random.default_rng(21).standard_normal(400_000)
meilleur, arg = np.inf, None
for fv in np.linspace(0.90, 1.10, 21):
    for fe in np.linspace(0.90, 1.10, 21):
        v, e = V_VRAI * fv, E_VRAI * fe
        if e <= v:
            continue
        m = float(fz_loss(x_long, plat(v, e, n=x_long.size)).mean())
        if m < meilleur:
            meilleur, arg = m, (fv, fe)
print(f"  Minimum de la perte moyenne sur une grille : "
      f"VaR × {arg[0]:.3f}, ES × {arg[1]:.3f}")
print("  (la vérité est × 1,000 pour les deux)")
print()
print("  → La perte est bien minimisée à la vérité. La propriété tient.")
print()
print("  ET VOICI CE QUE « CONJOINTEMENT » VEUT DIRE EN PRATIQUE. On fixe")
print("  la VaR à une valeur fausse, et on cherche l'ES qui minimise la")
print("  perte :")
print()
print(f"    {'VaR imposée':<18}{'ES optimale':>14}")
print("    " + "-" * 32)
for fv in (0.75, 0.85, 1.00, 1.15, 1.25):
    best, arge = np.inf, None
    for fe in np.linspace(0.85, 1.35, 101):
        v, e = V_VRAI * fv, E_VRAI * fe
        if e <= v:
            continue
        m = float(fz_loss(x_long, plat(v, e, n=x_long.size)).mean())
        if m < best:
            best, arge = m, fe
    print(f"    × {fv:.2f}{'':<12}× {arge:.3f}")
print()
print("  → Avec une VaR fausse, l'ES qui minimise la perte n'est plus la")
print("    vraie ES. **On ne peut pas évaluer une prévision d'ES sans")
print("    s'engager aussi sur une prévision de VaR.** C'est le contenu")
print("    opérationnel de la non-élicitabilité de l'ES, et c'est mesuré")
print("    plutôt que cité.")
print()
print("  Application : comparer deux modèles sur les mêmes données.")
print()

x_cmp = np.random.default_rng(77).standard_normal(3000)
a = plat(V_VRAI, E_VRAI, n=x_cmp.size, nom="exact")
b = plat(V_VRAI * 0.9, E_VRAI * 0.9, n=x_cmp.size, nom="sous-estime de 10 %")
cmp_ = compare_tail_models(x_cmp, a, b, name_a="exact", name_b="−10 %")
print(f"    {cmp_}")
print()
print("    ⚠️ La p-value n'est pas corrigée de l'autocorrélation des")
print("       différences de perte. Sur données réelles à volatilité")
print("       groupée, elle est donc trop petite. C'est déclaré dans la")
print("       docstring plutôt que découvert par l'utilisateur.")
print()


# ══════════════════════════════════════════════════════════════════════
print(SEP)
print("PARTIE 6 — CE QUE LA SÉANCE 13 A AJOUTÉ")
print(SEP)
print()
print("  1. Les deux statistiques d'Acerbi-Székely, avec leurs valeurs")
print("     critiques simulées — il n'existe pas de forme close.")
print()
print("  2. La mesure de l'angle mort du test 1 : il ne bouge que de")
print(f"     {ecart_z1:.4f} quand la fréquence des dépassements double.")
print("     Le module expose `blind_to_frequency` pour qu'on ne puisse")
print("     pas l'ignorer par distraction.")
print()
print("  3. La puissance de Z₂ contre Kupiec, mesurée à niveau et à α")
print("     égaux :")
for c, pk, pz in puissances:
    if c == 1.00:
        continue
    print(f"       σ × {c:.2f}   Kupiec {pk:>6.1%}   Z₂ {pz:>6.1%}")
print()
print("  4. Une contrainte de cohérence entre l'hypothèse nulle et les")
print("     prévisions, qui refuse une nulle dont le rapport ES/VaR")
print("     contredit ce qu'elle teste. Elle s'est révélée plus utile que")
print("     prévu : c'est elle qui limite la dépendance distributionnelle")
print(f"     à {taux_pire:.1%} de faux positifs au pire, au lieu du désastre")
print("     que j'annonçais.")
print()
print("  5. `EmpiricalNull` — la recommandation pratique : rééchantillonner")
print("     les résidus du modèle plutôt que supposer une forme.")
print()
print("  6. La perte de Fissler-Ziegel, vérifiée minimisée à la vérité, et")
print("     la démonstration mesurée de ce que « élicitable conjointement »")
print("     signifie : une VaR fausse déplace l'ES apparemment optimale.")
print()
print("  7. Un sixième usage du typage structurel — `StandardizedNull` —")
print("     et le premier qui serve à rendre une HYPOTHÈSE explicite")
print("     plutôt qu'à étendre ou comparer du code.")
print()
print("  LA LEÇON QUI RESTE : la VaR se backteste sans rien supposer, l'ES")
print("  non. Passer à une mesure de risque cohérente a donc un coût qui")
print("  ne se voit pas dans la formule de la mesure, mais dans celle de")
print("  son contrôle. Bâle a choisi la cohérence ; le validateur paie en")
print("  hypothèses explicites.")
print()
print(SEP)
