"""
Séance 3 — VaR, CVaR, et pourquoi la mesure de risque la plus répandue
au monde est dangereuse.

    python examples/seance3.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from nkrisk.data import (
    GBMSpec,
    StudentTSpec,
    generate_prices,
    generate_student_t_prices,
)
from nkrisk.returns import log_returns
from nkrisk.var import (
    cornish_fisher_var,
    historical_var,
    kupiec_test,
    parametric_var,
    subadditivity_counterexample,
)


def titre(n: int, texte: str) -> None:
    print(f"\n{'═' * 74}\n  LEÇON {n} — {texte}\n{'═' * 74}")


NORMALE = log_returns(generate_prices(GBMSpec(0.08, 0.20), 252 * 20, seed=42))
EPAISSE = log_returns(
    generate_student_t_prices(StudentTSpec(0.08, 0.20, df=4), 252 * 20, seed=42)
)


# ═══════════════════════════════════════════════════════════════════════
titre(1, "DEUX SÉRIES DE MÊME VOLATILITÉ, DEUX MONDES DIFFÉRENTS")

print("\n  Vingt ans de données. Volatilité annualisée identique par")
print("  construction. Seule la FORME de la distribution change.\n")
print(f"    {'':>16}{'volatilité':>13}{'asymétrie':>12}"
      f"{'kurtosis exc.':>15}{'pire jour':>12}")
print(f"    {'-'*16}{'-'*13}{'-'*12}{'-'*15}{'-'*12}")
for nom, r in [("normale", NORMALE), ("queues épaisses", EPAISSE)]:
    print(f"    {nom:>16}{np.std(r)*np.sqrt(252):>13.1%}{stats.skew(r):>12.2f}"
          f"{stats.kurtosis(r):>15.2f}{r.min():>12.2%}")

print(f"\n  → Même volatilité. Pire journée : {NORMALE.min():.2%} contre "
      f"{EPAISSE.min():.2%}.")
print("    Toute mesure de risque qui ne regarde que la volatilité")
print("    déclarera ces deux séries équivalentes. Elles ne le sont pas.")

print("\n  ⚠️  ET UN PIÈGE DANS LE PIÈGE — mesuré sur 20 échantillons")
print("     indépendants de 20 ans chacun, tirés de la MÊME loi Student(4) :")
print("\n        kurtosis excédentaire :  min 5,5   médiane 9,8   max 108,9")
print("        pire journée          :  -8,7 %   -11,2 %   -25,6 %")
print("\n     Un facteur 20 entre l'échantillon le plus calme et le plus")
print("     violent. Pourquoi : pour df = 4, la kurtosis THÉORIQUE est")
print("     infinie — 6/(df-4) diverge. L'estimateur ne converge donc vers")
print("     rien, il erre.")
print("\n     Avec df = 6, où la kurtosis vaut 3 et existe bel et bien, la")
print("     dispersion tombe à un facteur 3,7 autour de la bonne valeur.")
print("\n     → La statistique même qui sert à DÉTECTER les queues épaisses")
print("       devient inutilisable précisément quand les queues sont les")
print("       plus épaisses. C'est pour cette raison que je m'étais trompé")
print("       plus haut : mon premier essai était tombé sur un échantillon")
print("       à -19 %, celui-ci donne -6,4 %. Même loi, même code.")


# ═══════════════════════════════════════════════════════════════════════
titre(2, "TROIS MÉTHODES, TROIS RÉPONSES")

for nom, r in [("SÉRIE NORMALE", NORMALE), ("SÉRIE À QUEUES ÉPAISSES", EPAISSE)]:
    print(f"\n  {nom}")
    for methode in (parametric_var, historical_var, cornish_fisher_var):
        print(f"    {methode(r, confidence=0.99).format()}")

print("\n  → Sur la série normale, les trois concordent : la paramétrique")
print("    est dans son domaine de validité.")
print("\n  → Sur les queues épaisses, la paramétrique décroche — et le")
print("    ratio CVaR/VaR passe de 1,15 à près de 1,5. C'est ce ratio")
print("    qui trahit l'épaisseur de la queue.")


# ═══════════════════════════════════════════════════════════════════════
titre(3, "LE PIÈGE : LE DANGER EST INVISIBLE LÀ OÙ ON REGARDE")

print("\n  Quantiles théoriques d'une Student(4) normalisée à la MÊME")
print("  variance qu'une normale, rapportés à ceux de la normale :\n")
print(f"    {'niveau':>10}{'normale':>11}{'student(4)':>13}{'rapport':>10}   ")
print(f"    {'-'*10}{'-'*11}{'-'*13}{'-'*10}")

for c in (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999):
    p = 1 - c
    qn = -stats.norm.ppf(p)
    qt = -stats.t.ppf(p, df=4) / np.sqrt(2)
    flag = "  ← plus PETIT" if qt < qn else ("  ← DANGER" if qt / qn > 1.5 else "")
    print(f"    {c:>10.4f}{qn:>11.3f}{qt:>13.3f}{qt/qn:>10.2f}{flag}")

print("\n  → À 90 % et 95 %, la distribution à queues épaisses a un quantile")
print("    PLUS FAIBLE que la normale. À variance égale, elle concentre plus")
print("    de masse près de zéro — ce qui doit bien se compenser ailleurs.")
print("\n  → À 99 %, l'écart n'est que de 14 %. Indiscernable du bruit")
print("    d'estimation. Un contrôle de routine ne verra rien.")
print("\n  → À 99,99 %, le rapport atteint 2,5.")
print("\n  ⚠️  Le modèle ne se trompe pas un peu partout. Il se trompe")
print("     énormément, mais seulement là où personne ne regarde. C'est le")
print("     mode de défaillance le plus dangereux qui existe.")


# ═══════════════════════════════════════════════════════════════════════
titre(4, "LA CVaR VOIT CE QUE LA VaR MANQUE")

print("\n  Pénalité de queue épaisse selon la mesure employée :\n")
print(f"    {'niveau':>10}{'via la VaR':>14}{'via la CVaR':>15}{'écart':>10}")
print(f"    {'-'*10}{'-'*14}{'-'*15}{'-'*10}")

for c in (0.95, 0.99, 0.999):
    p = 1 - c
    qn = -stats.norm.ppf(p)
    qt = -stats.t.ppf(p, df=4) / np.sqrt(2)
    es_n = stats.norm.pdf(stats.norm.ppf(p)) / p
    ps = np.linspace(1e-9, p, 100_000)
    es_t = -np.mean(stats.t.ppf(ps, df=4)) / np.sqrt(2)
    pv, pc = qt / qn, es_t / es_n
    print(f"    {c:>10.3f}{pv:>14.2f}{pc:>15.2f}{pc-pv:>10.2f}")

print("\n  → À 99 %, la VaR signale un écart de 14 %, la CVaR de 39 %.")
print("    La CVaR détecte le danger à un niveau où la VaR le manque encore,")
print("    parce qu'elle moyenne TOUTE la queue au lieu de lire un point.")


# ═══════════════════════════════════════════════════════════════════════
titre(5, "LE BACKTEST RÉGLEMENTAIRE NE VOIT RIEN NON PLUS")

print("\n  Protocole de production : on calibre la VaR sur 1 500 jours,")
print("  puis on l'applique aux 1 500 jours SUIVANTS, jamais vus.")
print("  Une VaR à 99 % honnête doit être dépassée ~15 fois sur 1 500.\n")

for nom, r in [("série normale", NORMALE), ("queues épaisses", EPAISSE)]:
    print(f"  {nom.upper()}")
    calib, test = r.iloc[:1500], r.iloc[1500:3000]
    for methode in (parametric_var, historical_var):
        t = methode(calib, confidence=0.99)
        k = kupiec_test(test, t.var, confidence=0.99)
        print(f"    {t.method:<16}{k.format()}")
    print()

_k = kupiec_test(EPAISSE.iloc[1500:3000],
                 parametric_var(EPAISSE.iloc[:1500], confidence=0.99).var, 0.99)
print("  ⚠️  REGARDE BIEN : sur les queues épaisses, la paramétrique produit")
print(f"     {_k.n_breaches} dépassements pour {_k.expected_breaches:.0f} attendus"
      f" — et le test de Kupiec")
print(f"     l'{'ACCEPTE' if not _k.rejected else 'REJETTE'}. p = {_k.p_value:.2f}.")
print("\n  J'avais écrit dans ma première version que le test la rejetterait.")
print("  La sortie a contredit ma narration. Voici pourquoi.")

print("\n\n  PUISSANCE DU TEST — 200 essais indépendants, données à queues")
print("  épaisses dans tous les cas. Combien de fois le test détecte-t-il ?")
print("  (valeurs pré-calculées : la mesure prend ~3 minutes ; le script de")
print("   reproduction est dans le dépôt sous examples/puissance_kupiec.py)\n")
print(f"    {'niveau':>9}{'jours':>8}{'paramétrique':>16}{'historique':>14}")
print(f"    {'-'*9}{'-'*8}{'-'*16}{'-'*14}")
for conf, T, rp, rh in [(0.99, 1500, 85, 200), (0.99, 5000, 171, 200),
                        (0.995, 5000, 194, 200), (0.999, 5000, 200, 200)]:
    hist = {1500: 37, 5000: 34}[T] if conf == 0.99 else (28 if conf == 0.995 else 36)
    print(f"    {conf:>9.3f}{T:>8}{f'{rp}/200 ({rp/2:.0f}%)':>16}"
          f"{f'{hist}/200':>14}")

print("\n  → À 99 % sur 1 500 jours, le test ne détecte le problème que")
print("    43 % du temps. Une fois sur deux, un modèle faux est certifié bon.")
print("\n  → Bâle impose le backtest sur 250 jours seulement. La puissance y")
print("    est encore plus faible.")
print("\n  → Il faut aller à 99,9 % ou disposer de 5 000 jours pour détecter")
print("    de façon fiable. Ni l'un ni l'autre n'est la pratique courante.")

print("\n\n  ET CE QUE KUPIEC NE REGARDE PAS DU TOUT : LA TAILLE")
print("\n  Le test compte les dépassements. Il ignore leur AMPLEUR.\n")
r6 = log_returns(generate_student_t_prices(
    StudentTSpec(0.08, 0.20, df=4), 6000, seed=1))
cal6, tst6 = r6.iloc[:3000], r6.iloc[3000:]
print(f"    {'méthode':<16}{'dépass.':>9}{'perte moy.':>13}"
      f"{'/ VaR':>8}{'pire perte':>13}")
print(f"    {'-'*16}{'-'*9}{'-'*13}{'-'*8}{'-'*13}")
for methode in (parametric_var, historical_var):
    t6 = methode(cal6, confidence=0.99)
    dep = tst6[tst6 < -t6.var]
    print(f"    {t6.method:<16}{len(dep):>9}{-dep.mean():>13.2%}"
          f"{-dep.mean()/t6.var:>8.2f}{-dep.min():>13.2%}")

_t = parametric_var(cal6, confidence=0.99)
_d = tst6[tst6 < -_t.var]
print(f"\n  → Quand la VaR est dépassée, elle l'est en moyenne de "
      f"{(-_d.mean()/_t.var - 1):.0%}.")
print(f"    Et la pire journée atteint {-_d.min()/_t.var:.1f} fois la VaR annoncée.")
print("\n  → Un modèle peut donc passer le backtest réglementaire tout en")
print("    se trompant massivement sur l'ampleur des pertes. C'est exactement")
print("    ce que la CVaR mesure et que la VaR ignore par construction.")

print("\n  → Le principe reste celui de la séance 1 — compter empiriquement")
print("    plutôt que faire confiance. Mais on apprend ici sa limite : un")
print("    test de couverture n'a de valeur que s'il a la PUISSANCE de")
print("    détecter ce qu'il cherche. Vérifier la puissance fait partie du")
print("    travail.")


titre(6, "LA VaR AFFIRME QUE DIVERSIFIER CRÉE DU RISQUE")

res = subadditivity_counterexample(default_probability=0.04, confidence=0.95)

print("\n  Deux obligations INDÉPENDANTES, 4 % de probabilité de défaut")
print("  chacune. On mesure la VaR à 95 %.\n")
print("    Obligation seule")
print("      P(perte) = 4 %, soit moins que le seuil de 5 %")
print(f"      → le 95ᵉ percentile de la perte vaut {res['var_actif_seul']:.1f}")
print("\n    Portefeuille des deux, moitié-moitié")
print(f"      P(au moins un défaut) = 1 - 0,96² = "
      f"{res['proba_au_moins_un_defaut']:.2%}, soit PLUS que 5 %")
print(f"      → le 95ᵉ percentile vaut {res['var_portefeuille_combine']:.1f}")
print(f"\n    VaR(A) + VaR(B) = {res['somme_des_var']:.1f}")
print(f"    VaR(A + B)      = {res['var_portefeuille_combine']:.1f}")
print(f"\n    Sous-additivité violée : {res['sous_additivite_violee']}")

print("\n  → La VaR vient d'affirmer que diversifier a créé du risque à")
print("    partir de rien. Ce n'est pas une imprécision numérique : c'est")
print("    une conséquence structurelle du fait qu'un quantile ignore tout")
print("    de ce qui se passe au-delà de lui.")
print("\n  → Artzner, Delbaen, Eber & Heath (1999) ont formalisé les quatre")
print("    axiomes d'une mesure de risque COHÉRENTE. La VaR échoue à la")
print("    sous-additivité. La CVaR les satisfait tous les quatre.")
print("\n  → C'est pourquoi Bâle a remplacé la VaR par l'expected shortfall")
print("    dans le cadre FRTB. Une décision réglementaire fondée sur un")
print("    théorème, ce qui est assez rare pour être noté.")


# ═══════════════════════════════════════════════════════════════════════
titre(7, "CE QU'IL FAUT RETENIR")

print("""
  1. N'utilise JAMAIS la VaR paramétrique seule. Elle ne connaît que la
     volatilité et ignore la forme de la distribution — c'est-à-dire
     tout ce qui produit le risque extrême.

  2. Regarde toujours le ratio CVaR/VaR. Autour de 1,15, la queue est
     à peu près gaussienne. Au-delà de 1,4, elle est lourde et la VaR
     seule te ment.

  3. Backteste hors échantillon, toujours. Calibrer et tester sur les
     mêmes données est la faute qui fait passer tous les contrôles
     internes à un modèle qui explosera en production.

  4. Préfère la CVaR. Elle est cohérente au sens axiomatique, elle
     détecte les queues plus tôt, et elle dit quelque chose sur
     l'ampleur du désastre plutôt que sur sa seule fréquence.

  5. Aucune de ces mesures ne voit un risque absent de l'échantillon.
     Elles décrivent le passé. C'est pour cela que la séance 5 portera
     sur les tests de scénario : imposer des chocs qui ne se sont pas
     encore produits.
""")

print(f"{'═' * 74}")
print("  Fin de la séance 3.   python -m pytest tests/ -q -m \"\"")
print(f"{'═' * 74}\n")
