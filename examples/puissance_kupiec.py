"""
Mesure de la puissance du test de Kupiec — reproduit le tableau de la
séance 3, leçon 5. Compter ~3 minutes.

    python examples/puissance_kupiec.py

POURQUOI CE FICHIER EXISTE SÉPARÉMENT
-------------------------------------
La démonstration de la séance 3 affiche des chiffres de puissance
PRÉ-CALCULÉS, parce que les recalculer prendrait trois minutes à chaque
lancement. Un nombre codé en dur dans une narration est exactement le
bug que j'ai commis quatre fois dans ce projet — la seule défense
acceptable est de fournir le script qui permet de le vérifier.

Si tu modifies le moteur et que ces chiffres changent, mets à jour la
séance 3. Un chiffre non reproductible est un chiffre qui dérive.
"""

from __future__ import annotations

from nkrisk.data import StudentTSpec, generate_student_t_prices
from nkrisk.returns import log_returns
from nkrisk.var import historical_var, kupiec_test, parametric_var


def serie(n: int, seed: int):
    return log_returns(
        generate_student_t_prices(StudentTSpec(0.08, 0.20, df=4), n, seed=seed)
    )


print("Puissance du test de Kupiec, 200 essais par ligne.")
print("Données à queues épaisses dans TOUS les cas : le modèle")
print("paramétrique est faux par construction, on mesure combien de")
print("fois le test s'en aperçoit.\n")
print(f"{'niveau':>9}{'jours':>8}{'paramétrique':>16}{'historique':>14}")
print(f"{'-'*9}{'-'*8}{'-'*16}{'-'*14}")

for conf, t in [(0.99, 1500), (0.99, 5000), (0.995, 5000), (0.999, 5000)]:
    rejets_p = rejets_h = 0
    for s in range(200):
        r = serie(2 * t, seed=s + 9000)
        calib, test = r.iloc[:t], r.iloc[t : 2 * t]
        if kupiec_test(test, parametric_var(calib, confidence=conf).var, conf).rejected:
            rejets_p += 1
        if kupiec_test(test, historical_var(calib, confidence=conf).var, conf).rejected:
            rejets_h += 1
    print(f"{conf:>9.3f}{t:>8}{f'{rejets_p}/200 ({rejets_p/2:.0f}%)':>16}"
          f"{f'{rejets_h}/200':>14}")

print("\n→ Un test de couverture ne vaut que s'il a la PUISSANCE de")
print("  détecter ce qu'il cherche. Mesurer cette puissance fait partie")
print("  du travail, pas de l'ornement.")
