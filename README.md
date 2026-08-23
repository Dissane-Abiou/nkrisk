# nkrisk

**Moteur de mesure de risque de portefeuille qui déclare son incertitude.**

Aucune fonction ne retourne un nombre nu. Chaque estimation transporte son
intervalle de confiance, son nombre d'observations, sa méthode et ses
hypothèses de validité. Quand les données ne suffisent pas, le moteur refuse
de répondre plutôt que de deviner.

```python
from nkrisk import annualized_volatility, log_returns
from nkrisk.data.market import load_yahoo

vol = annualized_volatility(log_returns(load_yahoo("XIC.TO", start="2015-01-01")))
print(vol.format(2, percent=True))
# 14.82%  [IC 95% : 14.31% — 15.37%]  n=2634
```

Sur un mois de données, la même fonction répond :

```
21.93%  [IC 95% : 16.78% — 31.67%]  n=21  ⚠️ PEU FIABLE
```

C'est la thèse du projet : **un chiffre de risque sans sa marge d'erreur est
une information trompeuse**, et beaucoup d'outils grand public en affichent
sans avertissement.

---

## Installation

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 41 tests rapides (~2 s)
pytest tests/ -q -m ""    # 47 tests, dont couverture Monte Carlo (~30 s)
```

Python ≥ 3.11 · numpy · pandas · scipy · scikit-learn (optionnel, pour une
validation croisée)

---

## Ce que le moteur calcule

| Module | Contenu |
|---|---|
| `returns` | rendements simples et log, volatilité, ratio de Sharpe, traînée de volatilité — tous avec intervalles de confiance |
| `covariance` | covariance empirique et contraction de Ledoit-Wolf, avec diagnostic de conditionnement |
| `risk` | décomposition du risque : contributions marginale et composante, ratio de diversification, nombre effectif de paris |
| `var` | VaR et CVaR par trois méthodes, backtest de Kupiec, contre-exemple de sous-additivité |
| `data` | générateurs synthétiques à vérité connue (GBM, Student-t corrélés) + chargeurs Yahoo/CSV |

---

## L'architecture des tests

C'est la partie du projet dont je suis le plus satisfait. Les tests sont
organisés en quatre niveaux de force croissante.

**Niveau 1 — propriétés mathématiques.** Des identités qui doivent tenir quelles
que soient les données : `r = e^l - 1`, additivité temporelle des log-rendements,
additivité des rendements simples entre actifs.

**Niveau 2 — calibration contre une vérité connue.** On génère des données dont
on a *fixé* la volatilité à 20 %, et on vérifie que l'estimateur la retrouve.
Impossible avec des données réelles, où la vraie valeur est inobservable.
C'est ce qui attrape les erreurs de facteur d'échelle.

**Niveau 3 — couverture de l'intervalle de confiance.** On simule mille
échantillons, on construit mille intervalles à 95 %, et on compte combien
contiennent la vraie valeur. Le résultat doit être ≈ 95 %.

> Ce niveau valide non pas l'estimation, mais **l'incertitude elle-même**.
> C'est le seul test qui garantit que le moteur ne ment pas sur ce qu'il sait.

**Niveau 4 — validation croisée indépendante.** L'implémentation de
Ledoit-Wolf est comparée à celle de scikit-learn, écrite par d'autres
personnes à partir du même article. Concordance à 10⁻¹⁸ sur δ et sur la
matrice complète, dans quatre régimes N/T différents.

---

## Quelques résultats produits par le moteur

**La volatilité s'estime, le rendement espéré non.** Sur dix ans de données,
la volatilité sort à ±2,8 % d'imprécision relative, le rendement moyen à
±488 % — intervalle contenant zéro. Pour distinguer un rendement espéré de
8 % d'un rendement de 4 % avec σ = 20 %, il faut de l'ordre de quatre-vingt-dix
ans d'historique.

**L'optimiseur de portefeuille maximise l'erreur.** Sur quinze échantillons
tirés de la même distribution, le portefeuille de variance minimale calculé
sur la covariance empirique produit un levier moyen de 19,8 à N/T = 0,92,
contre 6,1 avec contraction. La matrice empirique atteint un conditionnement
de 1,4 million à N/T = 0,98.

**La VaR paramétrique se trompe là où personne ne regarde.** À variance égale,
une distribution à queues épaisses a un quantile *plus faible* que la normale
à 90 % et 95 %. L'écart n'est que de 14 % à 99 %, et atteint un facteur 2,5
à 99,99 %.

**Le backtest réglementaire n'a pas la puissance de le détecter.** Le test de
Kupiec à 99 % sur 1 500 jours ne rejette un modèle faux que 42 % du temps.
Bâle impose ce test sur 250 jours.

**La VaR n'est pas une mesure de risque cohérente.** Deux obligations
indépendantes à 4 % de défaut ont chacune une VaR à 95 % de zéro ; leur
combinaison a une VaR de 0,5. La VaR affirme que diversifier crée du risque.

---

## Les bugs que j'ai commis, et pourquoi ils sont documentés

Quatre erreurs sont conservées dans le dépôt, avec le test de non-régression
qui les verrouille. Elles ont plus de valeur pédagogique que le code correct.

**Un bug de convention silencieux.** Je comparais un rendement annualisé
linéairement à un autre annualisé par composition. L'écart de convention
(0,33 pt) dépassait l'effet mesuré (0,13 pt) : la traînée de volatilité
sortait *négative*, l'inverse de la réalité. Rien n'a planté, aucun test ne
l'a vu — ils utilisaient tous un régime où le vrai effet écrase l'artefact.
→ `compound_annualize`, `test_trainee_positive_meme_a_faible_volatilite`

**Un indicateur conceptuellement faux.** J'avais défini le nombre effectif de
paris comme l'inverse de l'indice de Herfindahl sur les contributions au
risque. Sur huit actifs corrélés à 0,9999 et équipondérés, il retournait 8 —
alors qu'il n'y a qu'un seul pari détenu huit fois. La formule mesurait la
*concentration*, pas la *diversification*.
→ `effective_number_of_bets`, remplacé par la version entropique de Meucci

**Un domaine de validité non documenté dans la littérature.** En testant la
version corrigée, j'ai découvert qu'elle saute discontinûment de 10 à 1 entre
ρ = 0 et ρ = 0,001 sur un portefeuille parfaitement symétrique. C'est
mathématiquement correct et pratiquement inutilisable. Documenté dans le code.

**Des narrations codées en dur qui dérivent.** Trois affirmations du code de
démonstration contredisaient les tableaux affichés juste au-dessus, parce que
les nombres avaient été écrits à la main à partir d'une exécution antérieure.
Toutes les narrations sont désormais calculées à partir des mêmes données que
les tables.

La leçon commune : **un bug numérique ne lève pas d'exception.** Il produit un
nombre plausible, du bon ordre de grandeur, et faux. La seule défense connue
est de vérifier signe et ordre de grandeur contre une prédiction théorique
indépendante.

---

## Ce que ce moteur ne fera jamais

Il ne dit pas quoi acheter. Il mesure des expositions et des pertes
potentielles ; il ne prédit pas de rendements, parce que le projet démontre
lui-même que personne ne le peut de façon fiable.

C'est aussi le périmètre d'Aladdin, dont ce projet s'inspire : Aladdin ne
génère aucune recommandation d'achat. Il dit aux gérants ce que leurs
décisions leur font courir comme risque. Les décisions restent humaines.

---

## Références

Ledoit & Wolf (2004), *A Well-Conditioned Estimator for Large-Dimensional
Covariance Matrices* · Meucci (2009), *Managing Diversification* · Artzner,
Delbaen, Eber & Heath (1999), *Coherent Measures of Risk* · Kupiec (1995),
*Techniques for Verifying the Accuracy of Risk Measurement Models* · Lo (2002),
*The Statistics of Sharpe Ratios* · Bailey & López de Prado (2014), *The
Deflated Sharpe Ratio* · DeMiguel, Garlappi & Uppal (2009), *Optimal Versus
Naive Diversification* · Michaud (1989), *The Markowitz Optimization Enigma*

---

*Projet personnel. Aucun élément de ce dépôt ne constitue un conseil en
investissement.*
