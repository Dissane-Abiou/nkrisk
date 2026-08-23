# nkrisk — moteur de mesure de risque de portefeuille

**Un moteur qui sait ce qu'il ne sait pas.**

Aucune fonction ne retourne un nombre nu. Tout retourne un `Estimate`, qui
transporte son intervalle de confiance, son nombre d'observations, sa méthode
et ses hypothèses de validité. Quand les données ne suffisent pas, le moteur
refuse de répondre au lieu de deviner.

---

## Démarrage

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v          # 17 tests, dont 2 de couverture
python examples/seance1.py          # la démonstration commentée
```

Sans installation :

```bash
PYTHONPATH=src python3 examples/seance1.py
PYTHONPATH=src python3 -m pytest tests/ -q
```

---

## Ce que la séance 1 t'a appris

### 1. Il existe deux rendements, et les confondre est un bug silencieux

| | formule | additif selon | à utiliser pour |
|---|---|---|---|
| **simple** | `P_t/P_{t-1} - 1` | les **actifs** | agréger un portefeuille |
| **log** | `ln(P_t/P_{t-1})` | le **temps** | volatilité, cumul, simulation |

Sur un jour l'écart est de 10⁻⁸. Sur dix ans, la démonstration montre
−25,15 % contre −36,08 % réels. **L'erreur est invisible là où on la teste
et énorme là où elle compte.**

### 2. La volatilité s'estime. Le rendement espéré, non.

Sur les mêmes dix ans de données :

| grandeur | estimation | imprécision relative |
|---|---|---|
| volatilité annualisée | 19,80 % [19,27 – 20,36] | **2,8 %** |
| rendement moyen annuel | −2,52 % [−14,80 – +9,77] | **488 %** |

L'intervalle sur le rendement contient zéro, et le contiendra encore dans
vingt ans. Pour distinguer un rendement espéré de 8 % d'un rendement de 4 %
avec σ = 20 %, il faut de l'ordre de **quatre-vingt-dix ans** de données.

**C'est la raison mathématique pour laquelle un logiciel qui dit « quoi
acheter » repose sur une quantité que personne ne sait mesurer.** C'est aussi
pourquoi l'optimisation de Markowitz est surnommée un « maximiseur d'erreur »
(Michaud, 1989) et pourquoi le portefeuille naïf 1/N bat quatorze modèles
sophistiqués hors échantillon (DeMiguel, Garlappi & Uppal, 2009).

### 3. Combien de données pour savoir quelque chose

Volatilité vraie = 20 %, intervalles à 95 % :

| historique | n | estimation | fourchette | imprécision |
|---|---|---|---|---|
| 1 mois | 21 | 21,2 % | 16,2 – 30,6 % | ±34,0 % |
| 3 mois | 63 | 18,8 % | 16,0 – 22,8 % | ±18,1 % |
| 1 an | 252 | 19,6 % | 18,0 – 21,5 % | ±8,8 % |
| 10 ans | 2520 | 20,1 % | 19,5 – 20,6 % | ±2,8 % |

Un mois de données ne distingue pas une action à 15 % d'une action à 30 %.
Beaucoup d'applications de courtage affichent pourtant ce chiffre sans
avertissement.

### 4. La volatilité te coûte du rendement, mécaniquement

`rendement géométrique ≈ rendement arithmétique − σ²/2`

| σ | perdu (mesuré, 300 simulations) | σ²/2 (théorie) |
|---|---|---|
| 5 % | 0,14 % | 0,13 % |
| 20 % | 2,14 % | 2,00 % |
| 50 % | 12,72 % | 12,50 % |

La traînée croît en **carré** de la volatilité. Gagner 50 % puis perdre 50 %
te laisse à −25 %, pas à zéro. C'est l'argument quantitatif en faveur de la
diversification : réduire σ augmente ta richesse finale même si le rendement
espéré ne bouge pas.

### 5. Un an de bons résultats ne prouve rien

Stratégie de Sharpe vrai = 0,5 :

| historique | Sharpe mesuré | IC 95 % | prouvé ? |
|---|---|---|---|
| 1 an | −0,60 | [−2,57 ; +1,36] | non |
| 5 ans | 0,77 | [−0,11 ; +1,64] | non |
| 10 ans | 0,71 | [+0,09 ; +1,33] | **oui** |

Et tout ceci suppose **un seul essai**. Le théorème des fausses stratégies
(Bailey & López de Prado) établit qu'avec assez d'essais, aucun Sharpe n'est
assez élevé pour rejeter l'hypothèse que la stratégie est fausse.
**Ton backtest ne mesure pas la qualité de ta stratégie ; il mesure le nombre
de fois où tu as regardé.**

---

## Les trois niveaux de test — la partie à mettre sur ton CV

Tu m'as demandé « sans aucune erreur ». On ne le promet pas, on le **mesure**.

**Niveau 1 — propriétés mathématiques.** Des identités qui doivent tenir
quelles que soient les données. Attrape les fautes de frappe.

**Niveau 2 — calibration contre une vérité connue.** On génère des données
dont on a *fixé* la volatilité à 20 %, et on vérifie que l'estimateur la
retrouve. C'est impossible avec des données réelles, où la vraie valeur est
inobservable. Attrape les erreurs de formule et de facteur d'échelle.

**Niveau 3 — couverture de l'intervalle de confiance.** ★
On simule mille échantillons, on construit mille intervalles à 95 %, et on
compte combien contiennent la vraie valeur. Le résultat doit être ≈ 95 %.

Ce niveau valide non pas l'estimation, mais **l'incertitude elle-même**.
Presque personne ne l'écrit. C'est pourtant le seul qui garantit que le
moteur ne ment pas sur ce qu'il sait.

---

## Le bug que j'ai commis, et pourquoi il est dans le dépôt

La première version de la démonstration comparait un rendement arithmétique
annualisé **linéairement** (×252) à un rendement géométrique annualisé par
**composition**. L'écart de convention valait 0,33 point ; l'effet à mesurer
en valait 0,13. La traînée de volatilité sortait **négative** — le tableau
démontrait l'inverse de la réalité.

Rien n'a planté. Aucun test ne l'a vu, parce qu'ils utilisaient tous σ ≥ 20 %,
régime où le vrai effet écrase l'artefact. Je l'ai trouvé en **regardant la
sortie** et en constatant qu'un signe n'allait pas.

C'est conservé dans `test_trainee_positive_meme_a_faible_volatilite` et
documenté dans `compound_annualize`. Trois leçons :

1. Un bug de convention ne lève pas d'exception. Il produit un nombre
   plausible, du bon ordre de grandeur, et faux.
2. Les tests ne l'attrapent pas s'ils sont écrits dans le régime où
   l'artefact est masqué.
3. La seule défense est de vérifier **signe** et **ordre de grandeur** contre
   une prédiction théorique indépendante.

---

## Structure

```
src/nkrisk/
  types.py            Estimate — l'incertitude dans le système de types
  returns.py          rendements, volatilité, Sharpe, traînée
  data/
    synthetic.py      générateur à vérité connue (pour valider)
    market.py         Yahoo Finance et CSV (pour utiliser)
tests/
  test_returns.py     17 tests sur trois niveaux
examples/
  seance1.py          démonstration commentée
```

---

## Utiliser de vraies données

Mon environnement n'atteint pas Yahoo Finance ; le tien oui.

```python
from nkrisk import annualized_volatility, log_returns, sharpe_ratio
from nkrisk.data.market import load_yahoo

prix = load_yahoo("XIC.TO", start="2015-01-01")   # FNB actions canadiennes
vol = annualized_volatility(log_returns(prix))
print(vol.format(2, percent=True))
```

Trois pièges documentés dans `market.py` : prix **ajustés** obligatoires
(sinon chaque dividende ressemble à un krach), biais du survivant
incorrigible avec des données gratuites, et licence Yahoo limitée à l'usage
personnel.

---

## Prochaines séances

| # | sujet | ce que tu apprendras |
|---|---|---|
| 2 | covariance et corrélation | pourquoi la matrice empirique est inutilisable, et la contraction de Ledoit-Wolf |
| 3 | décomposition du risque | contribution marginale et composante — le cœur d'un Aladdin |
| 4 | VaR et CVaR | trois méthodes, et pourquoi la VaR paramétrique ment sur les queues |
| 5 | drawdown | le risque que vivent réellement les investisseurs |
| 6 | exposition factorielle | régression Fama-French : à quoi ton portefeuille est vraiment exposé |
| 7 | stress tests | 2008, mars 2020, 2022 rejoués sur ton portefeuille |
| 8 | Deflated Sharpe Ratio | corriger le biais de sélection — l'antidote au backtest |

---

## Ce que ce moteur ne fera jamais

Il ne dira pas quoi acheter. Il mesure des expositions et des pertes
potentielles ; il ne prédit pas de rendements — parce que la leçon 2 démontre
que personne ne le peut de façon fiable.

C'est exactement le périmètre d'Aladdin, d'ailleurs : Aladdin ne génère
aucune recommandation d'achat. Il dit aux gérants ce que leurs décisions leur
font courir comme risque. Les décisions restent humaines.

*Aucun élément de ce dépôt ne constitue un conseil en investissement.*
