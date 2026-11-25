# nkrisk — État d'avancement

**Dépôt** · github.com/Dissane-Abiou/nkrisk
**Dernière mise à jour** · 25 novembre 2026, après la séance 15

> Ce fichier est le journal de bord du projet : ce qui existe, ce que
> chaque séance a produit, ce qui ne marche pas encore, et ce qui vient
> ensuite. Il se lit de haut en bas quand on reprend après une pause.
>
> Il ne sert pas à vendre le projet — c'est le rôle de
> `nkrisk-cv-linkedin.md`. Ici, on écrit aussi ce qui va mal.

---

## 1. L'état en un coup d'œil

| | |
|---|---|
| **Séances** | 15 |
| **Commits** | 16, du 23 août au 25 novembre 2026 |
| **Modules** | 19 |
| **Lignes versionnées** | ~25 700, dont ~24 000 de Python |
| **Répartition** | 10 299 source · 8 042 tests · 5 617 démonstrations |
| **Tests** | 450 (427 rapides ~25 s · 23 Monte Carlo ~52 s) |
| **Couverture** | 89 % |
| **Analyse statique** | ruff propre · **`mypy --strict`** sans erreur (21 fichiers) |
| **CI** | GitHub Actions, 4 travaux, Python 3.11 / 3.12 / 3.13 |
| **Sécurité** | modèle de menace écrit, 40 tests d'attaque |
| **Licence** | MIT, avec `AUTEURS.md` et obligation d'attribution |
| **Démonstration** | dissane-abiou.github.io/nkrisk |

**Vérifié sur deux plateformes** : Linux / Python 3.11 (développement) et
Windows / Python 3.14 (machine de travail). Les démonstrations produisent
les mêmes chiffres au millième près sur les deux — c'est en partie ce que
paie la correction des signes propres de la séance 7.

---

## 1bis. Le premier verdict du moteur sur de vraies données

**24 novembre 2026** · 12 titres TSX et NYSE · 2 834 séances communes ·
2015-05-22 → 2026-11-23 · `docs/rapport-validation.html`

C'est la première fois que le moteur rend une opinion complète sur un
marché réel. Il se rejette lui-même, et c'est le résultat qui compte.

| modèle | inventaire | protocole modèle | queue | **verdict** | contraint par |
|---|---|---|---|---|---|
| empirique | VALIDÉ | VALIDÉ | REJETÉ | **REJETÉ** | queue |
| Ledoit-Wolf | VALIDÉ | VALIDÉ | REJETÉ | **REJETÉ** | queue |
| EWMA λ=0,94 | REJETÉ | REJETÉ | REJETÉ | **REJETÉ** | inventaire |
| facteurs (Marchenko-Pastur) | VALIDÉ | VALIDÉ | REJETÉ | **REJETÉ** | queue |

### Le chiffre à retenir

```
calibration globale                        0,76
calibration hors des 11 % périodes agitées 1,01
part de la variance dans ces 11 %           50 %
→ écart CONCENTRÉ
```

La séance 9 avait trouvé 0,79 sur une fenêtre plus courte, sans pouvoir
dire d'où venait l'écart. Onze ans de données le disent : **sur les jours
normaux le modèle est exact — 1,01, on ne fait pas mieux.** La totalité
des 24 % manquants vient de 11 % de périodes qui portent la moitié de la
variance de la décennie.

Ce n'est donc pas un biais, c'est un RETARD : le modèle voit la tempête
une fois qu'elle est entrée dans sa fenêtre d'estimation. Aucun facteur
correctif ne répare cela — il faut un estimateur qui réagit. Et la
séance 11 avait déjà mesuré pourquoi la pondération exponentielle ne
suffit pas : son avantage est en temps, son coût est en dimension, et
ici elle est le seul modèle rejeté dès l'inventaire.

### Ce que la qualité des données a signalé, et qui conditionne tout

- **SHOP.TO** démarre 97 dates après les autres. En intersection, ce seul
  titre ampute l'univers entier de 97 séances. Le rapport suggère de
  l'exclure plutôt que de payer ce prix.
- **114 dates sur 2 949** ne sont pas communes aux deux places boursières
  (3,9 %). La séance 9 a mesuré que le choix d'alignement déplace alors
  la corrélation d'environ 10 % — donc le risque du portefeuille.

Le conflit de PÉRIMÈTRE s'est déclenché : tous les verdicts ci-dessus
sont conditionnels à des données que le moteur lui-même tient pour
suspectes, et les instruments en aval n'ont pas été restreints.

### Une limite que seule cette exécution pouvait révéler

**Les quatre modèles sont rejetés sur la queue pour la même raison, et
cette raison n'est pas eux.** Le protocole impose à chaque modèle de
covariance une forme de queue GAUSSIENNE : VaR = σ·1,96 et
ES = VaR·1,193. Sur un marché réel, le rapport ES/VaR réalisé vaut entre
1,29 et 1,37 — franchement hors de l'intervalle attendu.

Le verdict de queue rejette donc l'hypothèse gaussienne que *j'ai*
imposée, pas la qualité des estimateurs de covariance. C'est une
information vraie et utile — une queue gaussienne est fausse sur un vrai
marché — mais elle ne discrimine pas entre les quatre modèles. Le
correctif est identifié : laisser chaque modèle déclarer sa propre forme
de queue, ajustée causalement sur ses résidus.

**La causalité, elle, est propre** : écart maximal 0,0 sur trois sondes.

---

## 2. Les quinze séances

| # | Date | Sujet | Le résultat qui reste |
|---|---|---|---|
| 1 | 23 août | `Estimate`, rendements, volatilité | Aucune fonction ne retourne un nombre nu |
| 2 | 23 août | Covariance, Ledoit-Wolf | Levier 19,8 → 6,1 par contraction |
| 3 | 23 août | Euler, VaR, CVaR, Kupiec | Kupiec ne détecte un modèle faux que 42 % du temps |
| 4 | 17 nov. | Drawdown | Sans référence simulée, « −37 % » ne veut rien dire |
| 5 | 17 nov. | Scénarios de stress | La diversification est ce qu'une crise détruit |
| 6 | 17 nov. | Intégration continue, page de démo | Le patron du test à oracle vérifié |
| 7 | 19 nov. | Modèle à facteurs, Marchenko-Pastur | 400 actions sur 8 ans battent 15 actions sur 2 siècles |
| 8 | 19 nov. | Backtest, causalité, Sharpe dégonflé | Une avance d'un jour sur 8 ans est détectée |
| 9 | 19 nov. | Données réelles, qualité, alignement | Calibration 0,79 sur vraies actions contre 0,99 simulé |
| 10 | 19 nov. | Sécurité | Une CWE-22 trouvée dans mon propre code |
| 11 | 19 nov. | Pondération exponentielle | L'avantage est en temps, le coût est en dimension |
| 12 | 22 nov. | Validation de modèles | Un angle mort appartient au protocole, pas au modèle |
| 13 | 24 nov. | Backtest d'Expected Shortfall | La VaR se backteste sans rien supposer, l'ES non |
| 14 | 24 nov. | Rapport de validation | Le moteur forme une opinion, et nomme ses propres conflits |
| 15 | 25 nov. | Durcissement d'ingénierie | Ce qui rend un code consommable n'est pas dans le code |

### Le détail, pour les séances qui portent le plus

**Séance 7 — modèle à facteurs.** Sélection du nombre de facteurs par la
loi de Marchenko-Pastur, sans seuil arbitraire. Taux de faux positifs
mesuré sur 1 000 univers de bruit pur : 6,4 %. Volatilité réalisée hors
échantillon contre Ledoit-Wolf : −9 % à N/T = 0,2, −17 % à 0,4, −36 % à
N/T = 1,5 où la covariance empirique est singulière. Un modèle à k = 5
décrit 500 actifs avec 3 000 paramètres au lieu de 125 250.
→ `src/nkrisk/factors.py`, `examples/seance7.py`, 45 tests

**Séance 8 — backtest et causalité.** `history = returns.iloc[debut:t]`,
avec `t` exclu : toute la sécurité du moteur tient dans cette ligne.
L'audit `detect_lookahead` ne lit pas le code évalué : il corrompt le
futur, rejoue, et compare les décisions. Trois fuites sur trois
détectées, zéro fausse alerte sur trois stratégies honnêtes, y compris
une avance d'un seul jour sur huit ans (0,05 % de l'historique). Plus le
Deflated Sharpe : un Sharpe de 1,0 après 12 650 essais voit sa
probabilité d'être réel passer de 99,6 % à 10,3 %.
→ `src/nkrisk/backtest.py`, 38 tests

**Séance 9 — données réelles.** La confrontation aux vraies données a
produit le résultat le plus utile du projet : le modèle annonce **0,79
fois** le risque subi sur douze actions de Toronto et New York
(2015-2026), contre 0,99 en simulé. Le diagnostic intégré montre que
l'écart est *concentré* : 11 % des périodes portent 59 % de la variance,
et la pire fenêtre — 28 février 2020 — annonçait 7,3 % pour 98,4 %
réalisée. Deux lois mesurées sur l'alignement de calendriers : à 10 % de
séances non communes, reporter le dernier prix ne biaise pas les
volatilités (−0,2 %) mais sous-estime la corrélation de 10,2 %.
→ `src/nkrisk/data/quality.py`, `src/nkrisk/data/market.py`, 34 tests

**Séance 10 — sécurité.** Une traversée de répertoire (CWE-22) réelle
dans `CsvSource.fetch`, avec deux évasions vérifiées :
`fetch("../prive/secrets")`, et pire, `fetch("/etc/passwd")` — l'opérateur
`/` de pathlib abandonne son membre de gauche quand celui de droite est
absolu. Corrigée par liste blanche **et** confinement du chemin résolu ;
un test par couche montre que chacune attrape l'angle mort de l'autre.
→ `SECURITE.md`, `tests/test_securite.py`, 40 tests d'attaque

**Séance 12 — validation de modèles.** `ModelInventory` et `ModelCard` :
l'inventaire et la fiche que la ligne directrice E-23 exige, produits par
la mesure. Deux résultats. D'abord, la statistique de biais usuelle de la
profession, dont l'intervalle `|b−1| < 1,96/√(2T)` a une couverture réelle
de 43,3 % sur un marché réaliste au lieu des 95 % annoncés — elle rejette
un modèle correct six fois sur dix. Ensuite, et c'est la leçon qui
dépasse la finance : même modèle, mêmes données, deux protocoles,
verdicts opposés (biais 1,002 → VALIDÉ sur portefeuille fixe ; 1,959 →
REJETÉ sur le portefeuille que le modèle recommande).
→ `src/nkrisk/validation.py`, `examples/seance12.py`, 50 tests

**Séance 13 — backtest d'Expected Shortfall.** Bâle a remplacé la VaR à
99 % par l'ES à 97,5 % dans FRTB, parce que la séance 3 avait raison : la
VaR n'est pas cohérente. Cette séance mesure le prix du changement, et
produit quatre résultats. Le test 1 d'Acerbi-Székely a un **angle mort
béant** : un modèle qui dépasse deux fois trop souvent le passe sans
broncher. Le test 2 est **nettement plus puissant que Kupiec** à α et
niveau égaux — 84,5 % contre 69,4 % de détections pour une volatilité
sous-estimée de 10 %. La **loi nulle n'est pas libre** : le couple
(VaR, ES) fixe le rapport de queue, et le module refuse une nulle qui le
contredit. Et ma quatrième prédiction était fausse : j'annonçais que la
dépendance distributionnelle résiduelle démolirait le test, elle fait
passer les faux positifs de 5,0 % à 7,4 % au pire.
→ `src/nkrisk/shortfall.py`, `examples/seance13.py`, 64 tests

**Séance 14 — le rapport de validation.** Aucune mesure nouvelle, et
c'est assumé. Les treize séances précédentes savaient répondre à treize
questions séparément ; le moteur n'avait jamais formé une opinion. Ce
module réconcilie, sous une règle défendable — le verdict d'un modèle est
le PIRE des verdicts rendus sur lui, parce qu'une absence de preuve ne
compense pas une preuve — et expose `binding_instrument`, qui nomme
l'instrument contraignant, donc ce qu'il faut réparer. Cinq familles de
conflits, chacune avec son interprétation, et le périmètre en premier :
quand la qualité des données signale des anomalies que les autres
instruments ont ignorées, tous les verdicts deviennent conditionnels. Un
conflit qui vise tous les modèles est replié en constat de régime, parce
qu'il ne désigne plus personne. Sortie : un HTML autonome, lisible en
soixante secondes par quelqu'un qui ne lira jamais le code.
→ `src/nkrisk/report.py`, `examples/seance14.py`,
`examples/rapport_reel.py`, 37 tests

**Séance 15 — durcissement d'ingénierie.** Déclenchée par un retour
extérieur : « des codes de développeur intermédiaire ». L'audit a donné
raison au retour, mais pas sur les statistiques — sur **l'emballage**.
Un relecteur voit le paquet avant le contenu, et le paquet ne livrait
pas `py.typed`, donc les dix mille lignes d'annotations étaient
invisibles pour quiconque l'installait. `mypy` n'était pas en `strict`,
donc « mypy sans erreur » voulait dire beaucoup moins que ce que le
README laissait croire ; le passage a demandé quatorze corrections
réelles. Le générateur d'univers était recopié dans cinq fichiers, celui
du GARCH dans cinq aussi, avec des divergences déjà installées. Et le
profilage — jamais fait en quatorze séances — a trouvé un `np.quantile`
recalculé 24 021 fois sur le même tableau : plus de la moitié du temps
d'un rapport passée sur un invariant. Quatre objets valeur avaient par
ailleurs un `__eq__` qui **levait une exception** au lieu de répondre,
défaut vivant depuis la séance 2 parce qu'aucun test ne comparait deux
estimations.
→ `py.typed`, `CHANGELOG.md`, `tests/conftest.py`, `tests/test_paquet.py`,
`src/nkrisk/rendering.py`, `src/nkrisk/selection.py`, 23 tests

---

## 3. Les dix-neuf modules

| Module | Lignes | Couv. | Contenu |
|---|---|---|---|
| `types` | 185 | 88 % | `Estimate`, `InsufficientData` |
| `returns` | 476 | 91 % | rendements, volatilité, Sharpe, traînée |
| `covariance` | 276 | 70 % | empirique, Ledoit-Wolf, conditionnement |
| `ewma` | 535 | 89 % | pondération exponentielle, taille effective, ajustement de λ |
| `factors` | 923 | 94 % | facteurs statistiques, Marchenko-Pastur, attribution |
| `risk` | 364 | 86 % | Euler, diversification, paris effectifs, parité |
| `var` | 434 | 86 % | VaR/CVaR ×3, Kupiec, sous-additivité |
| `drawdown` | 349 | 89 % | courbe sous l'eau, épisodes, ulcère, Calmar |
| `portfolio` | 184 | 93 % | objet valeur gelé |
| `stress` | 516 | 91 % | moteur de scénarios par `Protocol` |
| `backtest` | 1 152 | 87 % | rejeu causal, audit, coûts, Sharpe dégonflé |
| `validation` | 998 | 88 % | biais, intervalle par blocs, fiche, inventaire |
| `shortfall` | 820 | 94 % | Acerbi-Székely, nulles explicites, perte de Fissler-Ziegel |
| `report` | 862 | 91 % | réconciliation des instruments, conflits interprétés |
| `rendering` | 216 | 98 % | la présentation du rapport, séparée du domaine |
| `selection` | 270 | — | Sharpe dégonflé, longueur minimale de piste |
| `data.market` | 432 | 89 % | sources injectables, validées |
| `data.quality` | 742 | 91 % | anomalies, alignement, annualisation |
| `data.synthetic` | 288 | 81 % | générateurs à vérité connue |

**Le point faible visible : `covariance` à 70 %**, la plus basse du
dépôt. Ce n'est pas que le module soit mal testé — Ledoit-Wolf est
validé à 10⁻¹⁵ contre scikit-learn — mais ses branches de diagnostic
(`diagnostic()`, les messages de conditionnement) ne sont pas
parcourues. C'est le prochain trou à boucher, et il est petit.

---

## 4. Le registre des erreurs

Trente-trois erreurs trouvées et corrigées, chacune verrouillée par un
test de non-régression. C'est la partie du projet qui a le plus de
valeur en entrevue, et c'est pour ça qu'on la garde.

| Séance | L'erreur | Le mécanisme |
|---|---|---|
| 1 | Conventions d'annualisation mélangées | L'écart de convention (0,33 pt) dépassait l'effet mesuré (0,13 pt) |
| 2 | Paris effectifs par Herfindahl | Mesurait la concentration, pas la diversification |
| 2 | Discontinuité de l'indicateur corrigé | Saut de 10 à 1 entre ρ = 0 et ρ = 0,001 |
| 2–5 | Narrations codées en dur | Chiffres écrits à la main qui ont dérivé |
| 4 | « L'ulcère est plus stable » | Il est 1,6 fois **plus** dispersé |
| 5 | « Les chocs ne commutent pas » | Ils commutent : l'état se sépare en parts disjointes |
| 5 | Preuve de semi-définie positivité incomplète | La borne est ρ ≥ −1/(n−1) |
| 6 | Oracle sans point initial | Toute baisse en début de série invisible |
| 7 | « L'erreur de l'ACP décroît en T » | Elle plafonne : l'ACP n'est pas de l'analyse factorielle |
| 7 | Docstring citant des chiffres périmés | Première dérive dans un docstring |
| 8 | Sondes de l'audit mal placées | Une avance d'un jour passait ; il faut les poser sur les décisions |
| 8 | `pytest.raises(Exception)` avec `noqa` | Passe sur n'importe quoi, y compris une faute de frappe |
| 9 | IPO tardive confondue avec jour férié | Trouvé par les vraies données (SHOP.TO) |
| 9 | « Ce n'est pas testable » | Faux : c'était *intestable*, par couplage au réseau |
| 9 | Diagnostic de calibration incohérent | Moyennait des volatilités ; il faut agréger des variances |
| 9 | `calibration_path` vide à step=1 | Retournait un DataFrame vide, puis `KeyError` |
| 10 | **CWE-22 dans mon propre code** | Deux évasions, dont l'absorption du chemin absolu par pathlib |
| 10 | Avertissements triviaux noyant le vrai | Cinq alertes d'un jour cachaient l'anomalie réelle |
| 11 | « La matrice EWMA devient singulière » | Non : elle reste inversible, et c'est pire que la singularité |
| 11 | Quatre prédictions réfutées, deux oracles cassés | Dont un oracle à 1,57 au lieu de 1,00 |
| 12 | **Oracle anticipant invisible** | Dans un GARCH, la variance du bloc est causée par les rendements qu'elle normalise |
| 12 | Test écrit à l'envers | Le sens du biais suit celui de la corrélation moyenne, ×(N−1) |
| 12 | Protocole aveugle à ce qu'il devait voir | Un portefeuille fixe n'inverse pas la matrice |
| 13 | **Trois formules de Fissler-Ziegel de mémoire** | Les trois réfutées par le test d'élicitabilité ; il a fallu redériver |
| 13 | Prédiction trop pessimiste sur la loi nulle | L'effet est réel mais modéré : 5,0 % → 7,4 % |
| 13 | `int(math.ceil(...))` — deuxième fois | RUF046 : `math.ceil` retourne déjà un entier |
| 14 | **Un test rendu aveugle, troisième fois** | La nulle empirique était la forme réalisée : le test ne pouvait plus la rejeter |
| 14 | Seuil fixe là où il fallait une loi | Le bootstrap mesurait la variance, pas la largeur réelle de l'estimateur |
| 14 | Erreur de syntaxe masquée par un `.pyc` | `import` réussissait sur du bytecode périmé ; `ruff` l'a vue |
| 15 | **Un invariant recalculé 24 021 fois** | 2,67 s sur 5,24 ; un rapport passe de 3,8 s à 2,0 s |
| 15 | **Quatre `__eq__` qui levaient une exception** | `dataclass` compare un `ndarray` avec `==` ; vivant depuis la séance 2 |
| 15 | `mypy` n'était pas strict | « sans erreur » reposait sur deux options ; 14 corrections réelles |
| 15 | `py.typed` absent | 10 000 lignes d'annotations invisibles pour qui installe le paquet |

**La leçon commune, et c'est la seule phrase à retenir de ce tableau :**
un bug numérique ne lève pas d'exception. Il produit un nombre
plausible, du bon ordre de grandeur, et faux. La seule défense connue
est de vérifier signe et ordre de grandeur contre une prédiction
théorique indépendante — et de vérifier l'oracle avant de lui faire
confiance.

---

## 5. Ce qui ne va pas, et qu'il faut dire

- **Aucune revue de code externe.** Un seul auteur, aucun regard
  critique indépendant sur l'ensemble. C'est la limite la plus réelle du
  projet.
- **Jamais exécuté en production ni sous charge.**
- **Validation essentiellement synthétique**, par choix méthodologique
  assumé : la vraie valeur d'une volatilité est inobservable sur données
  réelles, donc on ne peut pas y calibrer un estimateur. Mais le projet
  n'a été confronté qu'une fois aux vraies données, et cette fois-là a
  révélé une sous-estimation de 21 %.
- **Les facteurs extraits sont statistiques, donc sans nom.** Ils disent
  qu'il y a une cause commune, jamais laquelle.
- **L'intervalle par blocs de la séance 12 ne referme pas l'écart de
  couverture.** 87,7 % au lieu de 95 % dans le régime le plus difficile :
  48 blocs ne suffisent pas à estimer un moment d'ordre quatre.
- **Les actions GitHub de la CI sont référencées par étiquette mutable**
  et non par empreinte de commit. La marche à suivre est dans
  `SECURITE.md` ; ce n'est pas fait.
- **Pas de documentation générée.** Elle vit dans les docstrings et le
  README.
- **Cinq modules dépassent encore 800 lignes** : `validation` (998),
  `backtest` (953), `factors` (929), `shortfall` (894), `report` (862).
  Deux scissions ont été faites à la séance 15 ; les autres attendent.
- **Les actions de la CI restent épinglées par étiquette mutable.** La
  marche à suivre est dans `SECURITE.md` ; elle demande un accès à
  l'API GitHub que l'environnement de développement n'a pas.
- **110 noms publics, sans tri par maturité.** Le contrat de stabilité
  est écrit, mais rien ne distingue encore l'API éprouvée de
  l'expérimentale.
- **Le verdict de queue teste l'hypothèse gaussienne imposée par le
  protocole, pas les modèles.** Les quatre sont rejetés pour la même
  raison, donc la colonne ne discrimine pas. Voir §1bis.
- **Le rapport réel n'exclut pas SHOP.TO**, alors que sa propre section
  « données » recommande de le faire : il signale le coût sans en tirer
  la conséquence.
- **Le rapport ne cherche que les conflits qu'on lui a appris à
  reconnaître.** Deux instruments peuvent diverger d'une façon qu'il ne
  voit pas — même limite que les tests des séances 12 et 13.
- **La comparaison de deux modèles de queue est anticonservatrice.**
  `compare_tail_models` ne corrige pas l'autocorrélation des
  différences de perte ; sa p-value est donc trop petite sur données
  réelles. Déclaré dans la docstring, pas corrigé.

---

## 6. Ce qui vient ensuite — technique

Par ordre de valeur décroissante, telle que je la vois aujourd'hui.

1. **Une forme de queue par modèle, ajustée causalement.** Le verdict de
   queue impose aujourd'hui une queue gaussienne à tout le monde et
   rejette les quatre pour la même raison. Laisser chaque modèle
   déclarer sa forme — par un ν de Student ajusté sur ses propres
   résidus, à la manière de `fit_decay` à la séance 11 — rendrait la
   colonne discriminante. C'est la suite directe de ce que le premier
   rapport réel a montré.
2. **Vérifier que GitHub Pages sert bien `docs/rapport-validation.html`**
   et mettre le lien en tête du README. Le rapport existe avec de vraies
   données dedans ; personne ne sait encore qu'il existe.
3. **Correction HAC (Newey-West) sur `compare_tail_models`.** La
   p-value de la comparaison de deux modèles de queue ne corrige pas
   l'autocorrélation des différences de perte : elle est donc trop
   petite sur données réelles à volatilité groupée. C'est déclaré dans
   la docstring, ce qui est honnête mais pas suffisant.
4. **Le test de Christoffersen** (indépendance des dépassements), que la
   séance 3 avait explicitement mis hors périmètre. Kupiec ne voit pas
   si les dépassements sont groupés, et un modèle qui produit tous les
   siens pendant la même semaine de krach le passe.
5. **Protocole de validation pour les scénarios de stress.** La
   séance 12 valide les covariances, la séance 13 les queues. Le stress
   reste sans protocole, et le trou est nommé.
6. **Boucher `covariance` à 70 %.** Petit, rapide, et ça remonte la
   couverture globale.
7. **Exclure SHOP.TO et relancer.** Son démarrage tardif ampute
   l'univers de 97 séances pour un seul titre. Mesurer ce que
   l'exclusion change aux verdicts serait une expérience en soi.
8. **Épingler les actions de la CI par SHA.** Dix minutes, et ça ferme
   un point du modèle de menace.

---

## 7. Ce qui vient ensuite — et qui ne se code pas

L'état honnête du projet, au 24 novembre 2026 : **le dépôt est prêt, la
recherche d'emploi n'a pas commencé.** Trois mois sans revenu, zéro appel
de découverte passé, et une candidature refusée avant l'entretien
technique — donc sur le dossier, pas sur la compétence. L'outil n'a jamais été le prérequis de ces appels,
et il l'est encore moins depuis la séance 12.

- [ ] Écrire le coussin de trésorerie en deux chiffres : ce qu'il y a,
      ce qui part par mois
- [ ] Les quatre agences : Alteo, Fed IT, S.i. Systems, Robert Half
      Technologie
- [ ] Chercher les offres « validation de modèles », « model risk »,
      « risque de modélisation », « gouvernance de l'IA » en institution
      sous BSIF ou AMF
- [ ] Trois contacts de découverte

**L'échéance qui compte** : la ligne directrice E-23 du BSIF entre en
vigueur le **1er mai 2027** pour toutes les institutions financières
fédérales — dans **158 jours**. Ces postes se créent maintenant, et
presque aucun candidat junior ne peut montrer un exemple travaillé de
validation de modèles. Depuis les séances 12 et 13, toi si.

Le détail du raisonnement est dans `nkrisk-plan-affaires.md` ; les
formulations prêtes à l'emploi sont dans `nkrisk-cv-linkedin.md`,
section 7.A.

---

## 8. Reprendre le travail

```bash
cd ~/Desktop/nkrisk

pip install -e ".[dev]"

pytest -q                     # 369 tests rapides
pytest -q -m ""               # 390 tests, Monte Carlo inclus
pytest -q -m "" --cov=src/nkrisk --cov-report=term
ruff check src/ tests/ examples/
mypy src/nkrisk

python examples/seance14.py   # la dernière séance
python examples/rapport_reel.py      # le rapport sur vraies actions (réseau + yfinance)
python examples/donnees_reelles.py   # demande le réseau (yfinance)
```

Les démonstrations `examples/seance1.py` à `seance14.py` se lisent dans
l'ordre : chacune raconte ce qu'elle a mesuré, y compris quand la mesure
a contredit ce que j'avais prédit.

---

*Ce fichier se met à jour à la fin de chaque séance. Tous les chiffres
sont reproductibles depuis le dépôt ; aucun n'est recopié d'une autre
source.*
