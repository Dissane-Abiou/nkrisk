# Dossier de défense — nkrisk
### Entretien technique, poste de développement logiciel

**Yannick Dissane Abiou · github.com/Dissane-Abiou/nkrisk**

---

## Comment utiliser ce dossier

Il n'est pas fait pour être lu, mais pour être **travaillé**. La section 7
contient huit exercices où tu modifies le code toi-même. C'est la seule
partie qui compte vraiment : on ne défend bien que ce qu'on a manipulé.

Le reste sert de référence pendant que tu fais les exercices.

---

## 1. La question de la paternité

Elle viendra peut-être, sous une forme ou une autre : « comment tu as
construit ça ? », « tu as utilisé des outils ? », ou simplement une
demande de modifier une fonction en direct.

**La réponse honnête est la bonne réponse.** Quelque chose comme :

> « Je l'ai construit en binôme avec une IA, en travaillant par séances
> thématiques. Elle écrivait, je validais et je questionnais. Ce que
> j'ai appris, c'est surtout la méthode de vérification — plusieurs
> erreurs ont été trouvées parce que les chiffres de sortie
> contredisaient ce qu'on attendait, pas parce qu'un test plantait. Je
> peux te dérouler n'importe quelle décision du projet. »

Personne ne te reprochera l'assistance IA en 2026. Ce qu'on sanctionne,
c'est l'incapacité à expliquer ou à modifier ce qu'on présente.

**Ce qui rend cette réponse crédible** : les six bugs documentés dans le
dépôt. Ils prouvent qu'il y a eu une vraie boucle de vérification, pas
une génération avalée telle quelle. Mets-les en avant au lieu de les
cacher.

🔴 **Ne dis jamais** « j'ai tout écrit moi-même » si tu ne peux pas
modifier le code en direct. C'est le seul scénario réellement
catastrophique.

---

## 2. Le pitch de 90 secondes

À répéter à voix haute jusqu'à ce qu'il sorte sans effort.

> nkrisk est un moteur de mesure de risque de portefeuille en Python,
> construit autour d'une contrainte unique : **aucune fonction ne
> retourne un nombre nu.**
>
> Chaque estimation revient dans un type `Estimate` qui transporte son
> intervalle de confiance, son nombre d'observations, la méthode
> utilisée et ses hypothèses de validité. Si l'imprécision relative
> dépasse 25 %, l'objet se déclare lui-même peu fiable, et certaines
> fonctions lèvent une exception plutôt que de répondre.
>
> L'idée vient d'un constat : une volatilité estimée sur un mois de
> données est comprise entre 16 % et 31 %, mais les outils grand public
> affichent « 21 % » sans fourchette. L'incertitude est dans le système
> de types pour qu'il soit impossible de l'oublier en aval.
>
> Il y a cinq modules — rendements, covariance, décomposition du risque,
> VaR, drawdown — et 64 tests organisés en quatre niveaux de rigueur
> croissante. Le niveau qui m'intéresse le plus est le troisième : on
> simule mille échantillons, on construit mille intervalles à 95 %, et
> on vérifie que 95 % d'entre eux contiennent la vraie valeur. Ça ne
> valide pas l'estimation, ça valide **l'incertitude elle-même**.

**Les trois accroches à placer**, selon ce qui intéresse ton
interlocuteur : l'incertitude dans le type, les quatre niveaux de test,
ou les bugs documentés.

---

## 3. Architecture — les décisions et leur justification

### 3.1 `types.py` — pourquoi un type plutôt qu'un float

**La décision.** Aucune fonction ne retourne `float`. Tout retourne
`Estimate`, un dataclass gelé qui valide sa cohérence à la construction.

**Ce qu'on te demandera.** « Pourquoi pas juste retourner un tuple, ou
un dict ? C'est plus lourd, non ? »

**Ce que tu réponds.**

> Un tuple n'empêche rien. Si une fonction en aval ne veut que la
> valeur, elle écrit `vol[0]` et l'incertitude disparaît sans que
> personne le remarque. Avec un type nommé, il faut écrire
> `vol.value` — un geste conscient qui dit « je choisis d'ignorer
> l'intervalle ».
>
> Et surtout, le type porte de la logique : `relative_uncertainty`,
> `is_meaningful()`, et la validation au `__post_init__`. Un `Estimate`
> dont la valeur sort de son propre intervalle ne peut pas exister —
> l'objet refuse d'être construit. C'est le principe « make illegal
> states unrepresentable ».

**L'anecdote qui prouve que ça sert.** En écrivant le module drawdown,
j'ai voulu encoder une sémantique dans l'intervalle — borne basse = moins
grave — en inversant `ci_low` et `ci_high`. Comme les drawdowns sont
négatifs, `ci_low` valait −0,19 et `ci_high` −0,54. La validation a
refusé l'objet. Une contrainte que j'avais écrite plusieurs séances plus
tôt a attrapé son propre auteur.

**Si on pousse** : « et le coût en performance ? » → Un dataclass gelé,
c'est une allocation. Sur un moteur qui calcule quelques centaines
d'estimations, c'est invisible. Si on passait à des millions
d'évaluations dans une boucle serrée, je séparerais un chemin rapide
rendant des tableaux numpy bruts, avec l'enveloppe `Estimate`
uniquement à la frontière publique.

---

### 3.2 La stratégie d'échec — « fail loud »

**La décision.** Données manquantes, prix négatifs, poids désalignés :
tout lève une exception. Jamais de `NaN` propagé, jamais de valeur par
défaut silencieuse.

**Ce qu'on te demandera.** « C'est agressif. Et si l'utilisateur veut
juste continuer ? »

**Ce que tu réponds.**

> Dans un moteur de risque, échouer bruyamment est une vertu. Un `NaN`
> traverse les calculs, finit dans une case de tableau, et l'utilisateur
> l'interprète comme un détail technique. Une exception arrête le
> programme et force à traiter le cas.
>
> Exemple concret : si deux dates sont en double dans un CSV, on refuse
> plutôt que de dédupliquer. Garder la première ou la dernière change le
> résultat — c'est une décision métier, pas une décision de bibliothèque.
> Le moteur ne doit pas la prendre à la place de l'utilisateur, surtout
> pas en silence.

**Si on pousse** : « donc impossible d'utiliser ça en production sur des
données sales ? » → C'est exactement le but : on nettoie explicitement en
amont, dans une couche dont c'est le travail. Mélanger nettoyage et
calcul est ce qui rend les pipelines financiers indébogables.

---

### 3.3 `data/` — pourquoi générer des données synthétiques

**La décision.** Un générateur de prix à propriétés statistiques connues
(mouvement brownien géométrique, Student-t corrélés), séparé des
chargeurs Yahoo/CSV. La graine aléatoire est un paramètre **obligatoire**,
pas optionnel.

**Ce qu'on te demandera.** « Pourquoi ne pas tester sur de vraies
données ? »

**Ce que tu réponds.** C'est la réponse la plus importante du dossier.

> Avec des données réelles, on ne peut pas valider un estimateur. On
> calcule une volatilité de 18,3 % et... c'est tout. La vraie volatilité
> du marché est inobservable, donc on n'a aucun moyen de savoir si le
> code est juste. Un bug qui multiplie tout par 1,02 passerait inaperçu
> pour toujours.
>
> Avec des données synthétiques, je **fixe** la vérité. Je génère une
> série dont je sais qu'elle a 20 % de volatilité, je lance
> l'estimateur, et je vérifie qu'il retrouve 20 %. C'est la seule façon
> rigoureuse de valider.
>
> Les données réelles servent à l'usage, jamais à la validation.

**Sur la graine obligatoire.** Un test qui utilise de l'aléatoire non
reproductible échouera un jour sans qu'on puisse comprendre pourquoi.
J'ai donc rendu `seed` obligatoire plutôt qu'optionnel avec défaut — le
type force à y penser. Preuve que ça marche : la sortie des scripts de
démonstration est identique au chiffre près entre mon Linux et un
Windows sous Python 3.14.

---

### 3.4 L'abstraction de la source de données

**La décision.** Toutes les fonctions prennent une `pd.Series` de prix.
Aucune ne sait d'où elle vient.

**Ce qu'on te répondra peut-être.** « C'est de l'inversion de dépendance
basique. »

**Ce que tu ajoutes pour montrer que c'est motivé, pas réflexe.**

> Oui, et en finance elle a une conséquence très concrète : les
> fournisseurs de données changent constamment — API dépréciées, quotas
> modifiés, licences révisées. Un moteur couplé à un fournisseur meurt
> avec lui. Un moteur qui prend une Series survit à tout.
>
> Et ça rend possible la stratégie de test : je peux injecter des
> données synthétiques exactement comme des données réelles.

---

### 3.5 `covariance.py` — le compromis biais-variance

**La décision.** Implémenter la covariance empirique *et* la contraction
de Ledoit-Wolf, avec un diagnostic de conditionnement sur l'objet.

**Ce qu'on te demandera.** « Explique-moi Ledoit-Wolf en une minute. »

**Ce que tu réponds** (version développeur, pas version quant) :

> Pour N actifs, la matrice de covariance a N(N+1)/2 paramètres. Avec
> 100 actifs ça fait 5 050 paramètres, estimés sur 252 jours
> d'historique. Le problème est massivement sous-déterminé.
>
> Conséquence concrète : la matrice devient très mal conditionnée. Son
> conditionnement — le rapport entre la plus grande et la plus petite
> valeur propre — passe de 75 à plus d'un million quand le rapport N/T
> approche 1. Et comme l'optimisation de portefeuille **inverse** cette
> matrice, l'inversion amplifie le bruit de façon catastrophique.
>
> Ledoit-Wolf tire l'estimation empirique vers une cible rigide et bien
> conditionnée, avec une intensité calculée en forme close. On accepte
> volontairement un biais pour effondrer la variance. Mesuré dans le
> projet : le portefeuille de variance minimale produit un levier moyen
> de 19,8 avec la covariance empirique contre 6,1 avec contraction, sur
> des échantillons tirés de la même distribution.

**Le détail qui impressionne.** Mon implémentation est comparée à celle
de scikit-learn dans un test. Concordance à 10⁻¹⁸ sur l'intensité de
contraction et sur la matrice complète, dans quatre régimes N/T
différents. Quand une implémentation de référence existe, s'en priver
est une faute.

---

### 3.6 `risk.py` — l'invariant qui protège le calcul

**La décision.** La décomposition du risque vérifie l'identité d'Euler
à chaque appel et **lève une exception** si elle est violée.

**Ce qu'on te demandera.** « Un assert en production, c'est sage ? »

**Ce que tu réponds.**

> Ce n'est pas un assert de débogage, c'est un invariant mathématique.
> σₚ = √(wᵀΣw) est homogène de degré 1, donc le théorème d'Euler garantit
> que la somme des contributions égale exactement la volatilité totale —
> à la précision machine, mesurée à 5·10⁻¹⁷.
>
> Si cette égalité est violée, ce n'est pas une imprécision : c'est la
> preuve d'un bug en amont, et tous les chiffres produits sont faux.
> Continuer serait pire que s'arrêter.
>
> Cela dit, si on me demandait de le mettre en production tel quel, je
> remplacerais l'`AssertionError` par une exception métier nommée, et
> j'ajouterais une métrique plutôt qu'un simple crash — pour qu'une
> alerte parte avant que quelqu'un prenne une décision sur un chiffre
> faux.

Cette dernière phrase vaut cher : elle montre que tu distingues un
projet personnel d'un système de production.

---

### 3.7 L'organisation des tests — ton meilleur atout

**La décision.** Quatre niveaux, du plus faible au plus fort.

| Niveau | Ce qu'il attrape | Exemple |
|---|---|---|
| 1 · Propriétés | fautes de frappe | `r = eˡ − 1` doit tenir exactement |
| 2 · Calibration | erreurs de formule, facteurs d'échelle | σ fixée à 20 %, l'estimateur doit la retrouver |
| 3 · **Couverture** | **une incertitude mal calibrée** | 1 000 intervalles à 95 % → ~95 % contiennent la vérité |
| 4 · Validation croisée | un bug partagé avec l'auteur | comparaison avec scikit-learn |

**Ce qu'on te demandera.** « Explique le niveau 3. »

> Un intervalle de confiance à 95 % n'est correct que s'il contient la
> vraie valeur dans 95 % des cas. Ce n'est pas une évidence, c'est une
> affirmation **testable**. Je génère mille échantillons d'une loi dont
> je connais la volatilité, je construis mille intervalles, et je
> compte. Si j'obtiens 80 %, mon moteur ment sur sa propre précision.
>
> Presque personne n'écrit ce niveau. C'est pourtant le seul qui garantit
> que le logiciel ne surestime pas ce qu'il sait.

**Les tests lents.** Les trois tests de couverture font mille simulations
et prennent 30 secondes. Ils sont marqués `@pytest.mark.slow` et exclus
par défaut : la boucle de développement reste à 2 secondes. Si on te
demande pourquoi exclure les plus importants → c'est un arbitrage
assumé entre rigueur et ergonomie, parce qu'une suite lente est une
suite qu'on cesse de lancer. En intégration continue, on lance tout.

---

## 4. Les six bugs — à raconter, pas à cacher

C'est la section qui te distinguera. Prépare-en **deux** à raconter en
détail ; les autres servent de réserve.

### Bug 1 — la convention d'annualisation *(le meilleur à raconter)*

Je comparais un rendement annualisé **linéairement** (×252) à un autre
annualisé par **composition** ((1+r)²⁵² − 1). L'écart de convention
valait 0,33 point ; l'effet que je voulais mesurer en valait 0,13. Le
tableau affichait donc une traînée de volatilité **négative** —
exactement l'inverse de la réalité.

Rien n'a planté. Aucun test ne l'a vu, parce qu'ils utilisaient tous un
régime de forte volatilité où le vrai effet écrase l'artefact. Je l'ai
trouvé en regardant la sortie et en constatant qu'un signe n'allait pas.

**La leçon à énoncer** : un bug numérique ne lève pas d'exception. Il
produit un nombre plausible, du bon ordre de grandeur, et faux. La seule
défense est de vérifier signe et ordre de grandeur contre une prédiction
théorique indépendante — ici σ²/2.

### Bug 2 — un indicateur conceptuellement faux

J'avais défini le « nombre effectif de paris » comme l'inverse de
l'indice de Herfindahl sur les contributions au risque. Sur huit actifs
corrélés à 0,9999 et équipondérés, il retournait 8 — alors qu'il n'y a
manifestement qu'un seul pari détenu huit fois.

La formule mesurait la **concentration** du risque entre les lignes, pas
la **diversification**. Deux notions différentes que le cas limite sépare.

**Bonus, si on creuse** : en testant la version corrigée (entropie des
composantes principales, Meucci 2009), j'ai découvert qu'elle saute
discontinûment de 10 à 1 entre ρ = 0 et ρ = 0,001 sur un portefeuille
parfaitement symétrique. C'est mathématiquement correct et pratiquement
inutilisable. **Un indicateur publié peut avoir un domaine de validité
que l'article ne met pas en avant.**

### Bug 3 — l'algorithme qui diverge

Ma première parité de risque utilisait une mise à jour multiplicative
naïve. L'obligation, peu volatile et négativement corrélée, faisait
exploser le terme en 1/(Σw)ᵢ et produisait des poids négatifs.

**Le symptôme qui l'a révélé** : un ratio de diversification de 0,91.
C'est mathématiquement impossible en position longue — il vaut au
minimum 1. Remplacé par une descente par coordonnées cycliques sur le
programme convexe de Spinu, qui garantit des poids strictement positifs.

### Bug 4 — l'affirmation invalidée par la mesure

J'avais présenté l'indice d'ulcère comme « le pendant de ce que la CVaR
est à la VaR : une moyenne plutôt qu'un extremum, donc plus stable ».
Mesuré sur 200 trajectoires : coefficient de variation 0,47 contre 0,30.
Il est **plus** dispersé, pas moins. L'analogie était séduisante et
fausse.

### Bug 5 — la narration codée en dur

Trois phrases des scripts de démonstration contredisaient les tableaux
affichés juste au-dessus, parce que les nombres avaient été écrits à la
main à partir d'une exécution antérieure. Toutes les narrations sont
désormais **calculées** à partir des mêmes données que les tables.

### Bug 6 — le type qui attrape son auteur

Raconté en 3.1.

---

## 5. Les quinze questions les plus probables

**Sur le projet**

1. *Raconte-moi ce projet en deux minutes.* → section 2
2. *Pourquoi un type plutôt qu'un float ?* → 3.1
3. *Comment tu sais que ton code est juste ?* → 3.7, et insiste sur le niveau 2
4. *Explique un bug que tu as trouvé.* → section 4, bug 1
5. *Qu'est-ce que tu ferais différemment ?* → section 6
6. *Comment tu ajouterais le support des devises multiples ?* → réfléchis-y avant, ne pas improviser
7. *Pourquoi Python et pas un langage compilé ?* → numpy/scipy délèguent à du BLAS compilé ; le goulot est l'algèbre, pas l'interprète. Pour du temps réel on repenserait.
8. *Ton code est-il thread-safe ?* → oui par construction : toutes les structures sont gelées et les fonctions sont pures. Aucun état mutable partagé.

**Sur les pratiques**

9. *Comment tu organises tes tests ?* → les quatre niveaux
10. *Tu fais du TDD ?* → honnêtement non, pas strictement. J'écris le calcul, puis la validation contre une vérité connue. Ne prétends pas.
11. *Comment tu gères les dépendances ?* → `pyproject.toml`, extras `[dev]` et `[data]`, scikit-learn optionnel avec `pytest.importorskip` pour qu'une dépendance de confort ne casse jamais la suite.
12. *Et l'intégration continue ?* → **il n'y en a pas**. C'est la première chose que j'ajouterais (voir exercice 8).

**Les pièges**

13. *Ce projet est-il utilisé en production ?* → Non. Sa valeur est de démontrer une méthode, pas un déploiement. Ne jamais prétendre le contraire.
14. *Combien de temps ça t'a pris ?* → Dis la vérité. Un rythme rapide avec assistance IA est normal et attendu.
15. *Tu pourrais ajouter telle fonctionnalité maintenant, devant moi ?* → **C'est pour ça que la section 7 existe.**

---

## 6. « Qu'est-ce que tu ferais différemment ? »

Prépare trois réponses honnêtes. Une critique lucide de son propre
travail est le signal de maturité le plus recherché.

**1. Aucune intégration continue.** Les tests existent mais rien ne les
lance automatiquement. Une action GitHub sur chaque push serait le
premier ajout — c'est exactement ce qui empêche une régression de passer.

**2. Pas de vérification de types statique.** Les annotations sont là
mais `mypy` n'est jamais exécuté. Des annotations non vérifiées sont de
la documentation, pas une garantie.

**3. Le découplage données/calcul est incomplet.** Les fonctions prennent
une `pd.Series`, donc elles dépendent quand même de pandas. Un protocole
explicite ou des tableaux numpy nus seraient plus propres pour une
bibliothèque destinée à être intégrée ailleurs.

**4. Le français dans le code.** Choix assumé — les commentaires
expliquent, et le marché visé est québécois. Mais pour une bibliothèque
publique internationale, l'anglais serait le bon choix. Sache défendre
le compromis plutôt que de l'ignorer.

---

## 7. Les huit exercices — la partie qui compte

Fais-les dans l'ordre. Chacun se vérifie par un test que tu écris aussi.
**Ne me demande pas la solution avant d'avoir essayé** : l'objectif n'est
pas d'avoir le code, c'est de pouvoir le réécrire en entretien.

### Exercice 1 — lire et expliquer *(1 h)*
Ouvre `types.py`. Écris, dans tes mots et sans regarder ce dossier, à
quoi sert chaque méthode de `Estimate` et pourquoi `__post_init__` fait
ces quatre vérifications. Compare ensuite avec la section 3.1.

### Exercice 2 — ajouter le ratio de Sortino *(2 h)*
Dans `returns.py`, ajoute `sortino_ratio`. C'est le Sharpe où le
dénominateur ne compte que la volatilité des rendements **négatifs**.
Retourne un `Estimate`. Écris trois tests : une propriété, une
calibration, un cas limite.
*Ce que ça vérifie* : tu sais naviguer le module et respecter sa convention.

### Exercice 3 — casser puis réparer *(1 h)*
Remplace `np.sqrt(252)` par `252` dans `annualized_volatility`. Lance la
suite. **Note quels tests échouent et lesquels passent encore.** Remets
en état. Tu dois pouvoir dire en entretien quel niveau de test attrape
quel type d'erreur.

### Exercice 4 — le drawdown conditionnel *(3 h)*
Ajoute `conditional_drawdown_at_risk(prices, confidence)` dans
`drawdown.py` : la moyenne des `(1−confidence)` pires valeurs de la
courbe de drawdown. C'est à la courbe de drawdown ce que la CVaR est aux
rendements. Teste qu'elle est toujours comprise entre le drawdown moyen
et le maximum.

### Exercice 5 — covariance à pondération exponentielle *(4 h)*
Ajoute `ewma_covariance(returns, halflife)` dans `covariance.py` : une
covariance où les observations récentes pèsent plus. Retourne un
`CovarianceEstimate`. Vérifie qu'avec une demi-vie infinie elle converge
vers la covariance empirique.
*Ce que ça vérifie* : tu peux étendre une abstraction existante sans la casser.

### Exercice 6 — trouver un vrai problème *(2 h)*
Dans `drawdown_episodes`, le `peak` du tout premier épisode est pris à
`debut − 1`, et à l'indice 0 si l'épisode commence au premier point.
Construis un cas où ce comportement est discutable. Décide s'il faut le
corriger ou le documenter, et justifie ton choix.
*Ce que ça vérifie* : tu lis le code de façon critique, pas en lecteur passif.

### Exercice 7 — faire du bruit utile *(2 h)*
Ajoute une option `strict=False` à `annualized_volatility` qui, quand
elle vaut `True`, lève une exception au lieu de retourner un `Estimate`
marqué peu fiable. Réfléchis à l'argument contre cette option avant de
l'écrire.
*Ce que ça vérifie* : tu sais discuter un choix d'API, pas seulement coder.

### Exercice 8 — l'intégration continue *(2 h)*
Crée `.github/workflows/tests.yml` : une action qui lance les tests
rapides sur chaque push et la suite complète sur la branche principale.
Ajoute le badge de statut au README.
*Ce que ça vérifie* : c'est la faiblesse n° 1 du projet, et la corriger
toi-même transforme une critique en démonstration.

---

## 8. Plan sur dix jours

| Jours | Travail |
|---|---|
| 1 | Exercice 1. Lire `types.py` et `returns.py` en entier, ligne à ligne. |
| 2 | Exercice 2 (Sortino). Répéter le pitch à voix haute cinq fois. |
| 3 | Exercice 3 (casser/réparer). Relire la section 4 sur les bugs. |
| 4 | Lire `covariance.py` et `risk.py`. Redémontrer l'identité d'Euler sur papier. |
| 5 | Exercice 5 (EWMA). Le plus long — prévois la journée. |
| 6 | Exercice 8 (intégration continue). Pousser et vérifier que le badge passe au vert. |
| 7 | Exercices 4 et 6. |
| 8 | Exercice 7. Préparer les trois réponses de la section 6. |
| 9 | Simulation : je te pose les quinze questions, tu réponds sans notes. |
| 10 | Révision des points faibles trouvés le jour 9. |

---

## 9. Ce qu'il ne faut jamais dire

- « J'ai tout écrit moi-même » *(si tu ne peux pas le modifier en direct)*
- « C'est en production »
- « J'ai construit un Aladdin » → dis « inspiré du périmètre d'Aladdin »
- « Il n'y a pas de bug » → il y en a toujours ; tu en as documenté six
- Inventer une réponse quand tu ne sais pas → **« Je ne sais pas, mais
  voilà comment je le vérifierais »** est une excellente réponse en
  entretien technique. Elle est même souvent testée exprès.

---

*Dernière mise à jour : novembre 2026*
