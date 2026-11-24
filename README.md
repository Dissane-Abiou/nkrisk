# nkrisk

[![CI](https://github.com/Dissane-Abiou/nkrisk/actions/workflows/ci.yml/badge.svg)](https://github.com/Dissane-Abiou/nkrisk/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/Dissane-Abiou/nkrisk)
[![Licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Sécurité](https://img.shields.io/badge/mod%C3%A8le%20de%20menace-document%C3%A9-blue)](SECURITE.md)

**Moteur de mesure de risque de portefeuille qui déclare son incertitude.**

Aucune fonction ne retourne un nombre nu. Chaque estimation transporte son
intervalle de confiance, son nombre d'observations, sa méthode et ses
hypothèses de validité. Quand les données ne suffisent pas, le moteur refuse
de répondre plutôt que de deviner.

```python
from nkrisk import annualized_volatility, log_returns
from nkrisk.data import YahooSource

prix = YahooSource().fetch("XIC.TO", start="2015-01-01")
vol = annualized_volatility(log_returns(prix))
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
pytest tests/ -q          # 369 tests rapides (~15 s)
pytest tests/ -q -m ""    # 390 tests, couverture Monte Carlo incluse (~50 s)
ruff check src/ tests/ examples/
mypy src/nkrisk
```

Python ≥ 3.11 · numpy · pandas · scipy · scikit-learn (optionnel, pour une
validation croisée)

---

## Ce que le moteur calcule

| Module | Contenu |
|---|---|
| `returns` | rendements simples et log, volatilité, ratio de Sharpe, traînée de volatilité — tous avec intervalles de confiance |
| `covariance` | covariance empirique et contraction de Ledoit-Wolf, avec diagnostic de conditionnement |
| `ewma` | pondération exponentielle, taille d'échantillon effective portée dans l'intervalle, ajustement de λ par vraisemblance |
| `risk` | décomposition du risque : contributions marginale et composante, ratio de diversification, nombre effectif de paris |
| `var` | VaR et CVaR par trois méthodes, backtest de Kupiec, contre-exemple de sous-additivité |
| `drawdown` | courbe sous l'eau, épisodes datés, indice d'ulcère, Calmar, distribution simulée du maximum |
| `factors` | modèle à facteurs statistiques, nombre de facteurs par Marchenko-Pastur, attribution systématique / spécifique, scénarios de facteur |
| `portfolio` | objet valeur gelé : poids + modèle de risque, validés une seule fois à la construction |
| `stress` | moteur de scénarios — tout choc est une fonction Portefeuille → Portefeuille |
| `backtest` | rejeu point-in-time strict, audit de causalité, coûts de transaction, Sharpe dégonflé |
| `data.market` | sources injectables (Yahoo, CSV, mémoire) — testables sans réseau |
| `data.quality` | détection d'anomalies, politiques d'alignement de calendriers, annualisation correcte |
| `validation` | statistique de biais, intervalle par blocs, fiche de modèle, inventaire — la validation indépendante du moteur par lui-même |
| `shortfall` | backtest d'Expected Shortfall (Acerbi-Székely), valeurs critiques simulées, perte de Fissler-Ziegel — la mesure que Bâle impose depuis FRTB |
| `data.synthetic` | générateurs à vérité connue (GBM, Student-t corrélés) |

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

**Un modèle à facteurs bat la contraction, et l'écart grandit quand les
données manquent.** Volatilité réalisée hors échantillon du portefeuille de
variance minimale, contre Ledoit-Wolf : −9 % à N/T = 0,2, −17 % à N/T = 0,4,
−36 % à N/T = 1,5 — régime où la covariance empirique est singulière et où il
n'y a tout simplement pas de portefeuille à construire.

**Mais quand son hypothèse est fausse, il devient sur-confiant.** Sur un
univers sans structure à facteurs, l'avantage disparaît *et* le rapport
risque prédit / risque réalisé tombe à 0,89, contre 1,05 pour Ledoit-Wolf. Se
tromper en sous-estimant est la pire façon de se tromper pour un moteur de
risque.

**Du bruit pur n'a pas les valeurs propres qu'on croit.** Cent actifs
indépendants sur 250 jours produisent un rapport de 21 entre la plus grande et
la plus petite valeur propre de leur corrélation — alors que la vraie matrice
est l'identité. Marchenko-Pastur prédit l'intervalle à 0,01 près. C'est ce qui
permet de choisir le nombre de facteurs sans seuil arbitraire ; le taux de
faux positifs mesuré est de 6,4 %.

**Une fuite d'un seul jour transforme un backtest en fiction.** Sur un marché
simulé où l'espérance de rendement de chaque actif est rigoureusement nulle,
une stratégie qui choisit ses quatre actifs en connaissant la suite multiplie
le capital par 19 là où la même stratégie honnête le multiplie par 3,3. Elle ne
plante pas, ne produit aucun NaN et respecte la signature du contrat.

**L'audit de causalité la démasque sans lire son code.** On corrompt les
données postérieures à une date, on rejoue, et on vérifie qu'aucune décision
antérieure n'a bougé. Trois fuites de natures différentes détectées sur trois,
zéro fausse alerte sur trois stratégies honnêtes — y compris une avance d'un
seul jour sur huit ans de données, soit 0,05 % de l'historique.

**Un Sharpe de 1,0 obtenu après 12 650 essais vaut moins que rien.** C'est le
nombre de façons de choisir 4 actifs parmi 25 — toutes essayées par une
stratégie qui « garde les quatre meilleurs ». Le Sharpe dégonflé fait tomber la
probabilité que le résultat soit réel de 99,6 % à 10,3 %, sur exactement les
mêmes rendements.

**Le choix d'alignement des calendriers déplace le risque plus que bien des
choix de modélisation.** Mesuré contre une vérité connue, à 10 % de séances non
communes : reporter le dernier prix ne biaise pas les volatilités (−0,2 %) mais
sous-estime la corrélation de 10,2 %, et donc le risque du portefeuille de
4,2 %. Toujours dans le sens optimiste. L'intersection, elle, n'est biaisée que
si on l'annualise par 252 alors qu'elle ne retient que 211 observations par an.

**Sur de vraies actions, le modèle de risque annonce 0,79 fois le risque
subi.** Mesuré sur douze titres de Toronto et New York, de 2015 à 2026 : la
variance minimale sur Ledoit-Wolf sous-estime sa propre volatilité réalisée de
21 %, contre 0,99 sur données simulées à volatilité constante. Le diagnostic
intégré montre que l'écart est **concentré** : une fenêtre glissante voit la
tempête une fois qu'elle est entrée dedans.

**Une faille de traversée de répertoire a survécu une séance dans ce dépôt.**
`Path(dossier) / f"{ticker}.csv"` laissait `fetch("../prive/secrets")` lire
hors du périmètre, et `fetch("/etc/passwd")` faire pire : l'opérateur `/` de
pathlib **abandonne son membre de gauche** quand celui de droite est absolu.
Corrigée par liste blanche *et* confinement du chemin résolu — un test par
couche montre que chacune attrape ce que l'autre laisse passer. Modèle de
menace complet dans [SECURITE.md](SECURITE.md).

**L'avantage d'un estimateur réactif est en TEMPS ; son coût est en
DIMENSION.** La pondération exponentielle bien réglée bat la fenêtre longue
quand on prévoit la volatilité d'une série (E[z²] de 1,09 contre 1,11, pour une
borne causale à 0,99), et perd dès qu'on passe par une matrice de covariance
(1,24 contre 1,17). Une série a un paramètre à estimer ; une matrice N×N en a
N(N+1)/2, portés par les mêmes 66 observations effectives.

**Et λ doit correspondre à l'horizon de prévision.** Le 0,94 de RiskMetrics est
calibré pour une VaR à un jour : employé pour prévoir un mois, il bruite plus
qu'il n'anticipe (1,14 contre 1,09 pour λ=0,97, dont la demi-vie de 23 jours
épouse l'horizon).

**La VaR n'est pas une mesure de risque cohérente.** Deux obligations
indépendantes à 4 % de défaut ont chacune une VaR à 95 % de zéro ; leur
combinaison a une VaR de 0,5. La VaR affirme que diversifier crée du risque.

---

## Les bugs que j'ai commis, et pourquoi ils sont documentés

Les erreurs commises pendant la construction sont conservées dans le dépôt,
chacune avec le test de non-régression qui la verrouille. Elles ont plus de
valeur pédagogique que le code correct.

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

**Un estimateur qui ne converge pas là où je le croyais.** J'avais écrit le
test évident : « l'erreur doit décroître quand l'historique s'allonge ». Il a
échoué. À univers fixé, l'erreur de l'ACP PLAFONNE : le dernier quadruplement
de l'historique ne gagne plus rien de mesurable. L'ACP n'est pas de l'analyse
factorielle — elle suppose les variances spécifiques égales, et le biais qui en
résulte dépend du nombre d'actifs, pas de la longueur de l'historique. Mesuré :
quatre cents actions sur huit ans battent quinze actions sur deux siècles.
→ `test_allonger_lhistorique_ne_fait_pas_converger_le_modele`,
`test_un_grand_univers_bat_un_long_historique`

**Un détecteur posé au mauvais endroit.** La première version de l'audit de
causalité plaçait ses sondes uniformément sur l'historique. Elle attrapait une
stratégie calibrée sur tout l'échantillon et ratait une avance d'un jour — la
fuite la plus fréquente. Une décision prise en position t avec a périodes
d'avance lit jusqu'à t+a−1 : elle n'est démasquée que si la corruption commence
à un indice qu'elle atteint. Avec des sondes au hasard et un rebalancement
mensuel, la probabilité valait 1/21. En posant la sonde SUR la date de
décision, toute avance ≥ 1 est vue. La sensibilité d'un détecteur ne vient pas
de sa finesse, mais de l'endroit où on le pose.
→ `test_une_sonde_mal_placee_laisse_passer_la_fuite`

**Une prédiction à moitié fausse sur un biais.** J'avais écrit que reporter le
dernier prix connu, en injectant des rendements nuls, réduisait la variance ET
la corrélation. La mesure a donné autre chose : la volatilité ne bouge
quasiment pas, parce que les zéros la tirent vers le bas pendant que la séance
suivante, qui absorbe le mouvement manqué, la tire vers le haut. Rien ne
compense pour la corrélation. Le biais est donc entier là où il fait le plus de
dégâts — la diversification apparente.
→ `test_le_report_ne_biaise_pas_la_volatilite_mais_la_correlation`

**« Ce n'est pas testable » était un mauvais diagnostic.** `data/market.py` est
resté à 0 % de couverture pendant huit séances, avec cette justification : la
fonction appelle le réseau. Faux — elle était intestable parce qu'elle
*décidait* d'appeler le réseau. Une fois la source passée en paramètre, le
module est passé à 87 % sans une ligne de test réseau, et il ne reste non
couvert que l'adaptateur Yahoo, quinze lignes sans logique.

**Deux causes additionnées sous un seul chiffre.** Le rapport de qualité
annonçait « 7 % de dates non communes » sur le premier jeu de vraies données,
en mélangeant une entrée en Bourse tardive et des jours fériés décalés entre
places. Les remèdes sont opposés : on exclut un titre à l'historique trop
court, on choisit une politique d'alignement pour des fériés. Un rapport qui
additionne deux causes empêche de choisir la bonne.
→ `test_un_historique_tronque_nest_pas_un_desalignement_de_calendrier`

**Deux méthodes du même objet, un seul nom, deux chiffres.** Le diagnostic de
calibration moyennait les volatilités par fenêtre et annonçait 1,02 là où
`calibration()` disait 0,91. La moyenne des écarts-types n'est pas
l'écart-type de l'ensemble — par Jensen, la seconde est toujours la plus
grande. Il fallait agréger les variances.
→ `test_le_diagnostic_saccorde_avec_la_calibration_agregee`

**Quatre prédictions, quatre réfutations, un oracle cassé.** J'avais annoncé
que la pondération exponentielle réparerait la calibration de 0,79. Elle
l'empire. Puis : « c'est ma mesure » — non, les deux concordent ; « c'est
l'optimiseur » — non, vrai sans optimiseur ; « c'est l'horizon » — non, vrai à
tout horizon. Le vrai résultat n'est apparu qu'après avoir réparé un oracle qui
marquait 1,57 là où il devait marquer 1,00, parce qu'il normalisait par la
moyenne *théorique* d'un processus quasi intégré au lieu de sa moyenne
d'échantillon.
→ `test_lavantage_de_lewma_ne_survit_pas_a_la_matrice`

**Un oracle qui regardait le futur sans que ça se voie.** Pour mesurer si
l'intervalle de confiance d'une statistique de biais est honnête, il faut un
modèle correct *par construction*. J'avais pris « la racine de la vraie
variance moyenne du bloc » : l'identité paraît imparable, la moyenne des h
divisée par elle-même valant 1. Mais dans un GARCH, h_{t+1} dépend de e_t — la
moyenne du bloc est donc en partie **causée** par les rendements qu'elle sert à
normaliser, et E[X/Y] < E[X]/E[Y]. L'oracle marquait 0,936 au lieu de 1,000, et
m'aurait fait publier que la statistique de biais est biaisée alors que le
défaut était dans mon harnais. Quatrième oracle cassé du projet.
→ `test_loracle_garch_est_un_vrai_oracle`

**Un test écrit à l'envers, corrigé par la mesure.** J'avais asserté qu'un
modèle ignorant les corrélations sous-estime toujours le risque d'un
portefeuille équipondéré. Faux : le sens suit celui de la corrélation moyenne
hors diagonale, et comme elle est multipliée par N−1 dans un portefeuille, une
corrélation de −0,02 — indistinguable de zéro à l'œil — déplace la variance
d'environ un tiers à N = 15.
→ `test_le_sens_du_biais_dune_diagonale_suit_le_signe_des_correlations`

**Une prose qui contredisait le tableau juste au-dessus d'elle.** La treizième
dérive de narration, trouvée pendant la séance 12 et non après : j'avais écrit
que la covariance empirique s'effondrerait dans le régime encombré, puis la
mesure l'a déclarée VALIDÉE — parce qu'un portefeuille *fixe* n'inverse pas la
matrice, et que l'inversion est ce qui amplifie le bruit. Le défaut était dans
mon protocole de validation, et sa limite était déjà écrite dans la fiche que
j'avais produite sans la lire.
→ `test_les_deux_protocoles_rendent_des_verdicts_opposes_si_mal_conditionnee`

**Trois formules de mémoire, trois réfutations.** La perte de
Fissler-Ziegel est le seul score dont l'espérance est minimisée au vrai
couple (VaR, ES) — c'est la définition de l'élicitabilité conjointe, donc
c'est *testable*. Mes trois premières écritures de cette formule, toutes
tirées de la mémoire, ont échoué ce test : leur minimum tombait au bord
de la grille. Il a fallu la redériver depuis la forme générale de
Fissler & Ziegel (2016) pour obtenir un minimum exactement à la vérité.
Une formule qu'on peut tester ne doit jamais être recopiée de mémoire.
→ `test_la_perte_fz_est_minimisee_a_la_verite`

**Une prédiction trop pessimiste, mesurée et corrigée.** J'avais annoncé
que la dépendance distributionnelle démolirait le backtest d'ES : la loi
nulle d'une statistique d'ES dépend de la forme de la queue, là où
Kupiec s'appuie sur une binomiale qui ne dépend que de α. L'effet est
réel et va dans le mauvais sens — supposer la normalité quand la vérité
est une t(3) fait passer les faux positifs de 5,0 % à 7,4 % — mais il est
modéré, parce que le couple (VaR, ES) fixe déjà le rapport de queue et
retire au problème l'essentiel de sa gravité.
→ `test_la_loi_nulle_depend_de_la_queue_mais_faiblement`

La leçon commune : **un bug numérique ne lève pas d'exception.** Il produit un
nombre plausible, du bon ordre de grandeur, et faux. La seule défense connue
est de vérifier signe et ordre de grandeur contre une prédiction théorique
indépendante.

---

## Sécurité

Le dépôt porte un modèle de menace écrit : qui peut faire entrer quoi, ce qui
est traité, et surtout **ce qui est explicitement hors périmètre**. Un
périmètre non déclaré est un périmètre que le lecteur suppose plus large qu'il
ne l'est.

`tests/test_securite.py` exécute réellement les attaques — traversée de
répertoire, chemin absolu, lien symbolique, injection ANSI dans les journaux,
épuisement mémoire — et chacune **fonctionnait** avant son correctif. Un test
de sécurité qui n'a jamais échoué ne prouve rien.

La menace principale contre un moteur de risque n'est cependant pas dans ce
fichier : c'est de **produire un chiffre faux et crédible**. Les
contre-mesures contre celle-là sont l'architecture entière du projet — les
intervalles de confiance, le refus de répondre, l'audit de causalité, le
Sharpe dégonflé, la calibration. *La sécurité d'un outil de décision, c'est
d'abord qu'il ne mente pas sur ce qu'il sait.*

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
Covariance Matrices* · Laloux, Cizeau, Bouchaud & Potters (1999), *Noise
Dressing of Financial Correlation Matrices* · Chamberlain & Rothschild (1983),
*Arbitrage, Factor Structure, and Mean-Variance Analysis on Large Asset
Markets* · Meucci (2009), *Managing Diversification* · Artzner,
Delbaen, Eber & Heath (1999), *Coherent Measures of Risk* · Kupiec (1995),
*Techniques for Verifying the Accuracy of Risk Measurement Models* · Bailey,
Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial
Charlatanism* · Lo (2002),
*The Statistics of Sharpe Ratios* · J.P. Morgan / Reuters (1996),
*RiskMetrics Technical Document* · Kish (1965), *Survey Sampling* ·
Bailey & López de Prado (2014), *The
Deflated Sharpe Ratio* · DeMiguel, Garlappi & Uppal (2009), *Optimal Versus
Naive Diversification* · Michaud (1989), *The Markowitz Optimization Enigma*

---

*Projet personnel. Aucun élément de ce dépôt ne constitue un conseil en
investissement.*

*Auteur et conditions d'attribution : [AUTEURS.md](AUTEURS.md) · Licence MIT —
la réutilisation est libre, la conservation de la notice d'auteur est exigée.*
