# Modèle de menace de nkrisk

*Dernière révision : séance 10. Ce document décrit ce que ce projet
protège, ce qu'il ne protège pas, et pourquoi.*

---

## Pourquoi ce document existe

Un moteur de risque est, par nature, un programme auquel on donne des
données qu'on n'a pas écrites, et dont on croit les sorties. Les deux
font de lui une cible intéressante.

Ce document n'est pas une déclaration de conformité. C'est un
raisonnement : qui peut faire entrer quoi, ce qui arrive alors, et ce
qu'on a décidé de ne pas traiter. **Un périmètre non déclaré est un
périmètre que le lecteur suppose plus large qu'il ne l'est**, et c'est
ainsi qu'on finit par s'appuyer sur une protection qui n'existe pas.

---

## Le modèle de menace

### Qui alimente ce moteur

| Entrée | Origine | Confiance |
|---|---|---|
| Symboles boursiers | configuration, parfois saisie utilisateur | **aucune** |
| Fichiers CSV | export de courtier, dépôt partagé | **aucune** |
| Réponses réseau | Yahoo Finance, via yfinance | **faible** |
| `pd.Series` passées en mémoire | code appelant | élevée |
| Paramètres numériques | code appelant | élevée |

La ligne qui compte est la première. Un symbole a l'air inoffensif —
quatre lettres majuscules — et c'est précisément ce qui le rend
dangereux : personne ne pense à le valider.

### Ce qu'un attaquant chercherait à obtenir

1. **Lire un fichier hors du périmètre.** Le moteur ouvre des fichiers
   dont le nom vient de l'extérieur. C'est la surface la plus directe,
   et c'est celle où une vulnérabilité réelle a été trouvée.
2. **Exécuter du code.** Par désérialisation, évaluation dynamique, ou
   une dépendance compromise.
3. **Faire tomber le processus.** Un fichier gigantesque suffit.
4. **Fausser une décision sans être détecté.** Le plus intéressant
   contre un moteur de risque, et le moins traité en général : il n'y a
   pas d'alarme quand un chiffre est simplement faux.
5. **Réécrire ce que lit l'opérateur.** Journaux, terminal, rapport.

---

## Vulnérabilité trouvée et corrigée

### CWE-22 — Traversée de répertoire dans `CsvSource` *(séance 9 → 10)*

**Code vulnérable, tel que livré :**

```python
chemin = Path(self.folder) / f"{ticker}.csv"
```

**Deux évasions, toutes deux vérifiées avant correction :**

```python
source.fetch("../prive/secrets")      # lit hors du dossier
source.fetch("/etc/passwd")           # pire — voir ci-dessous
```

Le second cas est le plus traître, et il vient de la bibliothèque
standard. L'opérateur `/` de `pathlib` **abandonne son membre de gauche**
quand celui de droite est absolu :

```python
Path("/data") / "/etc/passwd"   ==   Path("/etc/passwd")
```

Le dossier autorisé disparaît sans exception et sans avertissement.
C'est documenté, c'est cohérent avec la sémantique POSIX, et rien dans
la lecture de `a / b` ne suggère que le résultat puisse ne pas commencer
par `a`.

**Correction, en deux couches qui ne font pas double emploi :**

1. **Liste blanche sur le symbole** — `^\^?[A-Za-z0-9][A-Za-z0-9._=-]{0,23}$`.
   Une liste blanche, et non une liste noire : on pourrait interdire
   `..`, puis `/`, puis `%2e%2e`, puis les équivalents Unicode — chaque
   interdiction en oublie une, et l'histoire des traversées de
   répertoire est l'histoire de ces oublis. Une liste blanche transforme
   l'oubli en faux refus, visible et corrigeable, plutôt qu'en faux
   accès, silencieux.

2. **Confinement du chemin résolu** — `chemin.resolve()` doit rester
   sous `racine.resolve()`.

**Pourquoi les deux.** Un lien symbolique nommé `PIEGE.csv`, placé dans
le dossier autorisé et pointant ailleurs, porte un nom parfaitement
valide : la validation le laisse passer, et elle a raison — c'est le
système de fichiers qui redirige. Seul le confinement l'attrape.
Inversement, la validation bloque avant tout accès disque et protège
aussi l'adaptateur réseau, que le confinement ne concerne pas.

Chaque couche couvre exactement ce que l'autre laisse passer, et
`tests/test_securite.py` contient un test pour chacun de ces deux cas.

---

## Surfaces traitées

### Épuisement de ressources

`pd.read_csv` charge tout en mémoire. Un fichier de plusieurs
gigaoctets — malveillant, ou simplement mal exporté — fait tomber le
processus.

`CsvSource` borne la **taille du fichier** (64 Mio par défaut, vérifiée
avant d'en lire un octet) et le **nombre de lignes** (2 millions). Un
dépassement de lignes lève une erreur au lieu de tronquer : tronquer en
silence ferait croire à l'appelant qu'il analyse la série entière.

### Injection dans les journaux (CWE-117)

Un message d'erreur finit dans un journal, et un journal se lit dans un
terminal. Un symbole contenant des séquences d'échappement ANSI peut
effacer l'écran ou réécrire les lignes précédentes pour masquer une
attaque.

Les messages de refus donnent la **longueur** de l'entrée et la règle
attendue, **jamais la valeur reçue**. Le refus reste diagnosticable sans
rendre l'attaquant maître de ce qui s'affiche.

### Désérialisation et évaluation dynamique

Le paquet n'utilise ni `pickle`, ni `marshal`, ni `eval`, ni `exec`, ni
`os.system`, ni `subprocess`, ni `yaml.load`.

`pickle` est la vulnérabilité classique des bibliothèques scientifiques
Python, parce qu'il est commode et que son format a l'air d'un format de
données. Ce n'en est pas un : **c'est un langage de programmation
sérialisé**, et le lire exécute ce qu'il contient.

Un commentaire disant « on n'utilise pas pickle » vieillit mal. Un test
qui parcourt les sources et échoue le jour où quelqu'un l'ajoute,
non — c'est `test_aucune_deserialisation_ni_evaluation_dans_le_paquet`.

### Chaîne d'approvisionnement

Les dépendances d'exécution sont peu nombreuses et très utilisées :
numpy, pandas, scipy. L'extra `dev` ajoute pytest, ruff, mypy et
scikit-learn. `yfinance` n'est **pas** une dépendance déclarée : il est
importé à l'intérieur de l'adaptateur Yahoo, et son absence n'empêche
rien d'autre de fonctionner.

Ce choix a une raison de sécurité. `pip install yfinance` tire une
dizaine de paquets transitifs — `curl_cffi`, `protobuf`, `peewee`,
`lxml`, `websockets` — dont aucun n'est nécessaire pour calculer une
volatilité. Les maintenir hors du chemin d'installation par défaut
réduit la surface pour tous ceux qui n'utilisent pas Yahoo.

L'intégration continue exécute `pip-audit`, qui confronte l'arbre
installé aux avis de vulnérabilité publiés.

---

## Hors périmètre, explicitement

Ces points ne sont **pas** traités, et les énumérer fait partie du
travail :

- **Authentification et autorisation.** Il n'y en a aucune. nkrisk est
  une bibliothèque, pas un service. Si tu l'exposes derrière une API,
  l'authentification est entièrement à ta charge.
- **Secrets.** Le projet n'en lit, n'en stocke et n'en journalise aucun.
  Si tu ajoutes un fournisseur de données payant, sa clé est ton
  problème — et elle n'a rien à faire dans le code.
- **Chiffrement au repos ou en transit.** `yfinance` utilise HTTPS ;
  rien d'autre n'est chiffré, et rien n'est persisté.
- **Multilocataire.** Aucune isolation entre appelants. Tout le code
  s'exécute avec les droits du processus.
- **Intégrité des données de marché.** Le moteur ne peut pas savoir si
  Yahoo lui ment. Il détecte des anomalies *statistiques* — prix figés,
  sauts invraisemblables — mais un prix faux et plausible passera. Cette
  limite est structurelle, pas une lacune d'implémentation.
- **Exécution d'ordres.** Il n'y en a pas. Le moteur ne se connecte à
  aucun courtier, ne passe aucun ordre, et ne détient aucun
  identifiant.

---

## Une menace propre aux moteurs de risque

Les sections précédentes couvrent des catégories classiques. Il y en a
une qui ne l'est pas, et qui est la plus sérieuse ici.

**Un moteur de risque échoue surtout en produisant un chiffre faux et
crédible.** Aucune des protections ci-dessus n'empêche cela, parce que
ce n'est pas une faille de sécurité au sens habituel : il n'y a ni
intrusion, ni exécution, ni exfiltration. Il y a une décision prise sur
une base fausse.

Les contre-mesures du projet contre ce mode de défaillance ne sont pas
dans ce document, elles sont dans son architecture :

- toute estimation transporte son intervalle de confiance, et refuse de
  répondre sous un seuil de fiabilité ;
- les hypothèses de validité voyagent avec le résultat ;
- le rapport de qualité des données signale sans réparer, parce que
  réparer en silence est un mensonge différé ;
- l'audit de causalité vérifie expérimentalement qu'aucun backtest ne
  lit le futur ;
- le Sharpe dégonflé corrige du nombre d'essais ;
- et la calibration compare systématiquement le risque annoncé au
  risque subi.

**La sécurité d'un outil de décision, c'est d'abord qu'il ne mente pas
sur ce qu'il sait.** Le reste est nécessaire, et vient après.

---

## Signaler une vulnérabilité

Ouvrir une *issue* sur https://github.com/Dissane-Abiou/nkrisk en
décrivant le chemin d'attaque et, si possible, un cas reproductible.
Projet personnel, sans engagement de délai.

---

## Épingler les actions GitHub par empreinte

Le fichier `.github/workflows/ci.yml` référence encore les actions par
étiquette (`actions/checkout@v4`). Une étiquette est **mutable** : celui
qui contrôle le dépôt de l'action peut la déplacer vers un autre commit,
et ce commit s'exécutera avec le jeton de ton dépôt. C'est le vecteur
qui a servi à plusieurs compromissions de chaînes d'intégration
continue.

La correction consiste à référencer l'empreinte du commit, qui est
immuable :

```bash
# Récupérer l'empreinte correspondant à l'étiquette actuelle
gh api repos/actions/checkout/git/ref/tags/v4 --jq .object.sha
gh api repos/actions/setup-python/git/ref/tags/v5 --jq .object.sha
```

puis remplacer dans le workflow :

```yaml
- uses: actions/checkout@<empreinte>          # v4.x.y
```

Le commentaire garde l'étiquette lisible ; c'est l'empreinte qui fait
foi. Dependabot sait mettre à jour ces épingles automatiquement.

**Ces empreintes ne sont pas écrites ici parce qu'elles n'ont pas pu
être vérifiées depuis l'environnement où ce document a été rédigé.**
Écrire de mémoire une empreinte de commit dans une chaîne d'intégration
continue — c'est-à-dire exactement la valeur dont dépend la sécurité du
mécanisme — serait contraire à tout ce que ce projet défend par
ailleurs. Les deux commandes ci-dessus donnent la bonne valeur en trois
secondes.
