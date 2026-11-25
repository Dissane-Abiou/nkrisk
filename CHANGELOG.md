# Journal des versions

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).
Versionnement : [SemVer](https://semver.org/lang/fr/), avec la réserve
du `0.x` expliquée dans [la politique de stabilité](#politique-de-stabilité).

---

## [0.2.0] — 2026-11-25

Séance 15 : durcissement d'ingénierie. Aucune mesure nouvelle ; ce sont
les propriétés du paquet en tant que **bibliothèque** qui changent.

### ⚠️ Ruptures

- **`CovarianceEstimate`, `FactorModel`, `TailForecast` et
  `EmpiricalNull` ne sont plus hachables.** Ces objets portent un
  tableau numpy ; leur `__eq__` généré par `dataclass` comparait les
  champs un à un et **levait `ValueError`** au lieu de rendre vrai ou
  faux. Ils ont désormais une égalité de valeur correcte, fondée sur
  `np.array_equal` — et définir `__eq__` met `__hash__` à `None`.

  *Migration* : si vous les mettiez dans un `set` ou en clé de `dict`,
  utilisez `id()` ou une clé dérivée. Aucun code du dépôt ne le
  faisait ; c'est pourquoi le défaut est resté invisible depuis la
  séance 2.

### Ajouté

- **`py.typed`** (PEP 561). Sans ce marqueur, les quelque dix mille
  lignes d'annotations du paquet étaient **invisibles** pour quiconque
  l'installait : mypy et les éditeurs traitaient `nkrisk` comme du code
  non typé.
- `nkrisk.__version__`, et le présent journal.
- `tests/conftest.py` : générateurs d'univers et de trajectoires GARCH
  partagés. Ils étaient recopiés dans cinq fichiers chacun, avec des
  divergences déjà installées.
- `nkrisk.rendering` : la couche de présentation du rapport, extraite
  du domaine.
- `nkrisk.selection` : le Sharpe dégonflé et la longueur minimale de
  piste, extraits de `backtest`. Le rejeu est une question de
  simulation, le biais de sélection une question d'inférence ; elles
  partageaient un fichier de 1 152 lignes pour la seule raison
  d'être arrivées à la même séance. **Les anciens chemins d'import
  continuent de fonctionner** — casser un import pour une raison de
  rangement serait un mauvais échange.
- Matrice d'intégration continue sur trois systèmes d'exploitation, et
  **seuil de couverture bloquant** à 85 %.

### Modifié

- **`mypy` passe en `strict`.** L'ancienne configuration se limitait à
  deux options, donc « mypy sans erreur » voulait dire beaucoup moins
  que ce que le README laissait croire. Le passage a demandé quatorze
  corrections, toutes réelles : dix retours `Any` non déclarés et
  quatre génériques sans paramètre.
- `report.py` passe de 1 020 à 862 lignes ; le CSS et la génération
  HTML vivent désormais dans `rendering.py`.
- `backtest.py` passe de 1 152 à 953 lignes.

### Corrigé

- **La version est redevenue un littéral**, et `pyproject.toml` la lit
  (`dynamic = ["version"]`). La première correction de cette séance
  remplaçait la constante dupliquée par une lecture de
  `importlib.metadata`, ce qui introduisait un mode de panne que la
  duplication n'avait pas : la fonction cherche un `.dist-info` sur
  `sys.path`, et `sys.path` diffère entre `pytest` et `python` nu. Sur
  une machine portant d'anciennes métadonnées d'installation éditable,
  `python -c` annonçait 0.1.0 pendant que la suite voyait 0.2.0 — **et
  le test de cohérence passait au vert**. Remplacer une duplication par
  un mécanisme n'est pas gratuit.
- **Deux retours `Any` que l'analyse ne voyait pas ici.** `mypy --strict`
  passait avec numpy 2.5.3 et trouvait deux erreurs avec numpy 2.4.4 :
  les **stubs de numpy changent d'une version à l'autre**, donc
  « mypy strict sans erreur » n'est pas une propriété du code seul mais
  du code ET de ses dépendances. Trouvé en livrant, sur une autre
  machine. `mypy` tourne désormais sur toute la matrice de CI, et plus
  seulement sur une version.
- **Performance : un rapport complet passe de 3,8 s à 2,0 s.**
  `EmpiricalNull.draw` recalculait un `np.quantile` sur l'intégralité
  du tableau de résidus à chaque tirage — 24 021 fois pour un seul
  rapport, soit 2,67 secondes sur 5,24 à recalculer un invariant. Le
  résultat est maintenant mis en cache par niveau de queue.

---

## [0.1.0] — 2026-08-23 → 2026-11-24

Les quatorze premières séances. Le détail séance par séance est dans
[`AVANCEMENT.md`](AVANCEMENT.md) ; en voici la trame.

### Ajouté

- `types` — `Estimate` : aucune fonction ne retourne un nombre nu.
- `returns`, `covariance` (Ledoit-Wolf), `ewma`, `factors`
  (Marchenko-Pastur), `risk` (Euler), `var` (VaR/CVaR, Kupiec),
  `drawdown`, `portfolio`, `stress`.
- `backtest` — rejeu point-in-time, audit de causalité, Sharpe dégonflé.
- `data.market`, `data.quality`, `data.synthetic`.
- `validation` — statistique de biais, intervalle par blocs, inventaire
  de modèles.
- `shortfall` — backtest d'Expected Shortfall (Acerbi-Székely), perte
  de Fissler-Ziegel.
- `report` — réconciliation des instruments en un verdict unique.
- `SECURITE.md` (modèle de menace, CWE-22 corrigée), `AUTEURS.md`,
  `AVANCEMENT.md`.

---

## Politique de stabilité

**Tant que la version majeure est `0`, aucune API n'est garantie.**
C'est la seule promesse honnête pour un projet qui a changé de forme à
chaque séance — la séance 15 vient encore de casser le hachage de quatre
types.

Ce que le projet s'engage malgré tout à tenir, dès maintenant :

1. **Toute rupture est écrite ici**, avec sa note de migration. Une
   rupture non documentée est un bogue, pas un changement.
2. **Les ruptures n'arrivent que sur une version mineure** (`0.x.0`),
   jamais sur un correctif (`0.x.y`).
3. **Ce qui est importable depuis `nkrisk` directement est l'API
   publique.** Ce qui exige d'importer un sous-module
   (`nkrisk.report.SEVERITE`) est un détail d'implémentation, et peut
   changer sans préavis.
4. Les noms préfixés d'un `_` ne sont jamais publics, où qu'ils soient.

À la `1.0`, la règle deviendra celle de SemVer sans réserve. Ce qui
manque pour y arriver est listé dans la section « ce qui ne va pas » de
`AVANCEMENT.md` — en tête, l'absence de toute revue de code externe.
