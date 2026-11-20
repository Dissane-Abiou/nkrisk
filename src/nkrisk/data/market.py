"""
Accès aux données de marché — et le point du projet où l'on cesse de
contrôler ce qu'on reçoit.

POURQUOI CE MODULE A ÉTÉ RÉÉCRIT
================================

Pendant huit séances, il est resté à **0 % de couverture de tests**.
C'était le seul trou du dépôt, et il était honnêtement documenté comme
tel : la fonction appelait le réseau, donc elle n'était pas testable
sans réseau.

Ce raisonnement était faux, et c'est une erreur de conception, pas de
test. Une fonction n'est pas « intestable parce qu'elle appelle le
réseau » : elle est intestable parce qu'elle DÉCIDE elle-même d'appeler
le réseau. La différence est entière.

    avant   load_yahoo(ticker)  →  appelle yfinance en dur
    après   source.fetch(ticker) →  la source est un paramètre

Dans la seconde version, un test passe une source en mémoire et vérifie
toute la logique — validation, alignement, détection d'anomalies — sans
une seule connexion. Le code qui parle réellement à Yahoo se réduit à
une quinzaine de lignes sans aucune logique, et c'est la seule partie
qui reste non couverte.

**« Ce n'est pas testable » veut presque toujours dire « ce n'est pas
encore bien découpé ».** C'est l'inversion de dépendance, et c'est la
seule raison pour laquelle on s'en sert : rendre vérifiable ce qui ne
l'était pas.

CE QUE LES DONNÉES RÉELLES APPORTENT QUE LE SYNTHÉTIQUE N'A JAMAIS EU
=====================================================================

Jusqu'ici, tout le moteur a été validé contre des données dont on
connaissait la vérité. C'est une décision méthodologique assumée depuis
la séance 1, et elle ne change pas : on ne valide pas un estimateur
contre des données dont on ignore la bonne réponse.

Mais les données réelles arrivent avec des pathologies que le
synthétique ne produit jamais :

- des prix qui ne bougent pas pendant six jours parce que le titre ne
  s'échange pas ;
- des calendriers qui ne coïncident pas — la Bourse de Toronto ferme le
  premier lundi d'août, pas celle de New York ;
- des sauts de 40 % qui sont soit un krach, soit un fractionnement
  d'actions mal corrigé, et rien dans la série ne permet de trancher ;
- des titres qui disparaissent, et dont l'absence même est une
  information qu'on ne reçoit pas.

Aucune de ces pathologies n'est une erreur de calcul. Toutes faussent
les estimations en silence. Le module `quality` les détecte ; celui-ci
se contente de les faire entrer proprement.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

# ══════════════════════════════════════════════════════════════════════
#  VALIDATION DES SYMBOLES — PREMIÈRE LIGNE DE DÉFENSE
# ══════════════════════════════════════════════════════════════════════

# Un symbole boursier : lettres, chiffres, et les quelques séparateurs
# réellement utilisés (RY.TO, BRK-B, ^GSPC). Rien d'autre.
#
# Ce motif interdit en particulier `/`, `\`, `..`, les octets nuls et
# les séquences d'échappement ANSI. Chacun de ces caractères ouvre une
# porte différente, et aucun n'a jamais servi à nommer un titre.
_SYMBOLE = re.compile(r"^\^?[A-Za-z0-9][A-Za-z0-9._=-]{0,23}$")

TAILLE_MAX_CSV = 64 * 1024 * 1024      # 64 Mio
LIGNES_MAX_CSV = 2_000_000


def _lisible(octets: int) -> str:
    """
    Formate une taille dans l'unité qui la rend lisible.

    Un message « fichier de 0 Mo, limite 0 Mo » est pire qu'inutile :
    il donne l'air d'un bogue alors que la limite a correctement joué.
    Un diagnostic illisible est un diagnostic qu'on ignore.
    """
    for seuil, unite in ((1e9, "Go"), (1e6, "Mo"), (1e3, "ko")):
        if octets >= seuil:
            return f"{octets / seuil:.1f} {unite}"
    return f"{octets} octets"


def valider_symbole(ticker: str) -> str:
    """
    Rejette tout ce qui ne ressemble pas à un symbole boursier.

    POURQUOI UNE LISTE BLANCHE ET NON UNE LISTE NOIRE
    --------------------------------------------------
    On pourrait interdire `..`, puis `/`, puis `\\`, puis `%2e%2e`, puis
    les formes Unicode équivalentes… Chaque interdiction en oublie une,
    et l'histoire des traversées de répertoire est l'histoire de ces
    oublis.

    Une liste blanche inverse la charge : tout est refusé sauf ce qu'on
    a explicitement décrit. L'oubli devient un faux refus — visible,
    signalé, corrigeable — au lieu d'un faux accès, silencieux.

    CE QUE CETTE FONCTION NE SUFFIT PAS À GARANTIR
    -----------------------------------------------
    Elle ne protège pas d'un lien symbolique placé DANS le dossier
    autorisé et pointant ailleurs : le nom `AAPL` est parfaitement
    valide, et c'est le système de fichiers qui redirige. D'où le
    confinement du chemin résolu dans `CsvSource`, qui attrape ce cas
    précis.

    Valider l'entrée et confiner le résultat ne font donc pas double
    emploi : chacun couvre ce que l'autre laisse passer.
    """
    if not isinstance(ticker, str):
        raise TypeError(f"symbole non textuel : {type(ticker).__name__}")
    if not _SYMBOLE.match(ticker):
        # On ne renvoie PAS la valeur reçue dans le message : un symbole
        # contenant des séquences d'échappement ANSI peut réécrire le
        # terminal de celui qui lit le journal, et un message d'erreur
        # est précisément ce qui finit dans un journal.
        raise ValueError(
            f"symbole invalide ({len(ticker)} caractères). Attendu : "
            f"lettres, chiffres et . _ - = , éventuellement précédés de "
            f"^, au plus 24 caractères."
        )
    return ticker

# ══════════════════════════════════════════════════════════════════════
#  LE CONTRAT D'UNE SOURCE
# ══════════════════════════════════════════════════════════════════════


@runtime_checkable
class PriceSource(Protocol):
    """
    Tout ce qui sait rendre une série de prix pour un symbole.

    Troisième protocole du projet, après `Shock` (séance 5) et
    `Strategy` (séance 8). Même raison à chaque fois : permettre
    d'étendre ou de remplacer sans hériter.

    Ici, le bénéfice est particulier. Les fournisseurs de données
    changent constamment — API dépréciées, quotas modifiés, licences
    révisées. Un moteur couplé à un fournisseur meurt avec lui. Un
    moteur qui prend une source survit à tout, et se teste sans réseau.
    """

    def fetch(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.Series: ...


@dataclass(frozen=True)
class InMemorySource:
    """
    Source alimentée par un dictionnaire. Pour les tests et les
    démonstrations reproductibles.

    Ce n'est pas un bouchon « juste pour les tests » : c'est la source
    qu'on utilise pour rejouer exactement un résultat six mois plus
    tard, quand le fournisseur aura changé ses données sous les pieds.
    """

    series: Mapping[str, pd.Series]
    name: str = "mémoire"

    def fetch(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.Series:
        # Pas de validation ici : cette source ne touche ni au système de
        # fichiers ni au réseau, et ses clés sont fournies par le code
        # appelant, pas par un tiers. Valider serait du rite.
        if ticker not in self.series:
            raise KeyError(
                f"symbole inconnu dans cette source : '{ticker}'. "
                f"Disponibles : {sorted(self.series)[:10]}"
            )
        s = self.series[ticker]
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        return s.rename(ticker)


@dataclass(frozen=True)
class CsvSource:
    """
    Un dossier de fichiers CSV, un par symbole : `<dossier>/<ticker>.csv`.

    Volontairement strict : on refuse les doublons de dates plutôt que
    de « réparer » en silence. Dans un moteur de risque, une réparation
    silencieuse est un mensonge différé — l'utilisateur ne saura jamais
    que la série qu'il analyse n'est pas celle qu'il croit.
    """

    folder: str | Path
    date_column: str = "Date"
    price_column: str = "Close"
    max_bytes: int = TAILLE_MAX_CSV
    max_rows: int = LIGNES_MAX_CSV
    name: str = "csv"

    def fetch(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.Series:
        valider_symbole(ticker)

        # ⚠️ CWE-22 — TRAVERSÉE DE RÉPERTOIRE. CE CODE ÉTAIT VULNÉRABLE.
        #
        # Version livrée à la séance 9 :
        #
        #     chemin = Path(self.folder) / f"{ticker}.csv"
        #
        # Deux évasions, toutes deux vérifiées :
        #
        #   fetch("../prive/secrets")       lit hors du dossier ;
        #   fetch("/etc/passwd")            encore pire — l'opérateur `/`
        #                                   de pathlib ABANDONNE le membre
        #                                   de gauche quand celui de droite
        #                                   est absolu. Le dossier autorisé
        #                                   disparaît sans un mot.
        #
        # Le second est le plus traître : rien dans la lecture du code ne
        # suggère que `a / b` puisse ne pas commencer par `a`.
        racine = Path(self.folder).resolve()
        chemin = (racine / f"{ticker}.csv").resolve()
        if not chemin.is_relative_to(racine):
            raise ValueError(
                f"chemin hors du dossier autorisé. Le symbole demandé "
                f"résout en dehors de {racine}."
            )

        if not chemin.exists():
            raise FileNotFoundError(f"fichier absent : {chemin}")

        # Déni de service par épuisement mémoire : `read_csv` charge tout.
        # Un fichier de plusieurs gigaoctets, volontaire ou non, fait
        # tomber le processus. Une borne vaut mieux qu'un plantage.
        taille = chemin.stat().st_size
        if taille > self.max_bytes:
            raise ValueError(
                f"fichier de {_lisible(taille)}, limite "
                f"{_lisible(self.max_bytes)}. Découpe-le ou relève "
                f"`max_bytes` en connaissance de cause."
            )

        df = pd.read_csv(
            chemin, parse_dates=[self.date_column], nrows=self.max_rows
        )
        if len(df) >= self.max_rows:
            raise ValueError(
                f"plus de {self.max_rows:,} lignes : lecture interrompue "
                f"plutôt que tronquée en silence."
            )
        if df[self.date_column].duplicated().any():
            n = int(df[self.date_column].duplicated().sum())
            raise ValueError(
                f"{n} date(s) en double dans {chemin}. Décide toi-même quoi "
                f"en faire — le moteur ne choisira pas à ta place."
            )

        df = df.sort_values(self.date_column)
        s = pd.Series(
            df[self.price_column].to_numpy(dtype=float),
            index=pd.DatetimeIndex(df[self.date_column]),
            name=ticker,
        ).dropna()

        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        return s


@dataclass(frozen=True)
class YahooSource:
    """
    Yahoo Finance, via yfinance. La seule partie du module qui touche
    le réseau, et la seule qui reste non couverte par les tests.

    Elle ne contient AUCUNE logique : elle appelle, vérifie que ce n'est
    pas vide, et rend une Series. Tout ce qui peut se tromper est
    ailleurs, donc testé.

    ⚠️ TROIS PIÈGES DE DONNÉES À CONNAÎTRE ⚠️

    1. PRIX AJUSTÉS, TOUJOURS. Le prix brut chute mécaniquement le jour
       du détachement de dividende. Calculer des rendements dessus
       comptabilise un faux krach à chaque trimestre. `auto_adjust=True`
       corrige dividendes ET fractionnements. C'est l'erreur numéro un
       des débutants en finance quantitative.

    2. BIAIS DU SURVIVANT. Yahoo ne donne que les titres qui existent
       encore. Les faillites ont disparu de la base. Toute étude
       historique bâtie sur les titres d'aujourd'hui surestime donc le
       rendement passé — parfois de plusieurs points par an. Aucune
       correction n'est possible avec des données gratuites, et c'est
       une limite qu'il faut déclarer, pas contourner.

    3. LICENCE. Usage personnel. Un usage commercial exige une licence
       explicite. C'est exactement le genre de détail qui bloque un
       produit en revue juridique chez un client institutionnel.
    """

    auto_adjust: bool = True
    name: str = "yahoo"

    def fetch(
        self, ticker: str, start: str | None = None, end: str | None = None
    ) -> pd.Series:
        # Le symbole part dans une URL construite par yfinance. On ne
        # sait pas comment il l'échappe, et on n'a pas à le savoir : ce
        # qu'on contrôle, c'est ce qu'on lui donne.
        valider_symbole(ticker)
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dépend de l'install
            raise ImportError(
                "yfinance n'est pas installé. Lance : pip install yfinance"
            ) from exc

        data = yf.download(
            ticker, start=start, end=end, progress=False,
            auto_adjust=self.auto_adjust,
        )
        if data is None or len(data) == 0:  # pragma: no cover - dépend du réseau
            raise ValueError(
                f"aucune donnée pour '{ticker}' entre {start} et {end}. "
                f"Vérifie le symbole (les titres canadiens finissent souvent "
                f"par .TO, ex. XIC.TO) et ta connexion."
            )

        close = data["Close"]
        if isinstance(close, pd.DataFrame):  # MultiIndex selon les versions
            close = close.iloc[:, 0]
        return close.dropna().rename(ticker)


# ══════════════════════════════════════════════════════════════════════
#  CHARGEMENT
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LoadResult:
    """Ce qu'un chargement a rendu, et ce qu'il n'a pas pu rendre."""

    series: dict[str, pd.Series]
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self.series)

    def __str__(self) -> str:
        lignes = [f"{len(self.series)} série(s) chargée(s)"]
        for t, s in self.series.items():
            lignes.append(
                f"  {t:<12} {len(s):>6} points  "
                f"{s.index.min().date()} → {s.index.max().date()}"
            )
        for t, raison in self.failures.items():
            lignes.append(f"  {t:<12} ÉCHEC : {raison}")
        return "\n".join(lignes)


def load_prices(
    tickers: Iterable[str],
    source: PriceSource,
    start: str | None = None,
    end: str | None = None,
    min_points: int = 2,
    strict: bool = True,
) -> LoadResult:
    """
    Charge plusieurs séries depuis une source, et dit ce qui a échoué.

    `strict=True` (défaut) : la première erreur arrête tout.
    `strict=False` : on continue et on consigne les échecs dans le
    résultat.

    POURQUOI LES DEUX MODES, ET POURQUOI LE STRICT EST LE DÉFAUT
    ------------------------------------------------------------
    Un univers amputé de trois titres sur cinquante n'est plus le même
    univers. Si ces trois-là ont disparu parce qu'ils ont fait faillite,
    continuer sans eux revient à construire soi-même un biais du
    survivant — exactement celui qu'on reproche aux données gratuites.

    Le mode souple existe parce qu'en exploration on veut avancer. Il
    doit rester un choix explicite, jamais un comportement par défaut :
    le défaut d'un moteur de risque est de s'arrêter quand il ne sait
    pas.
    """
    series: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}

    for ticker in tickers:
        try:
            s = source.fetch(ticker, start, end)
            if not isinstance(s, pd.Series):
                raise TypeError(
                    f"la source a rendu un {type(s).__name__} pour '{ticker}', "
                    f"attendu une pd.Series"
                )
            s = s.dropna()
            if not isinstance(s.index, pd.DatetimeIndex):
                raise TypeError(
                    f"la série de '{ticker}' n'est pas indexée par des dates"
                )
            if len(s) < min_points:
                raise ValueError(
                    f"série trop courte pour '{ticker}' : {len(s)} point(s), "
                    f"{min_points} requis"
                )
            series[ticker] = s.sort_index().rename(ticker)
        except Exception as exc:
            if strict:
                raise
            failures[ticker] = f"{type(exc).__name__} : {exc}"

    return LoadResult(series=series, failures=failures)
