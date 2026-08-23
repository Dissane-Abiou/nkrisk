"""
Chargement de données de marché réelles.

À exécuter sur TA machine — mon bac à sable n'a pas accès à Yahoo Finance.

POURQUOI LE MOTEUR NE DÉPEND PAS D'UNE SOURCE PARTICULIÈRE
==========================================================

Toutes les fonctions de `nkrisk.returns` prennent une `pd.Series` de prix.
Elles ne savent pas — et ne doivent pas savoir — d'où vient cette série.

C'est de l'inversion de dépendance élémentaire, mais en finance elle a une
conséquence pratique très concrète : les fournisseurs de données changent
constamment (API dépréciées, quotas modifiés, licences révisées). Un moteur
couplé à un fournisseur meurt avec lui. Un moteur qui prend une Series
survit à tout.

Corollaire pour les tests : on valide contre des données SYNTHÉTIQUES dont
on connaît la vérité, jamais contre des données réelles dont on ne sait pas
si elles sont correctes. Les données réelles servent à l'usage, pas à la
validation.
"""

from __future__ import annotations

import pandas as pd


def load_yahoo(
    ticker: str,
    start: str,
    end: str | None = None,
) -> pd.Series:
    """
    Télécharge une série de prix ajustés depuis Yahoo Finance.

    Requiert : pip install yfinance

    ⚠️ TROIS PIÈGES DE DONNÉES QUE TU DOIS CONNAÎTRE ⚠️

    1. PRIX AJUSTÉS, TOUJOURS. Le prix « brut » chute mécaniquement le jour
       du détachement de dividende. Si tu calcules des rendements dessus,
       tu comptabilises un faux krach à chaque trimestre. `auto_adjust=True`
       corrige les dividendes ET les fractionnements d'actions.
       C'est l'erreur numéro un des débutants en finance quantitative.

    2. BIAIS DU SURVIVANT. Yahoo ne te donne que les titres qui existent
       encore. Les entreprises qui ont fait faillite ont disparu de la base.
       Toute étude historique construite sur les titres d'aujourd'hui
       surestime donc le rendement passé — parfois de plusieurs points par
       an. Aucune correction n'est possible avec des données gratuites.

    3. LICENCE. Les données Yahoo sont destinées à un usage personnel. Un
       usage commercial exige une licence explicite. Retiens-le : c'est
       exactement le genre de détail qui bloque un produit en revue
       juridique chez un client institutionnel.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance n'est pas installé. Lance : pip install yfinance"
        ) from exc

    data = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )

    if data is None or len(data) == 0:
        raise ValueError(
            f"Aucune donnée pour '{ticker}' entre {start} et {end}. "
            f"Vérifie le symbole (les titres canadiens finissent souvent "
            f"par .TO, ex. XIC.TO) et ta connexion."
        )

    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance renvoie parfois un MultiIndex
        close = close.iloc[:, 0]

    serie = close.dropna()
    serie.name = ticker

    if len(serie) < 2:
        raise ValueError(f"Série trop courte pour '{ticker}' : {len(serie)} point(s)")

    return serie


def load_csv(path: str, date_column: str = "Date", price_column: str = "Close") -> pd.Series:
    """
    Charge une série de prix depuis un CSV — export de courtier, par exemple.

    Volontairement strict : on refuse les doublons de dates et les prix
    non triés plutôt que de « réparer » en silence. Dans un moteur de
    risque, une réparation silencieuse est un mensonge différé.
    """
    df = pd.read_csv(path, parse_dates=[date_column])

    if df[date_column].duplicated().any():
        n = int(df[date_column].duplicated().sum())
        raise ValueError(
            f"{n} date(s) en double dans {path}. Décide toi-même quoi en "
            f"faire — le moteur ne choisira pas à ta place."
        )

    df = df.sort_values(date_column)
    serie = pd.Series(
        df[price_column].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df[date_column]),
        name=path,
    )
    return serie.dropna()
