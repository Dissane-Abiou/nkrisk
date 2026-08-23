"""
Types fondamentaux du moteur.

LA DÉCISION D'ARCHITECTURE LA PLUS IMPORTANTE DE TOUT LE PROJET
===============================================================

Un moteur de risque ne calcule JAMAIS une vérité. Il produit une *estimation*
tirée d'un échantillon fini du passé. Deux échantillons différents du même
marché donnent deux réponses différentes.

La plupart des outils grand public affichent « volatilité : 18,3 % » et
s'arrêtent là. C'est un mensonge par omission : l'utilisateur croit tenir un
fait, alors qu'il tient un tirage.

Ici, on rend l'incertitude IMPOSSIBLE À IGNORER en la mettant dans le type
lui-même. Aucune fonction de ce moteur ne retourne un `float` nu. Tout retourne
un `Estimate`, qui transporte sa propre marge d'erreur.

C'est une contrainte qu'on s'impose au niveau du système de types, exactement
comme on utiliserait un type `Money` plutôt qu'un `float` pour éviter les
erreurs de devise. Le compilateur (ou ici, le lecteur) ne peut plus oublier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    """
    Une grandeur estimée, avec son incertitude.

    Attributs
    ---------
    value : float
        L'estimation ponctuelle (« point estimate »). Le chiffre que les
        autres logiciels afficheraient tout seul.
    ci_low, ci_high : float
        Bornes de l'intervalle de confiance. Interprétation correcte : si on
        répétait l'expérience un grand nombre de fois, `confidence` % des
        intervalles construits ainsi contiendraient la vraie valeur.
        ATTENTION : ce n'est PAS « il y a 95 % de chances que la vraie valeur
        soit dans cet intervalle » — cette phrase est une interprétation
        bayésienne et elle est fausse dans le cadre fréquentiste. Nuance
        pédante en apparence, mais elle t'évitera de dire des bêtises en
        entrevue.
    confidence : float
        Niveau de confiance, dans (0, 1). Typiquement 0.95.
    n_obs : int
        Nombre d'observations ayant servi à l'estimation. C'est le levier
        principal sur la largeur de l'intervalle.
    method : str
        Comment l'intervalle a été construit. Indispensable : un intervalle
        de confiance n'a de sens que relativement à ses hypothèses.
    assumptions : tuple[str, ...]
        Les hypothèses sous lesquelles l'intervalle est valide. Si elles sont
        fausses, l'intervalle est faux — et il vaut mieux que ce soit écrit.
    """

    value: float
    ci_low: float
    ci_high: float
    confidence: float
    n_obs: int
    method: str
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # On valide à la construction. Un Estimate incohérent ne doit jamais
        # exister — c'est le principe « make illegal states unrepresentable ».
        if not (0.0 < self.confidence < 1.0):
            raise ValueError(f"confidence doit être dans (0,1), reçu {self.confidence}")
        if self.n_obs < 0:
            raise ValueError(f"n_obs ne peut pas être négatif, reçu {self.n_obs}")
        if self.ci_low > self.ci_high:
            raise ValueError(
                f"intervalle inversé : [{self.ci_low}, {self.ci_high}]"
            )
        if math.isfinite(self.value) and math.isfinite(self.ci_low):
            if not (self.ci_low <= self.value <= self.ci_high):
                raise ValueError(
                    f"la valeur {self.value} est hors de son propre intervalle "
                    f"[{self.ci_low}, {self.ci_high}] — il y a un bug dans "
                    f"l'estimateur '{self.method}'"
                )

    @property
    def half_width(self) -> float:
        """Demi-largeur de l'intervalle. Mesure absolue de l'imprécision."""
        return (self.ci_high - self.ci_low) / 2.0

    @property
    def relative_uncertainty(self) -> float:
        """
        Imprécision relative : demi-largeur / |valeur|.

        C'est LE chiffre à regarder. Une volatilité estimée à 20 % avec une
        incertitude relative de 25 % signifie que la vraie valeur est
        plausiblement entre 15 % et 25 %. Beaucoup de décisions
        d'investissement ne survivent pas à cette fourchette.
        """
        if self.value == 0 or not math.isfinite(self.value):
            return math.inf
        return self.half_width / abs(self.value)

    # Seuil au-delà duquel on considère l'estimation trop imprécise pour
    # servir à une décision. 0,25 signifie : « l'intervalle ne doit pas
    # dépasser ±25 % de la valeur estimée ».
    #
    # Pourquoi 25 % et pas 50 % : à ±25 %, une volatilité estimée à 20 %
    # est « entre 15 % et 25 % » — déjà large, mais on peut encore
    # distinguer un actif prudent d'un actif agressif. À ±50 %, on aurait
    # « entre 10 % et 30 % », ce qui ne discrimine plus rien.
    #
    # Ce seuil correspond en pratique à environ 3 mois de données
    # quotidiennes pour une volatilité. Un mois ne passe pas — et c'est
    # exactement le message qu'on veut faire passer.
    MEANINGFUL_THRESHOLD = 0.25

    def is_meaningful(self, threshold: float | None = None) -> bool:
        """
        L'estimation est-elle assez précise pour qu'on puisse en dire
        quelque chose ?

        Si l'incertitude relative dépasse le seuil, la réponse est non :
        l'intervalle est si large que l'estimation ponctuelle ne porte
        quasiment pas d'information.

        C'est cette méthode qui permet au moteur de REFUSER de répondre.
        """
        limit = self.MEANINGFUL_THRESHOLD if threshold is None else threshold
        return self.relative_uncertainty <= limit

    def format(self, decimals: int = 2, percent: bool = False) -> str:
        """
        Affichage honnête : jamais la valeur sans son intervalle.

        `percent=True` multiplie par 100 et ajoute le signe %. Une
        volatilité stockée comme 0.198 s'affiche alors « 19.80% ».
        Le stockage reste toujours en unités décimales — mélanger les
        deux conventions dans un même code est une source de bug
        classique en finance.
        """
        if not math.isfinite(self.value):
            return "indéterminé (données insuffisantes)"
        k = 100.0 if percent else 1.0
        suffix = "%" if percent else ""
        v = f"{self.value * k:.{decimals}f}{suffix}"
        lo = f"{self.ci_low * k:.{decimals}f}{suffix}"
        hi = f"{self.ci_high * k:.{decimals}f}{suffix}"
        pct = int(round(self.confidence * 100))
        flag = "" if self.is_meaningful() else "  ⚠️ PEU FIABLE"
        return f"{v}  [IC {pct}% : {lo} — {hi}]  n={self.n_obs}{flag}"

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True)
class InsufficientData(Exception):
    """
    Levée quand il n'y a pas assez d'observations pour estimer honnêtement.

    Pourquoi une exception plutôt qu'un retour de NaN : un NaN se propage
    silencieusement à travers les calculs et finit par s'afficher comme un
    trou dans un tableau, que l'utilisateur interprète comme un détail
    technique. Une exception, elle, ARRÊTE le programme et force à traiter
    le cas. Dans un moteur de risque, échouer bruyamment est une vertu.
    """

    n_obs: int
    n_required: int
    what: str

    def __str__(self) -> str:
        return (
            f"Impossible d'estimer '{self.what}' : {self.n_obs} observations "
            f"disponibles, {self.n_required} requises au minimum."
        )
