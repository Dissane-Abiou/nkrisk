"""nkrisk — moteur de mesure de risque de portefeuille.

Principe directeur : aucune fonction ne retourne un nombre nu.
Tout retourne un `Estimate`, qui transporte son incertitude.
"""

from .types import Estimate, InsufficientData
from .returns import (
    simple_returns,
    log_returns,
    annualized_volatility,
    annualized_mean_return,
    geometric_return,
    volatility_drag,
    compound_annualize,
    sharpe_ratio,
    TRADING_DAYS_PER_YEAR,
)
from .covariance import CovarianceEstimate, sample_covariance, ledoit_wolf
from .var import (
    TailRisk,
    KupiecResult,
    historical_var,
    parametric_var,
    cornish_fisher_var,
    kupiec_test,
    subadditivity_counterexample,
)
from .risk import (
    RiskDecomposition,
    decompose_risk,
    portfolio_volatility,
    minimum_variance_weights,
    equal_weights,
)

__version__ = "0.1.0"
__all__ = [
    "Estimate",
    "InsufficientData",
    "simple_returns",
    "log_returns",
    "annualized_volatility",
    "annualized_mean_return",
    "geometric_return",
    "volatility_drag",
    "compound_annualize",
    "sharpe_ratio",
    "TRADING_DAYS_PER_YEAR",
    "CovarianceEstimate",
    "sample_covariance",
    "ledoit_wolf",
    "RiskDecomposition",
    "decompose_risk",
    "portfolio_volatility",
    "minimum_variance_weights",
    "equal_weights",
    "TailRisk",
    "KupiecResult",
    "historical_var",
    "parametric_var",
    "cornish_fisher_var",
    "kupiec_test",
    "subadditivity_counterexample",
]
