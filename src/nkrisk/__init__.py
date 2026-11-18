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
from .portfolio import Portfolio, equal_weighted
from .stress import (
    Shock, VolatilityShock, CorrelationShock, PriceShock, Combined,
    StressResult, stress, stress_table,
    CRISE_2008, CHOC_LIQUIDITE, HAUSSE_TAUX, DECORRELATION, SCENARIOS_STANDARD,
)
from .drawdown import (
    DrawdownEpisode,
    drawdown_curve,
    max_drawdown,
    drawdown_episodes,
    ulcer_index,
    calmar_ratio,
    expected_max_drawdown,
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
    "DrawdownEpisode",
    "drawdown_curve",
    "max_drawdown",
    "drawdown_episodes",
    "ulcer_index",
    "calmar_ratio",
    "expected_max_drawdown",
    "Portfolio", "equal_weighted",
    "Shock", "VolatilityShock", "CorrelationShock", "PriceShock", "Combined",
    "StressResult", "stress", "stress_table",
    "CRISE_2008", "CHOC_LIQUIDITE", "HAUSSE_TAUX", "DECORRELATION",
    "SCENARIOS_STANDARD",
]
