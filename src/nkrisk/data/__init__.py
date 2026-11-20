"""Données : générateurs synthétiques à vérité connue, sources réelles, qualité."""

from .market import (
    CsvSource,
    InMemorySource,
    LoadResult,
    PriceSource,
    YahooSource,
    load_prices,
)
from .quality import (
    AlignmentComparison,
    Anomaly,
    QualityReport,
    align_prices,
    compare_alignment,
    inspect_prices,
    observations_per_year,
)
from .synthetic import (
    GBMSpec,
    StudentTSpec,
    generate_correlated_prices,
    generate_prices,
    generate_student_t_prices,
)

__all__ = [
    "AlignmentComparison",
    "Anomaly",
    "CsvSource",
    "GBMSpec",
    "InMemorySource",
    "LoadResult",
    "PriceSource",
    "QualityReport",
    "StudentTSpec",
    "YahooSource",
    "align_prices",
    "compare_alignment",
    "generate_correlated_prices",
    "generate_prices",
    "generate_student_t_prices",
    "inspect_prices",
    "load_prices",
    "observations_per_year",
]
