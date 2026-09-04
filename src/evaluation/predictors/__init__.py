"""Predictor base class and plug-in registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Interface every predictor must implement."""

    # Override in subclass to skip specific CV scenarios (e.g. expensive pairwise splits).
    skip_scenarios: frozenset = frozenset()
    # Override in subclass with a list of pickle cache paths this predictor needs.
    cache_paths: list = []
    # Override in subclass to cap per-predictor parallelism (0 = use global --n-jobs).
    preferred_n_jobs: int = 0

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Train on a DataFrame of labelled PPI variants."""

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return probability scores (float in [0, 1]) for perturbed=True."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier used in result tables."""


# ── registry ──────────────────────────────────────────────────────────────────

PREDICTOR_REGISTRY: dict[str, type[BasePredictor]] = {}


def register(cls: type[BasePredictor]) -> type[BasePredictor]:
    """Class decorator that adds a predictor to the global registry."""
    PREDICTOR_REGISTRY[cls().name] = cls
    return cls


def get_all_predictors(seed: int = 42, device: str = "") -> list[BasePredictor]:
    """Return one instance of every registered predictor.

    Args:
        seed:   Random seed for seeding predictor RNGs.
        device: Optional device string (e.g. "cuda:3") forwarded to predictors
                whose constructor accepts a ``device`` keyword argument.
    """
    predictors = []
    for cls in PREDICTOR_REGISTRY.values():
        for kwargs in (
            [{"seed": seed, "device": device}] if device else []
        ) + [{"seed": seed}, {}]:
            try:
                pred = cls(**kwargs)
                break
            except TypeError:
                continue
        predictors.append(pred)
    return predictors
