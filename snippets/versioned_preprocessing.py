"""
Versioned preprocessing state for regression on tabular data — the pattern.

Same idea as the classification variant in the lapse case study, adapted to two
transformations that show up repeatedly in premium and revenue prediction:

  1. Winsorising at p99. Numeric features with heavy right tails (premium sums,
     policy counts, transaction volumes) get their upper tail clipped to a
     percentile computed once on the training set. Applied verbatim at scoring
     time. Extreme upstream values, whether genuine outliers or data errors,
     stop being able to silently pull the score around.

  2. Age imputation with a low-side threshold + high-side p99 cap. Owner age
     features often arrive with sentinel zeros (unknown filled as 0) and a long
     tail of implausibly high values (data entry). A single training-time
     policy handles both: below a low threshold, replace with the training
     median; above p99, cap at p99.

The invariant across both is the same as in classification: the parameters
(caps, thresholds, medians) are learned once on the training set, persisted as
JSON alongside the model artefact, and reloaded at scoring and validation time.
Never recomputed at scoring time.

This is illustrated below with a small, deterministic example so the shape of
the artefact and the roundtrip are visible without pulling in any real data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─── p99 winsorising ────────────────────────────────────────────────────────


@dataclass
class P99Caps:
    """Training-time p99 caps for a set of numeric columns.

    fit(...)  computes and stores the caps.
    apply(...) clips at those caps. Rows with NaN are left alone.
    save/load  persist the caps as JSON.
    """

    caps: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "P99Caps":
        self.caps = {
            col: float(df[col].quantile(0.99))
            for col in cols
            if col in df.columns
        }
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, cap in self.caps.items():
            if col in out.columns:
                out[col] = out[col].clip(upper=cap)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.caps, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "P99Caps":
        data: dict[str, float] = json.loads(Path(path).read_text())
        return cls(caps=data)


# ─── owner-age imputation and capping ───────────────────────────────────────


@dataclass
class AgeCapping:
    """Training-time age policy.

    Values below `low_threshold` (typical sentinel zeros or implausibly low)
    are imputed with `median`. Values above `p99_cap` are clipped.
    """

    low_threshold: float = 0.0
    median: float = 0.0
    p99_cap: float = 0.0

    def fit(
        self,
        df: pd.DataFrame,
        col: str,
        low_threshold: float,
    ) -> "AgeCapping":
        valid = df[col][(df[col] >= low_threshold) & df[col].notna()]
        self.low_threshold = float(low_threshold)
        self.median = float(valid.median())
        self.p99_cap = float(valid.quantile(0.99))
        return self

    def apply(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        out = df.copy()
        out[col] = out[col].where(
            (out[col] >= self.low_threshold) & out[col].notna(),
            other=self.median,
        )
        out[col] = out[col].clip(upper=self.p99_cap)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "AgeCapping":
        data: dict[str, Any] = json.loads(Path(path).read_text())
        return cls(**data)


# ─── example ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Deterministic synthetic training set with the two shapes the patterns
    # were designed for: a right-skewed numeric feature and an owner age
    # column with sentinel zeros and implausibly high values.
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "portfolio_value": np.concatenate(
                [
                    rng.exponential(scale=500, size=9800),
                    rng.uniform(50_000, 200_000, size=200),  # tail
                ]
            ),
            "owner_age": np.concatenate(
                [
                    np.zeros(400),  # sentinel unknowns
                    rng.normal(52, 15, size=9400).clip(min=1),
                    rng.uniform(120, 999, size=200),  # bad entries
                ]
            ),
        }
    )

    caps = P99Caps().fit(train, cols=["portfolio_value"])
    age = AgeCapping().fit(train, col="owner_age", low_threshold=18)

    caps.save("artefacts/p99_caps.json")
    age.save("artefacts/age_capping.json")

    scoring = pd.DataFrame(
        {
            "portfolio_value": [300, 1200, 300_000],  # last one gets clipped
            "owner_age": [0, 45, 600],  # first imputed, third capped
        }
    )

    loaded_caps = P99Caps.load("artefacts/p99_caps.json")
    loaded_age = AgeCapping.load("artefacts/age_capping.json")

    out = loaded_caps.apply(scoring)
    out = loaded_age.apply(out, col="owner_age")

    print("Loaded p99 caps:", loaded_caps.caps)
    print("Loaded age policy:", asdict(loaded_age))
    print("Scoring input:")
    print(scoring)
    print("Scoring output:")
    print(out)
