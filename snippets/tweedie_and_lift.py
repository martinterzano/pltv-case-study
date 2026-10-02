"""
Tweedie regression + top-K lift analysis — the pattern.

Two ideas illustrated together because they live at the same operational seam
for value-prediction problems.

  1. A LightGBM Tweedie regressor with early stopping, wrapped so that the
     variance power (the shape parameter of the Tweedie family) is set once,
     visible in the config, and not re-decided at every training run. Tweedie
     with variance_power in (1, 2) targets a compound Poisson-Gamma
     distribution, which describes premium and revenue targets: mass at zero
     (clients who lapse or churn), positive real support, heavy right tail
     (a small number of high-value clients dominating).

  2. Top-K lift analysis for regression predictions. Ranking metrics (Spearman,
     top-K lift) are the operational metrics when the score feeds a
     prioritisation system with fixed contact capacity. Lift is computed by
     ordering predictions from high to low, taking the top-K share, and
     comparing the sum of true values in that top-K against the expected sum
     under random selection.

The comparison against a naive baseline is built in. Every result is stated as
"model vs baseline delta", not model in isolation. This is the analytical
posture that surfaces segmentation effects (see §8 of the case study README):
without a baseline running alongside, small deltas hide behind large absolute
numbers.

Neither the model wrapper nor the lift function is novel. The point is the
discipline: aligning the loss with the target distribution, and evaluating the
model against a signal the business already trusts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


# ─── Tweedie training wrapper ───────────────────────────────────────────────


def train_tweedie_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    variance_power: float = 1.5,
    n_estimators: int = 5_000,
    early_stopping_rounds: int = 100,
    categorical_features: list[str] | None = None,
    **lgbm_kwargs: Any,
) -> LGBMRegressor:
    """Train a LightGBM Tweedie regressor with early stopping.

    The variance power is an explicit argument, not a default hyperparameter.
    Setting it wrong (say 1.9 when the target is closer to Poisson) produces a
    quiet degradation in the tail predictions that is easy to miss on average
    metrics. Fix it here, log it, version it.

    variance_power = 1.0 corresponds to Poisson; 2.0 corresponds to Gamma.
    Values in between (1.3 to 1.7 in practice) correspond to compound
    Poisson-Gamma distributions with different mass-at-zero behaviour.
    """
    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=variance_power,
        n_estimators=n_estimators,
        random_state=42,
        **lgbm_kwargs,
    )

    fit_kwargs: dict[str, Any] = {
        "eval_set": [(X_valid, y_valid)],
        "callbacks": [early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    }
    if categorical_features:
        fit_kwargs["categorical_feature"] = categorical_features

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ─── top-K lift ─────────────────────────────────────────────────────────────


@dataclass
class LiftReport:
    """Top-K lift for a regression prediction against a baseline.

    - top_k_share: the K used, e.g. 0.01 for the top 1%.
    - model_lift: sum of true values in the top-K by model prediction,
                  divided by the expected sum under uniform random selection.
    - baseline_lift: same but ranked by the baseline prediction.
    - delta: model_lift - baseline_lift. This is where the model earns its keep.
    """

    top_k_share: float
    model_lift: float
    baseline_lift: float
    delta: float


def compute_top_k_lift(
    y_true: np.ndarray | pd.Series,
    y_pred_model: np.ndarray | pd.Series,
    y_pred_baseline: np.ndarray | pd.Series,
    top_k_share: float = 0.01,
) -> LiftReport:
    """Compare model and baseline top-K lift on the same holdout.

    Both are ranked descending by their own prediction. The top-K rows are
    selected, and their true-value sum is compared against the expected sum
    under uniform random selection.
    """
    y_true = np.asarray(y_true)
    y_pred_model = np.asarray(y_pred_model)
    y_pred_baseline = np.asarray(y_pred_baseline)

    n = len(y_true)
    k = max(1, int(round(top_k_share * n)))
    expected_sum = y_true.sum() * (k / n)

    top_model_idx = np.argsort(-y_pred_model)[:k]
    top_baseline_idx = np.argsort(-y_pred_baseline)[:k]

    model_lift = y_true[top_model_idx].sum() / expected_sum if expected_sum > 0 else np.nan
    baseline_lift = y_true[top_baseline_idx].sum() / expected_sum if expected_sum > 0 else np.nan

    return LiftReport(
        top_k_share=top_k_share,
        model_lift=float(model_lift),
        baseline_lift=float(baseline_lift),
        delta=float(model_lift - baseline_lift),
    )


# ─── unified evaluation ─────────────────────────────────────────────────────


def evaluate_against_baseline(
    y_true: np.ndarray | pd.Series,
    y_pred_model: np.ndarray | pd.Series,
    y_pred_baseline: np.ndarray | pd.Series,
    top_k_shares: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Standard side-by-side evaluation.

    Prints MAE, RMSE, Spearman, and top-K lift at each requested K, for both
    the model and the baseline, along with the delta. Returns the same as a
    DataFrame for logging.
    """
    y_true = np.asarray(y_true)
    y_pred_model = np.asarray(y_pred_model)
    y_pred_baseline = np.asarray(y_pred_baseline)

    rows: list[dict[str, Any]] = []

    for name, pred in [("model", y_pred_model), ("baseline", y_pred_baseline)]:
        rows.append(
            {
                "candidate": name,
                "metric": "MAE",
                "value": mean_absolute_error(y_true, pred),
            }
        )
        rows.append(
            {
                "candidate": name,
                "metric": "RMSE",
                "value": np.sqrt(mean_squared_error(y_true, pred)),
            }
        )
        rows.append(
            {
                "candidate": name,
                "metric": "Spearman",
                "value": spearmanr(y_true, pred).correlation,
            }
        )

    for k in top_k_shares:
        lift = compute_top_k_lift(y_true, y_pred_model, y_pred_baseline, top_k_share=k)
        rows.append({"candidate": "model", "metric": f"top{int(k*100)}%_lift", "value": lift.model_lift})
        rows.append({"candidate": "baseline", "metric": f"top{int(k*100)}%_lift", "value": lift.baseline_lift})

    return pd.DataFrame(rows)


# ─── example ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n = 20_000
    # Two-feature synthetic problem with a heavy-tailed target that mimics
    # a compound Poisson-Gamma premium distribution.
    current_annual = np.concatenate(
        [
            rng.gamma(shape=1.5, scale=200, size=int(n * 0.6)),  # mono-policy-like
            rng.gamma(shape=3.5, scale=800, size=int(n * 0.4)),  # multi-policy-like
        ]
    )
    portfolio_breadth = np.concatenate(
        [np.ones(int(n * 0.6)), rng.integers(2, 8, size=int(n * 0.4))]
    )

    # True target is roughly 2x current annual, with some multiplicative noise
    # that scales with portfolio breadth (more variance in multi-policy tail).
    noise = rng.normal(loc=0, scale=0.15 * portfolio_breadth)
    y_true = np.maximum(0, current_annual * 2 * (1 + noise))

    X = pd.DataFrame({"current_annual": current_annual, "portfolio_breadth": portfolio_breadth})
    y = pd.Series(y_true, name="premium_24m")

    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train
    model = train_tweedie_model(X_train, y_train, X_valid, y_valid, variance_power=1.5)

    # Predict
    y_pred_model = model.predict(X_valid)

    # Baseline: current_annual * 2, no model
    y_pred_baseline = X_valid["current_annual"].to_numpy() * 2

    # Report
    report = evaluate_against_baseline(y_valid.values, y_pred_model, y_pred_baseline)
    print(report.pivot(index="metric", columns="candidate", values="value").round(3))

    lift = compute_top_k_lift(y_valid.values, y_pred_model, y_pred_baseline, top_k_share=0.01)
    print(f"\nTop-1% lift: model={lift.model_lift:.3f}, baseline={lift.baseline_lift:.3f}, delta={lift.delta:+.3f}")
