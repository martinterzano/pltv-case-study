# Predictive LTV: Case Study

> Monthly LightGBM Tweedie regressor that predicts individual 24-month cumulative
> premium for an insurance portfolio of ~2M policyholders. Score is used as the
> value-prioritisation signal in a Next-Best-Action system. Production on Google
> Cloud, monthly cadence, live since early 2026.

**Sector:** insurance · **Role:** end-to-end (framing, data, model, production, monitoring) · **Stack:** LightGBM (Tweedie) · Python · BigQuery · GCP (Vertex AI Workbench, Cloud Composer, Cloud Storage) · **Status:** in production, monthly cadence

*Writeup prepared October 2026; the implementation is client property.*

The client owns the business metrics and internal identifiers. This case study
describes how the problem was framed, how the model was built, and how the system
reasons, without publishing client data, feature names, or exact performance
numbers. Every metric below is expressed as a range or order of magnitude, and
predictive features are described by concept and group rather than by column name.

---

## Contents

1. [The problem](#1-the-problem)
2. [Choosing Tweedie for premium prediction](#2-choosing-tweedie-for-premium-prediction)
3. [The baseline that made the model earn its place](#3-the-baseline-that-made-the-model-earn-its-place)
4. [Model comparison](#4-model-comparison)
5. [Feature engineering: ~10 features, one dominant signal](#5-feature-engineering-10-features-one-dominant-signal)
6. [Feature importance and the architectural ceiling](#6-feature-importance-and-the-architectural-ceiling)
7. [Validation strategy](#7-validation-strategy)
8. [Segmentation: where the model actually adds value](#8-segmentation-where-the-model-actually-adds-value)
9. [Data quality guardrails](#9-data-quality-guardrails)
10. [Production system](#10-production-system)
11. [Model card: scope, assumptions and limitations](#11-model-card-scope-assumptions-and-limitations)
12. [Scoping decisions and repository contents](#12-scoping-decisions-and-repository-contents)

---

## 1. The problem

An insurance business runs a Next-Best-Action (NBA) system to prioritise which
active policyholders to target for cross-sell, upsell, renewal, and retention
actions. As part of this strategy, NBA rankings are only as useful as the value signal underneath them.
Without a reliable estimate of how much each client is worth going forward, any
prioritisation collapses into "reach out to whoever comes up first", which is
what the system was trying to replace. The company also aims to build a system where diverse models, each built for a specific goal, work together in the same prioritisation framework.

The model built here estimates **individual 24-month cumulative premium** for
every active policyholder in the portfolio. The 24-month horizon is a business
decision, matching the NBA planning cycle.
Individual-level (not policy-level) aggregation is a modelling decision: NBA
operates on the client, not the contract.

Constraints going in:

- Portfolio scale in the millions of active policyholders. Monthly batch scoring, not real-time.
- The target (24-month cumulative premium) is a positive real-valued number with a heavy right-skew, a large mass close to zero (mono-policy clients with small annual premiums), and a long tail (multi-policy high-value clients). This drove the algorithm choice and brought up meaningful discussions with the business when framing the modelling approach, see §2 and §4.
- The score would feed a downstream NBA ranking. Ranking performance matters more than point accuracy on the individual prediction.
- The system had to co-exist with an existing naive baseline the business already trusted. That baseline had to be beaten *and* preserved as a companion signal, see §3.

## 2. Choosing Tweedie for premium prediction

The target distribution is what pushed the choice. Individual 24-month premium
has three properties that most standard regression setups handle poorly.

1. **Positive real-valued with a non-trivial mass at zero.** Some policyholders lapse fully within the window and contribute nothing further. Standard Gaussian regression assumes symmetric errors and produces negative predictions, which are meaningless here.
2. **Heavy right skew.** Multi-policy high-value clients dominate the tail. MSE loss on the raw target penalises tail errors quadratically but treats them on the wrong scale, since a €50 error on a €100 target is very different from a €50 error on a €10,000 target.
3. **Compound Poisson-Gamma structure.** Each policyholder's 24-month premium is the sum over their active policies of a payment amount times the number of paid periods. That is textbook compound Poisson-Gamma, the exact distribution the Tweedie family with `variance_power ∈ (1, 2)` was designed to model.

The Tweedie GLM family lets the model directly optimise for this distributional
shape. Setting `tweedie_variance_power=1.5` places the loss in the middle of the
Poisson-Gamma range, which matched the empirical variance-to-mean ratio observed
in the training set.

Alternatives evaluated and dropped, see §4 for the empirical comparison:

- **MSE regression on the raw target.** Trained but tail-biased in the wrong direction, and produces negative predictions at the low end.
- **MSE regression on log-transformed target.** Handles the skew but re-introduces the mass-at-zero problem (log of a small number diverges), and requires a delicate re-transformation for inference.
- **Gamma GLM.** Handles the skew but assumes strictly positive support, so policyholders with zero premium in the window have to be modelled separately.
- **Two-stage compound model (churn classifier + Gamma regressor for survivors).** Technically sound in principle: train a binary lapse classifier first, then a Gamma regressor on the sub-population that stays. Three arguments pushed against it in this context. First, a separate early-lapse model already existed targeting a different time horizon. Repurposing it as the first stage would require retraining and temporal reconciliation between two different label windows. Second, prediction errors compound: the Gamma stage inherits the lapse model's miscalibration, and explaining the combined uncertainty to the business becomes genuinely hard. Third, it doubles the operational footprint: two models, two retraining runbooks, two monitoring streams. Tweedie handles both stages in a single objective, with a single artefact to maintain.

Tweedie captures all three properties in a single objective.

## 3. The baseline that made the model earn its place

Throughout development and validation, the model was compared against a **naive
baseline**: predicted 24-month premium = current annual premium × 2. Anyone with
access to the portfolio can produce this baseline in a single SQL query.

The baseline was chosen for three reasons.

1. **It is operationally realistic.** If the model cannot consistently beat what anyone can produce in a single SQL query, there is no justification for the added complexity of a machine learning system.
2. **It is a strong benchmark, not a straw man.** Annual premium is a highly stable signal. Most individuals don't change their premium year over year. The baseline is naturally well-correlated with the 24-month target, setting a high bar for the model to clear.
3. **It makes tradeoffs the team should see.** The baseline wins on some metrics (MAE) and the model wins on others (RMSE, ranking, tail identification). The model is not universally better. It is better where it matters most for NBA prioritisation.

The baseline is computed at scoring time and stored alongside the model
prediction, so any downstream consumer can compare both signals per policyholder
and inspect where they diverge.

## 4. Model comparison

Four candidates evaluated on the same training and validation split, same feature
set, same random seed. All metrics on the OOT validation set.

| Candidate | RMSE (relative) | MAE (relative) | Spearman | Top-1% lift | Notes |
|---|---|---|---|---|---|
| Baseline (annual × 2) | high | low | ~0.94 | ~8.4 | The bar to clear. Wins MAE by design (see §3). |
| LightGBM, MSE loss | mid-high | mid | ~0.94 | ~8.7 | Symmetric loss produces some negative predictions and under-weights the tail. |
| LightGBM, MSE on log(target) | mid | mid-high | ~0.94 | ~8.6 | Skew handled but mass-at-zero returns as a re-transformation problem. |
| XGBoost, MSE loss | mid | mid | ~0.95 | ~8.7 | Comparable to LightGBM MSE on this data shape. Not enough of an edge to overcome categorical-encoding overhead. |
| **LightGBM, Tweedie (var. power=1.5)** ← **adopted** | **lowest** | mid | **~0.95** | **~8.9** | Wins ranking metrics and tail RMSE. Loss aligned with target distribution. |

**Why not time-series models.** The problem was framed as a cross-sectional regression, not as a time-series forecast. The target (24-month cumulative premium per individual) is not a single signal observed over time. It is a label derived from the future behaviour of millions of different individuals at a single snapshot date. Time-series models would require individual-level histories of sufficient length and uniform observation frequency, with the same individuals appearing across periods. That does not hold cleanly at portfolio scale or with the customer turnover typical of an insurance book. A cross-sectional regressor that uses client-level features as of the snapshot date is both more tractable and a better use of the breadth of signals available in the portfolio data. The richer the feature set covering demographics, portfolio composition, payment behaviour, and relationship time, the more a cross-sectional approach benefits relative to a univariate temporal one.

**Why Tweedie beats LightGBM MSE.** MSE minimises squared error on the raw
target, which under-weights the tail relative to its business importance and
allows negative predictions. Tweedie's Poisson-Gamma loss aligns the training
objective with the target distribution and produces strictly positive
predictions by construction.

## 5. Feature engineering: ~10 features, one dominant signal

The final feature set is small on purpose. Around 10 features, all computable
directly from the client-level view of the portfolio without additional derived
tables.

| Group | Concept | Type |
|---|---|---|
| **Portfolio value** | Relative to annual premiums (**dominant signal**) | Numeric |
| **Portfolio breadth** | Relative to policy distribution | Numeric |
| **Relationship time** | Relative to recency | Numeric |
| **Demographics** | Relative to demographics | Numeric |
| **Product mix** | Relative to individual portfolio | Numeric |
| **Acquisition** | Relative to acquisition characteristics | Categorical |

The individual-level aggregation logic is the load-bearing design decision.
Policies belong to owners, and one owner can hold several policies. The features are
computed at the *owner* level: sums, counts, etc across all of that owner's policies. This is where the model's
value over the baseline (which only sees today's annual premium sum) will come
from, since portfolio breadth and time-since signals only exist at the owner
level and only matter for multi-policy owners.

### Preprocessing captured as versioned artefacts

Three transformations run at both training and scoring time, and all three read
their parameters from JSON files persisted alongside the model.

- **Winsorising at p99** on the three numeric portfolio-value and count features. Caps computed on the training set, applied verbatim at scoring time. Extreme upstream values (data errors or genuine outliers) do not silently corrupt scoring.
- **Numerical imputation and capping.** A pragmatic rule with a training-time low threshold and a training-time p99 cap. Persisted as JSON, loaded at scoring time.

The pattern is illustrated in [`snippets/versioned_preprocessing.py`](snippets/versioned_preprocessing.py).

## 6. Feature importance and the architectural ceiling

Post-training SHAP analysis shows what any experienced practitioner in this
domain expects: **a high concentration of importance in a single feature**,
related to active annual premiums per individual. This is neither surprising nor
a bug. Today's annual premium is the strongest available signal for a 24-month
cumulative premium prediction.

Two implications of this concentration shaped the case study going forward.

**The model and the baseline are structurally similar.** The baseline (annual
premium × 2) is a linear function of the single dominant feature. The model
uses the same feature plus a small number of complementary signals. Their global
correlation is very high.
The model's *incremental* value over the baseline lives in the remaining share
of importance, which is where the complementary features act. See §8 for the
segmentation that makes this concrete.

**There is an architectural ceiling.** Adding features that are correlated with
the dominant one produces no lift. Adding features that describe genuinely
different phenomena (portfolio composition, product mix, cross-line behaviour)
produces marginal lift because their contribution is diluted by the dominant
signal in the global loss. A meaningful next step is a dedicated model trained
only on the segment where the complementary features actually matter, see §8
and the v2 note in §11.

**Why we don't drop the dominant feature.** Removing it degrades the model
substantially and produces a worse ranker than the baseline. The dominance is a
property of the problem, not of the modelling choice. The right response is
segmentation, not de-signalling.

## 7. Validation strategy

Three layers of validation, each catching a different failure mode. The
comparison against baseline runs through all three as a validation companion,
not as a separate check.

### 7.1 Snapshot-based training

Training uses a snapshot of the active portfolio at date `T`, with the label
computed as the cumulative premium paid over `[T, T+24 months]`. The snapshot
logic (which policyholders were active at `T` and only using data known before
`T`) is enforced in the SQL that builds the training set, and does not depend on
the model code.

### 7.2 Out-of-time validation

A completely independent, later snapshot `T'` with the label computed over
`[T', T'+24 months]` is used as the primary generalisation test. The OOT window
matches the training label window (24 months), so that ranking performance is
measured on the same task the model was trained for.

Validation metrics are always computed for **both** the model and the baseline
on the same OOT set. Every result is stated as *"model X, baseline Y, delta Z"*.
This forces the analysis to answer "does the model beat what the business
already has?", not just "how well does the model do?". It is the analytical
posture that made the segmentation finding in §8 visible.

### 7.3 Global metrics on OOT

*All numbers below are approximate. The client owns the exact metrics.*

| Metric | Model | Baseline | Winner | Interpretation |
|---|---|---|---|---|
| MAE | mid | **low** | Baseline | Baseline wins point accuracy on average. This is expected. |
| RMSE | **low** | mid | Model | Model handles tail errors better. |
| Spearman | **~0.95** | ~0.94 | Model | Both rank well. Model marginally better globally. |
| Top-1% lift | **~9x** | ~8x | Model | Model concentrates more value at the very top. |
| Calibration ratio | ~1.0 | ~1.0 | Tie | Both are well-calibrated globally. |

Two things worth noting.

**MAE goes to the baseline.** The model wins RMSE (tail errors) but loses MAE
(mean error). This is the honest tradeoff. Point accuracy on the average
policyholder is not the metric NBA prioritisation optimises for. Ranking is,
and the model wins there. Publishing the baseline's MAE win alongside the
model's ranking win is an audit trail of the design choice, not a weakness.

**Global metrics understate the operational value.** The globally small delta
(Spearman +0.005, top-1% lift +0.5) is the average of a *very* uneven
distribution across segments. Section 8 breaks this down.

## 8. Segmentation: where the model actually adds value

This is the most operationally important finding of the validation, and the
insight that shaped the deployment strategy.

The portfolio splits cleanly into two segments.

- **Mono-policy clients** (~60% of the portfolio, ~25% of total premium): individuals holding exactly one active policy.
- **Multi-policy clients** (~40% of the portfolio, ~75% of total premium): individuals holding two or more.

The model's incremental value over the baseline is **not evenly distributed**.
It is almost entirely concentrated in multi-policy clients.

| Segment | Delta top-1% lift (model minus baseline) | Interpretation |
|---|---|---|
| Mono-policy | marginal | Model and baseline are effectively equivalent for these clients. |
| Multi-policy | sizeable | Model concentrates additional premium in the top 1% that the baseline misses. |

**The multi-policy delta is roughly 20x the mono-policy delta.** The bulk of the
incremental premium captured by the model in the top 1% comes from the
multi-policy segment.

### Why mono-policy converges with the baseline

For mono-policy clients, the dominant feature (active annual premiums per
individual) reduces to a single policy's annual premium. The baseline (that
number times 2) is a near-perfect predictor. The remaining features have very little room to contribute, since
the client has no portfolio breadth and their only relationship-time signal is
already collinear with maturity of the single policy.

### Why multi-policy is where the model adds value

For multi-policy clients, the dominant feature aggregates premiums across
multiple policies of different types, tenures, and risk profiles. Here the
complementary features carry real predictive weight. The model can distinguish
between an owner with three high-tenure, high-value policies and an owner with
three low-tenure, low-value policies whose current annual premium happens to be
similar. The baseline cannot.

### Decile composition in production

Ranking the full population by predicted premium and looking at the multi-policy
share by decile makes the segmentation-model interaction visible.

| Decile | Approx. multi-policy share | Behaviour |
|---|---|---|
| 1 to 3 (lowest predicted premium) | ~1 to 17% multi | Dominated by mono-policy. Both model and baseline agree on ranking. |
| 4 to 5 (transition zone) | ~20 to 45% multi | Composition inflection point. |
| 6 to 8 (upper-middle) | ~55 to 75% multi | Multi-policy majority. Model starts pulling away from baseline. |
| 9 to 10 (top) | ~85 to 90% multi | **This is where the model earns its place.** In the top decile, the model predicts systematically higher than the baseline for multi-policy clients and slightly lower for the remaining mono-policy ones. |

### Implication for NBA prioritisation

Any NBA action targeting the top decile should expect approximately 90% of the
audience to be multi-policy. This is the correct behaviour, and it reflects
where value concentration and predictive signal overlap. Prioritising the top
decile by the model score rather than by the baseline surfaces roughly the
right proportion of additional multi-policy value that the business was
previously under-weighting.

The v2 roadmap (see §11) builds on this segmentation directly, with a dedicated
model trained only on multi-policy clients and enriched with portfolio-composition
features (product mix, contract maturity, cross-line concentration).

## 9. Data quality guardrails

The high concentration of importance in a single feature (see §6) creates a
specific class of silent failure. If the upstream ETL pipeline that computes
that feature drifts (a schema change, a units mismatch, a missing join), the
score at every policyholder shifts by the same multiplicative factor without
producing any visible error. Downstream NBA prioritisation stays internally
consistent but is anchored on the wrong scale.

The scoring script therefore checks two quantities before running the model.

- **Mean deviation of the dominant feature.** The mean of the dominant feature in the current scoring input is compared against the training-time reference mean. If the deviation exceeds a threshold of ~20%, scoring is halted with an error rather than allowed to produce shifted predictions.
- **p99 deviation of the dominant feature.** Same check on the p99, with a slightly wider threshold. Catches asymmetric drift where the mean stays stable but the tail moves.

The reference values (training-time mean and p99, plus the thresholds) are
loaded from a JSON artefact persisted alongside the model. This makes the
guardrail portable across retraining runs.

Beyond the guardrails, monthly scheduled queries land distribution snapshots
into a monitoring table. Three signals trigger investigation.

- Month-over-month deviation of the baseline mean above ~10%. Suggests an upstream input problem, since the baseline is a single-feature transformation.
- Month-over-month change in the active population above ~5%. Sharp changes in scoring volume indicate portfolio-level events (large-scale renewals, campaign effects, or data pipeline issues).
- Multi-policy share of the top decile dropping below ~85%. Signals portfolio composition drift that would eventually degrade the model's incremental value.

Gradual month-over-month growth in the mean is expected (premium inflation) and
is not an alert.

## 10. Production system

The system runs in three planes. Training is manual and infrequent, scoring is
automated and monthly, monitoring is scheduled and continuous.

### 10.1 Architecture

```
BigQuery (training snapshot)  →  Vertex AI Workbench (training)  →  Cloud Storage (model + JSON artefacts)
                                                                                      ↓
BigQuery (scoring input)      →  Cloud Composer DAG (monthly)     →  BigQuery (scored table)
                                                                                      ↓
                                                                     BigQuery (monitoring table, scheduled query)
```

### 10.2 Config as single source of truth

A `config.yaml` file at the root of the repository declares every path, table ID,
GCS URI, and tunable parameter used by the training notebooks, the scoring
script, the DAG, and the monitoring query. Scripts import this file rather than
hardcoding values.

Two reasons this matters more than in a typical project.

- The system has three execution surfaces (training notebook, scoring script, DAG) that must all point to the same artefacts. Divergence between them produces silent corruption that only surfaces at monitoring time.
- The data-quality guardrail thresholds (see §9) are tuning parameters, and having them live in one file makes it explicit that changing them is a deliberate act, not a code edit.

### 10.3 Scoring output schema

Storing the baseline in the same row as the model prediction is what makes it
possible for any downstream consumer, or any future audit, to reproduce the
comparisons in §7 and §8 without needing to re-run training or recompute the
baseline retroactively.

### 10.4 Deliberate scoping

- **No online scoring.** Building a real-time serving layer would add complexity and cost for a use case that the business case does not need it.
- **No automated retraining.** Retraining requires a new OOT window that has not been used before, which is a temporal question, not a scheduling question. The retraining process is documented as a runbook rather than automated end-to-end.
- **No feature-store integration.** The feature set is small enough (~10 columns) that the marginal complexity of a feature store outweighs its benefit here. If the feature count grows meaningfully in v2, this decision is worth revisiting.

## 11. Model card: scope, assumptions and limitations

### Intended use

Rank active policyholders of an existing insurance portfolio by predicted
24-month cumulative premium, to feed a Next-Best-Action prioritisation system
that plans cross-sell, upsell, and renewal actions on a monthly cadence with
human agents in the loop.

### Out of scope

- **Automatic actions on the client** without human review. The model outputs a value estimate, not a decision. NBA logic (which action, when, by whom) is downstream and human-governed.
- **New products or new business lines.** The model has no signal on portfolio composition it has never seen. Scores for policyholders whose portfolio is dominated by a newly launched product should be treated with caution until a retrained model incorporates the new mix.
- **Individual-level causal claims.** SHAP explains what pushed a *score*, and does not explain what is *causing* future premium. A high score is a signal that a client will likely pay more premium going forward, and it is not a claim that any specific action will change that.
- **Very low predicted premium as a lapse signal.** A low score means the model predicts low future premium, and that does not mean the client is at risk of lapsing. Lapse risk is a different modelling problem with a different target and a different feature set.

### Known limitations

1. **High concentration of importance in a single feature.** The dominant feature drives a very large share of the model output. Consequences: the model correlates strongly with the naive baseline, the incremental value over the baseline is small on average, and the value is concentrated in the multi-policy segment (see §8). This is the architectural ceiling. The v2 roadmap addresses it with a dedicated multi-policy model.
2. **Marginal value for mono-policy clients.** For the ~60% of the portfolio that holds a single active policy, the model and the baseline are effectively equivalent. Running both is not wrong, and it is not where the model is doing meaningful work either.
3. **MAE loss to the baseline is real.** The model wins ranking and tail metrics, and it loses average point accuracy to the baseline. If a downstream use case ever prioritises point accuracy over ranking (for example, individual premium forecasting for accounting), this model is not the right tool.
4. **Snapshot cadence is monthly.** Between two monthly runs, a policyholder's ranking can only change if new data lands in the scoring input for the next run. Rapid intra-month changes (a new policy conversion mid-month) are not reflected until the next scoring.

### V2 roadmap

The most promising direction is a **dedicated multi-policy model** trained
exclusively on the multi-policy segment, enriched with portfolio-composition
features (product mix, contract maturity distribution, cross-line concentration,
premium concentration index). This targets the segment where the current model
already adds value, and removes the dilution effect of training jointly with
mono-policy clients whose behaviour is fully captured by a single feature.

## 12. Scoping decisions and repository contents

### What is in this repo

```
pltv-case-study/
├── README.md                          ← this document
└── snippets/
    ├── versioned_preprocessing.py     ← pattern: JSON-persisted winsorising and imputation
    └── tweedie_and_lift.py            ← pattern: Tweedie training wrapper and top-K lift analysis
```

The full training code, feature engineering SQL, and production scoring script
are the client's property and are not published here. The snippets illustrate
transferable technique (versioned preprocessing state, Tweedie training with
early stopping, top-K lift computation on regression predictions), and they do
not contain client logic.

### What this case study is meant to show

- End-to-end delivery on GCP, from framing to production and monitoring.
- **Choosing a loss aligned with the target distribution** (Tweedie for compound Poisson-Gamma), not a default MSE.
- **A naive baseline treated as a first-class validation companion**, computed at every scoring run.
- **Honest reporting of tradeoffs**: baseline wins MAE, model wins ranking, and both facts are published side by side.
- **Segmentation-driven value analysis**: recognising that the incremental value of a model is not evenly distributed, and letting that shape both the deployment story and the v2 roadmap.
- **Data-quality guardrails targeted at the failure mode** (silent drift of the dominant feature), not a generic ETL check.
- **A model card that names its architectural ceiling** and describes the concrete direction to break through it.

---

Martín Terzano · [helliumlab.com](https://helliumlab.com) · [LinkedIn](https://www.linkedin.com/in/martinterzano) · martin@helliumlab.com
