"""
simulate_treatment.py — Treatment effect simulator.

Injects realistic, heterogeneous treatment effects into the treatment
group's metrics. Stores ground truth so the stats engine can be
validated against known effects.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

# decorator
@dataclass
class TreatmentEffect:
    """Define a treatment effect for a specific metric and segment."""
    metric: str    # converted, revenue, pageviews, add_to_cart_events...
    effect_type: str         # "multiplicative" or "additive"
    effect_size: float
    segment_col: Optional[str] = None
    segment_value: Optional[str] = None


# ground truth effects for the checkout redesign experiment
DEFAULT_EFFECTS = [
    # Global: 3% relative conversion lift
    TreatmentEffect("converted", "multiplicative", 1.03),
    # Mobile: stronger 5% lift (checkout was worst on mobile)
    TreatmentEffect("converted", "multiplicative", 1.05,
                     segment_col="device_type", segment_value="mobile"),
    # Global: 2% revenue lift
    TreatmentEffect("revenue", "multiplicative", 1.02),
    # Desktop: slight pageview decrease (fewer steps in checkout)
    TreatmentEffect("pageviews", "multiplicative", 0.97,
                     segment_col="device_type", segment_value="desktop"),
    # Global: small add-to-cart improvement
    TreatmentEffect("add_to_cart_events", "multiplicative", 1.02),
    # Global: +5 seconds increase in session duration (users spend slightly longer)
    TreatmentEffect("session_duration_sec", "additive", 5.0),
]


def simulate_treatment_effects(
    df: pd.DataFrame,
    effects: list[TreatmentEffect] = DEFAULT_EFFECTS,
    seed: int = 42,
) -> pd.DataFrame:
    """Applying treatment effects to the treatment group only."""
    df = df.copy()
    rng = np.random.RandomState(seed)
    treatment_mask = df["variant"] == "treatment"

    for effect in effects:
        # building target mask
        mask = treatment_mask.copy()
        if effect.segment_col and effect.segment_value:
            # if segment_col is specified, further filter to that segment (e.g. mobile users)
            mask = mask & (df[effect.segment_col] == effect.segment_value)

        if mask.sum() == 0:
            continue

        if effect.metric == "converted":
            _simulate_binary(df, mask, effect, rng)
        else:
            _simulate_continuous(df, mask, effect, rng)

    # Recompute derived fields after injection
    df["converted"] = (df["transactions"] > 0).astype(int)
    df["is_bounce_proxy"] = (
        (df["pageviews"] <= 1) & (df["session_duration_sec"] < 10)
    ).astype(int)

    return df

# internal functions
def _simulate_binary(df, mask, effect, rng):
    """
    FLipping non-converters to converters so we can achieve the higher
    conversion rate to 1.03 or whatever the target is.
    Then sample realistic revenue values from existing purchasers for the new converters.
    """
    # getting mean of people in treatment group and specific segment and if they converted or not
    current_rate = df.loc[mask, "converted"].mean()
    target_rate = min(current_rate * effect.effect_size, 1.0)

    non_converters = mask & (df["converted"] == 0)
    num_non_converters = non_converters.sum()
    additional_needed = int((target_rate - current_rate) * mask.sum())
    additional_needed = min(max(additional_needed, 0), num_non_converters)

    if additional_needed == 0:
        return

    # flip selected non-converters - finding the row index of non converters and randomly selecting
    # replace=False so we don't flip the same person twice
    flip_idx = rng.choice(
        df.index[non_converters], size=additional_needed, replace=False
    )
    # convert values in converted and transactions to 1 for row index in flip_idx
    df.loc[flip_idx, "converted"] = 1
    df.loc[flip_idx, "transactions"] = 1

    # give new converters realistic revenue from existing distribution
    existing_rev = df.loc[(df["converted"] == 1) & (df["revenue"] > 0), "revenue"]
    if len(existing_rev) > 0:
        df.loc[flip_idx, "revenue"] = rng.choice(
            existing_rev.values, size=len(flip_idx)
        )


def _simulate_continuous(df, mask, effect, rng):
    """
    For continuous metrics: apply effect with per-row noise
    so it's not a perfectly uniform shift.
    """
    col = effect.metric
    # looking at specific ppl in treatment and metric 
    vals = df.loc[mask, col].copy()

    if effect.effect_type == "multiplicative":
        noise = rng.normal(1.0, 0.01, size=mask.sum())
        # Scale values by effect_size and multiply by noise (normal distribution)
        # Noise adds realistic variation across users while preserving the expected value 
        new_vals = (vals * effect.effect_size * noise).clip(lower=0)
    else:
        # Scale values by effect_size and add noise (normal distribution cenered 
        # around 0 with std dev of 10% of the effect size)
        noise = rng.normal(0, abs(effect.effect_size) * 0.1, size=mask.sum())
        new_vals = (vals + effect.effect_size + noise).clip(lower=0)

    # round integer columns
    if col in ["pageviews", "event_count", "add_to_cart_events", "transactions", "session_duration_sec"]:
        new_vals = new_vals.round().astype(int)

    df.loc[mask, col] = new_vals


def get_ground_truth(effects: list[TreatmentEffect] = DEFAULT_EFFECTS) -> dict:
    """Return ground truth for validation against stats engine output."""
    truth = {}
    for e in effects:
        key = e.metric
        if e.segment_col:
            key = f"{e.metric}_{e.segment_col} = {e.segment_value}"
        truth[key] = {
            "effect_type": e.effect_type,
            "effect_size": e.effect_size,
            "segment": f"{e.segment_col} = {e.segment_value}" if e.segment_col else "global",
        }
    return truth


if __name__ == "__main__":
    from clean import build_clean_sessions
    from assign_experiment import assign_users

    df = build_clean_sessions()
    df = assign_users(df)

    print("BEFORE SIMULATION:")
    print(df.groupby("variant")[["converted", "revenue", "pageviews", "session_duration_sec"]].mean())

    # simulate treatment effects in the treatment group and store ground truth for validation
    df = simulate_treatment_effects(df)

    print("\nAFTER SIMULATION:")
    print(df.groupby("variant")[["converted", "revenue", "pageviews", "session_duration_sec"]].mean())

    print("\nGROUND TRUTH:")
    for k, v in get_ground_truth().items():
        print(f"  {k}: {v}")

