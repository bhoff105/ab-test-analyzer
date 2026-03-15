"""
generate_test_data.py
=====================
Synthetic A/B Test Data Generator
Homepage Banner Personalization — Spring Sale Campaign

Generates ~50,000 user-level session records for an e-commerce A/B test
comparing a static promotional banner (Control A) against a personalized
browsing-history-driven banner (Variant B).

The dataset encodes a nuanced outcome:
  - Variant B wins overall on conversion (+0.4pp)
  - But the lift is concentrated in returning/desktop/email segments
  - New mobile visitors actually convert WORSE on Variant B
  - AOV dips slightly for B (more low-price conversions), but RPV nets positive
  - ~2% of users saw both variants (cookie deletion), flagged for exclusion

Author: Brendan Hoffman
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os

# ──────────────────────────────────────────────────────────────
# CONFIG — All test parameters in one place
# ──────────────────────────────────────────────────────────────

SEED = 42
TOTAL_VISITORS = 50_000
TEST_START = datetime(2025, 3, 15)
TEST_END = datetime(2025, 3, 28)
CROSS_CONTAMINATION_RATE = 0.02  # ~2% saw both variants

# Segment distributions
DEVICE_DIST = {"mobile": 0.58, "desktop": 0.34, "tablet": 0.08}
VISITOR_TYPE_DIST = {"new": 0.60, "returning": 0.40}
TRAFFIC_SOURCE_DIST = {
    "organic": 0.30,
    "paid_search": 0.25,
    "email": 0.20,
    "social": 0.15,
    "direct": 0.10,
}
REGION_DIST = {
    "northeast": 0.28,
    "southeast": 0.24,
    "midwest": 0.22,
    "west": 0.26,
}

# ──────────────────────────────────────────────────────────────
# CONTROL (A) BASELINE RATES
# ──────────────────────────────────────────────────────────────

# Note: Base conversion rate is set higher than the 3.2% target because
# bounced users (38%) are forced to converted=False downstream.
# Effective rate ≈ BASE_RATE × (1 - BOUNCE_RATE) ≈ 0.052 × 0.62 ≈ 3.2%
CONTROL_CONVERSION_RATE = 0.052       # pre-bounce effective → ~3.2%
CONTROL_CTR = 0.195                   # pre-bounce effective → ~12%
CONTROL_BOUNCE_RATE = 0.38            # 38.0%
CONTROL_AOV_MEAN = 85.0              # $85
CONTROL_AOV_STD = 32.0               # standard deviation for order values

# ──────────────────────────────────────────────────────────────
# VARIANT (B) EFFECT MODIFIERS
# These define how Variant B shifts each metric relative to Control,
# broken out by segment. This is where the "nuanced story" lives.
# ──────────────────────────────────────────────────────────────

# Conversion rate ADDITIVE modifiers (in percentage points)
# Scaled up to account for bounce suppression downstream
VARIANT_B_CONVERSION_EFFECTS = {
    # Device × Visitor Type interactions (the core story)
    ("desktop", "returning"):  +0.018,    # effective → ~+1.1pp
    ("desktop", "new"):        +0.005,    # effective → ~+0.3pp
    ("mobile", "returning"):   +0.007,    # effective → ~+0.4pp
    ("mobile", "new"):         -0.005,    # effective → ~-0.3pp (NEGATIVE)
    ("tablet", "returning"):   +0.010,    # effective → ~+0.6pp
    ("tablet", "new"):         +0.002,    # effective → ~+0.1pp
}

# Traffic source conversion modifiers (additive, stacks with above)
VARIANT_B_SOURCE_EFFECTS = {
    "email":       +0.010,    # effective → ~+0.6pp — warm users
    "organic":     +0.003,    # effective → ~+0.2pp
    "paid_search": +0.002,    # effective → ~+0.1pp
    "social":      +0.000,    # zero — top-of-funnel browsers
    "direct":      +0.005,    # effective → ~+0.3pp
}

# CTR modifiers for Variant B (scaled for bounce suppression)
VARIANT_B_CTR_EFFECT = +0.040         # effective → ~+2.5pp overall
VARIANT_B_CTR_MOBILE_NEW_EFFECT = +0.013  # effective → weaker on mobile new

# Bounce rate modifiers for Variant B
VARIANT_B_BOUNCE_EFFECT = -0.015      # -1.5pp overall
VARIANT_B_BOUNCE_MOBILE_NEW_EFFECT = +0.012  # INCREASES for mobile new

# AOV modifier for Variant B (multiplicative — variant nudges cheaper purchases)
VARIANT_B_AOV_MULTIPLIER = 0.953      # ~4.7% lower AOV

# Precomputed log-normal parameters (derived from CONTROL_AOV_* constants)
_AOV_MU = np.log(CONTROL_AOV_MEAN) - 0.5 * (CONTROL_AOV_STD / CONTROL_AOV_MEAN) ** 2
_AOV_SIGMA = np.sqrt(np.log(1 + (CONTROL_AOV_STD / CONTROL_AOV_MEAN) ** 2))

# Hour-of-day traffic weights indexed 0–23: peak 10am–9pm, trough 1–6am
_HOUR_WEIGHTS = np.array([
    0.3, 0.15, 0.1, 0.08, 0.08, 0.1,
    0.2, 0.35, 0.55, 0.75, 0.9, 0.95,
    1.0, 0.95, 0.9, 0.85, 0.88, 0.9,
    0.95, 1.0, 0.95, 0.85, 0.65, 0.45,
])

_PEAK_HOURS = np.array([10, 11, 12, 13, 14, 18, 19, 20])


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def weighted_choice(rng, options_dict, size=1):
    """Draw from a weighted categorical distribution."""
    categories = list(options_dict.keys())
    probabilities = list(options_dict.values())
    return rng.choice(categories, size=size, p=probabilities)


def generate_timestamp(rng, start, end, size):
    """Generate random timestamps within the test window with realistic
    daily traffic patterns. Fully vectorized — no Python-level loop."""
    total_seconds = int((end - start).total_seconds())
    offsets = rng.integers(0, total_seconds, size=size)

    hours = (offsets % 86400) // 3600
    accept = rng.random(size) < _HOUR_WEIGHTS[hours]

    # Rejected sessions are reassigned to a random peak hour on the same day
    day_offsets = (offsets // 86400) * 86400
    replacement_hours = rng.choice(_PEAK_HOURS, size=size)
    replacement_minutes = rng.integers(0, 60, size=size)
    rejected_offsets = day_offsets + replacement_hours * 3600 + replacement_minutes * 60

    final_offsets = np.where(accept, offsets, rejected_offsets)
    return [start + timedelta(seconds=int(s)) for s in final_offsets]


def generate_order_value(rng, size):
    """Generate log-normal order values clipped to a realistic range.
    Parameters are precomputed at module level from CONTROL_AOV_* constants."""
    values = rng.lognormal(mean=_AOV_MU, sigma=_AOV_SIGMA, size=size)
    return np.round(np.clip(values, 12.0, 650.0), 2)


# ──────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ──────────────────────────────────────────────────────────────

def generate_ab_test_data(seed=SEED, total_visitors=TOTAL_VISITORS, output_path=None):
    """Generate the full synthetic A/B test dataset.

    Returns a pandas DataFrame with one row per visitor session.
    """
    rng = np.random.default_rng(seed)
    print(f"Generating {total_visitors:,} visitor sessions...")

    # ── Step 1: Assign user IDs and segments ──

    user_ids = [f"usr_{str(i).zfill(5)}" for i in range(total_visitors)]

    devices = weighted_choice(rng, DEVICE_DIST, size=total_visitors)
    visitor_types = weighted_choice(rng, VISITOR_TYPE_DIST, size=total_visitors)
    traffic_sources = weighted_choice(rng, TRAFFIC_SOURCE_DIST, size=total_visitors)
    regions = weighted_choice(rng, REGION_DIST, size=total_visitors)

    # ── Step 2: Assign variants (50/50 split) ──

    variants = rng.choice(["A", "B"], size=total_visitors, p=[0.5, 0.5])
    is_b = variants == "B"

    # ── Step 3: Flag cross-contaminated users (~2%) ──

    n_contaminated = int(total_visitors * CROSS_CONTAMINATION_RATE)
    contaminated_indices = rng.choice(total_visitors, size=n_contaminated, replace=False)
    saw_both_variants = np.zeros(total_visitors, dtype=bool)
    saw_both_variants[contaminated_indices] = True

    # ── Step 4: Generate timestamps ──

    timestamps = generate_timestamp(rng, TEST_START, TEST_END, total_visitors)

    # ── Step 5: Compute per-user metric probabilities (vectorized) ──

    mobile_new = (devices == "mobile") & (visitor_types == "new")

    # --- Conversion probabilities ---
    conv_probs = np.full(total_visitors, CONTROL_CONVERSION_RATE)
    for (dev, vtype), effect in VARIANT_B_CONVERSION_EFFECTS.items():
        conv_probs[is_b & (devices == dev) & (visitor_types == vtype)] += effect
    for source, effect in VARIANT_B_SOURCE_EFFECTS.items():
        conv_probs[is_b & (traffic_sources == source)] += effect
    conv_probs += rng.normal(0, 0.001, size=total_visitors)
    conv_probs = np.clip(conv_probs, 0.005, 0.15)
    converted = rng.random(total_visitors) < conv_probs

    # --- Banner CTR ---
    ctr_probs = np.full(total_visitors, CONTROL_CTR)
    ctr_probs[is_b & mobile_new] += VARIANT_B_CTR_MOBILE_NEW_EFFECT
    ctr_probs[is_b & ~mobile_new] += VARIANT_B_CTR_EFFECT
    ctr_probs += rng.normal(0, 0.003, size=total_visitors)
    ctr_probs = np.clip(ctr_probs, 0.02, 0.40)
    banner_clicked = rng.random(total_visitors) < ctr_probs

    # --- Bounce rate ---
    bounce_probs = np.full(total_visitors, CONTROL_BOUNCE_RATE)
    bounce_probs[is_b & mobile_new] += VARIANT_B_BOUNCE_MOBILE_NEW_EFFECT
    bounce_probs[is_b & ~mobile_new] += VARIANT_B_BOUNCE_EFFECT
    bounce_probs += rng.normal(0, 0.005, size=total_visitors)
    bounce_probs = np.clip(bounce_probs, 0.10, 0.70)
    bounced = rng.random(total_visitors) < bounce_probs

    # Bounced users didn't convert; 95% also didn't click the banner
    converted[bounced] = False
    banner_clicked[bounced & (rng.random(total_visitors) > 0.05)] = False

    # ── Step 6: Generate order values for converters ──

    converter_mask = converted
    n_converters = converter_mask.sum()
    converter_is_b = variants[converter_mask] == "B"

    order_values = np.full(total_visitors, np.nan)
    all_order_values = generate_order_value(rng, size=n_converters)
    all_order_values[converter_is_b] *= VARIANT_B_AOV_MULTIPLIER
    order_values[converter_mask] = all_order_values

    # ── Step 7: Assemble DataFrame ──

    df = pd.DataFrame({
        "user_id": user_ids,
        "timestamp": timestamps,
        "variant": variants,
        "device": devices,
        "visitor_type": visitor_types,
        "traffic_source": traffic_sources,
        "region": regions,
        "banner_clicked": banner_clicked,
        "bounced": bounced,
        "converted": converted,
        "order_value": order_values,
        "saw_both_variants": saw_both_variants,
    })

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Step 8: Print Summary Stats ──

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total visitors:        {len(df):,}")
    print(f"Date range:            {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"Cross-contaminated:    {df['saw_both_variants'].sum():,} ({df['saw_both_variants'].mean():.1%})")

    print(f"\n{'Variant':<12} {'Visitors':>10} {'Conversions':>13} {'Conv Rate':>11} {'Avg AOV':>10} {'RPV':>10}")
    print("-" * 68)

    for variant in ["A", "B"]:
        subset = df[df["variant"] == variant]
        n = len(subset)
        conversions = subset["converted"].sum()
        conv_rate = conversions / n
        avg_aov = subset.loc[subset["converted"], "order_value"].mean()
        rpv = subset["order_value"].fillna(0).sum() / n
        print(f"{variant:<12} {n:>10,} {conversions:>13,} {conv_rate:>10.2%} ${avg_aov:>9.2f} ${rpv:>9.2f}")

    print(f"\n{'Metric':<25} {'Control (A)':>13} {'Variant (B)':>13} {'Delta':>10}")
    print("-" * 63)

    a = df[df["variant"] == "A"]
    b = df[df["variant"] == "B"]

    metrics = {
        "Conversion Rate": (a["converted"].mean(), b["converted"].mean()),
        "Banner CTR": (a["banner_clicked"].mean(), b["banner_clicked"].mean()),
        "Bounce Rate": (a["bounced"].mean(), b["bounced"].mean()),
        "Avg Order Value": (
            a.loc[a["converted"], "order_value"].mean(),
            b.loc[b["converted"], "order_value"].mean(),
        ),
        "Revenue per Visitor": (
            a["order_value"].fillna(0).sum() / len(a),
            b["order_value"].fillna(0).sum() / len(b),
        ),
    }

    for metric_name, (val_a, val_b) in metrics.items():
        delta = val_b - val_a
        if "Rate" in metric_name or "CTR" in metric_name:
            print(f"{metric_name:<25} {val_a:>12.2%} {val_b:>12.2%} {delta:>+9.2%}")
        else:
            print(f"{metric_name:<25} ${val_a:>11.2f} ${val_b:>11.2f} ${delta:>+8.2f}")

    # Segment breakdown for the key story
    print(f"\nKEY SEGMENT BREAKDOWN — Conversion Rate")
    print("-" * 63)
    print(f"{'Segment':<30} {'Control':>10} {'Variant':>10} {'Delta':>10}")
    print("-" * 63)

    segment_pairs = [
        ("Desktop + Returning", (df["device"] == "desktop") & (df["visitor_type"] == "returning")),
        ("Desktop + New", (df["device"] == "desktop") & (df["visitor_type"] == "new")),
        ("Mobile + Returning", (df["device"] == "mobile") & (df["visitor_type"] == "returning")),
        ("Mobile + New", (df["device"] == "mobile") & (df["visitor_type"] == "new")),
    ]

    for label, mask in segment_pairs:
        seg_a = df[mask & (df["variant"] == "A")]
        seg_b = df[mask & (df["variant"] == "B")]
        rate_a = seg_a["converted"].mean()
        rate_b = seg_b["converted"].mean()
        delta = rate_b - rate_a
        print(f"{label:<30} {rate_a:>9.2%} {rate_b:>9.2%} {delta:>+9.2%}")

    print()

    # ── Step 9: Export ──

    if output_path:
        df.to_csv(output_path, index=False)
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Saved to: {output_path} ({file_size_mb:.1f} MB)")
        print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

    return df


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic A/B test data for Homepage Banner Personalization test"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_test_homepage_banner.csv"),
        help="Output CSV path (default: ab_test_homepage_banner.csv in script directory)"
    )
    parser.add_argument(
        "-n", "--visitors",
        type=int,
        default=TOTAL_VISITORS,
        help=f"Total number of visitors (default: {TOTAL_VISITORS:,})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})"
    )

    args = parser.parse_args()

    df = generate_ab_test_data(
        seed=args.seed,
        total_visitors=args.visitors,
        output_path=args.output,
    )
