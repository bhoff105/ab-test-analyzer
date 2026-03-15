"""
stats_engine.py
===============
Statistical testing engine for A/B test analysis.

Handles:
  - Two-proportion z-test (conversion rate, CTR, bounce rate)
  - Welch's t-test (AOV, RPV)
  - Confidence intervals for observed lift
  - Statistical power calculation
  - Effect size estimation (Cohen's h for proportions, Cohen's d for means)

All functions return structured dictionaries so downstream consumers
(Streamlit UI, AI summary builder) have a consistent interface.

Author: Brendan Hoffman
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Literal, Optional


# ──────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Structured output from any statistical test."""

    metric_name: str
    metric_type: Literal["proportion", "continuous"]
    test_used: str

    # Sample sizes
    n_control: int
    n_variant: int

    # Observed values
    value_control: float
    value_variant: float

    # Lift
    absolute_lift: float
    relative_lift_pct: float

    # Statistical test output
    test_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # Verdict
    is_significant: bool
    verdict: Literal["significant", "not_significant", "borderline"]
    direction: Literal["positive", "negative", "neutral"]

    # Power analysis
    observed_power: float
    effect_size: float
    min_sample_needed: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        """One-line plain-English summary for logging or display."""
        dir_word = "increased" if self.direction == "positive" else "decreased"
        if self.direction == "neutral":
            dir_word = "showed no change in"
        return (
            f"{self.metric_name}: Variant B {dir_word} by "
            f"{abs(self.absolute_lift):.4f} ({self.relative_lift_pct:+.2f}%) -- "
            f"p={self.p_value:.4f} [{self.verdict}]"
        )


# ──────────────────────────────────────────────────────────────
# PROPORTION TEST (z-test for two independent proportions)
# Used for: conversion rate, CTR, bounce rate
# ──────────────────────────────────────────────────────────────

def _two_proportion_z_test(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    confidence: float = 0.95,
) -> dict:
    """Run a two-sided two-proportion z-test.

    Uses the pooled proportion for the standard error under H0,
    and the unpooled proportion for the confidence interval
    (Agresti-Caffo style adjustment for stability).
    """
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_b - p_a

    # Pooled proportion under H0: p_a = p_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    # Test statistic
    z_stat = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval (unpooled SE for the CI, not the test)
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_lower = diff - z_crit * se_unpooled
    ci_upper = diff + z_crit * se_unpooled

    return {
        "test_statistic": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se_unpooled,
    }


def _cohens_h(p_a: float, p_b: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))


def _power_from_ncp(ncp: float, alpha: float = 0.05) -> float:
    """Compute statistical power from a non-centrality parameter using
    normal approximation (shared by both proportion and t-test power functions)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    return float(np.clip(power, 0, 1))


def _power_proportion(
    p_a: float, effect_size_h: float, n: int, alpha: float = 0.05
) -> float:
    """Compute statistical power for a two-proportion z-test
    using Cohen's h and normal approximation."""
    ncp = abs(effect_size_h) * np.sqrt(n / 2)
    return _power_from_ncp(ncp, alpha)


def _min_sample_proportion(
    p_a: float, effect_size_h: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Minimum sample size per variant to detect the given effect."""
    if abs(effect_size_h) < 1e-8:
        return None
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n_per_arm = ((z_alpha + z_beta) / effect_size_h) ** 2 * 2
    return int(np.ceil(n_per_arm))


# ──────────────────────────────────────────────────────────────
# CONTINUOUS TEST (Welch's t-test)
# Used for: AOV, RPV
# ──────────────────────────────────────────────────────────────

def _welch_t_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Run a two-sided Welch's t-test for unequal variances."""
    n_a, n_b = len(values_a), len(values_b)
    mean_a, mean_b = values_a.mean(), values_b.mean()
    var_a, var_b = values_a.var(ddof=1), values_b.var(ddof=1)
    diff = mean_b - mean_a

    # Welch-Satterthwaite degrees of freedom
    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return {
            "test_statistic": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "se": 0.0,
        }

    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else 1

    t_stat = diff / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    return {
        "test_statistic": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
    }


def _cohens_d(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Cohen's d effect size for two independent samples."""
    n_a, n_b = len(values_a), len(values_b)
    var_a, var_b = values_a.var(ddof=1), values_b.var(ddof=1)
    # Pooled standard deviation
    s_pool = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if s_pool < 1e-12:
        return 0.0
    return float((values_b.mean() - values_a.mean()) / s_pool)


def _power_t_test(
    effect_d: float, n_a: int, n_b: int, alpha: float = 0.05
) -> float:
    """Approximate power for a two-sample t-test using normal approximation."""
    n_harm = 2 * (n_a * n_b) / (n_a + n_b)  # harmonic mean
    ncp = abs(effect_d) * np.sqrt(n_harm / 2)
    return _power_from_ncp(ncp, alpha)


def _min_sample_t_test(
    effect_d: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Minimum sample size per arm for a two-sample t-test."""
    if abs(effect_d) < 1e-8:
        return None
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n_per_arm = 2 * ((z_alpha + z_beta) / effect_d) ** 2
    return int(np.ceil(n_per_arm))


# ──────────────────────────────────────────────────────────────
# METRIC DEFINITIONS
# ──────────────────────────────────────────────────────────────

METRIC_REGISTRY = {
    "conversion_rate": {
        "display_name": "Purchase Conversion Rate",
        "type": "proportion",
        "success_col": "converted",
        "format": "percent",
    },
    "banner_ctr": {
        "display_name": "Banner Click-Through Rate",
        "type": "proportion",
        "success_col": "banner_clicked",
        "format": "percent",
    },
    "bounce_rate": {
        "display_name": "Bounce Rate",
        "type": "proportion",
        "success_col": "bounced",
        "format": "percent",
        "lower_is_better": True,
    },
    "aov": {
        "display_name": "Average Order Value",
        "type": "continuous",
        "value_col": "order_value",
        "filter_col": "converted",  # only include converters
        "format": "currency",
    },
    "rpv": {
        "display_name": "Revenue per Visitor",
        "type": "continuous",
        "value_col": "order_value",
        "fill_na": 0.0,  # non-converters contribute $0
        "format": "currency",
    },
}


# Reverse lookup: display_name → metric_key (used by format_lift)
_DISPLAY_NAME_TO_KEY = {v["display_name"]: k for k, v in METRIC_REGISTRY.items()}


# ──────────────────────────────────────────────────────────────
# SHARED VERDICT / DIRECTION HELPER
# ──────────────────────────────────────────────────────────────

def _compute_verdict_and_direction(
    p_val: float,
    absolute_lift: float,
    alpha: float,
    lower_is_better: bool = False,
) -> tuple[bool, str, str]:
    """Return (is_significant, verdict, direction) from test outputs."""
    if p_val < alpha:
        verdict, is_sig = "significant", True
    elif p_val < alpha * 2:
        verdict, is_sig = "borderline", False
    else:
        verdict, is_sig = "not_significant", False

    if abs(absolute_lift) < 1e-6:
        direction = "neutral"
    elif lower_is_better:
        direction = "positive" if absolute_lift < 0 else "negative"
    else:
        direction = "positive" if absolute_lift > 0 else "negative"

    return is_sig, verdict, direction


# ──────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ──────────────────────────────────────────────────────────────

def run_test(
    df: pd.DataFrame,
    metric_key: str,
    confidence: float = 0.95,
    alpha: float = 0.05,
    power_target: float = 0.80,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
    col_map: Optional[dict] = None,
) -> TestResult:
    """Run the appropriate statistical test for a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        The A/B test dataset.
    metric_key : str
        Key from METRIC_REGISTRY (e.g., 'conversion_rate', 'aov').
    confidence : float
        Confidence level for the interval (default 0.95).
    alpha : float
        Significance threshold (default 0.05).
    power_target : float
        Target power for minimum sample size calculation (default 0.80).
    variant_col : str
        Column name that identifies which variant each row belongs to (default 'variant').
    control_label : str
        Value in variant_col that denotes the control group (default 'A').
    variant_label : str
        Value in variant_col that denotes the treatment group (default 'B').
    col_map : dict, optional
        Per-metric column overrides. Keys are metric_keys; values are dicts that
        override any field in METRIC_REGISTRY for that metric.
        Example: {"conversion_rate": {"success_col": "purchased"}}

    Returns
    -------
    TestResult
        Structured result object with all test outputs.
    """
    if metric_key not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric_key}'. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )

    if variant_col not in df.columns:
        raise ValueError(
            f"Variant column '{variant_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    metric = {**METRIC_REGISTRY[metric_key], **(col_map or {}).get(metric_key, {})}
    df_a = df[df[variant_col] == control_label].copy()
    df_b = df[df[variant_col] == variant_label].copy()

    if metric["type"] == "proportion":
        return _run_proportion_test(
            df_a, df_b, metric_key, metric, confidence, alpha, power_target
        )
    else:
        return _run_continuous_test(
            df_a, df_b, metric_key, metric, confidence, alpha, power_target
        )


def _run_proportion_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric_key: str,
    metric: dict,
    confidence: float,
    alpha: float,
    power_target: float,
) -> TestResult:
    """Execute a proportion-based test."""
    col = metric["success_col"]

    if col not in df_a.columns:
        raise KeyError(f"Expected column '{col}' not found in dataset.")

    # Exclude NaN rows from both numerator and denominator
    n_a = int(df_a[col].notna().sum())
    n_b = int(df_b[col].notna().sum())

    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"One or both variant groups are empty after removing NaNs "
            f"(control={n_a}, variant={n_b}). "
            f"Check variant_col, control_label, and variant_label."
        )

    successes_a = int(df_a[col].sum())  # pandas sum() skips NaN by default
    successes_b = int(df_b[col].sum())
    p_a = successes_a / n_a
    p_b = successes_b / n_b

    # Run z-test
    result = _two_proportion_z_test(successes_a, n_a, successes_b, n_b, confidence)

    # Effect size and power
    h = _cohens_h(p_a, p_b)
    observed_power = _power_proportion(p_a, h, min(n_a, n_b), alpha)
    min_n = _min_sample_proportion(p_a, h, alpha, power_target)

    # Lift
    absolute_lift = p_b - p_a
    relative_lift = (p_b - p_a) / p_a * 100 if p_a > 0 else 0.0

    # Verdict + direction
    p_val = result["p_value"]
    is_sig, verdict, direction = _compute_verdict_and_direction(
        p_val, absolute_lift, alpha, lower_is_better=metric.get("lower_is_better", False)
    )

    return TestResult(
        metric_name=metric["display_name"],
        metric_type="proportion",
        test_used="Two-proportion z-test",
        n_control=n_a,
        n_variant=n_b,
        value_control=p_a,
        value_variant=p_b,
        absolute_lift=absolute_lift,
        relative_lift_pct=relative_lift,
        test_statistic=result["test_statistic"],
        p_value=p_val,
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        confidence_level=confidence,
        is_significant=is_sig,
        verdict=verdict,
        direction=direction,
        observed_power=observed_power,
        effect_size=h,
        min_sample_needed=min_n,
    )


def _run_continuous_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric_key: str,
    metric: dict,
    confidence: float,
    alpha: float,
    power_target: float,
) -> TestResult:
    """Execute a continuous metric test."""
    col = metric["value_col"]

    if col not in df_a.columns:
        raise KeyError(f"Expected column '{col}' not found in dataset.")

    # Apply filters (e.g., AOV only for converters)
    if "filter_col" in metric:
        df_a = df_a[df_a[metric["filter_col"]].astype(bool)]
        df_b = df_b[df_b[metric["filter_col"]].astype(bool)]

    # Apply fill (e.g., RPV fills NaN with 0 for non-converters)
    if "fill_na" in metric:
        values_a = df_a[col].fillna(metric["fill_na"]).values.astype(float)
        values_b = df_b[col].fillna(metric["fill_na"]).values.astype(float)
    else:
        values_a = df_a[col].dropna().values.astype(float)
        values_b = df_b[col].dropna().values.astype(float)

    n_a = len(values_a)
    n_b = len(values_b)

    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"One or both variant groups are empty after filtering/dropping NaNs "
            f"(control={n_a}, variant={n_b}). "
            f"Check variant_col, control_label, and variant_label."
        )
    if n_a < 2 or n_b < 2:
        raise ValueError(
            f"Welch's t-test requires at least 2 observations per group "
            f"(control={n_a}, variant={n_b})."
        )
    mean_a = values_a.mean()
    mean_b = values_b.mean()

    # Run Welch's t-test
    result = _welch_t_test(values_a, values_b, confidence)

    # Effect size and power
    d = _cohens_d(values_a, values_b)
    observed_power = _power_t_test(d, n_a, n_b, alpha)
    min_n = _min_sample_t_test(d, alpha, power_target)

    # Lift
    absolute_lift = mean_b - mean_a
    relative_lift = (mean_b - mean_a) / mean_a * 100 if abs(mean_a) > 1e-8 else 0.0

    # Verdict + direction
    p_val = result["p_value"]
    is_sig, verdict, direction = _compute_verdict_and_direction(p_val, absolute_lift, alpha)

    return TestResult(
        metric_name=metric["display_name"],
        metric_type="continuous",
        test_used="Welch's t-test",
        n_control=n_a,
        n_variant=n_b,
        value_control=mean_a,
        value_variant=mean_b,
        absolute_lift=absolute_lift,
        relative_lift_pct=relative_lift,
        test_statistic=result["test_statistic"],
        p_value=p_val,
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        confidence_level=confidence,
        is_significant=is_sig,
        verdict=verdict,
        direction=direction,
        observed_power=observed_power,
        effect_size=d,
        min_sample_needed=min_n,
    )


# ──────────────────────────────────────────────────────────────
# CONVENIENCE: RUN ALL METRICS
# ──────────────────────────────────────────────────────────────

def run_all_tests(
    df: pd.DataFrame,
    confidence: float = 0.95,
    alpha: float = 0.05,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
    col_map: Optional[dict] = None,
) -> dict[str, TestResult]:
    """Run tests on all registered metrics and return a dict of results.

    Parameters
    ----------
    variant_col, control_label, variant_label, col_map
        Passed through to run_test — see run_test docstring for details.

    Returns
    -------
    dict[str, TestResult]
        Keyed by metric_key from METRIC_REGISTRY.
    """
    results = {}
    for key in METRIC_REGISTRY:
        try:
            results[key] = run_test(
                df, key, confidence, alpha,
                variant_col=variant_col,
                control_label=control_label,
                variant_label=variant_label,
                col_map=col_map,
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: skipping '{key}' ({type(e).__name__}: {e})")
    return results


# ──────────────────────────────────────────────────────────────
# DATA PREPARATION HELPERS
# ──────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    exclude_contaminated: bool = True,
    bool_cols: Optional[list] = None,
    numeric_cols: Optional[list] = None,
    timestamp_col: str = "timestamp",
    contamination_col: str = "saw_both_variants",
) -> pd.DataFrame:
    """Clean and prepare the raw dataset for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Raw A/B test data.
    exclude_contaminated : bool
        If True, drop rows where contamination_col is True (default True).
    bool_cols : list, optional
        Columns to cast to bool. Defaults to the standard set:
        ['banner_clicked', 'bounced', 'converted', 'saw_both_variants'].
    numeric_cols : list, optional
        Columns to cast to numeric. Defaults to ['order_value'].
    timestamp_col : str
        Column to parse as datetime (default 'timestamp').
    contamination_col : str
        Column flagging cross-contaminated users (default 'saw_both_variants').

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for analysis.
    """
    df = df.copy()

    if bool_cols is None:
        bool_cols = ["banner_clicked", "bounced", "converted", "saw_both_variants"]
    if numeric_cols is None:
        numeric_cols = ["order_value"]

    existing_bool_cols = [c for c in bool_cols if c in df.columns]
    if existing_bool_cols:
        df[existing_bool_cols] = df[existing_bool_cols].astype(bool)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if exclude_contaminated and contamination_col in df.columns:
        n_before = len(df)
        df = df[~df[contamination_col]].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"Excluded {n_removed:,} cross-contaminated users.")

    return df


def get_data_summary(
    df: pd.DataFrame,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
    timestamp_col: str = "timestamp",
    contamination_col: str = "saw_both_variants",
) -> dict:
    """Generate a quick summary of the dataset for display.

    Returns a dict with keys: total_visitors, control_count, variant_count,
    date_start, date_end, test_duration_days, contaminated_count, contaminated_pct.
    """
    summary = {
        "total_visitors": len(df),
        "control_count": int((df[variant_col] == control_label).sum()),
        "variant_count": int((df[variant_col] == variant_label).sum()),
    }

    if timestamp_col in df.columns:
        summary["date_start"] = df[timestamp_col].min().strftime("%Y-%m-%d")
        summary["date_end"] = df[timestamp_col].max().strftime("%Y-%m-%d")
        summary["test_duration_days"] = (
            df[timestamp_col].max() - df[timestamp_col].min()
        ).days

    if contamination_col in df.columns:
        contaminated = df[contamination_col].sum()
        summary["contaminated_count"] = int(contaminated)
        summary["contaminated_pct"] = contaminated / len(df) * 100

    return summary


# ──────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ──────────────────────────────────────────────────────────────

def format_value(value: float, metric_key: str) -> str:
    """Format a metric value for display based on its type."""
    fmt = METRIC_REGISTRY.get(metric_key, {}).get("format", "number")
    if fmt == "percent":
        return f"{value:.2%}"
    elif fmt == "currency":
        return f"${value:.2f}"
    else:
        return f"{value:.4f}"


def format_lift(result: TestResult) -> str:
    """Format the lift for display."""
    key = _DISPLAY_NAME_TO_KEY.get(result.metric_name)
    fmt = METRIC_REGISTRY[key].get("format", "number") if key else "number"

    if fmt == "percent":
        return f"{result.absolute_lift:+.2%} ({result.relative_lift_pct:+.1f}%)"
    elif fmt == "currency":
        return f"${result.absolute_lift:+.2f} ({result.relative_lift_pct:+.1f}%)"
    else:
        return f"{result.absolute_lift:+.4f} ({result.relative_lift_pct:+.1f}%)"


# ──────────────────────────────────────────────────────────────
# CLI SELF-TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A/B test statistical analysis on a CSV file.")
    parser.add_argument("csv_file", nargs="?", default=None,
                        help="Path to the CSV file (default: ab_test_homepage_banner.csv in script directory)")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for intervals (default: 0.95)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold (default: 0.05)")
    parser.add_argument("--variant-col", default="variant",
                        help="Column identifying which variant each row belongs to (default: 'variant')")
    parser.add_argument("--control", default="A",
                        help="Label for the control group in variant-col (default: 'A')")
    parser.add_argument("--variant", default="B",
                        help="Label for the treatment group in variant-col (default: 'B')")
    parser.add_argument("--no-exclude-contaminated", action="store_true",
                        help="Keep cross-contaminated users instead of dropping them")
    args = parser.parse_args()

    csv_path = Path(args.csv_file) if args.csv_file else Path(__file__).parent / "ab_test_homepage_banner.csv"

    print(f"Loading dataset from: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    print("\n-- Preparing data --")
    df = prepare_data(raw_df, exclude_contaminated=not args.no_exclude_contaminated)

    summary = get_data_summary(
        df,
        variant_col=args.variant_col,
        control_label=args.control,
        variant_label=args.variant,
    )
    print(f"\nDataset: {summary['total_visitors']:,} visitors")
    if "date_start" in summary:
        print(f"Period:  {summary['date_start']} to {summary['date_end']}")
    print(f"Split:   {args.control}={summary['control_count']:,}  {args.variant}={summary['variant_count']:,}")

    print("\n-- Running all metric tests --\n")
    results = run_all_tests(
        df,
        confidence=args.confidence,
        alpha=args.alpha,
        variant_col=args.variant_col,
        control_label=args.control,
        variant_label=args.variant,
    )

    for key, result in results.items():
        print(result.summary_line())
        print(f"  CI ({result.confidence_level:.0%}): [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"  Power: {result.observed_power:.1%} | Effect size: {result.effect_size:.4f}")
        if result.min_sample_needed:
            print(f"  Min sample needed (80% power): {result.min_sample_needed:,} per arm")
        print()
