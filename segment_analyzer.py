"""
segment_analyzer.py
===================
Segment-level A/B test analysis engine.

Slices the dataset by user dimensions (device, visitor type, traffic source,
region), runs statistical tests on each slice, and detects anomalies — segments
where the treatment effect reverses direction or loses significance relative
to the topline result.

Designed to feed:
  - Tab 3 (Segment Explorer) in the Streamlit app
  - Tab 4 (AI Summary) with structured anomaly data

Author: Brendan Hoffman
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from stats_engine import run_test, TestResult, METRIC_REGISTRY


# ──────────────────────────────────────────────────────────────
# SEGMENT DIMENSION REGISTRY
# ──────────────────────────────────────────────────────────────

SEGMENT_REGISTRY = {
    "device": {
        "display_name": "Device Type",
        "column": "device",
        "values": ["mobile", "desktop", "tablet"],
    },
    "visitor_type": {
        "display_name": "Visitor Type",
        "column": "visitor_type",
        "values": ["new", "returning"],
    },
    "traffic_source": {
        "display_name": "Traffic Source",
        "column": "traffic_source",
        "values": ["organic", "paid_search", "email", "social", "direct"],
    },
    "region": {
        "display_name": "Region",
        "column": "region",
        "values": ["northeast", "southeast", "midwest", "west"],
    },
}

# Compound segments — cross two dimensions for deeper analysis
COMPOUND_SEGMENTS = {
    "device_x_visitor": {
        "display_name": "Device × Visitor Type",
        "columns": ["device", "visitor_type"],
    },
}

_SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


# ──────────────────────────────────────────────────────────────
# RESULT DATACLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class SegmentSlice:
    """Test result for a single segment value."""

    segment_dimension: str
    segment_value: str
    n_control: int
    n_variant: int
    test_result: TestResult

    def to_dict(self) -> dict:
        d = {
            "segment_dimension": self.segment_dimension,
            "segment_value": self.segment_value,
            "n_control": self.n_control,
            "n_variant": self.n_variant,
        }
        d.update(self.test_result.to_dict())
        return d


@dataclass
class Anomaly:
    """A detected anomaly in the segment breakdown."""

    segment_dimension: str
    segment_value: str
    anomaly_type: str  # "direction_reversal", "significance_loss", "outsized_effect"
    severity: str  # "high", "medium", "low"
    description: str
    topline_direction: str
    segment_direction: str
    segment_p_value: float
    segment_lift: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentReport:
    """Complete segment analysis for one metric across one dimension."""

    metric_key: str
    metric_name: str
    segment_dimension: str
    segment_display_name: str
    topline_result: TestResult
    slices: list[SegmentSlice] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metric_key": self.metric_key,
            "metric_name": self.metric_name,
            "segment_dimension": self.segment_dimension,
            "segment_display_name": self.segment_display_name,
            "topline": self.topline_result.to_dict(),
            "slices": [s.to_dict() for s in self.slices],
            "anomalies": [a.to_dict() for a in self.anomalies],
        }

    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0

    def high_severity_anomalies(self) -> list[Anomaly]:
        return [a for a in self.anomalies if a.severity == "high"]


# ──────────────────────────────────────────────────────────────
# CORE SEGMENT ANALYSIS
# ──────────────────────────────────────────────────────────────

def analyze_segment(
    df: pd.DataFrame,
    metric_key: str,
    segment_key: str,
    confidence: float = 0.95,
    alpha: float = 0.05,
    min_segment_size: int = 100,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
    topline: Optional[TestResult] = None,
) -> SegmentReport:
    """Run a metric test across every value in a segment dimension.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared A/B test data (output of stats_engine.prepare_data).
    metric_key : str
        Key from METRIC_REGISTRY (e.g., 'conversion_rate').
    segment_key : str
        Key from SEGMENT_REGISTRY (e.g., 'device') or COMPOUND_SEGMENTS.
    confidence : float
        Confidence level for tests.
    alpha : float
        Significance threshold.
    min_segment_size : int
        Minimum visitors per variant in a segment to run the test.
        Segments below this threshold are skipped to avoid noisy results.
    variant_col : str
        Column identifying which variant each row belongs to.
    control_label : str
        Label for the control group in variant_col.
    variant_label : str
        Label for the treatment group in variant_col.

    Returns
    -------
    SegmentReport
        Complete analysis with per-segment results and detected anomalies.
    """
    if metric_key not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_key}")

    metric_meta = METRIC_REGISTRY[metric_key]

    # Run the topline test if not supplied by the caller
    if topline is None:
        topline = run_test(
            df, metric_key, confidence, alpha,
            variant_col=variant_col, control_label=control_label, variant_label=variant_label,
        )

    # Determine if this is a single or compound segment
    is_compound = segment_key in COMPOUND_SEGMENTS

    if is_compound:
        return _analyze_compound_segment(
            df, metric_key, segment_key, topline, confidence, alpha, min_segment_size,
            variant_col=variant_col, control_label=control_label, variant_label=variant_label,
        )

    if segment_key not in SEGMENT_REGISTRY:
        raise ValueError(
            f"Unknown segment: {segment_key}. "
            f"Available: {list(SEGMENT_REGISTRY.keys()) + list(COMPOUND_SEGMENTS.keys())}"
        )

    seg_meta = SEGMENT_REGISTRY[segment_key]
    column = seg_meta["column"]
    values = seg_meta["values"]

    if column not in df.columns:
        raise ValueError(
            f"Segment column '{column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    n_null = int(df[column].isna().sum())
    if n_null > 0:
        print(f"Warning: '{column}' has {n_null:,} NaN rows -- these will be excluded from all segment slices.")

    report = SegmentReport(
        metric_key=metric_key,
        metric_name=metric_meta["display_name"],
        segment_dimension=segment_key,
        segment_display_name=seg_meta["display_name"],
        topline_result=topline,
    )

    for value in values:
        df_slice = df[df[column] == value]
        n_a = int((df_slice[variant_col] == control_label).sum())
        n_b = int((df_slice[variant_col] == variant_label).sum())

        # Skip segments too small for reliable testing
        if n_a < min_segment_size or n_b < min_segment_size:
            continue

        try:
            result = run_test(
                df_slice, metric_key, confidence, alpha,
                variant_col=variant_col, control_label=control_label, variant_label=variant_label,
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: skipping segment '{segment_key}={value}' for '{metric_key}' ({type(e).__name__}: {e})")
            continue

        slice_obj = SegmentSlice(
            segment_dimension=segment_key,
            segment_value=value,
            n_control=n_a,
            n_variant=n_b,
            test_result=result,
        )
        report.slices.append(slice_obj)

    # Detect anomalies
    report.anomalies = _detect_anomalies(topline, report.slices)

    return report


def _analyze_compound_segment(
    df: pd.DataFrame,
    metric_key: str,
    segment_key: str,
    topline: TestResult,
    confidence: float,
    alpha: float,
    min_segment_size: int,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
) -> SegmentReport:
    """Handle compound (two-dimension) segments like device × visitor type."""

    seg_meta = COMPOUND_SEGMENTS[segment_key]
    columns = seg_meta["columns"]
    metric_meta = METRIC_REGISTRY[metric_key]

    report = SegmentReport(
        metric_key=metric_key,
        metric_name=metric_meta["display_name"],
        segment_dimension=segment_key,
        segment_display_name=seg_meta["display_name"],
        topline_result=topline,
    )

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Compound segment columns {missing} not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    for col in columns:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            print(f"Warning: '{col}' has {n_null:,} NaN rows -- these combinations will be excluded from compound segment slices.")

    # Get all unique combinations present in the data
    combos = df.groupby(columns).size().reset_index(name="_count")

    for _, row in combos.iterrows():
        label_parts = [str(row[col]).title() for col in columns]
        label = " + ".join(label_parts)
        df_slice = df
        for col in columns:
            df_slice = df_slice[df_slice[col] == row[col]]
        n_a = int((df_slice[variant_col] == control_label).sum())
        n_b = int((df_slice[variant_col] == variant_label).sum())

        if n_a < min_segment_size or n_b < min_segment_size:
            continue

        try:
            result = run_test(
                df_slice, metric_key, confidence, alpha,
                variant_col=variant_col, control_label=control_label, variant_label=variant_label,
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: skipping compound segment '{label}' for '{metric_key}' ({type(e).__name__}: {e})")
            continue

        slice_obj = SegmentSlice(
            segment_dimension=segment_key,
            segment_value=label,
            n_control=n_a,
            n_variant=n_b,
            test_result=result,
        )
        report.slices.append(slice_obj)

    report.anomalies = _detect_anomalies(topline, report.slices)
    return report


# ──────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ──────────────────────────────────────────────────────────────

def _detect_anomalies(
    topline: TestResult,
    slices: list[SegmentSlice],
) -> list[Anomaly]:
    """Scan segment results for anomalies relative to the topline.

    Detects three types:
      1. Direction reversal — segment effect goes opposite to topline
      2. Significance loss — topline is significant but segment is not
         (when segment is large enough to reasonably detect the effect)
      3. Outsized effect — segment lift is 3x+ the topline lift
    """
    anomalies = []

    for s in slices:
        result = s.test_result

        # ── Direction Reversal ──
        if (
            topline.direction != "neutral"
            and result.direction != "neutral"
            and topline.direction != result.direction
        ):
            # High severity if the reversal is itself significant
            severity = "high" if result.is_significant else "medium"

            anomalies.append(Anomaly(
                segment_dimension=s.segment_dimension,
                segment_value=s.segment_value,
                anomaly_type="direction_reversal",
                severity=severity,
                description=(
                    f"{s.segment_value.title()} shows a {result.direction} effect "
                    f"({result.absolute_lift:+.4f}) while the topline is "
                    f"{topline.direction} ({topline.absolute_lift:+.4f}). "
                    f"{'This reversal is statistically significant.' if result.is_significant else 'This reversal is not statistically significant but warrants investigation.'}"
                ),
                topline_direction=topline.direction,
                segment_direction=result.direction,
                segment_p_value=result.p_value,
                segment_lift=result.absolute_lift,
            ))

        # ── Significance Loss ──
        # Only flag if topline is significant AND the segment is large
        # enough (>20% of total traffic) that we'd expect to see the effect
        elif (
            topline.is_significant
            and not result.is_significant
            and (s.n_control + s.n_variant) > 0.20 * (topline.n_control + topline.n_variant)
        ):
            anomalies.append(Anomaly(
                segment_dimension=s.segment_dimension,
                segment_value=s.segment_value,
                anomaly_type="significance_loss",
                severity="low",
                description=(
                    f"{s.segment_value.title()} does not show a significant effect "
                    f"(p={result.p_value:.4f}) despite representing a substantial "
                    f"portion of traffic ({s.n_control + s.n_variant:,} visitors). "
                    f"The topline effect may not hold uniformly across this segment."
                ),
                topline_direction=topline.direction,
                segment_direction=result.direction,
                segment_p_value=result.p_value,
                segment_lift=result.absolute_lift,
            ))

        # ── Outsized Effect ──
        if (
            abs(topline.absolute_lift) > 1e-6
            and abs(result.absolute_lift) > abs(topline.absolute_lift) * 3
            and result.is_significant
        ):
            anomalies.append(Anomaly(
                segment_dimension=s.segment_dimension,
                segment_value=s.segment_value,
                anomaly_type="outsized_effect",
                severity="medium",
                description=(
                    f"{s.segment_value.title()} shows a disproportionately large "
                    f"effect ({result.absolute_lift:+.4f}) — "
                    f"{abs(result.absolute_lift / topline.absolute_lift):.1f}x the "
                    f"topline lift. This segment may be driving the overall result."
                ),
                topline_direction=topline.direction,
                segment_direction=result.direction,
                segment_p_value=result.p_value,
                segment_lift=result.absolute_lift,
            ))

    # Sort: high severity first, then medium, then low
    anomalies.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 3))

    return anomalies


# ──────────────────────────────────────────────────────────────
# FULL SCAN — ALL DIMENSIONS FOR A METRIC
# ──────────────────────────────────────────────────────────────

def full_segment_scan(
    df: pd.DataFrame,
    metric_key: str,
    confidence: float = 0.95,
    alpha: float = 0.05,
    include_compound: bool = True,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
    topline: Optional[TestResult] = None,
) -> list[SegmentReport]:
    """Run segment analysis across all registered dimensions for a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared A/B test data.
    metric_key : str
        Metric to analyze.
    include_compound : bool
        Whether to include compound segments (e.g., device × visitor type).
    variant_col, control_label, variant_label
        Passed through to analyze_segment — see analyze_segment docstring.

    Returns
    -------
    list[SegmentReport]
        One report per segment dimension.
    """
    # Use pre-computed topline if provided, otherwise compute it once
    # and share it across all dimension scans
    shared_topline = topline or run_test(
        df, metric_key, confidence, alpha,
        variant_col=variant_col, control_label=control_label, variant_label=variant_label,
    )

    reports = []

    for seg_key in SEGMENT_REGISTRY:
        report = analyze_segment(
            df, metric_key, seg_key, confidence, alpha,
            variant_col=variant_col, control_label=control_label, variant_label=variant_label,
            topline=shared_topline,
        )
        reports.append(report)

    if include_compound:
        for seg_key in COMPOUND_SEGMENTS:
            report = analyze_segment(
                df, metric_key, seg_key, confidence, alpha,
                variant_col=variant_col, control_label=control_label, variant_label=variant_label,
                topline=shared_topline,
            )
            reports.append(report)

    return reports


def full_scan_all_metrics(
    df: pd.DataFrame,
    confidence: float = 0.95,
    alpha: float = 0.05,
    include_compound: bool = True,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
    topline_results: Optional[dict] = None,
) -> dict[str, list[SegmentReport]]:
    """Run segment analysis for every metric across every dimension.

    Parameters
    ----------
    topline_results : dict, optional
        Pre-computed topline TestResults keyed by metric_key (e.g. from
        run_all_tests). When provided, avoids re-running the full-dataset
        test for each metric inside full_segment_scan.

    Returns
    -------
    dict[str, list[SegmentReport]]
        Keyed by metric_key. Each value is a list of SegmentReports
        (one per dimension).
    """
    all_reports = {}
    for metric_key in METRIC_REGISTRY:
        precomputed = (topline_results or {}).get(metric_key)
        try:
            all_reports[metric_key] = full_segment_scan(
                df, metric_key, confidence, alpha, include_compound,
                variant_col=variant_col, control_label=control_label, variant_label=variant_label,
                topline=precomputed,
            )
        except (ValueError, KeyError) as e:
            print(f"Warning: skipping segment scan for '{metric_key}' ({type(e).__name__}: {e})")
    return all_reports


# ──────────────────────────────────────────────────────────────
# SUMMARY HELPERS (for AI summary consumption)
# ──────────────────────────────────────────────────────────────

def collect_all_anomalies(
    reports: list[SegmentReport],
) -> list[Anomaly]:
    """Flatten anomalies from multiple reports into a single sorted list."""
    all_anomalies = []
    for report in reports:
        all_anomalies.extend(report.anomalies)

    all_anomalies.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 3))
    return all_anomalies


def build_summary_payload(
    df: pd.DataFrame,
    confidence: float = 0.95,
    alpha: float = 0.05,
    variant_col: str = "variant",
    control_label: str = "A",
    variant_label: str = "B",
) -> dict:
    """Build a complete JSON-serializable payload for the AI summary.

    This is the structured input that ai_summary.py will send to the
    Claude API. Contains topline results for all metrics, segment
    breakdowns, and all detected anomalies.

    Returns
    -------
    dict
        Ready to be serialized to JSON for the Claude API prompt.
    """
    from stats_engine import run_all_tests, get_data_summary

    payload = {
        "test_summary": get_data_summary(df, variant_col=variant_col, control_label=control_label, variant_label=variant_label),
        "topline_results": {},
        "segment_reports": {},
        "all_anomalies": [],
    }

    # Topline results
    topline_results = run_all_tests(
        df, confidence, alpha,
        variant_col=variant_col, control_label=control_label, variant_label=variant_label,
    )
    for key, result in topline_results.items():
        payload["topline_results"][key] = result.to_dict()

    # Segment reports — pass topline_results so each metric's full-dataset
    # test is not recomputed inside full_segment_scan
    all_reports = full_scan_all_metrics(
        df, confidence, alpha, include_compound=True,
        variant_col=variant_col, control_label=control_label, variant_label=variant_label,
        topline_results=topline_results,
    )
    all_anomalies = []

    for metric_key, reports in all_reports.items():
        payload["segment_reports"][metric_key] = []
        for report in reports:
            payload["segment_reports"][metric_key].append(report.to_dict())
            all_anomalies.extend(report.anomalies)

    # Deduplicate and sort anomalies
    seen = set()
    unique_anomalies = []
    for a in all_anomalies:
        key = (a.segment_dimension, a.segment_value, a.anomaly_type)
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    unique_anomalies.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 3))
    payload["all_anomalies"] = [a.to_dict() for a in unique_anomalies]

    return payload


# ──────────────────────────────────────────────────────────────
# CLI SELF-TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from stats_engine import prepare_data

    print("Loading dataset...")
    raw_df = pd.read_csv(Path(__file__).parent / "ab_test_homepage_banner.csv")
    df = prepare_data(raw_df, exclude_contaminated=True)

    # ── Single dimension scan ──
    print("\n" + "=" * 60)
    print("SEGMENT ANALYSIS: Conversion Rate × Device Type")
    print("=" * 60)

    report = analyze_segment(df, "conversion_rate", "device")
    print(f"\nTopline: {report.topline_result.summary_line()}")

    for s in report.slices:
        r = s.test_result
        sig_marker = "[sig]" if r.is_significant else "[ns]"
        print(
            f"  {s.segment_value:<12} "
            f"A={r.value_control:.2%}  B={r.value_variant:.2%}  "
            f"d={r.absolute_lift:+.2%}  p={r.p_value:.4f} {sig_marker}  "
            f"(n={s.n_control + s.n_variant:,})"
        )

    if report.anomalies:
        print(f"\n  Anomalies detected: {len(report.anomalies)}")
        for a in report.anomalies:
            print(f"  [{a.severity.upper()}] {a.anomaly_type}: {a.description}")

    # ── Compound segment ──
    print("\n" + "=" * 60)
    print("SEGMENT ANALYSIS: Conversion Rate × Device × Visitor Type")
    print("=" * 60)

    report_compound = analyze_segment(df, "conversion_rate", "device_x_visitor")
    print(f"\nTopline: {report_compound.topline_result.summary_line()}")

    for s in report_compound.slices:
        r = s.test_result
        sig_marker = "[sig]" if r.is_significant else "[ns]"
        print(
            f"  {s.segment_value:<25} "
            f"A={r.value_control:.2%}  B={r.value_variant:.2%}  "
            f"d={r.absolute_lift:+.2%}  p={r.p_value:.4f} {sig_marker}  "
            f"(n={s.n_control + s.n_variant:,})"
        )

    if report_compound.anomalies:
        print(f"\n  Anomalies detected: {len(report_compound.anomalies)}")
        for a in report_compound.anomalies:
            print(f"  [{a.severity.upper()}] {a.anomaly_type}: {a.description}")

    # ── Traffic source scan ──
    print("\n" + "=" * 60)
    print("SEGMENT ANALYSIS: Conversion Rate × Traffic Source")
    print("=" * 60)

    report_source = analyze_segment(df, "conversion_rate", "traffic_source")
    print(f"\nTopline: {report_source.topline_result.summary_line()}")

    for s in report_source.slices:
        r = s.test_result
        sig_marker = "[sig]" if r.is_significant else "[ns]"
        print(
            f"  {s.segment_value:<15} "
            f"A={r.value_control:.2%}  B={r.value_variant:.2%}  "
            f"d={r.absolute_lift:+.2%}  p={r.p_value:.4f} {sig_marker}  "
            f"(n={s.n_control + s.n_variant:,})"
        )

    if report_source.anomalies:
        print(f"\n  Anomalies detected: {len(report_source.anomalies)}")
        for a in report_source.anomalies:
            print(f"  [{a.severity.upper()}] {a.anomaly_type}: {a.description}")

    # ── Full anomaly summary ──
    print("\n" + "=" * 60)
    print("ALL ANOMALIES — Conversion Rate (all dimensions)")
    print("=" * 60)

    all_reports = full_segment_scan(df, "conversion_rate", include_compound=True)
    all_anomalies = collect_all_anomalies(all_reports)

    if all_anomalies:
        for a in all_anomalies:
            print(f"\n  [{a.severity.upper()}] {a.anomaly_type}")
            print(f"  Segment: {a.segment_dimension} = {a.segment_value}")
            print(f"  {a.description}")
    else:
        print("  No anomalies detected.")

    # ── Verify summary payload builds ──
    print("\n" + "=" * 60)
    print("SUMMARY PAYLOAD TEST")
    print("=" * 60)

    payload = build_summary_payload(df)
    print(f"\n  Topline metrics: {len(payload['topline_results'])}")
    print(f"  Segment dimensions analyzed: {sum(len(v) for v in payload['segment_reports'].values())}")
    print(f"  Total unique anomalies: {len(payload['all_anomalies'])}")

    for a in payload["all_anomalies"]:
        print(f"    [{a['severity'].upper()}] {a['segment_value']}: {a['anomaly_type']}")
