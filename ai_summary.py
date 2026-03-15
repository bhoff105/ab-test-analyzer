"""
ai_summary.py
=============
AI-powered executive summary generator for A/B test results.

Takes structured statistical output from stats_engine and segment_analyzer,
constructs a carefully engineered prompt, and calls the Claude API to
generate a stakeholder-ready narrative.

Prompt Engineering Techniques Used
----------------------------------
1. **Persona priming** — The system prompt establishes a concrete identity
   (senior analytics consultant) with specific communication norms, which
   anchors Claude's tone, vocabulary, and reasoning depth.

2. **Structured output scaffolding** — The prompt prescribes an exact section
   order with descriptions of what each section should contain, reducing
   ambiguity and ensuring the output is scannable by busy executives.

3. **Constraint layering** — Multiple constraints (word limit, no hedging,
   quantify everything, no methodology unless decision-relevant) work together
   to prevent common LLM failure modes: verbosity, vagueness, and excessive
   qualification.

4. **Chain-of-thought steering** — The user prompt is organized into labeled
   data blocks (TEST CONTEXT, TOPLINE RESULTS, SEGMENTS, ANOMALIES) so the
   model can reason over each data category sequentially rather than parsing
   a wall of text.

5. **Few-shot quality anchoring** — The system prompt includes an example
   snippet showing the exact caliber of language expected, giving the model
   a concrete quality target.

6. **Anti-pattern guardrails** — Explicit "do not" instructions target the
   most common failure modes for analytical summaries: burying the lead,
   restating numbers without interpretation, and offering vague next steps.

7. **Temperature calibration** — Set to 0.3 for analytical precision while
   retaining enough variability for natural prose.

Author: Brendan Hoffman
"""

import json
import os
from typing import Optional
from anthropic import Anthropic


# ──────────────────────────────────────────────────────────────
# CLIENT INITIALIZATION
# ──────────────────────────────────────────────────────────────

def _get_client() -> Anthropic:
    """Initialize the Anthropic client.

    Reads ANTHROPIC_API_KEY from environment. Raises a clear error
    if not set, since this is a common setup issue.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found in environment variables.\n"
            "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )
    return Anthropic(api_key=api_key)


# ──────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior analytics consultant at a top-tier firm. Your client is a \
VP of Marketing at an e-commerce company. You have five minutes of their \
attention to present A/B test results and a clear recommendation.

<communication_principles>
- Lead with the decision, not the data. The first sentence should tell \
the VP what to do.
- Every claim must be grounded in a specific number. Never say "significant \
improvement" without stating the magnitude.
- Translate statistical concepts into business impact. Instead of "p < 0.01", \
say "we're highly confident this is real — less than a 1% chance it's noise."
- Be direct about risks. Executives trust advisors who surface problems, not \
those who bury them.
- Next steps must be concrete enough that someone could act on them Monday \
morning — include who, what, and a suggested timeline.
</communication_principles>

<output_structure>
## Recommendation
One to two sentences. State whether to ship, hold, or ship conditionally. \
Include the single most important number that supports the decision.

## What We Found
Three to five bullet points. Each bullet should pair a metric result with \
its business interpretation. Lead each bullet with the business implication, \
then support it with the number.

## Where It Breaks Down
Surface the most important segment-level risks. For each risk, explain:
(a) which audience is affected, (b) the magnitude of the problem, and \
(c) why it likely happens. If the data suggests a plausible causal \
explanation, state it as a hypothesis.

## Recommended Next Steps
Two to four numbered actions. Each action should specify the team or role \
responsible and a timeframe. Prioritize by business impact.
</output_structure>

<quality_standards>
- Total length: 300-450 words. This is a briefing, not a report.
- Use markdown formatting: ## headers, **bold** for key figures, bullet points.
- Never restate raw numbers without interpreting them. Every number needs a \
"so what."
- Do not hedge with phrases like "it appears that" or "results seem to \
suggest." State your assessment directly, then qualify confidence where needed.
- Do not describe your own methodology or reasoning process.
- Write at the reading level of a Wall Street Journal article — precise, \
confident, accessible.
</quality_standards>

<example_calibration>
This is the caliber of language expected for the Recommendation section:

"Ship the personalized banner to returning desktop visitors immediately — \
they convert at 4.89% vs. 3.04% for the static banner, a 61% lift we're \
highly confident in. Hold off on new mobile visitors, where conversion \
actually drops, likely because the personalization engine has no browsing \
history to work with."
</example_calibration>"""


def _build_user_prompt(payload: dict) -> str:
    """Construct the user message from the analysis payload.

    The prompt is organized into clearly labeled data blocks so the model
    can reason over each category sequentially. This structured input
    reduces hallucination risk and improves factual grounding.
    """
    topline_section = _format_topline(payload["topline_results"])
    anomaly_section = _format_anomalies(payload["all_anomalies"])
    segment_section = _format_key_segments(payload["segment_reports"])

    summary = payload["test_summary"]
    test_context = (
        f"Company: Evergreen Home & Outdoor (mid-market DTC e-commerce, ~$45M annual revenue)\n"
        f"Test: Homepage Banner Personalization — Spring Sale Campaign\n"
        f"Control (A): Static promotional banner — 'Spring Sale: Up to 40% Off'\n"
        f"Variant (B): Personalized banner showing products from visitor's browsing history\n"
        f"Fallback: First-time visitors with no history see the static banner\n"
        f"Duration: {summary.get('test_duration_days', 'N/A')} days "
        f"({summary.get('date_start', '?')} to {summary.get('date_end', '?')})\n"
        f"Total visitors analyzed: {summary.get('total_visitors', 0):,}\n"
        f"Split: Control={summary.get('control_count', 0):,} / Variant={summary.get('variant_count', 0):,}\n"
        f"Confidence level: 95% (alpha = 0.05)"
    )

    prompt = f"""\
Analyze the following A/B test results and generate an executive summary \
for the VP of Marketing. Follow the output structure exactly.

<test_context>
{test_context}
</test_context>

<topline_results>
{topline_section}
</topline_results>

<segment_breakdowns>
{segment_section}
</segment_breakdowns>

<detected_anomalies>
{anomaly_section}
</detected_anomalies>

Important: Ground every claim in the data above. Do not invent numbers or \
cite metrics not present in the results. If a segment shows a reversal, \
explain the likely business reason based on the test context (e.g., the \
personalization engine lacks data for first-time visitors)."""

    return prompt


def _format_topline(topline_results: dict) -> str:
    """Format topline results into readable text for the prompt."""
    lines = []

    metric_format = {
        "conversion_rate": ("Purchase Conversion Rate", "percent"),
        "banner_ctr": ("Banner Click-Through Rate", "percent"),
        "bounce_rate": ("Bounce Rate", "percent"),
        "aov": ("Average Order Value", "currency"),
        "rpv": ("Revenue per Visitor", "currency"),
    }

    for key, (display_name, fmt) in metric_format.items():
        if key not in topline_results:
            continue
        r = topline_results[key]

        if fmt == "percent":
            val_a = f"{r['value_control']:.2%}"
            val_b = f"{r['value_variant']:.2%}"
            lift = f"{r['absolute_lift']:+.2%}"
        else:
            val_a = f"${r['value_control']:.2f}"
            val_b = f"${r['value_variant']:.2f}"
            lift = f"${r['absolute_lift']:+.2f}"

        rel_lift = f"{r['relative_lift_pct']:+.1f}%"
        p_val = f"p={r['p_value']:.4f}"
        verdict = r['verdict'].upper()
        power = f"{r['observed_power']:.0%} power"
        effect = f"effect_size={r['effect_size']:.4f}"

        lines.append(
            f"- {display_name}: Control={val_a}, Variant={val_b}, "
            f"Lift={lift} ({rel_lift}), {p_val} [{verdict}], {power}, {effect}"
        )

    return "\n".join(lines)


def _format_anomalies(anomalies: list[dict]) -> str:
    """Format anomalies, prioritizing high and medium severity."""
    if not anomalies:
        return "No significant anomalies detected."

    important = [a for a in anomalies if a["severity"] in ("high", "medium")]

    if not important:
        return "Only minor anomalies detected (low severity). No major segment-level concerns."

    lines = []
    for a in important[:10]:
        lines.append(
            f"- [{a['severity'].upper()}] {a['anomaly_type'].replace('_', ' ').title()}: "
            f"{a['description']}"
        )

    remaining = len(important) - 10
    if remaining > 0:
        lines.append(f"  ... and {remaining} additional medium/high severity anomalies.")

    return "\n".join(lines)


def _format_key_segments(segment_reports: dict) -> str:
    """Format the conversion rate segment breakdown — the primary story."""
    lines = []

    if "conversion_rate" not in segment_reports:
        return "No segment data available for the primary metric."

    for report in segment_reports["conversion_rate"]:
        dim_name = report["segment_display_name"]
        lines.append(f"\n{dim_name}:")

        for s in report["slices"]:
            r = s
            val_a = f"{r['value_control']:.2%}"
            val_b = f"{r['value_variant']:.2%}"
            lift = f"{r['absolute_lift']:+.2%}"
            rel_lift = f"{r['relative_lift_pct']:+.1f}%"
            p_val = f"p={r['p_value']:.4f}"
            sig = "[SIGNIFICANT]" if r['is_significant'] else "[not significant]"
            n_total = r['n_control'] + r['n_variant']
            direction = r.get('direction', 'n/a')

            lines.append(
                f"  {s['segment_value']:<25} Control={val_a}  Variant={val_b}  "
                f"Lift={lift} ({rel_lift})  {p_val} {sig}  direction={direction}  (n={n_total:,})"
            )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# API CALL
# ──────────────────────────────────────────────────────────────

def generate_summary(
    payload: dict,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Generate an executive summary from the analysis payload.

    Parameters
    ----------
    payload : dict
        Output of segment_analyzer.build_summary_payload().
    model : str
        Claude model to use.
    max_tokens : int
        Maximum response length.
    temperature : float
        Lower = more deterministic. 0.3 balances analytical precision
        with natural prose variation.

    Returns
    -------
    str
        Markdown-formatted executive summary.
    """
    client = _get_client()

    user_prompt = _build_user_prompt(payload)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    summary = ""
    for block in message.content:
        if block.type == "text":
            summary += block.text

    return summary


def generate_summary_streaming(
    payload: dict,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
    temperature: float = 0.3,
):
    """Generate an executive summary with streaming output.

    Yields text chunks as they arrive — used by the Streamlit UI
    to display the summary progressively.

    Yields
    ------
    str
        Text chunks as they stream from the API.
    """
    client = _get_client()

    user_prompt = _build_user_prompt(payload)

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ──────────────────────────────────────────────────────────────
# PROMPT PREVIEW (for development and debugging)
# ──────────────────────────────────────────────────────────────

def preview_prompt(payload: dict) -> tuple[str, str]:
    """Return the system and user prompts without calling the API.

    Useful for debugging prompt construction and for showing
    the prompt in the Streamlit UI as a "see what the AI sees" feature.

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    return SYSTEM_PROMPT, _build_user_prompt(payload)


# ──────────────────────────────────────────────────────────────
# PROMPT TECHNIQUE DESCRIPTIONS (for UI display)
# ──────────────────────────────────────────────────────────────

PROMPT_TECHNIQUES = [
    {
        "name": "Persona Priming",
        "description": (
            "The system prompt establishes a specific professional identity — "
            "a senior analytics consultant at a top-tier firm presenting to a "
            "VP of Marketing. This anchors the model's tone, vocabulary level, "
            "and reasoning depth to match what a real executive audience expects."
        ),
        "location": "System prompt, opening paragraph",
    },
    {
        "name": "Structured Output Scaffolding",
        "description": (
            "The prompt prescribes an exact four-section output structure "
            "(Recommendation, What We Found, Where It Breaks Down, Next Steps) "
            "with descriptions of each section's purpose. This eliminates "
            "ambiguity about format and ensures the output is scannable."
        ),
        "location": "System prompt, <output_structure> block",
    },
    {
        "name": "Constraint Layering",
        "description": (
            "Multiple reinforcing constraints work together to prevent common "
            "LLM failure modes. Word limits prevent verbosity. 'No hedging' "
            "rules prevent vagueness. 'Quantify everything' prevents hand-waving. "
            "Each constraint targets a specific failure pattern."
        ),
        "location": "System prompt, <quality_standards> block",
    },
    {
        "name": "XML-Tagged Data Blocks",
        "description": (
            "The user prompt organizes input data into clearly labeled XML blocks "
            "(<test_context>, <topline_results>, <segment_breakdowns>, "
            "<detected_anomalies>). This lets the model reason over each data "
            "category sequentially, reducing hallucination and improving "
            "factual grounding."
        ),
        "location": "User prompt, data sections",
    },
    {
        "name": "Few-Shot Quality Anchoring",
        "description": (
            "The system prompt includes a concrete example showing the expected "
            "caliber of writing for the Recommendation section. This gives the "
            "model a specific quality target rather than relying on abstract "
            "instructions like 'write well.'"
        ),
        "location": "System prompt, <example_calibration> block",
    },
    {
        "name": "Anti-Pattern Guardrails",
        "description": (
            "Explicit 'do not' instructions target the most common failure modes "
            "for analytical summaries: burying the lead, restating numbers without "
            "interpretation, hedging with weak language, and offering vague next "
            "steps. Negative constraints are often more effective than positive ones."
        ),
        "location": "System prompt, <quality_standards> and <communication_principles>",
    },
    {
        "name": "Grounding Instruction",
        "description": (
            "The user prompt closes with an explicit instruction to ground every "
            "claim in the provided data and not invent numbers. This acts as a "
            "final guardrail against hallucination — particularly important when "
            "the model is generating analytical content."
        ),
        "location": "User prompt, closing instruction",
    },
]


# ──────────────────────────────────────────────────────────────
# CLI SELF-TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from stats_engine import prepare_data
    from segment_analyzer import build_summary_payload

    print("Loading dataset...")
    from pathlib import Path
    raw_df = pd.read_csv(Path(__file__).parent / "ab_test_homepage_banner.csv")
    df = prepare_data(raw_df, exclude_contaminated=True)

    print("Building summary payload...")
    payload = build_summary_payload(df)

    system_prompt, user_prompt = preview_prompt(payload)

    print("\n" + "=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(system_prompt)

    print("\n" + "=" * 60)
    print("USER PROMPT")
    print("=" * 60)
    print(user_prompt)

    total_chars = len(system_prompt) + len(user_prompt)
    est_tokens = total_chars // 4
    print(f"\n  Estimated prompt tokens: ~{est_tokens:,}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print("\n" + "=" * 60)
        print("GENERATED EXECUTIVE SUMMARY")
        print("=" * 60)
        summary = generate_summary(payload)
        print(summary)
    else:
        print("\n  ANTHROPIC_API_KEY not set — skipping API call.")
        print("  Set it to test the full pipeline:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("  python ai_summary.py")
