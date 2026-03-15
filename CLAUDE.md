# AI-Powered A/B Test Analyzer — CLAUDE.md

## Project Purpose

Portfolio project by Brendan Hoffman demonstrating how generative AI turns raw A/B test data into a boardroom-ready executive report. The intended reader of the AI output is a VP-level executive — time-constrained, not statistically trained, focused on decisions. All UI and AI changes should serve that reader.

## Quick Start

```bash
# Install dependencies (no requirements.txt — install manually)
pip install streamlit plotly pandas scipy numpy anthropic python-dotenv

# Run the app
streamlit run app.py
```

## Environment

| Variable | Required | Notes |
|----------|----------|-------|
| `ANTHROPIC_API_KEY` | No — optional | Only needed to regenerate the AI summary live; cached output in `ai_generated_summary.md` is displayed by default |

The app runs fully without an API key. The AI Summary tab displays the cached output from `ai_generated_summary.md`. Setting `ANTHROPIC_API_KEY` enables live regeneration via the Claude API.

## App Structure

Four tabs, progressively deeper:

1. **Executive Summary** — bottom-line story and recommendation
2. **Detailed Results** — full statistical evidence per metric
3. **Segment Analysis** ("Who Does It Work For?") — device, user type, time, geography, compound segments; anomaly detection
4. **AI Summary** — Claude-powered narrative with streaming output and a "prompt transparency" panel

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Visualization | Plotly |
| Data / Stats | pandas, scipy, numpy |
| AI | Anthropic Claude API (`claude-sonnet-4-6`) |
| Styling | Custom CSS + Google Fonts via Streamlit config |

## Key Files

| File | Role |
|------|------|
| `app.py` | Streamlit UI — four-tab dashboard, all rendering logic |
| `stats_engine.py` | Statistical tests: z-test, Welch's t-test, power analysis, metric registry |
| `segment_analyzer.py` | Segment breakdowns, compound segments, anomaly detection |
| `ai_summary.py` | Claude API integration — prompt construction, streaming, prompt technique descriptions |
| `generate_test_data.py` | Synthetic dataset generator |
| `ab_test_homepage_banner.csv` | Synthetic dataset (the "input") |
| `ai_generated_summary.md` | Cached example AI output |
| `.streamlit/config.toml` | Theme configuration (parchment background, fonts) |

## Visual Design

The app uses a **"Field Research Report"** aesthetic — premium, editorial, intentionally non-generic:

- Background: light parchment (`#F6F2EA`)
- Sidebar: deep forest green (`#1C2B22`)
- Typography: Playfair Display (headers), JetBrains Mono (data/code), Lato (body)
- Variant colors: cerulean blue for Control A (`#2B6CB0`), forest green for Variant B (`#276749`)
- Outcome colors: green for positive lift, red for negative

**Preserve this aesthetic in all changes.** It is intentional and differentiates the app from generic dashboard templates.

## Prompt Engineering (ai_summary.py)

Seven techniques are implemented, documented in `ai_summary.py`'s module docstring, and surfaced in the UI as a "prompt transparency" panel. This panel is a portfolio-facing feature — keep the `PROMPT_TECHNIQUES` list in `ai_summary.py` maintained and visible in the UI.

## Constraints and Preferences

- **Do not change the visual design direction** without good reason — the aesthetic is a deliberate portfolio choice
- **Do not simplify the statistics** — the rigorous statistical implementation (z-test, Welch's t-test, power analysis, effect size) is a demonstration of methodology fluency
- **Do not genericize the AI output** — the point is that the output should be sharp, direct, and boardroom-ready; resist changes that make it more hedged or verbose
- **Do not use Material Icons on elements that inherit font-family** — it breaks glyph rendering; apply icon fonts only on isolated spans with explicit font-family overrides
- **Keep prompt techniques documented** — the `PROMPT_TECHNIQUES` list in `ai_summary.py` and the UI panel that renders it are portfolio-facing features, not developer notes
