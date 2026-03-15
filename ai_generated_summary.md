## Recommendation

**Ship the personalized banner to returning visitors immediately** — they convert at 4.77% vs. 3.23% for the static banner, a **47.7% lift** we're highly confident in (p < 0.0001). Hold off on new visitors, where the personalization effect is negligible (+4.6%, not statistically significant), almost certainly because the engine has no browsing history to personalize against.

## What We Found

- **Banner engagement jumped 15.4%** — click-through rate rose from 12.6% to 14.5% (p < 0.001), confirming that personalized product recommendations capture more attention than a generic sale message.
- **Conversion rate improved 22.0% overall** — from 3.20% to 3.91% (p < 0.001, 98.8% statistical power), but this topline number masks a critical split between new and returning visitors.
- **Average order value dropped 8.0%** — from $86.08 to $79.22 (p < 0.001). Personalized recommendations likely steer shoppers toward browsed items rather than higher-margin bundles promoted in the static banner.
- **Revenue per visitor still rose 12.3%** — from $2.76 to $3.09 (p = 0.02). The conversion lift more than offsets the lower basket size, but the modest statistical power (63.7%) means this estimate carries more uncertainty than the other metrics.
- **Bounce rate was unchanged** — 37.6% vs. 37.5% (p = 0.77). The personalized banner doesn't repel visitors, but it doesn't reduce exits either.

## Where It Breaks Down

- **New visitors see almost no benefit.** Conversion lifted just 4.6% (not significant, p = 0.48) for the 29,388 new visitors in the test. The personalization engine has no browsing history to work with, so these visitors effectively see a weaker version of the static banner. This is the single biggest risk: new visitors represent 60% of traffic.
- **Desktop + New visitors actually declined slightly** — conversion dropped 1.0% (not significant), suggesting the personalization fallback experience on desktop may underperform the curated static creative.
- **Mobile drives the volume but returning visitors drive the win.** Mobile + Returning visitors converted at 4.67% vs. 3.21% (+45.5%, p = 0.0001), while Mobile + New showed only a 7.8% lift that isn't statistically reliable.
- **Email and direct traffic responded strongest** — email visitors converted 34.1% higher (p = 0.004) and direct visitors 54.6% higher (p = 0.004), likely because these audiences have the deepest browsing history for the engine to leverage.

## Recommended Next Steps

1. **Engineering team (this week):** Implement audience-based banner logic — serve the personalized banner to returning visitors and the static banner to first-time visitors. This captures the proven uplift without degrading the new-visitor experience.
2. **Product/UX team (next 2 weeks):** Investigate the $6.86 drop in average order value. Test whether adding a "complete the look" or bundle recommendation alongside the personalized product can recover basket size.
3. **Marketing team (next sprint):** Design a dedicated first-visit personalization strategy — consider using referral source or landing page context instead of browsing history to personalize for new visitors.
4. **Analytics team (ongoing):** Monitor revenue per visitor closely after rollout. The 12.3% lift is promising but had only 63.7% statistical power — a larger post-launch sample will confirm whether the net revenue impact holds.
