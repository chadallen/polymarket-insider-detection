# Insider Trading Detection — Product Requirements

## Vision

A proof-of-concept system that automatically scans resolved Polymarket prediction markets and surfaces the ones most likely to have involved insider trading — based on price anomalies and on-chain wallet behavior. Research and educational use only.

**Live dashboard:** https://dashboard-rouge-pi-13.vercel.app

---

## Features — Shipped

### Price signal pipeline

Fetches CLOB hourly price history (fidelity=60) for resolved political markets (Polymarket tag_id=2, volume ≥ $1M, resolved since 2024-01-01). Computes 7 features per market:

| Feature | Description |
|---|---|
| `surprise_score` | `|outcome - starting_price|` — how unexpected the resolution was |
| `late_move_ratio` | Fraction of total price move concentrated in the final tick |
| `price_volatility` | Std dev of absolute per-tick price changes |
| `max_single_move` | Largest single price step |
| `total_price_move` | `|final_price - starting_price|` |
| `price_momentum_6h` | Price change in final 6h window |
| `price_momentum_12h` | Price change in final 12h window |

An IsolationForest (contamination=0.1) on all 7 price features produces `suspicion_score` — a price-only anomaly signal used to rank markets for the Dune wallet query (not fed directly to the ensemble).

Markets with starting price outside 0.15–0.85 are skipped as uncontested. Labeled cases receive an extended price window back to their recorded start date for better feature coverage.

### Wallet signal pipeline

Queries on-chain trade data via Dune Analytics (`polymarket_polygon.market_trades`) for the top 50 markets by `suspicion_score` (~4 Dune credits). Computes 6 wallet features per market:

| Feature | Description |
|---|---|
| `new_wallet_ratio` | Fraction of trading wallets with no prior 12h Polymarket activity |
| `new_wallet_ratio_6h` | Same, restricted to the final 6h window |
| `burst_score` | Ratio of peak-hour trade count to median hourly count |
| `order_flow_imbalance` | `|yes_volume - no_volume| / total_volume` |
| `wallet_concentration` | Gini-like concentration of trade sizes across wallets |
| `wallet_age_median_days` | Median age (days since first Polygon tx) of top-20 wallets, via Polygonscan V2 |

Cross-market wallet flag (`cross_market_wallet_count`) is computed locally from the top-20 wallet lists already fetched — counts wallets appearing in ≥3 other markets. No additional Dune credits required (see ADR-0003).

### Ensemble classifier

Three-model ensemble outputs `insider_trading_prob` ∈ [0, 1] from 14 merged features (7 price + 7 wallet including `cross_market_wallet_count`):

| Model | Weight | Description |
|---|---|---|
| PU-LightGBM | 0.5 | Elkan & Noto PU learning: positives vs unlabeled. `pu_prob = clip(raw / c, 0, 1)` where `c = mean(raw[labeled_positives])`. Soft label weights applied. |
| IsolationForest | 0.05 | Full 14-feature matrix, contamination=0.1, score normalized [0, 1]. |
| One-Class SVM | 0.2 | Trained on CONFIRMED cases only (rbf kernel, nu=0.5). Falls back to 0.5 if no CONFIRMED matches. |

Ground truth: **22 labeled cases** in `data/labeled_cases.csv` — 5 CONFIRMED, 4 SUSPECTED, 13 POSSIBLE. Soft label weights: CONFIRMED=1.0, SUSPECTED=0.6, POSSIBLE=0.3. Zero-variance features are silently dropped before training.

### Dashboard ("Prediction Market Forensics")

React/Vite/Tailwind frontend deployed on Vercel. Key sections:

**Charts (top row):**
- Model Agreement scatter: PU-LightGBM (x) vs IsolationForest (y), colored by ensemble score
- Volume vs Suspicion scatter: log market volume (x) vs ensemble score (y)

**Ranked table** with sortable columns: market question, resolved date, Price IF score, IsoForest score, PU-LightGBM score, ensemble score. Color-coded: High ≥35% (rose), Medium ≥25% (amber), Low <25% (zinc).

**Expanded detail panel** (click any row):
- Ensemble score breakdown with per-model components and weights
- 8-axis radar chart (surprise_score, late_move_ratio, max_single_move, price_momentum_6h, new_wallet_ratio, order_flow_imbalance, wallet_concentration, burst_score)
- Price signals detail: resolved date, volume, starting/final price, all 7 price features
- Wallet signals detail: unique wallets, trade count, all 7 wallet features
- Anomaly highlighting on values that cross detection thresholds

---

## Features — Planned

### Phase 4a: News coverage signal (GDELT)

**Goal:** Add `news_article_count_48h` as a 15th feature. Low news coverage + strong price move is more suspicious than the same move with heavy coverage — this is the highest-signal unbuilt addition. ADR-0006 accepted GDELT via BigQuery free tier as the approach.

**Approach:** Query GDELT via BigQuery free tier (1 TB/month, no credit card). Extract 2–3 key terms from the market question, count articles in the 48h pre-resolution window, cache per market in `gdelt_counts.pkl`.

**Fallback:** DuckDuckGo News scraping (no API key, rate-limited) to unblock development if BigQuery setup is slow. Migrate to BigQuery later.

**Acceptance criteria:**
- `news_article_count_48h` populated for ≥ 80% of markets in a full run
- Feature in `MODEL_FEATURES` with median imputation fallback
- `--skip-dune --classifier-only` imputation path works
- Leave-one-out CV rank holds or improves for CONFIRMED cases

### Phase 4b: Kalshi cross-reference (evaluate after 4a)

Two labeled cases (`kalshi_mrbeast`, `kalshi_langford`) are Kalshi-only and untestable in the current pipeline. If scope is extended: fetch Kalshi prices via their free read-only API and compute `kalshi_price_divergence = poly_price - kalshi_price` in the 48h window.

### Phase 4c: Twitter/X API (evaluate after 4a)

Higher recency than GDELT, noisier. Free basic tier: 500K tweets/month. Evaluate only if GDELT signal proves insufficient.

### Phase 5: Validation & calibration

1. Leave-one-out CV on all 22 labeled cases — held-out positive must rank in top 10% of scored markets
2. Calibrate ensemble weights from CV results — current 0.5/0.05/0.2 are unvalidated starting points
3. Manual review of top 10 flagged historical markets — spot-check false positive rate
4. Backtest: model still flags current top suspects after any Phase 4 changes

---

## Known Limitations

1. **Small labeled set** — 22 cases, 5 CONFIRMED. PU `c` prior estimate is unstable with few confirmed positives.
2. **Ensemble weights don't sum to 1.0** — current 0.5 + 0.05 + 0.2 = 0.75. Intentional starting point pending Phase 5 calibration.
3. **CLOB data availability** — ~30% of fetched markets survive the price history filter; low-volume political markets often have no CLOB data.
4. **Cross-market flag is partial** — covers only top-20 wallets per market, not the full trading population (ADR-0003).
5. **No entity resolution** — cannot link wallets across accounts beyond the cross-market count.
6. **No news signal yet** — model cannot distinguish public-information trading from insider trading until Phase 4a lands.
7. **Live mode is POC** — `--live` mode is unvalidated; resolution-proxy features are heuristic.
8. **OC-SVM fragile** — falls back to neutral 0.5 if no CONFIRMED cases match the current dataset (e.g., after label changes).

---

## Open Questions

1. **Ensemble weight normalization**: Should weights be forced to sum to 1.0 now, or wait for Phase 5 CV calibration?
2. **GDELT vs DuckDuckGo**: Start with DDG scraping to unblock Phase 4a, migrate to BigQuery later?
3. **Kalshi scope**: Two labeled cases are Kalshi-only — decide before Phase 4b whether to extend the pipeline.
4. **Polygonscan rate limits**: 50 markets × 20 wallets = 1,000 API calls against 5 req/sec free tier. If too slow, scope to top 20 markets by `suspicion_score` only.
