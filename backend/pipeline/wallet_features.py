"""
Wallet-based feature extraction via Dune Analytics.
Corresponds to notebook Cells 7–11.
"""
import os
import re
import pandas as pd
from backend.config import TOP_N_MARKETS
from backend.pipeline.dune import run_query, sql_quote


# ── Labeled cases (ground truth) ─────────────────────────────────────────
# Source of truth: data/labeled_cases.csv
# To add a case: add a row to the CSV, then run: python run.py --classifier-only

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "labeled_cases.csv")


def load_labeled_cases() -> pd.DataFrame:
    """Load labeled_cases.csv. Columns: key, label, question_filter, start, end, resolution, notes."""
    return pd.read_csv(_CSV_PATH)


def _load_labeled_market_configs(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    configs = {}
    for _, row in df.iterrows():
        configs[row["key"]] = {
            "label": row["label"],
            "question_filter": row["question_filter"],
            "start": str(row["start"]),
            "end": str(row["end"]),
            "resolution": str(row["resolution"]),
        }
    return configs

LABELED_MARKET_CONFIGS = _load_labeled_market_configs(_CSV_PATH)


# ── Question filter matching ───────────────────────────────────────────────

def _sql_like_to_regex(like_pattern: str) -> re.Pattern:
    """Convert a SQL LIKE pattern (% = wildcard) to a compiled Python regex."""
    parts = like_pattern.split('%')
    return re.compile('.*'.join(re.escape(p) for p in parts), re.IGNORECASE)


def question_matches_filter(question: str, question_filter: str) -> bool:
    """
    Return True if question matches the SQL LIKE question_filter from labeled_cases.csv.
    Handles OR-separated LIKE clauses, e.g.:
      "question LIKE '%Maduro out%' OR question LIKE '%Maduro%custody%'"
    """
    like_patterns = re.findall(r"LIKE\s+'([^']+)'", question_filter, re.IGNORECASE)
    return any(_sql_like_to_regex(p).search(question) for p in like_patterns)


# ── Dune SQL builder ───────────────────────────────────────────────────────

def _build_labeled_sql(config: dict) -> str:
    from datetime import date as _date
    res_ts = config["resolution"].replace("T", " ")
    qf = config["question_filter"]
    s, e = config["start"], config["end"]
    # Derive block_month bounds for partition pruning
    s_month = _date.fromisoformat(s[:10]).replace(day=1).isoformat()
    e_month = _date.fromisoformat(e[:10]).replace(day=1).isoformat()
    return f"""
WITH trades AS (
    SELECT block_time, maker, price, amount, token_outcome_name
    FROM polymarket_polygon.market_trades
    WHERE ({qf})
    AND block_time BETWEEN TIMESTAMP '{s}' AND TIMESTAMP '{e}'
    AND block_month BETWEEN DATE '{s_month}' AND DATE '{e_month}'
),
market_stats AS (
    SELECT COUNT(*) AS trade_count,
           COUNT(DISTINCT maker) AS unique_wallets,
           SUM(amount) AS total_volume
    FROM trades
),
resolution_times AS (SELECT TIMESTAMP '{res_ts}' AS res_time),
new_wallets_12h AS (
    SELECT SUM(t.amount) AS new_wallet_volume_12h FROM trades t
    CROSS JOIN resolution_times r
    WHERE t.maker IN (
        SELECT maker FROM (
            SELECT maker, MIN(block_time) AS first_seen FROM trades GROUP BY maker
        ) fw WHERE fw.first_seen >= r.res_time - INTERVAL '12' HOUR
    )
),
new_wallets_6h AS (
    SELECT SUM(t.amount) AS new_wallet_volume_6h FROM trades t
    CROSS JOIN resolution_times r
    WHERE t.maker IN (
        SELECT maker FROM (
            SELECT maker, MIN(block_time) AS first_seen FROM trades GROUP BY maker
        ) fw WHERE fw.first_seen >= r.res_time - INTERVAL '6' HOUR
    )
),
burst AS (
    SELECT MAX(cnt) AS burst_score
    FROM (SELECT DATE_TRUNC('hour', block_time) AS h, COUNT(*) AS cnt
          FROM trades GROUP BY DATE_TRUNC('hour', block_time)) x
),
directional AS (
    SELECT MAX(ov) * 1.0 / NULLIF(SUM(ov), 0) AS directional_consensus
    FROM (SELECT token_outcome_name, SUM(amount) AS ov
          FROM trades GROUP BY token_outcome_name) x
),
order_flow AS (
    SELECT ABS(yes_vol - no_vol) / NULLIF(yes_vol + no_vol, 0) AS order_flow_imbalance
    FROM (SELECT SUM(CASE WHEN price > 0.5  THEN amount ELSE 0 END) AS yes_vol,
                 SUM(CASE WHEN price <= 0.5 THEN amount ELSE 0 END) AS no_vol
          FROM trades) x
),
wallet_vols AS (
    SELECT maker, SUM(amount) AS wallet_vol
    FROM trades
    GROUP BY maker
),
wallet_ranked AS (
    SELECT wallet_vol,
           ROW_NUMBER() OVER (ORDER BY wallet_vol ASC) AS rk,
           COUNT(*) OVER () AS n_wallets
    FROM wallet_vols
),
concentration AS (
    SELECT
        CASE
            WHEN SUM(wallet_vol) = 0 OR MAX(n_wallets) <= 1 THEN 0.0
            ELSE (2.0 * SUM(CAST(rk AS DOUBLE) * wallet_vol) /
                  (CAST(MAX(n_wallets) AS DOUBLE) * SUM(wallet_vol)))
                 - CAST(MAX(n_wallets) + 1 AS DOUBLE) / CAST(MAX(n_wallets) AS DOUBLE)
        END AS wallet_concentration
    FROM wallet_ranked
)
SELECT ms.trade_count, ms.unique_wallets, ms.total_volume,
    COALESCE(nw12.new_wallet_volume_12h, 0) / NULLIF(ms.total_volume, 0) AS new_wallet_ratio_12h,
    COALESCE(nw6.new_wallet_volume_6h,  0) / NULLIF(ms.total_volume, 0) AS new_wallet_ratio_6h,
    b.burst_score, d.directional_consensus, o.order_flow_imbalance,
    c.wallet_concentration
FROM market_stats ms
CROSS JOIN new_wallets_12h nw12 CROSS JOIN new_wallets_6h nw6
CROSS JOIN burst b CROSS JOIN directional d CROSS JOIN order_flow o
CROSS JOIN concentration c
"""


# ── Fetch labeled market data from Dune (Cell 8) ──────────────────────────

def fetch_labeled_market_trades() -> dict:
    """
    Runs one Dune query per labeled market. ~4 credits total.
    Returns dict: name -> DataFrame.
    """
    dune_results = {}
    configs = LABELED_MARKET_CONFIGS
    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"\n[{i}/{len(configs)}] {name.upper()} ({config['label']})")
        df, _ = run_query(_build_labeled_sql(config), label=name)
        dune_results[name] = df
        if not df.empty:
            row = df.iloc[0]
            print(
                f"  trades={int(row.get('trade_count') or 0):,} "
                f"wallets={int(row.get('unique_wallets') or 0):,} "
                f"vol=${float(row.get('total_volume') or 0):,.0f}"
            )
        else:
            print("  No results")

    print(f"\nFetched {sum(1 for d in dune_results.values() if not d.empty)}/{len(dune_results)} markets")
    return dune_results


# ── Extract wallet features from Dune results (Cell 9) ───────────────────

_COLUMN_MAP = {
    "new_wallet_ratio_12h":  "new_wallet_ratio",
    "new_wallet_ratio_6h":   "new_wallet_ratio_6h",
    "order_flow_imbalance":  "order_flow_imbalance",
    "burst_score":           "burst_score",
    "directional_consensus": "directional_consensus",
    "wallet_concentration":  "wallet_concentration",
    "total_volume":          "total_volume",
    "unique_wallets":        "unique_wallets",
    "trade_count":           "trade_count",
}

MIN_VOLUME_USD = 1000


def extract_wallet_features(dune_results: dict) -> dict:
    """Parse Dune result rows into feature dicts."""
    wallet_features = {}
    for name in LABELED_MARKET_CONFIGS:
        df = dune_results.get(name)
        if df is None or len(df) == 0:
            print(f"  {name}: no data")
            continue
        row = df.iloc[0]
        feats = {col: row.get(src, 0) for src, col in _COLUMN_MAP.items()}
        wallet_features[name] = feats
        flag = " LOW VOLUME" if feats.get("total_volume", 0) < MIN_VOLUME_USD else ""
        print(
            f"{name.upper()} ({LABELED_MARKET_CONFIGS[name]['label']}){flag}\n"
            f"  Vol ${feats['total_volume']:>14,.0f} | Wallets {int(feats['unique_wallets']):,}\n"
            f"  New 6h {feats['new_wallet_ratio_6h']:.1%} | OFI {feats['order_flow_imbalance']:.3f} "
            f"| Burst {int(feats['burst_score'])} | Dir {feats['directional_consensus']:.1%} "
            f"| Conc {feats.get('wallet_concentration', 0):.3f}\n"
        )
    print(f"Extracted features for {len(wallet_features)}/{len(LABELED_MARKET_CONFIGS)} markets")
    return wallet_features


# ── Top-N wallet query from Dune (Cell 11) ────────────────────────────────

def fetch_top_n_wallet_data(
    df_scored: pd.DataFrame,
    df_markets: pd.DataFrame,
    top_n: int = TOP_N_MARKETS,
) -> pd.DataFrame:
    """
    Fetches wallet features for top N markets by suspicion_score.
    Uses per-market end_date (not a rolling window) so older resolved
    markets are included. Also computes new_wallet_ratio_12h/6h.
    ~4 credits.
    """
    top_markets = df_scored.nlargest(top_n, "suspicion_score")

    # Build per-market end_date lookup from df_markets
    end_dates = (
        df_markets[["question", "end_date"]]
        .set_index("question")["end_date"]
        .to_dict()
    )

    # VALUES clause: one row per market with its effective cutoff timestamp.
    # For live (open) markets the end_date is in the future, so we clamp to now
    # so the wallet query only sees trades that have actually happened.
    from datetime import datetime, timezone as _tz
    now_ts = datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S")

    from datetime import date as _date
    values_rows = []
    effective_ts_list = []
    for q in top_markets["question"].tolist():
        end_date = end_dates.get(q, "")
        if not end_date:
            continue
        # Normalise to plain timestamp (strip trailing Z or timezone)
        ts = str(end_date).replace("Z", "").replace("+00:00", "").replace("T", " ")[:19]
        # Clamp future dates to now so live markets don't return all-time trades
        if ts > now_ts:
            ts = now_ts
        values_rows.append(f"    ({sql_quote(q)}, TIMESTAMP '{ts}')")
        effective_ts_list.append(ts)

    if not values_rows:
        print("No markets with end_date — skipping wallet query")
        return pd.DataFrame()

    values_clause = ",\n".join(values_rows)

    # Derive block_month lower bound from the earliest effective end_date for
    # partition pruning (markets start well before their end_date, but this
    # still eliminates pre-2024 data and dramatically reduces scan cost).
    min_ts = min(effective_ts_list)
    min_month = _date.fromisoformat(min_ts[:10]).replace(day=1).isoformat()

    sql = f"""
WITH market_times AS (
    SELECT *
    FROM (VALUES
{values_clause}
    ) AS t(question, end_date)
),
trades AS (
    SELECT t.question, t.maker, t.price, t.amount, t.token_outcome_name,
           t.block_time, mt.end_date
    FROM polymarket_polygon.market_trades t
    JOIN market_times mt ON t.question = mt.question
    WHERE t.block_time <= mt.end_date
    AND t.block_month >= DATE '{min_month}'
),
wallet_first_seen AS (
    SELECT question, maker, amount, end_date,
           MIN(block_time) OVER (PARTITION BY question, maker) AS first_seen
    FROM trades
),
new_wallets_12h AS (
    SELECT question,
           SUM(CASE WHEN first_seen >= end_date - INTERVAL '12' HOUR
                    THEN amount ELSE 0 END) AS new_wallet_volume_12h
    FROM wallet_first_seen
    GROUP BY question
),
new_wallets_6h AS (
    SELECT question,
           SUM(CASE WHEN first_seen >= end_date - INTERVAL '6' HOUR
                    THEN amount ELSE 0 END) AS new_wallet_volume_6h
    FROM wallet_first_seen
    GROUP BY question
),
burst AS (
    SELECT question, MAX(cnt) AS burst_score
    FROM (SELECT question, DATE_TRUNC('hour', block_time) AS h, COUNT(*) AS cnt
          FROM trades GROUP BY question, DATE_TRUNC('hour', block_time)) x
    GROUP BY question
),
directional AS (
    SELECT question,
           MAX(ov) * 1.0 / NULLIF(SUM(ov), 0) AS directional_consensus
    FROM (SELECT question, token_outcome_name, SUM(amount) AS ov
          FROM trades GROUP BY question, token_outcome_name) x
    GROUP BY question
),
order_flow AS (
    SELECT question,
           ABS(SUM(CASE WHEN price > 0.5  THEN amount ELSE 0 END) -
               SUM(CASE WHEN price <= 0.5 THEN amount ELSE 0 END)) /
           NULLIF(SUM(amount), 0) AS order_flow_imbalance,
           SUM(amount)            AS total_volume,
           COUNT(*)               AS trade_count,
           COUNT(DISTINCT maker)  AS unique_wallets
    FROM trades GROUP BY question
),
wallet_vols AS (
    SELECT question, maker, SUM(amount) AS wallet_vol
    FROM trades
    GROUP BY question, maker
),
wallet_ranked AS (
    SELECT question, maker, wallet_vol,
           ROW_NUMBER() OVER (PARTITION BY question ORDER BY wallet_vol ASC) AS rk,
           COUNT(*) OVER (PARTITION BY question) AS n_wallets
    FROM wallet_vols
),
concentration AS (
    SELECT question,
           CASE
               WHEN SUM(wallet_vol) = 0 OR MAX(n_wallets) <= 1 THEN 0.0
               ELSE (2.0 * SUM(CAST(rk AS DOUBLE) * wallet_vol) /
                     (CAST(MAX(n_wallets) AS DOUBLE) * SUM(wallet_vol)))
                    - CAST(MAX(n_wallets) + 1 AS DOUBLE) / CAST(MAX(n_wallets) AS DOUBLE)
           END AS wallet_concentration
    FROM wallet_ranked
    GROUP BY question
),
top_wallet_addrs AS (
    SELECT question,
           ARRAY_JOIN(
               ARRAY_AGG(maker ORDER BY wallet_vol DESC),
               ','
           ) AS top_wallet_addresses
    FROM (
        SELECT question, maker, wallet_vol,
               ROW_NUMBER() OVER (PARTITION BY question ORDER BY wallet_vol DESC) AS rn
        FROM wallet_vols
    ) ranked
    WHERE rn <= 20
    GROUP BY question
)
SELECT o.question, o.order_flow_imbalance, o.total_volume, o.trade_count,
       o.unique_wallets,
       b.burst_score * 1.0 / NULLIF(o.trade_count, 0) AS burst_score,
       d.directional_consensus,
       COALESCE(nw12.new_wallet_volume_12h, 0) / NULLIF(o.total_volume, 0) AS new_wallet_ratio,
       COALESCE(nw6.new_wallet_volume_6h,  0) / NULLIF(o.total_volume, 0) AS new_wallet_ratio_6h,
       c.wallet_concentration,
       wa.top_wallet_addresses
FROM order_flow o
JOIN burst b            ON o.question = b.question
JOIN directional d      ON o.question = d.question
LEFT JOIN new_wallets_12h nw12 ON o.question = nw12.question
LEFT JOIN new_wallets_6h  nw6  ON o.question = nw6.question
LEFT JOIN concentration c      ON o.question = c.question
LEFT JOIN top_wallet_addrs wa  ON o.question = wa.question
"""

    print(f"Wallet query for top {top_n} markets...")
    df_wallet_agg, _ = run_query(sql, label="top_n_wallet", timeout=300)

    if not df_wallet_agg.empty:
        print(f"\n{len(df_wallet_agg)} markets returned")
    else:
        print("No results — check query or Dune status")

    return df_wallet_agg


# ── Wallet age via Polygonscan (Phase 2) ──────────────────────────────────

def fetch_wallet_age_features(
    df_wallet_agg: "pd.DataFrame",
    polygonscan_api_key: str = "",
) -> "pd.DataFrame":
    """
    Compute wallet_age_median_days for each market by calling the Polygonscan
    API for each wallet address found in the top_wallet_addresses column.

    Requires df_wallet_agg to contain a 'top_wallet_addresses' column
    (comma-separated wallet addresses, added by fetch_top_n_wallet_data in
    Phase 2). If the column is absent the DataFrame is returned unchanged.

    Free tier: ~4 req/sec, no API key required (slower without one).
    """
    import pandas as pd
    from backend.pipeline.polygonscan import fetch_wallet_ages, compute_wallet_age_median

    if df_wallet_agg.empty or "top_wallet_addresses" not in df_wallet_agg.columns:
        print("  top_wallet_addresses column missing — skipping Polygonscan lookup")
        return df_wallet_agg

    # Collect all unique addresses across all markets
    all_addrs: list[str] = []
    for addr_str in df_wallet_agg["top_wallet_addresses"].dropna():
        all_addrs.extend(str(addr_str).split(","))
    all_addrs = [a.strip() for a in all_addrs if a.strip()]

    if not all_addrs:
        return df_wallet_agg

    unique_count = len(set(a.lower() for a in all_addrs))
    print(f"  Fetching wallet ages for {unique_count} unique wallets via Polygonscan...")
    age_map = fetch_wallet_ages(all_addrs, api_key=polygonscan_api_key)
    n_found = sum(1 for v in age_map.values() if v is not None)
    print(f"  Resolved {n_found}/{unique_count} wallet ages")

    df = df_wallet_agg.copy()
    df["wallet_age_median_days"] = df["top_wallet_addresses"].apply(
        lambda s: compute_wallet_age_median(
            str(s).split(",") if pd.notna(s) else [], age_map
        )
    )
    return df


# ── Cross-market wallet flag via Dune (Phase 2) ───────────────────────────

def compute_cross_market_wallet_flags(
    df_wallet_agg: "pd.DataFrame",
    min_shared_markets: int = 3,
) -> "pd.DataFrame":
    """
    Compute cross_market_wallet_count locally from the top_wallet_addresses
    column already fetched by fetch_top_n_wallet_data(). No Dune credits needed.

    For each market, counts how many of its top wallets also appear in at least
    `min_shared_markets` other markets in df_wallet_agg.

    Limitation: only covers the top-N wallets per market (currently top 20),
    not the full trader population. Concentrated insiders (large bets) are
    likely captured; distributed/layered trading across many small wallets may
    be missed.

    NOTE (2026-03-13): Replaced the Dune-based fetch_cross_market_wallet_flags()
    below because the Dune query consistently hit the 10-credit resource cap.
    The original Dune query is preserved (commented out) in case full-population
    coverage is needed in future. To re-enable, raise DUNE_MAX_CREDITS and call
    fetch_cross_market_wallet_flags() in run.py instead.

    Returns DataFrame with columns: question, cross_market_wallet_count.
    """
    if df_wallet_agg.empty or "top_wallet_addresses" not in df_wallet_agg.columns:
        return pd.DataFrame(columns=["question", "cross_market_wallet_count"])

    # Build wallet → set of questions mapping
    wallet_to_questions: dict[str, set[str]] = {}
    for _, row in df_wallet_agg.iterrows():
        addrs = str(row["top_wallet_addresses"]).split(",")
        for addr in addrs:
            addr = addr.strip().lower()
            if addr:
                wallet_to_questions.setdefault(addr, set()).add(row["question"])

    # Wallets active in >= min_shared_markets markets
    cross_market_wallets = {
        w for w, qs in wallet_to_questions.items() if len(qs) >= min_shared_markets
    }

    # Count per market
    records = []
    for _, row in df_wallet_agg.iterrows():
        addrs = {a.strip().lower() for a in str(row["top_wallet_addresses"]).split(",") if a.strip()}
        count = len(addrs & cross_market_wallets)
        records.append({"question": row["question"], "cross_market_wallet_count": count})

    print(f"Cross-market wallet flag (local, top-wallet addresses, min_shared={min_shared_markets}): "
          f"{len(cross_market_wallets)} cross-market wallets found")
    return pd.DataFrame(records)


# ── Dune-based cross-market wallet flag (DISABLED — hits 10-credit resource cap) ──
# To re-enable: raise DUNE_MAX_CREDITS env var and call this instead of
# compute_cross_market_wallet_flags() in run.py.
#
# def fetch_cross_market_wallet_flags(
#     questions: list[str],
#     min_shared_markets: int = 3,
# ) -> "pd.DataFrame":
#     """
#     For each market in `questions`, count how many wallets traded in at least
#     `min_shared_markets` of the listed markets (cross_market_wallet_count).
#     Scans full polymarket_polygon.market_trades — catches distributed trading
#     patterns missed by the top-wallet local approach. ~10-15 Dune credits.
#     """
#     if not questions:
#         return pd.DataFrame()
#     q_list = ", ".join(sql_quote(q) for q in questions)
#     sql = f"""
# WITH top_market_wallets AS (
#     SELECT maker, question
#     FROM polymarket_polygon.market_trades
#     WHERE question IN ({q_list})
#     GROUP BY maker, question
# ),
# wallet_market_counts AS (
#     SELECT maker, COUNT(DISTINCT question) AS markets_active
#     FROM top_market_wallets
#     GROUP BY maker
# ),
# cross_market_per_question AS (
#     SELECT tmw.question,
#            COUNT(DISTINCT tmw.maker) AS cross_market_wallet_count
#     FROM top_market_wallets tmw
#     JOIN wallet_market_counts wmc ON tmw.maker = wmc.maker
#     WHERE wmc.markets_active >= {min_shared_markets}
#     GROUP BY tmw.question
# )
# SELECT question, cross_market_wallet_count
# FROM cross_market_per_question
# """
#     print(f"Cross-market wallet flag query ({len(questions)} markets, min_shared={min_shared_markets})...")
#     df, _ = run_query(sql, label="cross_market_wallet_flags")
#     return df
