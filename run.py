#!/usr/bin/env python3
"""
Insider trading detection pipeline — CLI entrypoint.

Run modes:
  python run.py                    Full pipeline (fetch + score + classify)
  python run.py --skip-fetch       Use cached markets/histories, re-run scoring
  python run.py --skip-dune        Price pipeline only — no Dune credits spent
  python run.py --classifier-only  Retrain ensemble on saved df_combined (fastest, 0 credits)
  python run.py --push             Push output CSVs to GitHub after running
  python run.py --live             POC: score open markets ending within --hours-ahead hours

Example iteration loop (model tuning):
  1. Edit labeled_cases.csv to add or adjust cases
  2. python run.py --classifier-only
  3. Review output, repeat
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

_dune_key = os.environ.get("DUNE_API_KEY", "")
if _dune_key:
    print(f"[env] DUNE_API_KEY loaded ({len(_dune_key)} chars)")
else:
    print("[env] WARNING: DUNE_API_KEY not found — wallet queries will fail")

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Polymarket insider trading detector")
    p.add_argument("--skip-fetch",       action="store_true", help="Skip Gamma + CLOB fetch, use cached data")
    p.add_argument("--skip-dune",        action="store_true", help="Skip all Dune queries (0 credits, price signals only)")
    p.add_argument("--classifier-only",  action="store_true", help="Retrain RF on saved df_combined only")
    p.add_argument("--push",             action="store_true", help="Push output CSVs to GitHub when done")
    p.add_argument("--top-n",            type=int, default=None, help="Override TOP_N_MARKETS for wallet query")
    p.add_argument("--contamination",    type=float, default=0.1, help="Isolation Forest contamination (default: 0.1)")
    p.add_argument("--n-neg",            type=int, default=30, help="Number of implicit negatives for RF (default: 30)")
    p.add_argument("--live",             action="store_true", help="POC: score open markets resolving within --hours-ahead hours")
    p.add_argument("--hours-ahead",      type=int, default=48, help="Hours ahead to look for live markets (default: 48)")
    p.add_argument("--live-min-volume",  type=float, default=1_000_000, help="Min volume for live markets (default: 1000000)")
    p.add_argument("--refresh-labeled",  action="store_true", help="Re-fetch labeled case wallet features from Dune (~52 credits) and write data/labeled_features.pkl")
    return p.parse_args()


def push_to_github(df_combined, df_scored, df_wallet_agg):
    from github import Github
    from backend.config import GITHUB_TOKEN, GITHUB_REPO, GITHUB_BRANCH

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def push_csv(df, filename, message):
        if df is None or (hasattr(df, "empty") and df.empty):
            print(f"  Skipping {filename} (empty)")
            return
        content = df.to_csv(index=False)
        path = f"outputs/{filename}"
        try:
            existing = repo.get_contents(path, ref=GITHUB_BRANCH)
            repo.update_file(path, message, content, existing.sha, branch=GITHUB_BRANCH)
            print(f"  Updated  {path}")
        except Exception:
            repo.create_file(path, message, content, branch=GITHUB_BRANCH)
            print(f"  Created  {path}")

    print(f"Pushing to {GITHUB_REPO} ({ts})...")
    push_csv(df_combined,   "df_combined.csv",   f"Update combined scores {ts}")
    push_csv(df_scored,     "df_scored.csv",      f"Update price scores {ts}")
    push_csv(df_wallet_agg, "df_wallet_agg.csv",  f"Update wallet scores {ts}")
    print("Done — dashboard will update within ~1 minute.")


def _annotate_labeled_windows(df_markets: pd.DataFrame) -> pd.DataFrame:
    """
    For markets matching labeled cases:
      - set price_start_time to the case's start timestamp (extended CLOB window)
      - set is_labeled_case=True to bypass the starting-price filter in
        fetch_price_histories() — insider trading cases often start at low
        probability (e.g. Maduro 0.7%, Israel strike 7.5%) which would
        otherwise exclude them as "uncontested" markets.
    """
    from backend.pipeline.wallet_features import load_labeled_cases, question_matches_filter
    labeled = load_labeled_cases()
    df = df_markets.copy()
    df["price_start_time"] = None
    df["is_labeled_case"] = False
    for _, case in labeled.iterrows():
        mask = df["question"].apply(
            lambda q: question_matches_filter(q, case["question_filter"])
        )
        df.loc[mask, "price_start_time"] = case["start"]
        df.loc[mask, "is_labeled_case"] = True
    n = df["is_labeled_case"].sum()
    if n:
        print(f"  {n} markets annotated with labeled case price windows")
    return df


def main():
    args = parse_args()

    import backend.checkpoints as cp
    from backend.pipeline.fetcher import fetch_markets, fetch_price_histories, fetch_live_markets
    from backend.pipeline.price_features import build_price_features, score_with_isolation_forest
    from backend.pipeline.wallet_features import (
        fetch_top_n_wallet_data, fetch_wallet_age_features, compute_cross_market_wallet_flags,
        build_and_cache_labeled_features,
    )
    from backend.pipeline.scorer import merge_features, train_classifier
    from backend.config import POLYGONSCAN_API_KEY

    # top_n is None unless --top-n is explicitly passed; None means "all markets"
    top_n = args.top_n  # may be None

    # ── Refresh labeled case features from Dune ───────────────────────────
    if args.refresh_labeled:
        print("Refreshing labeled case features from Dune (~52 credits)...")
        df_labeled = build_and_cache_labeled_features()
        print(f"\nLabeled features shape: {df_labeled.shape}")
        print(f"Columns: {df_labeled.columns.tolist()}")
        return

    # ── Classifier-only mode: load df_combined, retrain, save ────────────
    if args.classifier_only:
        print("=== Classifier-only mode ===")
        state = cp.load_all()
        df_combined = state.get("df_combined")
        if df_combined is None:
            print("No saved df_combined found. Run full pipeline first.")
            sys.exit(1)
        print(f"Loaded df_combined: {len(df_combined)} markets")
        if "suspicion_score" not in df_combined.columns:
            print("WARNING: df_combined has no suspicion_score column.")
            print("This checkpoint predates Phase 1. Run the full pipeline to rebuild it.")
            sys.exit(1)

        df_combined = _merge_labeled_features(df_combined, state.get("labeled_features"))

        _preflight(df_combined)
        print("\n=== Training ensemble classifier (PU-LightGBM + IsoForest + OC-SVM) ===")
        df_combined, rf_model, rf_scaler, _ = train_classifier(
            df_combined, n_neg=args.n_neg
        )
        df_markets = state.get("df_markets")
        if df_markets is not None and "end_date" in df_markets.columns:
            end_dates = df_markets[["question", "end_date"]].drop_duplicates("question")
            df_combined = df_combined.drop(columns=["end_date"], errors="ignore").merge(end_dates, on="question", how="left")
        cp.save("df_combined", df_combined)
        _write_outputs(df_combined, state.get("df_scored"), state.get("df_wallet_agg"))
        if args.push:
            push_to_github(df_combined, state.get("df_scored"), state.get("df_wallet_agg"))
        return

    # ── Live mode: score open markets resolving soon (POC) ───────────────
    if args.live:
        _run_live(args, cp, build_price_features, score_with_isolation_forest,
                  fetch_top_n_wallet_data, merge_features, train_classifier,
                  fetch_live_markets, top_n)
        return

    # ── Load or fetch markets / histories ────────────────────────────────
    state = cp.load_all()

    if args.skip_fetch and state["df_markets"] is not None:
        print("=== Using cached markets + histories ===")
        df_markets = state["df_markets"]
        histories  = state["histories"]
    else:
        print("=== Fetching markets ===")
        df_markets = fetch_markets()
        cp.save("df_markets", df_markets)

        print("\n=== Fetching price histories ===")
        df_markets = _annotate_labeled_windows(df_markets)
        histories = fetch_price_histories(df_markets)
        cp.save("histories", histories)

    # ── Price features ────────────────────────────────────────────────────
    print("\n=== Computing price features ===")
    df_scored = build_price_features(df_markets, histories)

    print("\n=== Isolation Forest scoring ===")
    df_scored = score_with_isolation_forest(df_scored, contamination=args.contamination)
    cp.save("df_scored", df_scored)

    # ── Wallet features (optional, ~4 credits) ───────────────────────────
    df_wallet_agg = state.get("df_wallet_agg")
    if not args.skip_dune:
        if top_n is not None:
            print(f"\n=== Wallet query from Dune — top {top_n} markets (override) ===")
        else:
            print(f"\n=== Wallet query from Dune — all {len(df_scored)} markets ===")
        df_wallet_agg = fetch_top_n_wallet_data(df_scored, df_markets, top_n=top_n)
    else:
        print("\n=== Skipping Dune wallet query (--skip-dune) ===")
        if df_wallet_agg is not None:
            print(f"  Using cached df_wallet_agg ({len(df_wallet_agg)} markets)")

    # ── Polygonscan + cross-market flag (free, runs on cached or fresh wallet data) ──
    if df_wallet_agg is not None and not df_wallet_agg.empty:
        # Wallet age via Polygonscan (free, no extra Dune credits)
        print("\n=== Wallet age lookup via Polygonscan ===")
        df_wallet_agg = fetch_wallet_age_features(
            df_wallet_agg, polygonscan_api_key=POLYGONSCAN_API_KEY
        )

        # Cross-market wallet overlap (local — no Dune credits)
        # Uses top_wallet_addresses already fetched; covers top-N wallets only.
        # See wallet_features.py compute_cross_market_wallet_flags() for details.
        print("\n=== Cross-market wallet flag (local computation) ===")
        df_cross = compute_cross_market_wallet_flags(df_wallet_agg)
        if not df_cross.empty:
            # Drop stale columns before merge to avoid _x/_y suffix conflicts
            # when re-running against a cached df_wallet_agg pickle
            df_wallet_agg = df_wallet_agg.drop(
                columns=[c for c in ("cross_market_wallet_count", "cross_market_wallet_flag")
                         if c in df_wallet_agg.columns]
            )
            df_wallet_agg = df_wallet_agg.merge(
                df_cross[["question", "cross_market_wallet_count"]],
                on="question",
                how="left",
            )
            df_wallet_agg["cross_market_wallet_flag"] = (
                pd.to_numeric(
                    df_wallet_agg["cross_market_wallet_count"],
                    errors="coerce",
                ).fillna(0)
            )

    cp.save("df_wallet_agg", df_wallet_agg)

    # ── Merge features ────────────────────────────────────────────────────
    print("\n=== Merging features ===")
    df_combined = merge_features(df_scored, df_wallet_agg)

    df_combined = _merge_labeled_features(df_combined, state.get("labeled_features"))

    # ── Train classifier ──────────────────────────────────────────────────
    _preflight(df_combined)
    print("\n=== Training ensemble classifier (PU-LightGBM + IsoForest + OC-SVM) ===")
    df_combined, rf_model, rf_scaler, _ = train_classifier(df_combined, n_neg=args.n_neg)
    if "end_date" in df_markets.columns:
        end_dates = df_markets[["question", "end_date"]].drop_duplicates("question")
        df_combined = df_combined.drop(columns=["end_date"], errors="ignore").merge(end_dates, on="question", how="left")
    cp.save("df_combined", df_combined)

    # ── Write outputs ─────────────────────────────────────────────────────
    _write_outputs(df_combined, df_scored, df_wallet_agg)

    if args.push:
        push_to_github(df_combined, df_scored, df_wallet_agg)

    print("\nDone.")


def _run_live(args, cp, build_price_features, score_with_isolation_forest,
              fetch_top_n_wallet_data, merge_features, train_classifier,
              fetch_live_markets, top_n):
    """
    POC live mode: fetch open markets ending within --hours-ahead, score them,
    then apply the RF trained on historical labeled cases.

    NOTE: positives are drawn from labeled_cases.csv; predictions on live markets
    are indicative only since labeled cases are all historical.
    """
    from backend.pipeline.fetcher import fetch_price_histories

    print(f"=== LIVE MODE (POC) — markets ending within {args.hours_ahead}h ===\n")

    # 1. Fetch open markets ending soon
    df_markets = fetch_live_markets(
        hours_ahead=args.hours_ahead,
        min_volume=args.live_min_volume,
    )
    if df_markets.empty:
        print("No live markets found — try increasing --hours-ahead or lowering --live-min-volume.")
        return

    print(f"\n=== Fetching price histories ({len(df_markets)} markets) ===")
    histories = fetch_price_histories(df_markets)

    if not histories:
        print("No price histories returned — cannot score.")
        return

    # 2. Price features + isolation forest
    print("\n=== Computing price features ===")
    df_scored = build_price_features(df_markets, histories)

    print("\n=== Isolation Forest scoring ===")
    # Higher contamination = flags more markets; useful when we have few data points
    contamination = max(args.contamination, min(0.3, 2 / max(len(df_scored), 1)))
    df_scored = score_with_isolation_forest(df_scored, contamination=contamination)

    # 3. Wallet features from Dune (uses now as cutoff since resolution_time = now)
    df_wallet_agg = None
    if not args.skip_dune:
        if top_n is not None:
            live_top_n = min(top_n, len(df_scored))
            print(f"\n=== Wallet query for top {live_top_n} live markets (override) ===")
        else:
            live_top_n = None
            print(f"\n=== Wallet query for all {len(df_scored)} live markets ===")
        df_wallet_agg = fetch_top_n_wallet_data(df_scored, df_markets, top_n=live_top_n)

    # 4. Merge + classify
    print("\n=== Merging features ===")
    df_combined = merge_features(df_scored, df_wallet_agg)

    # Load historical df_combined to give the RF more training context
    state = cp.load_all()
    df_hist = state.get("df_combined")
    if df_hist is not None and not df_hist.empty:
        print(f"\nAppending {len(df_hist)} historical markets for RF training context...")
        df_for_training = pd.concat([df_hist, df_combined], ignore_index=True)
        df_for_training = df_for_training.drop_duplicates(subset=["question"], keep="last")
    else:
        print("\nNo historical data found — RF trained only on live markets (very limited).")
        df_for_training = df_combined

    _preflight(df_for_training)
    print("\n=== Training ensemble classifier (PU-LightGBM + IsoForest + OC-SVM) ===")
    df_for_training, _, _, _ = train_classifier(df_for_training, n_neg=args.n_neg)

    # Extract scores for only the live markets
    live_questions = set(df_combined["question"])
    df_live_results = df_for_training[df_for_training["question"].isin(live_questions)].copy()

    # Merge end_date so the dashboard can show time-to-resolution
    end_dates = df_markets[["question", "end_date"]].drop_duplicates("question")
    df_live_results = df_live_results.merge(end_dates, on="question", how="left")

    # Write outputs to both outputs/ and dashboard/public/ (for Vercel)
    root = os.path.dirname(__file__)
    for dest in [os.path.join(root, "outputs"), os.path.join(root, "dashboard", "public")]:
        os.makedirs(dest, exist_ok=True)
        df_live_results.to_csv(os.path.join(dest, "df_live.csv"), index=False)
        df_scored.to_csv(os.path.join(dest, "df_live_scored.csv"), index=False)

    print(f"\nOutputs written to outputs/ and dashboard/public/")

    # Print summary
    cols = ["question", "insider_trading_prob", "suspicion_score"]
    available = [c for c in cols if c in df_live_results.columns]

    print(f"\n{'='*70}")
    print(f"LIVE MARKET INSIDER TRADING RISK — top results (POC, not validated)")
    print(f"{'='*70}")
    top = df_live_results.dropna(subset=["insider_trading_prob"]).nlargest(15, "insider_trading_prob")
    if top.empty:
        print("No scored markets — check price history coverage.")
    else:
        for _, row in top.iterrows():
            prob   = row.get("insider_trading_prob", float("nan"))
            susp   = row.get("suspicion_score", float("nan"))
            q      = row["question"][:70]
            print(f"  {prob:.3f}  suspicion={susp:.3f}  {q}")
    print(f"{'='*70}\n")

    if args.push:
        push_to_github(df_live_results, df_scored, df_wallet_agg)


def _merge_labeled_features(df_combined: pd.DataFrame, df_labeled) -> pd.DataFrame:
    """
    Concat cached labeled case wallet features into df_combined, then
    drop_duplicates(subset=["question"], keep="first") so the top-50 pipeline
    row (with price features) wins over the labeled-only row (NaN price features)
    when a market appears in both.

    df_labeled columns: question, label, new_wallet_ratio, new_wallet_ratio_6h,
      burst_score, order_flow_imbalance, wallet_age_median_days,
      cross_market_wallet_flag.  Price feature columns are absent and become NaN
      after concat — they will be median-imputed at training time.

    Silently skips if df_labeled is None or empty (e.g. --refresh-labeled not
    yet run).
    """
    if df_labeled is None or (hasattr(df_labeled, "empty") and df_labeled.empty):
        return df_combined

    before = len(df_combined)
    combined = pd.concat([df_combined, df_labeled], ignore_index=True)
    combined = combined.drop_duplicates(subset=["question"], keep="first")
    added = len(combined) - before
    total = len(combined)
    print(f"  Added {added} labeled case rows to df_combined (total: {total})")
    return combined


def _preflight(df_combined):
    """Print NaN counts per model feature so you can see what's populated before training."""
    from backend.pipeline.scorer import MODEL_FEATURES, MODEL_WALLET_FEATURES
    print("\n=== Pre-flight: feature NaN counts ===")
    for feat in MODEL_FEATURES:
        if feat in df_combined.columns:
            n_null = df_combined[feat].isna().sum()
            n_ok   = df_combined[feat].notna().sum()
            tag    = " <- needs wallet query" if n_null > 0 and feat in MODEL_WALLET_FEATURES else ""
            print(f"  {feat:<25} {n_ok:>4} present  {n_null:>4} NaN{tag}")
        else:
            print(f"  {feat:<25}  MISSING COLUMN — check scorer.py")


def _write_outputs(df_combined, df_scored, df_wallet_agg):
    """Write CSVs to outputs/ and mirror to dashboard/public/ for local dev."""
    root = os.path.dirname(__file__)
    dirs = [
        os.path.join(root, "outputs"),
        os.path.join(root, "dashboard", "public"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    pairs = [
        (df_combined,   "df_combined.csv"),
        (df_scored,     "df_scored.csv"),
        (df_wallet_agg, "df_wallet_agg.csv"),
    ]
    for df, fname in pairs:
        if df is not None:
            for d in dirs:
                df.to_csv(os.path.join(d, fname), index=False)

    print(f"\nOutputs written to outputs/ and dashboard/public/")


if __name__ == "__main__":
    main()
