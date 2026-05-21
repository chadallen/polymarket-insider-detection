"""
Merges price + wallet features and trains the Phase 3 ensemble classifier.

Phase 3 model architecture:
  1. PU Learning     — LightGBM (two-step Elkan & Noto)
  2. Unified ISO     — IsolationForest on merged feature matrix
  3. One-Class SVM   — trained only on CONFIRMED labeled cases
  4. Ensemble        — 0.5 × pu_prob + 0.3 × iso_score + 0.2 × ocsvm_score

The price-only IsolationForest in price_features.py is still used upstream to
select which markets get Dune wallet queries (suspicion_score). The unified
IsolationForest here runs after merge_features() on all 9 features.

Soft label weights (CONFIRMED=1.0 / SUSPECTED=0.6 / POSSIBLE=0.3) are applied
as sample_weight in LightGBM training. The PU prior c is estimated from the
average model confidence on labeled positives after fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from backend.pipeline.wallet_features import load_labeled_cases, question_matches_filter


# ── Feature lists ─────────────────────────────────────────────────────────

MODEL_PRICE_FEATURES = [
    "surprise_score",
    "late_move_ratio",
    "price_volatility",
    "max_single_move",
    "total_price_move",
    "price_momentum_6h",
    "price_momentum_12h",
]

MODEL_WALLET_FEATURES = [
    "new_wallet_ratio",
    "new_wallet_ratio_6h",
    "burst_score",
    "order_flow_imbalance",
    "wallet_age_median_days",
    "cross_market_wallet_flag",
]

MODEL_FEATURES = MODEL_PRICE_FEATURES + MODEL_WALLET_FEATURES

# Backward-compat aliases (run.py imports RF_FEATURES, RF_WALLET_FEATURES)
RF_FEATURES        = MODEL_FEATURES
RF_PRICE_FEATURES  = MODEL_PRICE_FEATURES
RF_WALLET_FEATURES = MODEL_WALLET_FEATURES

# ── Label and ensemble configuration ──────────────────────────────────────

# Soft confidence weights applied as sample_weight during LightGBM training.
LABEL_WEIGHTS = {
    "CONFIRMED": 1.0,
    "SUSPECTED": 0.6,
    "POSSIBLE":  0.3,
}

# Ensemble component weights (must sum to 1.0).
# Calibrated by label type; adjust after Phase 5 leave-one-out CV.
ENSEMBLE_WEIGHTS = {
    "pu":    0.75,  # PU-LightGBM probability (primary)
    "iso":   0.05,  # Unified IsolationForest anomaly score
    "ocsvm": 0.2,   # One-Class SVM similarity to CONFIRMED cases
}


# ── Merge price + wallet features ─────────────────────────────────────────

def merge_features(df_scored: pd.DataFrame, df_wallet_agg: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge price features and wallet features into a single DataFrame.
    No intermediate scoring — the ensemble trains directly on raw features.
    suspicion_score (price-only IsolationForest) is preserved; it was used
    upstream to select markets for Dune wallet queries.
    """
    price_cols = [
        "question", "volume", "end_date", "market_url", "suspicion_score",
        "surprise_score", "late_move_ratio", "price_volatility",
        "max_single_move", "total_price_move",
        "price_momentum_6h", "price_momentum_12h",
    ]
    df_price = df_scored[[c for c in price_cols if c in df_scored.columns]].copy()

    df_wallet = (
        df_wallet_agg.copy()
        if df_wallet_agg is not None and not df_wallet_agg.empty
        else pd.DataFrame()
    )
    if not df_wallet.empty:
        numeric_cols = [
            "burst_score", "order_flow_imbalance", "directional_consensus",
            "new_wallet_ratio", "new_wallet_ratio_6h",
            "wallet_concentration", "wallet_age_median_days",
            "cross_market_wallet_flag",
            "total_volume", "trade_count", "unique_wallets",
        ]
        for col in numeric_cols:
            if col in df_wallet.columns:
                df_wallet[col] = pd.to_numeric(df_wallet[col], errors="coerce")

    wallet_cols = [
        "question", "new_wallet_ratio", "new_wallet_ratio_6h",
        "burst_score", "directional_consensus", "order_flow_imbalance",
        "wallet_concentration", "wallet_age_median_days",
        "cross_market_wallet_flag",
    ]
    df_combined = df_price.merge(
        df_wallet[[c for c in wallet_cols if c in df_wallet.columns]]
        if not df_wallet.empty
        else pd.DataFrame(columns=["question"]),
        on="question",
        how="left",
    )

    with_wallet = (
        df_combined["order_flow_imbalance"].notna().sum()
        if "order_flow_imbalance" in df_combined.columns else 0
    )
    print(f"df_combined: {len(df_combined)} markets ({with_wallet} with wallet data)")
    return df_combined


# ── Helpers ───────────────────────────────────────────────────────────────

def _normalize_0_1(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns 0.5 for constant arrays."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def _impute_wallet_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill NaNs for all model features with column medians. Returns (df, log_msgs)."""
    imputed = []
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        if df[col].isna().any():
            fill = df[col].median()
            fill = fill if pd.notna(fill) else 0.0
            df[col] = df[col].fillna(fill)
            imputed.append(f"{col}→{fill:.3f}")
    return df, imputed


def _label_positives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row with label_weight and is_confirmed from labeled_cases.csv.
    is_positive: any matched case (weight > 0).
    is_confirmed: matched a CONFIRMED case (weight == 1.0).
    """
    labeled_df = load_labeled_cases()

    def _weight(question: str) -> tuple[float, bool]:
        best_w, best_confirmed = 0.0, False
        for _, case in labeled_df.iterrows():
            if question_matches_filter(question, case["question_filter"]):
                w = LABEL_WEIGHTS.get(case["label"], 0.0)
                if w > best_w:
                    best_w = w
                    best_confirmed = (case["label"] == "CONFIRMED")
        return best_w, best_confirmed

    weights_confirmed = df["question"].apply(_weight)
    df["label_weight"]  = [x[0] for x in weights_confirmed]
    df["is_confirmed"]  = [x[1] for x in weights_confirmed]
    df["is_positive"]   = df["label_weight"] > 0
    return df


# ── Ensemble classifier ───────────────────────────────────────────────────

def train_classifier(
    df_combined: pd.DataFrame,
    features: list[str] | None = None,
    n_neg: int = 30,       # unused in PU learning; kept for API compatibility
) -> tuple[pd.DataFrame, object, object, list[str]]:
    """
    Train the Phase 3 ensemble on df_combined.
    Returns (df_with_probs, model_bundle, scaler, active_features).

    model_bundle = {"lgbm": ..., "iso": ..., "ocsvm": ..., "c": float}

    Positives are all cases in labeled_cases.csv with soft sample weights.
    Unlabeled markets (all others) are class 0 in PU training — they are NOT
    assumed to be clean negatives.

    Two-step Elkan & Noto adjustment:
      raw_prob = lgbm.predict_proba(X)[:, 1]
      c        = mean(raw_prob[positives])        # P(labeled | positive)
      pu_prob  = clip(raw_prob / c, 0, 1)

    Ensemble:
      insider_trading_prob = 0.5 * pu_prob + 0.3 * iso_score + 0.2 * ocsvm_score
    """
    if features is None:
        features = MODEL_FEATURES

    df = df_combined.copy()

    # ── Step 1: Ensure all expected feature columns exist, then impute ───
    for feat in features:
        if feat not in df.columns:
            df[feat] = np.nan
    df, imputed_cols = _impute_wallet_features(df)
    if imputed_cols:
        print(f"  Imputed NaNs:      {', '.join(imputed_cols)}")
    else:
        print("  Imputed NaNs:      none (all wallet features present)")

    # ── Step 2: Label positives from labeled_cases.csv ────────────────────
    df = _label_positives(df)

    n_pos       = df["is_positive"].sum()
    n_confirmed = df["is_confirmed"].sum()
    print(f"  Positives matched: {n_pos} ({n_confirmed} CONFIRMED)")
    if n_pos > 0:
        for _, row in df[df["is_positive"]].iterrows():
            tag = " [CONFIRMED]" if row["is_confirmed"] else ""
            print(f"    + [{row['label_weight']:.1f}] {str(row['question'])[:80]}{tag}")
    else:
        print("  WARNING: No labeled cases matched markets in df_combined.")
        print("  Check that labeled_cases.csv question_filters match current market questions.")
        df["insider_trading_prob"] = np.nan
        return df, None, None, features

    # ── Step 3: Drop zero-variance features ───────────────────────────────
    print(f"\n  {'Feature':<25} {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print("  " + "─" * 50)
    active_features = []
    df_feat = df.dropna(subset=features)
    for feat in features:
        vals = df_feat[feat]
        std  = vals.std()
        mn, mx = vals.min(), vals.max()
        if std < 1e-6:
            print(f"  {feat:<25} {std:>7.4f}  {mn:>7.4f}  {mx:>7.4f}  ZERO VARIANCE — dropping")
        else:
            print(f"  {feat:<25} {std:>7.4f}  {mn:>7.4f}  {mx:>7.4f}")
            active_features.append(feat)

    if not active_features:
        print("  All features have zero variance — cannot train.")
        df["insider_trading_prob"] = np.nan
        return df, None, None, features

    if len(active_features) < len(features):
        dropped = set(features) - set(active_features)
        print(f"\n  Dropped {len(dropped)} zero-variance feature(s): {dropped}")

    # ── Step 4: Scale all markets ─────────────────────────────────────────
    df_scoreable = df.dropna(subset=active_features).copy()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_scoreable[active_features].values)
    # Named DataFrame for LightGBM (preserves feature names across fit/predict)
    X_lgbm   = pd.DataFrame(X_scaled, columns=active_features)
    X_all    = X_scaled   # numpy array for IsolationForest / OC-SVM
    pos_mask = df_scoreable["is_positive"].values
    confirmed_mask = df_scoreable["is_confirmed"].values

    # Determine which rows are labeled-only (backward compat: treat missing column as False)
    if "is_labeled_only" in df_scoreable.columns:
        labeled_only_mask = df_scoreable["is_labeled_only"].fillna(False).astype(bool).values
    else:
        labeled_only_mask = np.zeros(len(df_scoreable), dtype=bool)

    # ── Step 5: PU Learning — LightGBM (two-step Elkan & Noto) ───────────
    # The unlabeled pool is current pipeline markets only: rows that are NOT
    # positive AND NOT labeled-only.  Labeled-only rows that are NOT matched as
    # positives must be excluded so they don't corrupt the unlabeled set when
    # the positive/unlabeled ratio is inverted.
    print("\n  [1/3] PU Learning — LightGBM")

    # Build training mask: positives + unlabeled pipeline markets (exclude
    # labeled-only rows that are not positives)
    unlabeled_mask = ~pos_mask & ~labeled_only_mask
    train_mask = pos_mask | unlabeled_mask
    n_unlabeled    = int(unlabeled_mask.sum())
    n_labeled_only = int((labeled_only_mask & ~pos_mask).sum())

    print(f"     {pos_mask.sum()} positives | {n_unlabeled} unlabeled | "
          f"{n_labeled_only} labeled-only (excluded from negatives)")

    X_train = X_lgbm[train_mask]
    y_train = pos_mask[train_mask].astype(int)
    sample_weights = np.where(
        pos_mask[train_mask],
        df_scoreable["label_weight"].values[train_mask],
        1.0,
    )

    lgbm_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_samples=1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgbm_model.fit(X_train, y_train, sample_weight=sample_weights)

    raw_pu = lgbm_model.predict_proba(X_lgbm)[:, 1]

    # Estimate c = P(labeled | positive) from positives' predicted probabilities
    c = float(raw_pu[pos_mask].mean()) if pos_mask.sum() > 0 else 1.0
    c = max(c, 0.01)   # guard against division by near-zero
    pu_prob = np.clip(raw_pu / c, 0.0, 1.0)
    df_scoreable["pu_prob"] = pu_prob

    # ── Step 6: Unified IsolationForest ───────────────────────────────────
    print("  [2/3] Unified IsolationForest")
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X_all)
    iso_score = _normalize_0_1(-iso.decision_function(X_all))
    df_scoreable["iso_score"] = iso_score

    # ── Step 7: One-Class SVM (CONFIRMED cases only) ──────────────────────
    print("  [3/3] One-Class SVM")
    ocsvm_score = np.full(len(df_scoreable), 0.5)   # neutral default
    ocsvm = None
    if confirmed_mask.sum() >= 1:
        ocsvm = OneClassSVM(nu=0.5, kernel="rbf", gamma="scale")
        ocsvm.fit(X_all[confirmed_mask])
        ocsvm_raw   = ocsvm.decision_function(X_all)
        ocsvm_score = _normalize_0_1(ocsvm_raw)
        print(f"     Trained on {confirmed_mask.sum()} CONFIRMED cases")
    else:
        print(f"     Only {confirmed_mask.sum()} CONFIRMED case(s) matched — skipping "
              f"(need ≥ 1); using neutral 0.5 for ocsvm_score")
    df_scoreable["ocsvm_score"] = ocsvm_score

    # ── Step 8: Ensemble ──────────────────────────────────────────────────
    w = ENSEMBLE_WEIGHTS
    df_scoreable["insider_trading_prob"] = np.clip(
        w["pu"]    * df_scoreable["pu_prob"]    +
        w["iso"]   * df_scoreable["iso_score"]  +
        w["ocsvm"] * df_scoreable["ocsvm_score"],
        0.0, 1.0,
    )

    # ── Merge scores back into df ─────────────────────────────────────────
    score_cols = ["pu_prob", "iso_score", "ocsvm_score", "insider_trading_prob"]
    for col in score_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.join(df_scoreable[score_cols], how="left")

    # ── Diagnostics ───────────────────────────────────────────────────────
    print("\nLightGBM feature importances (gain):")
    importances = lgbm_model.feature_importances_
    for feat, imp in sorted(zip(active_features, importances), key=lambda x: -x[1]):
        bar = "█" * max(1, int(imp / max(importances) * 30))
        print(f"  {feat:<25} {bar} {imp:.1f}")

    print("\nTop 15 by insider_trading_prob:")
    top = (
        df[df["insider_trading_prob"].notna()]
        .nlargest(15, "insider_trading_prob")
        [["question", "insider_trading_prob", "pu_prob", "iso_score", "ocsvm_score", "suspicion_score"]]
        .reset_index(drop=True)
    )
    top.index += 1
    with pd.option_context("display.max_colwidth", 60):
        print(top.to_string())

    n_scored = df["insider_trading_prob"].notna().sum()
    print(f"\nScored {n_scored}/{len(df)} markets")

    model_bundle = {
        "lgbm":  lgbm_model,
        "iso":   iso,
        "ocsvm": ocsvm,
        "c":     c,
    }
    return df, model_bundle, scaler, active_features
