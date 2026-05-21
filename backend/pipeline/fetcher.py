"""
Fetches market metadata (Gamma API) and price histories (CLOB API).
Corresponds to notebook Cells 3 and 4.
"""
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from backend.config import (
    FETCH_TAG_IDS, MARKETS_PER_PAGE, MAX_PAGES,
    MIN_VOLUME_USD, MIN_END_DATE, PRICE_HOURS_BEFORE,
    question_is_blocked,
)

TAG_NAMES = {
    2:      "Politics",
    100265: "Geopolitics",
    596:    "Culture",
    1401:   "Tech",
    101999: "Big Tech",
    107:    "Business",
    120:    "Finance",
    101970: "World",
    100328: "Economy",
}


def _parse_markets_from_events(events: list, seen_ids: set, tag_id: int = 0) -> list:
    """Extract market rows from Gamma API event objects."""
    rows = []
    for event in events:
        event["_tag_id"] = tag_id
        for mkt in event.get("markets", []):
            mid = mkt.get("conditionId")
            question = mkt.get("question", "")
            if not mid or mid in seen_ids or question_is_blocked(question):
                continue
            seen_ids.add(mid)
            try:
                token_ids = json.loads(mkt.get("clobTokenIds", "[]"))
                token_id = token_ids[0] if token_ids else None
            except Exception:
                token_id = None
            rows.append({
                "market_id":       mid,
                "question":        mkt.get("question", ""),
                "end_date":        mkt.get("endDate", ""),
                "volume":          float(mkt.get("volume") or 0),
                "resolution_time": mkt.get("endDate"),
                "token_id":        token_id,
                "category":        TAG_NAMES.get(event.get("_tag_id"), "unknown"),
                "event_title":     event.get("title", ""),
                "market_url":      f"https://polymarket.com/event/{event.get('slug', '')}" if event.get("slug") else "",
            })
    return rows


def fetch_markets() -> pd.DataFrame:
    """
    Fetch closed markets from Gamma API across all FETCH_TAG_IDS.
    Filters: volume >= MIN_VOLUME_USD, end_date >= MIN_END_DATE.
    Deduplicates on conditionId across all tags.
    """
    all_markets, seen_ids = [], set()
    tag_counts: dict[int, int] = {}

    print(f"Fetching closed markets across {len(FETCH_TAG_IDS)} tag categories...")

    for tag_id in FETCH_TAG_IDS:
        tag_name = TAG_NAMES.get(tag_id, str(tag_id))
        tag_start = len(all_markets)
        print(f"\n  [{tag_name}] tag_id={tag_id}")

        for page in range(MAX_PAGES):
            offset = page * MARKETS_PER_PAGE
            url = (
                f"https://gamma-api.polymarket.com/events"
                f"?tag_id={tag_id}&closed=true"
                f"&limit={MARKETS_PER_PAGE}&offset={offset}"
                f"&order=volume&ascending=false"
            )
            events = requests.get(url, timeout=30).json()
            if not events:
                print(f"    No more results at offset {offset}")
                break

            count_before = len(all_markets)
            all_markets.extend(_parse_markets_from_events(events, seen_ids, tag_id))
            new_count = len(all_markets) - count_before

            print(f"    Page {page + 1}: +{new_count} new | Running total: {len(all_markets)}")
            time.sleep(0.3)

        tag_counts[tag_id] = len(all_markets) - tag_start

    # Per-tag summary
    print(f"\n{'─' * 42}")
    print(f"  {'Tag':<12} {'ID':>7}  {'Markets':>8}")
    print(f"{'─' * 42}")
    for tag_id in FETCH_TAG_IDS:
        tag_name = TAG_NAMES.get(tag_id, str(tag_id))
        print(f"  {tag_name:<12} {tag_id:>7}  {tag_counts[tag_id]:>8}")
    print(f"{'─' * 42}")
    print(f"  {'TOTAL':<12} {'':>7}  {len(all_markets):>8} (before filters)")

    df = pd.DataFrame(all_markets)
    df = df[df["volume"] >= MIN_VOLUME_USD].reset_index(drop=True)
    df = df[df["end_date"] >= MIN_END_DATE].reset_index(drop=True)

    print(f"\n{len(df)} markets | volume >= ${MIN_VOLUME_USD:,} | end_date >= {MIN_END_DATE}")
    return df


def fetch_live_markets(hours_ahead: int = 48, min_volume: float = 1_000_000) -> pd.DataFrame:
    """
    Fetch open markets resolving within the next `hours_ahead` hours across
    all FETCH_TAG_IDS. Sets resolution_time = now so price histories are
    pulled up to the present. Filters: volume >= min_volume.
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    all_markets, seen_ids = [], set()
    print(
        f"Fetching live markets ending within {hours_ahead}h across "
        f"{len(FETCH_TAG_IDS)} tag categories (vol >= ${min_volume:,.0f})..."
    )

    for tag_id in FETCH_TAG_IDS:
        tag_name = TAG_NAMES.get(tag_id, str(tag_id))
        print(f"\n  [{tag_name}] tag_id={tag_id}")

        for page in range(MAX_PAGES):
            offset = page * MARKETS_PER_PAGE
            url = (
                f"https://gamma-api.polymarket.com/events"
                f"?tag_id={tag_id}&closed=false&active=true"
                f"&limit={MARKETS_PER_PAGE}&offset={offset}"
                f"&order=volume&ascending=false"
            )
            events = requests.get(url, timeout=30).json()
            if not events:
                break

            new_count_before = len(all_markets)
            all_markets.extend(_parse_markets_from_events(events, seen_ids, tag_id))
            new_count = len(all_markets) - new_count_before
            print(f"    Page {page + 1}: +{new_count} | Running total: {len(all_markets)}")
            time.sleep(0.3)

    if not all_markets:
        print("No live markets found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_markets)
    df = df[df["volume"] >= min_volume].reset_index(drop=True)

    # Keep only markets resolving within the window
    now_iso = now.isoformat()
    cutoff_iso = cutoff.isoformat()
    df = df[(df["end_date"] > now_iso) & (df["end_date"] <= cutoff_iso)].reset_index(drop=True)

    # Use current time as the effective resolution time so price histories
    # are fetched up to now rather than a future timestamp.
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df["resolution_time"] = now_str

    print(f"\n{len(df)} live markets ending within {hours_ahead}h | volume >= ${min_volume:,.0f}")
    return df


def fetch_price_history(
    token_id: str,
    resolution_time,
    hours_before: int = PRICE_HOURS_BEFORE,
    start_time=None,
) -> pd.DataFrame:
    """
    Fetch CLOB price history for a single market token.

    start_time: optional datetime or ISO string. If provided, overrides the
    hours_before calculation with min(start_time, resolution - hours_before)
    so labeled cases with early suspicious windows get full coverage.
    """
    if isinstance(resolution_time, str):
        res_time = datetime.fromisoformat(resolution_time.replace("Z", "+00:00"))
    else:
        res_time = resolution_time

    default_start = res_time - timedelta(hours=hours_before)

    if start_time is not None:
        if isinstance(start_time, str):
            st_str = start_time.strip()
            if "T" not in st_str:
                st_str += "T00:00:00+00:00"
            elif st_str.endswith("Z"):
                st_str = st_str[:-1] + "+00:00"
            elif "+" not in st_str and st_str[-6] != "+":
                st_str += "+00:00"
            parsed_start = datetime.fromisoformat(st_str)
        else:
            parsed_start = start_time
        # Use whichever start is earlier (never less than hours_before of history)
        effective_start = min(parsed_start, default_start)
    else:
        effective_start = default_start

    params = {
        "market":    token_id,
        "interval":  "max",
        "fidelity":  60,
        "startTs":   int(effective_start.timestamp()),
        "endTs":     int(res_time.timestamp()),
    }
    try:
        r = requests.get("https://clob.polymarket.com/prices-history", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data or "history" not in data or not data["history"]:
            return pd.DataFrame()
        df = pd.DataFrame(data["history"]).rename(columns={"t": "timestamp", "p": "price"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["price"] = df["price"].astype(float)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_price_histories(df_markets: pd.DataFrame) -> dict:
    """
    Fetch price histories for all markets. Skips uncontested markets
    (starting price outside 0.15–0.85).
    Returns dict: token_id -> DataFrame.
    """
    histories = {}
    print(f"Fetching price histories for {len(df_markets)} markets...")

    for i, (_, row) in enumerate(df_markets.iterrows()):
        if i % 25 == 0:
            print(f"  {i}/{len(df_markets)}...")
        if row["resolution_time"] is None:
            continue
        history = fetch_price_history(
            row["token_id"],
            row["resolution_time"],
            start_time=row.get("price_start_time"),
        )
        if len(history) < 3:
            continue
        # Bypass starting-price filter for labeled case markets — insider
        # trading cases often start at low probability (e.g. Maduro 0.7%,
        # Israel strike 7.5%) which would otherwise exclude them.
        if not row.get("is_labeled_case", False):
            if not (0.15 <= history["price"].iloc[0] <= 0.85):
                continue
        histories[row["token_id"]] = history

    print(f"\nPrice histories cached for {len(histories)}/{len(df_markets)} markets")
    return histories
