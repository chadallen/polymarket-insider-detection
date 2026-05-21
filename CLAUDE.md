# Insider Trading Detection — Developer Context

## What This Is

A proof-of-concept ML pipeline that detects potential insider trading on
[Polymarket](https://polymarket.com) by analyzing price anomalies and on-chain
wallet behavior in resolved political prediction markets.

---

## Stack

- **Python 3.14** — pipeline and ML
- **LightGBM, scikit-learn, pandas** — ensemble classifier and feature computation
- **Dune Analytics** — on-chain wallet data (`polymarket_polygon.market_trades`)
- **Polygonscan V2 API** — wallet age lookup (Polygon PoS chain)
- **Polymarket Gamma API + CLOB API** — market metadata and price history
- **React / Vite / Tailwind / Recharts** — dashboard frontend
- **Vercel** — dashboard hosting

---

## Run Commands

```bash
python3 run.py                             # Full pipeline (~25 min, ~5 Dune credits)
python3 run.py --skip-fetch                # Cached markets, fresh Dune + Polygonscan
python3 run.py --skip-fetch --skip-dune   # All cached, fresh Polygonscan only
python3 run.py --skip-dune                # Price signals only (0 credits)
python3 run.py --classifier-only          # Retrain ensemble only (~5 sec, 0 credits)
python3 run.py --classifier-only --push   # Retrain + update live dashboard
python3 run.py --live --hours-ahead 48    # POC: score open markets
cd dashboard && npm run dev               # Dashboard dev server: http://localhost:5173
cd dashboard && vercel --prod --yes       # Deploy dashboard
```

---

## Environment Variables

Required in `.env` at repo root (gitignored):

```
DUNE_API_KEY=...
GITHUB_TOKEN=...              # only needed for --push
GITHUB_REPO=chadallen/polymarket-insider-detection
GITHUB_BRANCH=main
TOP_N_MARKETS=50
POLYGONSCAN_API_KEY=...       # optional; free key at polygonscan.com
```

---

## Directory Structure

```
run.py                        # CLI entrypoint
backend/
  config.py                   # All tunable constants + env vars
  pipeline/
    fetcher.py                # Gamma API + CLOB API
    price_features.py         # Price features + IsolationForest scoring
    wallet_features.py        # Dune queries + wallet feature computation
    scorer.py                 # Ensemble classifier (PU-LightGBM + IsoForest + OC-SVM)
    polygonscan.py            # Wallet age lookup
    dune.py                   # Dune Analytics HTTP client
data/
  labeled_cases.csv           # 22 ground-truth cases (5 CONFIRMED, 4 SUSPECTED, 13 POSSIBLE)
  *.pkl                       # Cached pipeline intermediates
outputs/                      # CSV outputs (mirrored to dashboard/public/)
dashboard/                    # React frontend (Vite + Tailwind + Recharts)
  public/                     # CSVs read at runtime
docs/
  adr/                        # Architecture Decision Records
  plans/                      # Feature design docs
```

See `PRD.md` for full feature set, model architecture, and pipeline flow.

---

## Key Configuration

- `backend/config.py`: `MIN_END_DATE="2025-01-01"`, `TOP_N_MARKETS=50`, `MIN_VOLUME_USD=10M`, `PRICE_HOURS_BEFORE=48`
- `backend/pipeline/scorer.py`: ensemble weights pu=0.5, iso=0.05, ocsvm=0.2; label weights CONFIRMED=1.0, SUSPECTED=0.6, POSSIBLE=0.3
- `data/labeled_cases.csv`: add a row + run `--classifier-only` to incorporate new cases
- **Dune:** wallet query scales with market count. If resource cap hit, reduce `TOP_N_MARKETS` or check per-query credit limit in Dune account settings.
- **Vercel:** dashboard at https://dashboard-rouge-pi-13.vercel.app (`cgallen-1252s-projects/dashboard`), `rootDirectory=dashboard` set via REST API.
- **Worktrees:** need a `node_modules` symlink + `.claude/launch.json` with explicit vite path and non-conflicting port.

---

## Agent Behavior

- **Wait for approval before writing code**
- **Commit frequently** with task IDs: `git commit -m "<message> (<task-id>)"`
- **Do not read `scratch.md`**
- Task tracking: beads. Run `bd ready` for next tasks. Skills: /start-session, /end-session, /create-tasks, /build-tasks, /adr


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
