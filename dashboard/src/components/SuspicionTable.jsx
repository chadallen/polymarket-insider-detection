import { useState } from 'react'
import SignalRadar from './SignalRadar'

const getScore = (row) => row.insider_trading_prob ?? row.combined_score ?? 0

const level = (row) => {
  const s = getScore(row)
  return s >= 0.35 ? 'high' : s >= 0.25 ? 'medium' : 'low'
}

const LEVEL_COLORS = {
  high:   { bar: 'bg-rose-500',   badge: 'border border-rose-200 text-rose-600 bg-rose-50',   score: 'text-rose-600'   },
  medium: { bar: 'bg-amber-500',  badge: 'border border-amber-200 text-amber-600 bg-amber-50', score: 'text-amber-600'  },
  low:    { bar: 'bg-zinc-300',   badge: 'border border-zinc-200 text-zinc-500 bg-zinc-50',    score: 'text-zinc-500'   },
}

// ── formatting helpers ────────────────────────────────────────────────────────
const fmtVol = (n) =>
  n == null ? '—'
  : n >= 1_000_000 ? `$${(n / 1_000_000).toFixed(2)}M`
  : n >= 1_000     ? `$${(n / 1_000).toFixed(1)}K`
  :                  `$${Number(n).toFixed(0)}`

const fmtPct  = (n) => (n == null ? '—' : `${(n * 100).toFixed(1)}%`)
const fmtNum  = (n, dec = 3) => (n == null ? '—' : Number(n).toFixed(dec))
const fmtDate = (s) => {
  if (s == null || s === '') return '—'
  const d = new Date(String(s).replace(' UTC', 'Z'))
  return isNaN(d) ? String(s) : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

// ── column definitions ────────────────────────────────────────────────────────
const COLUMNS = [
  { key: null,                   label: '#',            width: 'w-7',    align: 'text-left'   },
  { key: null,                   label: 'Market',       width: 'flex-1', align: 'text-left'   },
  { key: null,                   label: 'Resolved',     width: 'w-28',   align: 'text-right'  },
  { key: 'suspicion_score',      label: 'Price IF',     width: 'w-28',   align: 'text-right', tooltip: 'IsolationForest on price features only' },
  { key: 'iso_score',            label: 'IsoForest',    width: 'w-28',   align: 'text-right', tooltip: 'IsolationForest on all 14 features' },
  { key: 'pu_prob',              label: 'PU-LGB',       width: 'w-20',   align: 'text-right', tooltip: 'PU-LightGBM adjusted probability' },
  { key: 'insider_trading_prob', label: 'Score',        width: 'w-28',   align: 'text-right', tooltip: 'Ensemble score: 0.5×PU + 0.3×ISO + 0.2×OCSVM' },
]

// ── sub-components ────────────────────────────────────────────────────────────
function ChevronIcon({ open }) {
  return (
    <svg
      className={`w-5 h-5 text-blue-800 transition-transform duration-200 shrink-0 ${open ? 'rotate-0' : '-rotate-90'}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
    >
      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function SortIcon({ direction }) {
  if (!direction) return <span className="text-zinc-300 ml-0.5 text-[10px]">↕</span>
  return <span className="text-orange-500 ml-0.5 text-[10px]">{direction === 'asc' ? '↑' : '↓'}</span>
}

function MiniBar({ value, colorClass }) {
  if (value == null || isNaN(value)) return <span className="text-zinc-300 text-xs">—</span>
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="flex items-center gap-1.5 justify-end">
      <div className="w-12 h-1.5 bg-zinc-100 rounded-full overflow-hidden shrink-0">
        <div className={`h-full rounded-full ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="tabular-nums text-zinc-500 text-xs w-10 text-right font-mono">{Number(value).toFixed(3)}</span>
    </div>
  )
}

function ScoreBreakdown({ row }) {
  const pu    = row.pu_prob    ?? null
  const iso   = row.iso_score  ?? null
  const ocsvm = row.ocsvm_score ?? null
  const ens   = getScore(row)
  const lvl   = level(row)

  const bars = [
    { label: 'PU-LightGBM', value: pu,    weight: '×0.5', color: 'bg-indigo-500' },
    { label: 'IsoForest',   value: iso,   weight: '×0.3', color: 'bg-violet-400' },
    { label: 'OC-SVM',      value: ocsvm, weight: '×0.2', color: 'bg-purple-400' },
  ]

  return (
    <div className="mb-5 p-4 rounded-lg border border-zinc-100 bg-zinc-50">
      <div className="flex items-baseline justify-between mb-3">
        <span className="text-[10px] font-medium text-zinc-400 uppercase tracking-wide">Ensemble Score</span>
        <span className={`text-2xl font-bold tabular-nums ${LEVEL_COLORS[lvl].score}`}>
          {(ens * 100).toFixed(1)}%
        </span>
      </div>
      <div className="space-y-2">
        {bars.map(({ label, value, weight, color }) => {
          const pct = value != null ? Math.max(0, Math.min(1, value)) * 100 : null
          return (
            <div key={label} className="flex items-center gap-2">
              <span className="text-xs text-zinc-400 w-24 shrink-0">{label} <span className="text-zinc-300">{weight}</span></span>
              <div className="flex-1 h-2 bg-white rounded-full overflow-hidden border border-zinc-200">
                {pct != null
                  ? <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
                  : <div className="h-full rounded-full bg-zinc-100" style={{ width: '50%' }} />
                }
              </div>
              <span className="tabular-nums text-xs text-zinc-500 w-10 text-right font-mono">
                {value != null ? value.toFixed(3) : '0.5*'}
              </span>
            </div>
          )
        })}
      </div>
      {ocsvm == null && (
        <p className="text-[10px] text-zinc-400 mt-2">* OC-SVM fell back to neutral (insufficient CONFIRMED matches)</p>
      )}
    </div>
  )
}

function isAnomalous(field, value) {
  if (value == null) return false
  switch (field) {
    case 'wallet_concentration':     return value > 0.4
    case 'order_flow_imbalance':     return value > 0.7 || value < -0.7
    case 'burst_score':              return value > 40
    case 'new_wallet_ratio':         return value > 0.5
    case 'cross_market_wallet_flag': return value > 2
    case 'wallet_age_median_days':   return value < 30
    default: return false
  }
}

function isAnomalousPrice(field, value) {
  if (value == null) return false
  switch (field) {
    case 'surprise_score':    return value > 0.6
    case 'late_move_ratio':   return value > 0.5
    case 'max_single_move':   return value > 0.2
    case 'price_momentum_6h': return Math.abs(value) > 0.15
    default: return false
  }
}

function DetailRow({ label, value, anomalous }) {
  return (
    <div className={`flex justify-between items-baseline gap-2 py-1.5 border-b border-zinc-100 last:border-0 ${anomalous ? 'rounded px-1 -mx-1 bg-amber-50' : ''}`}>
      <span className="text-zinc-400 text-xs">{label}</span>
      <span className={`text-xs tabular-nums font-mono text-right ${anomalous ? 'text-amber-600 font-semibold' : 'text-zinc-700'}`}>
        {value}
        {anomalous && <span className="ml-1 text-amber-400">▲</span>}
      </span>
    </div>
  )
}

function DetailSection({ title, children }) {
  return (
    <div>
      <p className="text-[10px] font-medium text-zinc-400 uppercase tracking-wide mb-2 pb-1 border-b border-zinc-200">{title}</p>
      {children}
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────
export default function SuspicionTable({ data, scored = {}, wallet = {}, onRowClick, selected }) {
  const [sortKey, setSortKey] = useState('insider_trading_prob')
  const [sortDir, setSortDir] = useState('desc')

  function handleSort(key) {
    if (!key) return
    if (key === sortKey) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sorted = [...data].sort((a, b) => {
    const av = a[sortKey] ?? -Infinity
    const bv = b[sortKey] ?? -Infinity
    return sortDir === 'desc' ? bv - av : av - bv
  })

  if (!data.length) return (
    <div className="px-6 py-12 text-center text-zinc-400 text-sm">No market data loaded.</div>
  )

  return (
    <div>
      {/* ── Column header row ── */}
      <div className="hidden sm:flex items-center gap-x-3 px-4 py-2 border-b border-zinc-100 bg-zinc-50 text-[11px] font-medium text-zinc-400 uppercase tracking-wide select-none">
        <span className="w-20 shrink-0"></span>
        <span className="flex-1">Market</span>
        {COLUMNS.slice(2).map((col) => (
          <button
            key={col.label}
            onClick={() => handleSort(col.key)}
            disabled={!col.key}
            title={col.tooltip}
            className={`${col.width} shrink-0 ${col.align} flex items-center justify-end gap-0.5 ${
              col.key ? 'hover:text-orange-600 cursor-pointer' : 'cursor-default'
            } ${sortKey === col.key ? 'text-orange-500' : ''}`}
          >
            {col.label}
            {col.key && <SortIcon direction={sortKey === col.key ? sortDir : null} />}
          </button>
        ))}
      </div>

      {/* ── Rows ── */}
      <div className="divide-y divide-zinc-100">
        {sorted.map((row, i) => {
          const lvl    = level(row)
          const isOpen = selected?.question === row.question
          const s      = scored[row.question]
          const w      = wallet[row.question]
          const colors = LEVEL_COLORS[lvl]

          const priceIF = row.suspicion_score != null
            ? Math.max(0, Math.min(1, row.suspicion_score + 0.5))
            : null

          return (
            <div key={i}>
              {/* ── Collapsed row ── */}
              <button
                onClick={() => onRowClick(isOpen ? null : row)}
                className={`group w-full text-left px-4 py-2.5 transition-colors duration-100 cursor-pointer ${
                  isOpen ? 'bg-orange-50/40' : 'hover:bg-blue-50'
                }`}
                aria-expanded={isOpen}
              >
                {/* Desktop */}
                <div className="hidden sm:flex items-center gap-x-3">
                  <span className="w-20 shrink-0 flex items-center gap-1">
                    <ChevronIcon open={isOpen} />
                    <span className="text-[10px] font-mono text-blue-800 uppercase tracking-wide">details</span>
                  </span>
                  <span
                    className="flex-1 text-zinc-800 text-[13px] truncate pr-2 group-hover:underline"
                    title={row.question}
                  >
                    {row.question}
                  </span>

                  <div className="w-28 shrink-0 text-right">
                    <span className="tabular-nums text-xs text-zinc-400 font-mono">
                      {fmtDate(row.end_date)}
                    </span>
                  </div>

                  <div className="w-28 shrink-0">
                    <MiniBar value={priceIF} colorClass="bg-zinc-400" />
                  </div>
                  <div className="w-28 shrink-0">
                    <MiniBar value={row.iso_score} colorClass="bg-violet-400" />
                  </div>
                  <div className="w-20 shrink-0 text-right">
                    <span className="tabular-nums text-xs text-zinc-400 font-mono">
                      {row.pu_prob != null ? row.pu_prob.toFixed(3) : '—'}
                    </span>
                  </div>
                  <div className="w-28 shrink-0 text-right">
                    <span className={`tabular-nums font-bold text-sm ${colors.score}`}>
                      {(getScore(row) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Mobile */}
                <div className="sm:hidden flex items-start gap-3">
                  <ChevronIcon open={isOpen} />
                  <span className="text-zinc-500 text-xs tabular-nums mt-0.5 shrink-0 w-5">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-zinc-800 text-sm leading-snug mb-1.5 group-hover:underline">{row.question}</p>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`tabular-nums font-bold text-sm ${colors.score}`}>
                        {(getScore(row) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </button>

              {/* ── Expanded detail panel ── */}
              {isOpen && (
                <div className="px-4 pb-5 pt-3 bg-white border-t border-zinc-100">
                  <div className="mb-3">
                    <a
                      href={row.market_url || `https://polymarket.com/markets?_q=${encodeURIComponent(row.question)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 text-xs font-mono text-blue-600 hover:text-blue-800 hover:underline"
                    >
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                        <polyline points="15 3 21 3 21 9" />
                        <line x1="10" y1="14" x2="21" y2="3" />
                      </svg>
                      View on Polymarket
                    </a>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-4 mb-5">
                    <ScoreBreakdown row={row} />
                    <SignalRadar row={row} />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
                    {s ? (
                      <DetailSection title="Price Signals">
                        <DetailRow label="Resolved"         value={fmtDate(row.end_date)} />
                        <DetailRow label="Volume"           value={fmtVol(s.volume)} />
                        <DetailRow label="Starting Price"   value={fmtNum(s.starting_price)} />
                        <DetailRow label="Final Price"      value={fmtNum(s.final_price)} />
                        <DetailRow label="Total Price Move" value={fmtPct(s.total_price_move)}   anomalous={isAnomalousPrice('total_price_move', s.total_price_move)} />
                        <DetailRow label="Max Single Move"  value={fmtPct(s.max_single_move)}    anomalous={isAnomalousPrice('max_single_move', s.max_single_move)} />
                        <DetailRow label="Surprise Score"   value={fmtNum(s.surprise_score)}     anomalous={isAnomalousPrice('surprise_score', s.surprise_score)} />
                        <DetailRow label="Late Move Ratio"  value={fmtNum(s.late_move_ratio)}    anomalous={isAnomalousPrice('late_move_ratio', s.late_move_ratio)} />
                        <DetailRow label="Momentum 6h"      value={fmtNum(s.price_momentum_6h)}  anomalous={isAnomalousPrice('price_momentum_6h', s.price_momentum_6h)} />
                        <DetailRow label="Momentum 12h"     value={fmtNum(s.price_momentum_12h)} />
                        <DetailRow label="Price Volatility" value={fmtNum(s.price_volatility)} />
                        <DetailRow label="Price IF Flag"    value={s.anomaly_score === -1 ? 'ANOMALOUS' : 'normal'} anomalous={s.anomaly_score === -1} />
                      </DetailSection>
                    ) : (
                      <DetailSection title="Price Signals">
                        <p className="text-zinc-300 text-xs py-2">No price data available.</p>
                      </DetailSection>
                    )}

                    {w ? (
                      <DetailSection title="Wallet Signals">
                        <DetailRow label="Unique Wallets"       value={w.unique_wallets} />
                        <DetailRow label="Trade Count"          value={w.trade_count} />
                        <DetailRow label="Total Volume"         value={fmtVol(w.total_volume)} />
                        <DetailRow label="New Wallet Ratio"     value={fmtPct(w.new_wallet_ratio)}     anomalous={isAnomalous('new_wallet_ratio', w.new_wallet_ratio)} />
                        <DetailRow label="New Wallet Ratio 6h"  value={fmtPct(w.new_wallet_ratio_6h)} />
                        <DetailRow label="Order Flow Imbalance" value={fmtNum(w.order_flow_imbalance)} anomalous={isAnomalous('order_flow_imbalance', w.order_flow_imbalance)} />
                        <DetailRow label="Burst Score"          value={w.burst_score != null ? fmtNum(w.burst_score, 1) : '—'} anomalous={isAnomalous('burst_score', w.burst_score)} />
                        <DetailRow label="Wallet Concentration" value={fmtNum(w.wallet_concentration)} anomalous={isAnomalous('wallet_concentration', w.wallet_concentration)} />
                        <DetailRow label="Wallet Age (median)"  value={w.wallet_age_median_days != null ? `${Math.round(w.wallet_age_median_days)}d` : '—'} anomalous={isAnomalous('wallet_age_median_days', w.wallet_age_median_days)} />
                        <DetailRow label="Cross-Market Wallets" value={w.cross_market_wallet_flag ?? '—'} anomalous={isAnomalous('cross_market_wallet_flag', w.cross_market_wallet_flag)} />
                      </DetailSection>
                    ) : (
                      <DetailSection title="Wallet Signals">
                        <p className="text-zinc-300 text-xs py-2">No wallet data available.</p>
                      </DetailSection>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
