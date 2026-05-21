import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ZAxis,
} from 'recharts'

const getScore = (d) => d.insider_trading_prob ?? d.combined_score ?? 0

const getColor = (score) =>
  score >= 0.35 ? '#e11d48' : score >= 0.25 ? '#d97706' : '#94a3b8'

const fmtVol = (v) =>
  v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B`
  : v >= 1e6 ? `$${(v / 1e6).toFixed(0)}M`
  : v >= 1e3 ? `$${(v / 1e3).toFixed(0)}K`
  : `$${v}`

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  const score = getScore(d)
  return (
    <div className="border border-zinc-200 rounded p-3 shadow-lg text-xs max-w-[260px] bg-white">
      <p className="text-zinc-700 font-medium mb-2 leading-snug">{d.question}</p>
      <div className="space-y-0.5 font-mono">
        <div className="flex justify-between gap-6">
          <span className="text-zinc-400">Volume</span>
          <span className="text-zinc-600">{fmtVol(d.volume)}</span>
        </div>
        <div className="flex justify-between gap-6">
          <span className="text-zinc-400">Ensemble</span>
          <span style={{ color: getColor(score) }} className="font-semibold">
            {(score * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  )
}

// Log-scale tick values and labels
const LOG_TICKS = [1e6, 1e7, 1e8, 1e9]

export default function VolumeScatter({ data }) {
  const plotData = data
    .filter((d) => d.volume != null && d.volume > 0)
    .map((d) => ({
      ...d,
      logVol: Math.log10(d.volume),
      score: getScore(d),
    }))

  if (!plotData.length) return (
    <div className="flex items-center justify-center h-full text-zinc-300 text-xs font-mono">
      no data
    </div>
  )

  const logDomain = [
    6, // fixed at $1M minimum
    Math.ceil(Math.log10(Math.max(...plotData.map((d) => d.volume)) * 2)),
  ]

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-4 mb-2 text-[10px] font-mono text-zinc-500">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-rose-600 inline-block" /> High ≥35%</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-amber-600 inline-block" /> Medium ≥25%</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-slate-400 inline-block" /> Low</span>
      </div>
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 12, right: 20, bottom: 28, left: 8 }}>
        <CartesianGrid stroke="#f4f4f5" />
        <XAxis
          type="number"
          dataKey="logVol"
          domain={logDomain}
          ticks={LOG_TICKS.map((v) => Math.log10(v)).filter((v) => v >= logDomain[0] && v <= logDomain[1])}
          tickFormatter={(v) => fmtVol(Math.pow(10, v))}
          tick={{ fill: '#3f3f46', fontSize: 10, fontFamily: 'ui-monospace, monospace' }}
          tickLine={false}
          axisLine={{ stroke: '#e4e4e7' }}
          label={{ value: 'market volume (log scale)', position: 'insideBottom', offset: -14, fill: '#3f3f46', fontSize: 10 }}
        />
        <YAxis
          type="number"
          dataKey="score"
          domain={[0, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fill: '#3f3f46', fontSize: 10, fontFamily: 'ui-monospace, monospace' }}
          tickLine={false}
          axisLine={false}
          label={{ value: 'ensemble score', angle: -90, position: 'insideLeft', offset: 14, fill: '#3f3f46', fontSize: 10 }}
        />
        <ZAxis range={[44, 44]} />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#e4e4e7', strokeDasharray: '3 3' }} />
        <ReferenceLine y={0.35} stroke="#fca5a5" strokeDasharray="4 3" strokeWidth={1} />
        <ReferenceLine y={0.25} stroke="#fde68a" strokeDasharray="4 3" strokeWidth={1} />
        <Scatter
          data={plotData}
          shape={(props) => {
            const { cx, cy, payload } = props
            const color = getColor(getScore(payload))
            return (
              <circle
                cx={cx}
                cy={cy}
                r={5}
                fill={color}
                fillOpacity={0.8}
                stroke="white"
                strokeWidth={1.5}
              />
            )
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
    </div>
  )
}
