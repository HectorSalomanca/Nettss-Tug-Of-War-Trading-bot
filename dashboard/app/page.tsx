import { supabase, TugResult, Trade, RegimeState, OfiSnapshot } from "@/lib/supabase";
import TugMeter from "@/app/components/TugMeter";
import TradeRow from "@/app/components/TradeRow";
import { Activity, TrendingUp, TrendingDown, Zap, BarChart2, Brain, Waves } from "lucide-react";

async function getLatestTugResults(): Promise<TugResult[]> {
  const { data } = await supabase
    .from("tug_results")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(30);
  return (data as TugResult[]) ?? [];
}

async function getRecentTrades(): Promise<Trade[]> {
  const { data } = await supabase
    .from("trades")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(20);
  return (data as Trade[]) ?? [];
}

async function getLatestRegime(): Promise<RegimeState | null> {
  const { data } = await supabase
    .from("regime_states")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(1);
  return data?.[0] ?? null;
}

async function getLatestOfi(): Promise<Record<string, OfiSnapshot>> {
  const { data } = await supabase
    .from("ofi_snapshots")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(50);
  const map: Record<string, OfiSnapshot> = {};
  for (const row of (data as OfiSnapshot[]) ?? []) {
    if (!map[row.symbol]) map[row.symbol] = row;
  }
  return map;
}

function getStats(trades: Trade[], tugResults: TugResult[]) {
  const filled = trades.filter((t) => t.status === "filled");
  const totalPnl = filled.reduce((sum, t) => sum + (t.pnl ?? 0), 0);
  const wins = filled.filter((t) => (t.pnl ?? 0) > 0).length;
  const winRate = filled.length > 0 ? (wins / filled.length) * 100 : 0;
  const executed = tugResults.filter((r) => r.verdict === "execute").length;
  const crowded = tugResults.filter((r) => r.verdict === "crowded_skip").length;
  const avgShortfall = trades
    .filter((t) => t.implementation_shortfall_bps != null)
    .reduce((sum, t, _, arr) => sum + (t.implementation_shortfall_bps ?? 0) / arr.length, 0);
  return { totalPnl, winRate, executed, crowded, totalTrades: filled.length, avgShortfall };
}

const REGIME_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  trend:  { bg: "bg-emerald-500/20 border-emerald-500/40", text: "text-emerald-400", label: "TREND" },
  chop:   { bg: "bg-yellow-500/20 border-yellow-500/40",   text: "text-yellow-400",  label: "CHOP" },
  crisis: { bg: "bg-red-500/20 border-red-500/40",         text: "text-red-400",     label: "CRISIS" },
};

export const revalidate = 30;

export default async function Home() {
  const [tugResults, trades, regime, ofiMap] = await Promise.all([
    getLatestTugResults(),
    getRecentTrades(),
    getLatestRegime(),
    getLatestOfi(),
  ]);

  const stats = getStats(trades, tugResults);

  const latestBySymbol = new Map<string, TugResult>();
  for (const r of tugResults) {
    if (!latestBySymbol.has(r.symbol)) latestBySymbol.set(r.symbol, r);
  }
  const latestResults = Array.from(latestBySymbol.values());

  const regimeStyle = REGIME_STYLES[regime?.state ?? "trend"];

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-orange-500 flex items-center justify-center">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-white font-bold text-lg leading-none">Tug-Of-War</h1>
            <p className="text-zinc-500 text-xs mt-0.5">Sovereign Quant System V2</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {regime && (
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-bold ${regimeStyle.bg} ${regimeStyle.text}`}>
              <Brain className="w-3 h-3" />
              {regimeStyle.label}
              <span className="text-xs font-normal opacity-70">
                {(regime.confidence * 100).toFixed(0)}% conf
              </span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-zinc-400 text-sm">Paper Trading</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-10">

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <BarChart2 className="w-4 h-4 text-zinc-500" />
              <span className="text-zinc-500 text-xs uppercase tracking-widest">Total P&L</span>
            </div>
            <div className={`text-2xl font-bold ${stats.totalPnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {stats.totalPnl >= 0 ? "+" : ""}${stats.totalPnl.toFixed(2)}
            </div>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-4 h-4 text-zinc-500" />
              <span className="text-zinc-500 text-xs uppercase tracking-widest">Win Rate</span>
            </div>
            <div className="text-2xl font-bold text-white">{stats.winRate.toFixed(1)}%</div>
            <div className="text-zinc-500 text-xs mt-1">{stats.totalTrades} filled</div>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-4 h-4 text-zinc-500" />
              <span className="text-zinc-500 text-xs uppercase tracking-widest">Executed</span>
            </div>
            <div className="text-2xl font-bold text-emerald-400">{stats.executed}</div>
            <div className="text-zinc-500 text-xs mt-1">conflict trades</div>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <TrendingDown className="w-4 h-4 text-zinc-500" />
              <span className="text-zinc-500 text-xs uppercase tracking-widest">Crowded Skip</span>
            </div>
            <div className="text-2xl font-bold text-yellow-400">{stats.crowded}</div>
            <div className="text-zinc-500 text-xs mt-1">filtered out</div>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Waves className="w-4 h-4 text-zinc-500" />
              <span className="text-zinc-500 text-xs uppercase tracking-widest">Avg Shortfall</span>
            </div>
            <div className="text-2xl font-bold text-blue-400">{stats.avgShortfall.toFixed(1)}<span className="text-sm font-normal text-zinc-500 ml-1">bps</span></div>
            <div className="text-zinc-500 text-xs mt-1">implementation cost</div>
          </div>
        </div>

        {/* Regime detail bar */}
        {regime && (
          <div className={`rounded-xl border px-5 py-4 flex items-center gap-6 ${regimeStyle.bg}`}>
            <div className="flex items-center gap-2">
              <Brain className={`w-4 h-4 ${regimeStyle.text}`} />
              <span className={`font-bold text-sm ${regimeStyle.text}`}>HMM Regime: {regimeStyle.label}</span>
            </div>
            <div className="text-zinc-400 text-sm">SPY Vol: <span className="text-white">{(regime.spy_volatility * 100).toFixed(1)}%</span></div>
            <div className="text-zinc-400 text-sm">SPY Mom: <span className={regime.spy_momentum >= 0 ? "text-emerald-400" : "text-red-400"}>{regime.spy_momentum >= 0 ? "+" : ""}{(regime.spy_momentum * 100).toFixed(2)}%</span></div>
            <div className="text-zinc-400 text-sm">Confidence: <span className="text-white">{(regime.confidence * 100).toFixed(0)}%</span></div>
            <div className="text-zinc-500 text-xs ml-auto">
              {regime.state === "crisis" && "⚠️ All trading halted"}
              {regime.state === "chop" && "Scalp mode — tighter stops (1%/2%)"}
              {regime.state === "trend" && "Day-trade mode — full size (2%/4%)"}
            </div>
          </div>
        )}

        {/* Tug meters */}
        <section>
          <h2 className="text-zinc-300 font-semibold text-sm uppercase tracking-widest mb-4">
            Live Tug-Of-War Meters
          </h2>
          {latestResults.length === 0 ? (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-10 text-center text-zinc-600">
              No signals yet. Run the referee engine to start generating data.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {latestResults.map((r) => {
                const ofi = ofiMap[r.symbol];
                return (
                  <TugMeter
                    key={r.id}
                    symbol={r.symbol}
                    sovereignDir={r.sovereign_direction}
                    madmanDir={r.madman_direction}
                    sovereignConf={r.sovereign_confidence}
                    madmanConf={r.madman_confidence}
                    tugScore={r.tug_score}
                    verdict={r.verdict}
                    ofiZ={ofi?.ofi_z_score ?? null}
                    iceberg={ofi?.iceberg_detected ?? false}
                  />
                );
              })}
            </div>
          )}
        </section>

        {/* Trade history */}
        <section>
          <h2 className="text-zinc-300 font-semibold text-sm uppercase tracking-widest mb-4">
            Trade History
          </h2>
          {trades.length === 0 ? (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-10 text-center text-zinc-600">
              No trades executed yet.
            </div>
          ) : (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500 text-xs uppercase tracking-widest">
                    <th className="py-3 px-4 text-left">Symbol</th>
                    <th className="py-3 px-4 text-left">Side</th>
                    <th className="py-3 px-4 text-left">Qty</th>
                    <th className="py-3 px-4 text-left">Type</th>
                    <th className="py-3 px-4 text-left">Status</th>
                    <th className="py-3 px-4 text-left">Fill Price</th>
                    <th className="py-3 px-4 text-left">Shortfall</th>
                    <th className="py-3 px-4 text-left">P&L</th>
                    <th className="py-3 px-4 text-left">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((trade) => (
                    <TradeRow key={trade.id} trade={trade} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      <footer className="border-t border-zinc-800 px-6 py-4 text-center text-zinc-600 text-xs">
        Tug-Of-War Sovereign System V2 · Paper Trading · AFD + HMM + OFI
      </footer>
    </div>
  );
}
