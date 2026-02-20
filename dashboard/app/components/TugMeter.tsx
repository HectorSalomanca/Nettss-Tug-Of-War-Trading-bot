"use client";

type Props = {
  symbol: string;
  sovereignDir: string;
  madmanDir: string;
  sovereignConf: number;
  madmanConf: number;
  tugScore: number;
  verdict: string;
  ofiZ: number | null;
  iceberg: boolean;
};

const directionColor = (dir: string) => {
  if (dir === "buy") return "text-emerald-400";
  if (dir === "sell") return "text-red-400";
  return "text-zinc-400";
};

const verdictBadge = (verdict: string) => {
  if (verdict === "execute")
    return "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
  if (verdict === "crowded_skip")
    return "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40";
  return "bg-zinc-700/40 text-zinc-400 border border-zinc-600/40";
};

const verdictLabel = (verdict: string) => {
  if (verdict === "execute") return "EXECUTE";
  if (verdict === "crowded_skip") return "CROWDED — SKIP";
  return "NO SIGNAL";
};

export default function TugMeter({
  symbol,
  sovereignDir,
  madmanDir,
  sovereignConf,
  madmanConf,
  tugScore,
  verdict,
  ofiZ,
  iceberg,
}: Props) {
  const tugPct = Math.round(tugScore * 100);
  const sovereignPct = Math.round(sovereignConf * 100);
  const madmanPct = Math.round(madmanConf * 100);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-white font-bold text-lg tracking-wide">{symbol}</span>
        <span className={`text-xs font-semibold px-3 py-1 rounded-full ${verdictBadge(verdict)}`}>
          {verdictLabel(verdict)}
        </span>
      </div>

      {(ofiZ !== null || iceberg) && (
        <div className="flex items-center gap-3 text-xs">
          <span className="text-zinc-500">OFI Z:</span>
          <span className={ofiZ !== null && ofiZ < -1.5 ? "text-orange-400 font-bold" : ofiZ !== null && ofiZ > 1.5 ? "text-emerald-400 font-bold" : "text-zinc-300"}>
            {ofiZ !== null ? (ofiZ >= 0 ? "+" : "") + ofiZ.toFixed(2) : "—"}
          </span>
          {iceberg && (
            <span className="ml-1 px-2 py-0.5 rounded-full bg-orange-500/20 border border-orange-500/40 text-orange-400 font-semibold">
              🧊 ICEBERG
            </span>
          )}
        </div>
      )}

      <div className="flex items-center gap-3">
        <div className="flex-1 text-center">
          <div className="text-xs text-zinc-500 mb-1 uppercase tracking-widest">Sovereign</div>
          <div className={`text-xl font-bold ${directionColor(sovereignDir)}`}>
            {sovereignDir.toUpperCase()}
          </div>
          <div className="text-xs text-zinc-400 mt-1">{sovereignPct}% conf</div>
        </div>

        <div className="flex flex-col items-center gap-1 px-2">
          <div className="text-zinc-600 text-xs uppercase tracking-widest">vs</div>
          <div className="w-16 h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-orange-500 rounded-full transition-all duration-500"
              style={{ width: `${tugPct}%` }}
            />
          </div>
          <div className="text-zinc-500 text-xs">{tugPct}% tension</div>
        </div>

        <div className="flex-1 text-center">
          <div className="text-xs text-zinc-500 mb-1 uppercase tracking-widest">Madman</div>
          <div className={`text-xl font-bold ${directionColor(madmanDir)}`}>
            {madmanDir.toUpperCase()}
          </div>
          <div className="text-xs text-zinc-400 mt-1">{madmanPct}% conf</div>
        </div>
      </div>
    </div>
  );
}
