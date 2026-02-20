"use client";

import { Trade } from "@/lib/supabase";

type Props = { trade: Trade };

const statusColor = (status: string) => {
  if (status === "filled") return "text-emerald-400";
  if (status === "pending") return "text-yellow-400";
  if (status === "rejected" || status === "cancelled") return "text-red-400";
  return "text-zinc-400";
};

const sideColor = (side: string) =>
  side === "buy" ? "text-emerald-400" : "text-red-400";

export default function TradeRow({ trade }: Props) {
  const date = new Date(trade.created_at).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <tr className="border-b border-zinc-800 hover:bg-zinc-800/40 transition-colors">
      <td className="py-3 px-4 text-zinc-300 font-medium">{trade.symbol}</td>
      <td className={`py-3 px-4 font-bold ${sideColor(trade.side)}`}>
        {trade.side.toUpperCase()}
      </td>
      <td className="py-3 px-4 text-zinc-300">{trade.qty}</td>
      <td className="py-3 px-4 text-zinc-400">
        {trade.order_type_detail ?? trade.order_type}
      </td>
      <td className={`py-3 px-4 font-medium ${statusColor(trade.status)}`}>
        {trade.status.toUpperCase()}
      </td>
      <td className="py-3 px-4 text-zinc-400">
        {trade.fill_price ? `$${Number(trade.fill_price).toFixed(2)}` : "—"}
      </td>
      <td className="py-3 px-4 text-xs">
        {trade.implementation_shortfall_bps !== null && trade.implementation_shortfall_bps !== undefined ? (
          <span className="text-blue-400">{Number(trade.implementation_shortfall_bps).toFixed(1)} bps</span>
        ) : (
          <span className="text-zinc-600">—</span>
        )}
      </td>
      <td className="py-3 px-4">
        {trade.pnl !== null ? (
          <span className={trade.pnl >= 0 ? "text-emerald-400" : "text-red-400"}>
            {trade.pnl >= 0 ? "+" : ""}${Number(trade.pnl).toFixed(2)}
          </span>
        ) : (
          <span className="text-zinc-600">—</span>
        )}
      </td>
      <td className="py-3 px-4 text-zinc-500 text-sm">{date}</td>
    </tr>
  );
}
