import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export type Signal = {
  id: string;
  created_at: string;
  symbol: string;
  bot: "sovereign" | "madman";
  direction: "buy" | "sell" | "neutral";
  confidence: number;
  signal_type: string;
  raw_data: Record<string, unknown>;
  source_url: string | null;
};

export type TugResult = {
  id: string;
  created_at: string;
  symbol: string;
  sovereign_direction: "buy" | "sell" | "neutral";
  madman_direction: "buy" | "sell" | "neutral";
  conflict: boolean;
  verdict: "execute" | "crowded_skip" | "no_signal";
  sovereign_confidence: number;
  madman_confidence: number;
  tug_score: number;
};

export type Trade = {
  id: string;
  created_at: string;
  tug_result_id: string | null;
  symbol: string;
  side: "buy" | "sell";
  qty: number;
  order_type: "market" | "limit";
  order_type_detail: string | null;
  limit_price: number | null;
  alpaca_order_id: string | null;
  status: "pending" | "filled" | "cancelled" | "rejected";
  fill_price: number | null;
  fill_qty: number | null;
  pnl: number | null;
  implementation_shortfall_bps: number | null;
};

export type RegimeState = {
  id: string;
  created_at: string;
  state: "chop" | "trend" | "crisis";
  confidence: number;
  spy_volatility: number;
  spy_momentum: number;
};

export type OfiSnapshot = {
  id: string;
  created_at: string;
  symbol: string;
  ofi_z_score: number;
  iceberg_detected: boolean;
  mid_price: number;
};
