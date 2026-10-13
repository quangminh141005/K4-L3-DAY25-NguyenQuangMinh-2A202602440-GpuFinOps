"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    reasoning_extra_usd = reasoning_extra_wh = capped_savings = capped_wh = 0.0
    cache_accepted = cache_rejected = 0
    cache_overhead = 0.0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        # Missing reuse evidence cannot justify a write/storage investment.
        reads = num(r.get("cache_expected_reads"))
        write = num(r.get("cache_write_multiplier"), 1.25)
        storage = num(r.get("cache_storage_per_m_token_hour"))
        ttl = num(r.get("cache_ttl_hours"), 1)
        mult = .5 if is_batch else 1.0
        worth = pricing.cache_is_worth_it(cached, reads, pin, write_multiplier=write,
                    storage_per_m_token_hour=storage, ttl_hours=ttl, batch_multiplier=mult)
        cache_accepted += int(worth)
        cache_rejected += int(cached > 0 and not worth)
        overhead = (cached / 1e6) * (max(0, write - 1) * pin * mult + storage * ttl) / reads if worth else 0
        cache_overhead += overhead
        actual = pricing.request_cost(inp, out, pin, pout, cached_in=cached if worth else 0, batch=is_batch) + overhead
        opt_cost += actual
        if bool(int(num(r["is_reasoning"]))):
            # Generator expands output by 6x. Counterfactual keeps the same route.
            ordinary_out = out // 6
            extra = pricing.request_cost(inp, out, pin, pout, batch=is_batch) - pricing.request_cost(inp, ordinary_out, pin, pout, batch=is_batch)
            energy = sustainability.wh_per_query(inp + out, is_reasoning=True)
            ordinary_energy = sustainability.wh_per_query(inp + ordinary_out)
            reasoning_extra_usd += extra
            reasoning_extra_wh += energy - ordinary_energy
            # Proposed cap: only explicit approvals keep reasoning; unapproved
            # traffic routes to small non-reasoning with the ordinary output cap.
            capped_savings += actual - pricing.request_cost(inp, ordinary_out, *MODEL_PRICES["small"], batch=is_batch)
            capped_wh += energy - ordinary_energy

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"cache accepted={cache_accepted}, rejected={cache_rejected}")
        print(f"reasoning premium: ${reasoning_extra_usd:.4f}/day, {reasoning_extra_wh:,.2f} Wh/day")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

    return {
        "cache_accepted": cache_accepted, "cache_rejected": cache_rejected,
        "cache_overhead_daily": round(cache_overhead, 6),
        "reasoning_extra_daily_usd": round(reasoning_extra_usd, 4),
        "reasoning_extra_daily_wh": round(reasoning_extra_wh, 2),
        "reasoning_cap_savings_daily": round(capped_savings, 4),
        "reasoning_cap_saved_daily_wh": round(capped_wh, 2),
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    run()
