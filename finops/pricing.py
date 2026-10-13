"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Illustrative scenario assumptions, not measured provider interruption rates.
GPU_INTERRUPT_RATES = {"H100": .05, "H200": .08, "A100": .10,
                       "A10G": .15, "L4": .12, "B200": .10, "MI300X": .08}


def tier_costs(hours_per_day, interruptible, *, on_demand_hr=1.0,
               spot_hr=.6, reserved_1yr_hr=.8, reserved_3yr_hr=.55,
               gpu_type="H100", interrupt_rate=None, active_days=30,
               period_days=30, commitment_years=3):
    """Period costs per GPU; reservations pay for all capacity, including idle time.

    commitment_years is an explicit planning horizon, not an inferred renewal.
    """
    if not 0 <= hours_per_day <= 24 or not 0 <= active_days <= period_days:
        raise ValueError("Invalid duty cycle or active days")
    hours = hours_per_day * active_days
    costs = {"on_demand": hours * on_demand_hr}
    if interruptible:
        rate = GPU_INTERRUPT_RATES.get(gpu_type, .10) if interrupt_rate is None else interrupt_rate
        costs["spot"] = hours * (1.03 + rate * .5) * spot_hr
    if commitment_years >= 1:
        costs["reserved_1yr"] = period_days * 24 * reserved_1yr_hr
    if commitment_years >= 3:
        costs["reserved_3yr"] = period_days * 24 * reserved_3yr_hr
    return costs


def recommend_tier(hours_per_day, interruptible, reserved_discount=.45, **kwargs):
    """Choose the cheapest eligible tier including checkpoint/rework and commitments.

    Legacy calls return 'reserved'; catalog-aware calls identify the term.
    """
    legacy = not kwargs
    kwargs.setdefault("reserved_3yr_hr", 1 - reserved_discount)
    tier = min((costs := tier_costs(hours_per_day, interruptible, **kwargs)), key=costs.get)
    return "reserved" if legacy and tier.startswith("reserved_") else tier


def cache_is_worth_it(input_tokens, expected_reads, price_in_per_m,
                     cache_discount=.10, write_multiplier=1.25,
                     storage_per_m_token_hour=0.0, ttl_hours=1.0, batch_multiplier=1.0):
    """Cache only when read savings exceed write premium plus storage.

    Reads exclude the initial cache creation. Storage is not batch-discounted.
    Gemini-style storage and provider write tariffs must be supplied explicitly.
    """
    if min(input_tokens, expected_reads, price_in_per_m, ttl_hours) < 0:
        raise ValueError("Cache inputs must be nonnegative")
    saved = expected_reads * price_in_per_m * (1 - cache_discount) * batch_multiplier
    overhead = max(0, write_multiplier - 1) * price_in_per_m * batch_multiplier
    overhead += storage_per_m_token_hour * ttl_hours
    return input_tokens > 0 and saved > overhead


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
