"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


def schedule_training(jobs, catalog, hourly_profiles=None, carbon_price_per_tonne=100,
                      baseline_region="us-east-1"):
    """Select a contiguous daily window minimizing electricity + priced carbon.

    Profiles: {region: [{carbon_g_kwh, price_kwh}, ...24 hours...]}. Defaults
    use the supplied constant regional snapshots, so hours then tie. GPU rental
    is held constant; electricity savings are diagnostic, not added to cloud bills.
    """
    profiles = hourly_profiles if hourly_profiles is not None else {
        region: [dict(carbon_g_kwh=carbon, price_kwh=REGION_PRICE_KWH[region])] * 24
        for region, carbon in REGION_CARBON.items()}
    if baseline_region not in profiles or any(len(v) != 24 for v in profiles.values()):
        raise ValueError("Require 24 hourly values and a baseline region")
    results = []
    for job in jobs:
        if job["kind"] != "train" or not int(job["interruptible"]):
            continue
        hours = float(job["hours_per_day"])
        if not 0 < hours <= 24:
            raise ValueError("Training hours must be in (0, 24]")
        kw = float(catalog[job["gpu_type"]]["watts"]) / 1000 * int(job["num_gpus"])
        days = float(job["days"])
        def evaluate(region, start):
            cost = carbon = 0.0
            remaining = hours
            offset = 0
            while remaining > 0:
                slot = profiles[region][(start + offset) % 24]
                kwh = kw * min(1, remaining) * days
                cost += kwh * slot["price_kwh"]
                carbon += kwh * slot["carbon_g_kwh"]
                remaining -= min(1, remaining)
                offset += 1
            return cost, carbon
        region, hour = min(((r, h) for r in profiles for h in range(24)),
            key=lambda x: (lambda cost, carbon: cost + carbon / 1e6 * carbon_price_per_tonne)(*evaluate(*x)))
        cost, carbon = evaluate(region, hour)
        base_cost, base_carbon = evaluate(baseline_region, 0)
        results.append(dict(job_id=job["job_id"], region=region, start_hour_utc=hour,
                            hours_per_day=hours, electricity_saved_usd=round(base_cost-cost, 2),
                            carbon_saved_g=round(base_carbon-carbon, 2),
                            baseline_carbon_g=round(base_carbon, 2), optimized_carbon_g=round(carbon, 2)))
    return {"jobs": results, "carbon_saved_g": sum(r["carbon_saved_g"] for r in results),
            "electricity_saved_usd": sum(r["electricity_saved_usd"] for r in results),
            "hourly_data_available": hourly_profiles is not None,
            "carbon_price_per_tonne": carbon_price_per_tonne}
