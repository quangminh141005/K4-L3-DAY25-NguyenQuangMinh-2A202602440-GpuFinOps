"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    legacy_monthly = 0.0
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        days = num(j["days"])
        gpu_hours = hpd * days * ngpu
        on_demand_cost = gpu_hours * od
        options = dict(on_demand_hr=od, spot_hr=num(c["spot_hr"]),
                       reserved_1yr_hr=num(c["reserved_1yr_hr"]),
                       reserved_3yr_hr=num(c["reserved_3yr_hr"]),
                       gpu_type=gtype, active_days=days)
        tier = pricing.recommend_tier(hpd, interruptible, **options)
        opt_cost = pricing.tier_costs(hpd, interruptible, **options)[tier] * ngpu

        # Original policy on the same workload hours, with corrected commitment billing.
        legacy_tier = "spot" if interruptible and hpd < 24 else ("reserved_3yr" if hpd / 24 >= .55 else "on_demand")
        legacy_monthly += pricing.tier_costs(hpd, interruptible, **options)[legacy_tier] * ngpu
        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print("Reservations billed 24h/day; 3-year planning horizon assumed.")
        print(f"Policy improvement vs legacy on identical accounting: ${legacy_monthly - optimized_monthly:,.2f}/month")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

    return {"legacy_policy_monthly": round(legacy_monthly, 2),
            "policy_improvement_monthly": round(legacy_monthly - optimized_monthly, 2),
            "recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1)}


if __name__ == "__main__":
    run()
