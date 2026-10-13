"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT, load_csv
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for rec in r1["rightsize_recommendations"]:
        rightsize_savings += rec["daily_savings"] * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "MBU-aware right-sizing (standalone estimate)": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    scheduling = sustainability.schedule_training(load_csv("workloads.csv"), cat)
    # Fleet telemetry and purchased workloads have no mapping: avoid adding
    # standalone fleet estimates to savings already claimed on purchased capacity.
    standalone = {k: levers.pop(k) for k in list(levers) if k.startswith("MBU") or k == "Kill idle GPUs"}
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0
    md = report.build_report(baseline, optimized, levers, sustainability=sust)
    md += "\n\n## Extended policy analysis\n\n"
    md += f"Purchasing improvement over the old rule on identical hours and full reservation billing: ${r3['policy_improvement_monthly']:,.2f}/month. Three-year eligibility assumes a three-year planning horizon; shorter horizons can be supplied to the policy. GPU interruption rates are illustrative assumptions.\n\n"
    md += f"Cache: {r2['cache_accepted']} accepted, {r2['cache_rejected']} rejected. Logs lack cache identity/reuse/TTL/provider tariffs; no cache savings are counted without expected-read metadata. Supply cache_expected_reads, cache_write_multiplier, cache_storage_per_m_token_hour and cache_ttl_hours per row. Write overhead is amortized across the declared reads.\n\n"
    md += f"Reasoning premium: ${r2['reasoning_extra_daily_usd']:.4f}/day and {r2['reasoning_extra_daily_wh']:,.2f} Wh/day (${r2['reasoning_extra_daily_usd']*DAYS:.2f} and {r2['reasoning_extra_daily_wh']*DAYS:,.2f} Wh/month). Counterfactual uses the generator's 6x output expansion and the illustrative 80x energy multiplier.\n\n"
    md += f"Routing rule: default to small non-reasoning; allow reasoning only on approved tasks while the estimated daily reasoning premium stays below $0.50 and 10,000 Wh, with a 1,200 output-token cap. Reset budgets daily and fall back when either budget is exhausted. Disabling all unapproved reasoning in this sample projects ${r2['reasoning_cap_savings_daily']:.4f}/day savings and {r2['reasoning_cap_saved_daily_wh']:,.2f} Wh/day. Quality needs evaluation before deployment; these additional savings are not in the headline total.\n\n"
    md += "| GPU | Target | Required VRAM GB | Required TB/s | $/GB-VRAM-hour | Target TB/s |\n|---|---|---:|---:|---:|---:|\n"
    for rec in r1['rightsize_recommendations']:
        md += f"| {rec['gpu_id']} | {rec['target_gpu']} | {rec['required_vram_gb']} | {rec['required_bw_tbs']} | {rec['dollars_per_gb']:.4f} | {rec['peak_bw_tbs']} |\n"
    md += f"\nStandalone fleet estimates (excluded from headline to prevent overlap with purchasing): {standalone}. MBU > MFU with MFU < 30% identifies candidates; observed peak memory/bandwidth plus 10% headroom constrains targets. Validate compatibility and latency.\n\n"
    md += f"Training scheduling saves {scheduling['carbon_saved_g']/1000:,.2f} kgCO2 and ${scheduling['electricity_saved_usd']:.2f} electricity over the actual job days. Objective: electricity + carbon at $100/tonne. Constant regional snapshots make hours tie; hour 00 UTC is deterministic, not evidence of an hourly advantage. Full GPU nameplate power assumed; GPU rent held constant, electricity excluded from cloud-bill savings.\n\n"
    for job in scheduling['jobs']:
        md += f"- {job['job_id']}: {job['region']}, {job['start_hour_utc']:02d}:00 UTC, {job['hours_per_day']:g} hours/day; {job['carbon_saved_g']/1000:.2f} kgCO2 saved.\n"
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"scheduling": scheduling, "standalone_fleet_savings": standalone, "reasoning": r2, "baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
