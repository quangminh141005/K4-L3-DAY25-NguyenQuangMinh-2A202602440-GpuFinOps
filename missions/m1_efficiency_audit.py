"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def rightsize_memory_bound(rows, catalog):
    """Fit peak observed VRAM/bandwidth with 10% headroom; rank balanced unit costs.

    Low MFU alone does not establish a memory bottleneck: require MBU > MFU.
    GPU compatibility and latency still need deployment validation.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["workload"] in ("infer", "embed"):
            groups[row["gpu_id"]].append(row)
    recs = []
    for gid, samples in groups.items():
        cur = samples[0]["gpu_type"]
        c = catalog[cur]
        mfu = sum(num(r["achieved_tflops"]) for r in samples) / len(samples) / num(c["peak_tflops_fp16"])
        mbu = sum(num(r["achieved_bw_tbs"]) for r in samples) / len(samples) / num(c["peak_bw_tbs"])
        if not (mfu < .3 and mbu > mfu):
            continue
        mem = max(num(r["mem_used_gb"]) for r in samples) * 1.1
        bw = max(num(r["achieved_bw_tbs"]) for r in samples) * 1.1
        candidates = []
        for gpu, spec in catalog.items():
            price, vram, bandwidth = (num(spec[k]) for k in ("on_demand_hr", "hbm_gb", "peak_bw_tbs"))
            if vram >= mem and bandwidth >= bw and price <= num(c["on_demand_hr"]):
                # Equal relative weight for VRAM and bandwidth unit economics.
                score = .5 * ((price / vram) / (num(c["on_demand_hr"]) / num(c["hbm_gb"])) + (price / bandwidth) / (num(c["on_demand_hr"]) / num(c["peak_bw_tbs"])))
                candidates.append((score, price, gpu))
        target = min(candidates)[2] if candidates else cur
        recs.append(dict(gpu_id=gid, gpu_type=cur, target_gpu=target,
                         required_vram_gb=round(mem, 2), required_bw_tbs=round(bw, 3),
                         dollars_per_gb=num(catalog[target]["on_demand_hr"]) / num(catalog[target]["hbm_gb"]),
                         peak_bw_tbs=num(catalog[target]["peak_bw_tbs"]),
                         daily_savings=24 * (num(c["on_demand_hr"]) - num(catalog[target]["on_demand_hr"]))))
    return recs


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print("MBU-aware targets:", rightsize_memory_bound(tel, cat))
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

    return {"rightsize_recommendations": rightsize_memory_bound(tel, cat), "summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2)}


if __name__ == "__main__":
    run()
