# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $18,005  
**Optimized spend:** $10,475  
**Projected savings:** $7,530  (**42%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,176 |
| Purchasing (spot/reserved) | $6,354 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Lowest-carbon region: europe-north1

_Figures are June-2026 as-of snapshots; re-baseline before acting._

## Extended policy analysis

Purchasing improvement over the old rule on identical hours and full reservation billing: $0.00/month. Three-year eligibility assumes a three-year planning horizon; shorter horizons can be supplied to the policy. GPU interruption rates are illustrative assumptions.

Cache: 0 accepted, 2400 rejected. Logs lack cache identity/reuse/TTL/provider tariffs; no cache savings are counted without expected-read metadata. Supply cache_expected_reads, cache_write_multiplier, cache_storage_per_m_token_hour and cache_ttl_hours per row. Write overhead is amortized across the declared reads.

Reasoning premium: $1.0279/day and 29,610.14 Wh/day ($30.84 and 888,304.20 Wh/month). Counterfactual uses the generator's 6x output expansion and the illustrative 80x energy multiplier.

Routing rule: default to small non-reasoning; allow reasoning only on approved tasks while the estimated daily reasoning premium stays below $0.50 and 10,000 Wh, with a 1,200 output-token cap. Reset budgets daily and fall back when either budget is exhausted. Disabling all unapproved reasoning in this sample projects $1.3358/day savings and 29,610.14 Wh/day. Quality needs evaluation before deployment; these additional savings are not in the headline total.

| GPU | Target | Required VRAM GB | Required TB/s | $/GB-VRAM-hour | Target TB/s |
|---|---|---:|---:|---:|---:|
| gpu-a100-0 | A100 | 74.47 | 0.778 | 0.0224 | 2.0 |
| gpu-a100-1 | A100 | 73.7 | 0.745 | 0.0224 | 2.0 |
| gpu-a10g-0 | A10G | 22.33 | 0.2 | 0.0417 | 0.6 |
| gpu-a10g-1 | A10G | 22.44 | 0.233 | 0.0417 | 0.6 |

Standalone fleet estimates (excluded from headline to prevent overlap with purchasing): {'MBU-aware right-sizing (standalone estimate)': 0, 'Kill idle GPUs': 600}. MBU > MFU with MFU < 30% identifies candidates; observed peak memory/bandwidth plus 10% headroom constrains targets. Validate compatibility and latency.

Training scheduling saves 485.23 kgCO2 and $108.76 electricity over the actual job days. Objective: electricity + carbon at $100/tonne. Constant regional snapshots make hours tie; hour 00 UTC is deterministic, not evidence of an hourly advantage. Full GPU nameplate power assumed; GPU rent held constant, electricity excluded from cloud-bill savings.

- job-train-llm: us-east-wa, 00:00 UTC, 20 hours/day; 454.72 kgCO2 saved.
- job-train-embed: us-east-wa, 00:00 UTC, 10 hours/day; 23.20 kgCO2 saved.
- job-finetune: us-east-wa, 00:00 UTC, 6 hours/day; 7.31 kgCO2 saved.
