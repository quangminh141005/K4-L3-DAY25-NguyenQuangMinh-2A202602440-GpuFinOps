import pytest
from finops import pricing, sustainability
from missions.m1_efficiency_audit import rightsize_memory_bound


def test_interruption_and_commitment_horizon_change_choice():
    options = dict(on_demand_hr=1, spot_hr=.9, reserved_1yr_hr=.8,
                   reserved_3yr_hr=.5, active_days=30, commitment_years=0)
    assert pricing.recommend_tier(10, True, interrupt_rate=0, **options) == 'spot'
    assert pricing.recommend_tier(10, True, interrupt_rate=1, **options) == 'on_demand'
    assert pricing.recommend_tier(24, False, commitment_years=1) == 'reserved_1yr'
    assert pricing.recommend_tier(24, False, commitment_years=3) == 'reserved_3yr'
    assert pricing.recommend_tier(24, False, active_days=1) == 'on_demand'


def test_cache_threshold_storage_and_batch():
    assert not pricing.cache_is_worth_it(1000, 0, 3)
    assert pricing.cache_is_worth_it(1000, 1, 3)
    assert not pricing.cache_is_worth_it(1000, 1, 3, storage_per_m_token_hour=3)
    assert not pricing.cache_is_worth_it(1000, 1, 3, storage_per_m_token_hour=2, batch_multiplier=.5)
    assert not pricing.cache_is_worth_it(1000, 1, 3, cache_discount=.75)


def test_rightsize_rejects_insufficient_memory_and_bandwidth():
    catalog = {g: dict(on_demand_hr=p, hbm_gb=m, peak_bw_tbs=b, peak_tflops_fp16=100)
               for g, p, m, b in [('source', 3, 80, 4), ('too_small', .1, 24, 4),
                                  ('too_slow', .2, 80, .1), ('target', 1, 80, 3)]}
    rows = [dict(gpu_id='gpu', gpu_type='source', workload='infer',
                 achieved_tflops=10, achieved_bw_tbs=2, mem_used_gb=60)]
    assert rightsize_memory_bound(rows, catalog)[0]['target_gpu'] == 'target'


def test_hourly_schedule_uses_contiguous_window_and_fractional_hour():
    profiles = {'us-east-1': [dict(carbon_g_kwh=400, price_kwh=.2)] * 24,
                'clean': [dict(carbon_g_kwh=100, price_kwh=.1)] * 24}
    profiles['clean'] = list(profiles['clean'])
    profiles['clean'][5] = dict(carbon_g_kwh=0, price_kwh=0)
    job = dict(job_id='train', kind='train', interruptible='1', hours_per_day='.5',
               days='2', gpu_type='gpu', num_gpus='1')
    result = sustainability.schedule_training([job], {'gpu': {'watts': '1000'}}, profiles)
    assert result['jobs'][0]['start_hour_utc'] == 5
    assert result['carbon_saved_g'] == 400
    assert result['electricity_saved_usd'] == .2
