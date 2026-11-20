"""Mechanistic and integration tests for the optional v0.2 CA3/EC3/OLM system.

The tests deliberately distinguish *functional* offline reactivation from
experimental sharp-wave ripple / spike replay (which is not modeled).
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import csv
import json
import numpy as np
import pytest
from btsp_lab.learning.learning_rules import A2Parameters
from btsp_lab.models.microcircuit import CA3SymmetricConfig, SymmetricCA3BTSP
from btsp_lab.models.model import Circuit
from btsp_lab.experiments.runner import load_config, run_one, evaluate_policy

ROOT = Path(__file__).resolve().parents[1]
PAR = A2Parameters(**load_config(ROOT/'configs/experiments/experiment.yaml')['btsp_parameters'])


def micro_config(*, ec3=True, olm=True, recurrent=True, replay=True,
                 stdp='none', plateau_mode='dendritic'):
    return {
        'dt_s': .004, 'dwell_s': .04, 'n_ca1': 3, 'btsp': True,
        'plateau_mode': plateau_mode,
        'plateau_hazard': {'max_rate_hz': 4., 'gain': 3., 'threshold': .5,
                           'refractory_s': .1,
                           'tonic_drive': .6 if not ec3 else 0.},
        'ec3': {'enabled': ec3, 'gain': 1.8},
        'olm': {'enabled': olm, 'feedback_gain': .2, 'inhibition_gain': 1.5},
        'ca3_recurrent': {'enabled': recurrent, 'eta': .25,
                          'online_recurrent_gain': .05, 'recurrent_gain': 1.2},
        'offline_replay': {'enabled': replay, 'duration_s': .08,
                           'symmetric_stdp': True, 'adaptation_gain': 1.,
                           'noise_sd': 0.},
        'stdp': stdp, 'spike_rate_hz_per_unit': {'ca3': 15, 'ca1': 20},
        'stdp_params': {'eta': .003},
        'inter_episode_rest_s': .008,
        'neurons': {'ca3': {'tau_s': .02, 'max_rate': 1.},
                    'ca1': {'tau_s': .05, 'threshold': .2, 'max_rate': 3.},
                    'ec3': {'tau_s': .04},
                    'olm': {'tau_s': .04}}
    }


def test_symmetric_ca3_btsp_temporal_kernel_both_orders():
    def updated_at(lag):
        weights = np.zeros((2, 2))
        rule = SymmetricCA3BTSP(weights, CA3SymmetricConfig(
            tau_s=.4, eta=.02, wmax=.8), dt=.01)
        for tick in range(150):
            pre = [0., 1.] if tick == 70+lag else [0., 0.]
            rule.step(pre, (0,) if tick == 70 else ())
        return weights[0, 1]
    for lag in [1, 5, 15, 40]:
        assert updated_at(-lag) == pytest.approx(updated_at(lag), rel=1e-9, abs=1e-12)
        assert updated_at(lag) > 0.
    assert updated_at(5) > updated_at(40)


def test_ca3_recurrent_connectivity_learns_and_is_finite():
    c = Circuit(3, micro_config(replay=False), PAR, seed=1)
    assert c.ca3_btsp is not None
    assert np.all(c.W_ca3_ca3 == 0.)
    for s in (0, 1, 2):
        c.present(s)
    assert c.ca3_btsp.stats['plateau'] == 3
    assert np.any(c.W_ca3_ca3 > 0.)
    assert np.all(c.W_ca3_ca3 >= 0.) and np.all(c.W_ca3_ca3 <= c.ca3_recurrent_cfg.wmax)
    np.testing.assert_array_equal(np.diag(c.W_ca3_ca3), [0., 0., 0.])


def test_ec3_excitatory_distal_signal_without_direct_somatic_projection():
    cfg = micro_config(recurrent=False, replay=False, olm=False)
    c = Circuit(3, cfg, PAR, seed=3)
    c.plastic = False
    c.present(1)
    assert c.ec3.rate[1] > 0.
    assert c.W_ec3_ca1.shape == (3, 3)
    with_ec3_hazard = c._plateau_hazard().copy()
    assert with_ec3_hazard[1] > 0.
    # EC3 changes the dendritic instructive drive, but not somatic CA1
    # currents when CA3 activity and recurrent synapses are fixed.
    baseline = Circuit(3, micro_config(ec3=False, olm=False, recurrent=False,
                                        replay=False), PAR, seed=3)
    baseline.plastic = False
    baseline.present(1)
    np.testing.assert_allclose(c.ca1.rate, baseline.ca1.rate, atol=1e-13)
    assert c.last_dendritic_drive[1] > baseline.last_dendritic_drive[1]


def test_olm_feedback_suppresses_plateau_hazard_during_ca1_activation():
    c = Circuit(3, micro_config(recurrent=False, replay=False), PAR, seed=5)
    c.plastic = False
    c.present(1)
    assert c.olm.rate.sum() > 0. and c.ca1.rate.sum() > 0.
    feedback_on = c._plateau_hazard().copy()
    # Acute OLM silencing while holding EC3 and CA1 signals fixed.
    c.olm.rate.fill(0.)
    feedback_off = c._plateau_hazard().copy()
    assert np.all(feedback_on < feedback_off)
    assert np.isfinite(feedback_on).all()


def test_plateau_hazard_is_per_second_not_per_visit():
    c = Circuit(3, micro_config(recurrent=False, replay=False), PAR)
    c.ec3.rate.fill(1.)
    hz = c._plateau_hazard()
    for dt in [.002, .004, .01, .02]:
        n = round(.2/dt)
        per_tick = -np.expm1(-hz*dt)
        cumulative = 1-(1-per_tick)**n
        np.testing.assert_allclose(cumulative, 1-np.exp(-hz*.2), atol=1e-12)


def test_full_microcircuit_integrates_and_replay_offline_is_separate():
    c = Circuit(3, micro_config(), PAR, seed=19)
    for s in [0, 1, 2, 1]:
        c.present(s)
    assert c.ec3 is not None and c.olm is not None
    before_stats = c.btsp.stats.copy()
    w_before = c.W.copy()
    c.end_episode()
    assert len(c.replay_records) == 1
    record = c.replay_records[0]
    assert record['cue_state'] == 1
    assert record['n_ticks'] == 20
    assert len(record['decoded_states']) == 20
    assert record['offline_ca1_updates'] == 20
    assert record['offline_ca3_updates'] == 20
    assert c.btsp.stats['plateau'] == before_stats['plateau']
    assert c.ca3_btsp.stats['plateau'] == 4
    assert np.isfinite(c.W).all() and np.isfinite(c.W_ca3_ca3).all()
    assert np.max(c.W) <= PAR.wmax
    assert np.max(c.W_ca3_ca3) <= c.ca3_recurrent_cfg.wmax
    # Offline rate-STDP changes shared CA1 weights independently of BTSP.
    assert np.linalg.norm(c.W-w_before) > 0.


def test_learned_ca3_local_recurrence_can_generate_sequential_offline_reactivation():
    # Controlled sweep with deterministic path; no environment transition
    # matrix or future trajectory passed to replay.
    cfg = micro_config(ec3=False, olm=False, replay=True, plateau_mode='off')
    cfg['dwell_s'] = .2
    cfg['n_ca1'] = 5
    cfg['inter_episode_rest_s'] = 0.
    cfg['ca3_recurrent'].update({'tau_s': .15, 'eta': .4,
                                  'online_recurrent_gain': .05,
                                  'recurrent_gain': 1.2,
                                  'feedback_inhibition': .05})
    cfg['offline_replay'].update({'duration_s': 1., 'symmetric_stdp': False,
                                  'adaptation_gain': 1.})
    c = Circuit(5, cfg, PAR, seed=8)
    for _ in range(6):
        for state in range(5):
            c.present(state)
    assert np.max(c.W_ca3_ca3) > .1
    c.end_episode()
    decoded = c.replay_records[0]['decoded_states']
    sequence = [v for i, v in enumerate(decoded) if v >= 0 and
                (i == 0 or v != decoded[i-1])]
    assert len(set(sequence)) >= 3, sequence
    assert sequence[0] == 4
    assert c.replay_records[0]['active_ticks'] > 10


def test_eval_does_not_update_ca3_or_ca1_weights_in_replay_network():
    cfg = micro_config()
    c = Circuit(3, cfg, PAR, seed=15)
    for s in [0, 1, 2]:
        c.present(s)
    before = (c.W.copy(), c.W_ca3_ca3.copy())
    c.plastic = False
    c.stdp_kind = 'none'
    for s in [2, 1, 0]:
        c.present(s)
    np.testing.assert_array_equal(c.W, before[0])
    np.testing.assert_array_equal(c.W_ca3_ca3, before[1])


@pytest.mark.parametrize('ec3,olm,recurrent', [(e, o, r) for e in (False, True)
                                      for o in (False, True) for r in (False, True)])
@pytest.mark.parametrize('stdp', ['none', 'rate', 'pair'])
def test_all_2x2x2_component_ablations_with_stdp(ec3, olm, recurrent, stdp):
    cfg = micro_config(ec3=ec3, olm=olm, recurrent=recurrent,
                       replay=recurrent, stdp=stdp)
    c = Circuit(3, cfg, PAR, seed=14)
    for s in [0, 1, 2]:
        c.present(s)
    c.end_episode()
    assert bool(c.ec3 is not None) == ec3
    assert bool(c.olm is not None) == olm
    assert bool(c.ca3_btsp is not None) == recurrent
    assert len(c.replay_records) == int(recurrent)
    assert c.W.shape == (3, 3) and np.isfinite(c.W).all()
    if recurrent:
        assert np.isfinite(c.W_ca3_ca3).all()
    assert (c.btsp.stats['plateau'] >= 0)


@pytest.mark.parametrize('kind', ['sarsa', 'q_learning', 'actor_critic',
                                   'reward_hebb', 'motor_actor_critic'])
def test_full_microcircuit_rl_smoke_writes_replay_and_weights(kind, tmp_path):
    cfg = load_config(ROOT/'configs/experiments/smoke_rate.yaml')
    cfg['output'] = str(tmp_path)
    cfg['environment'] = {'kind': 'linear', 'rows': 1, 'cols': 3,
                          'goal': 2, 'start': 0, 'max_steps': 2}
    cfg['experiment'] = {'phases': [{'name': 'A', 'episodes': 2}]}
    cfg['network'] = micro_config(stdp='rate')
    cfg['network']['neurons']['motor'] = {'tau_s': .03}
    cfg['evaluation'] = {'enabled': True, 'every_episodes': 2, 'n_episodes': 1}
    cfg['rl'] = {'kind': kind, **({'lam': 0.} if kind == 'q_learning' else {})}
    cfg['metrics'] = {'save_events': True, 'save_rates': True, 'rate_stride': 5}
    episodes, c, _ = run_one(cfg, seed=20, write=True)
    assert len(episodes) == 2
    path = tmp_path/'seed_20'
    record = json.loads((path/'replay.json').read_text())
    assert len(record) == 2
    weights = np.load(path/'network.npz')
    assert weights['W_ca3_ca3'].shape == (3, 3)
    assert weights['W_ec3_ca1'].shape == (3, 3)
    assert weights['W_ca1_olm'].shape == (3, 3)
    assert (path/'evaluations.csv').exists()
    assert all(np.isfinite(weights[k]).all() for k in ('W_ca3_ca3','W_ca3_ca1'))


def test_terminal_goal_observation_recorded_without_rl_bootstrap(tmp_path):
    cfg = load_config(ROOT/'configs/experiments/smoke_rate.yaml')
    cfg['output'] = str(tmp_path)
    cfg['environment'] = {'kind':'linear','rows':1,'cols':2,'goal':1,
                          'start':0,'max_steps':1}
    cfg['experiment'] = {'phases':[{'name':'A','episodes':1}]}
    cfg['network'].update({'n_ca1':2,'terminal_observation':True})
    cfg['rl'] = {'kind': 'sarsa', 'epsilon': 0., 'lam': 0.}
    recorded_next = []
    from btsp_lab.learning.rl import LinearActionValue
    original_update = LinearActionValue.update
    def record_update(self, rate, action, reward, next_rate, next_action,
                      terminated, truncated):
        recorded_next.append((next_rate.copy(), terminated, truncated))
        return original_update(self, rate, action, reward, next_rate,
                               next_action, terminated, truncated)
    with patch('btsp_lab.learning.rl.LinearActionValue.action', lambda self, rate, train=True: 3):
        with patch('btsp_lab.learning.rl.LinearActionValue.update', record_update):
            episodes, c, _ = run_one(cfg, seed=8, write=True)
    assert episodes[0]['success'] == 1
    assert c.btsp.stats['plateau'] == 2
    observed = [json.loads(x)['state'] for x in
                (tmp_path/'seed_8'/'observations.jsonl').read_text().splitlines()]
    assert observed == [0, 1]
    assert len(recorded_next) == 1
    next_rate, terminated, truncated = recorded_next[0]
    assert terminated and not truncated
    assert np.array_equal(next_rate, np.zeros_like(next_rate))
    assert np.load(tmp_path/'seed_8'/'occupancy.npz')['counts'].sum() == 1


def test_terminal_observation_can_be_disabled_for_legacy_behavior(tmp_path):
    cfg = load_config(ROOT/'configs/experiments/smoke_rate.yaml')
    cfg['output'] = str(tmp_path)
    cfg['environment'] = {'kind':'linear','rows':1,'cols':2,'goal':1,
                          'start':0,'max_steps':1}
    cfg['experiment'] = {'phases':[{'name':'A','episodes':1}]}
    cfg['network'].update({'n_ca1':2,'terminal_observation':False})
    with patch('btsp_lab.learning.rl.LinearActionValue.action', lambda self, rate, train=True: 3):
        ep, c, _ = run_one(cfg, seed=12, write=True)
    assert ep[0]['success'] == 1
    assert c.btsp.stats['plateau'] == 1
    observed = (tmp_path/'seed_12'/'observations.jsonl').read_text().splitlines()
    assert len(observed) == 1


@pytest.mark.parametrize('cfg,match', [
    ({'offline_replay':{'enabled':True, 'duration_s':.1}}, 'ca3_recurrent'),
    ({'plateau_mode':'dendritic'}, 'require EC3'),
    ({'ca3_recurrent':{'enabled':True, 'recurrent_gain':-1}}, 'CA3 parameter'),
    ({'olm':{'enabled':True, 'feedback_gain':-1}}, 'OLM'),
])
def test_reject_invalid_microcircuit_settings(cfg, match):
    with pytest.raises(ValueError, match=match):
        Circuit(3, cfg, PAR)


def test_symmetric_offline_stdp_respects_both_spike_orders():
    from btsp_lab.models.microcircuit import SymmetricRateSTDP
    def change(reverse):
        w = np.array([[.1]], dtype=float)
        rule = SymmetricRateSTDP(w, dt=.005, tau_s=.025, eta=.4, wmax=1.)
        pre_first = ([1.], [0.]) if not reverse else ([0.], [1.])
        post_second = ([0.], [1.]) if not reverse else ([1.], [0.])
        rule.step(*pre_first)
        for _ in range(3):
            rule.step([0.], [0.])
        rule.step(*post_second)
        return w[0, 0] - .1
    assert change(False) > 0.
    assert change(False) == pytest.approx(change(True), rel=1e-12)


def test_replay_freezes_btsp_weights_and_advances_trace_clocks():
    cfg = micro_config(stdp='none')
    cfg['offline_replay']['symmetric_stdp'] = False
    cfg['inter_episode_rest_s'] = 0.
    c = Circuit(3, cfg, PAR, seed=7)
    c.present(0)
    previous_w = c.W.copy()
    previous_ca3 = c.W_ca3_ca3.copy()
    previous_clock = c.btsp.ticks
    previous_weight_ticks = c.btsp.stats['weight_ticks']
    previous_online_plateaus = c.ca3_btsp.stats['plateau']
    c.end_episode()
    assert c.btsp.ticks == previous_clock + 20
    assert c.btsp.ticks == c.ticks
    assert c.btsp.stats['weight_ticks'] == previous_weight_ticks
    assert c.ca3_btsp.stats['plateau'] == previous_online_plateaus
    np.testing.assert_array_equal(c.W, previous_w)
    np.testing.assert_array_equal(c.W_ca3_ca3, previous_ca3)


def test_offline_replay_stdp_ablation_changes_weight_updates():
    cfg = micro_config()
    cfg['inter_episode_rest_s'] = 0.
    run = []
    for symmetric_stdp in (False, True):
        conf = deepcopy(cfg)
        conf['offline_replay']['symmetric_stdp'] = symmetric_stdp
        c = Circuit(3, conf, PAR, seed=21)
        for state in [0, 1, 2]:
            c.present(state)
        w = c.W.copy()
        ca3 = c.W_ca3_ca3.copy()
        c.end_episode()
        run.append((c, np.linalg.norm(c.W-w), np.linalg.norm(c.W_ca3_ca3-ca3)))
    no_std, enabled = run
    assert no_std[1] == 0. and no_std[2] == 0.
    assert enabled[1] > 0. and enabled[2] > 0.
    assert no_std[0].replay_records[0]['offline_ca1_updates'] == 0
    assert enabled[0].replay_records[0]['offline_ca1_updates'] == 20


def test_replay_without_plasticity_does_not_modify_weights():
    c = Circuit(3, micro_config(), PAR, seed=44)
    c.plastic = False
    c.present(1)
    before_ca3 = c.W_ca3_ca3.copy()
    before_ca1 = c.W.copy()
    c.end_episode()
    np.testing.assert_array_equal(c.W, before_ca1)
    np.testing.assert_array_equal(c.W_ca3_ca3, before_ca3)
    assert c.replay_records[-1]['offline_ca1_updates'] == 0


def test_zero_step_goal_episode_has_consistent_observation_and_logging(tmp_path):
    cfg = load_config(ROOT/'configs/experiments/smoke_rate.yaml')
    cfg['output'] = str(tmp_path)
    cfg['environment'] = {'kind':'linear', 'rows':1, 'cols':2, 'goal':1,
                          'start':1, 'max_steps':2}
    cfg['experiment'] = {'phases':[{'name':'A','episodes':1}]}
    cfg['network'].update({'n_ca1':2,'terminal_observation':True})
    ev, c, _ = run_one(cfg, seed=100, write=True)
    with (tmp_path/'seed_100'/'episodes.csv').open() as f:
        row = list(csv.DictReader(f))[0]
    assert ev[0]['success'] == int(row['success']) == 1
    assert ev[0]['steps'] == int(row['steps']) == 0
    assert c.btsp.stats['plateau'] == 1
    obs = [json.loads(x)['state'] for x in
           (tmp_path/'seed_100'/'observations.jsonl').read_text().splitlines()]
    assert obs == [1]
