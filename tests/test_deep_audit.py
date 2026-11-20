"""Regression tests retained from audit; formerly expected failures are now PASS."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pytest
from btsp_lab.models.neurons import RateConfig, CA1Population, CA3Population, OLMPopulation
from btsp_lab.learning.learning_rules import A2BTSP, A2Parameters, STDPParameters, PairSTDP, RateSTDP
from btsp_lab.models.model import Circuit
from btsp_lab.experiments.runner import run_one, load_config, build_rl, evaluate_policy

ROOT=Path(__file__).resolve().parents[1]
PAR=A2Parameters(**load_config(ROOT/'configs/experiments/experiment.yaml')['btsp_parameters'])

@pytest.mark.parametrize('activation', ['relu','sigmoid','softplus'])
@pytest.mark.parametrize('dt', [.001,.03,.2])
def test_rate_ode_closed_form_transient(activation,dt):
    net=CA1Population(RateConfig(size=3,tau_s=.037,activation=activation,threshold=.2,gain=1.3,max_rate=4))
    drive=np.array([-1.,.4,2.])
    target=net.response(drive)
    for n in range(1,8):
        got=net.integrate(drive,dt)
        np.testing.assert_allclose(got,target*(1-np.exp(-n*dt/.037)),atol=1e-13)
        assert (0<=got).all() and (got<=4).all()

@pytest.mark.parametrize('mode', ['none','rate','pair','theta_rate'])
@pytest.mark.parametrize('btsp', [False,True])
@pytest.mark.parametrize('with_motor', [False,True])
def test_circuit_factorial_two_plasticity_classes(mode,btsp,with_motor):
    cfg={'dt_s':.02,'dwell_s':.04,'plateau_s':.30,'n_ca1':3,
         'btsp':btsp,'stdp':mode,'plateau_mode':'every_visit',
         'spike_rate_hz_per_unit':{'ca3':15.,'ca1':20.},
         'stdp_params':{'eta':.05},
         'neurons':{'ca3':{'tau_s':.017},'ca1':{'tau_s':.03}}}
    c=Circuit(3,cfg,PAR,seed=721)
    if with_motor: c.attach_motor(np.ones((4,3))*.03,{'tau_s':.03})
    w_init=c.W.copy()
    for s in (0,1,0,2):
        r=c.present(s)
        assert r.shape==(3,) and np.all(np.isfinite(r))
        assert np.all(c.W>=0.) and np.all(c.W<=PAR.wmax)
        if with_motor:assert c.motor.rate.shape==(4,) and np.isfinite(c.motor.rate).all()
    c.end_episode()
    assert np.isfinite(c.W).all()
    if btsp:
        assert c.btsp.stats['plateau']==4
        assert not np.array_equal(w_init,c.W)
    else:
        assert c.btsp.stats['plateau']==0
        if mode=='none':np.testing.assert_array_equal(w_init,c.W)

@pytest.mark.parametrize('kind', ['sarsa','q_learning','actor_critic','reward_hebb','motor_actor_critic'])
@pytest.mark.parametrize('stdp', ['none','rate','pair','theta_rate'])
def test_end_to_end_factorial_rl_stdp_smoke(kind,stdp,tmp_path):
    cfg=load_config(ROOT/'configs/experiments/smoke_motor.yaml')
    cfg['output']=str(tmp_path)
    cfg['environment']={'kind':'linear','rows':1,'cols':3,'goal':2,'start':0,
                        'blocked_edges':{'A':[]},'max_steps':2}
    cfg['network']={'dt_s':.02,'dwell_s':.04,'plateau_s':.30,'n_ca1':3,'btsp':True,
                    'plateau_mode':'every_visit','stdp':stdp,
                    'spike_rate_hz_per_unit':{'ca3':10.,'ca1':10.},
                    'stdp_params':{'eta':.01},
                    'neurons':{'ca3':{'tau_s':.025},'ca1':{'tau_s':.04},'motor':{'tau_s':.04}}}
    cfg['experiment']={'phases':[{'name':'A','episodes':1}]}
    cfg['rl']={'kind':kind}
    if kind=='q_learning':cfg['rl']['lam']=0.
    cfg['evaluation']={'enabled':True,'every_episodes':1,'n_episodes':1}
    cfg['metrics']={'save_rates':True,'rate_stride':1,'save_events':True}
    ev,net,agent=run_one(cfg,seed=227,write=True)
    assert len(ev)==1 and 1<=ev[0]['steps']<=2
    assert np.isfinite(net.W).all()
    data=np.load(tmp_path/'seed_227'/'occupancy.npz')
    assert data['counts'].sum()==ev[0]['steps']
    assert (tmp_path/'seed_227'/'evaluations.csv').exists()
    rateidx=json.loads((tmp_path/'seed_227'/'rates_index.json').read_text())
    assert rateidx['shards']

@pytest.mark.parametrize('kind', ['sarsa','q_learning','actor_critic','reward_hebb','motor_actor_critic'])
def test_rl_training_deterministic_by_seed(kind,tmp_path):
    cfg=load_config(ROOT/'configs/experiments/smoke_motor.yaml')
    cfg['rl']={'kind':kind,**({'lam':0.} if kind=='q_learning' else {})}
    cfg['environment']['max_steps']=2
    cfg['experiment']['phases']=[{'name':'A','episodes':2}]
    cfg['metrics']={'save_events':False,'save_rates':False}
    a,net_a,rl_a=run_one(deepcopy(cfg),seed=44,write=False)
    b,net_b,rl_b=run_one(deepcopy(cfg),seed=44,write=False)
    assert a==b
    np.testing.assert_array_equal(net_a.W,net_b.W)


def test_no_learning_when_all_rules_disabled():
    cfg={'dt_s':.004,'dwell_s':.04,'btsp':False,'stdp':'none','plateau_mode':'off'}
    c=Circuit(3,cfg,PAR)
    w=c.W.copy()
    for s in range(3):c.present(s)
    np.testing.assert_array_equal(c.W,w)
    assert c.btsp.stats['weight_ticks']==0


def test_olm_rate_cell_obeys_ode_without_circuit_wiring():
    # Tests only basic rate class; NOT a functional OLM->EC3/CA1 feedback loop.
    c=OLMPopulation(RateConfig(size=2,tau_s=.02))
    np.testing.assert_allclose(c.integrate([1.,0.],.02),[1-np.exp(-1),0])
    assert c.type_name()=='OLM_interneuron'

def test_ec3_population_and_instructive_to_ca1_integration():
    from btsp_lab.models.neurons import EC3Population  # noqa: F401

def test_olm_feedback_gate_on_plateaus():
    c=Circuit(2,{'dt_s':.02,'dwell_s':.04,
                 'ec3':{'enabled':True},'olm':{'enabled':True}},PAR)
    assert c.olm is not None and c.W_olm_ca1.shape==(2,2)

def test_ca3_recurrent_synaptic_dynamics():
    c=Circuit(2,{'dt_s':.02,'dwell_s':.04,'ca3_recurrent':{'enabled':True}},PAR)
    assert c.W_ca3_ca3.shape==(2,2) and c.ca3_btsp is not None

def test_shared_weight_ceiling_consistency_validation():
    with pytest.raises(ValueError,match='wmax'):
        Circuit(3,{'stdp':'pair','stdp_params':{'wmax':.1},'spike_rate_hz_per_unit':{'ca3':10.,'ca1':10.}},PAR)

def test_stdp_rejects_nan_presynaptic_input():
    w=np.ones((1,1));r=RateSTDP(w,STDPParameters())
    with pytest.raises(ValueError):r.step(np.array([np.nan]),np.array([.5]))

def test_neuron_config_rejects_nonfinite_tau():
    with pytest.raises(ValueError):RateConfig(size=2,tau_s=float('nan'))

def test_terminal_goal_btsp_encodes_rewarded_location(tmp_path):
    cfg=load_config(ROOT/'configs/experiments/smoke_rate.yaml')
    cfg['environment']={'kind':'linear','rows':1,'cols':2,'goal':1,'start':0,'blocked_edges':{'A':[]},'max_steps':1}
    cfg['experiment']={'phases':[{'name':'A','episodes':1}]}
    cfg['network'].update({'n_ca1':2,'dt_s':.004,'dwell_s':.04,'plateau_mode':'every_visit'})
    cfg['rl']={'kind':'sarsa','epsilon':0.,'lam':0.}
    with patch('btsp_lab.learning.rl.LinearActionValue.action',lambda self,rate,train=True:3):
        ev,c,_=run_one(cfg,seed=202,write=False)
    assert ev[0]['success']==1
    assert c.btsp.stats['plateau']==2  # expected if terminal state processed


@pytest.mark.parametrize('delay_ticks',[-250,-125,0,125,250])
def test_btsp_lagged_eligibility_bidirectional_bounded(delay_ticks):
    # 100ms CA3 burst at +/- seconds relative to 300ms plateau.
    p=PAR
    r=A2BTSP(1,1,p,dt=.004,plateau_s=.3)
    r.W[:]=.01*p.wmax
    total=2000;plateau_tick=750
    for tick in range(total):
        if tick==plateau_tick:r.induce([0])
        r.step([1.] if plateau_tick+delay_ticks<=tick<plateau_tick+delay_ticks+25 else [0.])
    assert 0<=r.W[0,0]<=p.wmax
    assert np.isfinite(r.W).all()

@pytest.mark.parametrize('delta', [1,2,5,10])
def test_pair_stdp_potentiation_exponential_delay(delta):
    p=STDPParameters(tau_pre_s=.02,tau_post_s=.04,eta=.01,wmax=2.)
    w=np.array([[1.]])
    stdp=PairSTDP(w,p,dt=.004)
    stdp.step([1],[0])
    for _ in range(delta-1):stdp.step([0],[0])
    stdp.step([0],[1])
    expected=.01*np.exp(-delta*.004/.02)*.5
    assert w[0,0]-1==pytest.approx(expected,abs=1e-14)

def test_zero_step_goal_episode_log_matches_returned_success(tmp_path):
    cfg=load_config(ROOT/'configs/experiments/smoke_rate.yaml')
    cfg['output']=str(tmp_path)
    cfg['environment']={'kind':'linear','rows':1,'cols':2,'goal':1,'start':1,'blocked_edges':{'A':[]},'max_steps':2}
    cfg['experiment']={'phases':[{'name':'A','episodes':1}]}
    cfg['evaluation']={'enabled':False}
    cfg['network']['n_ca1']=2
    ev,_,_=run_one(cfg,seed=12,write=True)
    import csv
    with (tmp_path/'seed_12'/'episodes.csv').open() as f: logged=list(csv.DictReader(f))
    assert ev[0]['success']==1
    assert int(logged[0]['success'])==ev[0]['success']

def test_btsp_negative_instructive_signal_fails_parameter_validation():
    from dataclasses import replace
    with pytest.raises(ValueError):replace(PAR,is_scale=-1.)

def test_reward_hebb_nonpositive_tau_rejected():
    from btsp_lab.learning.learning_rules import RewardModulatedHebbian
    with pytest.raises(ValueError):RewardModulatedHebbian(np.zeros((2,2)),eta=.1,tau_s=-1.)

@pytest.mark.parametrize('stdp', ['rate','pair'])
def test_stdponly_updates_without_plateaus(stdp):
    cfg={'dt_s':.004,'dwell_s':.04,'plateau_s':.3,'n_ca1':2,'btsp':False,'stdp':stdp,
         'spike_rate_hz_per_unit':{'ca3':150.,'ca1':150.},
         'stdp_params':{'eta':.7}}
    c=Circuit(2,cfg,PAR,seed=3)
    a=c.W.copy()
    for state in [0,1,0,1,0,1]:c.present(state)
    assert c.btsp.stats['plateau']==0
    assert np.any(np.abs(c.W-a)>1e-10)


def test_terminated_and_truncated_rl_are_not_identical():
    from btsp_lab.learning.rl import LinearActionValue
    x=np.array([1.,0.]);y=np.array([0.,1.])
    ag=LinearActionValue('q_learning',2,lam=0.,gamma=.9,alpha=0.,bootstrap_trunc=True)
    ag.theta[1,1]=2
    e_trunc=ag.update(x,0,0.,y,None,False,True)
    e_terminal=ag.update(x,0,0.,y,None,True,False)
    assert e_trunc==pytest.approx(1.8)
    assert e_terminal==pytest.approx(0.)

def test_btsp_rest_uses_actual_presynaptic_activity():
    conf={'dt_s':.004,'dwell_s':.008,'plateau_s':.3,'btsp':True,'inter_episode_rest_s':.004,'flush_btsp_traces':False}
    c=Circuit(2,conf,PAR,seed=321)
    c.present(0)
    captured=[]
    original=c.btsp.step
    def record(x):
        captured.append((np.asarray(x).copy(),c.ca3.rate.copy()))
        original(x)
    c.btsp.step=record
    c.end_episode()
    assert len(captured)==1
    x,actual=captured[0]
    assert actual[0]>0
    np.testing.assert_allclose(x,actual)

def test_stdp_rest_uses_actual_post_and_pre_rates():
    conf={'dt_s':.004,'dwell_s':.008,'plateau_s':.3,'btsp':False,'inter_episode_rest_s':.004,
          'stdp':'rate','stdp_params':{'eta':.01}}
    c=Circuit(2,conf,PAR,seed=321)
    c.present(0)
    captured=[]
    original=c.stdp.step
    def record(x,y):
        captured.append((np.asarray(x).copy(),np.asarray(y).copy(),c.ca3.rate.copy(),c.ca1.rate.copy()))
        original(x,y)
    c.stdp.step=record
    c.end_episode()
    assert len(captured)==1
    x,y,pre,post=captured[0]
    assert pre[0]>0 and post[0]>0
    np.testing.assert_allclose(x,pre)
    np.testing.assert_allclose(y,post)

@pytest.mark.parametrize('btsp_source', ['ca3_rate','sensory_onehot'])
@pytest.mark.parametrize('plateau_mode', ['off','every_visit','bernoulli'])
@pytest.mark.parametrize('rest_s', [0.,.016])
def test_plasticity_eligibility_and_episode_rest_sweep(btsp_source,plateau_mode,rest_s):
    cfg={'dt_s':.004,'dwell_s':.016,'plateau_s':.3,'n_ca1':2,'btsp':True,
         'btsp_input':btsp_source,'plateau_mode':plateau_mode,'plateau_probability':.25,
         'inter_episode_rest_s':rest_s,'flush_btsp_traces':False}
    c=Circuit(2,cfg,PAR,seed=0)
    for _ in range(3):
        for s in [0,1,1,0]:c.present(s)
        c.end_episode()
    assert np.isfinite(c.W).all() and np.max(c.W)<=PAR.wmax
    if plateau_mode=='off':assert c.btsp.stats['plateau']==0
    if plateau_mode=='every_visit':assert c.btsp.stats['plateau']==12
    assert c.ticks==(3*4*4 + 3*int(round(rest_s/.004)))

@pytest.mark.parametrize('mode', ['pair','rate','theta_rate'])
def test_stdp_bounded_under_repeated_inputs(mode):
    cfg={'dt_s':.004,'dwell_s':.024,'plateau_s':.3,'n_ca1':2,
         'btsp':True,'stdp':mode,'stdp_params':{'eta':.1,'wmax':PAR.wmax},
         'spike_rate_hz_per_unit':{'ca3':60.,'ca1':60.}}
    c=Circuit(2,cfg,PAR,seed=2)
    for n in range(15):
        c.present(n%2)
        assert np.isfinite(c.W).all() and (c.W>=0).all() and (c.W<=PAR.wmax).all()
    c.end_episode()


def test_complete_btsp_ec3_olm_microcircuit():
    c=Circuit(3,{'dt_s':.004,'dwell_s':.04,'btsp':True,
                 'ec3':{'enabled':True},'olm':{'enabled':True},
                 'plateau_mode':'dendritic'},PAR)
    assert c.ec3 is not None and c.olm is not None
    c.ec3.rate[:]=1.
    c.olm.rate[:]=0.
    without=c._plateau_hazard().copy()
    c.olm.rate[:]=10.
    with_olm=c._plateau_hazard().copy()
    assert np.all(with_olm < without)
