from pathlib import Path
import json
import numpy as np
import pytest
from btsp_lab.experiments.runner import load_config,run_one,optimize
from btsp_lab.models.model import Circuit
from btsp_lab.learning.learning_rules import A2Parameters
from btsp_lab.analysis.metrics import MetricsLogger

ROOT=Path(__file__).resolve().parents[1]
PARAM=A2Parameters(**load_config(ROOT/'configs/experiments/experiment.yaml')['btsp_parameters'])

def test_circuit_only_neural_rate_is_read_out_without_normalizing():
    # Force a known, non-unit, nonuniform W. Evaluate raw network response.
    c=Circuit(3,{'dt_s':.004,'dwell_s':.004,'n_ca1':2,'btsp':False,'plateau_mode':'off',
                  'neurons':{'ca3':{'tau_s':.001},'ca1':{'tau_s':.01}}},PARAM)
    c.W[:]=np.array([[2.,0.,0.],[0.,1.,0.]])
    for i in range(50):output=c.present(0)
    assert output[0]>1.9 and output[1]<1e-10
    assert np.linalg.norm(output)>1.9 # not L2-normalized
    assert c.W.shape==(2,3)

def test_theta_stdp_off_equivalent_to_empty_rule():
    base={'dt_s':.004,'dwell_s':.04,'n_ca1':3,'btsp':True,'btsp_input':'sensory_onehot',
          'plateau_mode':'every_visit','neurons':{'ca3':{'tau_s':.01},'ca1':{'tau_s':.03}}}
    a=Circuit(3,base,PARAM)
    b=Circuit(3,{**base,'stdp':'theta_rate','stdp_params':{'eta':0.},'theta_history':4},PARAM)
    for state in [0,1,2,1]:
        a.present(state);b.present(state)
    np.testing.assert_allclose(a.W,b.W,atol=0,rtol=0)
    np.testing.assert_allclose(a.btsp.et,b.btsp.et,atol=0,rtol=0)
    np.testing.assert_allclose(a.btsp.ins,b.btsp.ins,atol=0,rtol=0)

def test_pair_rate_units_must_be_declared():
    with pytest.raises(ValueError,match='spike_rate_hz_per_unit'):
        Circuit(3,{'stdp':'pair'},PARAM)
    c=Circuit(3,{'stdp':'pair','spike_rate_hz_per_unit':{'ca3':15.,'ca1':20.},
                 'dt_s':.004,'dwell_s':.012},PARAM)
    assert np.isfinite(c.present(0)).all()

def test_logger_phase_and_rate_shards(tmp_path):
    m=MetricsLogger(tmp_path,3,2,save_rates=True,rate_stride=1,max_rate_buffer=3)
    m.begin_phase('A_0');m.begin_episode(0)
    for i in range(8):m.tick(np.array([i,2*i]),i)
    m.presentation(0,0,8);m.transition(0,'A',0,3,1,-.01,False,False,False,False)
    m.finish_episode(0,'A',0,0);m.end_phase();m.finish()
    idx=json.loads((tmp_path/'rates_index.json').read_text())['shards']
    assert len(idx)==3
    assert sum(len(np.load(tmp_path/p)['rates']) for p in idx)==8
    assert json.loads((tmp_path/'activity_by_phase.json').read_text())['A_0']['n_samples']==8
    assert json.loads((tmp_path/'observations.jsonl').read_text().splitlines()[0])['state']==0

def test_all_rl_drivers_without_plasticity(tmp_path):
    spec=load_config(ROOT/'configs/experiments/motor_actor.yaml')
    spec['output']=str(tmp_path);spec['network']['btsp']=False
    spec['environment']['max_steps']=3
    for kind,params in [('sarsa',{'alpha':.01,'gamma':.9,'lam':.5}),
                        ('q_learning',{'alpha':.01,'gamma':.9,'lam':0.}),
                        ('actor_critic',{'alpha_actor':.005,'alpha_critic':.005}),
                        ('reward_hebb',{'alpha':.005}),
                        ('motor_actor_critic',{'alpha_actor':.005,'alpha_critic':.005})]:
        cfg={**spec,'rl':{'kind':kind,**params}}
        episodes,net,agent=run_one(cfg,seed=2,write=False)
        assert len(episodes)==3 and np.isfinite(episodes[-1]['reward'])

def test_optuna_dev_seeds_are_disjoint_and_tuning_runs(tmp_path):
    cfg=load_config(ROOT/'configs/experiments/motor_actor.yaml')
    cfg['output']=str(tmp_path)
    cfg['network']['btsp']=False
    cfg['environment']['max_steps']=2
    cfg['experiment']['phases']=[{'name':'A','episodes':1}]
    cfg['optuna']={'enabled':True,'trials':1,'jobs':1,'dev_seeds':[11],
                   'test_seeds':[99],'seed':1,'search_space':{
                       'rl.alpha_actor':{'kind':'float','low':.001,'high':.005}}}
    out=optimize(cfg)
    assert out['experiment']['seeds']==[99]
    assert 'rl.alpha_actor' in json.loads((tmp_path/'optuna_best.json').read_text())['best_params']
    assert 0.001<=out['rl']['alpha_actor']<=0.005

def test_network_stdp_modes_and_common_synapses_are_finite():
    for mode,extra in [('rate',{}),('theta_rate',{'theta_history':3,'theta_replay_ticks':4,'theta_silent_ticks':2}),
                       ('pair',{'spike_rate_hz_per_unit':{'ca3':15.,'ca1':20.}})]:
        net=Circuit(3,{'dt_s':.004,'dwell_s':.04,'stdp':mode,'n_ca1':3,'btsp':True,
                     'neurons':{'ca3':{'tau_s':.005},'ca1':{'tau_s':.01}},
                     'stdp_params':{'eta':.001},**extra},PARAM,seed=123)
        for state in [0,1,2,1,0]:net.present(state)
        assert net.W.shape==(3,3)
        assert np.all(np.isfinite(net.W))
        assert np.all(net.W>=0) and np.all(net.W<=PARAM.wmax)
        assert net.btsp.stats['plateau']==5


def test_rate_stdp_agrees_with_direct_local_update_formula():
    from btsp_lab.learning.learning_rules import RateSTDP,STDPParameters
    w=np.array([[1.,1.2],[.4,.8]])
    p=STDPParameters(eta=.7,ltd_ltp_ratio=.8,wmax=2.9)
    rule=RateSTDP(w,p,dt=.004)
    rule.pre[:]=[.1,.2];rule.post[:]=[.3,.4]
    x=np.array([.5,.6]);y=np.array([.7,.8]);w0=w.copy()
    expected=.004*.7*(np.outer(y,[.1,.2])-.8*np.outer([.3,.4],x))
    update=rule.step(x,y)
    np.testing.assert_allclose(w,w0+expected,atol=1e-14)
    np.testing.assert_allclose(update,expected,atol=1e-14)
