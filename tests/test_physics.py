from btsp_lab.experiments.runner import load_config
from pathlib import Path
import sys
import numpy as np
import pytest
from btsp_lab.learning.learning_rules import A2Parameters,A2BTSP,RateSTDP,PairSTDP,STDPParameters,ThetaCompressedSTDP,TDLambda,LinearSuccessorFeatures
from btsp_lab.models.neurons import RateConfig,CA1Population

ROOT=Path(__file__).resolve().parents[1]
PARAM=A2Parameters(**load_config(ROOT/'configs/experiments/experiment.yaml')['btsp_parameters'])

def test_a2_source_parity_3_visits(legacy_reference):
    legacy=legacy_reference
    n=8;rule=A2BTSP(n,n,PARAM)
    W=np.full((n,n),legacy.PAR[10]);et=np.zeros(n);ins=np.zeros(n);ends=np.zeros(n,dtype=np.int64)
    stats=np.zeros(4);p,d=legacy.make_luts(legacy.PAR);clock=0
    for s in [0,2,1,2,4,6,0]:
        rule.advance(np.eye(n)[s],100,[s]);clock=legacy.btsp_dwell(W,et,ins,ends,clock,s,True,legacy.PAR,p,d,stats)
        np.testing.assert_allclose(rule.W,W,rtol=0,atol=5e-12)
        np.testing.assert_allclose(rule.et,et,rtol=0,atol=5e-15)
        np.testing.assert_allclose(rule.ins,ins,rtol=0,atol=5e-15)
    rule.flush(5.0);legacy.btsp_flush(W,et,ins,ends,clock,legacy.PAR,p,d,stats)
    np.testing.assert_allclose(rule.W,W,rtol=0,atol=5e-11)
    assert rule.stats['plateau']==7 and rule.stats['ltp']>0 and rule.stats['ltd']>0


def test_a2_persistent_and_bidirectional_bounds():
    r=A2BTSP(2,2,PARAM)
    r.W[0,:]=[0.,PARAM.wmax]
    r.induce([0]);r.advance(np.array([1.,1.]),100)
    assert r.W[0,0]>0 and r.W[0,1]<PARAM.wmax
    assert (r.W>=0).all() and (r.W<=PARAM.wmax).all()


def test_static_neuron_limit_and_transient():
    cfg=RateConfig(size=2,tau_s=.1,activation='relu',threshold=.5)
    n=CA1Population(cfg)
    W=np.eye(2)
    s=n.integrate_synapses(W,np.array([2.,0.]),.1)
    np.testing.assert_allclose(s,np.array([1.5*(1-np.exp(-1)),0]),atol=1e-14)
    for _ in range(50):n.integrate_synapses(W,np.array([2.,0.]),.1)
    np.testing.assert_allclose(n.rate,n.response(np.array([2.,0.])),atol=1e-12)
    with pytest.raises(ValueError):n.integrate_synapses(np.ones((2,3)),np.ones(2),.1)


def test_rate_stdp_pre_then_post_changes_sign():
    p=STDPParameters(eta=1.0,ltd_ltp_ratio=1.,wmax=2.)
    W=np.ones((1,1));r=RateSTDP(W,p,dt=.004)
    for _ in range(10):r.step([1.],[0.])
    r.step([0.],[1.]);assert W[0,0]>1
    W2=np.ones((1,1));r2=RateSTDP(W2,p,dt=.004)
    for _ in range(10):r2.step([0.],[1.])
    r2.step([1.],[0.]);assert W2[0,0]<1


def test_pair_stdp_pre_before_post_positive_and_reverse():
    p=STDPParameters(eta=.1,wmax=2.)
    W=np.ones((1,1));r=PairSTDP(W,p)
    r.step([1],[0]);r.step([0],[1]);assert W[0,0]>1
    W2=np.ones((1,1));r2=PairSTDP(W2,p)
    r2.step([0],[1]);r2.step([1],[0]);assert W2[0,0]<1


def test_theta_causal_buffer_and_no_prelearned_future():
    W=np.ones((2,2));r=RateSTDP(W,STDPParameters(wmax=2.))
    th=ThetaCompressedSTDP(r,history=2,replay_ticks=4,silent_ticks=2)
    ca1=CA1Population(RateConfig(size=2))
    th.observe(np.array([1,0.]));th.observe(np.array([0,1.]));th.observe(np.array([1,0.]))
    assert len(th.events)==2
    assert np.array_equal(th.events[0],[0.,1.])
    assert np.array_equal(th.events[1],[1.,0.])
    for _ in range(6):th.step(ca1,W)
    assert np.isfinite(W).all()
    assert th.tick==6


def test_td_trace_and_sf_shape():
    rule=TDLambda((2,),.1,.9,.8);v=np.zeros(2)
    rule.update(v,np.array([1.,0.]),1.)
    assert v[0]==pytest.approx(.1)
    rule.update(v,np.array([0.,1.]),2.)
    assert rule.elig[0]==pytest.approx(.9*.8)
    sf=LinearSuccessorFeatures(2,3,.1,.9,.8)
    e=sf.update(np.array([1.,0]),np.array([1.,2.,3.]),np.array([0.,1.]),False)
    assert e.shape==(3,) and sf.W.shape==(2,3)
