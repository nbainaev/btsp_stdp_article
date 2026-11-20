from pathlib import Path
import json
import numpy as np
import pytest
from btsp_lab.environments.grid import EnvConfig,GridWorld,legacy_5x6_config,make_linear
from btsp_lab.analysis.metrics import OnlineNeuronActivity,OccupancyMatrix
from btsp_lab.learning.rl import LinearActionValue,LinearCritic,ActorCritic,DopamineMotorAgent
from btsp_lab.experiments.runner import load_config,run_one

ROOT=Path(__file__).resolve().parents[1]

def test_grid_exact_topology_and_collisions():
    env=GridWorld(legacy_5x6_config())
    assert env.n==30 and set(env.walls)=={8,14,20}
    assert len(env.reachable('A',env.goal))==27
    assert len(env.reachable('B',env.goal))==27
    assert env.nexts['A'][4,3]==-1 and env.nexts['B'][4,3]==5
    assert env.nexts['A'][10,3]==-1 and env.nexts['B'][10,3]==11
    assert env.distances('A')[0]==8
    assert env.nexts['A'][22,3]==23 and env.nexts['B'][22,3]==-1
    env.reset(phase='A',start=4)
    ns,r,done,trunc,meta=env.step(3)
    assert ns==4 and meta['collision'] and r==pytest.approx(-.11)
    assert env.distances('A')[env.goal]==0
    assert env.spl(env.goal,0,True,'A')==1
    assert env.spl(0,3,False,'A')==0

def test_invalid_edges_fail_fast():
    with pytest.raises(ValueError,match='touches wall'):
        GridWorld(EnvConfig(rows=2,cols=3,walls=[1],blocked_edges={'A':[(0,1)]},require_connected=False))
    with pytest.raises(ValueError,match='nonadjacent'):
        GridWorld(EnvConfig(rows=2,cols=3,blocked_edges={'A':[(0,5)]}))

def test_custom_start_goal_rewards_truncation_and_spl():
    env=GridWorld(EnvConfig(kind='linear',rows=1,cols=4,start=0,goal=3,
        blocked_edges={'A':[]},max_steps=1,default_reward=-.2,collision_reward=-.5,
        goal_reward=2.,state_rewards={1:.1}))
    assert env.reset()==0
    ns,r,terminal,truncated,info=env.step(3)
    assert (ns,r,terminal,truncated)==(1,.1,False,True)
    assert env.distances()[0]==3
    assert env.spl(0,3,True)==1.

def test_sarsa_and_q_learning_correct_bootstrap():
    for kind in ('sarsa','q_learning'):
        ag=LinearActionValue(kind,3,alpha=.5,gamma=.9,lam=(.8 if kind=='sarsa' else 0.),epsilon=0,seed=1)
        x=np.array([1.,0.,0.]);xp=np.array([0.,1.,0.]);ag.theta[1,1]=2.
        err=ag.update(x,0,1.,xp,1,False,False)
        assert err==pytest.approx(2.8)
        assert ag.theta[0,0]==pytest.approx(1.4)
        ag.update(x,0,1.,xp,1,False,True)
        assert np.sum(np.abs(ag.rule.elig))==0
    a=LinearActionValue('sarsa',3,bootstrap_trunc=False,alpha=.1)
    assert a.update(np.array([1.,0,0]),0,1.,np.ones(3),None,False,True)==pytest.approx(1.)

def test_q_learning_lambda_requires_explicit_variant():
    with pytest.raises(ValueError,match='Q-learning'):
        LinearActionValue('q_learning',4,lam=.7)

def test_metric_online_mean_and_occupancy():
    m=OnlineNeuronActivity(2)
    for rates in [[1,0],[3,2],[2,4]]:m.observe(rates=rates)
    out=m.result();np.testing.assert_allclose(out['mean'],[2.,2.]);np.testing.assert_allclose(out['variance'],[1.,4.])
    o=OccupancyMatrix(3)
    o.observe(state=0,action=3,next_state=1);o.observe(state=0,action=3,next_state=1)
    assert o.counts[0,1]==2 and o.result()['transition_probabilities'][0,1]==1

def test_smoke_sarsa_run_and_log_reconstruction(tmp_path):
    cfg=load_config(ROOT/'configs/experiments/experiment.yaml');cfg['output']=str(tmp_path)
    cfg['experiment']['phases']=[{'name':'A','episodes':2}]
    cfg['environment']['max_steps']=5
    ev,net,ag=run_one(cfg,seed=42)
    assert len(ev)==2 and net.W.shape==(30,30)
    out=tmp_path/'seed_42'
    assert (out/'network.npz').exists() and (out/'activity.json').exists()
    logs=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
    assert len(logs)==sum(row['steps'] for row in ev)
    assert sum(l['collision'] for l in logs)==sum(row['collisions'] for row in ev)
    assert sum(l['reward'] for l in logs)==pytest.approx(sum(row['reward'] for row in ev))
    assert json.loads((out/'activity.json').read_text())['n_samples']>0
    counts=np.load(out/'occupancy.npz')['counts']
    assert counts.sum()==len(logs)
    assert net.ticks>0

def test_motor_actor_real_rate_population(tmp_path):
    cfg=load_config(ROOT/'configs/experiments/motor_actor.yaml');cfg['output']=str(tmp_path)
    cfg['environment']['max_steps']=3
    cfg['experiment']['phases'][0]['episodes']=1
    ev,net,ag=run_one(cfg,seed=5)
    assert isinstance(ag,DopamineMotorAgent)
    assert net.motor is not None and net.motor.type_name()=='action_selection'
    assert net.motor.W.shape if hasattr(net.motor,'W') else True
    assert net.ticks>=100
    assert (tmp_path/'seed_5'/'rates.npz').exists()
    assert len(ev)==1 and np.isfinite(ag.W).all()

def test_policy_evaluation_no_learning_or_state_change(tmp_path):
    from btsp_lab.experiments.runner import evaluate_policy
    cfg=load_config(ROOT/'configs/experiments/experiment.yaml');cfg['output']=str(tmp_path)
    cfg['environment']['max_steps']=3
    cfg['experiment']['phases']=[{'name':'A','episodes':1}]
    events,net,ag=run_one(cfg,seed=4,write=False)
    W=net.W.copy();T=ag.theta.copy();trace=net.btsp.et.copy();ticks=net.ticks
    test=evaluate_policy(cfg,net,ag,'A',seed=198, n_episodes=3)
    assert len(test)==3
    assert np.array_equal(W,net.W) and np.array_equal(T,ag.theta)
    assert np.array_equal(trace,net.btsp.et) and ticks==net.ticks
    assert all(np.isfinite(row['spl']) for row in test)

def test_legacy_grid_all_edges_and_shortest_paths_parity(legacy_reference):
    GraphGrid=legacy_reference.GraphGrid
    original=GraphGrid(5,6,walls=[8,14,20],edge_blocks_A=[(4,5),(10,11)],
                       edge_blocks_B=[(22,23),(28,29)],goal=11)
    migrated=GridWorld(legacy_5x6_config())
    for phase in ['A','B']:
        np.testing.assert_array_equal(original.nexts[phase],migrated.nexts[phase])
        np.testing.assert_array_equal(original.shortest[phase][migrated.states],
                                      migrated.distances(phase)[migrated.states])
