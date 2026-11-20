"""YAML configuration -> reproducible instantiated model, environment, RL, metrics.

Optional Optuna tunes only `tuning` development seeds and NEVER test seeds.
Call as python -m btsp_lab.experiments.runner --config configs/experiments/experiment.yaml
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,csv
import numpy as np
import yaml
from ..environments.grid import EnvConfig,GridWorld
from ..learning.learning_rules import A2Parameters
from ..models.model import Circuit
from ..learning.rl import LinearActionValue,ActorCritic,RewardHebbianActor,DopamineMotorAgent
from ..analysis.metrics import MetricsLogger

def build_rl(spec,n_features,seed):
    kind=spec.get('kind','sarsa');args={k:v for k,v in spec.items() if k!='kind'}
    if kind in ('sarsa','q_learning'):return LinearActionValue(kind,n_features,seed=seed,**args)
    if kind=='actor_critic':return ActorCritic(n_features,seed=seed,**args)
    if kind=='reward_hebb':return RewardHebbianActor(n_features,seed=seed,**args)
    if kind=='motor_actor_critic':return DopamineMotorAgent(n_features,seed=seed,**args)
    raise ValueError('unknown RL algorithm '+kind)


def evaluate_policy(config,network,rl,phase,seed=190001,n_episodes=8):
    """Greedy, immutable evaluation on deep copies of learner and CA1 synapses.

    BTSP/STDP are OFF during evaluation; downstream weights are frozen. No
    post-evaluation state leaks into subsequent training. Fresh start RNG.
    """
    eval_net,eval_agent=deepcopy((network,rl))
    eval_net.plastic=False;eval_net.stdp_kind='none'
    env=GridWorld(EnvConfig(**config['environment']),seed=seed)
    out=[]
    for k in range(n_episodes):
        s=env.reset(phase=phase);start=s
        eval_net.reset_dynamics()
        if s in env.goals:
            out.append({'reward':0.,'steps':0,'success':1,'spl':1.,'collisions':0});continue
        reward_sum=0.;steps=0;success=False;collisions=0
        for _ in range(env.cfg.max_steps):
            rate=eval_net.present(s)
            motor=eval_net.motor.rate if eval_net.motor is not None else None
            action=eval_agent.action(rate,train=False,motor_rates=motor) if isinstance(eval_agent,DopamineMotorAgent) else eval_agent.action(rate,train=False)
            ns,rew,terminated,truncated,info=env.step(action)
            reward_sum+=rew;steps+=1;collisions+=int(info['collision'])
            success|=info['goal_reached']
            if terminated or truncated:break
            s=ns
        out.append({'reward':float(reward_sum),'steps':steps,'success':int(success),
                   'spl':env.spl(start,steps,success,phase),'collisions':collisions})
    return out

def run_one(config,seed=0,output=None,write=True):
    env=GridWorld(EnvConfig(**config['environment']),seed=seed)
    params_spec=config['btsp_parameters']
    params=A2Parameters(**params_spec) if isinstance(params_spec,dict) else A2Parameters.from_json(params_spec)
    network_conf=config.get('network',{})
    network=Circuit(env.n,network_conf,params,seed=seed+210001)
    rl=build_rl(config.get('rl',{}),network.ca1.cfg.size,seed=seed+100023)
    if isinstance(rl,DopamineMotorAgent):
        network.attach_motor(rl.W,network_conf.get('neurons',{}).get('motor',{}))
    phases=config.get('experiment',{}).get('phases',[{'name':'A','episodes':5}])
    paths=Path(output or config.get('output','run_outputs')) / f'seed_{seed}'
    mconf=config.get('metrics',{})
    logger=MetricsLogger(paths,env.n,network.ca1.cfg.size,
         save_events=mconf.get('save_events',True),save_rates=mconf.get('save_rates',False),rate_stride=mconf.get('rate_stride',10)) if write else None
    episodes=[];epid=0;eval_rows=[]
    eval_spec=config.get('evaluation',{})
    for phase_index,phase_spec in enumerate(phases):
        phase=phase_spec['name']
        if phase not in env.nexts:raise ValueError(f'phase {phase} missing in topology')
        if logger is not None:logger.begin_phase(f'{phase_index}_{phase}')
        for _ in range(int(phase_spec['episodes'])):
            s=env.reset(phase=phase);start=s
            if logger is not None:logger.begin_episode(epid)
            if hasattr(rl,'reset_episode'):rl.reset_episode()
            network.reset_dynamics()
            if s in env.goals and env.cfg.terminate_on_goal:
                entry={'episode':epid,'phase':phase,'reward':0.,'steps':0,'success':1,'spl':1.,'collisions':0}
                episodes.append(entry)
                if network.terminal_observation:network.present(s,logger)
                if logger is not None:logger.finish_episode(epid,phase,start,1.,success_override=True)
                network.end_episode();epid+=1
                continue
            episode_reward=0.;collisions=0;success=False;steps=0
            rates=network.present(s,logger)
            motor=network.motor.rate if network.motor is not None else None
            a=rl.action(rates,train=True,motor_rates=motor) if isinstance(rl,DopamineMotorAgent) else rl.action(rates,train=True)
            while True:
                ns,reward,terminated,truncated,info=env.step(a)
                episode_reward+=reward;steps+=1;collisions+=int(info['collision'])
                success=success or bool(info['goal_reached'])
                # Separate sensory registration from RL bootstrap. Terminal
                # goals can still shape CA1 and receive plateaus, but target
                # values never bootstrap after a terminated transition.
                bootstrap=not terminated and (not truncated or getattr(rl,'bootstrap_trunc',True))
                do_observe=bootstrap or ((terminated or truncated) and network.terminal_observation)
                observed_next=network.present(ns,logger) if do_observe else np.zeros_like(rates)
                next_rate=observed_next if bootstrap else np.zeros_like(rates)
                # SARSA's next action must be chosen BEFORE its TD update.
                # Other policies choose after updating the current transition.
                if bootstrap and isinstance(rl,LinearActionValue) and rl.kind=='sarsa':
                    na=rl.action(next_rate,train=True)
                else:na=None
                if isinstance(rl,LinearActionValue):
                    rl.update(rates,a,reward,next_rate,na,terminated,truncated)
                elif isinstance(rl,DopamineMotorAgent):
                    rl.update(rates,a,reward,next_rate,terminated,truncated,dt=network.dwell)
                elif isinstance(rl,RewardHebbianActor):
                    rl.update(rates,a,reward,next_rate,terminated,truncated,dt=network.dwell)
                else:rl.update(rates,a,reward,next_rate,terminated,truncated)
                if logger is not None:
                    logger.transition(epid,phase,s,a,ns,reward,terminated,truncated,info['collision'],info['goal_reached'])
                if terminated or truncated:break
                if na is None:
                    motor_next=network.motor.rate if network.motor is not None else None
                    na=rl.action(next_rate,train=True,motor_rates=motor_next) if isinstance(rl,DopamineMotorAgent) else rl.action(next_rate,train=True)
                s,rates,a=ns,next_rate,na
            spl=env.spl(start,steps,success,phase)
            entry={'episode':epid,'phase':phase,'reward':episode_reward,'steps':steps,'success':int(success),'spl':spl,'collisions':collisions}
            episodes.append(entry)
            if logger is not None:logger.finish_episode(epid,phase,start,spl)
            network.end_episode();epid+=1
            interval=int(eval_spec.get('every_episodes',0))
            if write and eval_spec.get('enabled',False) and interval>0 and (epid%interval==0):
                evaluations=evaluate_policy(config,network,rl,phase,seed=seed+1000000+epid,
                                          n_episodes=int(eval_spec.get('n_episodes',8)))
                for item in evaluations:eval_rows.append({'after_training_episode':epid,'phase':phase,**item})
        if logger is not None:logger.end_phase()
    if logger is not None:
        logger.finish()
        np.savez_compressed(paths/'network.npz',W_ca3_ca1=network.W,
             W_ca3_ca3=(network.W_ca3_ca3 if network.W_ca3_ca3 is not None else np.empty((0,0))),
             W_ec3_ca1=(network.W_ec3_ca1 if network.W_ec3_ca1 is not None else np.empty((0,0))),
             W_ca1_olm=(network.W_ca1_olm if network.W_ca1_olm is not None else np.empty((0,0))),
             W_olm_ca1=(network.W_olm_ca1 if network.W_olm_ca1 is not None else np.empty((0,0))),
             W_actor=getattr(rl,'W',getattr(rl,'theta',np.empty((0,)))) ,
             W_critic=getattr(getattr(rl,'critic',None),'w',np.empty((0,))))
        (paths/'config.json').write_text(json.dumps({'config':config,'seed':seed},indent=2,ensure_ascii=False))
        (paths/'replay.json').write_text(json.dumps(network.replay_records))
        (paths/'plasticity.json').write_text(json.dumps({'btsp':network.btsp.stats,
            'ca3_btsp':network.ca3_btsp.stats if network.ca3_btsp is not None else None,
            'offline_replay':{'enabled':network.replay_enabled,
                'episodes':len(network.replay_records)},
            'stdp_kind':network.stdp_kind,'stdp_params':network_conf.get('stdp_params',{}),
            'plateau_mode':network.plateau_mode,'dt_s':network.dt,'dwell_s':network.dwell},indent=2))
        if eval_rows:
            with (paths/'evaluations.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(eval_rows[0]));w.writeheader();w.writerows(eval_rows)
    return episodes,network,rl

def _score(result):
    vals=[v['success'] for v in result]
    return float(np.mean(vals)) if vals else -np.inf

def optimize(config):
    try:import optuna
    except ImportError as e:raise RuntimeError('install pip install -e ".[tune]"') from e
    tune=config['optuna']
    if not tune.get('enabled',False):return config
    dev=tune.get('dev_seeds',[11,12]);test=tune.get('test_seeds',[101,102])
    if set(dev)&set(test):raise ValueError('Optuna dev/test overlap')
    search=tune['search_space']
    direction=tune.get('direction','maximize')
    study=optuna.create_study(direction=direction,sampler=optuna.samplers.TPESampler(seed=tune.get('seed',123)))
    def objective(trial):
        cfg=deepcopy(config)
        for key,params in search.items():
            node=cfg
            parts=key.split('.')
            for p in parts[:-1]:node=node[p]
            if params['kind']=='float':
                value=trial.suggest_float(key,params['low'],params['high'],log=params.get('log',False))
            elif params['kind']=='int':value=trial.suggest_int(key,params['low'],params['high'])
            else:raise ValueError('space kind')
            node[parts[-1]]=value
        return float(np.mean([_score(run_one(cfg,seed=s,write=False)[0]) for s in dev]))
    study.optimize(objective,n_trials=int(tune.get('trials',10)),n_jobs=int(tune.get('jobs',1)))
    cfg=deepcopy(config)
    for key,value in study.best_params.items():
        node=cfg;parts=key.split('.')
        for p in parts[:-1]:node=node[p]
        node[parts[-1]]=value
    Path(cfg.get('output','run_outputs')).mkdir(parents=True,exist_ok=True)
    (Path(cfg.get('output','run_outputs'))/'optuna_best.json').write_text(json.dumps({'best_value':study.best_value,'best_params':study.best_params,'dev_seeds':dev,'test_seeds':test},indent=2))
    cfg.setdefault('experiment',{})['seeds']=test
    return cfg

def _merge_config(base, changes):
    """Recursively merge mappings; lists and scalar values replace earlier values."""
    result=deepcopy(base)
    for key,value in changes.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=_merge_config(result[key],value)
        else:result[key]=deepcopy(value)
    return result

def load_config(path):
    path=Path(path)
    data=yaml.safe_load(path.read_text())
    if not isinstance(data,dict):raise ValueError('configuration must be a mapping')
    if 'components' in data:
        if set(data)-{'components','overrides'}:raise ValueError('unknown experiment manifest keys')
        parts=data['components']
        allowed={'plasticity':{'btsp_parameters'},'environment':{'environment'},
                 'model':{'network'},'learner':{'rl'},
                 'runner':{'output','experiment','metrics','evaluation','optuna'}}
        if not isinstance(parts,dict) or set(parts)!=set(allowed):
            raise ValueError('manifest must select plasticity, environment, model, learner and runner')
        result={}
        for kind,keys in allowed.items():
            if not isinstance(parts[kind],str) or not parts[kind].strip():
                raise ValueError(f'{kind} component path must be a non-empty string')
            fragment_path=(path.parent/parts[kind]).resolve()
            fragment=yaml.safe_load(fragment_path.read_text())
            if not isinstance(fragment,dict) or not fragment or not set(fragment)<=keys:
                raise ValueError(f'invalid {kind} component: {fragment_path}')
            if kind=='plasticity' and not isinstance(fragment.get('btsp_parameters'),dict):
                raise ValueError('plasticity component must contain a btsp_parameters mapping')
            result=_merge_config(result,fragment)
        overrides=data.get('overrides',{})
        if not isinstance(overrides,dict) or not set(overrides)<=set().union(*allowed.values()):
            raise ValueError('invalid experiment overrides')
        data=_merge_config(result,overrides)
    if 'btsp_parameters' not in data:raise ValueError('must supply btsp_parameters')
    if isinstance(data['btsp_parameters'],str):
        p=Path(data['btsp_parameters'])
        if not p.is_absolute():data['btsp_parameters']=str((path.parent/p).resolve())
    elif not isinstance(data['btsp_parameters'],dict):raise ValueError('btsp_parameters must be a mapping or JSON path')
    return data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output');args=parser.parse_args()
    cfg=load_config(args.config)
    if args.output:cfg['output']=args.output
    if cfg.get('optuna',{}).get('enabled',False):cfg=optimize(cfg)
    seeds=cfg.get('experiment',{}).get('seeds',[0]);summary=[]
    for seed in seeds:
        episodes,_,_=run_one(cfg,seed,write=True)
        row={'seed':seed,'success':_score(episodes),'reward':float(np.mean([e['reward'] for e in episodes]))}
        print(json.dumps(row,ensure_ascii=False),flush=True);summary.append(row)
    out=Path(cfg.get('output','run_outputs'));out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
