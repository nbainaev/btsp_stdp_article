"""Validated and fully configured deterministic grid/linear environments.

State numbering row-major r*cols+c. Edges and walls are separate primitives.
Collision is a self-transition and the configured collision reward ADDs to the
normal step reward. Goal reward replaces normal step reward. No hidden schedules.
"""
from dataclasses import dataclass,field
from collections import deque
import numpy as np

MOVES=((-1,0),(1,0),(0,-1),(0,1))  # UP DOWN LEFT RIGHT

@dataclass
class EnvConfig:
    kind: str='gridworld'
    rows: int=7
    cols: int=7
    walls: list=field(default_factory=list)
    blocked_edges: dict=field(default_factory=lambda:{'A':[]})
    start_states: list|None=None
    start: int|None=None
    goal: int|None=None
    goals: list|None=None
    start_distribution: list|None=None
    default_reward: float=-0.01
    goal_reward: float=0.99
    collision_reward: float=-0.10
    state_rewards: dict=field(default_factory=dict)
    max_steps: int=50
    terminate_on_goal: bool=True
    require_connected: bool=True
    def __post_init__(self):
        if self.kind not in ('gridworld','linear'):raise ValueError('kind gridworld/linear only')
        if self.kind=='linear' and self.rows!=1:raise ValueError('linear requires rows=1')
        if self.rows<1 or self.cols<1 or self.max_steps<1:raise ValueError('bad grid dimensions/max_steps')

class GridWorld:
    def __init__(self,config:EnvConfig,seed=0):
        self.cfg=config;self.rows=config.rows;self.cols=config.cols;self.n=config.rows*config.cols
        self.rng=np.random.default_rng(seed);self.walls=set(map(int,config.walls))
        if any(not 0<=s<self.n for s in self.walls):raise ValueError('invalid wall indices')
        self.states=np.array([s for s in range(self.n) if s not in self.walls],dtype=int)
        if not len(self.states):raise ValueError('no traversable cells')
        goals=config.goals if config.goals is not None else ([config.goal] if config.goal is not None else [int(self.states[-1])])
        self.goals=set(map(int,goals));self.goal=int(sorted(self.goals)[0])
        if not self.goals or not self.goals.issubset(set(self.states)):raise ValueError('invalid goals')
        if config.start is not None and config.start_states is not None: raise ValueError('choose start or start_states')
        starts=config.start_states if config.start_states is not None else ([config.start] if config.start is not None else [int(s) for s in self.states if s not in self.goals])
        self.starts=np.array(starts,dtype=int)
        if not len(self.starts) or not set(self.starts).issubset(set(self.states)):raise ValueError('bad start states')
        if config.start_distribution is not None:
            probs=np.asarray(config.start_distribution,dtype=float)
            if probs.shape!=(len(self.starts),) or np.any(probs<0) or not np.isclose(probs.sum(),1): raise ValueError('bad start distribution')
            self.start_prob=probs
        else:self.start_prob=None
        self.state_rewards={int(k):float(v) for k,v in config.state_rewards.items()}
        if any(x not in self.states for x in self.state_rewards):raise ValueError('state reward placed in a wall')
        self.edges={}
        for phase, raw in config.blocked_edges.items():
            block=set()
            for u,v in raw:
                u,v=int(u),int(v)
                if u not in self.states or v not in self.states:raise ValueError(f'blocked edge touches wall: {u},{v}')
                ur,uc=divmod(u,self.cols);vr,vc=divmod(v,self.cols)
                if abs(ur-vr)+abs(uc-vc)!=1:raise ValueError(f'blocked edge nonadjacent: {u},{v}')
                block.add(tuple(sorted((u,v))))
            self.edges[phase]=block
        if not self.edges:raise ValueError('no topology phases')
        self.nexts={p:self._table(block) for p,block in self.edges.items()}
        for ph in self.nexts:
            if config.require_connected and len(self.reachable(ph,int(self.states[0])))!=len(self.states):
                raise ValueError(f'phase {ph} is disconnected')
        self.phase=next(iter(self.nexts));self.position=None;self.steps=0
    def _table(self,blocked):
        nxt=np.full((self.n,4),-1,dtype=int)
        for s in self.states:
            r,c=divmod(int(s),self.cols)
            for a,(dr,dc) in enumerate(MOVES):
                rr,cc=r+dr,c+dc
                if 0<=rr<self.rows and 0<=cc<self.cols:
                    to=rr*self.cols+cc
                    if to not in self.walls and tuple(sorted((int(s),to))) not in blocked:nxt[s,a]=to
        return nxt
    def reachable(self,phase,start):
        if phase not in self.nexts:raise KeyError(phase)
        seen={int(start)};q=deque([int(start)])
        while q:
            for nxt in self.nexts[phase][q.popleft()]:
                if nxt>=0 and int(nxt) not in seen:seen.add(int(nxt));q.append(int(nxt))
        return seen
    def distances(self,phase=None,goals=None):
        ph=phase or self.phase
        goalset=self.goals if goals is None else set(goals)
        if not goalset.issubset(set(self.states)):raise ValueError('goal outside free cells')
        d=np.full(self.n,np.inf);q=deque(goalset)
        for s in goalset:d[s]=0
        # Graph is undirected: reverse BFS along valid edges.
        reverse=[[] for _ in range(self.n)]
        for s in self.states:
            for ns in self.nexts[ph][s]:
                if ns>=0: reverse[ns].append(int(s))
        while q:
            s=q.popleft()
            for prev in reverse[s]:
                if not np.isfinite(d[prev]):d[prev]=d[s]+1;q.append(prev)
        return d
    def reset(self,seed=None,phase=None,start=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        if phase is not None:
            if phase not in self.nexts:raise KeyError(phase)
            self.phase=phase
        self.position=int(self.rng.choice(self.starts,p=self.start_prob)) if start is None else int(start)
        if self.position not in self.states:raise ValueError('start is not navigable')
        self.steps=0
        return self.position
    def step(self,action):
        if self.position is None:raise RuntimeError('call reset')
        if not 0<=int(action)<4:raise ValueError('action must be 0-3')
        old=self.position;ns=int(self.nexts[self.phase][old,int(action)]);collision=ns<0
        if collision:ns=old
        self.position=ns;self.steps+=1
        goal=ns in self.goals
        terminated=bool(goal and self.cfg.terminate_on_goal)
        truncated=bool(self.steps>=self.cfg.max_steps and not terminated)
        reward=float(self.cfg.goal_reward if goal else self.state_rewards.get(ns,self.cfg.default_reward))
        if collision:reward+=self.cfg.collision_reward
        return ns,reward,terminated,truncated,{'collision':collision,'goal_reached':goal,'phase':self.phase}
    def spl(self,start,steps,success,phase=None):
        if not success:return 0.
        optimal=self.distances(phase)[start]
        if not np.isfinite(optimal):return 0.
        if optimal==0:return 1.0
        return float(optimal/max(optimal,steps))

def legacy_5x6_config():
    # Preserve exact original graph: (10,11) is valid, despite stale legacy comment.
    return EnvConfig(rows=5,cols=6,walls=[8,14,20],blocked_edges={'A':[[4,5],[10,11]],'B':[[22,23],[28,29]]},goal=11)

def make_linear(length=10,**kwargs):
    return GridWorld(EnvConfig(kind='linear',rows=1,cols=length,**kwargs))
