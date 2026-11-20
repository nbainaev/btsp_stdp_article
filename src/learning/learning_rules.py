"""Numerically explicit plasticity and TD rules.

A2BTSP replicates 2026-11-16 *discrete* 4-ms eligibility/instructive LUT update.
Other rules are separately labelled phenomenological abstractions, not claimed
fitted to electrophysiology. Rule implementations never access environment map/P.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from pathlib import Path
import numpy as np

class LearningRule(ABC):
    @abstractmethod
    def reset(self): ...

@dataclass(frozen=True)
class A2Parameters:
    tau_e: float; tau_i: float; is_scale: float; k_plus: float; k_minus: float
    pot_center: float; pot_width: float; dep_center: float; dep_width: float
    wmax: float; initial_w: float; eta: float
    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text())['parameters'])
    def __post_init__(self):
        if all(np.isfinite(v) for v in vars(self).values()) and min(self.tau_e,self.tau_i,self.pot_width,self.dep_width,self.wmax)>0 and 0<=self.initial_w<=self.wmax and min(self.eta,self.k_plus,self.k_minus,self.is_scale)>=0: return
        raise ValueError('invalid A2 parameters')

def _curve(x,center,width):
    def sig(z):return 1/(1+np.exp(-np.clip(z,-60.,60.)))
    y0=sig(-center/width);y1=sig((1-center)/width)
    return np.clip((sig((np.clip(x,0,1)-center)/width)-y0)/(y1-y0),0,1)

class A2BTSP(LearningRule):
    """Exact original per-tick ET/IS and LUT-interpolated saturating LTP/LTD.

    Persistent between visits; event durations must be a whole number of dt.
    No independent neuron firing is used as an instructive plateau signal.
    """
    def __init__(self, n_ca3, n_ca1, params:A2Parameters, dt=0.004, plateau_s=0.300, W=None):
        if dt<=0 or plateau_s<=0 or abs(plateau_s/dt-round(plateau_s/dt))>1e-9:
            raise ValueError('dt and plateau duration incompatible')
        self.n_pre=n_ca3;self.n_post=n_ca1;self.p=params;self.dt=dt
        self.duration_ticks=int(round(plateau_s/dt))
        self.W=np.full((n_ca1,n_ca3),params.initial_w,dtype=np.float64) if W is None else W
        if self.W.shape!=(n_ca1,n_ca3):raise ValueError('W wrong shape')
        self.et=np.zeros(n_ca3);self.ins=np.zeros(n_ca1)
        self.end_ticks=np.zeros(n_ca1,dtype=np.int64)
        self.ticks=0
        self.ptab=_curve(np.arange(4097)/4096,params.pot_center,params.pot_width)
        self.dtab=_curve(np.arange(4097)/4096,params.dep_center,params.dep_width)
        self.stats={'plateau':0,'ltp':0.,'ltd':0.,'weight_ticks':0}
    def reset(self):
        self.et.fill(0);self.ins.fill(0);self.end_ticks.fill(0);self.ticks=0
        for k in self.stats:self.stats[k]=0
    def clear_traces(self, reset_clock=False):
        self.et.fill(0);self.ins.fill(0);self.end_ticks.fill(0)
        if reset_clock:self.ticks=0
    def induce(self, post_indices):
        for j in post_indices:
            if not 0<=int(j)<self.n_post:raise IndexError('plateau neuron out of bounds')
            self.end_ticks[int(j)]=self.ticks+self.duration_ticks
            self.stats['plateau']+=1
    def step(self, pre_rate):
        x=np.asarray(pre_rate,dtype=float)
        if x.shape!=(self.n_pre,) or np.any(x<0) or np.any(~np.isfinite(x)):
            raise ValueError('BTSP pre must be nonnegative rate (n_CA3,)')
        p=self.p
        self.et += (-np.expm1(-self.dt/p.tau_e))*(x-self.et)
        target=p.is_scale*(self.ticks<self.end_ticks)
        self.ins += (-np.expm1(-self.dt/p.tau_i))*(target-self.ins)
        for j in range(self.n_post):
            if self.ins[j]<1.e-12:continue
            coord=np.minimum(4096,np.maximum(0,self.et*self.ins[j]*4096))
            ix=coord.astype(np.int64);fraction=coord-ix
            # When ix=4096, f==0, no out-of-bound index needed.
            pot=self.ptab[ix]*(1-fraction)+self.ptab[np.minimum(ix+1,4096)]*fraction
            dep=self.dtab[ix]*(1-fraction)+self.dtab[np.minimum(ix+1,4096)]*fraction
            plus=self.dt*p.eta*p.k_plus*pot*(p.wmax-self.W[j]);minus=self.dt*p.eta*p.k_minus*dep*self.W[j]
            self.W[j]=np.clip(self.W[j]+plus-minus,0,p.wmax)
            self.stats['ltp']+=float(plus.sum());self.stats['ltd']+=float(minus.sum())
            self.stats['weight_ticks']+=self.n_pre
        self.ticks+=1
    def advance_without_learning(self, steps):
        """Advance physical time with NO BTSP updates during offline replay.

        Online plateau and eligibility traces cannot stand still while network
        time advances; plateau signals can naturally finish during this phase.
        No replay-related presynaptic activity enters this online BTSP trace.
        """
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        for _ in range(steps):
            self.et *= np.exp(-self.dt/self.p.tau_e)
            target = self.p.is_scale*(self.ticks < self.end_ticks)
            self.ins += (-np.expm1(-self.dt/self.p.tau_i))*(target-self.ins)
            self.ticks += 1

    def advance(self, pre, steps, plateau_at_start=()):
        self.induce(plateau_at_start)
        for _ in range(steps):self.step(pre)
    def flush(self, duration_s=5.0, reset_traces=True):
        nticks=round(duration_s/self.dt)
        if abs(nticks*self.dt-duration_s)>1e-9:raise ValueError('flush duration not divisible by dt')
        for _ in range(nticks):self.step(np.zeros(self.n_pre))
        if reset_traces:self.clear_traces(reset_clock=False)

@dataclass(frozen=True)
class STDPParameters:
    tau_pre_s: float=0.020
    tau_post_s: float=0.040
    eta: float=0.01
    ltd_ltp_ratio: float=1.
    wmax: float=1.
    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()) or min(self.tau_pre_s,self.tau_post_s,self.wmax)<=0 or self.eta<0 or self.ltd_ltp_ratio<0: raise ValueError('invalid STDP')

class RateSTDP(LearningRule):
    """Rate-based asymmetric pre/post traces; source grid5study/neural.py lines 369-425.

    Uses OLD traces for all weight changes. Exponential traces updated AFTER
    the synaptic change. No hidden time compression/normalization.
    """
    def __init__(self,W,params:STDPParameters,dt=0.004):
        self.W=W;self.p=params;self.dt=dt
        self.pre=np.zeros(W.shape[1]);self.post=np.zeros(W.shape[0])
    def reset(self):self.pre.fill(0);self.post.fill(0)
    def step(self,pre_rate,post_rate):
        x=np.asarray(pre_rate,dtype=float);y=np.asarray(post_rate,dtype=float)
        if x.shape!=self.pre.shape or y.shape!=self.post.shape or np.any(x<0) or np.any(y<0) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('bad STDP input')
        dw=self.dt*self.p.eta*(y[:,None]*self.pre[None,:] - self.p.ltd_ltp_ratio*self.post[:,None]*x[None,:])
        self.W[:]=np.clip(self.W+dw,0,self.p.wmax)
        self.pre+=(-np.expm1(-self.dt/self.p.tau_pre_s))*(x-self.pre)
        self.post+=(-np.expm1(-self.dt/self.p.tau_post_s))*(y-self.post)
        return dw

class PairSTDP(LearningRule):
    """Event-based trace rule; pre-before-post potentiates, opposite depresses.

    Uses W-dependent bounds like 2026 hybrid starter, not physiological fit.
    """
    def __init__(self,W,params:STDPParameters,dt=0.004):
        self.W=W;self.p=params;self.dt=dt
        self.pre=np.zeros(W.shape[1]);self.post=np.zeros(W.shape[0])
    def reset(self):self.pre.fill(0.);self.post.fill(0.)
    def step(self,pre_spikes,post_spikes):
        x=np.asarray(pre_spikes,dtype=float);y=np.asarray(post_spikes,dtype=float)
        if x.shape!=self.pre.shape or y.shape!=self.post.shape or np.any(x<0) or np.any(y<0) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('bad spikes')
        self.pre*=np.exp(-self.dt/self.p.tau_pre_s);self.post*=np.exp(-self.dt/self.p.tau_post_s)
        lp=self.p.eta*y[:,None]*self.pre[None,:]*(self.p.wmax-self.W)/self.p.wmax
        ld=self.p.eta*self.p.ltd_ltp_ratio*self.post[:,None]*x[None,:]*self.W/self.p.wmax
        self.W[:]=np.clip(self.W+lp-ld,0,self.p.wmax)
        self.pre+=x;self.post+=y
        return lp-ld

class ThetaCompressedSTDP(LearningRule):
    """Exploratory *rate-coded* compressed replay of observed CA3 patterns.

    On replay each tick uses the same CA3->CA1 transduction as real activity.
    BTSP must ONLY receive the actual CA3 state, never these replay samples.
    No future states, no oracle P, no claim of real theta-phase-precession.
    """
    def __init__(self, base_stdp:RateSTDP, history=8, replay_ticks=16, silent_ticks=4):
        if history<1 or replay_ticks<1 or silent_ticks<0:raise ValueError('invalid theta parameters')
        self.base=base_stdp;self.history=history;self.replay_ticks=replay_ticks;self.silent_ticks=silent_ticks
        self.events=[];self.tick=0
    def reset(self):self.base.reset();self.events=[];self.tick=0
    def observe(self,ca3_pattern):
        self.events.append(np.asarray(ca3_pattern,dtype=float).copy())
        if len(self.events)>self.history:self.events.pop(0)
    def step(self, CA1, W, inhibition=None):
        cycle=self.replay_ticks+self.silent_ticks
        phase=self.tick%cycle
        if phase<self.replay_ticks and self.events:
            i=min(len(self.events)-1,int(phase*len(self.events)/self.replay_ticks))
            pre=self.events[i]
            drive=W@pre
            if inhibition is not None:drive=drive-np.broadcast_to(inhibition,drive.shape)
            post=CA1.response(drive)
        else:
            pre=np.zeros(self.base.pre.shape);post=np.zeros(self.base.post.shape)
        result=self.base.step(pre,post)
        self.tick+=1
        return result

class Hebbian(LearningRule):
    def __init__(self,W,eta=0.001,decay=0.,bounds=None):self.W=W;self.eta=eta;self.decay=decay;self.bounds=bounds
    def reset(self):pass
    def step(self,pre,post,dt):
        dw=dt*self.eta*np.outer(post,pre)-dt*self.decay*self.W
        self.W[:]+=dw
        if self.bounds is not None:np.clip(self.W,*self.bounds,out=self.W)
        return dw

class RewardModulatedHebbian(LearningRule):
    """Local coactivity eligibility multiplied by external reinforcement signal."""
    def __init__(self,W,eta=0.001,tau_s=1.,bounds=None):
        if not np.isfinite(tau_s) or tau_s <= 0 or not np.isfinite(eta) or eta < 0: raise ValueError('invalid reward eligibility parameters')
        self.W=W;self.eta=eta;self.tau=tau_s;self.bounds=bounds;self.elig=np.zeros_like(W)
    def reset(self):self.elig.fill(0)
    def observe(self,pre,post,dt):
        self.elig+=(-np.expm1(-dt/self.tau))*(np.outer(post,pre)-self.elig)
    def reinforce(self,signal):
        self.W[:]+=self.eta*signal*self.elig
        if self.bounds is not None:np.clip(self.W,*self.bounds,out=self.W)

class TDLambda(LearningRule):
    """General semi-gradient TD(lambda) for a linear parameter matrix/vector.

    'features' is gradient of the selected current value wrt theta, already
    formatted to theta.shape. Bootstrap target is produced by caller; for SF
    use a vector-valued delta and a matrix eligibility outer product.
    """
    def __init__(self,shape,alpha,gamma,lam):
        if not(0<=gamma<=1 and 0<=lam<=1 and alpha>=0): raise ValueError('bad TD parameters')
        self.alpha=alpha;self.gamma=gamma;self.lam=lam;self.elig=np.zeros(shape)
    def reset(self):self.elig.fill(0)
    def update(self,theta,gradient,td_error):
        if theta.shape!=self.elig.shape or gradient.shape!=self.elig.shape:
            raise ValueError('TD gradient/parameter shape')
        if np.asarray(td_error).ndim != 0: raise ValueError('use update_vector for vector-valued cumulants')
        self.elig[:]=self.gamma*self.lam*self.elig+gradient
        theta[:]+=self.alpha*float(td_error)*self.elig
        return theta
    def update_vector(self,theta,gradient,td_error):
        # Same eligibility update as scalar TD, one independent cumulant error
        # per output; weights share the same local input eligibility vector.
        err=np.asarray(td_error,dtype=float)
        if (theta.ndim!=2 or theta.shape[0]!=self.elig.size or
            gradient.shape!=self.elig.shape or err.shape!=(theta.shape[1],)):
            raise ValueError('vector TD shape mismatch')
        self.elig[:]=self.gamma*self.lam*self.elig+gradient
        theta[:]+=self.alpha*np.outer(self.elig,err)
        return theta

class LinearSuccessorFeatures:
    """TD(lambda) SF in arbitrary fixed-feature coordinates; separate from CA1."""
    def __init__(self,n_input,n_cumulants,alpha,gamma,lam):
        self.W=np.zeros((n_input,n_cumulants))
        self.rule=TDLambda((n_input,),alpha,gamma,lam)
    def update(self,features,cumulants,next_features,terminated,truncated=False,bootstrap_trunc=True):
        terminal=terminated or (truncated and not bootstrap_trunc)
        target=np.asarray(cumulants)+ (0. if terminal else self.rule.gamma*np.asarray(next_features)@self.W)
        err=target-np.asarray(features)@self.W
        self.rule.update_vector(self.W,np.asarray(features),err)
        if terminated or truncated:self.rule.reset()
        return err
