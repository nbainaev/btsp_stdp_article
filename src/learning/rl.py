"""RL agents use CA1 output *rates*, never internal CA3->CA1 weights."""
import numpy as np
from .learning_rules import TDLambda,RewardModulatedHebbian
from .critic import LinearCritic

class LinearActionValue:
    def __init__(self,kind,n_features,n_actions=4,alpha=.05,gamma=.9,lam=.8,epsilon=.15,seed=0,bootstrap_trunc=True):
        if kind not in ('sarsa','q_learning'):raise ValueError('kind')
        if kind=='q_learning' and lam!=0:
            raise ValueError('q_learning implements Q-learning(0) only; nonzero lambda requires an explicitly specified Watkins/Peng variant')
        if not 0<=epsilon<=1:raise ValueError('epsilon')
        self.kind=kind;self.theta=np.zeros((n_actions,n_features),dtype=float)
        self.rule=TDLambda(self.theta.shape,alpha,gamma,lam)
        self.epsilon=epsilon;self.rng=np.random.default_rng(seed);self.bootstrap_trunc=bootstrap_trunc
    def q_values(self,rates):return self.theta@rates
    def action(self,rates,train=True):
        qs=self.q_values(rates)
        if train and self.rng.random()<self.epsilon:return int(self.rng.integers(len(qs)))
        ties=np.flatnonzero(np.isclose(qs,np.max(qs),rtol=1e-11,atol=1e-11))
        return int(self.rng.choice(ties))
    def update(self,rates,action,reward,next_rates,next_action,terminated,truncated):
        old=float(self.theta[action]@rates)
        terminal=terminated or (truncated and not self.bootstrap_trunc)
        if terminal:target=reward
        elif self.kind=='sarsa':
            if next_action is None:raise ValueError('SARSA requires next_action for bootstrapping')
            target=reward+self.rule.gamma*float(self.theta[next_action]@next_rates)
        else:target=reward+self.rule.gamma*float(np.max(self.q_values(next_rates)))
        g=np.zeros_like(self.theta);g[action]=rates
        self.rule.update(self.theta,g,target-old)
        if terminated or truncated:self.rule.reset()
        return float(target-old)
    def reset_episode(self):self.rule.reset()


class ActorCritic:
    """Rate input, stochastic competing action population, TD-critic.

    The actor eligibility (action-onehot minus expected probability) is a
    policy-gradient reference; NOT claimed synapse-local without competition
    circuitry implementing expected activity.
    """
    def __init__(self,n_features,n_actions=4,seed=0,alpha_actor=.01,alpha_critic=.01,gamma=.9,lam=.8,temperature=1.):
        if temperature<=0:raise ValueError('temperature')
        self.W=np.zeros((n_actions,n_features));self.critic=LinearCritic(n_features,alpha_critic,gamma,lam)
        self.rule=TDLambda(self.W.shape,alpha_actor,gamma,lam)
        self.temp=temperature;self.rng=np.random.default_rng(seed)
    def probs(self,rate):
        z=self.W@rate/self.temp;z-=np.max(z);p=np.exp(z);return p/p.sum()
    def action(self,rate,train=True):
        prob=self.probs(rate)
        return int(self.rng.choice(len(prob),p=prob)) if train else int(np.argmax(prob))
    def update(self,rate,action,reward,next_rate,terminated,truncated):
        p=self.probs(rate)
        grad=(-p)[:,None]*rate[None,:]/self.temp
        grad[action,:]+=rate/self.temp
        delta=self.critic.update(rate,reward,next_rate,terminated,truncated)
        self.rule.update(self.W,grad,delta)
        if terminated or truncated:self.rule.reset()
        return delta
    def reset_episode(self):self.rule.reset();self.critic.reset_episode()

class RewardHebbianActor:
    """Experimental reward-modulated coactivity actor, no policy-gradient claim."""
    def __init__(self,n_features,n_actions=4,seed=0,alpha=.01,tau_s=.5,epsilon=.15):
        self.W=np.zeros((n_actions,n_features));self.rng=np.random.default_rng(seed)
        self.rule=RewardModulatedHebbian(self.W,alpha,tau_s);self.epsilon=epsilon
    def action(self,rate,train=True):
        q=self.W@rate;ties=np.flatnonzero(np.isclose(q,q.max()))
        if train and self.rng.random()<self.epsilon:return int(self.rng.integers(len(q)))
        return int(self.rng.choice(ties))
    def update(self,rate,action,reward,next_rate,terminated,truncated,dt=0.4):
        post=np.eye(self.W.shape[0])[action]
        self.rule.observe(rate,post,dt);self.rule.reinforce(reward)
        if terminated or truncated:self.rule.reset()
        return float(reward)
    def reset_episode(self):self.rule.reset()

class DopamineMotorAgent:
    """Phenomenological striatal/motor-population actor with TD-modulated
    coactivity plasticity and Welford-compatible continuous motor rates.

    Motor population is integrated by Circuit at EACH numerical tick; at choice,
    epsilon-greedy samples competition of motor rates. Dopamine gates a local
    action-activity eligibility. This is NOT an exact policy gradient.
    """
    def __init__(self,n_features,n_actions=4,seed=0,alpha_actor=.002,alpha_critic=.01,
                 gamma=.9,lam=.8,epsilon=.12,tau_elig_s=.4,weight_init=.001):
        self.rng=np.random.default_rng(seed)
        self.W=self.rng.uniform(0,weight_init,(n_actions,n_features))
        self.rule=RewardModulatedHebbian(self.W,alpha_actor,tau_elig_s,bounds=(0,None))
        self.critic=LinearCritic(n_features,alpha_critic,gamma,lam)
        self.epsilon=epsilon;self.n_actions=n_actions
    def action(self,rates,train=True,motor_rates=None):
        if motor_rates is None:raise ValueError('motor rates required; integrate action population')
        if train and self.rng.random()<self.epsilon:return int(self.rng.integers(self.n_actions))
        q=np.asarray(motor_rates);ties=np.flatnonzero(np.isclose(q,q.max(),atol=1e-12,rtol=1e-11))
        return int(self.rng.choice(ties))
    def update(self,rate,action,reward,next_rate,terminated,truncated,dt=.4):
        # Actor learns from choice-conditioned coactivity and critic TD-error.
        # Action selection signal is external disinhibition / efference-copy proxy.
        delta=self.critic.update(rate,reward,next_rate,terminated,truncated)
        post=np.eye(self.n_actions)[action]
        self.rule.observe(rate,post,dt)
        self.rule.reinforce(delta)
        if terminated or truncated:self.rule.reset()
        return delta
    def reset_episode(self):self.rule.reset();self.critic.reset_episode()
