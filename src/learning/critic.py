"""Scalar TD(lambda) critic on current rate-based CA1 activity."""
import numpy as np
from .learning_rules import TDLambda

class LinearCritic:
    """TD critic uses the same TDLambda rule and raw CA1 rates."""
    def __init__(self,n_features,alpha=.01,gamma=.9,lam=.8,bootstrap_trunc=True):
        self.w=np.zeros(n_features);self.rule=TDLambda(self.w.shape,alpha,gamma,lam)
        self.bootstrap_trunc=bootstrap_trunc
    def value(self,rate):return float(self.w@rate)
    def update(self,r,reward,next_rate,terminated,truncated):
        terminal=terminated or (truncated and not self.bootstrap_trunc)
        err=reward+(0 if terminal else self.rule.gamma*self.value(next_rate))-self.value(r)
        self.rule.update(self.w,np.asarray(r),err)
        if terminated or truncated:self.rule.reset()
        return float(err)
    def reset_episode(self):self.rule.reset()
