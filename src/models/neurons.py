"""Rate-population dynamics. All activity is actual continuous network output, not W rows.

The model is a phenomenological rate abstraction, not an ion-channel model.
The exact-exponential Euler drive assumes piecewise constant synaptic drive.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RateConfig:
    size: int
    tau_s: float = 0.05
    activation: str = "relu"
    gain: float = 1.0
    threshold: float = 0.0
    max_rate: float | None = None
    baseline: float = 0.0
    def __post_init__(self):
        if self.size < 1 or not all(np.isfinite(v) for v in (self.tau_s, self.gain, self.threshold, self.baseline)) or self.tau_s <= 0 or self.gain < 0: raise ValueError('invalid neuron parameters')
        if self.activation not in ('relu','sigmoid','softplus'): raise ValueError('unknown activation')
        if self.max_rate is not None and (not np.isfinite(self.max_rate) or self.max_rate <= 0): raise ValueError('max_rate must be >0')

class RatePopulation(ABC):
    """Shared rate dynamics; type-specific populations may add circuit-specific signals."""
    def __init__(self, config: RateConfig):
        self.cfg = config
        self.rate = np.zeros(config.size, dtype=np.float64)
        self.current = np.zeros(config.size, dtype=np.float64)
    def response(self, current):
        z = self.cfg.gain * (np.asarray(current, dtype=float) - self.cfg.threshold) + self.cfg.baseline
        if self.cfg.activation == 'relu': z = np.maximum(z,0.)
        elif self.cfg.activation == 'sigmoid': z = 1. / (1. + np.exp(-np.clip(z,-60,60)))
        elif self.cfg.activation == 'softplus': z = np.logaddexp(0.,z)
        if self.cfg.max_rate is not None: z = np.minimum(self.cfg.max_rate,z)
        return z
    def integrate(self, drive, dt):
        drive = np.asarray(drive, dtype=float)
        if not np.isfinite(dt) or dt <= 0 or drive.shape != self.rate.shape or not np.all(np.isfinite(drive)):
            raise ValueError('invalid dt or drive shape/values')
        self.current[:] = drive
        self.rate += (-np.expm1(-dt/self.cfg.tau_s))*(self.response(drive)-self.rate)
        if not np.all(np.isfinite(self.rate)): raise FloatingPointError('nonfinite rate')
        return self.rate.copy()
    def reset(self):
        self.rate.fill(0.); self.current.fill(0.)
    @abstractmethod
    def type_name(self): ...

class CA3Population(RatePopulation):
    def type_name(self): return 'CA3'
    def integrate_observation(self, feature, dt, recurrent_weights=None,
                              recurrent_gain=1., inhibition=0.):
        drive = np.asarray(feature, dtype=float)
        if recurrent_weights is not None:
            weights = np.asarray(recurrent_weights, dtype=float)
            if weights.shape != (self.cfg.size, self.cfg.size):
                raise ValueError('CA3 recurrent W shape mismatch')
            # This read precedes the rate update, preserving causal time order.
            drive = drive + recurrent_gain * (weights @ self.rate) - inhibition
        return self.integrate(drive, dt)

class CA1Population(RatePopulation):
    def type_name(self): return 'CA1'
    def integrate_synapses(self, W, ca3_rate, dt, inhibition=None):
        W=np.asarray(W); ca3_rate=np.asarray(ca3_rate)
        if W.shape != (self.cfg.size,ca3_rate.size): raise ValueError('CA3->CA1 W shape mismatch')
        total = W @ ca3_rate
        if inhibition is not None: total -= np.broadcast_to(inhibition,total.shape)
        return self.integrate(total,dt)

class ActionPopulation(RatePopulation):
    def type_name(self): return 'action_selection'
    def integrate_synapses(self, W, ca1_rate, dt, inhibition=None):
        drive=np.asarray(W) @ np.asarray(ca1_rate)
        if inhibition is not None: drive -= np.broadcast_to(inhibition,drive.shape)
        return self.integrate(drive,dt)

class OLMPopulation(RatePopulation):
    def type_name(self): return 'OLM_interneuron'

class EC3Population(RatePopulation):
    """Entorhinal L3-like rate units; project to CA1 distal compartment."""
    def type_name(self): return 'EC3_entorhinal_layer3'
    def integrate_observation(self, feature, dt): return self.integrate(feature, dt)
