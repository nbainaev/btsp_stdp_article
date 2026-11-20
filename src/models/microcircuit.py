"""Optional, phenomenological CA3 recurrent plasticity and offline learning.

CA3: symmetric-in-time exponential BTSP-like pre/plateau association. This
is deliberately distinct from the calibrated, asymmetric A2 CA3->CA1 rule.
The symmetric online kernel is implemented *causally*: a presynaptic trace
covers pre-before-plateau, a plateau trace covers pre-after-plateau.

Offline replay: autonomous recurrent CA3 rate dynamics after an initial cue
from the last *observed* state. No access to environment transition tables or
future states. Symmetric rate-STDP during this offline phase is a
phenomenological substitute for a spike-based SWR/replay rule.
"""
from dataclasses import dataclass
import numpy as np


def _nonnegative_finite(x, shape, name):
    arr = np.asarray(x, dtype=float)
    if arr.shape != shape or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f'{name} must be finite nonnegative vector with shape {shape}')
    return arr


@dataclass(frozen=True)
class CA3SymmetricConfig:
    tau_s: float = 1.0
    eta: float = 1.0
    wmax: float = 0.8
    initial_w: float = 0.0
    ltd_ratio: float = 0.05
    recurrent_gain: float = 1.2
    online_recurrent_gain: float = 0.15
    feedback_inhibition: float = 0.05

    def __post_init__(self):
        for name in ('tau_s', 'eta', 'wmax', 'initial_w', 'ltd_ratio', 'recurrent_gain', 'online_recurrent_gain', 'feedback_inhibition'):
            val = getattr(self, name)
            if not np.isfinite(val) or val < 0:
                raise ValueError(f'invalid CA3 parameter {name}')
        if self.tau_s <= 0 or self.wmax <= 0 or self.initial_w > self.wmax:
            raise ValueError('invalid CA3 timescale or weight bounds')


class SymmetricCA3BTSP:
    """Online causal symmetric exponential kernel for recurrent CA3 synapses.

    If a plateau occurs at t0, contributions of pre at t0±d are both
    proportional to (1-exp(-dt/tau))*exp(-d/tau). All weights share [0,wmax].
    No autapses (diagonal kept at zero). The update is a *model proposal*,
    not the Li et al. (2024) biophysical or exact fitted learning law.
    """
    def __init__(self, W, config: CA3SymmetricConfig, dt):
        W = np.asarray(W)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError('CA3 recurrence must be square')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive')
        self.W = W
        self.cfg = config
        self.dt = dt
        self.decay = float(np.exp(-dt / config.tau_s))
        self.pre_trace = np.zeros(W.shape[1])
        self.plateau_trace = np.zeros(W.shape[0])
        self.stats = {'plateau': 0, 'ltp': 0., 'ltd': 0.}

    def reset_traces(self):
        self.pre_trace.fill(0)
        self.plateau_trace.fill(0)

    def advance_without_learning(self, steps):
        """Advance offline time while freezing recurrent BTSP weights."""
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError('steps must be nonnegative integer')
        self.pre_trace *= self.decay**steps
        self.plateau_trace *= self.decay**steps

    def step(self, pre, plateau_indices=()):
        x = _nonnegative_finite(pre, self.pre_trace.shape, 'CA3 presynaptic rate')
        plateau = np.zeros_like(self.plateau_trace)
        for j in plateau_indices:
            if not 0 <= int(j) < plateau.size:
                raise IndexError('CA3 plateau index outside population')
            plateau[int(j)] = 1.
        self.stats['plateau'] += int(np.count_nonzero(plateau))
        # Pre trace contains activity strictly before this tick. Plateau trace
        # after adding today's event covers simultaneous and future activity.
        self.pre_trace *= self.decay
        past = np.outer(plateau, self.pre_trace)
        self.plateau_trace *= self.decay
        self.plateau_trace += plateau
        future = (1. - self.decay) * np.outer(self.plateau_trace, x)
        overlap = past + future
        lp = self.cfg.eta * overlap * (self.cfg.wmax - self.W)
        ld = self.cfg.eta * self.cfg.ltd_ratio * overlap * self.W
        np.fill_diagonal(lp, 0.)
        np.fill_diagonal(ld, 0.)
        self.W[:] = np.clip(self.W + lp - ld, 0., self.cfg.wmax)
        np.fill_diagonal(self.W, 0.)
        self.stats['ltp'] += float(lp.sum())
        self.stats['ltd'] += float(ld.sum())
        self.pre_trace += (1. - self.decay) * x


class SymmetricRateSTDP:
    """Offline positive time-symmetric coactivity with bounded weights.

    Uses pre/post traces from earlier ticks and both temporal directions;
    intentionally not called during online exploration. This is rate-STDP,
    not a fit to the millisecond spike-train protocol in Cho & McClelland.
    """
    def __init__(self, W, dt, tau_s=.025, eta=.02, wmax=1., autapses=True):
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(tau_s) or tau_s <= 0:
            raise ValueError('invalid offline STDP timescale')
        if not np.isfinite(eta) or eta < 0 or not np.isfinite(wmax) or wmax <= 0:
            raise ValueError('invalid offline STDP bounds')
        self.W = W
        self.dt = dt
        self.decay = float(np.exp(-dt/tau_s))
        self.eta = eta
        self.wmax = wmax
        self.autapses = autapses
        self.pre = np.zeros(W.shape[1])
        self.post = np.zeros(W.shape[0])
        self.n_updates = 0

    def reset(self):
        self.pre.fill(0.)
        self.post.fill(0.)

    def step(self, pre, post):
        x = _nonnegative_finite(pre, self.pre.shape, 'offline STDP pre')
        y = _nonnegative_finite(post, self.post.shape, 'offline STDP post')
        self.pre *= self.decay
        self.post *= self.decay
        # Both orders of pairwise activity produce potentiation, unlike
        # asymmetric online RateSTDP. dt yields per-second learning rate.
        overlap = np.outer(y, self.pre) + np.outer(self.post, x)
        delta = self.eta * self.dt * overlap * (self.wmax - self.W)
        if not self.autapses:
            np.fill_diagonal(delta, 0.)
        self.W[:] = np.clip(self.W + delta, 0., self.wmax)
        if not self.autapses:
            np.fill_diagonal(self.W, 0.)
        self.pre += x
        self.post += y
        self.n_updates += 1
