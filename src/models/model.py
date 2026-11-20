"""Composable online CA3->CA1 circuit, optional EC3/OLM and CA3 replay.

Legacy configuration (no new enable flags) preserves the original rate-only
CA3->CA1 pathway and A2 BTSP rule. Mechanistic additions are *opt in*:

* ca3_recurrent.enabled: recurrent CA3 plus symmetric BTSP-like learning;
* ec3.enabled: entorhinal input to a separate distal CA1 drive;
* olm.enabled: feedback CA1->OLM and OLM inhibition of that distal drive;
* plateau_mode=dendritic: seconds-rate hazard driven by EC3 minus OLM;
* offline_replay.enabled: autonomous CA3 recurrent reactivation, optionally
  consolidated by *symmetric* offline rate-STDP (not SWR spiking dynamics).

The somatic CA1 response continues to depend on W_CA3_CA1 @ CA3 activity;
EC3 and OLM gate *plateau initiation*, not somatic firing directly.
"""
import numpy as np
from .neurons import (RateConfig, CA3Population, CA1Population,
                      ActionPopulation, EC3Population, OLMPopulation)
from ..learning.learning_rules import (A2BTSP, A2Parameters, RateSTDP,
                             STDPParameters, ThetaCompressedSTDP, PairSTDP)
from .microcircuit import (CA3SymmetricConfig, SymmetricCA3BTSP,
                           SymmetricRateSTDP)


class Circuit:
    def __init__(self, n_states, config, a2_params: A2Parameters, seed=0):
        self.dt = float(config.get('dt_s', .004))
        self.dwell = float(config.get('dwell_s', .4))
        if (not np.isfinite(self.dt) or not np.isfinite(self.dwell) or
                self.dt <= 0 or self.dwell <= 0 or
                abs(self.dwell/self.dt-round(self.dwell/self.dt)) > 1e-9):
            raise ValueError('invalid dt/dwell')
        self.n_ticks = int(round(self.dwell/self.dt))
        self.rng = np.random.default_rng(seed)
        neuron_cfg = config.get('neurons', {})
        self.ca3 = CA3Population(RateConfig(size=n_states, **neuron_cfg.get('ca3', {})))
        self.ca1 = CA1Population(RateConfig(size=config.get('n_ca1', n_states),
                                            **neuron_cfg.get('ca1', {})))
        self.btsp = A2BTSP(n_states, self.ca1.cfg.size, a2_params, dt=self.dt,
                           plateau_s=float(config.get('plateau_s', .3)))
        self.W = self.btsp.W
        self.plastic = bool(config.get('btsp', True))
        self.btsp_input = config.get('btsp_input', 'ca3_rate')
        if self.btsp_input not in ('ca3_rate', 'sensory_onehot'):
            raise ValueError('invalid BTSP source')

        self.stdp_kind = config.get('stdp', 'none')
        self.spike_scales = config.get('spike_rate_hz_per_unit', None)
        if self.stdp_kind == 'pair':
            if (not isinstance(self.spike_scales, dict) or
                not all(k in self.spike_scales for k in ('ca3', 'ca1'))):
                raise ValueError('pair STDP needs spike_rate_hz_per_unit: {ca3,ca1}')
            if any(not np.isfinite(float(self.spike_scales[k])) or
                   float(self.spike_scales[k]) <= 0 for k in ('ca3', 'ca1')):
                raise ValueError('invalid spike rate units')
        self.stdp = None
        if self.stdp_kind != 'none':
            sp = STDPParameters(**{'wmax': a2_params.wmax,
                                   **config.get('stdp_params', {})})
            if not np.isclose(sp.wmax, a2_params.wmax, atol=1e-12, rtol=1e-12):
                raise ValueError('shared BTSP+STDP matrix requires identical wmax')
            if self.stdp_kind == 'rate':
                self.stdp = RateSTDP(self.W, sp, self.dt)
            elif self.stdp_kind == 'pair':
                self.stdp = PairSTDP(self.W, sp, self.dt)
            elif self.stdp_kind == 'theta_rate':
                self.stdp = ThetaCompressedSTDP(
                    RateSTDP(self.W, sp, self.dt),
                    history=int(config.get('theta_history', 8)),
                    replay_ticks=int(config.get('theta_replay_ticks', 16)),
                    silent_ticks=int(config.get('theta_silent_ticks', 4)))
            else:
                raise ValueError(f'unsupported STDP kind {self.stdp_kind}')

        self.plateau_mode = config.get('plateau_mode', 'every_visit')
        if self.plateau_mode not in ('every_visit', 'bernoulli', 'off', 'dendritic'):
            raise ValueError('bad plateau mode')
        self.plateau_prob = float(config.get('plateau_probability', 1.0))
        if not np.isfinite(self.plateau_prob) or not 0 <= self.plateau_prob <= 1:
            raise ValueError('invalid plateau_probability')
        self.plateau_map = {int(k): int(v) for k, v in config.get('plateau_map', {}).items()}
        self.inhibition = config.get('ca1_inhibition', None)
        if self.inhibition is not None:
            self.inhibition = np.broadcast_to(np.asarray(self.inhibition, dtype=float),
                                               self.ca1.rate.shape).copy()
            if not np.isfinite(self.inhibition).all():
                raise ValueError('nonfinite somatic inhibition')
        self.rest_s = float(config.get('inter_episode_rest_s', 0.))
        self.flush = bool(config.get('flush_btsp_traces', False))
        if (not np.isfinite(self.rest_s) or self.rest_s < 0 or
                abs(self.rest_s/self.dt-round(self.rest_s/self.dt)) > 1e-9):
            raise ValueError('bad rest duration')
        self.terminal_observation = bool(config.get('terminal_observation', True))
        self.motor = None
        self.motor_W = None
        self.motor_lateral_gain = float(config.get('motor_lateral_gain', 0.))
        if not np.isfinite(self.motor_lateral_gain) or self.motor_lateral_gain < 0:
            raise ValueError('invalid motor lateral inhibition')

        # Recurrent CA3: distinct weight matrix and temporally symmetric
        # plasticity. Never copy the asymmetric A2 CA3->CA1 kernel to CA3.
        c3conf = dict(config.get('ca3_recurrent', {}))
        self.ca3_recurrent_enabled = bool(c3conf.pop('enabled', False))
        self.ca3_btsp_enabled = bool(c3conf.pop('btsp', True))
        self.W_ca3_ca3 = None
        self.ca3_btsp = None
        self.ca3_recurrent_cfg = None
        if self.ca3_recurrent_enabled:
            self.ca3_recurrent_cfg = CA3SymmetricConfig(**c3conf)
            cc = self.ca3_recurrent_cfg
            self.W_ca3_ca3 = np.full((n_states, n_states), cc.initial_w)
            np.fill_diagonal(self.W_ca3_ca3, 0.)
            self.ca3_btsp = SymmetricCA3BTSP(self.W_ca3_ca3, cc, self.dt)

        # EC3 provides a distinct state-dependent dendritic input. The map is
        # topographic, NOT a learned EC3 cortical representation.
        econf = dict(config.get('ec3', {}))
        self.ec3_enabled = bool(econf.get('enabled', False))
        self.ec3 = None
        self.W_ec3_ca1 = None
        self.ec3_gain = float(econf.get('gain', 1.))
        self.ec3_width = float(econf.get('tuning_width_states', 1.))
        if not all(np.isfinite(z) and z > 0 for z in (self.ec3_gain, self.ec3_width)):
            raise ValueError('invalid EC3 gain/width')
        if self.ec3_enabled:
            self.ec3 = EC3Population(RateConfig(size=n_states,
                                                **neuron_cfg.get('ec3', {})))
            positions = np.arange(self.ca1.cfg.size) * n_states / self.ca1.cfg.size
            positions_pre = np.arange(n_states)
            projection = np.exp(-.5*((positions[:, None]-positions_pre[None, :]) /
                                       self.ec3_width)**2)
            self.W_ec3_ca1 = self.ec3_gain * projection / np.maximum(
                1e-12, np.max(projection, axis=1, keepdims=True))

        oconf = dict(config.get('olm', {}))
        self.olm_enabled = bool(oconf.get('enabled', False))
        self.olm = None
        self.W_ca1_olm = None
        self.W_olm_ca1 = None
        if self.olm_enabled:
            feedback_gain = float(oconf.get('feedback_gain', 1.))
            inhibition_gain = float(oconf.get('inhibition_gain', 1.))
            if any(not np.isfinite(v) or v < 0 for v in (feedback_gain, inhibition_gain)):
                raise ValueError('invalid OLM connection gains')
            # Local rate feedback. Explicit matrices allow later replacement
            # with anatomically grounded, non-diagonal projections.
            self.olm = OLMPopulation(RateConfig(size=self.ca1.cfg.size,
                                                **neuron_cfg.get('olm', {})))
            self.W_ca1_olm = np.eye(self.ca1.cfg.size) * feedback_gain
            self.W_olm_ca1 = np.eye(self.ca1.cfg.size) * inhibition_gain

        hconf = config.get('plateau_hazard', {})
        self.hazard_max_hz = float(hconf.get('max_rate_hz', .8))
        self.hazard_gain = float(hconf.get('gain', 4.))
        self.hazard_threshold = float(hconf.get('threshold', .5))
        self.tonic_dendritic_drive = float(hconf.get('tonic_drive', 0.))
        self.plateau_refractory_s = float(hconf.get('refractory_s', 1.))
        if any(not np.isfinite(v) for v in (self.hazard_max_hz, self.hazard_gain,
                 self.hazard_threshold, self.tonic_dendritic_drive,
                 self.plateau_refractory_s)) or min(self.hazard_max_hz,
                 self.hazard_gain, self.plateau_refractory_s) < 0:
            raise ValueError('invalid dendritic plateau hazard')
        self.last_dendritic_drive = np.zeros(self.ca1.cfg.size)
        self.last_plateau_hazard_hz = np.zeros(self.ca1.cfg.size)
        self.last_plateau_tick = np.full(self.ca1.cfg.size, -10**12, dtype=np.int64)
        if self.plateau_mode == 'dendritic' and not self.ec3_enabled and self.tonic_dendritic_drive == 0:
            raise ValueError('dendritic plateaus require EC3 excitation or tonic_drive')

        # Offline replay is not the online theta_rate proxy; separate learning
        # occurs only in the explicitly enabled offline phase.
        rep = dict(config.get('offline_replay', {}))
        self.replay_enabled = bool(rep.get('enabled', False))
        self.replay_duration_s = float(rep.get('duration_s', 0.))
        self.replay_noise_sd = float(rep.get('noise_sd', 0.))
        self.replay_adaptation_gain = float(rep.get('adaptation_gain', 1.))
        self.replay_adaptation_tau_s = float(rep.get('adaptation_tau_s', .2))
        self.replay_cue_strength = float(rep.get('cue_strength', 1.))
        self.replay_min_active_rate = float(rep.get('min_active_rate', .05))
        self.offline_stdp_enabled = bool(rep.get('symmetric_stdp', True))
        self.offline_stdp_tau_s = float(rep.get('stdp_tau_s', .03))
        self.offline_stdp_eta = float(rep.get('stdp_eta', .02))
        if self.replay_enabled:
            if not self.ca3_recurrent_enabled:
                raise ValueError('offline replay requires ca3_recurrent.enabled')
            if (not all(np.isfinite(z) for z in (self.replay_duration_s,
                    self.replay_noise_sd, self.replay_adaptation_gain,
                    self.replay_adaptation_tau_s, self.replay_cue_strength,
                    self.replay_min_active_rate, self.offline_stdp_tau_s,
                    self.offline_stdp_eta)) or
                self.replay_duration_s <= 0 or self.replay_noise_sd < 0 or
                self.replay_adaptation_gain < 0 or self.replay_adaptation_tau_s <= 0 or
                self.replay_cue_strength < 0 or self.replay_min_active_rate < 0 or
                self.offline_stdp_tau_s <= 0 or self.offline_stdp_eta < 0 or
                abs(self.replay_duration_s/self.dt-round(self.replay_duration_s/self.dt)) > 1e-9):
                raise ValueError('invalid offline replay parameters')
        self.offline_stdp_ca3 = None
        self.offline_stdp_ca1 = None
        if self.replay_enabled and self.offline_stdp_enabled:
            self.offline_stdp_ca3 = SymmetricRateSTDP(
                self.W_ca3_ca3, self.dt, self.offline_stdp_tau_s,
                self.offline_stdp_eta, self.ca3_recurrent_cfg.wmax, autapses=False)
            self.offline_stdp_ca1 = SymmetricRateSTDP(
                self.W, self.dt, self.offline_stdp_tau_s,
                self.offline_stdp_eta, a2_params.wmax)
        self.replay_records = []
        self.last_state = None
        self.ticks = 0

    def attach_motor(self, motor_weights, motor_config=None):
        self.motor_W = motor_weights
        self.motor = ActionPopulation(RateConfig(size=motor_weights.shape[0],
                                                  **(motor_config or {})))

    def reset_dynamics(self):
        self.ca3.reset()
        self.ca1.reset()
        if self.ec3 is not None:
            self.ec3.reset()
        if self.olm is not None:
            self.olm.reset()
        if self.motor is not None:
            self.motor.reset()
        if self.stdp is not None:
            self.stdp.reset()
        self.last_plateau_tick.fill(-10**12)

    def _plateau_indices(self, state):
        if not self.plastic or self.plateau_mode in ('off', 'dendritic'):
            return ()
        if self.plateau_mode == 'bernoulli' and self.rng.random() >= self.plateau_prob:
            return ()
        if state in self.plateau_map:
            return (self.plateau_map[state],)
        if 0 <= state < self.ca1.cfg.size:
            return (state,)
        raise ValueError(f'plateau target undefined for state {state}; supply plateau_map')

    def _ca3_step(self, phi):
        if self.ca3_recurrent_enabled:
            feedback = self.ca3_recurrent_cfg.feedback_inhibition * float(self.ca3.rate.mean())
            return self.ca3.integrate_observation(phi, self.dt, self.W_ca3_ca3,
                       self.ca3_recurrent_cfg.online_recurrent_gain, feedback)
        return self.ca3.integrate_observation(phi, self.dt)

    def _plateau_hazard(self):
        """Per-neuron hazard in Hz; event probability is 1-exp(-rate*dt)."""
        drive = np.full(self.ca1.cfg.size, self.tonic_dendritic_drive)
        if self.ec3 is not None:
            drive += self.W_ec3_ca1 @ self.ec3.rate
        if self.olm is not None:
            drive -= self.W_olm_ca1 @ self.olm.rate
        self.last_dendritic_drive[:] = drive
        z = np.clip(self.hazard_gain * (drive-self.hazard_threshold), -60., 60.)
        rate = self.hazard_max_hz/(1.+np.exp(-z))
        self.last_plateau_hazard_hz[:] = rate
        return rate

    def _dendritic_plateaus(self):
        if not self.plastic or self.plateau_mode != 'dendritic':
            return ()
        rate = self._plateau_hazard()
        prob = -np.expm1(-rate*self.dt)
        eligible = (self.ticks-self.last_plateau_tick)*self.dt >= self.plateau_refractory_s
        indices = np.flatnonzero((self.rng.random(rate.size) < prob) & eligible)
        self.last_plateau_tick[indices] = self.ticks
        return indices

    def _motor_tick(self, post):
        if self.motor is None:
            return
        m = self.motor.rate
        lateral = self.motor_lateral_gain*(m.sum()-m)
        self.motor.integrate_synapses(self.motor_W, post, self.dt, lateral)

    def _online_stdp(self, pre, post):
        if self.stdp_kind == 'rate':
            self.stdp.step(pre, post)
        elif self.stdp_kind == 'theta_rate':
            self.stdp.step(self.ca1, self.W, self.inhibition)
        elif self.stdp_kind == 'pair':
            pre_sp = self.rng.poisson(pre*float(self.spike_scales['ca3'])*self.dt)
            post_sp = self.rng.poisson(post*float(self.spike_scales['ca1'])*self.dt)
            self.stdp.step(pre_sp, post_sp)

    def present(self, state, logger=None):
        if not 0 <= state < self.ca3.cfg.size:
            raise ValueError('bad observation state')
        self.last_state = int(state)
        if logger is not None:
            logger.presentation(state, self.ticks, self.n_ticks)
        phi = np.zeros(self.ca3.cfg.size)
        phi[state] = 1.
        if self.plastic:
            self.btsp.induce(self._plateau_indices(state))
        for k in range(self.n_ticks):
            pre = self._ca3_step(phi)
            post = self.ca1.integrate_synapses(self.W, pre, self.dt, self.inhibition)
            if self.ec3 is not None:
                self.ec3.integrate_observation(phi, self.dt)
            if self.olm is not None:
                # Feedback follows the current CA1 output, not future activity.
                self.olm.integrate(self.W_ca1_olm @ post, self.dt)
            if self.plastic:
                self.btsp.induce(self._dendritic_plateaus())
                self.btsp.step(pre if self.btsp_input == 'ca3_rate' else phi)
            if self.ca3_btsp is not None and self.plastic:
                self.ca3_btsp.step(pre, (state,) if (k == 0 and
                                      self.ca3_btsp_enabled) else ())
            self._online_stdp(pre, post)
            self._motor_tick(post)
            if logger is not None:
                logger.tick(post, self.ticks)
            self.ticks += 1
        if self.stdp_kind == 'theta_rate':
            self.stdp.observe(self.ca3.rate)
        return self.ca1.rate.copy()

    def _rest_tick(self):
        """No sensory input, but integrate real residual rates through rules."""
        pre = self._ca3_step(np.zeros(self.ca3.cfg.size))
        post = self.ca1.integrate_synapses(self.W, pre, self.dt, self.inhibition)
        if self.ec3 is not None:
            self.ec3.integrate_observation(np.zeros(self.ec3.cfg.size), self.dt)
        if self.olm is not None:
            self.olm.integrate(self.W_ca1_olm @ post, self.dt)
        self._motor_tick(post)
        if self.plastic:
            self.btsp.step(pre if self.btsp_input == 'ca3_rate' else np.zeros_like(pre))
        if self.ca3_btsp is not None and self.plastic:
            self.ca3_btsp.step(pre)
        if self.stdp_kind == 'rate':
            self.stdp.step(pre, post)
        elif self.stdp_kind == 'theta_rate':
            # Rest is physical residual activity, not a fabricated theta replay.
            self.stdp.base.step(pre, post)
        elif self.stdp_kind == 'pair':
            self._online_stdp(pre, post)
        self.ticks += 1

    def _offline_replay(self):
        if not self.replay_enabled:
            return
        # Start from a previously observed, never future, place cue. The seed
        # is an explicit assumption, not a claim of spontaneous SWR initiation.
        if self.last_state is not None:
            self.ca3.rate[self.last_state] = max(self.ca3.rate[self.last_state],
                                                  self.replay_cue_strength)
        n = int(round(self.replay_duration_s/self.dt))
        fatigue = np.zeros(self.ca3.cfg.size)
        decoded = []
        peaks = []
        if self.offline_stdp_ca3 is not None:
            self.offline_stdp_ca3.reset()
            self.offline_stdp_ca1.reset()
        for _ in range(n):
            drive = (self.ca3_recurrent_cfg.recurrent_gain*(self.W_ca3_ca3 @ self.ca3.rate)
                     - self.ca3_recurrent_cfg.feedback_inhibition*self.ca3.rate.mean()
                     - self.replay_adaptation_gain*fatigue)
            if self.replay_noise_sd:
                drive += self.rng.normal(0., self.replay_noise_sd, size=drive.shape)
            pre = self.ca3.integrate_observation(drive, self.dt)
            post = self.ca1.integrate_synapses(self.W, pre, self.dt, self.inhibition)
            fatigue += (-np.expm1(-self.dt/self.replay_adaptation_tau_s))*(pre-fatigue)
            if self.offline_stdp_ca3 is not None and self.plastic:
                self.offline_stdp_ca3.step(pre, pre)
                self.offline_stdp_ca1.step(pre, post)
            peak = float(np.max(pre))
            peaks.append(peak)
            decoded.append(int(np.argmax(pre)) if peak > self.replay_min_active_rate else -1)
            self.ticks += 1
        self.replay_records.append({
            'cue_state': self.last_state,
            'n_ticks': n,
            'active_ticks': int(sum(i >= 0 for i in decoded)),
            'decoded_states': decoded,
            'peak_rates': peaks,
            'offline_ca3_updates': n if self.offline_stdp_ca3 and self.plastic else 0,
            'offline_ca1_updates': n if self.offline_stdp_ca1 and self.plastic else 0,
            'n_transitions': int(sum(a >= 0 and b >= 0 and a != b
                                     for a, b in zip(decoded, decoded[1:]))),
        })
        # Replay consumes physical time but cannot trigger CA1/CA3 BTSP.
        self.btsp.advance_without_learning(n)
        if self.ca3_btsp is not None:
            self.ca3_btsp.advance_without_learning(n)

    def end_episode(self):
        for _ in range(int(round(self.rest_s/self.dt))):
            self._rest_tick()
        self._offline_replay()
        if self.flush:
            self.btsp.clear_traces()
            if self.ca3_btsp is not None:
                self.ca3_btsp.reset_traces()
        self.reset_dynamics()
