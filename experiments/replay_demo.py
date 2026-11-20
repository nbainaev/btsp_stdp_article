"""Reproducible *rate-based* replay capacity probe (not a SWR reproduction).

Run from project root:
python experiments/replay_demo.py --output verification/replay_demo.json
No environment transition table or future state is supplied to offline replay.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from btsp_lab.models.model import Circuit
from btsp_lab.learning.learning_rules import A2Parameters
from btsp_lab.experiments.runner import load_config


def experiment():
    p = A2Parameters(**load_config(Path(__file__).resolve().parents[1]/'configs/experiments/experiment.yaml')['btsp_parameters'])
    cfg = {
        'dt_s': .004, 'dwell_s': .2, 'n_ca1': 5,
        'btsp': True, 'plateau_mode': 'off',
        'neurons': {'ca3': {'tau_s': .03, 'max_rate': 1.}},
        'ca3_recurrent': {'enabled': True, 'tau_s': .15, 'eta': .4,
                          'online_recurrent_gain': .05, 'recurrent_gain': 1.2,
                          'feedback_inhibition': .05},
        'offline_replay': {'enabled': True, 'duration_s': 1.,
                            'symmetric_stdp': False, 'adaptation_gain': 1.,
                            'noise_sd': 0., 'cue_strength': 1.},
    }
    circuit = Circuit(5, cfg, p, seed=8)
    for lap in range(6):
        for state in range(5):
            circuit.present(state)
    ca3_before_replay = circuit.W_ca3_ca3.copy()
    circuit.end_episode()
    r = circuit.replay_records[-1]
    raw = r['decoded_states']
    seq = [s for i, s in enumerate(raw) if s >= 0 and (i == 0 or raw[i-1] != s)]
    return {
        'protocol': '6 fixed online laps 0-1-2-3-4; offline seeded by last observed 4',
        'seed': 8,
        'kernel': 'symmetric CA3 rate BTSP-like',
        'n_states': 5,
        'ca3_w_min': float(np.min(ca3_before_replay)),
        'ca3_w_max': float(np.max(ca3_before_replay)),
        'offline_spike_model': False,
        'offline_swr_model': False,
        'offline_active_ticks': r['active_ticks'],
        'offline_distinct_states': sorted(set(seq)),
        'offline_transition_count': r['n_transitions'],
        'offline_compressed_sequence': seq,
        'conclusion': 'sequential rate reactivation' if len(set(seq)) >= 3 else 'no convincing sequence',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='verification/replay_demo.json')
    args = parser.parse_args()
    result = experiment()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
