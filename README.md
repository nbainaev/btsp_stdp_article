# BTSP Lab

A rate-based CA3 → CA1 plasticity and reinforcement-learning research framework. The implementation lives under `src/`, mapped to the `btsp_lab` Python namespace. Optional recurrent CA3, EC3/OLM and offline rate reactivation are model variants; replay-like activity is not a sharp-wave ripple model, and smoke configurations do not establish biological fit or RL performance.

## Installation

Python 3.10 or newer is required. From the project root, install in an environment with access to the dependencies:

```sh
python -m pip install -e '.[test,tune]'
```

NumPy and PyYAML are required. The test extra includes pytest and Numba; the tune extra includes Optuna. Version ranges in `pyproject.toml` do not pin a complete environment.

## Experiments and configuration

From the project root, after installing the package:

```sh
btsp-run --config configs/experiments/smoke_rate.yaml --output run_outputs/smoke_rate
btsp-run --config configs/experiments/smoke_motor.yaml --output run_outputs/smoke_motor
btsp-run --config configs/experiments/smoke_full_microcircuit.yaml --output run_outputs/smoke_full
btsp-run --config configs/experiments/experiment.yaml --output run_outputs/experiment
btsp-run --config configs/experiments/motor_actor.yaml --output run_outputs/motor_actor
python experiments/replay_demo.py --output run_outputs/replay_demo.json
```

Each manifest selects fragments in `configs/models`, `configs/env`, `configs/learners`, and `configs/runner`; paths resolve relative to the manifest. The model parameters are in `configs/models/btsp_a2.yaml`. `overrides` recursively merge mappings; lists and scalars replace the corresponding values. `load_config` returns an assembled dictionary accepted by `run_one`. Legacy flat YAML with a `btsp_parameters` JSON path remains supported, with the path resolved relative to that file. Output paths resolve relative to the working directory; use a fresh directory for each retained run.

Optuna is disabled in the standard presets. The explicit `configs/experiments/experiment_tuned.yaml` preset enables the original grid search profile in `configs/runner/grid.yaml`; it uses the existing development and held-out test seeds. It does not run unless explicitly selected. `python -m btsp_lab.experiments.runner` is equivalent to `btsp-run`.

## Checks and provenance

```sh
python -m pytest tests -q
```

Run tests and simulations only where authorized. Two parity checks require the optional ignored `reference/closed_loop_2026_11_16.py` and the original byte-identical parameter file at `reference/A2_SECONDS_STRICT.json`; they skip individually if either is absent. The fixture copies these files to a temporary layout because the original source checks the parameter file's hash. The historical JSON is not an active configuration. Archived manuscript, historical code and prior run data remain local outside the active package.
