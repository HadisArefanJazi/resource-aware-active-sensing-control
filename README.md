# Resource-Aware Active Sensing and Control

A compact PyTorch benchmark for learning two decisions at the same time: how to control a partially observed system and when extra information is worth its cost.

## How it works

The environment is an unstable linear system:

```text
x[t+1] = a * x[t] + b * u[t] + disturbance
```

The controller does not observe the true state. It receives a belief estimate, uncertainty, measurement age, information use, episode progress, and the previous control.

At each step, the policy chooses one of nine joint actions:

- control: `-1`, `0`, or `+1`
- information: none, local sensing, or communication

Local sensing costs 1 unit, while higher-quality communication costs 3 units. A failed sensor or communication attempt still uses resources.

The policy is trained with a PyTorch actor-critic and projected Lagrange multipliers. It aims to maximize control reward while meeting two average constraints:

```text
information cost per step <= 0.80
state-limit violation rate <= 0.05
```

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
resource-control experiment --episodes 1500 --eval-episodes 100
resource-control demo --scenario sensor_dropout
```

The experiment saves metrics to `results/metrics.json` and policy weights to `artifacts/`.

## Evaluation

The default run trains constrained and unconstrained RL policies for 1,500 episodes. It evaluates five controllers over 100 episodes in six scenarios: nominal, noisy, disturbed, sensor dropout, limited communication, and shifted dynamics.

| Policy | Reward | Control RMSE | Information cost | State violations |
| --- | ---: | ---: | ---: | ---: |
| Resource-aware RL | -10.84 | 0.507 | **31.85** | **0.000** |
| Full information | **-7.09** | **0.378** | 120.00 | **0.000** |
| Fixed sensing | -20.19 | 0.669 | 10.00 | 0.002 |
| Unconstrained RL | -15.49 | 0.608 | 71.27 | 0.002 |
| Classical control | -11.95 | 0.513 | 40.00 | **0.000** |

The information budget is 32 units per 40-step episode. These results describe the bundled simulation and seed; they are not claims about physical systems. Full metrics are available in [`results/metrics.json`](results/metrics.json).

## Project structure

- `environment.py`: dynamics, sensing channels, and belief updates
- `networks.py`: policy and value networks
- `training.py`: primal-dual actor-critic training
- `baselines.py`: comparison controllers
- `evaluation.py`: episode and constraint metrics
- `experiment.py`: training and robustness evaluation
- `tests/`: unit and integration tests

## Scope

This is a focused research implementation for constrained active sensing and control. Physical deployment would require hardware-specific validation and safeguards.

## License

MIT
