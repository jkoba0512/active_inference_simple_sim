# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                           # Install dependencies
uv run python sim_1d.py           # Run 1D deterministic
uv run python sim_2d.py           # Run 2D deterministic (+ animation)
uv run python sim_1d_prob.py      # Run 1D probabilistic
uv run python sim_2d_prob.py      # Run 2D probabilistic (+ animation)
uv run python sim_1d_efe.py       # Run 1D with EFE for action selection
uv run ruff check .               # Lint
uv run ruff format .              # Format
```

## Architecture

| File | Belief Type | Dimensions | Animation |
|------|-------------|------------|-----------|
| `sim_1d.py` | Deterministic (μ) | 1D | No |
| `sim_2d.py` | Deterministic (μ) | 2D | Yes |
| `sim_1d_prob.py` | Probabilistic (μ, σ) | 1D | No |
| `sim_2d_prob.py` | Probabilistic (μ, σ) | 2D | Yes |
| `sim_1d_efe.py` | VFE + EFE (μ, μ_v) | 1D | No |

Each simulation has:
1. **`update_physics`**: Generative process (JIT-compiled)
2. **`compute_vfe`** / **`compute_vfe_prob`**: VFE computation (JIT-compiled)
3. **Gradient functions**: Auto-diff for belief, action, (and σ for probabilistic)
4. **`run_experiment`**: Main loop - 1500 steps, friction change at step 500

Probabilistic versions add:
- `log_sigma`: Log belief uncertainty (ensures σ > 0)
- KL divergence term in VFE
- Uncertainty visualization (bands in 1D, ellipses in 2D)

EFE version (`sim_1d_efe.py`) separates:
- **VFE**: Used for perception (updating beliefs μ, μ_v from observations)
- **EFE**: Used for action selection (planning based on predicted outcomes)
- **Internal model**: `predict_next_state` with agent's assumed friction `b_model`
- Demonstrates model mismatch when true friction changes at step 500

## Key Parameters

**Deterministic:**
| Parameter | Description |
|-----------|-------------|
| `p_obs`, `p_prior`, `p_action` | Precision weights |
| `action_gain` | Expected action scaling |

**Probabilistic:**
| Parameter | Description |
|-----------|-------------|
| `sigma_obs` | Observation noise model |
| `sigma_prior` | Prior uncertainty |
| `lr_mu`, `lr_sigma`, `lr_action` | Separate learning rates |

**EFE:**
| Parameter | Description |
|-----------|-------------|
| `pi_obs`, `pi_v` | Observation precisions (position, velocity) |
| `pi_target` | Goal prior precision |
| `b_model` | Agent's internal model of friction |
| `lr_mu`, `lr_mu_v`, `lr_action` | Separate learning rates |

## Code Style

- Ruff: E, F, I rules; line-length 88; double quotes
