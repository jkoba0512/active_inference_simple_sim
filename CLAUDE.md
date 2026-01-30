# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                           # Install dependencies
uv run python sim_1d.py           # Run 1D deterministic
uv run python sim_2d.py           # Run 2D deterministic (+ animation)
uv run python sim_1d_prob.py      # Run 1D probabilistic
uv run python sim_2d_prob.py      # Run 2D probabilistic (+ animation)
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

Each simulation has:
1. **`update_physics`**: Generative process (JIT-compiled)
2. **`compute_vfe`** / **`compute_vfe_prob`**: VFE computation (JIT-compiled)
3. **Gradient functions**: Auto-diff for belief, action, (and σ for probabilistic)
4. **`run_experiment`**: Main loop - 1500 steps, friction change at step 500

Probabilistic versions add:
- `log_sigma`: Log belief uncertainty (ensures σ > 0)
- KL divergence term in VFE
- Uncertainty visualization (bands in 1D, ellipses in 2D)

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

## Code Style

- Ruff: E, F, I rules; line-length 88; double quotes
