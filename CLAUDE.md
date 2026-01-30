# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                      # Install dependencies
uv run python sim_1d.py      # Run 1D simulation
uv run ruff check .          # Lint
uv run ruff format .         # Format
```

## Architecture

1D Active Inference simulation (`sim_1d.py`) with four components:

1. **`update_physics`**: Generative process - 1D point mass physics (JIT-compiled)
2. **`compute_vfe`**: Generative model - Variational Free Energy with three error terms (JIT-compiled)
3. **`grad_vfe_mu`, `grad_vfe_u`**: Auto-diff gradients for belief and action updates
4. **`run_experiment`**: Main loop - 1500 steps with friction change at step 500

## Key Parameters

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| `p_obs` | Sensory precision | `learning_rate * p_obs < 1` for stability |
| `p_prior` | Prior precision | |
| `p_action` | Action model precision | |
| `action_gain` | Expected action scaling | |

## Code Style

- Ruff with rules: E (pycodestyle), F (Pyflakes), I (isort)
- Line length: 88 (E501 ignored)
- Double quotes for strings
