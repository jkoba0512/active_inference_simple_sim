# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                      # Install dependencies
uv run python sim_1d.py      # Run 1D simulation
uv run python sim_2d.py      # Run 2D simulation (static plot + animation)
uv run ruff check .          # Lint
uv run ruff format .         # Format
```

## Architecture

Two Active Inference simulations with shared structure:

| File | Description |
|------|-------------|
| `sim_1d.py` | 1D point mass reaching target x=10 |
| `sim_2d.py` | 2D point mass reaching target (10, 7) with animation |

Each simulation has four components:
1. **`update_physics`**: Generative process - physics simulation (JIT-compiled)
2. **`compute_vfe`**: Generative model - VFE with three error terms (JIT-compiled)
3. **`grad_vfe_mu`, `grad_vfe_u`**: Auto-diff gradients for belief and action
4. **`run_experiment`**: Main loop - 1500 steps, friction change at step 500

`sim_2d.py` adds:
- **`plot_results`**: Static 4-panel plot
- **`animate_results`**: Animated trajectory (displays or saves to MP4)

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
