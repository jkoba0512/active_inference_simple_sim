# Active Inference Simulations

Minimal implementations of Active Inference demonstrating how an agent achieves goal-directed behavior by minimizing Variational Free Energy (VFE).

## What is Active Inference?

Active Inference is a theoretical framework from neuroscience that unifies perception and action under a single principle: minimizing surprise (or free energy). An agent maintains beliefs about the world and acts to make its observations match its predictions.

## Simulations

| File | Description | Target |
|------|-------------|--------|
| `sim_1d.py` | 1D point mass simulation | x = 10 |
| `sim_2d.py` | 2D point mass with animation | (x, y) = (10, 7) |

Both simulations model a point mass that must reach a target position while adapting to environmental changes:

- **Agent**: Has internal beliefs about position and generates control actions
- **Environment**: Physics world with mass, friction, and stiffness
- **Challenge**: At step 500, friction increases 10x to test robustness

The agent learns through two gradient descent updates:

1. **Perception**: Update belief (μ) to match observations
2. **Action**: Update control (u) to achieve the target

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# 1D simulation (static plot)
uv run python sim_1d.py

# 2D simulation (static plot + animation saved to MP4)
uv run python sim_2d.py
```

## Key Concepts in Code

```python
# Variational Free Energy combines three prediction errors:
vfe = 0.5 * (
    p_obs * (observation - belief)²     # Sensory error
  + p_prior * (target - belief)²        # Prior error
  + p_action * (action - expected)²     # Action model error
)

# Agent minimizes VFE by updating belief and action via gradients
belief -= learning_rate * ∂VFE/∂belief
action -= learning_rate * ∂VFE/∂action
```

## Parameters

| Parameter       | Description                   | Default |
| --------------- | ----------------------------- | ------- |
| `p_obs`         | Trust in sensory observations | 2.0     |
| `p_prior`       | Strength of goal preference   | 1.0     |
| `p_action`      | Action model precision        | 0.1     |
| `action_gain`   | Expected action scaling       | 0.5     |
| `learning_rate` | Gradient descent step size    | 0.2     |

## References

- Friston, K. (2010). **The free-energy principle: a unified brain theory?** _Nature Reviews Neuroscience_, 11(2), 127-138. [[Nature]](https://www.nature.com/articles/nrn2787) [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/20068583/)

- Buckley, C. L., Kim, C. S., McGregor, S., & Seth, A. K. (2017). **The free energy principle for action and perception: A mathematical review.** _Journal of Mathematical Psychology_, 81, 55-79. [[arXiv]](https://arxiv.org/abs/1705.09156) [[ScienceDirect]](https://www.sciencedirect.com/science/article/pii/S0022249617300962)
