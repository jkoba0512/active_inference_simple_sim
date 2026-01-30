# Active Inference Simulations

Minimal implementations of Active Inference demonstrating how an agent achieves goal-directed behavior by minimizing Variational Free Energy (VFE).

## What is Active Inference?

Active Inference is a theoretical framework from neuroscience that unifies perception and action under a single principle: minimizing surprise (or free energy). An agent maintains beliefs about the world and acts to make its observations match its predictions.

## Simulations

### Deterministic (point estimate beliefs)

| File        | Description                  | Target           |
| ----------- | ---------------------------- | ---------------- |
| `sim_1d.py` | 1D point mass simulation     | x = 10           |
| `sim_2d.py` | 2D point mass with animation | (x, y) = (10, 7) |

### Probabilistic (Gaussian beliefs with uncertainty)

| File             | Description                              | Target           |
| ---------------- | ---------------------------------------- | ---------------- |
| `sim_1d_prob.py` | 1D with belief uncertainty (σ)           | x = 10           |
| `sim_2d_prob.py` | 2D with uncertainty ellipses + animation | (x, y) = (10, 7) |

All simulations model a point mass that must reach a target position while adapting to environmental changes:

- **Agent**: Maintains beliefs about position and generates control actions
- **Environment**: Physics world with mass, friction, and stiffness
- **Challenge**: At step 500, friction increases 10x to test robustness

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# Deterministic simulations
uv run python sim_1d.py
uv run python sim_2d.py

# Probabilistic simulations (with uncertainty tracking)
uv run python sim_1d_prob.py
uv run python sim_2d_prob.py
```

## Key Concepts

### Deterministic VFE

```python
# Belief is a point estimate (μ)
vfe = 0.5 * (
    p_obs * (observation - μ)²      # Sensory error
  + p_prior * (target - μ)²         # Prior error
  + p_action * (action - expected)² # Action model error
)
```

### Probabilistic VFE

```python
# Belief is a Gaussian distribution N(μ, σ²)
vfe = accuracy + complexity
    = -log p(observation | μ, σ) + KL(q(x) || p(x))
```

The agent updates both belief mean (μ) and uncertainty (σ) via gradient descent. Uncertainty decreases when observations are consistent with beliefs.

## References

- Friston, K. (2010). **The free-energy principle: a unified brain theory?** _Nature Reviews Neuroscience_, 11(2), 127-138. [[Nature]](https://www.nature.com/articles/nrn2787) [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/20068583/)
