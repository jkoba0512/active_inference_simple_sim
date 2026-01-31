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

Belief is a point estimate $\mu$. The VFE is a sum of precision-weighted squared errors:

$$
F = \frac{1}{2} \left[ \pi_{obs}(o - \mu)^2 + \pi_{prior}(\mu_{target} - \mu)^2 + \pi_{action}(a - a_{expected})^2 \right]
$$

### Probabilistic VFE

Belief is a Gaussian distribution $q(x) = \mathcal{N}(\mu, \sigma^2)$. The VFE decomposes into accuracy and complexity:

$$
F = \underbrace{-\log p(o | \mu, \sigma)}_{\text{accuracy}} + \underbrace{D_{KL}(q(x) \| p(x))}_{\text{complexity}}
$$

The agent updates both belief mean $\mu$ and uncertainty $\sigma$ via gradient descent.

### Notation: p vs q

In variational inference, $p$ and $q$ denote different distribution families:

| Symbol        | Name              | Description                   | In code                                     |
| ------------- | ----------------- | ----------------------------- | ------------------------------------------- |
| $q(x)$        | Recognition model | Agent's belief about state    | $\mathcal{N}(\mu, \sigma^2)$                |
| $p(x)$        | Prior             | Where the agent expects to be | $\mathcal{N}(x_{target}, \sigma_{prior}^2)$ |
| $p(o \mid x)$ | Likelihood        | Observation model             | $\mathcal{N}(x, \sigma_{obs}^2)$            |

The letter $p$ is used for all distributions in the **generative model** (how the agent models the world), while $q$ is the **approximate posterior** (the agent's current belief).

### Accuracy Term

The accuracy term is derived from the negative log-likelihood of a Gaussian distribution.

**Step 1: Gaussian PDF**

The probability density of observing $o$ given belief $\mu$ with combined variance $\sigma^2 + \sigma_{obs}^2$:

$$
p(o | \mu, \sigma) = \frac{1}{\sqrt{2\pi(\sigma^2 + \sigma_{obs}^2)}} \exp\left( -\frac{(o - \mu)^2}{2(\sigma^2 + \sigma_{obs}^2)} \right)
$$

**Step 2: Take negative log**

$$
-\log p(o | \mu, \sigma) = \frac{1}{2}\log(2\pi) + \frac{1}{2}\log(\sigma^2 + \sigma_{obs}^2) + \frac{(o - \mu)^2}{2(\sigma^2 + \sigma_{obs}^2)}
$$

**Step 3: Drop constants**

Since $\frac{1}{2}\log(2\pi)$ doesn't depend on $\mu$ or $\sigma$, we can ignore it for optimization:

$$
-\log p(o | \mu, \sigma) \propto \frac{1}{2} \left[ \frac{(o - \mu)^2}{\sigma^2 + \sigma_{obs}^2} + \log(\sigma^2 + \sigma_{obs}^2) \right]
$$

**Why $\sigma^2 + \sigma_{obs}^2$?**

| Symbol         | Name               | Type              | Meaning                                        |
| -------------- | ------------------ | ----------------- | ---------------------------------------------- |
| $\sigma$       | Belief uncertainty | Dynamic (updated) | Agent's confidence about its position estimate |
| $\sigma_{obs}$ | Observation noise  | Fixed (parameter) | Agent's model of sensor noise                  |

The agent's belief is $q(x) = \mathcal{N}(\mu, \sigma^2)$ and the observation model is $p(o|x) = \mathcal{N}(x, \sigma_{obs}^2)$. Marginalizing over the uncertain state:

$$
p(o | \mu, \sigma) = \int p(o|x) \, q(x) \, dx = \mathcal{N}(\mu, \sigma^2 + \sigma_{obs}^2)
$$

The combined variance arises from integrating out the uncertain position.

**Precision**

Precision is the inverse of variance:

$$
\text{precision} = \frac{1}{\sigma^2}
$$

The **effective precision** combines both sources of uncertainty:

$$
\text{effective precision} = \frac{1}{\sigma^2 + \sigma_{obs}^2}
$$

In Active Inference, prediction errors are precision-weighted — errors matter more when the agent is confident (high precision), and less when uncertain (low precision).

**Two competing terms:**

| Term                                            | Effect on $\sigma$                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------- |
| $\frac{(o - \mu)^2}{\sigma^2 + \sigma_{obs}^2}$ | Large error + small $\sigma$ → big penalty (punishes overconfidence) |
| $\log(\sigma^2 + \sigma_{obs}^2)$               | Small $\sigma$ → smaller value (rewards confidence)                  |

This creates **adaptive uncertainty**: $\sigma$ shrinks when observations match predictions, and grows when the agent is surprised.

### Complexity Term

The complexity term is the KL divergence between the agent's belief and the prior (goal state).

**Derivation of KL divergence for Gaussians**

For $q(x) = \mathcal{N}(\mu, \sigma^2)$ and $p(x) = \mathcal{N}(\mu_{target}, \sigma_{prior}^2)$:

**Step 1: Definition of KL divergence**

$$
D_{KL}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)} dx = \mathbb{E}_q[\log q(x)] - \mathbb{E}_q[\log p(x)]
$$

**Step 2: Compute $\mathbb{E}_q[\log q(x)]$ (negative entropy)**

$$
\log q(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
$$

$$
\mathbb{E}_q[\log q(x)] = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2}
$$

(since $\mathbb{E}_q[(x-\mu)^2] = \sigma^2$)

**Step 3: Compute $\mathbb{E}_q[\log p(x)]$ (cross-entropy)**

$$
\log p(x) = -\frac{1}{2}\log(2\pi\sigma_{prior}^2) - \frac{(x-\mu_{target})^2}{2\sigma_{prior}^2}
$$

$$
\mathbb{E}_q[(x-\mu_{target})^2] = \mathbb{E}_q[(x-\mu+\mu-\mu_{target})^2] = \sigma^2 + (\mu - \mu_{target})^2
$$

$$
\mathbb{E}_q[\log p(x)] = -\frac{1}{2}\log(2\pi\sigma_{prior}^2) - \frac{\sigma^2 + (\mu - \mu_{target})^2}{2\sigma_{prior}^2}
$$

**Step 4: Subtract**

$$
D_{KL}(q \| p) = \mathbb{E}_q[\log q(x)] - \mathbb{E}_q[\log p(x)]
$$

$$
= \left( -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2} \right) - \left( -\frac{1}{2}\log(2\pi\sigma_{prior}^2) - \frac{\sigma^2 + (\mu - \mu_{target})^2}{2\sigma_{prior}^2} \right)
$$

$$
= \log\frac{\sigma_{prior}}{\sigma} + \frac{\sigma^2 + (\mu - \mu_{target})^2}{2\sigma_{prior}^2} - \frac{1}{2}
$$

**Three terms:**

| Term | Meaning |
|------|---------|
| $\log\frac{\sigma_{prior}}{\sigma}$ | Ratio of uncertainties — penalizes if belief is more uncertain than prior |
| $\frac{(\mu - \mu_{target})^2}{2\sigma_{prior}^2}$ | Distance from goal — penalizes beliefs far from target |
| $\frac{\sigma^2}{2\sigma_{prior}^2}$ | Uncertainty cost — penalizes high belief uncertainty |

**Intuition**

The complexity term pulls the agent's belief toward the goal state:
- $\mu \to \mu_{target}$ — "I should be at the target"
- $\sigma \to \sigma_{prior}$ — "My uncertainty should match my prior expectation"

Combined with the accuracy term, the agent balances:
- **Accuracy**: Match observations (sensory evidence)
- **Complexity**: Stay close to goals (prior preferences)

## References

- Friston, K. (2010). **The free-energy principle: a unified brain theory?** _Nature Reviews Neuroscience_, 11(2), 127-138. [[Nature]](https://www.nature.com/articles/nrn2787) [[PubMed]](https://pubmed.ncbi.nlm.nih.gov/20068583/)
