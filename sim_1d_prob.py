import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# --- 1. Generative Process (Physical Environment) ---
@jax.jit
def update_physics(x, v, u, b_current, dt=0.01):
    """
    Simulates the actual physical world (1D point mass).
    m: mass, b: friction (damping), k: natural stiffness.
    """
    m = 1.0
    k = 0.1  # Minimal natural stiffness
    a = (u - b_current * v - k * x) / m
    new_v = v + a * dt
    new_x = x + new_v * dt
    return new_x, new_v


# --- 2. Generative Model (Probabilistic Active Inference) ---
@jax.jit
def compute_vfe_prob(
    mu,
    log_sigma,
    action,
    observation,
    target_x,
    sigma_obs,
    sigma_prior,
    p_action=0.1,
    action_gain=0.5,
):
    """
    Computes Variational Free Energy with probabilistic beliefs.

    The agent maintains a Gaussian belief: q(x) = N(mu, sigma²)
    where sigma = exp(log_sigma) ensures positivity.

    VFE = Accuracy (prediction errors) + Complexity (KL divergence)

    For Gaussian distributions:
    - Accuracy: How well observations match beliefs
    - Complexity: How far beliefs deviate from prior
    """
    sigma = jnp.exp(log_sigma)  # Ensure sigma > 0

    # --- Accuracy Term ---
    # -log p(observation | belief) for Gaussian likelihood
    # The agent's belief uncertainty (sigma) affects how it weights observations
    # Higher sigma = less confident = observations weighted more
    error_sensory = observation - mu
    effective_precision = 1.0 / (sigma**2 + sigma_obs**2)
    accuracy = 0.5 * (
        error_sensory**2 * effective_precision + jnp.log(sigma**2 + sigma_obs**2)
    )

    # --- Complexity Term ---
    # KL divergence: KL(q(x) || p(x)) where q = N(mu, sigma²), p = N(target, sigma_prior²)
    # KL = log(sigma_prior/sigma) + (sigma² + (mu - target)²) / (2*sigma_prior²) - 0.5
    error_prior = mu - target_x
    kl_divergence = (
        jnp.log(sigma_prior / sigma)
        + (sigma**2 + error_prior**2) / (2 * sigma_prior**2)
        - 0.5
    )

    # --- Action Model ---
    # Agent's model: action should be proportional to goal-directed error
    action_target = action_gain * (target_x - mu)
    error_action = action - action_target
    action_cost = 0.5 * p_action * error_action**2

    # Total VFE
    vfe = accuracy + kl_divergence + action_cost
    return vfe


# Define gradients for belief mean, belief uncertainty, and action
grad_vfe_mu = jax.jit(jax.grad(compute_vfe_prob, argnums=0))
grad_vfe_log_sigma = jax.jit(jax.grad(compute_vfe_prob, argnums=1))
grad_vfe_u = jax.jit(jax.grad(compute_vfe_prob, argnums=2))


# --- 3. Simulation Execution ---
def run_experiment():
    dt = 0.01
    steps = 1500
    target_x = 10.0

    # Learning rates for different variables
    lr_mu = 0.2  # Learning rate for belief mean
    lr_sigma = 0.01  # Learning rate for belief uncertainty (slower)
    lr_action = 0.2  # Learning rate for action

    # Initialize states
    x, v = 0.0, 0.0  # True position and velocity
    mu = 0.0  # Belief mean
    log_sigma = jnp.log(2.0)  # Belief std dev (start with high uncertainty)
    u = 0.0  # Control action

    # Generative model parameters
    sigma_obs = 0.1  # Observation noise (agent's model of sensor noise)
    sigma_prior = 5.0  # Prior uncertainty about target (wide prior)

    history = {
        "x": [],
        "mu": [],
        "sigma": [],
        "u": [],
        "vfe": [],
    }

    for i in range(steps):
        # Friction change at step 500
        b_current = 0.5 if i < 500 else 5.0

        # 1. Sensory Observation (with Gaussian noise)
        key = jax.random.PRNGKey(i)
        obs = x + jax.random.normal(key, ()) * 0.02

        # 2. Perception: Update belief mean (mu) and uncertainty (sigma)
        dF_dmu = grad_vfe_mu(mu, log_sigma, u, obs, target_x, sigma_obs, sigma_prior)
        dF_dlog_sigma = grad_vfe_log_sigma(
            mu, log_sigma, u, obs, target_x, sigma_obs, sigma_prior
        )

        mu = mu - lr_mu * dF_dmu
        log_sigma = log_sigma - lr_sigma * dF_dlog_sigma

        # Clamp log_sigma to prevent numerical issues
        log_sigma = jnp.clip(log_sigma, -3.0, 3.0)

        # 3. Action: Update control input
        dF_du = grad_vfe_u(mu, log_sigma, u, obs, target_x, sigma_obs, sigma_prior)
        u = u - lr_action * dF_du

        # 4. Environment Step
        x, v = update_physics(x, v, u, b_current, dt)

        # Log
        vfe = compute_vfe_prob(mu, log_sigma, u, obs, target_x, sigma_obs, sigma_prior)
        sigma = jnp.exp(log_sigma)

        history["x"].append(float(x))
        history["mu"].append(float(mu))
        history["sigma"].append(float(sigma))
        history["u"].append(float(u))
        history["vfe"].append(float(vfe))

    return history


# --- 4. Visualization ---
def plot_results(h):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    steps = range(len(h["x"]))
    mu = jnp.array(h["mu"])
    sigma = jnp.array(h["sigma"])

    # Plot 1: Position with uncertainty band
    ax1 = axes[0]
    ax1.plot(h["x"], label="Actual Position (x)", color="royalblue", lw=2)
    ax1.plot(h["mu"], "--", label="Belief Mean (μ)", color="orange")
    ax1.fill_between(
        steps,
        mu - 2 * sigma,
        mu + 2 * sigma,
        alpha=0.2,
        color="orange",
        label="Belief ±2σ",
    )
    ax1.axhline(10, color="red", linestyle=":", label="Target")
    ax1.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax1.text(510, 2, "Friction Change", color="black", alpha=0.6)
    ax1.set_ylabel("Position")
    ax1.legend(loc="lower right")
    ax1.set_title("Probabilistic Active Inference: 1D Simulation")

    # Plot 2: Belief Uncertainty (sigma)
    ax2 = axes[1]
    ax2.plot(h["sigma"], label="Belief Uncertainty (σ)", color="coral")
    ax2.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax2.set_ylabel("Uncertainty (σ)")
    ax2.legend(loc="upper right")

    # Plot 3: Action
    ax3 = axes[2]
    ax3.plot(h["u"], label="Control Input (Force)", color="green")
    ax3.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax3.set_ylabel("Action (u)")
    ax3.legend(loc="lower right")

    # Plot 4: VFE
    ax4 = axes[3]
    ax4.plot(h["vfe"], label="VFE", color="purple")
    ax4.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax4.set_yscale("log")
    ax4.set_ylabel("VFE (Log Scale)")
    ax4.set_xlabel("Time Steps")
    ax4.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)
