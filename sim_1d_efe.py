import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# --- 1. Generative Process (Physical Environment) ---
@jax.jit
def update_physics(x, v, u, b_current, dt=0.01):
    """
    Simulates the actual physical world (1D point mass).
    """
    m = 1.0
    k = 0.1
    a = (u - b_current * v - k * x) / m
    new_v = v + a * dt
    new_x = x + new_v * dt
    return new_x, new_v


# --- 2. Agent's Generative Model ---
@jax.jit
def predict_next_state(mu, mu_v, u, b_model, dt=0.01):
    """
    Agent's internal model predicting next state given current belief and action.
    """
    m = 1.0
    k = 0.1
    a = (u - b_model * mu_v - k * mu) / m
    new_mu_v = mu_v + a * dt
    new_mu = mu + new_mu_v * dt
    return new_mu, new_mu_v


# --- 3. Variational Free Energy (for perception) ---
@jax.jit
def compute_vfe(mu, mu_v, observation, obs_v, pi_obs, pi_v):
    """
    VFE for state estimation (perception).
    Precision-weighted prediction errors for position and velocity.
    """
    error_pos = observation - mu
    error_vel = obs_v - mu_v
    vfe = 0.5 * (pi_obs * error_pos**2 + pi_v * error_vel**2)
    return vfe


grad_vfe_mu = jax.jit(jax.grad(compute_vfe, argnums=0))
grad_vfe_mu_v = jax.jit(jax.grad(compute_vfe, argnums=1))


# --- 4. Expected Free Energy (for action selection) ---
@jax.jit
def compute_efe(mu, mu_v, u, target_x, b_model, pi_target, dt=0.01):
    """
    Expected Free Energy for action selection.

    G(u) = pragmatic_value + epistemic_value

    Pragmatic: Expected distance from preferred outcome (goal-seeking)
    Epistemic: Expected uncertainty (simplified as constant here)
    """
    # Predict next state under action u
    mu_next, mu_v_next = predict_next_state(mu, mu_v, u, b_model, dt)

    # --- Pragmatic Value ---
    # How far will I be from my goal? (precision-weighted)
    pragmatic = 0.5 * pi_target * (mu_next - target_x) ** 2

    # --- Epistemic Value ---
    # Velocity should also settle (reach equilibrium near target)
    epistemic = 0.5 * 0.1 * mu_v_next**2

    efe = pragmatic + epistemic
    return efe


grad_efe_u = jax.jit(jax.grad(compute_efe, argnums=2))


# --- 5. Simulation Execution ---
def run_experiment():
    dt = 0.01
    steps = 1500
    target_x = 10.0

    # Learning rates
    lr_mu = 0.1  # For position belief
    lr_mu_v = 0.1  # For velocity belief
    lr_action = 1.0  # For action (EFE minimization)

    # Initialize states
    x, v = 0.0, 0.0  # True position and velocity
    mu, mu_v = 0.0, 0.0  # Believed position and velocity
    u = 0.0  # Control action

    # Precisions
    pi_obs = 10.0  # Trust in position observations
    pi_v = 1.0  # Trust in velocity observations
    pi_target = 1.0  # Precision of goal prior

    # Agent's model of friction (may differ from reality!)
    b_model = 0.5

    history = {
        "x": [],
        "v": [],
        "mu": [],
        "mu_v": [],
        "u": [],
        "vfe": [],
        "efe": [],
    }

    for i in range(steps):
        # TRUE friction (unknown to agent after step 500)
        b_true = 0.5 if i < 500 else 5.0

        # 1. Sensory Observations
        key = jax.random.PRNGKey(i)
        noise = jax.random.normal(key, (2,)) * 0.02
        obs = x + noise[0]
        obs_v = v + noise[1]

        # 2. Perception: Update beliefs (minimize VFE)
        dF_dmu = grad_vfe_mu(mu, mu_v, obs, obs_v, pi_obs, pi_v)
        dF_dmu_v = grad_vfe_mu_v(mu, mu_v, obs, obs_v, pi_obs, pi_v)

        mu = mu - lr_mu * dF_dmu
        mu_v = mu_v - lr_mu_v * dF_dmu_v

        # 3. Action Selection: Minimize Expected Free Energy
        dG_du = grad_efe_u(mu, mu_v, u, target_x, b_model, pi_target, dt)
        u = u - lr_action * dG_du

        # Clamp action
        u = jnp.clip(u, -20.0, 20.0)

        # 4. Environment Step (true physics)
        x, v = update_physics(x, v, u, b_true, dt)

        # Log
        vfe = compute_vfe(mu, mu_v, obs, obs_v, pi_obs, pi_v)
        efe = compute_efe(mu, mu_v, u, target_x, b_model, pi_target, dt)

        history["x"].append(float(x))
        history["v"].append(float(v))
        history["mu"].append(float(mu))
        history["mu_v"].append(float(mu_v))
        history["u"].append(float(u))
        history["vfe"].append(float(vfe))
        history["efe"].append(float(efe))

    return history


# --- 6. Visualization ---
def plot_results(h):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Position
    ax1 = axes[0]
    ax1.plot(h["x"], label="Actual Position (x)", color="royalblue", lw=2)
    ax1.plot(h["mu"], "--", label="Belief (μ)", color="orange")
    ax1.axhline(10, color="red", linestyle=":", label="Target")
    ax1.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax1.text(510, 2, "Friction Change (agent unaware)", color="black", alpha=0.6)
    ax1.set_ylabel("Position")
    ax1.legend(loc="lower right")
    ax1.set_title("Active Inference with Expected Free Energy")

    # Plot 2: Action
    ax2 = axes[1]
    ax2.plot(h["u"], label="Action (u)", color="green")
    ax2.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax2.set_ylabel("Action")
    ax2.legend(loc="lower right")

    # Plot 3: VFE (perception)
    ax3 = axes[2]
    ax3.plot(h["vfe"], label="VFE (perception)", color="purple")
    ax3.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax3.set_ylabel("VFE")
    ax3.legend(loc="upper right")

    # Plot 4: EFE (action selection)
    ax4 = axes[3]
    ax4.plot(h["efe"], label="EFE (action selection)", color="coral")
    ax4.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax4.set_ylabel("EFE")
    ax4.set_xlabel("Time Steps")
    ax4.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)
