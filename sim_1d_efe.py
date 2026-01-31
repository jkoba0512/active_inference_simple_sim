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


# --- 5. Friction Learning (model parameter estimation) ---
@jax.jit
def compute_prediction_error(b_model, mu, mu_v, u, obs_next, obs_v_next, dt=0.01):
    """
    Prediction error for learning the friction parameter.
    Compares predicted next state to actual observation.
    """
    mu_pred, mu_v_pred = predict_next_state(mu, mu_v, u, b_model, dt)
    error = 0.5 * ((obs_next - mu_pred) ** 2 + 0.1 * (obs_v_next - mu_v_pred) ** 2)
    return error


grad_pred_error_b = jax.jit(jax.grad(compute_prediction_error, argnums=0))


# --- 6. Simulation Execution ---
def run_experiment():
    dt = 0.01
    steps = 5000
    target_x = 10.0

    # Learning rates
    lr_mu = 0.1  # For position belief
    lr_mu_v = 0.1  # For velocity belief
    lr_action = 10.0  # For action (EFE minimization)
    lr_b = 5000.0  # For friction learning (large because gradient is small)

    # Initialize states
    x, v = 0.0, 0.0  # True position and velocity
    mu, mu_v = 0.0, 0.0  # Believed position and velocity
    u = 0.0  # Control action

    # Precisions
    pi_obs = 10.0  # Trust in position observations
    pi_v = 1.0  # Trust in velocity observations
    pi_target = 1.0  # Precision of goal prior

    # Agent's model of friction (learnable!)
    b_model = 0.5

    # Store previous state for friction learning
    mu_prev, mu_v_prev, u_prev = 0.0, 0.0, 0.0

    history = {
        "x": [],
        "v": [],
        "mu": [],
        "mu_v": [],
        "u": [],
        "vfe": [],
        "efe": [],
        "b_model": [],
    }

    for i in range(steps):
        # TRUE friction (unknown to agent after step 1500)
        b_true = 0.5 if i < 1500 else 5.0

        # 1. Sensory Observations
        key = jax.random.PRNGKey(i)
        noise = jax.random.normal(key, (2,)) * 0.02
        obs = x + noise[0]
        obs_v = v + noise[1]

        # 2. Friction Learning: Update b_model based on prediction error
        # Compare what we predicted (from previous step) to what we observe now
        if i > 0:
            dE_db = grad_pred_error_b(
                b_model, mu_prev, mu_v_prev, u_prev, obs, obs_v, dt
            )
            b_model = b_model - lr_b * dE_db
            # Clamp to reasonable range
            b_model = jnp.clip(b_model, 0.01, 20.0)

        # 3. Perception: Update beliefs (minimize VFE)
        dF_dmu = grad_vfe_mu(mu, mu_v, obs, obs_v, pi_obs, pi_v)
        dF_dmu_v = grad_vfe_mu_v(mu, mu_v, obs, obs_v, pi_obs, pi_v)

        mu = mu - lr_mu * dF_dmu
        mu_v = mu_v - lr_mu_v * dF_dmu_v

        # 4. Action Selection: Minimize Expected Free Energy
        dG_du = grad_efe_u(mu, mu_v, u, target_x, b_model, pi_target, dt)
        u = u - lr_action * dG_du

        # Clamp action
        u = jnp.clip(u, -20.0, 20.0)

        # Store current state for next iteration's friction learning
        mu_prev, mu_v_prev, u_prev = mu, mu_v, u

        # 5. Environment Step (true physics)
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
        history["b_model"].append(float(b_model))

    return history


# --- 7. Visualization ---
def plot_results(h):
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

    # Plot 1: Position
    ax1 = axes[0]
    ax1.plot(h["x"], label="Actual Position (x)", color="royalblue", lw=2)
    ax1.plot(h["mu"], "--", label="Belief (μ)", color="orange")
    ax1.axhline(10, color="red", linestyle=":", label="Target")
    ax1.axvline(1500, color="black", alpha=0.2, linestyle="--")
    ax1.text(510, 2, "Friction Change", color="black", alpha=0.6)
    ax1.set_ylabel("Position")
    ax1.legend(loc="lower right")
    ax1.set_title("Active Inference with Expected Free Energy + Friction Learning")

    # Plot 2: Action
    ax2 = axes[1]
    ax2.plot(h["u"], label="Action (u)", color="green")
    ax2.axvline(1500, color="black", alpha=0.2, linestyle="--")
    ax2.set_ylabel("Action")
    ax2.legend(loc="lower right")

    # Plot 3: Friction model (learned)
    ax3 = axes[2]
    ax3.plot(h["b_model"], label="Learned friction (b_model)", color="teal", lw=2)
    ax3.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="True b (before)")
    ax3.axhline(5.0, color="gray", linestyle="--", alpha=0.5, label="True b (after)")
    ax3.axvline(1500, color="black", alpha=0.2, linestyle="--")
    ax3.set_ylabel("Friction (b)")
    ax3.legend(loc="right")

    # Plot 4: VFE (perception)
    ax4 = axes[3]
    ax4.plot(h["vfe"], label="VFE (perception)", color="purple")
    ax4.axvline(1500, color="black", alpha=0.2, linestyle="--")
    ax4.set_ylabel("VFE")
    ax4.legend(loc="upper right")

    # Plot 5: EFE (action selection)
    ax5 = axes[4]
    ax5.plot(h["efe"], label="EFE (action selection)", color="coral")
    ax5.axvline(1500, color="black", alpha=0.2, linestyle="--")
    ax5.set_ylabel("EFE")
    ax5.set_xlabel("Time Steps")
    ax5.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)
