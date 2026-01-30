import jax
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


# --- 2. Generative Model (Active Inference Core) ---
@jax.jit
def compute_vfe(
    mu, action, observation, target_x, p_obs, p_prior, p_action=0.1, action_gain=0.5
):
    """
    Computes Variational Free Energy (VFE).
    Minimizing this leads to both accurate perception (inference)
    and goal-directed behavior (action).
    """
    # Prediction Error 1: Sensory Error (Observation vs. Internal Belief)
    error_sensory = observation - mu

    # Prediction Error 2: Prior Error (Internal Belief vs. Desired Target)
    # This represents the "stubborn" belief that "I should be at the target."
    error_prior = target_x - mu

    # Prediction Error 3: Action Model Error
    # Agent's model: action should be proportional to goal-directed error.
    # This couples action to the generative model, enabling action learning.
    action_target = action_gain * (target_x - mu)
    error_action = action - action_target

    # VFE = Sum of precision-weighted squared errors
    vfe = 0.5 * (
        p_obs * error_sensory**2 + p_prior * error_prior**2 + p_action * error_action**2
    )
    return vfe


# Define gradients for Inference (updating mu) and Action (updating u)
grad_vfe_mu = jax.jit(jax.grad(compute_vfe, argnums=0))
grad_vfe_u = jax.jit(jax.grad(compute_vfe, argnums=1))


# --- 3. Simulation Execution ---
def run_experiment():
    dt = 0.01
    steps = 1500
    target_x = 10.0
    learning_rate = 0.2  # Speed of adaptation (Gradient descent step size)

    # Initialize states
    x, v, mu, u = 0.0, 0.0, 0.0, 0.0

    # Precisions: p_obs (trust in sensors), p_prior (strength of desire for target)
    # Note: learning_rate * p_obs should be < 1 for stability
    p_obs, p_prior = 2.0, 1.0

    history = {"x": [], "mu": [], "u": [], "vfe": []}

    for i in range(steps):
        # TEST POINT: Suddenly increase friction at step 500
        # This simulates an unexpected environmental change (e.g., moving onto a sticky surface).
        b_current = 0.5 if i < 500 else 5.0

        # 1. Sensory Observation (with small Gaussian noise)
        key = jax.random.PRNGKey(i)
        obs = x + jax.random.normal(key, ()) * 0.02

        # 2. Perception (Inference): Update mu to minimize VFE
        dF_dmu = grad_vfe_mu(mu, u, obs, target_x, p_obs, p_prior)
        mu -= learning_rate * dF_dmu

        # 3. Action: Update control input u to minimize VFE
        # Action changes the world to make observations match the desired prior.
        dF_du = grad_vfe_u(mu, u, obs, target_x, p_obs, p_prior)
        u -= learning_rate * dF_du

        # 4. Environment Step
        x, v = update_physics(x, v, u, b_current, dt)

        # Log VFE (Surprise/Accuracy proxy)
        vfe = compute_vfe(mu, u, obs, target_x, p_obs, p_prior)

        history["x"].append(x)
        history["mu"].append(mu)
        history["u"].append(u)
        history["vfe"].append(vfe)

    return history


# --- 4. Visualization ---
def plot_results(h):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Trajectory
    ax1.plot(h["x"], label="Actual Position (x)", color="royalblue", lw=2)
    ax1.plot(h["mu"], "--", label="Internal Belief (mu)", color="orange")
    ax1.axhline(10, color="red", linestyle=":", label="Target")
    ax1.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax1.text(510, 2, "Friction Change (b=0.5 -> 5.0)", color="black", alpha=0.6)
    ax1.set_ylabel("Position")
    ax1.legend(loc="lower right")
    ax1.set_title("Active Inference: 1D Robustness Simulation")

    # Plot 2: Action (Control Effort)
    ax2.plot(h["u"], label="Control Input (Force)", color="green")
    ax2.set_ylabel("Action (u)")
    ax2.legend(loc="lower right")

    # Plot 3: Variational Free Energy
    ax3.plot(h["vfe"], label="VFE (Surprise)", color="purple")
    ax3.set_yscale("log")
    ax3.set_ylabel("VFE (Log Scale)")
    ax3.set_xlabel("Time Steps")
    ax3.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_experiment()
    plot_results(results)
