import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse


# --- 1. Generative Process (Physical Environment) ---
@jax.jit
def update_physics(pos, vel, u, b_current, dt=0.01):
    """
    Simulates the actual physical world (2D point mass).
    pos: (x, y) position
    vel: (vx, vy) velocity
    u: (ux, uy) control force
    m: mass, b: friction (damping), k: natural stiffness.
    """
    m = 1.0
    k = 0.1  # Minimal natural stiffness
    acc = (u - b_current * vel - k * pos) / m
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel


# --- 2. Generative Model (Probabilistic Active Inference) ---
@jax.jit
def compute_vfe_prob(
    mu,
    log_sigma,
    action,
    observation,
    target,
    sigma_obs,
    sigma_prior,
    p_action=0.1,
    action_gain=0.5,
):
    """
    Computes Variational Free Energy with probabilistic beliefs for 2D.

    The agent maintains a Gaussian belief: q(x) = N(mu, sigma²I)
    where sigma = exp(log_sigma) ensures positivity.
    (Isotropic covariance for simplicity)

    VFE = Accuracy (prediction errors) + Complexity (KL divergence)
    """
    sigma = jnp.exp(log_sigma)  # Ensure sigma > 0

    # --- Accuracy Term ---
    # -log p(observation | belief) for Gaussian likelihood
    error_sensory = observation - mu
    effective_precision = 1.0 / (sigma**2 + sigma_obs**2)
    accuracy = 0.5 * (
        jnp.sum(error_sensory**2) * effective_precision
        + 2 * jnp.log(sigma**2 + sigma_obs**2)  # 2D: 2 dimensions
    )

    # --- Complexity Term ---
    # KL divergence for 2D isotropic Gaussians
    # KL(N(mu, sigma²I) || N(target, sigma_prior²I))
    error_prior = mu - target
    kl_divergence = (
        2 * jnp.log(sigma_prior / sigma)  # 2D: 2 * log ratio
        + (2 * sigma**2 + jnp.sum(error_prior**2)) / (sigma_prior**2)
        - 2  # 2D: subtract 2
    ) * 0.5

    # --- Action Model ---
    action_target = action_gain * (target - mu)
    error_action = action - action_target
    action_cost = 0.5 * p_action * jnp.sum(error_action**2)

    # Total VFE
    vfe = accuracy + kl_divergence + action_cost
    return vfe


# Define gradients
grad_vfe_mu = jax.jit(jax.grad(compute_vfe_prob, argnums=0))
grad_vfe_log_sigma = jax.jit(jax.grad(compute_vfe_prob, argnums=1))
grad_vfe_u = jax.jit(jax.grad(compute_vfe_prob, argnums=2))


# --- 3. Simulation Execution ---
def run_experiment():
    dt = 0.01
    steps = 1500
    target = jnp.array([10.0, 7.0])  # 2D target position

    # Learning rates
    lr_mu = 0.2
    lr_sigma = 0.01
    lr_action = 0.2

    # Initialize states
    pos = jnp.array([0.0, 0.0])
    vel = jnp.array([0.0, 0.0])
    mu = jnp.array([0.0, 0.0])
    log_sigma = jnp.log(2.0)  # Start with uncertainty
    u = jnp.array([0.0, 0.0])

    # Generative model parameters
    sigma_obs = 0.1
    sigma_prior = 5.0

    history = {
        "pos_x": [],
        "pos_y": [],
        "mu_x": [],
        "mu_y": [],
        "sigma": [],
        "u_x": [],
        "u_y": [],
        "vfe": [],
    }

    for i in range(steps):
        b_current = 0.5 if i < 500 else 5.0

        # 1. Sensory Observation
        key = jax.random.PRNGKey(i)
        noise = jax.random.normal(key, (2,)) * 0.02
        obs = pos + noise

        # 2. Perception: Update belief mean and uncertainty
        dF_dmu = grad_vfe_mu(mu, log_sigma, u, obs, target, sigma_obs, sigma_prior)
        dF_dlog_sigma = grad_vfe_log_sigma(
            mu, log_sigma, u, obs, target, sigma_obs, sigma_prior
        )

        mu = mu - lr_mu * dF_dmu
        log_sigma = log_sigma - lr_sigma * dF_dlog_sigma
        log_sigma = jnp.clip(log_sigma, -3.0, 3.0)

        # 3. Action
        dF_du = grad_vfe_u(mu, log_sigma, u, obs, target, sigma_obs, sigma_prior)
        u = u - lr_action * dF_du

        # 4. Environment Step
        pos, vel = update_physics(pos, vel, u, b_current, dt)

        # Log
        vfe = compute_vfe_prob(mu, log_sigma, u, obs, target, sigma_obs, sigma_prior)
        sigma = jnp.exp(log_sigma)

        history["pos_x"].append(float(pos[0]))
        history["pos_y"].append(float(pos[1]))
        history["mu_x"].append(float(mu[0]))
        history["mu_y"].append(float(mu[1]))
        history["sigma"].append(float(sigma))
        history["u_x"].append(float(u[0]))
        history["u_y"].append(float(u[1]))
        history["vfe"].append(float(vfe))

    return history, target


# --- 4. Visualization ---
def plot_results(h, target):
    fig = plt.figure(figsize=(14, 12))

    # Plot 1: 2D Trajectory with uncertainty ellipses
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(h["pos_x"], h["pos_y"], label="Actual Path", color="royalblue", lw=2)
    ax1.plot(
        h["mu_x"], h["mu_y"], "--", label="Believed Path", color="orange", alpha=0.7
    )

    # Draw uncertainty ellipses at selected points
    for i in [0, 250, 500, 750, 1000, 1250, 1499]:
        ellipse = Ellipse(
            (h["mu_x"][i], h["mu_y"][i]),
            width=2 * h["sigma"][i] * 2,  # 2-sigma
            height=2 * h["sigma"][i] * 2,
            alpha=0.2,
            color="orange",
        )
        ax1.add_patch(ellipse)

    ax1.scatter([0], [0], color="green", s=100, marker="o", label="Start", zorder=5)
    ax1.scatter(
        [target[0]],
        [target[1]],
        color="red",
        s=100,
        marker="*",
        label="Target",
        zorder=5,
    )
    ax1.scatter(
        [h["pos_x"][500]],
        [h["pos_y"][500]],
        color="black",
        s=80,
        marker="x",
        label="Friction Change",
        zorder=5,
    )
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.legend(loc="lower right")
    ax1.set_title("Probabilistic Active Inference: 2D Trajectory")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Uncertainty over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(h["sigma"], label="Belief Uncertainty (σ)", color="coral", lw=2)
    ax2.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax2.text(510, max(h["sigma"]) * 0.9, "Friction Change", color="black", alpha=0.6)
    ax2.set_ylabel("Uncertainty (σ)")
    ax2.set_xlabel("Time Steps")
    ax2.legend(loc="upper right")
    ax2.set_title("Belief Uncertainty Over Time")

    # Plot 3: Position components
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(h["pos_x"], label="X Position", color="royalblue", lw=2)
    ax3.plot(h["pos_y"], label="Y Position", color="forestgreen", lw=2)
    ax3.axhline(target[0], color="royalblue", linestyle=":", alpha=0.5)
    ax3.axhline(target[1], color="forestgreen", linestyle=":", alpha=0.5)
    ax3.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax3.set_ylabel("Position")
    ax3.set_xlabel("Time Steps")
    ax3.legend(loc="lower right")
    ax3.set_title("Position Components Over Time")

    # Plot 4: VFE
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(h["vfe"], label="VFE", color="purple", lw=2)
    ax4.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax4.set_yscale("log")
    ax4.set_ylabel("VFE (Log Scale)")
    ax4.set_xlabel("Time Steps")
    ax4.legend(loc="upper right")
    ax4.set_title("Variational Free Energy")

    plt.tight_layout()
    plt.show()


# --- 5. Animation ---
def animate_results(h, target, interval=10, skip=5, save_path=None):
    """
    Animate the 2D trajectory with uncertainty visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-2, 14)
    ax.set_ylim(-2, 10)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Probabilistic Active Inference: 2D Simulation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # Static elements
    ax.scatter([0], [0], color="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter(
        [target[0]],
        [target[1]],
        color="red",
        s=150,
        marker="*",
        label="Target",
        zorder=5,
    )

    # Dynamic elements
    (trajectory_line,) = ax.plot([], [], color="royalblue", lw=2, alpha=0.5)
    (agent_dot,) = ax.plot([], [], "o", color="royalblue", markersize=12, label="Agent")
    uncertainty_ellipse = Ellipse(
        (0, 0), width=1, height=1, alpha=0.3, color="orange", label="Uncertainty (2σ)"
    )
    ax.add_patch(uncertainty_ellipse)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    sigma_text = ax.text(0.02, 0.93, "", transform=ax.transAxes, va="top", fontsize=10)
    friction_text = ax.text(
        0.02, 0.88, "", transform=ax.transAxes, va="top", fontsize=10
    )

    ax.legend(loc="lower right")

    frames = range(0, len(h["pos_x"]), skip)

    def init():
        trajectory_line.set_data([], [])
        agent_dot.set_data([], [])
        uncertainty_ellipse.set_center((0, 0))
        step_text.set_text("")
        sigma_text.set_text("")
        friction_text.set_text("")
        return (
            trajectory_line,
            agent_dot,
            uncertainty_ellipse,
            step_text,
            sigma_text,
            friction_text,
        )

    def update(frame):
        trajectory_line.set_data(h["pos_x"][:frame], h["pos_y"][:frame])
        agent_dot.set_data([h["pos_x"][frame]], [h["pos_y"][frame]])

        # Update uncertainty ellipse
        uncertainty_ellipse.set_center((h["mu_x"][frame], h["mu_y"][frame]))
        sigma_2 = h["sigma"][frame] * 2 * 2  # 2-sigma diameter
        uncertainty_ellipse.set_width(sigma_2)
        uncertainty_ellipse.set_height(sigma_2)

        step_text.set_text(f"Step: {frame}")
        sigma_text.set_text(f"Uncertainty (σ): {h['sigma'][frame]:.2f}")
        friction = "b=0.5" if frame < 500 else "b=5.0 (increased!)"
        friction_text.set_text(f"Friction: {friction}")

        return (
            trajectory_line,
            agent_dot,
            uncertainty_ellipse,
            step_text,
            sigma_text,
            friction_text,
        )

    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=interval
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="ffmpeg", fps=30)
        print("Done!")
    else:
        plt.show()

    return anim


if __name__ == "__main__":
    results, target = run_experiment()
    plot_results(results, target)
    animate_results(results, target, save_path="active_inference_2d_prob.mp4")
