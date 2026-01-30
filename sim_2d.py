import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


# --- 2. Generative Model (Active Inference Core) ---
@jax.jit
def compute_vfe(
    mu, action, observation, target, p_obs, p_prior, p_action=0.1, action_gain=0.5
):
    """
    Computes Variational Free Energy (VFE) for 2D case.
    All inputs are 2D vectors: mu, action, observation, target.
    Minimizing this leads to both accurate perception (inference)
    and goal-directed behavior (action).
    """
    # Prediction Error 1: Sensory Error (Observation vs. Internal Belief)
    error_sensory = observation - mu

    # Prediction Error 2: Prior Error (Internal Belief vs. Desired Target)
    # This represents the "stubborn" belief that "I should be at the target."
    error_prior = target - mu

    # Prediction Error 3: Action Model Error
    # Agent's model: action should be proportional to goal-directed error.
    # This couples action to the generative model, enabling action learning.
    action_target = action_gain * (target - mu)
    error_action = action - action_target

    # VFE = Sum of precision-weighted squared errors (summed over x,y components)
    vfe = 0.5 * (
        p_obs * jnp.sum(error_sensory**2)
        + p_prior * jnp.sum(error_prior**2)
        + p_action * jnp.sum(error_action**2)
    )
    return vfe


# Define gradients for Inference (updating mu) and Action (updating u)
grad_vfe_mu = jax.jit(jax.grad(compute_vfe, argnums=0))
grad_vfe_u = jax.jit(jax.grad(compute_vfe, argnums=1))


# --- 3. Simulation Execution ---
def run_experiment():
    dt = 0.01
    steps = 1500
    target = jnp.array([10.0, 7.0])  # 2D target position
    learning_rate = 0.2  # Speed of adaptation (Gradient descent step size)

    # Initialize states (all 2D vectors)
    pos = jnp.array([0.0, 0.0])  # Actual position
    vel = jnp.array([0.0, 0.0])  # Actual velocity
    mu = jnp.array([0.0, 0.0])  # Internal belief about position
    u = jnp.array([0.0, 0.0])  # Control action (force)

    # Precisions: p_obs (trust in sensors), p_prior (strength of desire for target)
    # Note: learning_rate * p_obs should be < 1 for stability
    p_obs, p_prior = 2.0, 1.0

    history = {
        "pos_x": [],
        "pos_y": [],
        "mu_x": [],
        "mu_y": [],
        "u_x": [],
        "u_y": [],
        "vfe": [],
    }

    for i in range(steps):
        # TEST POINT: Suddenly increase friction at step 500
        # This simulates an unexpected environmental change (e.g., moving onto a sticky surface).
        b_current = 0.5 if i < 500 else 5.0

        # 1. Sensory Observation (with small Gaussian noise)
        key = jax.random.PRNGKey(i)
        noise = jax.random.normal(key, (2,)) * 0.02
        obs = pos + noise

        # 2. Perception (Inference): Update mu to minimize VFE
        dF_dmu = grad_vfe_mu(mu, u, obs, target, p_obs, p_prior)
        mu = mu - learning_rate * dF_dmu

        # 3. Action: Update control input u to minimize VFE
        # Action changes the world to make observations match the desired prior.
        dF_du = grad_vfe_u(mu, u, obs, target, p_obs, p_prior)
        u = u - learning_rate * dF_du

        # 4. Environment Step
        pos, vel = update_physics(pos, vel, u, b_current, dt)

        # Log VFE (Surprise/Accuracy proxy)
        vfe = compute_vfe(mu, u, obs, target, p_obs, p_prior)

        history["pos_x"].append(float(pos[0]))
        history["pos_y"].append(float(pos[1]))
        history["mu_x"].append(float(mu[0]))
        history["mu_y"].append(float(mu[1]))
        history["u_x"].append(float(u[0]))
        history["u_y"].append(float(u[1]))
        history["vfe"].append(float(vfe))

    return history, target


# --- 4. Visualization ---
def plot_results(h, target):
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: 2D Trajectory (top-left, larger)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(h["pos_x"], h["pos_y"], label="Actual Path", color="royalblue", lw=2)
    ax1.plot(
        h["mu_x"], h["mu_y"], "--", label="Believed Path", color="orange", alpha=0.7
    )
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
    # Mark friction change point
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
    ax1.set_title("Active Inference: 2D Trajectory")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)

    # Plot 2: X and Y positions over time (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(h["pos_x"], label="X Position", color="royalblue", lw=2)
    ax2.plot(h["pos_y"], label="Y Position", color="forestgreen", lw=2)
    ax2.axhline(target[0], color="royalblue", linestyle=":", alpha=0.5)
    ax2.axhline(target[1], color="forestgreen", linestyle=":", alpha=0.5)
    ax2.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax2.text(510, 2, "Friction Change", color="black", alpha=0.6)
    ax2.set_ylabel("Position")
    ax2.legend(loc="lower right")
    ax2.set_title("Position Components Over Time")

    # Plot 3: Action (Control Effort)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(h["u_x"], label="Force X", color="coral", lw=1.5)
    ax3.plot(h["u_y"], label="Force Y", color="teal", lw=1.5)
    ax3.axvline(500, color="black", alpha=0.2, linestyle="--")
    ax3.set_ylabel("Action (Force)")
    ax3.set_xlabel("Time Steps")
    ax3.legend(loc="lower right")
    ax3.set_title("Control Actions")

    # Plot 4: Variational Free Energy
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(h["vfe"], label="VFE (Surprise)", color="purple")
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
    Animate the 2D trajectory.
    interval: milliseconds between frames
    skip: show every nth frame (to speed up animation)
    save_path: if provided, save animation to this path (e.g., "animation.mp4")
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up the plot
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 9)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Active Inference: 2D Simulation")
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
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    friction_text = ax.text(
        0.02, 0.93, "", transform=ax.transAxes, va="top", fontsize=10
    )

    ax.legend(loc="lower right")

    # Subsample frames
    frames = range(0, len(h["pos_x"]), skip)

    def init():
        trajectory_line.set_data([], [])
        agent_dot.set_data([], [])
        step_text.set_text("")
        friction_text.set_text("")
        return trajectory_line, agent_dot, step_text, friction_text

    def update(frame):
        # Trail
        trajectory_line.set_data(h["pos_x"][:frame], h["pos_y"][:frame])
        # Current position
        agent_dot.set_data([h["pos_x"][frame]], [h["pos_y"][frame]])
        # Step counter
        step_text.set_text(f"Step: {frame}")
        # Friction indicator
        friction = "b=0.5" if frame < 500 else "b=5.0 (increased!)"
        friction_text.set_text(f"Friction: {friction}")
        return trajectory_line, agent_dot, step_text, friction_text

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
    plot_results(results, target)  # Static plot
    animate_results(results, target, save_path="active_inference_2d.mp4")  # Animation
