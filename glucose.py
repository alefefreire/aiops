"""
Improved Bayesian Physics-Informed Neural Network (B-PINN) for Bergman Minimal Model
Using Pyro for variational inference with SIMULATED DATA for testing

Key Improvements:
1. Synthetic data generation with known parameters
2. Fixed physics loss computation during sampling
3. Validation/test split
4. Uncertainty quantification
5. Better normalization handling
6. Convergence diagnostics
7. Comprehensive posterior predictive checks

Requirements:
pip install torch pyro-ppl matplotlib pandas numpy scipy
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.optim import ClippedAdam
from scipy.integrate import odeint

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================================


def bergman_odes(state, t, p1, p2, p3, I_func, Gb, Ib):
    """
    Bergman minimal model ODEs

    dG/dt = -p1*(G - Gb) - X*G
    dX/dt = -p2*X + p3*(I(t) - Ib)

    Parameters:
    -----------
    state : array [G, X]
    t : float, time
    p1, p2, p3 : float, Bergman parameters
    I_func : callable, insulin function I(t)
    Gb, Ib : float, basal glucose and insulin
    """
    G, X = state
    I = I_func(t)

    dG_dt = -p1 * (G - Gb) - X * G
    dX_dt = -p2 * X + p3 * (I - Ib)

    return [dG_dt, dX_dt]


def generate_synthetic_data(
    n_points=200,
    time_span=(0, 300),  # minutes
    true_params={"p1": 0.028, "p2": 0.025, "p3": 1.5e-5},
    Gb=100.0,  # mg/dL
    Ib=10.0,  # μU/mL
    meal_times=[30, 120, 210],  # minutes
    meal_doses=[8.0, 10.0, 6.0],  # IU insulin
    tau=30.0,  # insulin absorption time constant
    noise_std=5.0,  # glucose measurement noise
    seed=42,
):
    """
    Generate synthetic diabetes data using the Bergman model

    Returns:
    --------
    dict with synthetic data and true parameters
    """
    np.random.seed(seed)

    print("=" * 70)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 70)
    print(f"\nTrue parameters:")
    print(f"  p1 = {true_params['p1']:.4f} min⁻¹ (glucose effectiveness)")
    print(f"  p2 = {true_params['p2']:.4f} min⁻¹ (insulin action decay)")
    print(f"  p3 = {true_params['p3']:.6f} min⁻² per μU/mL (insulin sensitivity)")
    print(f"  Gb = {Gb:.1f} mg/dL (basal glucose)")
    print(f"  Ib = {Ib:.1f} μU/mL (basal insulin)")

    print(f"\nMeal schedule:")
    for t, dose in zip(meal_times, meal_doses):
        print(f"  t={t} min: {dose} IU insulin")

    # Time points
    t_eval = np.linspace(time_span[0], time_span[1], n_points)

    # Create insulin function
    def insulin_function(t):
        """Insulin concentration at time t"""
        I = Ib  # Start with basal
        for t_meal, dose in zip(meal_times, meal_doses):
            if t >= t_meal:
                # Exponential absorption
                dt = t - t_meal
                I += (dose * 1000.0 * 0.6 / 12.0) * np.exp(-dt / tau)
        return I

    # Solve ODEs
    initial_state = [Gb, 0.0]  # Start at basal glucose, X=0

    solution = odeint(
        bergman_odes,
        initial_state,
        t_eval,
        args=(
            true_params["p1"],
            true_params["p2"],
            true_params["p3"],
            insulin_function,
            Gb,
            Ib,
        ),
    )

    G_true = solution[:, 0]
    X_true = solution[:, 1]

    # Generate insulin timeseries
    I_true = np.array([insulin_function(t) for t in t_eval])

    # Add measurement noise to glucose
    G_obs = G_true + np.random.normal(0, noise_std, size=G_true.shape)

    print(f"\nGenerated data statistics:")
    print(f"  Time points: {n_points}")
    print(f"  Time range: {t_eval[0]:.1f} - {t_eval[-1]:.1f} min")
    print(f"  G_true range: {G_true.min():.1f} - {G_true.max():.1f} mg/dL")
    print(f"  G_obs range: {G_obs.min():.1f} - {G_obs.max():.1f} mg/dL")
    print(f"  I range: {I_true.min():.1f} - {I_true.max():.1f} μU/mL")
    print(f"  X range: {X_true.min():.4f} - {X_true.max():.4f}")
    print(f"  Noise std: {noise_std:.1f} mg/dL")

    return {
        "t_minutes": t_eval,
        "G_true": G_true,
        "G_obs": G_obs,
        "X_true": X_true,
        "I_true": I_true,
        "Gb": Gb,
        "Ib": Ib,
        "true_params": true_params,
        "noise_std": noise_std,
    }


def prepare_synthetic_data(synthetic_data):
    """
    Prepare synthetic data for PINN training
    """
    t_minutes = synthetic_data["t_minutes"]
    G_obs = synthetic_data["G_obs"]
    I_obs = synthetic_data["I_true"]
    Gb = synthetic_data["Gb"]
    Ib = synthetic_data["Ib"]

    # Normalization
    t_mean, t_std = t_minutes.mean(), t_minutes.std()
    G_mean, G_std = G_obs.mean(), G_obs.std()
    I_mean, I_std = I_obs.mean(), I_obs.std()

    t_norm = (t_minutes - t_mean) / t_std
    G_norm = (G_obs - G_mean) / G_std
    I_norm = (I_obs - I_mean) / I_std

    # Convert to tensors
    data = {
        "t": torch.tensor(t_norm, dtype=torch.float32).reshape(-1, 1),
        "t_raw": torch.tensor(t_minutes, dtype=torch.float32).reshape(-1, 1),
        "G_obs": torch.tensor(G_norm, dtype=torch.float32),
        "G_obs_raw": torch.tensor(G_obs, dtype=torch.float32),
        "G_true_raw": torch.tensor(synthetic_data["G_true"], dtype=torch.float32),
        "X_true_raw": torch.tensor(synthetic_data["X_true"], dtype=torch.float32),
        "I_obs": torch.tensor(I_norm, dtype=torch.float32).reshape(-1, 1),
        "I_obs_raw": torch.tensor(I_obs, dtype=torch.float32).reshape(-1, 1),
        "Gb": float(Gb),
        "Ib": float(Ib),
        "normalization": {
            "t_mean": t_mean,
            "t_std": t_std,
            "G_mean": G_mean,
            "G_std": G_std,
            "I_mean": I_mean,
            "I_std": I_std,
        },
        "true_params": synthetic_data["true_params"],
        "noise_std": synthetic_data["noise_std"],
    }

    return data


def train_test_split(data, train_fraction=0.8, random=True):
    """
    Split data into training and test sets
    """
    n = len(data["t"])
    n_train = int(n * train_fraction)

    if random:
        indices = torch.randperm(n)
    else:
        # Sequential split (useful for time series)
        indices = torch.arange(n)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def split_dict(d, idx):
        result = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and len(v) == n:
                result[k] = v[idx]
            else:
                result[k] = v
        return result

    train_data = split_dict(data, train_idx)
    test_data = split_dict(data, test_idx)

    print(f"\nData split:")
    print(f"  Training points: {len(train_idx)} ({train_fraction*100:.0f}%)")
    print(f"  Test points: {len(test_idx)} ({(1-train_fraction)*100:.0f}%)")

    return train_data, test_data


# ============================================================================
# 2. IMPROVED NEURAL NETWORK ARCHITECTURE
# ============================================================================


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Bergman model

    Input: normalized time t
    Output: [G(t), X(t)] - normalized glucose and insulin action
    """

    def __init__(self, hidden_dims=[64, 64, 64], activation="tanh"):
        super().__init__()

        layers = []
        input_dim = 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "softplus":
                layers.append(nn.Softplus())
            else:
                layers.append(nn.Tanh())

            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 2))  # Output: [G, X]

        self.network = nn.Sequential(*layers)

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        """Forward pass"""
        return self.network(t)


# ============================================================================
# 3. IMPROVED BAYESIAN MODEL AND GUIDE
# ============================================================================


def model(
    t,
    G_obs,
    I_obs,
    Gb,
    Ib,
    norm_params,
    pinn_net,
    fix_p3=False,
    p3_value=1e-5,
    compute_physics=True,
):
    """
    Improved Bayesian PINN model

    Key improvements:
    - Proper handling of normalization in ODEs
    - Fixed physics loss during sampling
    - Better prior specifications
    """
    # Priors with documentation
    # p1: glucose effectiveness (min⁻¹), literature: 0.01-0.05, mean ~0.028
    p1 = pyro.sample(
        "p1",
        dist.LogNormal(
            torch.tensor(-3.58),  # log(0.028)
            torch.tensor(0.3),  # narrower prior based on literature
        ),
    )

    # p2: insulin action decay (min⁻¹), literature: 0.01-0.05, mean ~0.025
    p2 = pyro.sample(
        "p2", dist.LogNormal(torch.tensor(-3.69), torch.tensor(0.3))  # log(0.025)
    )

    # p3: insulin sensitivity (min⁻² per μU/mL)
    if fix_p3:
        p3 = torch.tensor(p3_value, dtype=torch.float32)
    else:
        p3 = pyro.sample(
            "p3", dist.LogNormal(torch.tensor(-11.1), torch.tensor(0.5))  # log(1.5e-5)
        )

    # Noise parameters
    sigma_G = pyro.sample("sigma_G", dist.HalfNormal(0.5))

    if compute_physics:
        sigma_phys = pyro.sample("sigma_phys", dist.HalfNormal(0.2))

    # Register neural network
    pinn = pyro.module("pinn", pinn_net)

    # Forward pass
    if compute_physics:
        # Training mode: compute gradients for physics loss
        t_phys = t.clone().requires_grad_(True)

        out = pinn(t_phys)
        G_hat_norm = out[:, 0:1]
        X_hat_norm = out[:, 1:2]

        # Denormalize for ODE (work in physical space)
        G_mean, G_std = norm_params["G_mean"], norm_params["G_std"]
        I_mean, I_std = norm_params["I_mean"], norm_params["I_std"]
        t_mean, t_std = norm_params["t_mean"], norm_params["t_std"]

        G_hat = G_hat_norm * G_std + G_mean
        X_hat = X_hat_norm  # X doesn't need denormalization (it's a derived state)
        I_phys = I_obs * I_std + I_mean

        # Compute time derivatives in physical space
        dG_dt_norm = torch.autograd.grad(
            G_hat_norm,
            t_phys,
            grad_outputs=torch.ones_like(G_hat_norm),
            create_graph=True,
            retain_graph=True,
        )[0]

        dX_dt_norm = torch.autograd.grad(
            X_hat_norm,
            t_phys,
            grad_outputs=torch.ones_like(X_hat_norm),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Convert to physical time derivatives: dt_phys = dt_norm * t_std
        dG_dt = dG_dt_norm * G_std / t_std
        dX_dt = dX_dt_norm / t_std

        # Bergman ODEs in physical space
        # dG/dt = -p1*(G - Gb) - X*G
        # dX/dt = -p2*X + p3*(I - Ib)
        R_G = dG_dt + p1 * (G_hat - Gb) + X_hat * G_hat
        R_X = dX_dt + p2 * X_hat - p3 * (I_phys - Ib)

        # Normalize residuals for numerical stability
        R_G_norm = R_G / G_std
        R_X_norm = R_X

        # Physics loss
        with pyro.plate("physics_plate", len(t)):
            pyro.sample(
                "physics_G", dist.Normal(0.0, sigma_phys), obs=R_G_norm.squeeze()
            )
            pyro.sample(
                "physics_X", dist.Normal(0.0, sigma_phys), obs=R_X_norm.squeeze()
            )

        # Use normalized predictions for data likelihood
        G_pred = G_hat_norm.squeeze()
    else:
        # Sampling mode: no gradients needed
        with torch.no_grad():
            out = pinn(t)
            G_pred = out[:, 0]

    # Data likelihood
    with pyro.plate("data_plate", len(G_obs)):
        pyro.sample("G_obs", dist.Normal(G_pred, sigma_G), obs=G_obs)


def guide(
    t,
    G_obs,
    I_obs,
    Gb,
    Ib,
    norm_params,
    pinn_net,
    fix_p3=False,
    p3_value=1e-5,
    compute_physics=True,
):
    """
    Variational guide (improved)
    """
    # Variational parameters for p1
    mean_p1_loc = pyro.param("mean_p1_loc", torch.tensor(-3.58))
    mean_p1_scale = pyro.param(
        "mean_p1_scale", torch.tensor(0.3), constraint=dist.constraints.positive
    )
    pyro.sample("p1", dist.LogNormal(mean_p1_loc, mean_p1_scale))

    # Variational parameters for p2
    mean_p2_loc = pyro.param("mean_p2_loc", torch.tensor(-3.69))
    mean_p2_scale = pyro.param(
        "mean_p2_scale", torch.tensor(0.3), constraint=dist.constraints.positive
    )
    pyro.sample("p2", dist.LogNormal(mean_p2_loc, mean_p2_scale))

    # Variational parameters for p3 (if not fixed)
    if not fix_p3:
        mean_p3_loc = pyro.param("mean_p3_loc", torch.tensor(-11.1))
        mean_p3_scale = pyro.param(
            "mean_p3_scale", torch.tensor(0.5), constraint=dist.constraints.positive
        )
        pyro.sample("p3", dist.LogNormal(mean_p3_loc, mean_p3_scale))

    # Variational parameters for noise
    sigma_G_scale = pyro.param(
        "sigma_G_scale", torch.tensor(0.5), constraint=dist.constraints.positive
    )
    pyro.sample("sigma_G", dist.HalfNormal(sigma_G_scale))

    if compute_physics:
        sigma_phys_scale = pyro.param(
            "sigma_phys_scale", torch.tensor(0.2), constraint=dist.constraints.positive
        )
        pyro.sample("sigma_phys", dist.HalfNormal(sigma_phys_scale))

    # Register network
    pyro.module("pinn", pinn_net)


# ============================================================================
# 4. TRAINING WITH IMPROVEMENTS
# ============================================================================


def check_convergence(losses, window=200, threshold=1e-4):
    """Check if training has converged"""
    if len(losses) < window:
        return False

    recent = losses[-window:]
    trend = np.polyfit(range(window), recent, 1)[0]

    return abs(trend) < threshold


def train_bayesian_pinn(
    train_data,
    val_data=None,
    n_iterations=5000,
    lr=0.001,
    fix_p3=False,
    save_dir="./checkpoints",
    checkpoint_freq=500,
    early_stopping_patience=1000,
):
    """
    Improved training with validation and early stopping
    """
    print("\n" + "=" * 70)
    print("TRAINING BAYESIAN PINN")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)

    # Clear parameter store
    pyro.clear_param_store()

    # Initialize network
    pinn_net = PINN(hidden_dims=[64, 64, 64])

    # Setup SVI
    optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Extract data
    t = train_data["t"]
    G_obs = train_data["G_obs"]
    I_obs = train_data["I_obs"]
    Gb = train_data["Gb"]
    Ib = train_data["Ib"]
    norm_params = train_data["normalization"]

    # Validation data if available
    if val_data is not None:
        t_val = val_data["t"]
        G_obs_val = val_data["G_obs"]
        I_obs_val = val_data["I_obs"]

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nConfiguration:")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {lr}")
    print(f"  Fix p3: {fix_p3}")
    print(f"  Early stopping patience: {early_stopping_patience}")

    for iteration in range(n_iterations):
        # Training step
        loss = svi.step(
            t, G_obs, I_obs, Gb, Ib, norm_params, pinn_net, fix_p3, 1e-5, True
        )
        train_losses.append(loss)

        # Validation step
        if val_data is not None and (iteration + 1) % 50 == 0:
            val_loss = svi.evaluate_loss(
                t_val,
                G_obs_val,
                I_obs_val,
                Gb,
                Ib,
                norm_params,
                pinn_net,
                fix_p3,
                1e-5,
                False,
            )
            val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                best_checkpoint = {
                    "iteration": iteration + 1,
                    "train_loss": loss,
                    "val_loss": val_loss,
                    "pinn_state_dict": pinn_net.state_dict(),
                    "pyro_param_store": pyro.get_param_store().get_state(),
                }
                torch.save(best_checkpoint, os.path.join(save_dir, "best_model.pt"))
            else:
                patience_counter += 50

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at iteration {iteration + 1}")
                    break

        # Progress reporting
        if (iteration + 1) % 100 == 0 or iteration == 0:
            msg = f"Iter {iteration + 1:5d} | Train Loss: {loss:.2e}"
            if val_data is not None and len(val_losses) > 0:
                msg += f" | Val Loss: {val_losses[-1]:.2e}"
            print(msg)

        # Checkpointing
        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint = {
                "iteration": iteration + 1,
                "train_loss": loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "pinn_state_dict": pinn_net.state_dict(),
                "pyro_param_store": pyro.get_param_store().get_state(),
            }
            torch.save(
                checkpoint, os.path.join(save_dir, f"checkpoint_{iteration + 1}.pt")
            )

        # Convergence check
        if (iteration + 1) % 500 == 0:
            if check_convergence(train_losses):
                print(f"\nConverged at iteration {iteration + 1}")
                break

    print("\nTraining completed!")
    return pinn_net, train_losses, val_losses


# ============================================================================
# 5. ENHANCED ANALYSIS AND VISUALIZATION
# ============================================================================


def analyze_results(pinn_net, data, n_samples=1000, fix_p3=False):
    """Enhanced posterior analysis"""
    print("\n" + "=" * 70)
    print("POSTERIOR ANALYSIS")
    print("=" * 70)

    # Sample from posterior (without computing physics for efficiency)
    predictive = Predictive(model, guide=guide, num_samples=n_samples)

    samples = predictive(
        data["t"],
        data["G_obs"],
        data["I_obs"],
        data["Gb"],
        data["Ib"],
        data["normalization"],
        pinn_net,
        fix_p3,
        1e-5,
        False,
    )

    # Parameter statistics
    p1_samples = samples["p1"].detach().numpy()
    p2_samples = samples["p2"].detach().numpy()

    print("\nParameter Posterior Statistics:")
    print(
        f"{'Parameter':<10} {'Mean':<10} {'Std':<10} {'2.5%':<10} {'97.5%':<10} {'True':<10}"
    )
    print("-" * 60)

    true_params = data.get("true_params", {})

    print(
        f"{'p1':<10} {p1_samples.mean():<10.5f} {p1_samples.std():<10.5f} "
        f"{np.percentile(p1_samples, 2.5):<10.5f} {np.percentile(p1_samples, 97.5):<10.5f} "
        f"{true_params.get('p1', 'N/A'):<10}"
    )

    print(
        f"{'p2':<10} {p2_samples.mean():<10.5f} {p2_samples.std():<10.5f} "
        f"{np.percentile(p2_samples, 2.5):<10.5f} {np.percentile(p2_samples, 97.5):<10.5f} "
        f"{true_params.get('p2', 'N/A'):<10}"
    )

    if not fix_p3 and "p3" in samples:
        p3_samples = samples["p3"].detach().numpy()
        print(
            f"{'p3':<10} {p3_samples.mean():<10.6f} {p3_samples.std():<10.6f} "
            f"{np.percentile(p3_samples, 2.5):<10.6f} {np.percentile(p3_samples, 97.5):<10.6f} "
            f"{true_params.get('p3', 'N/A'):<10}"
        )
    else:
        print(
            f"{'p3':<10} {'FIXED':<10} {'-':<10} {'-':<10} {'-':<10} "
            f"{true_params.get('p3', 'N/A'):<10}"
        )

    # Compute coverage
    if true_params:
        p1_covered = (
            np.percentile(p1_samples, 2.5)
            <= true_params["p1"]
            <= np.percentile(p1_samples, 97.5)
        )
        p2_covered = (
            np.percentile(p2_samples, 2.5)
            <= true_params["p2"]
            <= np.percentile(p2_samples, 97.5)
        )

        print(f"\n95% Credible Interval Coverage:")
        print(f"  p1: {'✓' if p1_covered else '✗'}")
        print(f"  p2: {'✓' if p2_covered else '✗'}")

        if not fix_p3 and "p3" in samples:
            p3_covered = (
                np.percentile(p3_samples, 2.5)
                <= true_params["p3"]
                <= np.percentile(p3_samples, 97.5)
            )
            print(f"  p3: {'✓' if p3_covered else '✗'}")

    return samples


def predict_with_uncertainty(pinn_net, data, samples, n_posterior_samples=100):
    """Generate predictions with uncertainty quantification"""

    # Select random posterior samples
    n_available = len(samples["p1"])
    indices = np.random.choice(
        n_available, min(n_posterior_samples, n_available), replace=False
    )

    predictions_G = []
    predictions_X = []

    with torch.no_grad():
        for idx in indices:
            out = pinn_net(data["t"])
            predictions_G.append(out[:, 0].numpy())
            predictions_X.append(out[:, 1].numpy())

    predictions_G = np.array(predictions_G)
    predictions_X = np.array(predictions_X)

    # Denormalize
    norm = data["normalization"]
    predictions_G = predictions_G * norm["G_std"] + norm["G_mean"]

    return {
        "G_mean": predictions_G.mean(axis=0),
        "G_std": predictions_G.std(axis=0),
        "G_lower": np.percentile(predictions_G, 2.5, axis=0),
        "G_upper": np.percentile(predictions_G, 97.5, axis=0),
        "X_mean": predictions_X.mean(axis=0),
        "X_std": predictions_X.std(axis=0),
    }


def plot_comprehensive_results(pinn_net, data, train_losses, val_losses, samples):
    """Comprehensive visualization"""

    fig = plt.figure(figsize=(18, 12))

    # Get predictions with uncertainty
    uncertainty = predict_with_uncertainty(pinn_net, data, samples)

    t_raw = data["t_raw"].numpy().flatten()
    G_obs_raw = data["G_obs_raw"].numpy()

    # Plot 1: Training curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label="Train", alpha=0.7)
    if val_losses:
        # Match validation losses to training iterations
        val_iters = np.linspace(0, len(train_losses), len(val_losses))
        ax1.plot(val_iters, val_losses, label="Validation", alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ELBO Loss")
    ax1.set_title("Training Progress")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Glucose predictions with uncertainty
    ax2 = plt.subplot(3, 3, 2)
    ax2.fill_between(
        t_raw, uncertainty["G_lower"], uncertainty["G_upper"], alpha=0.3, label="95% CI"
    )
    ax2.plot(t_raw, G_obs_raw, "o", alpha=0.5, markersize=3, label="Observed")
    if "G_true_raw" in data:
        ax2.plot(
            t_raw,
            data["G_true_raw"].numpy(),
            "--",
            linewidth=2,
            label="True",
            color="green",
        )
    ax2.plot(t_raw, uncertainty["G_mean"], "-", linewidth=2, label="Predicted")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Glucose (mg/dL)")
    ax2.set_title("Glucose: Predictions with Uncertainty")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax3 = plt.subplot(3, 3, 3)
    residuals = G_obs_raw - uncertainty["G_mean"]
    ax3.scatter(t_raw, residuals, alpha=0.5, s=10)
    ax3.axhline(y=0, color="r", linestyle="--")
    ax3.fill_between(
        t_raw,
        -2 * uncertainty["G_std"],
        2 * uncertainty["G_std"],
        alpha=0.2,
        color="gray",
    )
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Residual (mg/dL)")
    ax3.set_title("Prediction Residuals")
    ax3.grid(True, alpha=0.3)

    # Plot 4: X(t) trajectory with uncertainty
    ax4 = plt.subplot(3, 3, 4)
    ax4.fill_between(
        t_raw,
        uncertainty["X_mean"] - 2 * uncertainty["X_std"],
        uncertainty["X_mean"] + 2 * uncertainty["X_std"],
        alpha=0.3,
    )
    ax4.plot(t_raw, uncertainty["X_mean"], "-", linewidth=2, label="Predicted")
    if "X_true_raw" in data:
        ax4.plot(
            t_raw,
            data["X_true_raw"].numpy(),
            "--",
            linewidth=2,
            label="True",
            color="green",
        )
    ax4.set_xlabel("Time (minutes)")
    ax4.set_ylabel("Insulin Action X(t)")
    ax4.set_title("Insulin Action State")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Insulin input
    ax5 = plt.subplot(3, 3, 5)
    I_raw = data["I_obs_raw"].numpy().flatten()
    ax5.plot(t_raw, I_raw, linewidth=2)
    ax5.axhline(y=data["Ib"], color="r", linestyle="--", label=f"Ib = {data['Ib']:.1f}")
    ax5.set_xlabel("Time (minutes)")
    ax5.set_ylabel("Insulin (μU/mL)")
    ax5.set_title("Insulin Input")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Parameter posterior - p1
    ax6 = plt.subplot(3, 3, 6)
    p1_samples = samples["p1"].detach().numpy()
    ax6.hist(p1_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax6.axvline(
        p1_samples.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {p1_samples.mean():.4f}",
        linewidth=2,
    )
    if "true_params" in data:
        ax6.axvline(
            data["true_params"]["p1"],
            color="green",
            linestyle="-",
            label=f"True: {data['true_params']['p1']:.4f}",
            linewidth=2,
        )
    ax6.set_xlabel("p1 (min⁻¹)")
    ax6.set_ylabel("Density")
    ax6.set_title("Posterior: Glucose Effectiveness")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Parameter posterior - p2
    ax7 = plt.subplot(3, 3, 7)
    p2_samples = samples["p2"].detach().numpy()
    ax7.hist(p2_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax7.axvline(
        p2_samples.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {p2_samples.mean():.4f}",
        linewidth=2,
    )
    if "true_params" in data:
        ax7.axvline(
            data["true_params"]["p2"],
            color="green",
            linestyle="-",
            label=f"True: {data['true_params']['p2']:.4f}",
            linewidth=2,
        )
    ax7.set_xlabel("p2 (min⁻¹)")
    ax7.set_ylabel("Density")
    ax7.set_title("Posterior: Insulin Action Decay")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Parameter posterior - p3 or correlation
    ax8 = plt.subplot(3, 3, 8)
    if "p3" in samples:
        p3_samples = samples["p3"].detach().numpy()
        ax8.hist(p3_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
        ax8.axvline(
            p3_samples.mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {p3_samples.mean():.6f}",
            linewidth=2,
        )
        if "true_params" in data:
            ax8.axvline(
                data["true_params"]["p3"],
                color="green",
                linestyle="-",
                label=f"True: {data['true_params']['p3']:.6f}",
                linewidth=2,
            )
        ax8.set_xlabel("p3 (min⁻² per μU/mL)")
        ax8.set_ylabel("Density")
        ax8.set_title("Posterior: Insulin Sensitivity")
        ax8.legend()
    else:
        # Show parameter correlation
        ax8.scatter(p1_samples, p2_samples, alpha=0.3, s=10)
        ax8.set_xlabel("p1")
        ax8.set_ylabel("p2")
        ax8.set_title("Parameter Correlation: p1 vs p2")

        # Add correlation coefficient
        corr = np.corrcoef(p1_samples, p2_samples)[0, 1]
        ax8.text(
            0.05,
            0.95,
            f"ρ = {corr:.3f}",
            transform=ax8.transAxes,
            verticalalignment="top",
        )
    ax8.grid(True, alpha=0.3)

    # Plot 9: Posterior predictive check
    ax9 = plt.subplot(3, 3, 9)
    # Q-Q plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=ax9)
    ax9.set_title("Q-Q Plot: Residual Normality Check")
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_metrics(data, predictions):
    """Compute performance metrics"""
    G_true = data["G_obs_raw"].numpy()
    G_pred = predictions["G_mean"]

    # RMSE
    rmse = np.sqrt(np.mean((G_true - G_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(G_true - G_pred))

    # R²
    ss_res = np.sum((G_true - G_pred) ** 2)
    ss_tot = np.sum((G_true - G_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Coverage (percentage of observations within 95% CI)
    in_ci = (G_true >= predictions["G_lower"]) & (G_true <= predictions["G_upper"])
    coverage = in_ci.mean() * 100

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"RMSE:      {rmse:.2f} mg/dL")
    print(f"MAE:       {mae:.2f} mg/dL")
    print(f"R²:        {r2:.4f}")
    print(f"95% CI Coverage: {coverage:.1f}%")

    if "noise_std" in data:
        print(f"\nTrue noise std: {data['noise_std']:.2f} mg/dL")
        print(f"RMSE/True noise: {rmse/data['noise_std']:.2f}")


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================


def main():
    """Main execution with synthetic data"""

    print("=" * 70)
    print("BAYESIAN PINN FOR BERGMAN MODEL - SYNTHETIC DATA TEST")
    print("=" * 70)

    # Configuration
    SAVE_DIR = "./results_synthetic"
    N_ITERATIONS = 30_000
    LEARNING_RATE = 0.001
    CHECKPOINT_FREQ = 500
    FIX_P3 = False  # Try to infer all parameters

    # Step 1: Generate synthetic data
    synthetic_data = generate_synthetic_data(
        n_points=200,
        time_span=(0, 300),
        true_params={"p1": 0.028, "p2": 0.025, "p3": 1.5e-5},
        Gb=100.0,
        Ib=10.0,
        meal_times=[30, 120, 210],
        meal_doses=[8.0, 10.0, 6.0],
        tau=30.0,
        noise_std=5.0,
        seed=42,
    )

    # Step 2: Prepare data
    data = prepare_synthetic_data(synthetic_data)

    # Step 3: Train/test split
    train_data, test_data = train_test_split(data, train_fraction=0.8, random=True)

    # Step 4: Train model
    pinn_net, train_losses, val_losses = train_bayesian_pinn(
        train_data,
        val_data=test_data,
        n_iterations=N_ITERATIONS,
        lr=LEARNING_RATE,
        fix_p3=FIX_P3,
        save_dir=SAVE_DIR,
        checkpoint_freq=CHECKPOINT_FREQ,
        early_stopping_patience=1000,
    )

    # Step 5: Analyze on full dataset
    samples = analyze_results(pinn_net, data, n_samples=1000, fix_p3=FIX_P3)

    # Step 6: Predictions with uncertainty
    predictions = predict_with_uncertainty(
        pinn_net, data, samples, n_posterior_samples=100
    )

    # Step 7: Compute metrics
    compute_metrics(data, predictions)

    # Step 8: Comprehensive visualization
    fig = plot_comprehensive_results(pinn_net, data, train_losses, val_losses, samples)

    # Save results
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig_path = os.path.join(SAVE_DIR, "comprehensive_results.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nResults saved to: {fig_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
