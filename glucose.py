"""
Complete Bayesian Physics-Informed Neural Network (B-PINN) for Bergman Minimal Model
Using Pyro for variational inference

This script includes:
1. Data loading and preprocessing
2. Insulin absorption modeling
3. Neural network architecture
4. Bayesian model and guide definitions
5. Training with SVI
6. Results visualization and analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(filepath, tau=30.0, bioavailability=0.6, V_d=12.0):
    """
    Load diabetes data and prepare for Bergman model
    
    Parameters:
    -----------
    filepath : str
        Path to Excel file
    tau : float
        Insulin absorption time constant (minutes)
    bioavailability : float
        Subcutaneous insulin bioavailability (0-1)
    V_d : float
        Insulin distribution volume (liters)
    
    Returns:
    --------
    dict containing preprocessed data
    """
    print(f"Loading data from {filepath}...")
    
    # Read Excel file
    df = pd.read_excel(filepath)
    
    # Display available columns
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")
    
    # Find relevant columns
    time_cols = [col for col in df.columns if any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
    glucose_cols = [col for col in df.columns if any(x in col.lower() for x in ['glucose', 'cgm', 'bg'])]
    bolus_cols = [col for col in df.columns if 'bolus' in col.lower()]
    basal_cols = [col for col in df.columns if 'basal' in col.lower()]
    
    print(f"\nDetected columns:")
    print(f"  Time: {time_cols}")
    print(f"  Glucose: {glucose_cols}")
    print(f"  Bolus insulin: {bolus_cols}")
    print(f"  Basal insulin: {basal_cols}")
    
    # Extract data
    time_col = time_cols[0] if time_cols else df.columns[0]
    glucose_col = glucose_cols[0] if glucose_cols else None
    bolus_col = bolus_cols[0] if bolus_cols else None
    basal_col = basal_cols[0] if basal_cols else None
    
    if glucose_col is None:
        raise ValueError("Could not find glucose column. Please specify manually.")
    
    # Clean data
    required_cols = [time_col, glucose_col]
    df_clean = df[required_cols].copy()
    
    # Add bolus insulin if available
    if bolus_col:
        df_clean['bolus'] = df[bolus_col].fillna(0.0)
    else:
        print("  ⚠️  Warning: No bolus insulin column found, assuming no boluses")
        df_clean['bolus'] = 0.0
    
    # Add basal insulin if available
    if basal_col:
        df_clean['basal'] = df[basal_col].fillna(0.0)
        print(f"  ✓ Using basal insulin from column: {basal_col}")
    else:
        print("  ⚠️  Warning: No basal insulin column found, will estimate from data")
        df_clean['basal'] = 0.0
    
    df_clean = df_clean.dropna(subset=[glucose_col])
    
    # Convert time to minutes from start
    if pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        t_minutes = (df_clean[time_col] - df_clean[time_col].min()).dt.total_seconds() / 60.0
    else:
        t_minutes = df_clean[time_col].values
    
    df_clean['t_minutes'] = t_minutes
    
    # Extract glucose and insulin
    G_obs = df_clean[glucose_col].values
    bolus = df_clean['bolus'].values
    basal_insulin_data = df_clean['basal'].values
    
    print(f"\nData statistics:")
    print(f"  Number of observations: {len(G_obs)}")
    print(f"  Time range: {t_minutes.min():.1f} - {t_minutes.max():.1f} minutes")
    print(f"  Glucose range: {G_obs.min():.1f} - {G_obs.max():.1f} mg/dL")
    print(f"  Number of insulin boluses: {(bolus > 0).sum()}")
    if (bolus > 0).any():
        print(f"  Insulin bolus range: {bolus[bolus > 0].min():.2f} - {bolus[bolus > 0].max():.2f} IU")
    if (basal_insulin_data > 0).any():
        print(f"  Basal insulin range: {basal_insulin_data[basal_insulin_data > 0].min():.2f} - {basal_insulin_data[basal_insulin_data > 0].max():.2f} (units in file)")
    
    # Model bolus insulin absorption with proper unit conversion
    I_bolus = np.zeros_like(t_minutes, dtype=float)
    IU_to_microU = 1000.0  # 1 IU = 1000 μU
    
    for i, dose in enumerate(bolus):
        if dose > 0:
            # Convert IU to plasma concentration (μU/mL)
            dose_concentration = (dose * IU_to_microU * bioavailability) / V_d
            
            # Exponential absorption model
            time_diff = t_minutes - t_minutes[i]
            absorption = dose_concentration * np.exp(-time_diff / tau)
            absorption[time_diff < 0] = 0  # Causality
            I_bolus += absorption
    
    # Handle basal insulin from data
    if basal_col and (basal_insulin_data > 0).any():
        # Basal insulin found in dataset
        basal_mean = basal_insulin_data[basal_insulin_data > 0].mean()
        
        # Auto-detect units based on magnitude
        if basal_mean < 5.0:  
            # Likely IU/hr (typical pump basal rate: 0.5-2.0 IU/hr)
            print(f"\n  ✓ Basal insulin detected as IU/hr (mean={basal_mean:.2f} IU/hr)")
            print(f"    Converting to plasma concentration...")
            
            # Convert IU/hr to steady-state plasma concentration (μU/mL)
            # At steady state: infusion rate = clearance rate
            # Clearance half-life ~5-10 min → clearance constant ~0.1 min^-1
            clearance_constant = 0.1  # min^-1
            I_basal_concentration = (basal_mean * bioavailability * IU_to_microU) / (V_d * clearance_constant * 60)
            Ib = I_basal_concentration
            print(f"    Calculated Ib = {Ib:.1f} μU/mL")
            
        else:  
            # Likely already in concentration units (μU/mL)
            print(f"\n  ✓ Basal insulin detected as concentration (mean={basal_mean:.2f} μU/mL)")
            Ib = basal_mean
            
    else:
        # No basal insulin in data - use default
        Ib = 10.0  # Typical fasting insulin concentration
        print(f"\n  ⚠️  No basal insulin data found - using default: Ib = {Ib:.1f} μU/mL")
        print(f"     (Typical fasting insulin concentration)")
    
    # Total insulin = basal + bolus contribution
    I_obs = Ib + I_bolus
    
    # Estimate basal glucose
    Gb = np.percentile(G_obs, 10)  # Use 10th percentile as basal glucose
    
    print(f"\nBasal values:")
    print(f"  Gb (basal glucose): {Gb:.1f} mg/dL")
    print(f"  Ib (basal insulin): {Ib:.1f} μU/mL")
    print(f"\nInsulin statistics:")
    print(f"  I_obs range: {I_obs.min():.2f} - {I_obs.max():.2f} μU/mL")
    print(f"  (I_obs - Ib) range: {(I_obs - Ib).min():.2f} - {(I_obs - Ib).max():.2f} μU/mL")
    
    # Check identifiability
    max_insulin_excursion = (I_obs - Ib).max()
    if max_insulin_excursion < 5.0:
        print(f"\n⚠️  WARNING: Insulin excursion is very small ({max_insulin_excursion:.2f} μU/mL)")
        print("   Parameter p3 may not be identifiable. Consider fixing it to literature value.")
        fix_p3_recommended = True
    else:
        fix_p3_recommended = False
    
    # Normalize data for neural network
    t_mean, t_std = t_minutes.mean(), t_minutes.std()
    G_mean, G_std = G_obs.mean(), G_obs.std()
    I_mean, I_std = I_obs.mean(), I_obs.std()
    
    t_norm = (t_minutes - t_mean) / t_std
    G_norm = (G_obs - G_mean) / G_std
    I_norm = (I_obs - I_mean) / I_std
    
    # Convert to tensors
    data = {
        't': torch.tensor(t_norm, dtype=torch.float32).reshape(-1, 1),
        't_raw': torch.tensor(t_minutes, dtype=torch.float32).reshape(-1, 1),
        'G_obs': torch.tensor(G_norm, dtype=torch.float32),
        'G_obs_raw': torch.tensor(G_obs, dtype=torch.float32),
        'I_obs': torch.tensor(I_norm, dtype=torch.float32).reshape(-1, 1),
        'I_obs_raw': torch.tensor(I_obs, dtype=torch.float32).reshape(-1, 1),
        'Gb': float(Gb),
        'Ib': float(Ib),
        'normalization': {
            't_mean': t_mean, 't_std': t_std,
            'G_mean': G_mean, 'G_std': G_std,
            'I_mean': I_mean, 'I_std': I_std
        },
        'fix_p3_recommended': fix_p3_recommended
    }
    
    return data


# ============================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# ============================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Bergman model
    
    Input: time t
    Output: [G(t), X(t)] - glucose and insulin action
    """
    
    def __init__(self, hidden_dims=[32, 32, 32]):
        super().__init__()
        
        layers = []
        input_dim = 1
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 2))  # Output: [G, X]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t):
        """
        Forward pass
        
        Parameters:
        -----------
        t : torch.Tensor, shape (N, 1)
            Normalized time points
        
        Returns:
        --------
        output : torch.Tensor, shape (N, 2)
            [G(t), X(t)] predictions
        """
        return self.network(t)


# ============================================================================
# 3. BAYESIAN MODEL AND GUIDE
# ============================================================================

def model(t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3=True, p3_value=1e-5, compute_physics=True):
    """
    Bayesian PINN model for Bergman minimal model
    
    Parameters:
    -----------
    t : torch.Tensor
        Normalized time points
    G_obs : torch.Tensor
        Observed glucose (normalized)
    I_obs : torch.Tensor
        Observed insulin (normalized)
    Gb : float
        Basal glucose (raw units)
    Ib : float
        Basal insulin (raw units)
    pinn_net : PINN
        Neural network instance
    fix_p3 : bool
        Whether to fix p3 to literature value
    p3_value : float
        Fixed value for p3 if fix_p3=True
    compute_physics : bool
        Whether to compute physics loss (False during sampling for efficiency)
    """
    # Priors on Bergman parameters (in log-space for numerical stability)
    # p1 ~ 0.03 min^-1
    p1 = pyro.sample("p1", dist.LogNormal(-3.5, 0.5))
    
    # p2 ~ 0.02 min^-1
    p2 = pyro.sample("p2", dist.LogNormal(-3.9, 0.5))
    
    # p3 ~ 1e-5 - either fixed or sampled
    if fix_p3:
        p3 = torch.tensor(p3_value, dtype=torch.float32)
    else:
        # Use log-space parameterization for better numerical stability
        log_p3 = pyro.sample("log_p3", dist.Normal(-11.5, 1.0))
        p3 = torch.exp(log_p3)
    
    # Noise parameters
    sigma_G = pyro.sample("sigma_G", dist.Exponential(0.2))
    sigma_phys = pyro.sample("sigma_phys", dist.Exponential(0.1))
    
    # Register neural network
    pinn = pyro.module("pinn", pinn_net)
    
    # Forward pass (with or without gradients)
    if compute_physics:
        # Enable gradient computation for physics loss (training mode)
        t_phys = t.clone().detach().requires_grad_(True)
        
        out = pinn(t_phys)
        G_hat = out[:, 0:1]
        X_hat = out[:, 1:2]
        
        # Compute time derivatives
        dG_dt = torch.autograd.grad(
            G_hat, t_phys,
            grad_outputs=torch.ones_like(G_hat),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dX_dt = torch.autograd.grad(
            X_hat, t_phys,
            grad_outputs=torch.ones_like(X_hat),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Bergman ODE residuals (in normalized space)
        R_G = dG_dt + (p1 + X_hat) * G_hat - p1 * 0.0  # Normalized Gb ≈ 0
        R_X = dX_dt + p2 * X_hat - p3 * I_obs
    else:
        # Sampling mode - no gradients needed
        with torch.no_grad():
            out = pinn(t)
            G_hat = out[:, 0:1]
        
        # Use dummy residuals with zero variance (they won't affect sampling)
        R_G = torch.zeros_like(G_hat)
        R_X = torch.zeros_like(G_hat)
    
    # Likelihoods
    with pyro.plate("observations", G_obs.shape[0]):
        # Data likelihood
        pyro.sample("G_obs",
                    dist.Normal(G_hat.squeeze(), sigma_G),
                    obs=G_obs)
        
        if compute_physics:
            # Physics likelihoods (only during training)
            pyro.sample("physics_G",
                        dist.Normal(torch.zeros_like(R_G.squeeze()), sigma_phys),
                        obs=R_G.squeeze())
            
            pyro.sample("physics_X",
                        dist.Normal(torch.zeros_like(R_X.squeeze()), sigma_phys),
                        obs=R_X.squeeze())


def guide(t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3=True, p3_value=1e-5, compute_physics=True):
    """
    Variational guide for SVI
    """
    # Variational parameters for p1
    mean_p1_loc = pyro.param("mean_p1_loc", torch.tensor(-3.5))
    mean_p1_scale = pyro.param("mean_p1_scale", torch.tensor(0.5),
                                constraint=dist.constraints.positive)
    pyro.sample("p1", dist.LogNormal(mean_p1_loc, mean_p1_scale))
    
    # Variational parameters for p2
    mean_p2_loc = pyro.param("mean_p2_loc", torch.tensor(-3.9))
    mean_p2_scale = pyro.param("mean_p2_scale", torch.tensor(0.5),
                                constraint=dist.constraints.positive)
    pyro.sample("p2", dist.LogNormal(mean_p2_loc, mean_p2_scale))
    
    # Variational parameters for p3 (only if not fixed)
    if not fix_p3:
        mean_log_p3 = pyro.param("mean_log_p3", torch.tensor(-11.5))
        scale_log_p3 = pyro.param("scale_log_p3", torch.tensor(1.0),
                                   constraint=dist.constraints.positive)
        pyro.sample("log_p3", dist.Normal(mean_log_p3, scale_log_p3))
    
    # Variational parameters for noise
    sigma_G_rate = pyro.param("sigma_G_rate", torch.tensor(0.2),
                              constraint=dist.constraints.positive)
    pyro.sample("sigma_G", dist.Exponential(sigma_G_rate))
    
    sigma_phys_rate = pyro.param("sigma_phys_rate", torch.tensor(0.1),
                                  constraint=dist.constraints.positive)
    pyro.sample("sigma_phys", dist.Exponential(sigma_phys_rate))
    
    # Register network
    pyro.module("pinn", pinn_net)
    
    with pyro.plate("observations", G_obs.shape[0]):
        pass


# ============================================================================
# 4. TRAINING WITH CHECKPOINTING
# ============================================================================

def train_bayesian_pinn(data, n_iterations=5000, lr=0.001, fix_p3=True, 
                        save_dir='./checkpoints', checkpoint_freq=500):
    """
    Train Bayesian PINN using SVI with checkpointing
    
    Parameters:
    -----------
    data : dict
        Preprocessed data dictionary
    n_iterations : int
        Number of SVI iterations
    lr : float
        Learning rate
    fix_p3 : bool
        Whether to fix p3 to literature value
    save_dir : str
        Directory to save checkpoints (use Google Drive path in Colab)
    checkpoint_freq : int
        Save checkpoint every N iterations
    """
    print("\n" + "="*70)
    print("TRAINING BAYESIAN PINN")
    print("="*70)
    
    # Create save directory
    import os
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Clear parameter store
    pyro.clear_param_store()
    
    # Initialize network
    pinn_net = PINN(hidden_dims=[32, 32, 32])
    
    # Setup SVI
    optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Extract data
    t = data['t']
    G_obs = data['G_obs']
    I_obs = data['I_obs']
    Gb = data['Gb']
    Ib = data['Ib']
    
    # Training loop
    losses = []
    best_loss = float('inf')
    
    print(f"\nTraining for {n_iterations} iterations...")
    print(f"Fix p3: {fix_p3}")
    print(f"Checkpoint frequency: every {checkpoint_freq} iterations")
    
    for iteration in range(n_iterations):
        loss = svi.step(t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3, 1e-5, True)  # compute_physics=True
        losses.append(loss)
        
        # Print progress
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1:5d} | ELBO Loss: {loss:.2e}")
        
        # Save checkpoint
        if (iteration + 1) % checkpoint_freq == 0 or iteration == n_iterations - 1:
            checkpoint = {
                'iteration': iteration + 1,
                'loss': loss,
                'losses': losses,
                'pinn_state_dict': pinn_net.state_dict(),
                'pyro_param_store': pyro.get_param_store().get_state(),
                'config': {
                    'n_iterations': n_iterations,
                    'lr': lr,
                    'fix_p3': fix_p3,
                    'Gb': Gb,
                    'Ib': Ib
                }
            }
            
            checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{iteration + 1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if loss < best_loss:
                best_loss = loss
                best_path = os.path.join(save_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"  → Best model updated: {best_path}")
    
    # Save final losses as CSV
    losses_df = pd.DataFrame({
        'iteration': range(1, len(losses) + 1),
        'loss': losses
    })
    losses_csv_path = os.path.join(save_dir, 'training_losses.csv')
    losses_df.to_csv(losses_csv_path, index=False)
    print(f"\n  → Losses saved to: {losses_csv_path}")
    
    # Save final model
    final_checkpoint = {
        'iteration': n_iterations,
        'loss': losses[-1],
        'losses': losses,
        'pinn_state_dict': pinn_net.state_dict(),
        'pyro_param_store': pyro.get_param_store().get_state(),
        'config': {
            'n_iterations': n_iterations,
            'lr': lr,
            'fix_p3': fix_p3,
            'Gb': Gb,
            'Ib': Ib
        }
    }
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(final_checkpoint, final_path)
    print(f"  → Final model saved: {final_path}")
    
    print("\nTraining completed!")
    
    return pinn_net, losses


def load_checkpoint(checkpoint_path, pinn_net=None):
    """
    Load a saved checkpoint
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
    pinn_net : PINN, optional
        PINN instance to load weights into. If None, creates new one.
    
    Returns:
    --------
    pinn_net : PINN
        Network with loaded weights
    checkpoint : dict
        Full checkpoint data
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Create or use provided network
    if pinn_net is None:
        pinn_net = PINN(hidden_dims=[32, 32, 32])
    
    # Load network weights
    pinn_net.load_state_dict(checkpoint['pinn_state_dict'])
    
    # Restore Pyro parameter store
    pyro.clear_param_store()
    pyro.get_param_store().set_state(checkpoint['pyro_param_store'])
    
    print(f"  Loaded iteration: {checkpoint['iteration']}")
    print(f"  Loss at checkpoint: {checkpoint['loss']:.2e}")
    
    return pinn_net, checkpoint


# ============================================================================
# 5. RESULTS ANALYSIS
# ============================================================================

def analyze_results(pinn_net, data, n_samples=1000, fix_p3=True):
    """
    Analyze posterior distributions and predictions
    """
    print("\n" + "="*70)
    print("POSTERIOR ANALYSIS")
    print("="*70)
    
    from pyro.infer import Predictive
    
    # Sample from posterior WITHOUT computing physics loss (for efficiency)
    # We only need the parameter samples, not the physics residuals
    predictive = Predictive(
        lambda t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3: model(
            t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3, 1e-5, compute_physics=False
        ),
        guide=lambda t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3: guide(
            t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3, 1e-5, compute_physics=False
        ),
        num_samples=n_samples
    )
    
    samples = predictive(data['t'], data['G_obs'], data['I_obs'],
                        data['Gb'], data['Ib'], pinn_net, fix_p3)
    
    # Analyze parameter posteriors
    p1_samples = samples['p1'].detach().numpy()
    p2_samples = samples['p2'].detach().numpy()
    
    print("\nParameter Posterior Statistics:")
    print(f"p1: mean={p1_samples.mean():.4f}, std={p1_samples.std():.4f}, "
          f"95% CI=[{np.percentile(p1_samples, 2.5):.4f}, {np.percentile(p1_samples, 97.5):.4f}]")
    print(f"p2: mean={p2_samples.mean():.4f}, std={p2_samples.std():.4f}, "
          f"95% CI=[{np.percentile(p2_samples, 2.5):.4f}, {np.percentile(p2_samples, 97.5):.4f}]")
    
    if not fix_p3:
        p3_samples = torch.exp(samples['log_p3']).detach().numpy()
        print(f"p3: mean={p3_samples.mean():.6f}, std={p3_samples.std():.6f}, "
              f"95% CI=[{np.percentile(p3_samples, 2.5):.6f}, {np.percentile(p3_samples, 97.5):.6f}]")
    else:
        print(f"p3: FIXED at 1e-5 (not inferred)")
    
    sigma_G_samples = samples['sigma_G'].detach().numpy()
    sigma_phys_samples = samples['sigma_phys'].detach().numpy()
    
    print(f"\nsigma_G: mean={sigma_G_samples.mean():.4f}, std={sigma_G_samples.std():.4f}")
    print(f"sigma_phys: mean={sigma_phys_samples.mean():.4f}, std={sigma_phys_samples.std():.4f}")
    
    return samples


def plot_results(pinn_net, data, losses, samples=None):
    """
    Visualize training and results
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Training loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(losses)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('ELBO Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions
    with torch.no_grad():
        out = pinn_net(data['t'])
        G_pred = out[:, 0].numpy()
        X_pred = out[:, 1].numpy()
    
    t_raw = data['t_raw'].numpy().flatten()
    G_obs_raw = data['G_obs_raw'].numpy()
    
    # Denormalize predictions
    norm = data['normalization']
    G_pred_denorm = G_pred * norm['G_std'] + norm['G_mean']
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t_raw, G_obs_raw, 'o', label='Observed', alpha=0.6)
    ax2.plot(t_raw, G_pred_denorm, '-', label='Predicted', linewidth=2)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Glucose (mg/dL)')
    ax2.set_title('Glucose Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: X(t) trajectory
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t_raw, X_pred, '-', linewidth=2)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('X(t) (normalized)')
    ax3.set_title('Insulin Action X(t)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: Parameter posteriors
    if samples is not None:
        p1_samples = samples['p1'].detach().numpy()
        p2_samples = samples['p2'].detach().numpy()
        
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(p1_samples, bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(p1_samples.mean(), color='r', linestyle='--', label=f'Mean: {p1_samples.mean():.4f}')
        ax4.set_xlabel('p1')
        ax4.set_ylabel('Density')
        ax4.set_title('Posterior of p1')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(p2_samples, bins=50, alpha=0.7, edgecolor='black')
        ax5.axvline(p2_samples.mean(), color='r', linestyle='--', label=f'Mean: {p2_samples.mean():.4f}')
        ax5.set_xlabel('p2')
        ax5.set_ylabel('Density')
        ax5.set_title('Posterior of p2')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(2, 3, 6)
        if 'log_p3' in samples:
            p3_samples = torch.exp(samples['log_p3']).detach().numpy()
            ax6.hist(p3_samples, bins=50, alpha=0.7, edgecolor='black')
            ax6.axvline(p3_samples.mean(), color='r', linestyle='--', label=f'Mean: {p3_samples.mean():.6f}')
            ax6.set_xlabel('p3')
            ax6.set_ylabel('Density')
            ax6.set_title('Posterior of p3')
        else:
            ax6.text(0.5, 0.5, 'p3 FIXED\nat 1e-5', 
                    ha='center', va='center', fontsize=14, transform=ax6.transAxes)
            ax6.set_title('p3 (Fixed Parameter)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    # ========================================================================
    # GOOGLE COLAB SETUP (uncomment if using Colab)
    # ========================================================================
    # from google.colab import drive
    # drive.mount('/content/drive')
    # SAVE_DIR = '/content/drive/MyDrive/BayesianPINN_Checkpoints'
    # FILEPATH = '/content/drive/MyDrive/1001_0_20210730.xlsx'
    
    # ========================================================================
    # LOCAL SETUP (default)
    # ========================================================================
    SAVE_DIR = './checkpoints'
    FILEPATH = '1001_0_20210730.xlsx'
    
    # Configuration
    N_ITERATIONS = 5000
    LEARNING_RATE = 0.001
    CHECKPOINT_FREQ = 500  # Save every 500 iterations
    
    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")
        print("\n⚠️  IMPORTANT: Make sure to mount Google Drive!")
        print("Uncomment the drive.mount() lines in the main() function")
        print("and set your file paths accordingly.\n")
    except:
        IN_COLAB = False
        print("Running locally")
    
    # Step 1: Load and prepare data
    print("="*70)
    print("BAYESIAN PINN FOR BERGMAN MINIMAL MODEL")
    print("="*70)
    
    data = load_and_prepare_data(
        FILEPATH,
        tau=30.0,
        bioavailability=0.6,
        V_d=12.0
    )
    
    # Determine whether to fix p3 based on data
    fix_p3 = data['fix_p3_recommended']
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Fix p3: {fix_p3}")
    print(f"  Checkpoint frequency: {CHECKPOINT_FREQ}")
    print(f"  Save directory: {SAVE_DIR}")
    
    # Step 2: Train model
    pinn_net, losses = train_bayesian_pinn(
        data,
        n_iterations=N_ITERATIONS,
        lr=LEARNING_RATE,
        fix_p3=fix_p3,
        save_dir=SAVE_DIR,
        checkpoint_freq=CHECKPOINT_FREQ
    )
    
    # Step 3: Analyze results
    samples = analyze_results(pinn_net, data, n_samples=1000, fix_p3=fix_p3)
    
    # Step 4: Plot results
    fig = plot_results(pinn_net, data, losses, samples)
    
    # Save figure
    import os
    fig_path = os.path.join(SAVE_DIR, 'bayesian_pinn_results.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nResults figure saved to: {fig_path}")
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {SAVE_DIR}")
    print("Files saved:")
    print(f"  - best_model.pt (best checkpoint)")
    print(f"  - final_model.pt (final model)")
    print(f"  - checkpoint_iter_*.pt (periodic checkpoints)")
    print(f"  - training_losses.csv (loss history)")
    print(f"  - bayesian_pinn_results.png (visualization)")


# ============================================================================
# UTILITY FUNCTION FOR COLAB
# ============================================================================

def resume_training_from_checkpoint(checkpoint_path, data, additional_iterations=1000,
                                   lr=0.001, save_dir='./checkpoints', checkpoint_freq=500):
    """
    Resume training from a saved checkpoint (useful for Colab interruptions)
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file to resume from
    data : dict
        Preprocessed data dictionary
    additional_iterations : int
        How many more iterations to train
    lr : float
        Learning rate (can be different from original)
    save_dir : str
        Directory to save new checkpoints
    checkpoint_freq : int
        Checkpoint frequency
    
    Returns:
    --------
    pinn_net : PINN
        Trained network
    all_losses : list
        Combined loss history
    """
    print("="*70)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("="*70)
    
    # Load checkpoint
    pinn_net, checkpoint = load_checkpoint(checkpoint_path)
    
    # Get previous losses and config
    previous_losses = checkpoint['losses']
    start_iteration = checkpoint['iteration']
    fix_p3 = checkpoint['config']['fix_p3']
    
    print(f"\nResuming from iteration {start_iteration}")
    print(f"Previous best loss: {min(previous_losses):.2e}")
    print(f"Training for {additional_iterations} more iterations...")
    
    # Setup SVI with loaded parameters
    optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Extract data
    t = data['t']
    G_obs = data['G_obs']
    I_obs = data['I_obs']
    Gb = data['Gb']
    Ib = data['Ib']
    
    # Continue training
    new_losses = []
    best_loss = min(previous_losses)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for iteration in range(additional_iterations):
        loss = svi.step(t, G_obs, I_obs, Gb, Ib, pinn_net, fix_p3, 1e-5, True)  # compute_physics=True
        new_losses.append(loss)
        
        current_iteration = start_iteration + iteration + 1
        
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Iteration {current_iteration:5d} | ELBO Loss: {loss:.2e}")
        
        # Save checkpoint
        if (iteration + 1) % checkpoint_freq == 0 or iteration == additional_iterations - 1:
            all_losses = previous_losses + new_losses
            checkpoint = {
                'iteration': current_iteration,
                'loss': loss,
                'losses': all_losses,
                'pinn_state_dict': pinn_net.state_dict(),
                'pyro_param_store': pyro.get_param_store().get_state(),
                'config': {
                    'n_iterations': current_iteration,
                    'lr': lr,
                    'fix_p3': fix_p3,
                    'Gb': Gb,
                    'Ib': Ib
                }
            }
            
            checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{current_iteration}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
            
            if loss < best_loss:
                best_loss = loss
                best_path = os.path.join(save_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"  → Best model updated: {best_path}")
    
    all_losses = previous_losses + new_losses
    
    # Save updated losses
    losses_df = pd.DataFrame({
        'iteration': range(1, len(all_losses) + 1),
        'loss': all_losses
    })
    losses_csv_path = os.path.join(save_dir, 'training_losses.csv')
    losses_df.to_csv(losses_csv_path, index=False)
    
    print(f"\nResumed training completed!")
    print(f"Total iterations: {len(all_losses)}")
    
    return pinn_net, all_losses


if __name__ == "__main__":
    main()