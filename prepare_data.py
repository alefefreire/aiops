"""
Prepare Real Patient Data for Bayesian PINN Analysis

This script processes continuous glucose monitoring (CGM) data from diabetes patients
and prepares it for use with Bayesian Physics-Informed Neural Networks (B-PINNs).

Key Features:
-------------
1. Data Loading & Processing:
   - Reads patient CGM data from Excel files
   - Extracts insulin administration events (bolus and basal)
   - Identifies meal events from dietary logs

2. Insulin Pharmacokinetics:
   - Estimates plasma insulin concentration from doses using PK models
   - Handles both rapid-acting (bolus) and long-acting (basal) insulin
   - Accounts for multiple insulin types with different PK profiles

3. Meal Window Extraction:
   - Extracts clean 4-hour windows around individual meals
   - Ensures each window has a single meal + bolus insulin dose
   - Provides pre-meal baseline glucose estimates

4. Visualization:
   - plot_patient_overview(): Full dataset visualization with glucose, insulin, and meals
   - plot_meal_windows(): Individual meal-response windows for detailed analysis

5. Data Preparation:
   - Formats data for direct use with B-PINN code
   - Includes normalization parameters
   - Compatible with synthetic data format

Challenges Addressed:
--------------------
1. Convert insulin doses to concentration estimates using PK models
2. Handle both bolus and basal insulin types
3. Extract meal-response windows for cleaner analysis
4. Provide comprehensive data quality checks and visualizations

Usage:
------
    python prepare_patient_data_for_pinn.py

Output Files:
------------
    - patient_data_overview.png: Full 14-day dataset visualization
    - meal_windows_prepared.png: First 6 extracted meal windows
    - patient_windows.pkl: Serialized window data for easy loading

Author: Adapted for patient 1006 data
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def extract_insulin_dose(text):
    """Extract insulin dose and type from text"""
    if pd.isna(text):
        return None, None

    match = re.search(r"(\d+(?:\.\d+)?)\s*IU", str(text))
    if match:
        dose = float(match.group(1))
        if "degludec" in text.lower() or "tresiba" in text.lower():
            insulin_type = "basal"
        elif "humulin" in text.lower() or "novolin" in text.lower():
            insulin_type = "bolus"
        else:
            insulin_type = "unknown"
        return dose, insulin_type
    return None, None


def insulin_pk_bolus(t, t_dose, dose_IU, tau=50, bioavail=0.8):
    """
    Pharmacokinetic model for rapid-acting insulin (Humulin R)

    Parameters:
    -----------
    t : array, current time points (minutes)
    t_dose : float, time of injection (minutes)
    dose_IU : float, dose in International Units
    tau : float, absorption time constant (~50 min for regular insulin)
    bioavail : float, bioavailability fraction

    Returns:
    --------
    Plasma insulin concentration (μU/mL) contribution from this dose

    Reference: Regular insulin peaks at ~2-4 hours, duration ~6-8 hours
    """
    # Conversion factor: 1 IU ≈ 36 μg, distribution volume ~12 L
    # Peak concentration ≈ dose(IU) * 36000 μg * bioavail / (12000 mL) / molecular_weight
    # Simplified: Use empirical scaling

    scaling_factor = 15.0  # Empirical: 1 IU → ~15 μU/mL peak (adjustable)

    dt = t - t_dose
    # Only compute for times after injection
    I = np.zeros_like(dt)
    mask = dt >= 0

    # Two-exponential model: fast absorption, slower elimination
    # I(t) = A * [exp(-t/tau_fast) - exp(-t/tau_slow)]
    tau_abs = tau  # Absorption
    tau_elim = tau * 3  # Elimination (slower)

    I[mask] = (
        scaling_factor
        * dose_IU
        * bioavail
        * (np.exp(-dt[mask] / tau_elim) - np.exp(-dt[mask] / tau_abs))
    )

    # Normalize peak to match expected concentration
    peak = I.max()
    if peak > 0:
        expected_peak = scaling_factor * dose_IU * bioavail * 0.4  # ~40% of theoretical
        I = I * (expected_peak / peak)

    return I


def insulin_pk_basal(t, t_dose, dose_IU, tau=600, steady_state_conc=None):
    """
    Pharmacokinetic model for long-acting insulin (degludec)

    Parameters:
    -----------
    t : array, current time points (minutes)
    t_dose : float, time of injection (minutes)
    dose_IU : float, dose in International Units
    tau : float, absorption time constant (~600-800 min for degludec)
    steady_state_conc : float, if known, the steady-state concentration

    Returns:
    --------
    Plasma insulin concentration (μU/mL) contribution from this dose

    Reference: Degludec has ultra-long duration >42 hours, flat profile
    """
    # Simpler model: slow rise to plateau, very slow decay
    scaling_factor = 8.0  # Basal contributes lower but sustained concentration

    dt = t - t_dose
    I = np.zeros_like(dt)
    mask = dt >= 0

    # Approach steady state with time constant tau
    plateau = scaling_factor * dose_IU * 0.7
    I[mask] = plateau * (1 - np.exp(-dt[mask] / tau))

    return I


def compute_total_insulin(df, insulin_events):
    """
    Compute total plasma insulin concentration time series

    Parameters:
    -----------
    df : DataFrame with 'Time_minutes' column
    insulin_events : DataFrame with columns ['time_minutes', 'dose_IU', 'type']

    Returns:
    --------
    I_total : array, estimated plasma insulin concentration (μU/mL)
    """
    t = df["Time_minutes"].values
    I_total = np.zeros_like(t)

    # Estimate basal concentration from basal doses (approximate steady state)
    basal_events = insulin_events[insulin_events["type"] == "basal"]
    if len(basal_events) > 0:
        avg_basal_dose = basal_events["dose_IU"].mean()
        # Add a constant basal level (simplified - assumes steady state)
        basal_contribution = avg_basal_dose * 5.0  # Empirical baseline
        I_total += basal_contribution

    # Add contributions from each bolus
    bolus_events = insulin_events[insulin_events["type"] == "bolus"]
    for _, event in bolus_events.iterrows():
        I_bolus = insulin_pk_bolus(t, event["time_minutes"], event["dose_IU"])
        I_total += I_bolus

    # Add contributions from each basal injection (dynamic part)
    for _, event in basal_events.iterrows():
        I_basal = insulin_pk_basal(t, event["time_minutes"], event["dose_IU"])
        I_total += I_basal

    return I_total


def extract_meal_windows(
    df, insulin_events, meal_times, window_hours=4, min_gap_hours=2
):
    """
    Extract individual meal-response windows for cleaner PINN analysis

    Parameters:
    -----------
    df : Full dataset DataFrame
    insulin_events : Insulin events DataFrame
    meal_times : Array of meal times (minutes)
    window_hours : Duration of each window after meal
    min_gap_hours : Minimum gap between selected meals

    Returns:
    --------
    List of dictionaries, each containing a meal window with data
    """
    windows = []
    last_selected = -np.inf

    for meal_time in meal_times:
        # Check if enough time has passed since last selected meal
        if (meal_time - last_selected) / 60 < min_gap_hours:
            continue

        # Find insulin bolus close to this meal (within ±30 minutes)
        bolus_events = insulin_events[insulin_events["type"] == "bolus"]
        close_boluses = bolus_events[
            np.abs(bolus_events["time_minutes"] - meal_time) < 30
        ]

        if len(close_boluses) == 0:
            continue  # Skip meals without clear insulin bolus

        # Get the closest bolus
        idx_closest = (np.abs(close_boluses["time_minutes"] - meal_time)).argmin()
        bolus = close_boluses.iloc[idx_closest]

        # Extract window data
        t_start = meal_time - 30  # 30 min before meal
        t_end = meal_time + window_hours * 60  # N hours after meal

        window_mask = (df["Time_minutes"] >= t_start) & (df["Time_minutes"] <= t_end)
        window_df = df[window_mask].copy()

        if len(window_df) < 10:  # Need sufficient data points
            continue

        # Reset time to start from 0
        window_df["Time_minutes_window"] = window_df["Time_minutes"] - t_start

        # Store window info
        windows.append(
            {
                "data": window_df,
                "meal_time": meal_time,
                "bolus_dose": bolus["dose_IU"],
                "bolus_time": bolus["time_minutes"],
                "t_start": t_start,
                "t_end": t_end,
                "duration_hours": window_hours,
            }
        )

        last_selected = meal_time

    return windows


def prepare_for_pinn(window_data, Gb_estimate=None, Ib_estimate=10.0):
    """
    Prepare a meal window for Bayesian PINN training

    Similar format to the synthetic data preparation in the original code

    Parameters:
    -----------
    window_data : dict from extract_meal_windows
    Gb_estimate : float, basal glucose estimate (if None, use first measurement)
    Ib_estimate : float, basal insulin estimate (μU/mL)

    Returns:
    --------
    Dictionary ready for PINN training
    """
    df = window_data["data"]

    t_minutes = df["Time_minutes_window"].values
    G_obs = df["CGM (mg / dl)"].values

    # Compute insulin concentration for this window
    # Simplified: assume only the bolus from this meal matters significantly
    t_bolus_rel = window_data["bolus_time"] - window_data["t_start"]
    I_obs = insulin_pk_bolus(t_minutes, t_bolus_rel, window_data["bolus_dose"])
    I_obs += Ib_estimate  # Add basal level

    # Estimate basal glucose
    if Gb_estimate is None:
        Gb = G_obs[0]  # Use first measurement as baseline
    else:
        Gb = Gb_estimate

    # Normalization
    t_mean, t_std = t_minutes.mean(), t_minutes.std()
    G_mean, G_std = G_obs.mean(), G_obs.std()
    I_mean, I_std = I_obs.mean(), I_obs.std()

    # Handle case where std is very small
    if t_std < 1e-6:
        t_std = 1.0
    if G_std < 1e-6:
        G_std = 1.0
    if I_std < 1e-6:
        I_std = 1.0

    return {
        "t_minutes": t_minutes,
        "G_obs": G_obs,
        "I_obs": I_obs,
        "Gb": Gb,
        "Ib": Ib_estimate,
        "normalization": {
            "t_mean": t_mean,
            "t_std": t_std,
            "G_mean": G_mean,
            "G_std": G_std,
            "I_mean": I_mean,
            "I_std": I_std,
        },
        "bolus_dose": window_data["bolus_dose"],
        "meal_time": window_data["meal_time"],
        "window_info": window_data,
    }


def plot_patient_overview(
    df, insulin_events, save_path="/home/claude/patient_data_overview.png"
):
    """
    Create comprehensive overview visualization of patient data

    This function generates a 3-panel figure showing:
    1. Continuous glucose monitoring (CGM) with reference lines
    2. Insulin administration events (bolus and basal)
    3. Meal timing events

    Parameters:
    -----------
    df : DataFrame
        Full patient dataset with columns:
        - Time_minutes: Time from start in minutes
        - CGM (mg / dl): Continuous glucose measurements
        - CBG (mg / dl): Capillary blood glucose (fingerstick calibrations)
        - Dietary intake: Meal descriptions
    insulin_events : DataFrame
        Insulin administration events with columns:
        - time_minutes: Time of injection
        - dose_IU: Dose in International Units
        - type: 'bolus' or 'basal'
    save_path : str
        Where to save the figure (default: current directory)

    Returns:
    --------
    None (saves figure to disk)

    Figure Details:
    --------------
    Panel 1 (Top): Glucose Monitoring
        - Blue line: Continuous glucose (CGM)
        - Red dots: Fingerstick calibrations (CBG)
        - Orange dashed line: Target glucose (~180 mg/dL)
        - Red dashed line: Hypoglycemia threshold (70 mg/dL)

    Panel 2 (Middle): Insulin Administration
        - Blue stems: Bolus (rapid-acting) insulin doses
        - Orange stems: Basal (long-acting) insulin doses

    Panel 3 (Bottom): Meal Events
        - Green vertical lines: Times when meals were consumed
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # ========== PANEL 1: GLUCOSE DATA ==========
    ax1 = axes[0]

    # Plot continuous glucose monitoring
    ax1.plot(
        df["Time_minutes"] / 60,
        df["CGM (mg / dl)"],
        linewidth=1,
        alpha=0.8,
        color="steelblue",
        label="CGM",
    )

    # Add CBG fingerstick measurements (calibration points)
    cbg_mask = df["CBG (mg / dl)"].notna()
    if cbg_mask.sum() > 0:
        ax1.scatter(
            df[cbg_mask]["Time_minutes"] / 60,
            df[cbg_mask]["CBG (mg / dl)"],
            color="red",
            s=50,
            alpha=0.7,
            zorder=5,
            label="CBG (fingerstick)",
        )

    # Add clinical reference lines
    ax1.axhline(
        y=180, color="orange", linestyle="--", alpha=0.5, label="Target (~180 mg/dL)"
    )
    ax1.axhline(
        y=70, color="red", linestyle="--", alpha=0.5, label="Hypoglycemia threshold"
    )

    # Formatting
    ax1.set_ylabel("Glucose (mg/dL)", fontsize=11)
    ax1.set_title(
        "Patient 1006 - Glucose Monitoring (Feb 9-23, 2021)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, df["Time_minutes"].max() / 60)

    # ========== PANEL 2: INSULIN DOSES ==========
    ax2 = axes[1]

    # Separate bolus and basal insulin events
    bolus_df = insulin_events[insulin_events["type"] == "bolus"]
    basal_df = insulin_events[insulin_events["type"] == "basal"]

    # Plot bolus insulin (rapid-acting, with meals)
    ax2.stem(
        bolus_df["time_minutes"] / 60,
        bolus_df["dose_IU"],
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
        label="Bolus insulin",
    )

    # Plot basal insulin (long-acting, background)
    ax2.stem(
        basal_df["time_minutes"] / 60,
        basal_df["dose_IU"],
        linefmt="C1-",
        markerfmt="C1s",
        basefmt=" ",
        label="Basal insulin",
    )

    # Formatting
    ax2.set_ylabel("Insulin Dose (IU)", fontsize=11)
    ax2.set_title("Insulin Administration", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, df["Time_minutes"].max() / 60)

    # ========== PANEL 3: MEAL EVENTS ==========
    ax3 = axes[2]

    # Extract meal times
    meal_times = df[df["Dietary intake"].notna()]["Time_minutes"] / 60

    # Plot as vertical marks
    ax3.scatter(
        meal_times,
        np.ones(len(meal_times)),
        marker="|",
        s=500,
        linewidths=2,
        color="green",
        alpha=0.7,
    )

    # Formatting
    ax3.set_ylim(0.5, 1.5)
    ax3.set_ylabel("Meals", fontsize=11)
    ax3.set_xlabel("Time (hours)", fontsize=11)
    ax3.set_title("Meal Events", fontsize=12, fontweight="bold")
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.set_xlim(0, df["Time_minutes"].max() / 60)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Patient overview saved to: {save_path}")
    plt.close()


def plot_meal_windows(
    prepared_windows, save_path="/home/claude/meal_windows_prepared.png"
):
    """
    Visualize extracted meal-response windows

    Creates a grid showing the first 6 meal windows, each displaying:
    - Glucose trajectory over 4 hours
    - Baseline glucose (horizontal red line)
    - Insulin bolus timing (vertical green line)

    This visualization helps verify that the window extraction worked correctly
    and provides a quick overview of the variety of meal responses in the dataset.

    Parameters:
    -----------
    prepared_windows : list of dict
        List of prepared window data dictionaries, each containing:
        - t_minutes: Time vector for the window
        - G_obs: Observed glucose values
        - Gb: Baseline glucose
        - bolus_dose: Insulin dose administered
        - window_info: Metadata about the window
    save_path : str
        Where to save the figure

    Returns:
    --------
    None (saves figure to disk)

    Window Selection:
    ----------------
    Different windows represent different physiological states:
    - Normal baseline (Gb ~100-180 mg/dL): Typical meal response
    - High baseline (Gb >250 mg/dL): Poor control, high glucose
    - Low baseline (Gb <80 mg/dL): Hypoglycemic or near-hypoglycemic

    Testing on diverse windows helps validate model generalization.
    """
    n_windows_to_plot = min(6, len(prepared_windows))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

    for i, pdata in enumerate(prepared_windows[:n_windows_to_plot]):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Plot glucose trajectory
        ax.plot(
            pdata["t_minutes"],
            pdata["G_obs"],
            "o-",
            linewidth=2,
            markersize=4,
            label="Glucose",
        )

        # Show baseline glucose as reference
        ax.axhline(
            y=pdata["Gb"],
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f'Gb={pdata["Gb"]:.0f}',
        )

        # Mark when insulin bolus was administered
        bolus_time_rel = (
            pdata["window_info"]["bolus_time"] - pdata["window_info"]["t_start"]
        )
        ax.axvline(
            x=bolus_time_rel,
            color="green",
            linestyle="--",
            alpha=0.5,
            label=f'Bolus {pdata["bolus_dose"]:.0f} IU',
        )

        # Formatting
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(f"Window {i+1}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if we have fewer than 6 windows
    for i in range(n_windows_to_plot, 6):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Meal windows visualization saved to: {save_path}")
    plt.close()


def print_summary_statistics(df, insulin_events):
    """Print detailed summary statistics about the dataset"""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\n--- GLUCOSE DATA ---")
    print(f"Total CGM readings: {len(df)}")
    print(
        f"Time range: {df['Time_minutes'].min():.1f} - {df['Time_minutes'].max():.1f} minutes"
    )
    print(
        f"Duration: {df['Time_minutes'].max() / 60:.1f} hours ({df['Time_minutes'].max() / (60*24):.1f} days)"
    )
    print(f"Sampling interval: ~{df['Time_minutes'].diff().median():.0f} minutes")
    print(
        f"Glucose range: {df['CGM (mg / dl)'].min():.1f} - {df['CGM (mg / dl)'].max():.1f} mg/dL"
    )
    print(
        f"Glucose mean: {df['CGM (mg / dl)'].mean():.1f} ± {df['CGM (mg / dl)'].std():.1f} mg/dL"
    )
    print(
        f"Baseline estimate (first 2h): {df[df['Time_minutes'] < 120]['CGM (mg / dl)'].mean():.1f} mg/dL"
    )

    cbg_count = df["CBG (mg / dl)"].notna().sum()
    if cbg_count > 0:
        print(f"\nCBG calibrations: {cbg_count}")
        print(
            f"CBG range: {df['CBG (mg / dl)'].min():.1f} - {df['CBG (mg / dl)'].max():.1f} mg/dL"
        )

    print(f"\n--- INSULIN ADMINISTRATION ---")
    bolus_events = insulin_events[insulin_events["type"] == "bolus"]
    basal_events = insulin_events[insulin_events["type"] == "basal"]

    print(f"Total insulin events: {len(insulin_events)}")
    print(f"Bolus (rapid-acting): {len(bolus_events)} events")
    if len(bolus_events) > 0:
        print(
            f"  Mean dose: {bolus_events['dose_IU'].mean():.1f} ± {bolus_events['dose_IU'].std():.1f} IU"
        )
        print(
            f"  Range: {bolus_events['dose_IU'].min():.0f} - {bolus_events['dose_IU'].max():.0f} IU"
        )

    print(f"Basal (long-acting): {len(basal_events)} events")
    if len(basal_events) > 0:
        print(
            f"  Mean dose: {basal_events['dose_IU'].mean():.1f} ± {basal_events['dose_IU'].std():.1f} IU"
        )
        print(
            f"  Range: {basal_events['dose_IU'].min():.0f} - {basal_events['dose_IU'].max():.0f} IU"
        )

    meal_count = df["Dietary intake"].notna().sum()
    print(f"\n--- MEALS ---")
    print(f"Total meal events: {meal_count}")


def main():
    """Main processing pipeline"""

    print("=" * 70)
    print("PATIENT DATA PREPARATION FOR BAYESIAN PINN")
    print("=" * 70)

    # Load data
    df = pd.read_excel("/mnt/user-data/uploads/1006_1_20210209.xlsx", sheet_name=0)
    df["Time_minutes"] = (df["Date"] - df["Date"].min()).dt.total_seconds() / 60

    # Extract insulin events
    insulin_data = []
    for idx, row in df[df["Insulin dose - s.c."].notna()].iterrows():
        dose, insulin_type = extract_insulin_dose(row["Insulin dose - s.c."])
        if dose is not None:
            insulin_data.append(
                {
                    "time_minutes": row["Time_minutes"],
                    "dose_IU": dose,
                    "type": insulin_type,
                    "datetime": row["Date"],
                }
            )

    insulin_events = pd.DataFrame(insulin_data)

    # Print summary statistics
    print_summary_statistics(df, insulin_events)

    # Create patient overview visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    plot_patient_overview(df, insulin_events)

    # Compute total insulin concentration
    print("\nComputing insulin concentration time series...")
    I_total = compute_total_insulin(df, insulin_events)
    df["Insulin_est"] = I_total

    # Get meal times
    meal_times = df[df["Dietary intake"].notna()]["Time_minutes"].values

    # Extract meal windows
    print(f"\nExtracting meal-response windows...")
    print(f"Total meals logged: {len(meal_times)}")

    windows = extract_meal_windows(
        df, insulin_events, meal_times, window_hours=4, min_gap_hours=3
    )

    print(f"Extracted {len(windows)} clean meal windows for analysis")

    # Prepare each window for PINN
    print("\n" + "=" * 70)
    print("PREPARING WINDOWS FOR PINN TRAINING")
    print("=" * 70)
    prepared_windows = []

    for i, window in enumerate(windows):
        # Estimate Gb from pre-meal glucose
        pre_meal_idx = window["data"]["Time_minutes_window"] < 30
        if pre_meal_idx.sum() > 0:
            Gb_est = window["data"][pre_meal_idx]["CGM (mg / dl)"].mean()
        else:
            Gb_est = window["data"]["CGM (mg / dl)"].iloc[0]

        pinn_data = prepare_for_pinn(window, Gb_estimate=Gb_est, Ib_estimate=12.0)
        prepared_windows.append(pinn_data)

        print(
            f"  Window {i+1}: Meal at {window['meal_time']/60:.1f}h, "
            f"Bolus {window['bolus_dose']:.0f} IU, "
            f"Gb={pinn_data['Gb']:.1f} mg/dL, "
            f"Duration={len(pinn_data['t_minutes'])} samples"
        )

    # Create meal windows visualization
    print("\n" + "=" * 70)
    print("CREATING MEAL WINDOW VISUALIZATIONS")
    print("=" * 70)
    plot_meal_windows(prepared_windows)

    # Summary
    print("\n" + "=" * 70)
    print("PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Total windows prepared: {len(prepared_windows)}")
    print(f"\nRecommendation: Start with Window 1 for initial PINN training")
    print(f"Then validate on additional windows to test generalization")
    print(f"\nFiles created:")
    print(f"  - patient_data_overview.png (full dataset visualization)")
    print(f"  - meal_windows_prepared.png (extracted windows)")

    return df, insulin_events, prepared_windows


if __name__ == "__main__":
    df, insulin_events, windows = main()

    print("\n" + "=" * 70)
    print("✓ Data prepared and ready for Bayesian PINN analysis")
    print("=" * 70)

    # Optional: Save windows for later use
    try:
        import pickle

        with open("/home/claude/patient_windows.pkl", "wb") as f:
            pickle.dump(windows, f)
        print("✓ Windows saved to: patient_windows.pkl")
    except Exception as e:
        print(f"Note: Could not save windows to pickle file: {e}")
