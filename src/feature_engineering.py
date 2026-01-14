"""
AeroGuard RUL - Multi-Domain Feature Engineering Module
========================================================
The "X-Factor" for competition-winning performance:
- Temporal Features (Rolling Mean, Std Dev, Skewness)
- Degradation Rate (First-order derivative/gradient)
- Cross-Sensor Interaction (Physics-informed features)
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

# Rolling window size for temporal features
TEMPORAL_WINDOW = 30

# Cross-sensor pairs for interaction features (physics-informed)
# Based on thermodynamic relationships (PV=nRT)
CROSS_SENSOR_PAIRS = [
    ('sensor2', 'sensor4'),   # Total temperature at LPC outlet × Total temperature at HPC outlet
    ('sensor3', 'sensor9'),   # Total temperature at HPC outlet × Physical core speed
    ('sensor7', 'sensor12'),  # Total pressure at HPC outlet × Ratio of fuel flow to Ps30
    ('sensor11', 'sensor15'), # Static pressure at HPC outlet × Bleed enthalpy ratio
]


def calculate_rolling_features(df, feature_cols, window=TEMPORAL_WINDOW):
    """
    Calculate rolling/temporal features over a moving window.
    
    These features tell the model not just WHAT the value is, but 
    HOW "shaky" the sensor is becoming - a key indicator of degradation.
    
    Features:
    - Rolling Mean: Average value over window
    - Rolling Std Dev: Volatility/instability measure
    - Rolling Skewness: Distribution asymmetry (early warning of shift)
    
    Args:
        df: DataFrame with sensor data
        feature_cols: List of sensor column names
        window: Rolling window size (default: 30)
    
    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()
    
    for unit in df['unit'].unique():
        unit_mask = df['unit'] == unit
        unit_data = df.loc[unit_mask, feature_cols]
        
        for col in feature_cols:
            col_data = unit_data[col]
            
            # Rolling Mean - average value trend
            df.loc[unit_mask, f'{col}_roll_mean'] = col_data.rolling(
                window=window, min_periods=1, center=False
            ).mean()
            
            # Rolling Std Dev - volatility/shakiness measure
            df.loc[unit_mask, f'{col}_roll_std'] = col_data.rolling(
                window=window, min_periods=1, center=False
            ).std().fillna(0)
            
            # Rolling Skewness - distribution asymmetry
            rolling_skew = col_data.rolling(window=window, min_periods=5, center=False).apply(
                lambda x: skew(x) if len(x) >= 5 else 0, raw=True
            )
            df.loc[unit_mask, f'{col}_roll_skew'] = rolling_skew.fillna(0)
    
    return df


def calculate_degradation_rate(df, feature_cols):
    """
    Calculate degradation rate (Lag Difference: Current - Previous).
    
    A temperature that is 500° but RISING by 5°/cycle is far more 
    dangerous than one steady at 550°. This captures the VELOCITY of failure.
    
    Args:
        df: DataFrame with sensor data
        feature_cols: List of sensor column names
    
    Returns:
        DataFrame with added diff features
    """
    df = df.copy()
    
    for unit in df['unit'].unique():
        unit_mask = df['unit'] == unit
        
        # Calculate Lag Difference (Current - Previous)
        # We use fillna(0) for the first cycle
        diff_features = df.loc[unit_mask, feature_cols].diff().fillna(0)
        
        # Rename columns to indicate they are diffs
        diff_features.columns = [f'{col}_diff' for col in feature_cols]
        
        # Assign back to DataFrame
        df.loc[unit_mask, diff_features.columns] = diff_features
    
    return df


def calculate_cross_sensor_features(df, sensor_pairs=CROSS_SENSOR_PAIRS):
    """
    Calculate cross-sensor interaction features (Fused Sensors).
    
    Based on thermodynamic relationships (PV=nRT), these "Fused Sensors"
    often reveal "hidden" stress that a single sensor misses.
    
    Example: Temperature × Pressure can reveal compressor surge risk
    that neither sensor alone would indicate (early stage wear).
    
    Args:
        df: DataFrame with sensor data
        sensor_pairs: List of tuples of sensor column pairs
    
    Returns:
        DataFrame with added interaction features
    """
    df = df.copy()
    
    for s1, s2 in sensor_pairs:
        if s1 in df.columns and s2 in df.columns:
            # Multiplicative interaction
            df[f'{s1}_x_{s2}'] = df[s1] * df[s2]
            
            # Ratio interaction (with small epsilon to avoid division by zero)
            eps = 1e-8
            df[f'{s1}_div_{s2}'] = df[s1] / (df[s2] + eps)
    
    return df


def calculate_exponential_smoothing(df, feature_cols, alpha=0.3):
    """
    Apply Exponential Weighted Moving Average (EWMA) for trend detection.
    
    EWMA gives more weight to recent observations, making it better
    at detecting recent degradation trends.
    
    Args:
        df: DataFrame with sensor data
        feature_cols: List of sensor column names
        alpha: Smoothing factor (0-1), higher = more weight to recent
    
    Returns:
        DataFrame with added EWMA features
    """
    df = df.copy()
    
    for unit in df['unit'].unique():
        unit_mask = df['unit'] == unit
        unit_data = df.loc[unit_mask, feature_cols]
        
        for col in feature_cols:
            ewma = unit_data[col].ewm(alpha=alpha, adjust=False).mean()
            df.loc[unit_mask, f'{col}_ewma'] = ewma
    
    return df


def extract_all_features(df, verbose=True):
    """
    Complete feature extraction pipeline.
    
    Applies all three types of features:
    1. Temporal (Rolling statistics)
    2. Degradation Rate (Gradient)
    3. Cross-Sensor Interaction
    
    Args:
        df: DataFrame with preprocessed sensor data
        verbose: Print progress messages
    
    Returns:
        DataFrame with all engineered features
    """
    # Get base feature columns (sensors and operational settings)
    base_features = [col for col in df.columns 
                     if col.startswith('sensor') or col.startswith('op')]
    
    if verbose:
        print("\n" + "=" * 60)
        print("Feature Engineering Pipeline")
        print("=" * 60)
        print(f"Base features: {len(base_features)}")
    
    # 1. Temporal Features (Rolling statistics)
    if verbose:
        print(f"\n[1/4] Calculating Rolling Features (window={TEMPORAL_WINDOW})...")
    df = calculate_rolling_features(df, base_features)
    
    # 2. Degradation Rate (Gradient)
    if verbose:
        print("[2/4] Calculating Degradation Rate (gradient)...")
    df = calculate_degradation_rate(df, base_features)
    
    # 3. Cross-Sensor Interaction
    if verbose:
        print("[3/4] Calculating Cross-Sensor Interactions...")
    df = calculate_cross_sensor_features(df)
    
    # 4. EWMA for trend detection
    if verbose:
        print("[4/4] Calculating EWMA features...")
    df = calculate_exponential_smoothing(df, base_features)
    
    # Count new features
    new_features = [col for col in df.columns 
                    if col not in ['unit', 'time', 'RUL'] 
                    and col.startswith(('sensor', 'op'))]
    
    if verbose:
        print(f"\nTotal features after engineering: {len(new_features)}")
        print("=" * 60)
    
    return df


def get_engineered_feature_columns(df):
    """Get list of all engineered feature columns."""
    exclude_cols = ['unit', 'time', 'RUL']
    return [col for col in df.columns if col not in exclude_cols]


if __name__ == "__main__":
    import os
    from preprocessing import load_data, add_rul_labels, drop_constant_sensors
    
    # Load and preprocess data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CMAPSSData')
    train_df, test_df, rul_df = load_data(data_dir)
    train_df = add_rul_labels(train_df)
    train_df = drop_constant_sensors(train_df)
    
    # Apply feature engineering
    train_df_fe = extract_all_features(train_df)
    
    print(f"\n✓ Original columns: {len(train_df.columns)}")
    print(f"✓ After feature engineering: {len(train_df_fe.columns)}")
    print(f"\nSample engineered features:")
    new_cols = [c for c in train_df_fe.columns if c not in train_df.columns]
    for col in new_cols[:10]:
        print(f"  - {col}")
    print(f"  ... and {len(new_cols) - 10} more")
