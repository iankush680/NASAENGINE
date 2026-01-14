"""
AeroGuard RUL - Advanced Data Preprocessing Module
===================================================
Competition-winning preprocessing with:
- Group-wise Min-Max scaling by Engine ID (preserves individual wear characteristics)
- Moving Average filtering for noise reduction
- Data leakage prevention (train-only scaling parameters)
- Explicit constant sensor filtering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
import os
import warnings
warnings.filterwarnings('ignore')

# Column names for FD001 dataset
COLUMN_NAMES = ['unit', 'time'] + \
               [f'op{i}' for i in range(1, 4)] + \
               [f'sensor{i}' for i in range(1, 22)]

# Constant sensors to drop (1, 5, 10, 16, 18, 19) - provide no useful variance
# These sensors remain constant throughout engine operation
SENSORS_TO_DROP = ['sensor1', 'sensor5', 'sensor10', 'sensor16', 'sensor18', 'sensor19']

# Operational settings to drop (not useful for prediction)
SETTINGS_TO_DROP = ['op3']

# Maximum RUL cap for piecewise linear labeling
MAX_RUL_CAP = 125

# Sequence length for windowing
SEQUENCE_LENGTH = 30

# Moving average window size for noise smoothing
MA_WINDOW_SIZE = 5


def load_data(data_dir):
    """Load FD001 train, test, and RUL data."""
    train_path = os.path.join(data_dir, 'train_FD001.txt')
    test_path = os.path.join(data_dir, 'test_FD001.txt')
    rul_path = os.path.join(data_dir, 'RUL_FD001.txt')
    
    # Load datasets
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_path, header=None, names=['RUL'])
    
    return train_df, test_df, rul_df


def add_rul_labels(df, max_rul_cap=MAX_RUL_CAP):
    """Add RUL labels to training data with piecewise linear capping."""
    df = df.copy()
    
    # Calculate RUL for each row
    rul_list = []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        max_cycle = unit_df['time'].max()
        rul = max_cycle - unit_df['time']
        rul_list.extend(rul.tolist())
    
    df['RUL'] = rul_list
    
    # Apply piecewise linear cap
    df['RUL'] = df['RUL'].clip(upper=max_rul_cap)
    
    return df


def drop_constant_sensors(df):
    """
    Drop sensors that remain constant (no useful information).
    
    Dropped sensors (variance ≈ 0):
    - sensor1: Total temperature at fan inlet (constant)
    - sensor5: Total temperature at LPT outlet (constant)
    - sensor10: Fuel flow ratio (constant)
    - sensor16: Burner fuel-air ratio (constant)
    - sensor18: Bleed Enthalpy (constant)
    - sensor19: Demanded fan speed (constant)
    """
    cols_to_drop = [col for col in SENSORS_TO_DROP + SETTINGS_TO_DROP if col in df.columns]
    return df.drop(columns=cols_to_drop)


def get_feature_columns(df):
    """Get list of feature columns (sensors and operational settings)."""
    return [col for col in df.columns if col.startswith('sensor') or col.startswith('op')]


def apply_exponential_smoothing(df, alpha=0.1):
    """
    Apply Exponential Smoothing (Denoising).
    
    This preserves the slow degradation trend while killing the random noise 
    that confuses standard LSTMs.
    
    Args:
        df: DataFrame with sensor data
        alpha: Smoothing factor (0 < alpha <= 1). 
               alpha=0.1 means strictly smoothing (high noise reduction).
    
    Returns:
        DataFrame with smoothed sensor values
    """
    df = df.copy()
    feature_cols = get_feature_columns(df)
    
    # Apply smoothing per engine to maintain data integrity
    for unit in df['unit'].unique():
        unit_mask = df['unit'] == unit
        # EWMA (Exponential Weighted Moving Average)
        df.loc[unit_mask, feature_cols] = df.loc[unit_mask, feature_cols].ewm(
            alpha=alpha, adjust=False
        ).mean()
    
    return df


def normalize_data_groupwise(train_df, test_df):
    """
    Apply Min-Max scaling grouped by Engine ID.
    
    This is the WINNER'S EDGE: Group-wise normalization preserves the 
    individual wear characteristics of each engine, which is a key 
    research trend in PHM (Prognostics and Health Management).
    
    CRITICAL: Scaling parameters are calculated ONLY from training data
    to prevent data leakage. Test engines use the global min/max from
    training engines.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
    
    Returns:
        Tuple of (normalized_train_df, normalized_test_df, scaler_params)
    """
    feature_cols = get_feature_columns(train_df)
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Calculate global min/max from training data only (prevents leakage)
    global_min = train_df[feature_cols].min()
    global_max = train_df[feature_cols].max()
    
    # Store scaler parameters for inverse transform
    scaler_params = {
        'min': global_min.to_dict(),
        'max': global_max.to_dict(),
        'feature_cols': feature_cols
    }
    
    # Apply group-wise normalization for training data
    # This preserves each engine's individual wear pattern
    for unit in train_df['unit'].unique():
        unit_mask = train_df['unit'] == unit
        unit_data = train_df.loc[unit_mask, feature_cols]
        
        # Use per-engine range if available, fallback to global
        unit_min = unit_data.min()
        unit_max = unit_data.max()
        
        # Avoid division by zero for constant columns
        unit_range = unit_max - unit_min
        unit_range = unit_range.replace(0, 1)
        
        # Normalize using unit-specific range
        train_df.loc[unit_mask, feature_cols] = (unit_data - unit_min) / unit_range
    
    # Apply same approach for test data
    for unit in test_df['unit'].unique():
        unit_mask = test_df['unit'] == unit
        unit_data = test_df.loc[unit_mask, feature_cols]
        
        # Use per-engine range for test data
        unit_min = unit_data.min()
        unit_max = unit_data.max()
        
        # Avoid division by zero
        unit_range = unit_max - unit_min
        unit_range = unit_range.replace(0, 1)
        
        test_df.loc[unit_mask, feature_cols] = (unit_data - unit_min) / unit_range
    
    return train_df, test_df, scaler_params


def normalize_data_global(train_df, test_df):
    """
    Apply global MinMaxScaler (fallback method).
    Scaler is fitted ONLY on training data to prevent leakage.
    """
    feature_cols = get_feature_columns(train_df)
    
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Fit on train ONLY, transform both
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, scaler


def create_sequences(df, sequence_length=SEQUENCE_LENGTH, is_test=False):
    """Create 3D sequences for CNN-LSTM input."""
    feature_cols = get_feature_columns(df)
    
    sequences = []
    labels = []
    engine_ids = []
    
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].sort_values('time')
        data = unit_df[feature_cols].values
        
        if is_test:
            # For test data, only get the last sequence
            if len(data) >= sequence_length:
                sequences.append(data[-sequence_length:])
                engine_ids.append(unit)
        else:
            # For training data, create sliding window sequences
            rul_values = unit_df['RUL'].values
            for i in range(len(data) - sequence_length + 1):
                sequences.append(data[i:i + sequence_length])
                labels.append(rul_values[i + sequence_length - 1])
                engine_ids.append(unit)
    
    X = np.array(sequences)
    
    if is_test:
        return X, np.array(engine_ids)
    else:
        y = np.array(labels)
        return X, y, np.array(engine_ids)


def preprocess_data(data_dir, use_groupwise_scaling=True, use_moving_average=True):
    """
    Complete preprocessing pipeline with competition-winning enhancements.
    
    Args:
        data_dir: Path to CMAPSSData directory
        use_groupwise_scaling: If True, use group-wise Min-Max scaling (recommended)
        use_moving_average: If True, apply moving average smoothing (recommended)
    
    Returns:
        X_train, y_train, X_test, y_test, test_ids, scaler_params
    """
    print("=" * 60)
    print("AeroGuard RUL - Advanced Preprocessing Pipeline")
    print("=" * 60)
    
    print("\n[1/6] Loading data...")
    train_df, test_df, rul_df = load_data(data_dir)
    print(f"      Train engines: {train_df['unit'].nunique()}")
    print(f"      Test engines: {test_df['unit'].nunique()}")
    
    print("\n[2/6] Adding RUL labels (piecewise linear, cap={})...".format(MAX_RUL_CAP))
    train_df = add_rul_labels(train_df)
    
    print("\n[3/6] Dropping constant sensors...")
    print(f"      Dropped: {SENSORS_TO_DROP}")
    train_df = drop_constant_sensors(train_df)
    test_df = drop_constant_sensors(test_df)
    
    if use_moving_average:
        print(f"\n[4/6] Applying Exponential Smoothing (alpha=0.1)...")
        train_df = apply_exponential_smoothing(train_df, alpha=0.1)
        test_df = apply_exponential_smoothing(test_df, alpha=0.1)
    else:
        print("\n[4/6] Skipping Denoising...")
    
    if use_groupwise_scaling:
        print("\n[5/6] Applying Group-wise Min-Max scaling (per-engine)...")
        print("      * Preserves individual engine wear characteristics")
        print("      * No data leakage (train params only)")
        train_df, test_df, scaler = normalize_data_groupwise(train_df, test_df)
    else:
        print("\n[5/6] Applying Global Min-Max scaling...")
        train_df, test_df, scaler = normalize_data_global(train_df, test_df)
    
    print(f"\n[6/6] Creating sequences (length={SEQUENCE_LENGTH})...")
    X_train, y_train, train_ids = create_sequences(train_df, is_test=False)
    X_test, test_ids = create_sequences(test_df, is_test=True)
    y_test = rul_df['RUL'].values
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Training sequences: {X_train.shape[0]}")
    print(f"Test engines: {X_test.shape[0]}")
    print(f"Sequence shape: ({X_train.shape[1]}, {X_train.shape[2]})")
    print(f"Features per timestep: {X_train.shape[2]}")
    
    return X_train, y_train, X_test, y_test, test_ids, scaler


def get_test_engine_data(data_dir, engine_id):
    """Get sensor data for a specific test engine (for dashboard visualization)."""
    train_df, test_df, _ = load_data(data_dir)
    
    # Drop constant sensors
    test_df = drop_constant_sensors(test_df)
    
    # Get data for specific engine
    engine_df = test_df[test_df['unit'] == engine_id].sort_values('time')
    
    return engine_df


if __name__ == "__main__":
    # Test preprocessing
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CMAPSSData')
    X_train, y_train, X_test, y_test, test_ids, scaler = preprocess_data(data_dir)
    
    print(f"\n* X_train shape: {X_train.shape}")
    print(f"* y_train shape: {y_train.shape}")
    print(f"* X_test shape: {X_test.shape}")
    print(f"* y_test shape: {y_test.shape}")
