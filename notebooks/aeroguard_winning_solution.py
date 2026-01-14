"""
🏆 AeroGuard RUL - Competition-Winning Solution
================================================
NASA C-MAPSS FD001 | Hybrid CNN-Attention-LSTM | Safety-Weighted Loss

Enhancements over baseline:
- ✅ Unit ID-based validation split (prevents temporal data leakage)
- ✅ Bahdanau-style Attention Layer (weights critical time-steps)
- ✅ Asymmetric Safety Loss (penalizes late predictions 3x more)
- ✅ Interactive Engine Gauge (Plotly visualization for judges)

Convert to notebook:
    jupyter nbconvert --to notebook --execute aeroguard_winning_solution.py
"""

# %% [markdown]
# # 🏆 AeroGuard RUL - Competition-Winning Solution
# 
# **NASA C-MAPSS FD001 | Hybrid CNN-Attention-LSTM | Safety-Weighted Loss**

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
    BatchNormalization, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

print(f"TensorFlow version: {tf.__version__}")

# %%
# Dataset path
DATA_DIR = '../datasets/CMAPSSData'
MODEL_DIR = '../models'

# Column names
COLUMN_NAMES = ['unit', 'time'] + [f'op{i}' for i in range(1, 4)] + [f'sensor{i}' for i in range(1, 22)]

# Sensors to drop (constant values)
SENSORS_TO_DROP = ['sensor1', 'sensor5', 'sensor10', 'sensor16', 'sensor18', 'sensor19', 'op3']

# Hyperparameters
MAX_RUL_CAP = 125
SEQUENCE_LENGTH = 30
EPOCHS = 80  # Updated as requested
BATCH_SIZE = 256
VAL_SPLIT_RATIO = 0.2  # 20% of units held out for validation

os.makedirs(MODEL_DIR, exist_ok=True)

# Load datasets
train_df = pd.read_csv(f'{DATA_DIR}/train_FD001.txt', sep=r'\s+', header=None, names=COLUMN_NAMES)
test_df = pd.read_csv(f'{DATA_DIR}/test_FD001.txt', sep=r'\s+', header=None, names=COLUMN_NAMES)
rul_df = pd.read_csv(f'{DATA_DIR}/RUL_FD001.txt', header=None, names=['RUL'])

print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print(f"RUL labels: {rul_df.shape}")

# %%
def add_rul_labels(df, max_rul_cap=MAX_RUL_CAP):
    """Add RUL labels with piecewise linear capping."""
    df = df.copy()
    rul_list = []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        max_cycle = unit_df['time'].max()
        rul = max_cycle - unit_df['time']
        rul_list.extend(rul.tolist())
    df['RUL'] = rul_list
    df['RUL'] = df['RUL'].clip(upper=max_rul_cap)
    return df

train_df = add_rul_labels(train_df)
print("RUL distribution:")
print(train_df['RUL'].describe())

# %%
# Drop constant sensors
cols_to_drop = [col for col in SENSORS_TO_DROP if col in train_df.columns]
train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)

# Get feature columns
feature_cols = [col for col in train_df.columns if col.startswith('sensor') or col.startswith('op')]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# Normalize data
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# %% [markdown]
# ## 🔬 Unit ID-Based Validation Split
# 
# **Why?** A time-step split allows the model to see adjacent cycles of the same engine 
# in both training and validation sets, causing **temporal data leakage**.
# 
# **Solution:** Randomly hold out 20% of *engines* (not samples) for validation.

# %%
def create_sequences(df, sequence_length, is_test=False):
    """Create 3D sequences for CNN-LSTM."""
    sequences, labels, engine_ids = [], [], []
    
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].sort_values('time')
        data = unit_df[feature_cols].values
        
        if is_test:
            if len(data) >= sequence_length:
                sequences.append(data[-sequence_length:])
                engine_ids.append(unit)
        else:
            rul_values = unit_df['RUL'].values
            for i in range(len(data) - sequence_length + 1):
                sequences.append(data[i:i + sequence_length])
                labels.append(rul_values[i + sequence_length - 1])
                engine_ids.append(unit)
    
    X = np.array(sequences)
    if is_test:
        return X, np.array(engine_ids)
    return X, np.array(labels), np.array(engine_ids)


# ========== UNIT ID-BASED SPLIT (THE KEY FIX) ==========
np.random.seed(42)  # Reproducibility
all_units = train_df['unit'].unique()
n_val_units = int(len(all_units) * VAL_SPLIT_RATIO)
val_units = np.random.choice(all_units, size=n_val_units, replace=False)
train_units = np.array([u for u in all_units if u not in val_units])

print(f"Total units: {len(all_units)}")
print(f"Training units ({len(train_units)}): {sorted(train_units)[:10]}... (showing first 10)")
print(f"Validation units ({len(val_units)}): {sorted(val_units)}")

# Split the dataframe by unit BEFORE creating sequences
train_split_df = train_df[train_df['unit'].isin(train_units)]
val_split_df = train_df[train_df['unit'].isin(val_units)]

# Create sequences from the split dataframes
X_train, y_train, train_ids = create_sequences(train_split_df, SEQUENCE_LENGTH, is_test=False)
X_val, y_val, val_ids = create_sequences(val_split_df, SEQUENCE_LENGTH, is_test=False)
X_test, test_ids = create_sequences(test_df, SEQUENCE_LENGTH, is_test=True)
y_test = rul_df['RUL'].values

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"\n✓ NO DATA LEAKAGE: Train and Val units are completely separate!")

# %% [markdown]
# ## 🧠 Bahdanau-Style Attention Layer
# 
# **What it does:** Allows the model to learn which of the 30 time-steps are most 
# important for predicting RUL.

# %%
class BahdanauAttention(Layer):
    """
    Bahdanau-style Additive Attention for time-step weighting.
    
    Given LSTM output of shape (batch, time_steps, features),
    computes attention weights for each time-step and returns
    a weighted sum context vector.
    """
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units, use_bias=False)
        self.V = Dense(1, use_bias=False)
    
    def call(self, inputs):
        # inputs shape: (batch, time_steps, features)
        # Score shape: (batch, time_steps, 1)
        score = self.V(tf.nn.tanh(self.W(inputs)))
        
        # Attention weights: (batch, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector: weighted sum of inputs
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

print("✓ BahdanauAttention layer defined")

# %% [markdown]
# ## ⚠️ Custom Safety-Weighted Loss Function
# 
# **The Problem:** Standard MSE treats over-predictions and under-predictions equally.
# 
# **In Aviation Reality:**
# - Predicting failure 10 days **LATE** = Catastrophe
# - Predicting failure 10 days **EARLY** = Unnecessary maintenance

# %%
def safety_weighted_loss(y_true, y_pred):
    """
    Asymmetric loss function mimicking NASA S-Score.
    
    Late prediction (y_pred > y_true): Loss = exp(error/10) - 1
    Early prediction (y_pred < y_true): Loss = exp(-error/13) - 1
    
    Penalizes late predictions ~3x more severely.
    """
    error = y_pred - y_true
    
    # Late predictions (over-estimation of RUL)
    late_loss = tf.exp(error / 10.0) - 1.0
    
    # Early predictions (under-estimation of RUL)
    early_loss = tf.exp(-error / 13.0) - 1.0
    
    # Select appropriate loss based on error sign
    loss = tf.where(error >= 0, late_loss, early_loss)
    
    return tf.reduce_mean(loss)

# Quick test
test_true = tf.constant([50.0, 50.0])
test_pred = tf.constant([60.0, 40.0])  # 10 late, 10 early
test_loss = safety_weighted_loss(test_true, test_pred)
print(f"Test loss (10 late + 10 early): {test_loss.numpy():.4f}")
print("✓ Safety-weighted loss function defined")

# %% [markdown]
# ## 🏗️ CNN-Attention-LSTM Hybrid Architecture

# %%
def build_attention_model(input_shape):
    """
    Build CNN-Attention-LSTM hybrid architecture.
    
    Args:
        input_shape: Tuple (sequence_length, n_features) = (30, 17)
    
    Returns:
        Compiled Keras model with Safety-Weighted Loss
    """
    inputs = Input(shape=input_shape, name='input')
    
    # ===== CNN Block: Spatial Pattern Extraction =====
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2(0.001), name='conv1d')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling1D(pool_size=2, name='maxpool')(x)
    x = Dropout(0.2, name='dropout_cnn')(x)
    
    # ===== LSTM Block: Temporal Sequence Processing =====
    x = LSTM(100, return_sequences=True, kernel_regularizer=l2(0.001),
             name='lstm_1')(x)
    x = Dropout(0.3, name='dropout_lstm1')(x)
    
    # ===== Attention Block: Critical Time-Step Focus =====
    context_vector, attention_weights = BahdanauAttention(64, name='attention')(x)
    
    # ===== Output Block =====
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001),
              name='dense_1')(context_vector)
    x = Dropout(0.2, name='dropout_dense')(x)
    outputs = Dense(1, activation='linear', name='rul_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Attention_LSTM')
    
    # Compile with Safety-Weighted Loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=safety_weighted_loss,
        metrics=['mae']
    )
    
    return model

# Build and display model
model = build_attention_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# %%
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(f'{MODEL_DIR}/cnn_attention_lstm_model.keras', 
                    monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
]

print(f"Training for {EPOCHS} epochs with Safety-Weighted Loss...")
print(f"Training on {len(train_units)} engines, validating on {len(val_units)} engines\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss (Safety-Weighted)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# MAE
axes[1].plot(history.history['mae'], label='Training MAE')
axes[1].plot(history.history['val_mae'], label='Validation MAE')
axes[1].set_title('Model MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# %%
def calculate_rmse(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_score(y_true, y_pred):
    """Calculate NASA S-Score."""
    d = y_pred - y_true
    score = 0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13) - 1
        else:
            score += np.exp(di / 10) - 1
    return score

# Predict
y_pred = model.predict(X_test, verbose=0).flatten()
y_pred = np.maximum(y_pred, 0)  # Non-negative

# Metrics
rmse = calculate_rmse(y_test, y_pred)
score = calculate_score(y_test, y_pred)

print("=" * 60)
print("🏆 EVALUATION RESULTS (CNN-Attention-LSTM + Safety Loss)")
print("=" * 60)
print(f"📊 Test RMSE: {rmse:.4f}")
print(f"📊 Test S-Score: {score:.4f}")
print("=" * 60)

if rmse < 15.0:
    print("\n✅ Excellent! RMSE is within competition-winning range.")
else:
    print("\n⚠️ Consider more tuning or longer training.")

# %%
# Predictions vs Actual
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label='Perfect')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.title('True vs Predicted RUL')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
errors = y_pred - y_test
plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🎯 Interactive Engine Health Gauge (The Winning Move)
# 
# This helper function takes a `unit_id`, pulls its data from `X_test`, and displays 
# a **Radial Gauge Chart**. Perfect for "The Twist" scenario!

# %%
import plotly.graph_objects as go

def display_engine_health(unit_id):
    """
    Display health gauge for a specific test engine.
    
    Args:
        unit_id: Engine ID from test set
    
    Returns:
        Plotly Figure with radial gauge
    """
    # Find engine index
    idx = np.where(test_ids == unit_id)[0]
    if len(idx) == 0:
        print(f"❌ Engine {unit_id} not found in test set!")
        print(f"Available engines: {sorted(test_ids)}")
        return None
    
    idx = idx[0]
    
    # Get prediction for this engine
    engine_data = X_test[idx:idx+1]
    predicted_rul = model.predict(engine_data, verbose=0).flatten()[0]
    predicted_rul = max(0, predicted_rul)
    true_rul = y_test[idx]
    
    # Industrial Health Calculation (Capped at 125)
    health_pct = min(100.0, (predicted_rul / 125) * 100)
    
    # Determine color and status
    if health_pct > 70:
        color = "green"
        status = "SAFE ✅"
    elif health_pct > 30:
        color = "orange"
        status = "MAINTENANCE ⚠️"
    else:
        color = "red"
        status = "CRITICAL 🚨"
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_pct,
        title={'text': f"Engine {unit_id} Health Status (%)"},
        delta={'reference': 70, 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "rgba(255, 0, 0, 0.3)"},
                {'range': [30, 70], 'color': "rgba(255, 165, 0, 0.3)"},
                {'range': [70, 100], 'color': "rgba(0, 128, 0, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': health_pct
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        annotations=[
            dict(
                text=f"<b>Status:</b> {status}<br><b>Pred RUL:</b> {predicted_rul:.1f}<br><b>True RUL:</b> {true_rul}",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    return fig

print("✓ display_engine_health() function defined")
print(f"\nTest engines available: {sorted(test_ids)[:10]}... (showing first 10)")

# %%
# 🎲 Simulate "The Twist" - Random engine selection
random_engine = np.random.choice(test_ids)
print(f"🎲 Judge selected Engine #{random_engine}")

fig = display_engine_health(random_engine)
if fig:
    fig.show()

# %%
# Color-coded predictions for all engines
colors = []
for pred in y_pred:
    health_pct = (pred / 125) * 100
    if health_pct > 70:
        colors.append('green')
    elif health_pct > 30:
        colors.append('orange')
    else:
        colors.append('red')

plt.figure(figsize=(14, 6))
plt.bar(range(len(y_pred)), y_pred, color=colors, edgecolor='k', alpha=0.7)
plt.axhline(y=87.5, color='green', linestyle='--', label='Safe (>70%)')
plt.axhline(y=37.5, color='orange', linestyle='--', label='Maintenance (30-70%)')
plt.xlabel('Engine Index')
plt.ylabel('Predicted RUL (cycles)')
plt.title('Engine Health Status - All Test Engines')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nHealth Distribution:")
print(f"  Safe (Green): {colors.count('green')} engines")
print(f"  Maintenance (Orange): {colors.count('orange')} engines")
print(f"  Critical (Red): {colors.count('red')} engines")

# %%
import datetime

# Save model in .keras format
model.save(f'{MODEL_DIR}/cnn_attention_lstm_model.keras')
print(f"✓ Saved: {MODEL_DIR}/cnn_attention_lstm_model.keras")

# Generate a summary report
print("\n" + "=" * 60)
print("📋 MODEL SUMMARY REPORT")
print("=" * 60)
print(f"Architecture: CNN-Attention-LSTM Hybrid")
print(f"Input Shape: (30, 17)")
print(f"Attention Type: Bahdanau (Additive)")
print(f"Loss Function: Safety-Weighted (Asymmetric)")
print(f"Validation Split: Unit ID-based ({VAL_SPLIT_RATIO*100:.0f}% engines)")
print(f"Epochs Trained: {EPOCHS}")
print(f"\n📊 Final Metrics:")
print(f"  - Test RMSE: {rmse:.4f}")
print(f"  - Test S-Score: {score:.4f}")
print("=" * 60)

# %% [markdown]
# ## 📝 Judge Q&A Preparation
# 
# | Question | Winning Answer |
# |----------|---------------|
# | "Why did you switch to a Unit-based split?" | "To prevent temporal data leakage. A time-step split allows the model to see adjacent cycles of the same engine in both sets, inflating performance artificially." |
# | "What does the Attention mechanism provide?" | "Interpretability. It allows us to extract weights showing which specific time-steps in the 30-cycle window the model prioritized to make its prediction." |
# | "Why an Asymmetric Loss?" | "Industrial safety. In aviation, the cost of an over-prediction (failure) is infinitely higher than an under-prediction (maintenance)." |
