"""
AeroGuard RUL - Advanced Utility Functions
==========================================
Competition-winning metrics and analysis:
- RMSE & NASA S-Score (standard metrics)
- SHAP Interpretability for model explanations
- Uncertainty Quantification helpers
- Resilience Testing (sensor dropout evaluation)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_score(y_true, y_pred):
    """
    Calculate NASA S-Score (asymmetric penalty).
    
    Late predictions (positive error) are penalized more heavily than early predictions.
    - Early prediction (d < 0): S = exp(-d/13) - 1
    - Late prediction (d >= 0): S = exp(d/10) - 1
    
    Where d = predicted - actual (positive means late prediction)
    """
    d = y_pred - y_true
    score = 0
    
    for di in d:
        if di < 0:
            score += np.exp(-di / 13) - 1
        else:
            score += np.exp(di / 10) - 1
    
    return score


def get_health_status(predicted_rul, max_rul=125, uncertainty=None):
    """
    Calculate health percentage and status based on predicted RUL.
    
    Args:
        predicted_rul: Predicted RUL value
        max_rul: Maximum RUL cap
        uncertainty: Optional uncertainty (std) for risk assessment
    
    Returns:
        tuple: (health_percentage, color, status_text, status_emoji, risk_level)
    """
    # Calculate health percentage
    health_pct = min(100, max(0, (predicted_rul / max_rul) * 100))
    
    # Adjust risk based on uncertainty if provided
    risk_adjustment = 0
    if uncertainty is not None:
        # Higher uncertainty = higher risk
        risk_adjustment = min(20, uncertainty * 2)
        health_pct = max(0, health_pct - risk_adjustment)
    
    if health_pct > 70:
        return health_pct, '#00FF00', 'SAFE', '🟢', 'Low'
    elif health_pct > 30:
        return health_pct, '#FFD700', 'MAINTENANCE DUE', '🟡', 'Medium'
    else:
        return health_pct, '#FF0000', 'CRITICAL - IMMEDIATE ACTION REQUIRED', '🔴', 'High'


def format_predictions_table(engine_ids, y_true, y_pred, uncertainties=None):
    """
    Create a formatted table of predictions with optional uncertainty.
    
    Args:
        engine_ids: Array of engine IDs
        y_true: True RUL values
        y_pred: Predicted RUL values
        uncertainties: Optional array of prediction uncertainties
    
    Returns:
        List of dictionaries with prediction details
    """
    results = []
    for i, (eid, true_rul, pred_rul) in enumerate(zip(engine_ids, y_true, y_pred)):
        uncertainty = uncertainties[i] if uncertainties is not None else None
        health_pct, _, status, emoji, risk = get_health_status(pred_rul, uncertainty=uncertainty)
        
        result = {
            'Engine ID': int(eid),
            'True RUL': int(true_rul),
            'Predicted RUL': int(round(pred_rul)),
            'Health %': f'{health_pct:.1f}%',
            'Status': f'{emoji} {status}',
            'Risk': risk
        }
        
        if uncertainty is not None:
            result['Uncertainty (±)'] = f'{uncertainty:.1f}'
            result['95% CI'] = f'[{max(0, pred_rul - 1.96*uncertainty):.0f}, {pred_rul + 1.96*uncertainty:.0f}]'
        
        results.append(result)
    
    return results


# ============================================================================
# SHAP INTERPRETABILITY
# ============================================================================

def setup_shap_explainer(model, background_data, max_samples=100):
    """
    Set up SHAP DeepExplainer for CNN-LSTM model.
    
    Args:
        model: Trained Keras model
        background_data: Background samples for SHAP (typically training data subset)
        max_samples: Maximum background samples (reduces computation time)
    
    Returns:
        SHAP DeepExplainer object
    """
    try:
        import shap
        
        # Use a random subset for background
        if len(background_data) > max_samples:
            idx = np.random.choice(len(background_data), max_samples, replace=False)
            background_data = background_data[idx]
        
        explainer = shap.DeepExplainer(model, background_data)
        return explainer
    except Exception as e:
        print(f"Warning: Could not set up SHAP explainer: {e}")
        return None


def explain_prediction(explainer, X_sample, feature_names=None):
    """
    Generate SHAP explanation for a prediction.
    
    Args:
        explainer: SHAP DeepExplainer object
        X_sample: Sample to explain (shape: (1, seq_len, n_features))
        feature_names: List of feature names
    
    Returns:
        Dictionary with SHAP values and interpretation
    """
    if explainer is None:
        return None
    
    try:
        import shap
        
        # Get SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Aggregate SHAP values across time for feature importance
        # Shape: (1, seq_len, n_features) -> (n_features,)
        feature_importance = np.abs(shap_values[0]).mean(axis=0)
        
        # Create interpretation
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        explanation = {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'top_features': [
                {
                    'name': feature_names[i],
                    'importance': float(feature_importance[i]),
                    'rank': rank + 1
                }
                for rank, i in enumerate(sorted_idx[:10])
            ]
        }
        
        return explanation
    except Exception as e:
        print(f"Warning: Could not generate SHAP explanation: {e}")
        return None


def generate_shap_summary(model, X_data, feature_names=None, max_samples=50):
    """
    Generate SHAP summary for model interpretation.
    
    Args:
        model: Trained model
        X_data: Data samples to explain
        feature_names: List of feature names
        max_samples: Maximum samples to use
    
    Returns:
        Dictionary with overall feature importance
    """
    try:
        import shap
        
        # Limit samples
        if len(X_data) > max_samples:
            idx = np.random.choice(len(X_data), max_samples, replace=False)
            X_subset = X_data[idx]
        else:
            X_subset = X_data
        
        # Create explainer with subset
        background_idx = np.random.choice(len(X_data), min(50, len(X_data)), replace=False)
        explainer = shap.DeepExplainer(model, X_data[background_idx])
        
        # Get SHAP values
        shap_values = explainer.shap_values(X_subset)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Aggregate across time and samples
        mean_importance = np.abs(shap_values).mean(axis=(0, 1))
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(mean_importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(mean_importance)[::-1]
        
        return {
            'feature_names': [feature_names[i] for i in sorted_idx],
            'importance': [float(mean_importance[i]) for i in sorted_idx],
            'raw_shap_values': shap_values
        }
    except Exception as e:
        print(f"Warning: Could not generate SHAP summary: {e}")
        return None


# ============================================================================
# RESILIENCE TESTING
# ============================================================================

def test_sensor_dropout(model, X_test, y_test, sensor_indices, n_trials=5):
    """
    Test model resilience to sensor failures.
    
    Simulates sensor dropout by setting sensor values to zero and
    measuring the impact on prediction accuracy.
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: True RUL values
        sensor_indices: List of sensor indices to drop
        n_trials: Number of trials for averaging
    
    Returns:
        Dictionary with resilience metrics
    """
    # Baseline performance
    baseline_pred = model.predict(X_test, verbose=0).flatten()
    baseline_rmse = calculate_rmse(y_test, baseline_pred)
    
    # Test with sensor dropout
    X_dropout = X_test.copy()
    for idx in sensor_indices:
        X_dropout[:, :, idx] = 0  # Zero out the sensor
    
    dropout_pred = model.predict(X_dropout, verbose=0).flatten()
    dropout_rmse = calculate_rmse(y_test, dropout_pred)
    
    # Calculate degradation
    rmse_degradation = (dropout_rmse - baseline_rmse) / baseline_rmse * 100
    
    return {
        'baseline_rmse': baseline_rmse,
        'dropout_rmse': dropout_rmse,
        'rmse_increase_pct': rmse_degradation,
        'sensors_dropped': sensor_indices,
        'is_resilient': rmse_degradation < 20  # Less than 20% increase = resilient
    }


def test_noise_injection(model, X_test, y_test, noise_level=0.1, n_trials=5):
    """
    Test model resilience to noisy sensor readings.
    
    Injects Gaussian noise into sensor readings and measures
    the impact on prediction accuracy.
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: True RUL values
        noise_level: Standard deviation of noise (fraction of data std)
        n_trials: Number of trials for averaging
    
    Returns:
        Dictionary with resilience metrics
    """
    # Baseline performance
    baseline_pred = model.predict(X_test, verbose=0).flatten()
    baseline_rmse = calculate_rmse(y_test, baseline_pred)
    
    # Test with noise injection
    noisy_rmses = []
    for _ in range(n_trials):
        noise = np.random.normal(0, noise_level, X_test.shape)
        X_noisy = X_test + noise
        
        noisy_pred = model.predict(X_noisy, verbose=0).flatten()
        noisy_rmses.append(calculate_rmse(y_test, noisy_pred))
    
    mean_noisy_rmse = np.mean(noisy_rmses)
    rmse_degradation = (mean_noisy_rmse - baseline_rmse) / baseline_rmse * 100
    
    return {
        'baseline_rmse': baseline_rmse,
        'noisy_rmse': mean_noisy_rmse,
        'rmse_increase_pct': rmse_degradation,
        'noise_level': noise_level,
        'is_resilient': rmse_degradation < 15  # Less than 15% increase = resilient
    }


def full_resilience_test(model, X_test, y_test, feature_names=None):
    """
    Run comprehensive resilience testing.
    
    Tests:
    1. Individual sensor dropout
    2. Multiple sensor dropout (2 sensors)
    3. Noise injection at different levels
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: True RUL values
        feature_names: Optional list of feature names
    
    Returns:
        Comprehensive resilience report
    """
    n_features = X_test.shape[2]
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    print("\n" + "=" * 60)
    print("Resilience Testing Report")
    print("=" * 60)
    
    # 1. Individual sensor dropout
    print("\n[1/3] Testing individual sensor dropout...")
    individual_results = []
    for i in range(min(n_features, 10)):  # Test first 10 features
        result = test_sensor_dropout(model, X_test, y_test, [i])
        result['sensor_name'] = feature_names[i]
        individual_results.append(result)
        print(f"  {feature_names[i]}: RMSE +{result['rmse_increase_pct']:.1f}%")
    
    # 2. Multiple sensor dropout
    print("\n[2/3] Testing multiple sensor dropout...")
    # Drop 2 random non-critical sensors
    multi_drop_indices = [0, 1]  # First two sensors
    multi_result = test_sensor_dropout(model, X_test, y_test, multi_drop_indices)
    print(f"  Dropping 2 sensors: RMSE +{multi_result['rmse_increase_pct']:.1f}%")
    
    # 3. Noise injection
    print("\n[3/3] Testing noise injection...")
    noise_results = []
    for noise in [0.05, 0.1, 0.2]:
        result = test_noise_injection(model, X_test, y_test, noise_level=noise)
        noise_results.append(result)
        print(f"  Noise σ={noise}: RMSE +{result['rmse_increase_pct']:.1f}%")
    
    # Summarize
    resilient_count = sum(1 for r in individual_results if r['is_resilient'])
    overall_score = (resilient_count / len(individual_results)) * 100
    
    print("\n" + "-" * 60)
    print(f"Resilience Score: {overall_score:.0f}% ({resilient_count}/{len(individual_results)} sensors)")
    print("-" * 60)
    
    return {
        'individual_dropout': individual_results,
        'multi_dropout': multi_result,
        'noise_injection': noise_results,
        'overall_score': overall_score,
        'is_robust': overall_score >= 70
    }


# ============================================================================
# PITCH GENERATOR
# ============================================================================

def generate_prediction_pitch(predicted_rul, top_features, threshold_sensor=None):
    """
    Generate a judge-ready pitch for a prediction.
    
    Example output:
    "Judge, our model predicts 12 cycles left because Sensor Ps30 
    (Static Pressure) has crossed its safety threshold of 0.8."
    
    Args:
        predicted_rul: Predicted RUL value
        top_features: List of top influential features from SHAP
        threshold_sensor: Optional sensor that crossed threshold
    
    Returns:
        String pitch for the judge
    """
    if predicted_rul <= 10:
        urgency = "CRITICAL - the engine requires immediate attention"
    elif predicted_rul <= 30:
        urgency = "the engine is approaching its maintenance window"
    else:
        urgency = "the engine is operating within safe parameters"
    
    pitch = f"🎯 **Prediction: {int(predicted_rul)} cycles remaining**\n\n"
    pitch += f"Our model predicts {int(predicted_rul)} cycles of useful life remaining, "
    pitch += f"indicating that {urgency}.\n\n"
    
    if top_features:
        pitch += "**Key Contributing Factors:**\n"
        for i, feat in enumerate(top_features[:3], 1):
            pitch += f"{i}. **{feat['name']}** (importance: {feat['importance']:.3f})\n"
    
    if threshold_sensor:
        pitch += f"\n⚠️ **Alert:** {threshold_sensor} has crossed its safety threshold."
    
    return pitch


if __name__ == "__main__":
    # Test utility functions
    y_true = np.array([50, 80, 120, 20, 100])
    y_pred = np.array([55, 75, 115, 25, 95])
    
    print("=" * 50)
    print("Utility Functions Test")
    print("=" * 50)
    
    print(f"\nRMSE: {calculate_rmse(y_true, y_pred):.4f}")
    print(f"S-Score: {calculate_score(y_true, y_pred):.4f}")
    
    # Test health status with uncertainty
    print("\nHealth Status with Uncertainty:")
    for rul, unc in [(100, 5), (50, 10), (20, 15)]:
        health, color, status, emoji, risk = get_health_status(rul, uncertainty=unc)
        print(f"RUL {rul} ± {unc}: {emoji} {health:.1f}% - {status} (Risk: {risk})")
    
    # Test pitch generation
    print("\n" + "=" * 50)
    print("Sample Prediction Pitch:")
    print("=" * 50)
    
    sample_features = [
        {'name': 'sensor11 (Ps30 - Static Pressure)', 'importance': 0.45},
        {'name': 'sensor7 (T50 - Temperature)', 'importance': 0.32},
        {'name': 'sensor4 (T30 - Temperature)', 'importance': 0.18}
    ]
    
    pitch = generate_prediction_pitch(12, sample_features, "Sensor Ps30")
    print(pitch)
