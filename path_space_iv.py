import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def path_signature(series, depth=4):
    """
    Compute the truncated signature of a 1D series up to depth.
    """
    if len(series) < 2:
        return np.zeros(depth)
    increments = np.diff(series)
    sig = []
    sig.append(np.sum(increments))
    if depth >= 2:
        sum_inc = np.sum(increments)
        sum_sq = np.sum(increments**2)
        sig.append(0.5 * (sum_inc**2 - sum_sq))
    if depth >= 3:
        sig.append((np.sum(increments)**3) / 6.0)
    if depth >= 4:
        sig.append((np.sum(increments)**4) / 24.0)
    return np.array(sig)

def realized_volatility(returns, horizon=21):
    """
    Compute realized volatility over a horizon.
    """
    if len(returns) < horizon:
        return np.std(returns)
    return np.std(returns[-horizon:])

def path_space_implied_vol(returns, macro_df, depth=4, horizon=21):
    """
    Compute path-space implied volatility using signature features.
    The signature of the path predicts future realized volatility.
    """
    if len(returns) < horizon + 5 or macro_df is None or len(macro_df) < horizon + 5:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < horizon + 5:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    # Prepare features: path signature of returns and macro factor
    window = 20
    X, y = [], []
    for i in range(window, len(returns) - horizon):
        ret_seg = returns[i-window:i]
        macro_seg = macro_factor[i-window:i]
        # Signature of return path
        sig_ret = path_signature(ret_seg, depth)
        # Signature of macro path
        sig_macro = path_signature(macro_seg, depth)
        # Joint signature (interaction)
        joint = path_signature(np.concatenate([ret_seg, macro_seg]), depth)
        features = np.concatenate([sig_ret, sig_macro, joint])
        X.append(features)
        # Target: realized volatility over the next horizon
        y.append(realized_volatility(returns[i:i+horizon], horizon))
    X = np.array(X)
    y = np.array(y)
    if len(y) < 10:
        return 0.0
    # Train ridge regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    # Predict implied volatility for the current path
    # Use the last window
    last_ret_seg = returns[-window:]
    last_macro_seg = macro_factor[-window:]
    last_sig_ret = path_signature(last_ret_seg, depth)
    last_sig_macro = path_signature(last_macro_seg, depth)
    last_joint = path_signature(np.concatenate([last_ret_seg, last_macro_seg]), depth)
    last_features = np.concatenate([last_sig_ret, last_sig_macro, last_joint]).reshape(1, -1)
    last_scaled = scaler.transform(last_features)
    iv_pred = ridge.predict(last_scaled)[0]
    # Implied volatility should be positive
    iv_pred = max(iv_pred, 0.0)
    return float(iv_pred)
