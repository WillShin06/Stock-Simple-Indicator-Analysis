# src/analysis.py
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# 1. LOG TREND (log-price regression) 
# -----------------------------------------------------------
def compute_log_trend(series: pd.Series):
    """
    Fit log(price) ~ time to produce an exponential/trend line.
    Returns a pandas Series aligned with the original index.
    """
    series = series.dropna()
    y = np.log(series.values)
    x = np.arange(len(series))

    # Linear regression in log-space
    coef = np.polyfit(x, y, 1)
    trend_log = np.polyval(coef, x)
    trend = np.exp(trend_log)

    trend_series = pd.Series(trend, index=series.index, name="log_trend")
    return trend_series


# -----------------------------------------------------------
# 2. SMOOTH TREND (moving average)
# -----------------------------------------------------------
def compute_smooth_trend(series: pd.Series, window=50):
    """
    Returns a simple moving average to smooth volatility.
    """
    return series.rolling(window=window, min_periods=1).mean()


# -----------------------------------------------------------
# 3. % DISTANCE FROM TREND
# -----------------------------------------------------------
def pct_distance(series: pd.Series, trend: pd.Series):
    """
    Computes (price - trend) / trend as percent distance.
    """
    return (series - trend) / trend * 100.0
