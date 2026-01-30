import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_statistical_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def calculate_financial_metrics(y_true, y_pred, risk_free_rate=0.02):
    # sign of returns for directional accuracy
    da = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100
    
    # win rate (percentage of profitable trades)
    # Assuming we trade in the direction of y_pred
    profitable = (np.sign(y_pred) == np.sign(y_true)).astype(float)
    win_rate = np.mean(profitable) * 100
    
    # Cumulative returns (assuming daily trading)
    # y_true is daily % return. If we go long on positive y_pred and short on negative
    strategy_returns = np.sign(y_pred) * y_true
    cum_returns = np.cumprod(1 + strategy_returns) - 1
    
    # Sharpe Ratio
    avg_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe = (avg_ret * 252 - risk_free_rate) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
    
    # Max Drawdown
    peak = np.maximum.accumulate(1 + cum_returns)
    drawdown = ((1 + cum_returns) / peak) - 1
    mdd = np.min(drawdown) * 100
    
    # Profit Factor: gross profits / gross losses
    profits = strategy_returns[strategy_returns > 0].sum()
    losses = np.abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = profits / losses if losses > 0 else float('inf')
    
    return {
        'Sharpe Ratio': sharpe,
        'Win Rate (%)': win_rate,
        'Directional Accuracy (%)': da,
        'Max Drawdown (%)': mdd,
        'Cumulative Return (%)': cum_returns[-1] * 100 if len(cum_returns) > 0 else 0,
        'Profit Factor': profit_factor
    }
