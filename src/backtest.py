import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import zscore

def extract_residuals_pca(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    PCA to return idiosyncratic residuals.
    """
    def get_residual_returns(returns_df):
        pca = PCA(n_components=1)
        market_factor = pca.fit_transform(returns_df)
        market_reconstruction = pca.inverse_transform(market_factor)
        residuals = returns_df - market_reconstruction
        return pd.DataFrame(residuals, index=returns_df.index, columns=returns_df.columns)
    residual_returns = get_residual_returns(stock_returns.loc[degree_centrality.index])
    return residual_returns

def run_ridge_walk_forward(degree_df, neigh_mom_df, target_returns, stock_returns):
    """
    Makes expanding-window Ridge regression and builds long/short portfolio.
    Returns: Series of daily strategy returns
    """
    aligned_idx = target_returns.index.intersection(degree_centrality.index).intersection(neighbor_momentum.index)
    degree_df = degree_centrality.loc[aligned_idx]
    neighbor_momentum_df = neighbor_momentum.loc[aligned_idx]
    targets = target_returns.loc[aligned_idx]
    portfolio_returns = []
    prediction_dates = []

    for t in range(500, len(aligned_idx) - 1):
        # Train
        X_train_degree = degree_df.iloc[:t].values.flatten()
        X_train_momentum = neighbor_momentum_df.iloc[:t].values.flatten()
        y_train = targets.iloc[:t].values.flatten()
        X_train = np.column_stack((zscore(X_train_degree), zscore(X_train_momentum)))
        
        m = Ridge(alpha=1.0)
        m.fit(X_train, y_train)
        X_test_degree = degree_df.iloc[t].values
        X_test_momentum = neighbor_momentum_df.iloc[t].values
        X_test = np.column_stack((zscore(X_test_degree), zscore(X_test_momentum)))
        preds = m.predict(X_test)
        pred_series = pd.Series(preds, index=targets.columns)
        
        quantiles = pred_series.quantile([0.2, 0.8])
        longs = pred_series[pred_series >= quantiles[0.8]].index
        shorts = pred_series[pred_series <= quantiles[0.2]].index
        
        actual_next_day_returns = stock_returns.loc[aligned_idx[t+1]]
        long_returns = actual_next_day_returns[longs].mean() if len(longs) > 0 else 0
        short_returns = actual_next_day_returns[shorts].mean() if len(shorts) > 0 else 0
        
        strat_returns = (long_returns - short_returns) / 2 
        portfolio_returns.append(strat_returns)
        prediction_dates.append(aligned_idx[t+1])

    strat_series = pd.Series(portfolio_returns, index=prediction_dates)
    return strat_series