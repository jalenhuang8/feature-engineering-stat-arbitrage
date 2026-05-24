import pandas as pd
import numpy as np

def generate_network_features(stock_returns: pd.DataFrame, rolling_window=60, momentum_window=5, threshold=0.5):
    """
    Constructs adjacency matrices and makes network features.
    Returns: (degree_centrality_df, neighbor_momentum_df)
    """
    rolling_window = 60          
    threshold = 0.5     

    degree_centrality = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)
    neighbor_momentum = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)

    rolling_returns = stock_returns.rolling(window=5).sum()

    for i in range(rolling_window, len(stock_returns)):
        window_data = stock_returns.iloc[i-rolling_window : i]
        corr_matrix = window_data.corr().values
        np.fill_diagonal(corr_matrix, 0)
        adjacency = (corr_matrix > threshold).astype(int)
        degrees = adjacency.sum(axis=1)
        current_returns = rolling_returns.iloc[i-1].values
        safe_degrees = np.where(degrees == 0, 1, degrees)
        neighbors_momentum = np.dot(adjacency, current_returns) / safe_degrees
        date_idx = stock_returns.index[i]
        degree_centrality.loc[date_idx] = degrees
        neighbor_momentum.loc[date_idx] = neighbors_momentum
    degree_centrality = degree_centrality.dropna().astype(float)
    neighbor_momentum = neighbor_momentum.dropna().astype(float)
    
    return degree_centrality, neighbor_momentum