# Network-Centric Feature Engineering for Cross-Asset Statistical Arbitrage

A market-neutral backtesting framework that models equity markets as a dynamic network to capture cross-asset alpha. This framework strays away from standalone time-series metrics by using graph-theoretic factors, which are isolated from noise using PCA and modeled through Ridge Regression.

## Structure

```text
network-stat-arb/
│
├── backtest.pynb # Execution and visualization
├── src/
│   ├── data_pipeline.py      # Data preprocessing
│   ├── features.py           # Graph generation and topology feature math
│   └── backtest.py           # PCA and Ridge Walk-Forward validation
├── report.pdf paper
└── requirements.txt