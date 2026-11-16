**UM MADS 695 Milestone 2** — Diane O and Oleg N, Fall 2021<br>
Crypto price prediction and analysis

We analyzed and predicted daily prices for Bitcoin, Ethereum, Litecoin, and Dogecoin using market data from CoinMetrics and social media data from LunarCrush.

- **Supervised learning:** LSTM neural networks (TensorFlow/Keras) across several window sizes, and a Random Forest regressor with feature-importance (Pareto) selection, evaluated with MAE and residual plots
- **Unsupervised learning:** hierarchical (Ward) clustering of coin price time series using dynamic time warping
- **Interactive app:** a Streamlit app (in `streamlit/`) where you can choose a coin and data source and rerun the models
