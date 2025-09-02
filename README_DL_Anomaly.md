
# Deep Learning Demand Forecasting and Anomaly Detection

## Objective
Build a production-ready pipeline for retail demand forecasting and anomaly detection using deep learning (Temporal Fusion Transformer) and statistical/ML methods to improve inventory planning, flag unusual patterns, and support decisions.

## Dataset Description
- Sales Data: Weekly sales by store and department.
- Core Fields: store_id, dept_id, date, weekly_sales, holiday_flag, promotions.
- External Factors: temperature, fuel_price, CPI, seasonality/holiday indicators.
- Engineered Features: time-based features, lags, rolling means, promo/holiday windows.

## Methods & Techniques
### Deep Learning Forecasting
- Model: Temporal Fusion Transformer (PyTorch Forecasting)
- Framework: PyTorch Lightning (Trainer, callbacks, early stopping)
- Data: Grouped time series with static and time-varying features, normalization/encoding
- Metrics: MAE, RMSE, MAPE; backtesting splits for robustness
- Interpretability: Variable selection weights, attention, partial dependencies

### Anomaly Detection
- Decomposition: Seasonal-Trend decomposition (STL) to obtain residuals
- Stat Rules: SPC-style thresholds on residuals (e.g., z-score/IQR)
- ML-based: Isolation Forest on multivariate feature windows (trend, change rate, promo flags)
- Review Loop: Flag lists, drill-down plots, and store/department ranking

## Tasks Performed
- Data ingestion and ETL to build model-ready time series
- Feature engineering (lags, rolling windows, calendar, promotions, weather)
- TFT model training with hyperparameter tuning and early stopping
- Backtesting and error analysis across stores/departments
- STL + Isolation Forest anomaly detection on forecast errors and residuals
- Visualization: actual vs forecast, error bands, anomaly overlays, feature importance

## Key Visualizations
- Forecast vs actual by store/department with confidence bands
- Loss curves and validation metrics over epochs
- Variable importance and attention summaries (TFT)
- STL components (trend/seasonality/residual) and anomaly flags
- Anomaly score timelines and top-N anomalous segments

## Results
- TFT captured seasonality and promo/holiday effects with lower MAE/RMSE than baseline ML models (e.g., SARIMA/Prophet) on validation splits.
- Hybrid anomaly detection surfaced rare demand spikes/dips for targeted reviews.
- Dashboards/plots enable rapid triage of underperforming stores and SKUs.

## Business Recommendations
- Adjust inventory and replenishment cadence based on forecasted peaks and anomaly alerts.
- Tie promotions to segments with the highest uplift per forecast while monitoring anomaly risk.
- Establish a weekly review of flagged stores/departments to investigate root causes.
- Retrain the model on a rolling window and refresh anomaly thresholds as seasonality shifts.

## Reproducibility
- Deterministic seeds for model training
- Clear config for data paths, time windows, and hyperparameters
- Requirements pinned for consistent environments

## Project Structure
- `Demand_Forecasting_using_DL.ipynb` — TFT modeling and evaluation
- `anomaly_detection.ipynb` (optional) — STL, SPC thresholds, Isolation Forest
- `data/` — raw and processed data
- `reports/` — saved figures and metrics
- `requirements.txt` — dependencies

## How to Run
1. Create and activate an environment.
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate                            # Windows
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open the notebook(s).
   ```bash
   jupyter notebook
   ```
4. Run cells in order. Configure paths in the first config cell if necessary.
