# 5G RAN QoS Prediction Mini-Project

## Overview
This project demonstrates how to model and predict **application-layer throughput and latency** for 5G User Equipment (UE) based solely on **physical-layer RAN measurements** (RSRP, SINR, CQI) and network load indicators.

## Project Structure
```
5g_ran_qos_prediction/
├── data/                          # Generated and processed datasets
│   ├── 5g_ran_dataset.csv        # Raw generated data
│   ├── 5g_ran_dataset_clean.csv  # Cleaned data
│   └── 5g_ran_dataset_features.csv # Feature-engineered data
├── models/                        # Trained ML models
│   ├── throughput_lr_model.pkl
│   ├── throughput_dt_model.pkl
│   ├── throughput_rf_model.pkl
│   ├── latency_lr_model.pkl
│   ├── latency_dt_model.pkl
│   └── latency_rf_model.pkl
├── results/                       # Visualizations and analysis results
│   ├── distributions.png
│   ├── correlation_matrix.png
│   ├── throughput_predictions.png
│   ├── latency_predictions.png
│   └── model_comparison.csv
├── src/                          # Source code
│   ├── data_generator.py         # Synthetic data generation
│   ├── data_preprocessing.py     # Data cleaning and validation
│   ├── eda_analysis.py          # Exploratory data analysis
│   ├── feature_engineering.py    # Feature creation
│   ├── model_training.py        # ML model training
│   └── main_pipeline.py         # Complete pipeline orchestration
└── README.md                     # This file
```

## Features

### Physical-Layer RAN Metrics (Input Features)
- **RSRP** (Reference Signal Received Power): Signal strength in dBm
- **SINR** (Signal to Interference plus Noise Ratio): Signal quality in dB
- **CQI** (Channel Quality Indicator): Channel quality (0-15)
- **MCS** (Modulation and Coding Scheme): Modulation scheme (0-28)
- **RB Allocation**: Number of resource blocks allocated
- **Network Load**: Current network congestion level (0-1)

### Application-Layer QoS Metrics (Targets)
- **Throughput**: Data rate in Mbps
- **Latency**: Round-trip time in milliseconds

### Engineered Features
- Rolling window statistics (mean, std, min, max)
- Rate of change features
- Stability metrics (coefficient of variation)
- Interaction features (e.g., RSRP × SINR, CQI/Load ratio)
- Time-based features (hour, day of week, peak hour indicator)

## Installation

### Requirements
```bash
pip3 install numpy pandas scikit-learn matplotlib seaborn scipy
```

Or install from requirements file:
```bash
pip3 install -r requirements.txt
```

## Usage

### Option 1: Run Complete Pipeline
Execute the entire workflow (data generation → preprocessing → EDA → feature engineering → model training):

```bash
cd src
python3 main_pipeline.py
```

### Option 2: Run Individual Steps

#### 1. Generate Dataset
```bash
cd src
python3 data_generator.py
```

#### 2. Preprocess Data
```bash
python3 data_preprocessing.py
```

#### 3. Exploratory Data Analysis
```bash
python3 eda_analysis.py
```

#### 4. Feature Engineering
```bash
python3 feature_engineering.py
```

#### 5. Train Models
```bash
python3 model_training.py
```

## Models Implemented

1. **Linear Regression**: Baseline model for linear relationships
2. **Decision Tree**: Captures non-linear patterns
3. **Random Forest**: Ensemble method for robust predictions

## Key Findings

### Correlation Analysis
- **SINR** shows strongest correlation with throughput (positive)
- **Network Load** inversely correlates with throughput
- **CQI** is a strong predictor of both throughput and latency
- **RSRP** has moderate correlation with QoS metrics

### Model Performance
Models achieve strong predictive performance:
- **Throughput Prediction**: R² > 0.85 (Random Forest)
- **Latency Prediction**: R² > 0.80 (Random Forest)

### Impact of Network Load
- High network load (>60%) significantly degrades throughput
- Latency increases exponentially under heavy load
- Physical-layer metrics alone are insufficient without load context

## Dataset Characteristics

- **Samples**: 10,000 time-series measurements
- **Sampling Interval**: 100ms
- **Realistic Correlations**: Shannon capacity-based throughput modeling
- **Temporal Patterns**: Diurnal load variations (peak hours simulation)
- **Data Quality**: Includes missing values (2%) and outliers (1%)

## Visualization Outputs

1. **Distribution Plots**: Histograms of all key metrics
2. **Correlation Matrix**: Heatmap showing metric relationships
3. **Scatter Plots**: RAN metrics vs QoS metrics
4. **Time Series**: Temporal patterns in measurements
5. **Load Impact**: Box plots showing QoS degradation under load
6. **Prediction Plots**: Actual vs predicted values for all models

## Future Enhancements

- Deep learning models (LSTM, GRU) for time-series prediction
- Real-world dataset integration
- Multi-UE scenario modeling
- Handover event detection and impact analysis
- Real-time prediction API

## Technical Details

### Data Generation
The synthetic dataset uses realistic 5G NR parameters:
- RSRP range: -140 to -44 dBm
- SINR range: -10 to 30 dB
- Shannon capacity approximation with practical efficiency factors
- Temporal correlation modeling for mobility patterns

### Feature Engineering
- **Rolling Windows**: 5, 10, 20 samples
- **Rate of Change**: 1, 5, 10 sample periods
- **Stability Window**: 10 samples
- Total engineered features: 80+

### Model Training
- Train/Val/Test split: 70/10/20
- Feature standardization using StandardScaler
- Hyperparameter tuning for tree-based models
- Cross-validation ready architecture


## References

### 3GPP Technical Specifications
1. **3GPP TS 38.214** – *NR; Physical Layer Procedures for Data*. Defines CQI-to-MCS mapping tables, resource block allocation, and modulation schemes used in 5G NR.
   [https://www.3gpp.org/dynareport/38214.htm](https://www.3gpp.org/dynareport/38214.htm)

2. **3GPP TS 38.331** – *NR; Radio Resource Control (RRC) Protocol Specification*. Covers RSRP/RSRQ/SINR measurement reporting and RRC state management.
   [https://www.3gpp.org/dynareport/38331.htm](https://www.3gpp.org/dynareport/38331.htm)

3. **3GPP TS 38.133** – *NR; Requirements for Support of Radio Resource Management*. Specifies RSRP range (−156 to −31 dBm for SS-RSRP) and SINR measurement requirements.
   [https://www.3gpp.org/dynareport/38133.htm](https://www.3gpp.org/dynareport/38133.htm)

4. **3GPP TS 23.501** – *System Architecture for the 5G System*. Defines QoS models, 5QI values, and end-to-end QoS framework in 5G.
   [https://www.3gpp.org/dynareport/23501.htm](https://www.3gpp.org/dynareport/23501.htm)

### Foundational Theory
5. **Shannon, C.E.** (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal, 27(3), 379–423. Establishes the theoretical channel capacity formula C = B × log₂(1 + SNR) used for throughput estimation.
   [https://doi.org/10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)

### Machine Learning & Tools
6. **Pedregosa, F. et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. Library used for Linear Regression, Decision Tree, and Random Forest model implementations.
   [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

7. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5–32. Foundational paper for the Random Forest ensemble method used in this project.
   [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

### Related Research
8. **Narayanan, A. et al.** (2020). *A First Look at Commercial 5G Performance on Smartphones*. Proceedings of The Web Conference 2020 (WWW '20). Provides real-world 5G throughput and latency measurements for benchmarking.
   [https://doi.org/10.1145/3366423.3380169](https://doi.org/10.1145/3366423.3380169)

9. **Mei, J. et al.** (2021). *Realtime Mobile Bandwidth Prediction Using LSTM Neural Network and Bayesian Fusion*. Computer Networks, 182, 107515. Explores ML-based mobile network QoS prediction similar to this project's approach.
   [https://doi.org/10.1016/j.comnet.2020.107515](https://doi.org/10.1016/j.comnet.2020.107515)

10. **Zhang, C. et al.** (2019). *Deep Learning in Mobile and Wireless Networking: A Survey*. IEEE Communications Surveys & Tutorials, 21(3), 2224–2287. Comprehensive survey covering ML/DL techniques for wireless network performance prediction.
    [https://doi.org/10.1109/COMST.2019.2904897](https://doi.org/10.1109/COMST.2019.2904897)
