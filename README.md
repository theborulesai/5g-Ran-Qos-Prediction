# 5G RAN QoS Prediction — Mini Project

## Overview

This is my mini project on predicting **throughput and latency** in 5G Radio Access Networks using machine learning. Instead of synthetic data, **we use real captured network traffic** — a PCAP file containing SCTP packets from 5G NGAP (gNB ↔ AMF) and F1AP (CU ↔ DU) interfaces.

I implemented the whole pipeline from scratch: PCAP parsing with Scapy, data cleaning, exploratory analysis, feature engineering (ended up with 106 features), and then training **3 ML algorithms** — Linear Regression (Ridge), Decision Tree, and Random Forest. The project produces **12 graphs** that cover everything from data distributions to model comparison.

Random Forest turned out to be the best model (R²=0.9999 for throughput, R²=0.9525 for latency). All three models achieved excellent results because the features extracted from real PCAP traffic have strong predictive power.

---

## Data Source

The project uses a **real 5G RAN packet capture** (`dataset .pcap`), not synthetic data.

| Property | Value |
|----------|-------|
| File | `dataset .pcap` |
| Size | ~707 KB |
| Protocol | SCTP (Stream Control Transmission Protocol) |
| 5G Interfaces | NGAP (ports 38412/38413) and F1AP (ports 38472/38473) |
| Total packets | 5,386 |
| Packets after cleaning | 4,728 (~12% outliers removed) |

The PCAP contains real SCTP traffic between 5G network elements — gNB, AMF, CU, and DU. We extract per-packet features like packet size, inter-arrival time, jitter, and then compute windowed metrics like throughput and latency estimates using multiple methods (HEARTBEAT/ACK RTT matching, SACK-based flow pairing, and inter-arrival delay estimation).

---

## Project Structure

```
5g-Ran-Qos-Prediction/
├── dataset .pcap                      # Real 5G RAN SCTP packet capture
├── data/                              # Parsed and processed datasets
│   ├── 5g_ran_dataset.csv            # Raw parsed data (5,386 samples)
│   ├── 5g_ran_dataset_clean.csv      # Cleaned data (4,728 samples after outlier removal)
│   └── 5g_ran_dataset_features.csv   # Feature-engineered data (106 features)
├── models/                            # Trained ML models (8 .pkl files)
│   ├── throughput_lr_model.pkl
│   ├── throughput_dt_model.pkl
│   ├── throughput_rf_model.pkl
│   ├── latency_lr_model.pkl
│   ├── latency_dt_model.pkl
│   ├── latency_rf_model.pkl
│   ├── throughput_kbps_scaler.pkl    # StandardScaler for throughput features
│   └── latency_ms_scaler.pkl         # StandardScaler for latency features
├── results/                           # 12 output graphs (.png)
│   ├── distributions.png             # Graph 1
│   ├── correlation_matrix.png        # Graph 2
│   ├── throughput_scatter.png        # Graph 3
│   ├── latency_scatter.png           # Graph 4
│   ├── time_series.png               # Graph 5
│   ├── load_impact.png               # Graph 6
│   ├── model_comparison.png          # Graph 7
│   ├── throughput_predictions.png    # Graph 8
│   ├── latency_predictions.png       # Graph 9
│   ├── residual_analysis.png         # Graph 10
│   ├── feature_importance.png        # Graph 11
│   └── learning_curves.png           # Graph 12
├── src/                               # Source code (6 Python modules)
│   ├── pcap_parser.py                # PCAP parsing with Scapy — extracts 5G traffic features
│   ├── data_preprocessing.py         # Cleaning, outlier removal, validation
│   ├── eda_analysis.py               # Exploratory data analysis (Graphs 1–6)
│   ├── feature_engineering.py        # Feature creation (106 engineered features)
│   ├── model_training.py             # ML model training + evaluation (Graphs 7–12)
│   └── main_pipeline.py              # End-to-end pipeline orchestration
├── website/                           # Project showcase web page
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Input Features & Targets

### PCAP-Derived Network Metrics (Inputs)

These features are extracted directly from real SCTP packet captures:

| Metric | Description | Source |
|--------|-------------|--------|
| **packet_size** | Total packet length in bytes | Per-packet measurement |
| **ip_payload_size** | IP payload size in bytes | IP header extraction |
| **inter_arrival_ms** | Time between consecutive packets (ms) | Timestamp differencing |
| **jitter_ms** | Variation in inter-arrival time (ms) | Absolute change in inter-arrival |
| **is_burst** | Whether packet is part of a burst (IAT < 1ms) | Threshold detection |
| **direction** | Uplink (1) or Downlink (0) | IP address comparison |
| **interface_code** | NGAP=0, F1AP=1, OTHER=2 | SCTP port mapping |
| **chunk_code** | SCTP chunk type (SACK, HEARTBEAT, DATA, etc.) | SCTP header parsing |
| **network_load** | Normalised packet rate (0–1) | Packet rate / 99th percentile |
| **packet_rate_pps** | Packets per second (sliding window) | Windowed computation |

### QoS Prediction Targets (Outputs)

| Metric | Unit | Description |
|--------|------|-------------|
| **Throughput** | Kbps | Windowed throughput computed from packet sizes over a 5-second sliding window |
| **Latency** | ms | RTT estimated from HEARTBEAT/ACK matching, SACK flow pairing, and inter-arrival delay |

### Engineered Features (106 total)

| Group | Count | Description |
|-------|------:|-------------|
| Rolling window stats | 60 | Mean, std, min, max of 5 metrics at windows 5, 10, 20 |
| Rate of change | 20 | diff and pct_change at periods 1 and 5 for 5 network metrics |
| Stability metrics | 5 | Coefficient of variation (window 10) for 5 network metrics |
| Interaction terms | 5 | size/throughput ratio, jitter/latency ratio, arrival/load ratio, pps/throughput ratio, network quality score |
| Time features | 6 | hour, minute, second, hour_sin, hour_cos, elapsed_seconds |
| Original metrics | 10 | packet_size, ip_payload_size, inter_arrival_ms, jitter_ms, is_burst, direction, interface_code, chunk_code, network_load, packet_rate_pps |

---

## Installation

### Environment

| Item | Version |
|------|---------:|
| Python | ≥ 3.10 (tested on 3.13) |
| OS | Linux / macOS / Windows |

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy>=1.24 pandas>=2.0 scikit-learn>=1.1 matplotlib>=3.6 seaborn>=0.12 scipy>=1.9 scapy>=2.5
```

### Dependencies

| Package | Minimum | Tested | Purpose |
|---------|---------|--------|---------|
| numpy | ≥ 1.24.0 | 2.2.4 | Numerical operations |
| pandas | ≥ 2.0.0 | 2.2.3 | Data manipulation |
| scikit-learn | ≥ 1.1.0 | 1.4.2 | ML models & evaluation |
| matplotlib | ≥ 3.6.0 | 3.10.1 | Graph plotting |
| seaborn | ≥ 0.12.0 | 0.13.2 | Statistical visualizations |
| scipy | ≥ 1.9.0 | 1.15.3 | Statistical analysis |
| scapy | ≥ 2.5.0 | 2.7.0 | PCAP parsing |

> **Note:** `scapy` is required for parsing the PCAP file. It reads raw packet data and extracts IP/SCTP headers.

---

## How to Run

---

### 🐍 Run the ML Pipeline (Python Code)

#### Prerequisites

```bash
# 1. Clone / navigate to the project folder
cd "5g-Ran-Qos-Prediction"

# 2. (Optional but recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

#### Option A — Full Pipeline (Recommended)

Runs all 5 stages end-to-end in one command:

```bash
cd src
python3 main_pipeline.py
```

**Expected console output:**

```
============================================================
  5G RAN QoS Prediction — Full Pipeline
  Data Source: Real PCAP capture (NGAP/F1AP SCTP traffic)
  Algorithms: Linear Regression, Decision Tree, Random Forest
  Graphs: 12 significant plots
============================================================

[1/5] Parsing PCAP file...
  loading pcap: ../dataset .pcap
  loaded 5386 packets
  extracted 5386 packet records
  computing timing features...
  computing latency metrics...
  matched 2311 heartbeat RTT pairs
  matched 125 SACK/flow-based RTT estimates
  final dataset: (5386, 13)

[2/5] Preprocessing...
  loaded 5386 samples
  detecting and removing outliers...
  removing 658 rows (12.22%)
  final shape: (4728, 13)

[3/5] Exploratory Data Analysis (Graphs 1-6)...
  saved: distributions.png
  saved: correlation_matrix.png
  saved: throughput_scatter.png
  saved: latency_scatter.png
  saved: time_series.png
  saved: load_impact.png

[4/5] Feature engineering...
  done: 106 features total

[5/5] Training 3 Models + Generating Graphs 7-12...
  --- Throughput models ---
  --- Latency models ---
  saving models...
  generating model evaluation graphs...

============================================================
  Pipeline complete!
  Data Source: dataset .pcap (real 5G RAN SCTP capture)
  Algorithms: Linear Regression, Decision Tree, Random Forest
  Graphs: 12 (see results/ folder)
  Data:   data/
  Models: models/
  Plots:  results/
============================================================
```

**Files produced after running:**

| Location | Files |
|----------|-------|
| `data/` | `5g_ran_dataset.csv`, `5g_ran_dataset_clean.csv`, `5g_ran_dataset_features.csv` |
| `models/` | 6 `.pkl` model files + 2 scaler files |
| `results/` | 12 `.png` graph files |

---

#### Option B — Run Each Step Individually

```bash
cd src

# Step 1 — Parse PCAP file (saves to data/5g_ran_dataset.csv)
python3 pcap_parser.py

# Step 2 — Clean and validate (saves to data/5g_ran_dataset_clean.csv)
python3 data_preprocessing.py

# Step 3 — EDA graphs 1–6 (saves 6 .png to results/)
python3 eda_analysis.py

# Step 4 — Feature engineering (saves to data/5g_ran_dataset_features.csv)
python3 feature_engineering.py

# Step 5 — Train models + graphs 7–12 (saves 8 .pkl to models/ and 6 .png to results/)
python3 model_training.py
```

> **Order matters** — each step reads the output of the previous one.

---

### 🌐 Run the Website (Project Showcase)

The website shows all 12 graphs, model results, and project details in a nice UI.
**Run the pipeline first** so the graph images exist in `results/`.

#### How to open the website:

```bash
# Step 1: Open terminal and go to the project folder
cd "5g-Ran-Qos-Prediction"

# Step 2: Start a local server (from the project root, NOT from website/)
python3 -m http.server 8080

# Step 3: Open this URL in your browser
# http://localhost:8080/website/index.html
```

That's it. The website will open with all 12 graphs loading correctly.

> **Important:** Always start the server from the **project root folder** (where `README.md` is), not from inside `website/`. The graphs are loaded from `../results/` so the server needs to be one level above.

> **If you get "Address already in use" error:** Another server is already running on port 8080. Kill it first with `pkill -f "http.server 8080"` and try again.

> **To stop the server:** Press `Ctrl+C` in the terminal.

---

### 📁 File Output Summary

```
After running the pipeline:

data/
  ├── 5g_ran_dataset.csv              ← raw parsed PCAP data (5,386 samples)
  ├── 5g_ran_dataset_clean.csv        ← cleaned data (4,728 samples)
  └── 5g_ran_dataset_features.csv     ← 106 engineered features

models/
  ├── throughput_lr_model.pkl         ← Linear Regression / Ridge (throughput)
  ├── throughput_dt_model.pkl         ← Decision Tree (throughput)
  ├── throughput_rf_model.pkl         ← Random Forest (throughput)
  ├── latency_lr_model.pkl            ← Linear Regression / Ridge (latency)
  ├── latency_dt_model.pkl            ← Decision Tree (latency)
  ├── latency_rf_model.pkl            ← Random Forest (latency)
  ├── throughput_kbps_scaler.pkl      ← StandardScaler (throughput)
  └── latency_ms_scaler.pkl           ← StandardScaler (latency)

results/
  ├── distributions.png               ← Graph 1
  ├── correlation_matrix.png          ← Graph 2
  ├── throughput_scatter.png          ← Graph 3
  ├── latency_scatter.png             ← Graph 4
  ├── time_series.png                 ← Graph 5
  ├── load_impact.png                 ← Graph 6
  ├── model_comparison.png            ← Graph 7
  ├── throughput_predictions.png      ← Graph 8
  ├── latency_predictions.png         ← Graph 9
  ├── residual_analysis.png           ← Graph 10
  ├── feature_importance.png          ← Graph 11
  └── learning_curves.png             ← Graph 12
```

---

## 3 ML Algorithms

### Algorithm 1 — Linear Regression (Ridge)
- **Type:** Parametric, regularized linear model
- **How:** Ridge regression (L2 regularization) — prevents coefficient explosion from multicollinear features
- **Why:** Provides an interpretable baseline; handles our 106 correlated features better than plain OLS
- **Params:** `alpha=1.0`
- `sklearn.linear_model.Ridge()`

### Algorithm 2 — Decision Tree Regressor
- **Type:** Non-parametric, tree-based
- **How:** Recursively splits feature space on variance-minimizing thresholds
- **Hyperparameters:** `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`
- `sklearn.tree.DecisionTreeRegressor()`

### Algorithm 3 — Random Forest Regressor *(Best)*
- **Type:** Ensemble (bagging of decision trees)
- **How:** Averages predictions from 100 trees trained on random subsets — reduces overfitting
- **Hyperparameters:** `n_estimators=100`, `max_depth=15`, `min_samples_split=20`, `n_jobs=-1`
- `sklearn.ensemble.RandomForestRegressor()`

---

## Model Performance Results (Test Set)

### Throughput Prediction (Kbps)

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|----|----------|
| Linear Regression (Ridge) | 9.9748 | 7.6656 | 0.9998 | 284.44 |
| Decision Tree | 18.1900 | 2.4451 | 0.9994 | 4.31 |
| **Random Forest** | **5.8880** | **1.3171** | **0.9999** | **1.70** |

### Latency Prediction (ms)

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|----|----------|
| Linear Regression (Ridge) | 0.0013 | 0.0008 | 0.9788 | 3.86 |
| Decision Tree | 0.0024 | 0.0011 | 0.9215 | 5.82 |
| **Random Forest** | **0.0019** | **0.0008** | **0.9525** | **4.13** |

> Train/val/test split is **70 / 10 / 20 %** applied **chronologically** (no shuffle) to respect the time-series nature of the data.

### What I Noticed

- **Random Forest** wins on throughput — best R² (0.9999) and lowest RMSE (5.89), which makes sense for an ensemble method
- **All models achieve very high R²** because the PCAP-derived features have strong direct relationships with the computed QoS metrics
- **Linear Regression (Ridge)** has a high MAPE on throughput (284%) — this is because some throughput values are very close to zero (near-idle periods), making percentage error blow up, even though the absolute error (MAE=7.67) is quite small
- **Latency prediction** shows Ridge performing best overall (R²=0.9788) — the linear model captures the latency patterns well since RTT correlates linearly with inter-arrival and jitter features
- **Decision Tree** tends to overfit slightly (lower test R² compared to train) — RF bagging corrects this

---

## 12 Significant Graphs

### EDA Graphs (1–6)

| # | Graph | File | What It Shows |
|---|-------|------|---------------|
| 1 | Feature Distributions | `distributions.png` | Histograms of packet_size, inter_arrival_ms, jitter_ms, network_load, throughput_kbps, latency_ms with mean/median lines |
| 2 | Correlation Matrix | `correlation_matrix.png` | Lower-triangle Pearson correlation heatmap of all network metrics |
| 3 | Throughput Scatter | `throughput_scatter.png` | 5 scatter plots (packet_size, inter_arrival, jitter, load, packet_rate vs throughput) with trend lines |
| 4 | Latency Scatter | `latency_scatter.png` | 5 scatter plots (same metrics vs latency) with trend lines |
| 5 | Time Series | `time_series.png` | First 1,000 packets of all 6 metrics over time — shows traffic patterns |
| 6 | Network Load Impact | `load_impact.png` | Box plots of throughput & latency grouped by Low / Medium / High load |

### Model Evaluation Graphs (7–12)

| # | Graph | File | What It Shows |
|---|-------|------|---------------|
| 7 | Model Comparison | `model_comparison.png` | R², RMSE, MAE bar chart — 3 algorithms × 2 targets = 6 subplots |
| 8 | Throughput Predictions | `throughput_predictions.png` | Actual vs predicted scatter for all 3 models on throughput test set |
| 9 | Latency Predictions | `latency_predictions.png` | Actual vs predicted scatter for all 3 models on latency test set |
| 10 | Residual Analysis | `residual_analysis.png` | Residual histograms — 3 models × 2 targets, centred residuals verify no bias |
| 11 | Feature Importance | `feature_importance.png` | Top 15 Random Forest features for both throughput and latency |
| 12 | Learning Curves | `learning_curves.png` | Train R² vs CV R² over training set size — shows convergence & overfitting |

---

## Key Findings

### Correlation Analysis
- **packet_rate_pps** → strongest predictor of throughput (more packets = higher throughput)
- **Network Load** → derived from packet rate; highly correlated with throughput
- **inter_arrival_ms / jitter_ms** → correlate with latency estimates; longer gaps between packets indicate higher delay
- **packet_size** → moderate correlation with throughput; larger SCTP DATA chunks carry more payload

### Network Load Impact on QoS

| Load Level | Throughput | Latency |
|------------|-----------|---------|
| Low (< 30%) | Lower throughput (less traffic) | Lower latency |
| Medium (30–60%) | Moderate throughput | Moderate latency |
| High (> 60%) | Higher throughput (more active traffic) | Higher latency |

### Feature Importance (Random Forest)
- **Throughput:** Top features are rolling window stats of packet_rate_pps, packet_size, and their interaction terms
- **Latency:** Top features are inter_arrival_ms rolling stats, jitter stability metrics, and network quality score
- Rolling window features at window=20 capture medium-term trends that improve predictions beyond raw values

---

## How the PCAP Parsing Works

The `pcap_parser.py` module uses Scapy to extract features from real 5G RAN traffic:

### Step 1: Per-Packet Feature Extraction
- Read each packet from the PCAP file
- Extract IP headers (src/dst IP, TTL, payload size)
- Parse SCTP headers manually (src/dst port, verification tag, chunk type)
- Identify 5G interface from SCTP port (38412/38413 → NGAP, 38472/38473 → F1AP)
- Determine flow direction from IP address ordering

### Step 2: Timing Features
- **Inter-arrival time:** Difference in timestamps between consecutive packets
- **Jitter:** Absolute change in inter-arrival time (variation in delay)
- **Burst detection:** Packets arriving within 1ms flagged as burst traffic

### Step 3: Latency Estimation (Multi-Method)
Three methods are used to estimate RTT/latency for every packet:

1. **HEARTBEAT → HEARTBEAT_ACK RTT:** Direct RTT measurement by matching SCTP heartbeat requests to their acknowledgements on the reverse path (matched 2,311 pairs)
2. **SACK-based flow pairing:** Approximate RTT from consecutive packets going in opposite directions within the same flow (matched 125 pairs)
3. **Inter-arrival delay proxy:** For remaining packets, estimate latency from flow-level inter-arrival patterns, calibrated against the known RTT measurements

### Step 4: Windowed Metrics
- **Throughput:** Sliding window (5 seconds) of total bytes × 8 / duration → bits per second → Kbps
- **Packet rate:** Packets per second within the sliding window
- **Network load:** Normalised packet rate (0–1)

---

## Dataset Properties

| Property | Value |
|----------|-------|
| Data source | Real 5G RAN PCAP capture (`dataset .pcap`) |
| Protocol | SCTP over IP |
| 5G interfaces | NGAP (gNB ↔ AMF) and F1AP (CU ↔ DU) |
| Total raw packets | 5,386 |
| Samples after cleaning | 4,728 (~12.2% removed as outliers) |
| Features after engineering | 106 |
| Latency estimation | HEARTBEAT/ACK RTT + SACK flow pairing + inter-arrival proxy |
| Throughput computation | 5-second sliding window, bytes × 8 / duration |
| Outlier detection | IQR method, threshold = 3.0 |
| Missing value handling | Linear interpolation |

---

## Technical Notes

### PCAP Parsing
- Uses Scapy's `rdpcap()` to load the full capture into memory
- SCTP headers are parsed manually from raw IP payload bytes (Scapy doesn't fully decode SCTP)
- SCTP chunk types identified: DATA, INIT, INIT_ACK, SACK, HEARTBEAT, HEARTBEAT_ACK, ABORT, SHUTDOWN, COOKIE_ECHO, COOKIE_ACK
- 5G interface detected from SCTP destination/source port matching against known NGAP/F1AP port numbers

### Data Preprocessing
- Interpolation method: `linear` (pandas `interpolate()`) — forward + backward fill for edge cases
- Outlier removal: IQR × 3.0 on inter_arrival_ms, jitter_ms, throughput_kbps, latency_ms columns
- Range validation against expected bounds after cleaning

### Model Training
- Chronological 70/10/20 train/val/test split — `shuffle=False` to preserve time ordering
- `StandardScaler` fitted on train set, applied to val and test (no data leakage)
- Ridge regression used instead of plain LinearRegression to handle multicollinearity in 106 features
- Models and scalers serialised as `.pkl` files in `models/`
- Learning curves use 3-fold CV on a 2,000-sample subsample for speed

---

## What I Would Do Next (Future Work)

- Try LSTM or GRU networks — they should capture temporal patterns better since this is time-series packet data
- Collect more PCAP captures from different 5G scenarios (high load, handover, mobility)
- Add gradient boosting models (XGBoost, LightGBM) for potentially better performance
- Include upper-layer protocol features (e.g., HTTP response times) for more accurate latency prediction
- Build a real-time prediction system using live packet capture with tshark/Scapy
- Maybe build a simple REST API with FastAPI so the model can predict QoS in real-time from live traffic

---

## References

### 3GPP Technical Specifications
1. **3GPP TS 38.413** – *NG Application Protocol (NGAP)*. Defines the NGAP signalling between gNB and AMF over SCTP.
   https://www.3gpp.org/dynareport/38413.htm

2. **3GPP TS 38.473** – *F1 Application Protocol (F1AP)*. Defines the F1AP signalling between CU and DU over SCTP.
   https://www.3gpp.org/dynareport/38473.htm

3. **3GPP TS 38.401** – *NR; Architecture Description*. Overall 5G RAN architecture (gNB split into CU and DU).
   https://www.3gpp.org/dynareport/38401.htm

4. **3GPP TS 23.501** – *System Architecture for the 5G System*. QoS models and 5QI values.
   https://www.3gpp.org/dynareport/23501.htm

### Protocol Specifications
5. **RFC 4960** – *Stream Control Transmission Protocol (SCTP)*. Defines SCTP chunk types (DATA, SACK, HEARTBEAT, etc.) used for parsing.
   https://www.rfc-editor.org/rfc/rfc4960

### Machine Learning Tools
6. **Pedregosa, F. et al.** (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
   https://jmlr.org/papers/v12/pedregosa11a.html

7. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
   https://doi.org/10.1023/A:1010933404324

### Packet Capture Tools
8. **Scapy** – *Interactive Packet Manipulation Library*. Used for PCAP parsing and IP/SCTP header extraction.
   https://scapy.net/

### Related Research
9. **Narayanan, A. et al.** (2020). *Lumos5G: Mapping and Predicting Commercial mmWave 5G Throughput*. ACM Internet Measurement Conference (IMC) 2020.
   https://doi.org/10.1145/3419394.3423629

10. **Mei, J. et al.** (2021). *Realtime Mobile Bandwidth Prediction Using LSTM Neural Network and Bayesian Fusion*. Computer Networks, 182.
    https://doi.org/10.1016/j.comnet.2020.107515
