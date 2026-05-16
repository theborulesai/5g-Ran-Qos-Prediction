# feature_engineering.py
# creates extra features from the pcap-derived network measurements
# things like rolling averages, rate of change, stability metrics etc
# adapted for real 5G RAN SCTP traffic features

import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self):
        self.feature_names = []

    def create_rolling_features(self, df, columns, windows=[5, 10, 20]):
        # rolling stats (mean, std etc) at different window sizes
        # captures both short-term spikes and medium-term trends
        df_features = df.copy()

        for col in columns:
            for w in windows:
                df_features[f'{col}_rolling_mean_{w}'] = df[col].rolling(window=w, min_periods=1).mean()
                df_features[f'{col}_rolling_std_{w}'] = df[col].rolling(window=w, min_periods=1).std()
                df_features[f'{col}_rolling_min_{w}'] = df[col].rolling(window=w, min_periods=1).min()
                df_features[f'{col}_rolling_max_{w}'] = df[col].rolling(window=w, min_periods=1).max()

        return df_features

    def create_rate_of_change_features(self, df, columns, periods=[1, 5]):
        # first difference and percentage change
        df_features = df.copy()

        for col in columns:
            for p in periods:
                df_features[f'{col}_diff_{p}'] = df[col].diff(periods=p)
                df_features[f'{col}_pct_change_{p}'] = df[col].pct_change(periods=p)

        df_features = df_features.bfill()
        df_features = df_features.replace([np.inf, -np.inf], 0)

        return df_features

    def create_stability_features(self, df, columns, window=10):
        # coefficient of variation = std / mean, lower means more stable
        df_features = df.copy()

        for col in columns:
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            df_features[f'{col}_stability_{window}'] = rolling_std / (rolling_mean + 1e-6)

        return df_features

    def create_interaction_features(self, df):
        # feature combos that capture network behavior relationships
        df_features = df.copy()

        # packet size relative to throughput — efficiency indicator
        df_features['size_throughput_ratio'] = df['packet_size'] / (df['throughput_kbps'] + 0.001)

        # jitter relative to latency — how unstable is the delay
        df_features['jitter_latency_ratio'] = df['jitter_ms'] / (df['latency_ms'] + 0.001)

        # inter-arrival relative to network load
        df_features['arrival_load_ratio'] = df['inter_arrival_ms'] / (df['network_load'] + 0.01)

        # packet rate to throughput ratio — overhead indicator
        df_features['pps_throughput_ratio'] = df['packet_rate_pps'] / (df['throughput_kbps'] + 0.001)

        # composite network quality score
        # low jitter + low latency + high throughput = good quality
        jitter_norm = 1 - np.clip(df['jitter_ms'] / (df['jitter_ms'].quantile(0.99) + 0.001), 0, 1)
        latency_norm = 1 - np.clip(df['latency_ms'] / (df['latency_ms'].quantile(0.99) + 0.001), 0, 1)
        tp_norm = np.clip(df['throughput_kbps'] / (df['throughput_kbps'].quantile(0.99) + 0.001), 0, 1)
        df_features['network_quality_score'] = 0.3 * jitter_norm + 0.3 * latency_norm + 0.4 * tp_norm

        return df_features

    def create_time_features(self, df):
        # extract time-of-day info
        df_features = df.copy()

        df_features['hour'] = df['timestamp'].dt.hour
        df_features['minute'] = df['timestamp'].dt.minute
        df_features['second'] = df['timestamp'].dt.second

        # encode hour cyclically so 23:00 and 00:00 are close
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

        # fractional time within the capture for trend detection
        start_time = df['timestamp'].min()
        df_features['elapsed_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

        return df_features

    def create_all_features(self, df):
        print("engineering features...")
        df_features = df.copy()

        # key metrics to create rolling/rate/stability features for
        net_metrics = ['packet_size', 'inter_arrival_ms', 'jitter_ms',
                       'throughput_kbps', 'packet_rate_pps']

        print("  rolling stats...")
        df_features = self.create_rolling_features(df_features, net_metrics, windows=[5, 10, 20])

        print("  rate of change...")
        df_features = self.create_rate_of_change_features(df_features, net_metrics, periods=[1, 5])

        print("  stability metrics...")
        df_features = self.create_stability_features(df_features, net_metrics, window=10)

        print("  interaction features...")
        df_features = self.create_interaction_features(df_features)

        print("  time features...")
        df_features = self.create_time_features(df_features)

        # clean up any remaining NaN or inf
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)

        exclude_cols = ['timestamp', 'throughput_kbps', 'latency_ms']
        self.feature_names = [c for c in df_features.columns if c not in exclude_cols]

        print(f"  done: {len(self.feature_names)} features total")
        return df_features

    def get_feature_names(self):
        return self.feature_names


if __name__ == "__main__":
    df = pd.read_csv('../data/5g_ran_dataset_clean.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    df_features.to_csv('../data/5g_ran_dataset_features.csv', index=False)

    print(f"\nshape: {df_features.shape}")
    print(f"sample feature names: {engineer.get_feature_names()[:10]}")
