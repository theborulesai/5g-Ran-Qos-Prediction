# feature_engineering.py
# creates new features from the raw RAN metrics
# rolling stats, rate of change, stability, and some interaction terms

import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self):
        self.feature_names = []

    def create_rolling_features(self, df, columns, windows=[5, 10, 20]):
        # rolling mean, std, min, max for each column across different window sizes
        # this captures short-term and medium-term trends in the signal
        df_features = df.copy()

        for col in columns:
            for w in windows:
                df_features[f'{col}_rolling_mean_{w}'] = df[col].rolling(window=w, min_periods=1).mean()
                df_features[f'{col}_rolling_std_{w}'] = df[col].rolling(window=w, min_periods=1).std()
                df_features[f'{col}_rolling_min_{w}'] = df[col].rolling(window=w, min_periods=1).min()
                df_features[f'{col}_rolling_max_{w}'] = df[col].rolling(window=w, min_periods=1).max()

        return df_features

    def create_rate_of_change_features(self, df, columns, periods=[1, 5]):
        # first difference and percentage change - tells the model how fast things are changing
        df_features = df.copy()

        for col in columns:
            for p in periods:
                df_features[f'{col}_diff_{p}'] = df[col].diff(periods=p)
                df_features[f'{col}_pct_change_{p}'] = df[col].pct_change(periods=p)

        df_features = df_features.bfill()
        df_features = df_features.replace([np.inf, -np.inf], 0)

        return df_features

    def create_stability_features(self, df, columns, window=10):
        # coefficient of variation = std / mean, lower means more stable signal
        df_features = df.copy()

        for col in columns:
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            df_features[f'{col}_stability_{window}'] = rolling_std / (rolling_mean + 1e-6)

        return df_features

    def create_interaction_features(self, df):
        # combinations of features that might be more informative than individual ones
        df_features = df.copy()

        df_features['rsrp_sinr_product'] = df['rsrp_dbm'] * df['sinr_db']
        df_features['cqi_load_ratio'] = df['cqi'] / (df['network_load'] + 0.1)
        df_features['sinr_load_ratio'] = df['sinr_db'] / (df['network_load'] + 0.1)
        df_features['rb_per_cqi'] = df['rb_allocation'] / (df['cqi'] + 1)

        # composite signal quality score - weighted combination of RSRP, SINR, CQI
        df_features['signal_quality_score'] = (
            (df['rsrp_dbm'] + 100) / 60 * 0.3 +
            (df['sinr_db'] + 10) / 40 * 0.4 +
            df['cqi'] / 15 * 0.3
        )

        return df_features

    def create_time_features(self, df):
        # extract time-of-day info - useful since network load follows a daily pattern
        df_features = df.copy()

        df_features['hour'] = df['timestamp'].dt.hour
        df_features['minute'] = df['timestamp'].dt.minute
        df_features['day_of_week'] = df['timestamp'].dt.dayofweek

        # encode hour cyclically so 23:00 and 00:00 are close to each other
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

        # flag evening peak hours (5 PM to 9 PM)
        df_features['is_peak_hour'] = ((df_features['hour'] >= 17) & (df_features['hour'] <= 21)).astype(int)

        return df_features

    def create_all_features(self, df):
        print("engineering features...")
        df_features = df.copy()

        ran_metrics = ['rsrp_dbm', 'sinr_db', 'cqi']

        print("  rolling stats...")
        df_features = self.create_rolling_features(df_features, ran_metrics, windows=[5, 10, 20])

        print("  rate of change...")
        df_features = self.create_rate_of_change_features(df_features, ran_metrics, periods=[1, 5])

        print("  stability metrics...")
        df_features = self.create_stability_features(df_features, ran_metrics, window=10)

        print("  interaction features...")
        df_features = self.create_interaction_features(df_features)

        print("  time features...")
        df_features = self.create_time_features(df_features)

        # clean up any remaining NaN or inf
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)

        exclude_cols = ['timestamp', 'throughput_mbps', 'latency_ms']
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
