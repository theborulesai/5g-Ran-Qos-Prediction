# data_preprocessing.py
# cleans up the raw pcap-derived data - fills missing values, removes outliers etc
# adapted for real 5G RAN SCTP traffic features extracted from dataset.pcap

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def handle_missing_values(self, df, method='interpolate'):
        # linear interpolation works best for time-series stuff
        df_copy = df.copy()
        print(f"missing values before:\n{df_copy.isnull().sum()}\n")

        if method == 'interpolate':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].interpolate(
                method='linear', limit_direction='both')
        elif method == 'forward_fill':
            df_copy = df_copy.ffill().bfill()
        elif method == 'drop':
            df_copy = df_copy.dropna()

        print(f"missing values after:\n{df_copy.isnull().sum()}\n")
        return df_copy

    def detect_outliers(self, df, columns, method='iqr', threshold=1.5):
        # using IQR method - anything outside Q1-1.5*IQR to Q3+1.5*IQR is outlier
        outlier_indices = set()

        for col in columns:
            if col not in df.columns:
                continue
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)].index
            elif method == 'zscore':
                z = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z > threshold].index

            outlier_indices.update(outliers)
            print(f"  {col}: {len(outliers)} outliers found")

        return list(outlier_indices)

    def remove_outliers(self, df, columns, method='iqr', threshold=3.0):
        outlier_idx = self.detect_outliers(df, columns, method, threshold)
        print(f"\nremoving {len(outlier_idx)} rows ({len(outlier_idx)/len(df)*100:.2f}%)")
        return df.drop(outlier_idx).reset_index(drop=True)

    def validate_data(self, df):
        # check that all values are within expected ranges for pcap-derived metrics
        valid_ranges = {
            'packet_size': (0, 10000),
            'ip_payload_size': (0, 10000),
            'inter_arrival_ms': (0, 60000),       # up to 60 seconds between packets
            'jitter_ms': (0, 60000),
            'network_load': (0, 1),
            'throughput_kbps': (0, 1e6),           # up to 1 Gbps
            'latency_ms': (0, 500),
            'packet_rate_pps': (0, 100000),
        }

        issues = []
        for col, (lo, hi) in valid_ranges.items():
            if col in df.columns:
                bad = df[(df[col] < lo) | (df[col] > hi)]
                if len(bad) > 0:
                    issues.append(f"{col}: {len(bad)} values out of range [{lo}, {hi}]")

        if issues:
            print("validation issues:")
            for i in issues:
                print(f"  - {i}")
        else:
            print("all values within valid ranges")

        return len(issues) == 0

    def create_clean_dataset(self, filepath, output_path=None):
        print("loading data...")
        df = self.load_data(filepath)
        print(f"  loaded {len(df)} samples")

        print("\nhandling missing values...")
        df = self.handle_missing_values(df, method='interpolate')

        print("detecting and removing outliers...")
        outlier_cols = ['inter_arrival_ms', 'jitter_ms', 'throughput_kbps', 'latency_ms']
        df = self.remove_outliers(df, outlier_cols, method='iqr', threshold=3.0)

        print("\nvalidating data ranges...")
        self.validate_data(df)

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nsaved clean dataset to {output_path}")

        print(f"final shape: {df.shape}")
        return df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.create_clean_dataset(
        '../data/5g_ran_dataset.csv',
        '../data/5g_ran_dataset_clean.csv'
    )
    print("\nsummary stats:")
    print(df_clean.describe())
