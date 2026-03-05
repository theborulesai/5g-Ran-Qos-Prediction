# data_preprocessing.py
# cleans the raw dataset - handles missing values, outliers, and validates ranges
# referenced 3GPP TS 38.214 for valid parameter ranges

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
        # using linear interpolation since this is time-series data
        # forward/backward fill are options too but interpolation seems cleaner
        df_copy = df.copy()
        print(f"missing values before:\n{df_copy.isnull().sum()}\n")

        if method == 'interpolate':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].interpolate(
                method='linear', limit_direction='both')
        elif method == 'forward_fill':
            df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
        elif method == 'drop':
            df_copy = df_copy.dropna()

        print(f"missing values after:\n{df_copy.isnull().sum()}\n")
        return df_copy

    def detect_outliers(self, df, columns, method='iqr', threshold=1.5):
        # IQR method - mark anything outside Q1-1.5*IQR or Q3+1.5*IQR as outlier
        # can also use z-score but IQR is more robust
        outlier_indices = set()

        for col in columns:
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
        # check that all values are within expected 5G NR ranges
        valid_ranges = {
            'rsrp_dbm': (-140, -44),
            'sinr_db': (-10, 30),
            'cqi': (0, 15),
            'mcs': (0, 28),
            'network_load': (0, 1),
            'throughput_mbps': (0, 1000),
            'latency_ms': (0, 500)
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
        outlier_cols = ['rsrp_dbm', 'sinr_db', 'throughput_mbps', 'latency_ms']
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
