# data_generator.py
# generates synthetic 5G RAN dataset for the mini project
# I used Shannon capacity formula and 3GPP parameter ranges to make it realistic

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class RAN5GDataGenerator:
    # generates time-series measurement data for a single UE in a 5G cell
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate_dataset(self, n_samples=10000, sampling_interval_ms=100):
        # generates n_samples rows of RAN + QoS measurements
        # sampling_interval_ms: time between each sample (100ms = 10 samples/sec)
        
        # timestamps  
        start_time = datetime.now()
        timestamps = [start_time + timedelta(milliseconds=i * sampling_interval_ms)
                      for i in range(n_samples)]
        
        # network load - varies like a sinusoid to simulate busy/quiet hours
        # roughly peaks around 6 PM
        time_of_day = np.array([t.hour + t.minute / 60 for t in timestamps])
        base_load = 0.3 + 0.4 * np.sin(2 * np.pi * (time_of_day - 6) / 24)
        network_load = np.clip(base_load + np.random.normal(0, 0.1, n_samples), 0.1, 0.95)
        
        # RSRP in dBm (typical 5G NR range: -140 to -44 dBm)
        # adding temporal correlation to simulate slow mobility
        base_rsrp = np.random.uniform(-100, -70, n_samples)
        for i in range(1, n_samples):
            base_rsrp[i] = 0.95 * base_rsrp[i - 1] + 0.05 * base_rsrp[i]
        rsrp = base_rsrp + np.random.normal(0, 3, n_samples)
        rsrp = np.clip(rsrp, -140, -44)
        
        # SINR in dB (range: -10 to 30 dB)
        # correlated with RSRP, degrades under load
        base_sinr = (rsrp + 100) / 3 - 5 * network_load
        sinr = base_sinr + np.random.normal(0, 2, n_samples)
        sinr = np.clip(sinr, -10, 30)
        
        # CQI (0-15) - closely follows SINR
        cqi_continuous = (sinr + 10) / 40 * 15
        cqi = np.clip(np.round(cqi_continuous + np.random.normal(0, 0.5, n_samples)), 0, 15).astype(int)
        
        # MCS (0-28) - determined by CQI
        mcs = np.clip(np.round(cqi * 1.8 + np.random.normal(0, 1, n_samples)), 0, 28).astype(int)
        
        # RB allocation - fewer RBs available when load is high
        max_rbs = 100
        rb_allocation = max_rbs * (1 - 0.5 * network_load) * np.random.uniform(0.3, 1.0, n_samples)
        rb_allocation = np.clip(rb_allocation, 5, max_rbs).astype(int)
        
        # Throughput (Mbps) using Shannon capacity approximation
        # C = B * log2(1 + SNR), each RB is 180 kHz wide in NR
        spectral_efficiency = 0.6 * np.log2(1 + 10 ** (sinr / 10))
        bandwidth_per_rb = 180e3  # Hz
        theoretical_tp = spectral_efficiency * rb_allocation * bandwidth_per_rb / 1e6
        
        # apply practical efficiency reduction
        eff_factor = 0.7 + 0.2 * (cqi / 15)
        load_penalty = 1 - 0.3 * network_load
        throughput = theoretical_tp * eff_factor * load_penalty
        throughput = throughput + np.random.normal(0, throughput * 0.05, n_samples)
        throughput = np.clip(throughput, 0.1, 1000)
        
        # Latency (ms) - increases with load and poor SINR
        base_latency = 10 + 50 * network_load
        sinr_penalty = np.maximum(0, (15 - sinr) * 2)
        retx_delay = np.random.exponential(5 * (1 - cqi / 15), n_samples)
        latency = base_latency + sinr_penalty + retx_delay
        latency = latency + np.random.normal(0, 3, n_samples)
        latency = np.clip(latency, 1, 200)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'rsrp_dbm': rsrp,
            'sinr_db': sinr,
            'cqi': cqi,
            'mcs': mcs,
            'rb_allocation': rb_allocation,
            'network_load': network_load,
            'throughput_mbps': throughput,
            'latency_ms': latency
        })
        
        return df
    
    def add_missing_values(self, df, missing_rate=0.02):
        # randomly drop ~2% of values in a few columns to simulate sensor glitches
        df_copy = df.copy()
        for col in ['rsrp_dbm', 'sinr_db', 'cqi']:
            mask = np.random.random(len(df)) < missing_rate
            df_copy.loc[mask, col] = np.nan
        return df_copy
    
    def add_outliers(self, df, outlier_rate=0.01):
        # inject some extreme values to simulate measurement errors
        df_copy = df.copy()
        mask = np.random.random(len(df)) < outlier_rate
        df_copy.loc[mask, 'rsrp_dbm'] = np.random.uniform(-140, -120, mask.sum())
        mask = np.random.random(len(df)) < outlier_rate
        df_copy.loc[mask, 'sinr_db'] = np.random.uniform(-10, 0, mask.sum())
        return df_copy


if __name__ == "__main__":
    generator = RAN5GDataGenerator(seed=42)
    df = generator.generate_dataset(n_samples=10000)
    df = generator.add_missing_values(df, missing_rate=0.02)
    df = generator.add_outliers(df, outlier_rate=0.01)
    df.to_csv('../data/5g_ran_dataset.csv', index=False)
    print(f"dataset saved: {df.shape}")
    print(df.head())
