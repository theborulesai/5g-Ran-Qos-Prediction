# main_pipeline.py
# runs the complete pipeline end to end:
# data generation -> preprocessing -> EDA -> feature engineering -> model training

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import RAN5GDataGenerator
from data_preprocessing import DataPreprocessor
from eda_analysis import EDAAnalyzer
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer


def main():
    print("\n5G RAN QoS Prediction - Full Pipeline")
    print("=" * 50)

    # step 1: generate dataset
    print("\n[1/5] Generating dataset...")
    generator = RAN5GDataGenerator(seed=42)
    df_raw = generator.generate_dataset(n_samples=10000)
    df_raw = generator.add_missing_values(df_raw, missing_rate=0.02)
    df_raw = generator.add_outliers(df_raw, outlier_rate=0.01)
    df_raw.to_csv('../data/5g_ran_dataset.csv', index=False)
    print(f"  generated {len(df_raw)} samples")

    # step 2: preprocess
    print("\n[2/5] Preprocessing...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.create_clean_dataset(
        '../data/5g_ran_dataset.csv',
        '../data/5g_ran_dataset_clean.csv'
    )

    # step 3: EDA
    print("\n[3/5] Exploratory data analysis...")
    analyzer = EDAAnalyzer(output_dir='../results')
    analyzer.run_full_eda(df_clean)

    # step 4: feature engineering
    print("\n[4/5] Feature engineering...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    df_features.to_csv('../data/5g_ran_dataset_features.csv', index=False)

    # step 5: model training
    print("\n[5/5] Training models...")
    feature_cols = engineer.get_feature_names()
    trainer = ModelTrainer()

    print("  throughput models:")
    train_data, val_data, test_data = trainer.prepare_data(df_features, 'throughput_mbps', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'throughput')

    print("  latency models:")
    train_data, val_data, test_data = trainer.prepare_data(df_features, 'latency_ms', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'latency')

    print("  saving models...")
    trainer.save_models('../models')

    print("  generating prediction plots...")
    trainer.plot_predictions('throughput', '../results')
    trainer.plot_predictions('latency', '../results')
    trainer.create_comparison_table('../results')

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("  data   -> data/")
    print("  models -> models/")
    print("  plots  -> results/")
    print("=" * 50)


if __name__ == "__main__":
    main()
