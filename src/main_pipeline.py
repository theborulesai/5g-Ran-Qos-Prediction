# main_pipeline.py
# runs everything end to end - just run this file and it does all 5 steps
# pcap parsing -> cleaning -> EDA -> feature engineering -> model training
# uses real 5G RAN pcap data, 3 ML algorithms and produces 12 graphs total

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pcap_parser import PcapParser
from data_preprocessing import DataPreprocessor
from eda_analysis import EDAAnalyzer
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer


def main():
    print("\n" + "=" * 60)
    print("  5G RAN QoS Prediction — Full Pipeline")
    print("  Data Source: Real PCAP capture (NGAP/F1AP SCTP traffic)")
    print("  Algorithms: Linear Regression, Decision Tree, Random Forest")
    print("  Graphs: 12 significant plots")
    print("=" * 60)

    # step 1: parse pcap file
    print("\n[1/5] Parsing PCAP file...")
    parser = PcapParser(pcap_path='../dataset .pcap')
    df_raw = parser.build_dataset(window_sec=5.0)
    parser.save_dataset('../data/5g_ran_dataset.csv')
    print(f"  extracted {len(df_raw)} samples from pcap")

    # step 2: preprocess
    print("\n[2/5] Preprocessing...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.create_clean_dataset(
        '../data/5g_ran_dataset.csv',
        '../data/5g_ran_dataset_clean.csv'
    )

    # step 3: EDA — generates graphs 1-6
    print("\n[3/5] Exploratory Data Analysis (Graphs 1-6)...")
    analyzer = EDAAnalyzer(output_dir='../results')
    analyzer.run_full_eda(df_clean)

    # step 4: feature engineering
    print("\n[4/5] Feature engineering...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_clean)
    df_features.to_csv('../data/5g_ran_dataset_features.csv', index=False)

    # step 5: model training + graphs 7-12
    print("\n[5/5] Training 3 Models + Generating Graphs 7-12...")
    feature_cols = engineer.get_feature_names()
    trainer = ModelTrainer()

    print("\n  --- Throughput models ---")
    train_data, val_data, test_data = trainer.prepare_data(df_features, 'throughput_kbps', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'throughput')

    print("\n  --- Latency models ---")
    train_data, val_data, test_data = trainer.prepare_data(df_features, 'latency_ms', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'latency')

    print("\n  saving models...")
    trainer.save_models('../models')

    # Generate graphs 7-12
    print("\n  generating model evaluation graphs...")
    print("  [Graph 7]  Model comparison bar chart...")
    trainer.plot_model_comparison('../results')

    print("  [Graph 8]  Throughput predictions (actual vs predicted)...")
    trainer.plot_predictions('throughput', '../results')

    print("  [Graph 9]  Latency predictions (actual vs predicted)...")
    trainer.plot_predictions('latency', '../results')

    print("  [Graph 10] Residual analysis...")
    trainer.plot_residual_analysis('../results')

    print("  [Graph 11] Feature importance (Random Forest)...")
    trainer.plot_feature_importance(feature_cols, '../results')

    print("  [Graph 12] Learning curves...")
    trainer.plot_learning_curves('../results')

    trainer.print_comparison_table()

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Data Source: dataset .pcap (real 5G RAN SCTP capture)")
    print("  Algorithms: Linear Regression, Decision Tree, Random Forest")
    print("  Graphs: 12 (see results/ folder)")
    print("  Data:   data/")
    print("  Models: models/")
    print("  Plots:  results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
