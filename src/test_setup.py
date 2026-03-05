# test_setup.py
# quick sanity check - run this before the main pipeline to make sure everything is installed

def test_imports():
    print("checking required packages...")

    try:
        import numpy as np
        print("  numpy - ok")
    except ImportError as e:
        print(f"  numpy - MISSING: {e}")
        return False

    try:
        import pandas as pd
        print("  pandas - ok")
    except ImportError as e:
        print(f"  pandas - MISSING: {e}")
        return False

    try:
        from sklearn.linear_model import LinearRegression
        print("  scikit-learn - ok")
    except ImportError as e:
        print(f"  scikit-learn - MISSING: {e}")
        return False

    try:
        import matplotlib.pyplot as plt
        print("  matplotlib - ok")
    except ImportError as e:
        print(f"  matplotlib - MISSING: {e}")
        return False

    try:
        import seaborn as sns
        print("  seaborn - ok")
    except ImportError as e:
        print(f"  seaborn - MISSING: {e}")
        return False

    try:
        import scipy
        print("  scipy - ok")
    except ImportError as e:
        print(f"  scipy - MISSING: {e}")
        return False

    return True


def test_modules():
    print("\nchecking project modules...")

    try:
        from data_generator import RAN5GDataGenerator
        print("  data_generator - ok")
    except ImportError as e:
        print(f"  data_generator - FAILED: {e}")
        return False

    try:
        from data_preprocessing import DataPreprocessor
        print("  data_preprocessing - ok")
    except ImportError as e:
        print(f"  data_preprocessing - FAILED: {e}")
        return False

    try:
        from eda_analysis import EDAAnalyzer
        print("  eda_analysis - ok")
    except ImportError as e:
        print(f"  eda_analysis - FAILED: {e}")
        return False

    try:
        from feature_engineering import FeatureEngineer
        print("  feature_engineering - ok")
    except ImportError as e:
        print(f"  feature_engineering - FAILED: {e}")
        return False

    try:
        from model_training import ModelTrainer
        print("  model_training - ok")
    except ImportError as e:
        print(f"  model_training - FAILED: {e}")
        return False

    return True


def quick_test():
    print("\nrunning a small generation test...")

    try:
        from data_generator import RAN5GDataGenerator

        gen = RAN5GDataGenerator(seed=42)
        df = gen.generate_dataset(n_samples=100)

        assert df['rsrp_dbm'].min() >= -140 and df['rsrp_dbm'].max() <= -44, "RSRP out of range"
        assert df['sinr_db'].min() >= -10 and df['sinr_db'].max() <= 30, "SINR out of range"
        assert df['cqi'].min() >= 0 and df['cqi'].max() <= 15, "CQI out of range"

        print(f"  generated {len(df)} samples - all values in range")
        return True

    except Exception as e:
        print(f"  failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("5G RAN QoS Prediction - Setup Check")
    print("=" * 50)

    if not test_imports():
        print("\nsome packages are missing. install with:")
        print("  pip install numpy pandas scikit-learn matplotlib seaborn scipy")
        exit(1)

    if not test_modules():
        print("\nmodule import failed - check the src/ directory")
        exit(1)

    if not quick_test():
        print("\nfunctionality test failed")
        exit(1)

    print("\nall checks passed. run the pipeline with:")
    print("  python main_pipeline.py")
