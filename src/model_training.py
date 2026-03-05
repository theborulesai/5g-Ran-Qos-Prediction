# model_training.py
# trains 3 models (Linear Regression, Decision Tree, Random Forest) for throughput and latency prediction
# evaluates each with RMSE, MAE, R2, and MAPE
# I used a 70/10/20 train/val/test split

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}

    def prepare_data(self, df, target_column, feature_columns, test_size=0.2, val_size=0.1):
        # split chronologically (shuffle=False) since this is time-series data
        X = df[feature_columns].values
        y = df[target_column].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False)

        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=42, shuffle=False)

        # standardize - important for linear regression
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        self.scalers[target_column] = scaler

        print(f"split for {target_column} -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

        return (X_train_sc, y_train), (X_val_sc, y_val), (X_test_sc, y_test)

    def train_linear_regression(self, X_train, y_train, model_name):
        print(f"  training Linear Regression for {model_name}...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models[f'{model_name}_lr'] = model
        return model

    def train_decision_tree(self, X_train, y_train, model_name, max_depth=10):
        print(f"  training Decision Tree for {model_name}...")
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models[f'{model_name}_dt'] = model
        return model

    def train_random_forest(self, X_train, y_train, model_name, n_estimators=100):
        print(f"  training Random Forest for {model_name} (this takes a bit)...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1  # use all CPU cores
        )
        model.fit(X_train, y_train)
        self.models[f'{model_name}_rf'] = model
        return model

    def evaluate_model(self, model, X, y, dataset_name):
        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-6))) * 100

        print(f"    [{dataset_name}] RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}  MAPE: {mape:.2f}%")

        return {
            'mse': mse, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'mape': mape,
            'predictions': y_pred, 'actual': y
        }

    def train_all_models(self, train_data, val_data, test_data, target_name):
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        results = {}

        lr_model = self.train_linear_regression(X_train, y_train, target_name)
        results[f'{target_name}_lr'] = {
            'train': self.evaluate_model(lr_model, X_train, y_train, 'train'),
            'val': self.evaluate_model(lr_model, X_val, y_val, 'val'),
            'test': self.evaluate_model(lr_model, X_test, y_test, 'test')
        }

        dt_model = self.train_decision_tree(X_train, y_train, target_name)
        results[f'{target_name}_dt'] = {
            'train': self.evaluate_model(dt_model, X_train, y_train, 'train'),
            'val': self.evaluate_model(dt_model, X_val, y_val, 'val'),
            'test': self.evaluate_model(dt_model, X_test, y_test, 'test')
        }

        rf_model = self.train_random_forest(X_train, y_train, target_name)
        results[f'{target_name}_rf'] = {
            'train': self.evaluate_model(rf_model, X_train, y_train, 'train'),
            'val': self.evaluate_model(rf_model, X_val, y_val, 'val'),
            'test': self.evaluate_model(rf_model, X_test, y_test, 'test')
        }

        self.results[target_name] = results
        return results

    def save_models(self, output_dir):
        for name, model in self.models.items():
            path = f"{output_dir}/{name}_model.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  saved {name}_model.pkl")

        for name, scaler in self.scalers.items():
            path = f"{output_dir}/{name}_scaler.pkl"
            with open(path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"  saved {name}_scaler.pkl")

    def plot_predictions(self, target_name, output_dir):
        # actual vs predicted scatter for each model on the test set
        results = self.results[target_name]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{target_name.replace("_", " ").title()} - Predictions vs Actual', fontsize=16)

        model_names = ['lr', 'dt', 'rf']
        titles = ['Linear Regression', 'Decision Tree', 'Random Forest']

        for idx, (suffix, title) in enumerate(zip(model_names, titles)):
            ax = axes[idx]
            res = results[f'{target_name}_{suffix}']['test']

            y_actual = res['actual']
            y_pred = res['predictions']

            ax.scatter(y_actual, y_pred, alpha=0.5, s=10)
            ax.plot([y_actual.min(), y_actual.max()],
                    [y_actual.min(), y_actual.max()],
                    'r--', lw=2, label='Perfect prediction')

            ax.set_xlabel('Actual', fontsize=12)
            ax.set_ylabel('Predicted', fontsize=12)
            ax.set_title(f'{title}\nR² = {res["r2"]:.4f}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{target_name}_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  saved {target_name}_predictions.png")

    def create_comparison_table(self, output_dir):
        rows = []
        for target_name, results in self.results.items():
            for model_key, metrics in results.items():
                m = metrics['test']
                rows.append({
                    'Target': target_name,
                    'Model': model_key.split('_')[-1].upper(),
                    'RMSE': round(m['rmse'], 4),
                    'MAE': round(m['mae'], 4),
                    'R²': round(m['r2'], 4),
                    'MAPE (%)': round(m['mape'], 2)
                })

        df_cmp = pd.DataFrame(rows)
        df_cmp.to_csv(f'{output_dir}/model_comparison.csv', index=False)

        print("\n--- Model Comparison (test set) ---")
        print(df_cmp.to_string(index=False))

        return df_cmp


if __name__ == "__main__":
    df = pd.read_csv('../data/5g_ran_dataset_features.csv')

    exclude_cols = ['timestamp', 'throughput_mbps', 'latency_ms']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"using {len(feature_cols)} features")

    trainer = ModelTrainer()

    print("\n--- Throughput prediction ---")
    train_data, val_data, test_data = trainer.prepare_data(df, 'throughput_mbps', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'throughput')

    print("\n--- Latency prediction ---")
    train_data, val_data, test_data = trainer.prepare_data(df, 'latency_ms', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'latency')

    print("\n--- Saving models ---")
    trainer.save_models('../models')

    print("\n--- Saving plots ---")
    trainer.plot_predictions('throughput', '../results')
    trainer.plot_predictions('latency', '../results')

    trainer.create_comparison_table('../results')
