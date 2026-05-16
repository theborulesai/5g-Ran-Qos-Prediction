# model_training.py
# trains 3 ML algorithms on both throughput and latency prediction
# algorithms: Linear Regression, Decision Tree, Random Forest
# we evaluate with RMSE, MAE, R2, MAPE and make 6 graphs (graphs 7-12)
# adapted for real pcap-derived 5G RAN network metrics

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        # store train data for learning curves
        self.train_data_cache = {}

    def prepare_data(self, df, target_column, feature_columns, test_size=0.2, val_size=0.1):
        # chronological split - no shuffling because this is time-series
        X = df[feature_columns].values
        y = df[target_column].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False)

        val_size_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=42, shuffle=False)

        # standardize features
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        self.scalers[target_column] = scaler
        self.train_data_cache[target_column] = (X_train_sc, y_train)

        print(f"  split for {target_column} -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

        return (X_train_sc, y_train), (X_val_sc, y_val), (X_test_sc, y_test)

    def train_linear_regression(self, X_train, y_train, model_name):
        print(f"    training Linear Regression (Ridge) for {model_name}...")
        # using Ridge (L2 regularization) instead of plain LinearRegression
        # because our 106 features have multicollinearity which makes plain LR explode
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        self.models[f'{model_name}_lr'] = model
        return model

    def train_decision_tree(self, X_train, y_train, model_name, max_depth=10):
        print(f"    training Decision Tree for {model_name}...")
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
        print(f"    training Random Forest for {model_name}...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
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

        print(f"      [{dataset_name}] RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}  MAPE: {mape:.2f}%")

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

        # Algorithm 1: Linear Regression
        lr_model = self.train_linear_regression(X_train, y_train, target_name)
        results[f'{target_name}_lr'] = {
            'train': self.evaluate_model(lr_model, X_train, y_train, 'train'),
            'val': self.evaluate_model(lr_model, X_val, y_val, 'val'),
            'test': self.evaluate_model(lr_model, X_test, y_test, 'test')
        }

        # Algorithm 2: Decision Tree
        dt_model = self.train_decision_tree(X_train, y_train, target_name)
        results[f'{target_name}_dt'] = {
            'train': self.evaluate_model(dt_model, X_train, y_train, 'train'),
            'val': self.evaluate_model(dt_model, X_val, y_val, 'val'),
            'test': self.evaluate_model(dt_model, X_test, y_test, 'test')
        }

        # Algorithm 3: Random Forest
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
            print(f"    saved {name}_model.pkl")

        for name, scaler in self.scalers.items():
            path = f"{output_dir}/{name}_scaler.pkl"
            with open(path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"    saved {name}_scaler.pkl")

    def _setup_style(self):
        """Set clean white-background academic style."""
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'font.family': 'sans-serif',
            'font.size': 11,
            'figure.dpi': 150,
        })

    # -------------------------------------------------------
    #  GRAPH 7: Model Comparison Bar Chart
    # -------------------------------------------------------
    def plot_model_comparison(self, output_dir):
        """Bar chart comparing R2, RMSE, MAE for all 3 models across both targets."""
        self._setup_style()
        model_suffixes = ['lr', 'dt', 'rf']
        model_labels = ['Linear Reg.\n(Ridge)', 'Decision\nTree', 'Random\nForest']
        colors = ['#4285F4', '#34A853', '#EA4335']
        metrics_list = ['r2', 'rmse', 'mae']
        metrics_labels = ['R² Score', 'RMSE', 'MAE']

        target_names = list(self.results.keys())

        fig, axes = plt.subplots(len(target_names), 3, figsize=(18, 5 * len(target_names)))
        fig.suptitle('Model Performance Comparison — 3 Algorithms',
                     fontsize=16, fontweight='bold', y=0.98)

        for row, target_name in enumerate(target_names):
            results = self.results[target_name]
            target_label = 'Throughput (Kbps)' if 'throughput' in target_name else 'Latency (ms)'

            for col, (metric, ylabel) in enumerate(zip(metrics_list, metrics_labels)):
                ax = axes[row][col]
                vals = [results[f'{target_name}_{sfx}']['test'][metric] for sfx in model_suffixes]

                x = np.arange(len(model_labels))
                bars = ax.bar(x, vals, 0.55, color=colors, edgecolor='white', linewidth=1)

                ax.set_title(f'{target_label} — {ylabel}', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(model_labels, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print("    saved model_comparison.png")

    # -------------------------------------------------------
    #  GRAPH 8/9: Predictions (Actual vs Predicted)
    # -------------------------------------------------------
    def plot_predictions(self, target_name, output_dir):
        """Scatter plot: actual vs predicted for all 3 models on the test set."""
        self._setup_style()
        results = self.results[target_name]
        unit = 'Kbps' if 'throughput' in target_name else 'ms'
        target_label = f'Throughput ({unit})' if 'throughput' in target_name else f'Latency ({unit})'

        model_suffixes = ['lr', 'dt', 'rf']
        titles = ['Linear Regression (Ridge)', 'Decision Tree', 'Random Forest']
        colors = ['#4285F4', '#34A853', '#EA4335']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{target_label} — Actual vs Predicted (Test Set)',
                     fontsize=14, fontweight='bold')

        for idx, (suffix, title, clr) in enumerate(zip(model_suffixes, titles, colors)):
            ax = axes[idx]
            res = results[f'{target_name}_{suffix}']['test']
            y_actual = res['actual']
            y_pred = res['predictions']

            ax.scatter(y_actual, y_pred, alpha=0.35, s=8, color=clr,
                       edgecolor='none', label='Predictions')
            ax.plot([y_actual.min(), y_actual.max()],
                    [y_actual.min(), y_actual.max()],
                    color='#E53935', linestyle='--', lw=2, label='Perfect fit')

            ax.set_xlabel(f'Actual ({unit})', fontsize=10)
            ax.set_ylabel(f'Predicted ({unit})', fontsize=10)
            ax.set_title(f'{title}\nR² = {res["r2"]:.4f} | RMSE = {res["rmse"]:.4f}',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, framealpha=0.9, edgecolor='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{target_name}_predictions.png', dpi=150,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"    saved {target_name}_predictions.png")

    # -------------------------------------------------------
    #  GRAPH 10: Residual Analysis
    # -------------------------------------------------------
    def plot_residual_analysis(self, output_dir):
        """Residual histogram for all 3 models x 2 targets."""
        self._setup_style()
        model_suffixes = ['lr', 'dt', 'rf']
        model_labels = ['Linear Regression (Ridge)', 'Decision Tree', 'Random Forest']
        colors = ['#4285F4', '#34A853', '#EA4335']
        target_names = list(self.results.keys())

        fig, axes = plt.subplots(len(target_names), 3, figsize=(18, 5 * len(target_names)))
        fig.suptitle('Residual Analysis — All Models',
                     fontsize=16, fontweight='bold', y=0.98)

        for row, target_name in enumerate(target_names):
            results = self.results[target_name]
            target_label = 'Throughput' if 'throughput' in target_name else 'Latency'

            for col, (sfx, label, clr) in enumerate(zip(model_suffixes, model_labels, colors)):
                ax = axes[row][col]
                res = results[f'{target_name}_{sfx}']['test']
                residuals = res['actual'] - res['predictions']

                ax.hist(residuals, bins=40, color=clr, edgecolor='white',
                        linewidth=0.5, alpha=0.75, density=True)
                ax.axvline(x=0, color='#E53935', linestyle='--', linewidth=1.5, label='Zero')
                mean_r = np.mean(residuals)
                std_r = np.std(residuals)
                ax.axvline(x=mean_r, color='#FF9800', linestyle='-', linewidth=1.5,
                           label=f'Mean: {mean_r:.4f}')

                ax.set_xlabel('Residual (Actual - Predicted)', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.set_title(f'{target_label} — {label}\nσ = {std_r:.4f}',
                             fontsize=11, fontweight='bold')
                ax.legend(fontsize=8, framealpha=0.9, edgecolor='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/residual_analysis.png', dpi=150,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print("    saved residual_analysis.png")

    # -------------------------------------------------------
    #  GRAPH 11: Feature Importance from Random Forest
    # -------------------------------------------------------
    def plot_feature_importance(self, feature_names, output_dir, top_n=15):
        """Top 15 features from Random Forest for both targets."""
        self._setup_style()
        target_names = list(self.results.keys())

        fig, axes = plt.subplots(1, len(target_names), figsize=(16, 7))
        fig.suptitle('Top 15 Feature Importances — Random Forest',
                     fontsize=15, fontweight='bold')

        if len(target_names) == 1:
            axes = [axes]

        grad_colors = ['#4285F4', '#34A853']

        for idx, target_name in enumerate(target_names):
            ax = axes[idx]
            target_label = 'Throughput (Kbps)' if 'throughput' in target_name else 'Latency (ms)'

            rf_model = self.models[f'{target_name}_rf']
            importances = rf_model.feature_importances_
            sorted_idx = np.argsort(importances)[-top_n:]

            base_color = grad_colors[idx % len(grad_colors)]
            bar_colors = [base_color] * top_n

            ax.barh(range(top_n), importances[sorted_idx], color=bar_colors,
                    edgecolor='white', linewidth=0.5, alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
            ax.set_xlabel('Importance Score', fontsize=11)
            ax.set_title(target_label, fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for j, (val, pos) in enumerate(zip(importances[sorted_idx], range(top_n))):
                ax.text(val + 0.001, pos, f'{val:.3f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=150,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print("    saved feature_importance.png")

    # -------------------------------------------------------
    #  GRAPH 12: Learning Curves
    # -------------------------------------------------------
    def plot_learning_curves(self, output_dir):
        """Learning curves for all 3 models across both targets."""
        self._setup_style()
        model_configs = [
            ('Linear Reg. (Ridge)', Ridge(alpha=1.0), '#4285F4'),
            ('Decision Tree', DecisionTreeRegressor(max_depth=10, min_samples_split=20,
                                                     min_samples_leaf=10, random_state=42), '#34A853'),
            ('Random Forest', RandomForestRegressor(n_estimators=50, max_depth=15,
                                                     min_samples_leaf=10, random_state=42, n_jobs=-1), '#EA4335'),
        ]

        target_names = list(self.train_data_cache.keys())

        fig, axes = plt.subplots(len(target_names), 3, figsize=(18, 5 * len(target_names)))
        fig.suptitle('Learning Curves — 3 Algorithms',
                     fontsize=16, fontweight='bold', y=0.98)

        for row, target_col in enumerate(target_names):
            target_label = 'Throughput' if 'throughput' in target_col else 'Latency'
            X_train, y_train = self.train_data_cache[target_col]

            max_samples = 2000
            if len(X_train) > max_samples:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_train), max_samples, replace=False)
                X_lc, y_lc = X_train[idx], y_train[idx]
            else:
                X_lc, y_lc = X_train, y_train

            train_sizes_frac = np.linspace(0.1, 1.0, 8)

            for col, (label, estimator, clr) in enumerate(model_configs):
                ax = axes[row][col]
                try:
                    train_sizes, train_scores, test_scores = learning_curve(
                        estimator, X_lc, y_lc,
                        train_sizes=train_sizes_frac,
                        cv=3, scoring='r2', n_jobs=-1, random_state=42
                    )

                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    test_std = np.std(test_scores, axis=1)

                    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                                    alpha=0.12, color=clr)
                    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                                    alpha=0.12, color='#FF9800')
                    ax.plot(train_sizes, train_mean, 'o-', color=clr,
                            linewidth=2, markersize=4, label='Train R²')
                    ax.plot(train_sizes, test_mean, 's-', color='#FF9800',
                            linewidth=2, markersize=4, label='CV R²')
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error computing', transform=ax.transAxes,
                            ha='center', va='center', fontsize=10)

                ax.set_xlabel('Training Samples', fontsize=10)
                ax.set_ylabel('R² Score', fontsize=10)
                ax.set_title(f'{target_label} — {label}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/learning_curves.png', dpi=150,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print("    saved learning_curves.png")

    def print_comparison_table(self):
        rows = []
        for target_name, results in self.results.items():
            for model_key, metrics in results.items():
                m = metrics['test']
                suffix = model_key.split('_')[-1].upper()
                rows.append({
                    'Target': target_name,
                    'Model': suffix,
                    'RMSE': round(m['rmse'], 4),
                    'MAE': round(m['mae'], 4),
                    'R²': round(m['r2'], 4),
                    'MAPE (%)': round(m['mape'], 2)
                })

        df_cmp = pd.DataFrame(rows)
        print("\n  --- Model Comparison (test set) ---")
        print(df_cmp.to_string(index=False))
        return df_cmp


if __name__ == "__main__":
    df = pd.read_csv('../data/5g_ran_dataset_features.csv')

    exclude_cols = ['timestamp', 'throughput_kbps', 'latency_ms']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"using {len(feature_cols)} features")

    trainer = ModelTrainer()

    print("\n--- Throughput prediction ---")
    train_data, val_data, test_data = trainer.prepare_data(df, 'throughput_kbps', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'throughput')

    print("\n--- Latency prediction ---")
    train_data, val_data, test_data = trainer.prepare_data(df, 'latency_ms', feature_cols)
    trainer.train_all_models(train_data, val_data, test_data, 'latency')

    trainer.save_models('../models')
    trainer.plot_model_comparison('../results')
    trainer.plot_predictions('throughput', '../results')
    trainer.plot_predictions('latency', '../results')
    trainer.plot_residual_analysis('../results')
    trainer.plot_feature_importance(feature_cols, '../results')
    trainer.plot_learning_curves('../results')
    trainer.print_comparison_table()
