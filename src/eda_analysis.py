# eda_analysis.py
# exploratory data analysis - generating plots to understand the dataset
# I plotted distributions, correlations, scatter plots, time series, and load impact

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class EDAAnalyzer:

    def __init__(self, output_dir='../results'):
        self.output_dir = output_dir
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_distributions(self, df, columns, filename='distributions.png'):
        # histogram for each metric with mean/median lines
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, col in enumerate(columns):
            ax = axes[idx]
            df[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)

            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.legend()

        # turn off empty subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved: {filename}")

    def plot_correlation_matrix(self, df, columns, filename='correlation_matrix.png'):
        corr_matrix = df[columns].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - 5G RAN Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved: {filename}")

        return corr_matrix

    def plot_scatter_relationships(self, df, x_cols, y_col, filename='scatter.png'):
        # scatter each feature against the target with a trend line
        n_cols = len(x_cols)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, x_col in enumerate(x_cols):
            ax = axes[idx]
            ax.scatter(df[x_col], df[y_col], alpha=0.3, s=5)

            # linear trend line
            z = np.polyfit(df[x_col], df[y_col], 1)
            p = np.poly1d(z)
            ax.plot(df[x_col].sort_values(), p(df[x_col].sort_values()), "r--", linewidth=2, label='Trend')

            corr = df[x_col].corr(df[y_col])
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(y_col, fontsize=10)
            ax.set_title(f'{x_col} vs {y_col}\nCorr: {corr:.3f}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        for idx in range(len(x_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved: {filename}")

    def plot_time_series(self, df, columns, filename='time_series.png', sample_size=1000):
        # only plot the first 1000 samples otherwise it gets too crowded
        df_sample = df.head(sample_size).copy()

        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(15, 3 * n_cols))
        axes = [axes] if n_cols == 1 else axes

        for idx, col in enumerate(columns):
            ax = axes[idx]
            ax.plot(df_sample.index, df_sample[col], linewidth=1, alpha=0.8)
            ax.set_title(f'{col} over time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample Index', fontsize=10)
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved: {filename}")

    def plot_load_impact(self, df, filename='load_impact.png'):
        # bin network load into 3 categories and show throughput/latency distributions
        df_plot = df.copy()
        df_plot['load_category'] = pd.cut(df_plot['network_load'],
                                          bins=[0, 0.3, 0.6, 1.0],
                                          labels=['Low', 'Medium', 'High'])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        df_plot.boxplot(column='throughput_mbps', by='load_category', ax=axes[0])
        axes[0].set_title('Throughput vs Network Load', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Load Category', fontsize=12)
        axes[0].set_ylabel('Throughput (Mbps)', fontsize=12)
        axes[0].get_figure().suptitle('')

        df_plot.boxplot(column='latency_ms', by='load_category', ax=axes[1])
        axes[1].set_title('Latency vs Network Load', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Load Category', fontsize=12)
        axes[1].set_ylabel('Latency (ms)', fontsize=12)
        axes[1].get_figure().suptitle('')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved: {filename}")

    def generate_summary_statistics(self, df, filename='summary_statistics.csv'):
        summary = df.describe().T
        summary['missing'] = df.isnull().sum()
        summary['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
        summary.to_csv(f'{self.output_dir}/{filename}')
        print(f"saved: {filename}")
        print("\n--- Summary Statistics ---")
        print(summary)
        return summary

    def run_full_eda(self, df):
        print("\n--- Running EDA ---")

        self.generate_summary_statistics(df)

        key_metrics = ['rsrp_dbm', 'sinr_db', 'cqi', 'network_load', 'throughput_mbps', 'latency_ms']
        self.plot_distributions(df, key_metrics)

        corr_cols = ['rsrp_dbm', 'sinr_db', 'cqi', 'network_load', 'rb_allocation', 'throughput_mbps', 'latency_ms']
        corr_matrix = self.plot_correlation_matrix(df, corr_cols)

        ran_metrics = ['rsrp_dbm', 'sinr_db', 'cqi', 'network_load', 'rb_allocation']
        self.plot_scatter_relationships(df, ran_metrics, 'throughput_mbps', 'throughput_scatter.png')
        self.plot_scatter_relationships(df, ran_metrics, 'latency_ms', 'latency_scatter.png')

        self.plot_time_series(df, key_metrics)
        self.plot_load_impact(df)

        print("\nEDA done. plots saved to", self.output_dir)
        return corr_matrix


if __name__ == "__main__":
    df = pd.read_csv('../data/5g_ran_dataset_clean.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    analyzer = EDAAnalyzer(output_dir='../results')
    analyzer.run_full_eda(df)
