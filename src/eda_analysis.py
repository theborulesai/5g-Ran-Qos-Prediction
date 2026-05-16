# eda_analysis.py
# exploratory data analysis - makes 6 graphs (graphs 1-6)
# adapted for real pcap-derived 5G RAN network metrics
# clean white-background student-style graphs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class EDAAnalyzer:

    def __init__(self, output_dir='../results'):
        self.output_dir = output_dir

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
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.dpi': 150,
        })

    # ---------------------------------------------------
    #  GRAPH 1: Distribution Histograms
    # ---------------------------------------------------
    def plot_distributions(self, df, columns, filename='distributions.png'):
        """Clean histograms with KDE overlays for each key metric."""
        self._setup_style()
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
        fig.suptitle('Feature Distributions — 5G RAN PCAP Dataset',
                     fontsize=16, fontweight='bold', y=0.98)
        axes = axes.flatten()

        colors = ['#4285F4', '#34A853', '#EA4335', '#9C27B0', '#FF9800', '#00BCD4']
        labels = {
            'packet_size': 'Packet Size (bytes)',
            'inter_arrival_ms': 'Inter-Arrival Time (ms)',
            'jitter_ms': 'Jitter (ms)',
            'network_load': 'Network Load (0–1)',
            'throughput_kbps': 'Throughput (Kbps)',
            'latency_ms': 'Latency / RTT (ms)',
        }

        for idx, col in enumerate(columns):
            ax = axes[idx]
            clr = colors[idx % len(colors)]
            data = df[col].dropna()

            # use log-scaled bins for heavily-skewed data
            ax.hist(data, bins=40, edgecolor='white', linewidth=0.5,
                    alpha=0.75, color=clr, zorder=2)

            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='#E53935', linestyle='--', linewidth=1.5,
                       label=f'Mean: {mean_val:.2f}', zorder=3)
            ax.axvline(median_val, color='#212121', linestyle=':', linewidth=1.5,
                       label=f'Median: {median_val:.2f}', zorder=3)

            nice_name = labels.get(col, col)
            ax.set_title(nice_name, fontsize=12, fontweight='bold')
            ax.set_xlabel(nice_name, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"    saved: {filename}")

    # ---------------------------------------------------
    #  GRAPH 2: Correlation Matrix Heatmap
    # ---------------------------------------------------
    def plot_correlation_matrix(self, df, columns, filename='correlation_matrix.png'):
        """Clean correlation heatmap with readable labels."""
        self._setup_style()
        corr_matrix = df[columns].corr()

        nice_names = {
            'packet_size': 'Packet Size',
            'inter_arrival_ms': 'Inter-Arrival',
            'jitter_ms': 'Jitter',
            'network_load': 'Network Load',
            'packet_rate_pps': 'Packet Rate',
            'throughput_kbps': 'Throughput',
            'latency_ms': 'Latency (RTT)',
        }
        corr_renamed = corr_matrix.copy()
        corr_renamed.index = [nice_names.get(c, c) for c in corr_renamed.index]
        corr_renamed.columns = [nice_names.get(c, c) for c in corr_renamed.columns]

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_renamed, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr_renamed, mask=mask, annot=True, fmt='.2f',
                    cmap=cmap, center=0, square=True, linewidths=1.5,
                    linecolor='white', cbar_kws={"shrink": 0.75, "label": "Pearson r"},
                    ax=ax, vmin=-1, vmax=1)
        ax.set_title('Correlation Matrix — 5G RAN Network Metrics',
                      fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"    saved: {filename}")
        return corr_matrix

    # ---------------------------------------------------
    #  GRAPH 3 & 4: Scatter Plots (network metrics vs QoS)
    # ---------------------------------------------------
    def plot_scatter_relationships(self, df, x_cols, y_col, filename='scatter.png'):
        """Scatter each metric against the target with regression line."""
        self._setup_style()
        n_cols = len(x_cols)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))

        target_name = 'Throughput (Kbps)' if 'throughput' in y_col else 'Latency (ms)'
        fig.suptitle(f'Network Metrics vs {target_name}',
                     fontsize=15, fontweight='bold', y=0.98)
        axes = axes.flatten()

        colors = ['#4285F4', '#34A853', '#EA4335', '#9C27B0', '#FF9800']
        nice_names = {
            'packet_size': 'Packet Size (bytes)',
            'inter_arrival_ms': 'Inter-Arrival (ms)',
            'jitter_ms': 'Jitter (ms)',
            'network_load': 'Network Load',
            'packet_rate_pps': 'Packet Rate (pps)',
        }

        for idx, x_col in enumerate(x_cols):
            ax = axes[idx]
            clr = colors[idx % len(colors)]

            # subsample for cleaner scatter
            sample = df.sample(min(2000, len(df)), random_state=42)

            ax.scatter(sample[x_col], sample[y_col], alpha=0.35, s=8,
                       color=clr, edgecolor='none', zorder=2)

            # linear trend
            z = np.polyfit(df[x_col], df[y_col], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            ax.plot(x_range, p(x_range), color='#E53935', linewidth=2,
                    linestyle='--', label='Linear fit', zorder=3)

            corr = df[x_col].corr(df[y_col])
            nice_x = nice_names.get(x_col, x_col)
            ax.set_xlabel(nice_x, fontsize=10)
            ax.set_ylabel(target_name, fontsize=10)
            ax.set_title(f'{nice_x}\nr = {corr:.3f}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, edgecolor='gray')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for idx in range(len(x_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"    saved: {filename}")

    # ---------------------------------------------------
    #  GRAPH 5: Time Series
    # ---------------------------------------------------
    def plot_time_series(self, df, columns, filename='time_series.png', sample_size=1000):
        """Clean time-series panels of all key metrics."""
        self._setup_style()
        df_sample = df.head(sample_size).copy()

        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(14, 2.8 * n_cols))
        fig.suptitle('Time Series — 5G RAN PCAP Capture (First 1,000 Packets)',
                     fontsize=14, fontweight='bold', y=1.0)
        axes = [axes] if n_cols == 1 else axes

        colors = ['#4285F4', '#34A853', '#EA4335', '#9C27B0', '#FF9800', '#00BCD4']
        nice_names = {
            'packet_size': 'Packet Size (bytes)',
            'inter_arrival_ms': 'Inter-Arrival (ms)',
            'jitter_ms': 'Jitter (ms)',
            'network_load': 'Network Load',
            'throughput_kbps': 'Throughput (Kbps)',
            'latency_ms': 'Latency (ms)',
        }

        for idx, col in enumerate(columns):
            ax = axes[idx]
            clr = colors[idx % len(colors)]
            ax.plot(df_sample.index, df_sample[col], linewidth=0.8,
                    alpha=0.85, color=clr, zorder=2)

            nice = nice_names.get(col, col)
            ax.set_ylabel(nice, fontsize=9)
            ax.set_title(nice, fontsize=10, fontweight='bold', loc='left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if idx < len(columns) - 1:
                ax.set_xticklabels([])

        axes[-1].set_xlabel('Packet Index', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"    saved: {filename}")

    # ---------------------------------------------------
    #  GRAPH 6: Network Load Impact
    # ---------------------------------------------------
    def plot_load_impact(self, df, filename='load_impact.png'):
        """Box plots showing QoS under Low/Medium/High load."""
        self._setup_style()
        df_plot = df.copy()
        df_plot['Load Category'] = pd.cut(df_plot['network_load'],
                                          bins=[0, 0.3, 0.6, 1.0],
                                          labels=['Low (<30%)', 'Medium (30-60%)', 'High (>60%)'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Impact of Network Load on QoS Metrics',
                     fontsize=14, fontweight='bold')

        # throughput boxplot
        bp1 = df_plot.boxplot(column='throughput_kbps', by='Load Category',
                              ax=axes[0], patch_artist=True,
                              return_type='dict')
        colors_box = ['#4285F4', '#FF9800', '#EA4335']
        for patch, color in zip(bp1['throughput_kbps']['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].set_title('Throughput vs Network Load', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Load Category', fontsize=11)
        axes[0].set_ylabel('Throughput (Kbps)', fontsize=11)
        axes[0].get_figure().suptitle('')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # latency boxplot
        bp2 = df_plot.boxplot(column='latency_ms', by='Load Category',
                              ax=axes[1], patch_artist=True,
                              return_type='dict')
        for patch, color in zip(bp2['latency_ms']['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1].set_title('Latency vs Network Load', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Load Category', fontsize=11)
        axes[1].set_ylabel('Latency (ms)', fontsize=11)
        axes[1].get_figure().suptitle('')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"    saved: {filename}")

    def print_summary_statistics(self, df):
        summary = df.describe().T
        summary['missing'] = df.isnull().sum()
        summary['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
        print("    summary statistics computed")
        return summary

    def run_full_eda(self, df):
        """Generate all 6 EDA graphs."""
        print("  generating EDA graphs...")

        self.print_summary_statistics(df)

        key_metrics = ['packet_size', 'inter_arrival_ms', 'jitter_ms',
                       'network_load', 'throughput_kbps', 'latency_ms']

        print("  [Graph 1] Distributions...")
        self.plot_distributions(df, key_metrics)

        corr_cols = ['packet_size', 'inter_arrival_ms', 'jitter_ms', 'network_load',
                     'packet_rate_pps', 'throughput_kbps', 'latency_ms']
        print("  [Graph 2] Correlation matrix...")
        corr_matrix = self.plot_correlation_matrix(df, corr_cols)

        net_metrics = ['packet_size', 'inter_arrival_ms', 'jitter_ms',
                       'network_load', 'packet_rate_pps']
        print("  [Graph 3] Throughput scatter...")
        self.plot_scatter_relationships(df, net_metrics, 'throughput_kbps', 'throughput_scatter.png')
        print("  [Graph 4] Latency scatter...")
        self.plot_scatter_relationships(df, net_metrics, 'latency_ms', 'latency_scatter.png')

        print("  [Graph 5] Time series...")
        self.plot_time_series(df, key_metrics)

        print("  [Graph 6] Load impact...")
        self.plot_load_impact(df)

        print("  EDA complete (6 graphs saved)")
        return corr_matrix


if __name__ == "__main__":
    df = pd.read_csv('../data/5g_ran_dataset_clean.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    analyzer = EDAAnalyzer(output_dir='../results')
    analyzer.run_full_eda(df)
