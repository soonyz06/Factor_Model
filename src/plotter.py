import polars as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plots:
    def plot_null_heatmap(self, df):
        null_mask = df.select(pl.all().is_null()).to_pandas()
        plt.figure(figsize=(12, 8))
        sns.heatmap(null_mask, cbar=False, cmap="viridis")
        plt.title("Missing Data Heatmap (Yellow = Missing, Purple = Present)")
        plt.show()
        return self

    def plot_factor_performance(self, df, X_cols):
        pdf = df.to_pandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.set_index("date").sort_index()
        pdf.index = pd.DatetimeIndex(pdf.index)
        
        daily_rets = pdf[X_cols]
        cum_rets = daily_rets.cumsum()
        ann_ret = daily_rets.mean() * 252
        ann_vol = daily_rets.std() * np.sqrt(252)
        corrs = daily_rets.corr()

        fig1, ax_cum = plt.subplots(figsize=(14, 7))
        cum_rets.plot(ax=ax_cum, lw=2)
        ax_cum.set_title("Cumulative Factor Returns (1σ Tilt)", fontsize=14, fontweight='bold')
        ax_cum.set_ylabel("Log Ret")
        ax_cum.set_xlabel("Time")
        ax_cum.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

        fig2, (ax_corr, ax_table) = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(corrs, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_corr, cbar=False, center=0)
        ax_corr.set_title("Factor Correlation Matrix", fontweight='bold')
        ax_table.axis('off')
        stats_df = pd.DataFrame({
            "Avg. Return": ann_ret,
            "Avg. Vol": ann_vol,
        }).map(lambda x: f"{x:.2%}" if abs(x) < 1.0 else f"{x:.2f}")
        table = ax_table.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        ax_table.set_title("Factor Summary", fontweight='bold')
        plt.tight_layout()
        plt.show()
        return self
