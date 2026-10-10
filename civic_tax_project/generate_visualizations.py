import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm

INPUT_FEATURE_PATH = "data/processed/02_feature_matrix.csv"
OUTPUT_DIR = "reports/figures"
TARGET_ZIP = "06608"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
NAVY = "#1B3962"
SLATE = "#424D62"
CRIMSON = "#C12929"
LIGHT_GRAY = "#E6EBF2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_visualizations():
    if not os.path.exists(INPUT_FEATURE_PATH):
        raise FileNotFoundError(f"Missing dataset: {INPUT_FEATURE_PATH}. Run Step 2 first.")

    df = pd.read_csv(INPUT_FEATURE_PATH)
    df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)

    df['eitc_rank'] = df['eitc_rate_pct'].rank(method='first')
    df['eitc_quartile'] = pd.qcut(
        df['eitc_rank'], 
        q=4, 
        labels=['Q1 (Lowest)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (Highest)']
    )

    df_valid = df.dropna(subset=['eitc_rate_pct', 'paid_prep_rate_pct']).copy()
    X_ols = sm.add_constant(df_valid['eitc_rate_pct'])
    model = sm.OLS(df_valid['paid_prep_rate_pct'], X_ols).fit()
    b0 = model.params['const']
    b1 = model.params['eitc_rate_pct']
    r2 = model.rsquared
    pval = model.f_pvalue

    print("Generating Figure 1: Bivariate OLS Scatter Plot...")
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)

    sns.scatterplot(
        data=df_valid,
        x='eitc_rate_pct',
        y='paid_prep_rate_pct',
        color=SLATE,
        alpha=0.6,
        s=50,
        ax=ax,
        label='CT ZCTA (ZIP Code)'
    )

    sns.regplot(
        data=df_valid,
        x='eitc_rate_pct',
        y='paid_prep_rate_pct',
        scatter=False,
        ax=ax,
        color=NAVY,
        line_kws={'linewidth': 2.5, 'label': 'Univariate OLS Line'}
    )

    target_row = df_valid[df_valid['zip_code'] == TARGET_ZIP]
    if not target_row.empty:
        t_x = target_row['eitc_rate_pct'].values[0]
        t_y = target_row['paid_prep_rate_pct'].values[0]

        ax.scatter(t_x, t_y, color=CRIMSON, s=130, zorder=5, label=f'Target ZIP ({TARGET_ZIP})')
        
        ax.annotate(
            f"Bridgeport ({TARGET_ZIP})\nEITC Rate: {t_x:.1f}% | Paid Prep: {t_y:.1f}%",
            xy=(t_x, t_y),
            xytext=(-120, 30),
            textcoords="offset points",
            arrowprops=dict(facecolor=CRIMSON, edgecolor=CRIMSON, shrink=0.08, width=1.2, headwidth=6),
            fontsize=9,
            fontweight='bold',
            color=CRIMSON,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF5F5', edgecolor=CRIMSON, alpha=0.9)
        )

    stats_text = (
        "Univariate OLS Fit Metrics:\n"
        f"PaidPrepRate = {b0:.2f} + {b1:.2f} × (EITC_Rate)\n"
        f"R² = {r2:.4f} | p = {pval:.4f}\n"
        "Econometric Note: Low R² indicates Omitted\n"
        "Variable Bias (OVB) in simple bivariate model."
    )
    ax.text(
        0.03, 0.93, stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=LIGHT_GRAY, alpha=0.95)
    )

    ax.set_title("Paid Tax Preparer Reliance vs. EITC Utilization Density in Connecticut", fontsize=12, fontweight='bold', pad=15, color=NAVY)
    ax.set_xlabel("EITC Utilization Rate (% of Total Tax Filers)", fontsize=10, fontweight='bold', color=SLATE)
    ax.set_ylabel("Paid Tax Preparer Reliance Rate (% of Total Filers)", fontsize=10, fontweight='bold', color=SLATE)
    ax.legend(loc='lower right', frameon=True)
    
    plt.figtext(0.1, 0.01, "*Source: IRS SOI Line-Item Tax Data (2021) & US Census Bureau ACS 5-Year Estimates", fontsize=8, color=SLATE, style='italic')

    fig1_path = os.path.join(OUTPUT_DIR, "fig1_regression_plot.png")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1: {fig1_path}")

    print("Generating Figure 2: Capital Leakage Bar Chart...")
    
    quartile_summary = df.groupby('eitc_quartile', observed=False)['capital_leakage_dollars'].sum().reset_index()
    total_leakage = quartile_summary['capital_leakage_dollars'].sum()

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    colors = [SLATE, SLATE, SLATE, CRIMSON] 
    bars = ax.bar(
        quartile_summary['eitc_quartile'],
        quartile_summary['capital_leakage_dollars'] / 1000.0,
        color=colors,
        width=0.55
    )

    for bar in bars:
        height = bar.get_height()
        val_dollars = height * 1000.0
        pct_share = (val_dollars / total_leakage) * 100 if total_leakage > 0 else 0
        ax.annotate(
            f"${val_dollars:,.0f}\n({pct_share:.1f}% Share)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold',
            color=NAVY
        )

    ax.set_title("Estimated EITC Capital Leakage to Commercial Preparers by Tier ($)", fontsize=12, fontweight='bold', pad=15, color=NAVY)
    ax.set_xlabel("EITC Utilization Quartile (Q1 = Lowest EITC, Q4 = Highest EITC)", fontsize=10, fontweight='bold', color=SLATE)
    ax.set_ylabel("Total Estimated Capital Leakage ($ Thousands)", fontsize=10, fontweight='bold', color=SLATE)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x:,.0f}K'))
    ax.set_ylim(0, max(quartile_summary['capital_leakage_dollars'] / 1000.0) * 1.25)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.figtext(0.1, 0.01, "Capital leakage calculated assuming average commercial preparation fee of $250/return.", fontsize=8, color=SLATE, style='italic')

    fig2_path = os.path.join(OUTPUT_DIR, "fig2_capital_leakage.png")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[+] Saved Figure 2: {fig2_path}")

if __name__ == "__main__":
    generate_visualizations()
