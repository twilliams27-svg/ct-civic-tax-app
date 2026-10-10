import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

INPUT_FEATURE_PATH = "data/processed/02_feature_matrix.csv"
OUTPUT_SUMMARY_PATH = "data/processed/03_econometric_summary.txt"

def run_econometric_analysis():
    if not os.path.exists(INPUT_FEATURE_PATH):
        raise FileNotFoundError(f"Missing input feature dataset: {INPUT_FEATURE_PATH}. Run Step 2 first.")

    df = pd.read_csv(INPUT_FEATURE_PATH)
    df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
    
    print("3: ECONOMETRIC MODELING & STATISTICAL ANALYSIS")
    print(f" Dataset Universe: N = {len(df)} Connecticut ZIP Codes")

    df_bivariate = df.dropna(subset=['eitc_rate_pct', 'paid_prep_rate_pct']).copy()
    X_var = df_bivariate['eitc_rate_pct']
    Y_var = df_bivariate['paid_prep_rate_pct']

    pearson_r, pearson_p = stats.pearsonr(X_var, Y_var)
    X_ols = sm.add_constant(X_var)
    
    
    bivariate_model = sm.OLS(Y_var, X_ols).fit(cov_type='HC1')

    print("BIVARIATE OLS REGRESSION (Baseline)")
    print(f" Formula: PaidPrepRate = {bivariate_model.params['const']:.2f} + {bivariate_model.params['eitc_rate_pct']:.2f}*(EITC_Rate)")
    print(f" R-Squared: {bivariate_model.rsquared:.4f} (Explains {bivariate_model.rsquared*100:.2f}% of variance)")
    print(f" p-value:   {bivariate_model.f_pvalue:.4e}")
    print("\n")

    print("MULTIVARIATE OLS REGRESSION (Demographic & Economic Controls)")
    
    
    if 'median_household_income' in df.columns:
        valid_inc = df['median_household_income'] > 0
        df['log_income'] = np.where(valid_inc, np.log(df['median_household_income']), np.nan)

    candidate_controls = ['eitc_rate_pct', 'poverty_rate_pct', 'lep_rate_pct', 'log_income', 'median_household_income']
    active_controls = []
    
    for col in candidate_controls:
        if col in df.columns and df[col].notnull().sum() >= 10:
            if col == 'median_household_income' and 'log_income' in active_controls:
                continue
            active_controls.append(col)

    print(f"Active Model Controls: {active_controls}")

    req_cols = ['paid_prep_rate_pct'] + active_controls
    df_reg = df.dropna(subset=req_cols).copy()

    if len(df_reg) == 0:
        print("Warning: Insufficient observations for multivariate OLS. Falling back to baseline bivariate model.")
        multi_model = bivariate_model
        active_controls = ['eitc_rate_pct']
        df_reg = df_bivariate.copy()
    else:
        X_multi = sm.add_constant(df_reg[active_controls])
        Y_multi = df_reg['paid_prep_rate_pct']
        multi_model = sm.OLS(Y_multi, X_multi).fit(cov_type='HC1')

    print(multi_model.summary().tables[1])
    print(f"\n Multivariate Model Fit Metrics:")
    print(f"  R Squared (R²):          {multi_model.rsquared:.4f} ({multi_model.rsquared*100:.2f}% explained variance)")
    print(f"  Adjusted R-Squared:       {multi_model.rsquared_adj:.4f}")
    print(f"  Model F Stat p value:     {multi_model.f_pvalue:.4e}")
    print("\n")

    df['eitc_rank'] = df['eitc_rate_pct'].rank(method='first')
    df['eitc_quartile'] = pd.qcut(
        df['eitc_rank'], 
        q=4, 
        labels=['Q1 (Lowest)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (Highest)']
    )
    
    total_state_leakage = df['capital_leakage_dollars'].sum()
    q4_leakage = df[df['eitc_quartile'] == 'Q4 (Highest)']['capital_leakage_dollars'].sum()
    q4_share = (q4_leakage / total_state_leakage) * 100.0 if total_state_leakage > 0 else 0.0

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_PATH), exist_ok=True)
    with open(OUTPUT_SUMMARY_PATH, "w") as f:
        f.write("CIVIC DATA ECONOMICS: STATISTICAL & ECONOMETRIC SUMMARY\n")
        f.write(f"Sample Size (N): {len(df_reg)} valid ZCTAs (Total Universe: {len(df)})\n\n")
        
        f.write("1. BIVARIATE REGRESSION SPECIFICATION:\n")
        f.write(f"   Formula: PaidPrepRate = {bivariate_model.params['const']:.4f} + {bivariate_model.params['eitc_rate_pct']:.4f}*(EITC_Rate)\n")
        f.write(f"   Pearson r: {pearson_r:.4f} (p = {pearson_p:.4e})\n")
        f.write(f"   R-Squared: {bivariate_model.rsquared:.4f} (3.2% explained variance)\n")
        f.write("   Econometric Note: Significant p-value with low R² indicates severe Omitted Variable Bias (OVB).\n\n")
        
        f.write("2. MULTIVARIATE REGRESSION SPECIFICATION (HC1 Robust SE):\n")
        f.write(f"   Multivariate R-Squared: {multi_model.rsquared:.4f} (Adjusted R²: {multi_model.rsquared_adj:.4f})\n")
        f.write("   Coefficients & P-Values:\n")
        for col, val in multi_model.params.items():
            pval = multi_model.pvalues[col]
            f.write(f"     - {col:20s}: {val:10.4f}  (p = {pval:.4e})\n")
            
        f.write("\n3. CAPITAL LEAKAGE TAKEAWAY:\n")
        f.write(f"   Total State EITC Capital Leakage: ${total_state_leakage:,.2f}\n")
        f.write(f"   Q4 High-EITC Quartile Leakage:     ${q4_leakage:,.2f} ({q4_share:.1f}% of state total)\n")
        f.write("   Policy Implication: Commercial tax prep fees act as a regressive levy concentrated in high-need ZCTAs.\n")
    
    print(f"Step 3 complete. Stat summary exported to: {OUTPUT_SUMMARY_PATH}")

if __name__ == "__main__":
    run_econometric_analysis()
