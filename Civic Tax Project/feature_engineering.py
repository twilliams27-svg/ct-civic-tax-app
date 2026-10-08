import os
import requests
import pandas as pd
import numpy as np

INPUT_TAX_PATH = "data/processed/01_cleaned_tax_data.csv"
OUTPUT_FEATURE_PATH = "data/processed/02_feature_matrix.csv"
TARGET_ZIP = "06608"  # Bridgeport, CT

AVG_TAX_PREP_FEE = 250.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

def fetch_census_acs_data(target_zips_list):
    print("[*] Fetching Census ACS Demographic Data via API...")
    acs_vars = ["NAME", "B17001_001E", "B17001_002E", "B19013_001E"]
    var_str = ",".join(acs_vars)
    census_url = f"https://api.census.gov/data/2021/acs/acs5?get={var_str}&for=zip%20code%20tabulation%20area:*"
    
    try:
        res = requests.get(census_url, headers=HEADERS, timeout=30)
        if res.status_code == 200 and res.text.startswith('['):
            data = res.json()
            headers = data[0]
            rows = data[1:]
            df_census = pd.DataFrame(rows, columns=headers)
            
            df_census['zip_code'] = df_census['zip code tabulation area'].astype(str).str.zfill(5)
            df_census = df_census[df_census['zip_code'].isin(target_zips_list)].copy()
            
            for col in ["B17001_001E", "B17001_002E", "B19013_001E"]:
                df_census[col] = pd.to_numeric(df_census[col], errors='coerce')
                df_census[col] = df_census[col].replace([-666666666, -888888888, -999999999], np.nan)
            
            df_census['poverty_rate_pct'] = (
                df_census['B17001_002E'] / df_census['B17001_001E'].replace(0, np.nan)
            ) * 100.0
            df_census['median_household_income'] = df_census['B19013_001E']
            df_census['lep_rate_pct'] = np.nan
            
            return df_census[['zip_code', 'median_household_income', 'poverty_rate_pct', 'lep_rate_pct']].copy()
        else:
            return None
    except Exception as e:
        print(f"[-] Census API fetch failed: {e}")
        return None

def build_feature_matrix():
    if not os.path.exists(INPUT_TAX_PATH):
        raise FileNotFoundError(f"Missing input tax dataset: {INPUT_TAX_PATH}. Run Step 1 first.")

    df_tax = pd.read_csv(INPUT_TAX_PATH)
    df_tax['zip_code'] = df_tax['zip_code'].astype(str).str.zfill(5)

    ct_zips = df_tax['zip_code'].unique().tolist()
    df_census = fetch_census_acs_data(ct_zips)
    
    if df_census is not None and not df_census.empty:
        df_merged = pd.merge(df_tax, df_census, on='zip_code', how='left')
    else:
        df_merged = df_tax.copy()
        df_merged['median_household_income'] = np.nan
        df_merged['poverty_rate_pct'] = np.nan
        df_merged['lep_rate_pct'] = np.nan

    # Force 0–100 Percentage Scale for both metrics
    df_merged['paid_prep_rate_pct'] = (
        df_merged['paid_prep_returns'] / df_merged['total_returns'].replace(0, np.nan)
    ) * 100.0

    df_merged['eitc_rate_pct'] = (
        df_merged['eitc_returns'] / df_merged['total_returns'].replace(0, np.nan)
    ) * 100.0

    # Capital Leakage Calculation
    df_merged['estimated_eitc_paid_returns'] = df_merged['eitc_returns'] * (df_merged['paid_prep_rate_pct'] / 100.0)
    df_merged['capital_leakage_dollars'] = df_merged['estimated_eitc_paid_returns'] * AVG_TAX_PREP_FEE

    # Rank-based Quartile Segmentation
    df_merged['eitc_rank'] = df_merged['eitc_rate_pct'].rank(method='first')
    df_merged['eitc_quartile'] = pd.qcut(
        df_merged['eitc_rank'],
        q=4,
        labels=['Q1 (Lowest EITC)', 'Q2 (Low-Mid EITC)', 'Q3 (Mid-High EITC)', 'Q4 (Highest EITC)']
    )

    if df_merged['median_household_income'].notnull().sum() >= 4:
        df_merged['income_rank'] = df_merged['median_household_income'].rank(method='first')
        df_merged['income_quartile'] = pd.qcut(
            df_merged['income_rank'],
            q=4,
            labels=['Q1 (Lowest)', 'Q2 (Low-Mid)', 'Q3 (Mid-High)', 'Q4 (Highest)']
        )

    # Rounding
    df_merged['paid_prep_rate_pct'] = df_merged['paid_prep_rate_pct'].round(2)
    df_merged['eitc_rate_pct'] = df_merged['eitc_rate_pct'].round(2)
    df_merged['poverty_rate_pct'] = df_merged['poverty_rate_pct'].round(2)
    df_merged['capital_leakage_dollars'] = df_merged['capital_leakage_dollars'].round(2)

    os.makedirs(os.path.dirname(OUTPUT_FEATURE_PATH), exist_ok=True)
    df_merged.to_csv(OUTPUT_FEATURE_PATH, index=False)
    print(f"[✓] Step 2 Complete! Saved feature matrix to: {OUTPUT_FEATURE_PATH}")

    target_row = df_merged[df_merged['zip_code'] == TARGET_ZIP]
    if not target_row.empty:
        r = target_row.iloc[0]
        print(f"\n--- Target ZIP ({TARGET_ZIP}) Summary ---")
        print(f" Total Returns:             {int(r['total_returns']):,}")
        print(f" EITC Returns:              {int(r['eitc_returns']):,} ({r['eitc_rate_pct']:.1f}%)")
        print(f" Paid Preparer Rate:        {r['paid_prep_rate_pct']:.1f}%")
        print(f" Est. Capital Leakage:      ${r['capital_leakage_dollars']:,.2f}")

if __name__ == "__main__":
    build_feature_matrix()