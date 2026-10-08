import os
import io
import requests
import pandas as pd

TARGET_STATE = "CT"
TARGET_ZIP = "06608"  # Bridgeport, CT
IRS_DATA_URLS = [
    "https://www.irs.gov/pub/irs-soi/21zpallagi.csv",
    "https://www.irs.gov/pub/irs-soi/20zpallagi.csv"
]
PROCESSED_DATA_PATH = "data/processed/01_cleaned_tax_data.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# IRS SOI Field Definitions:
# N1: Total returns
# N59660 / N07180: EITC returns count
# A59660 / A07180: EITC amount ($ in thousands)
# PREP: Paid preparer returns count
# N11902: Direct deposit refund returns count
REQUIRED_COLS = ['STATE', 'ZIPCODE', 'AGI_STUB', 'N1', 'N59660', 'A59660', 'N07180', 'A07180', 'PREP', 'N11902', 'A11902']

def fetch_and_process_irs_data(target_state: str = TARGET_STATE):
    print(f"[*] Initializing IRS SOI Data Intake for State: {target_state}")
    
    df_raw = None
    for url in IRS_DATA_URLS:
        print(f"[*] Attempting download from: {url}")
        try:
            res = requests.get(url, headers=HEADERS, timeout=60)
            if res.status_code == 200:
                buffer = io.BytesIO(res.content)
                preview = pd.read_csv(buffer, nrows=2)
                cols_upper = [c.upper() for c in preview.columns]
                use_cols = [c for c in cols_upper if c in REQUIRED_COLS]
                
                buffer.seek(0)
                df_raw = pd.read_csv(
                    buffer,
                    usecols=lambda col: col.upper() in use_cols,
                    low_memory=False
                )
                df_raw.columns = [c.upper() for c in df_raw.columns]
                print(f"[+] Download successful! Raw Shape: {df_raw.shape}")
                break
        except Exception as e:
            print(f"[-] Download attempt failed: {e}")
            
    if df_raw is None:
        raise RuntimeError("Failed to fetch IRS dataset from configured URLs.")

    # Format ZIP codes with leading zeros
    df_raw['ZIPCODE'] = df_raw['ZIPCODE'].astype(str).str.zfill(5)
    
    # Exclude aggregate row (AGI_STUB == 0) and filter for Target State
    df_state = df_raw[(df_raw['STATE'] == target_state) & (df_raw['AGI_STUB'] > 0)].copy()

    # Drop non-geographic state summary ZIPs
    df_state = df_state[~df_state['ZIPCODE'].isin(['00000', '99999'])].copy()

    # Standardize EITC column across IRS SOI schema versions
    if 'N59660' in df_state.columns:
        df_state['eitc_returns_raw'] = df_state['N59660']
        df_state['eitc_amount_raw'] = df_state.get('A59660', 0)
    elif 'N07180' in df_state.columns:
        df_state['eitc_returns_raw'] = df_state['N07180']
        df_state['eitc_amount_raw'] = df_state.get('A07180', 0)
    else:
        df_state['eitc_returns_raw'] = 0
        df_state['eitc_amount_raw'] = 0

    metric_cols = ['N1', 'eitc_returns_raw', 'eitc_amount_raw', 'PREP', 'N11902']
    for col in metric_cols:
        if col in df_state.columns:
            df_state[col] = pd.to_numeric(df_state[col], errors='coerce').fillna(0)

    # Aggregate across AGI brackets per ZIP code
    print("[*] Aggregating AGI brackets by ZIP code...")
    df_grouped = df_state.groupby(['STATE', 'ZIPCODE'])[metric_cols].sum().reset_index()

    cols_map = {
        'STATE': 'state',
        'ZIPCODE': 'zip_code',
        'N1': 'total_returns',
        'eitc_returns_raw': 'eitc_returns',
        'eitc_amount_raw': 'eitc_amount_thousands',
        'PREP': 'paid_prep_returns',
        'N11902': 'direct_deposit_returns'
    }
    df_clean = df_grouped.rename(columns=cols_map)

    # Derived proportion: Paid Preparer Rate
    df_clean['paid_preparer_share'] = (
        df_clean['paid_prep_returns'] / df_clean['total_returns'].replace(0, pd.NA)
    ).fillna(0)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[✓] Step 1 Complete! Active ZIP Count: N = {len(df_clean)}")
    
    sample = df_clean[df_clean['zip_code'] == TARGET_ZIP]
    if not sample.empty:
        print(f"\n--- Target ZIP ({TARGET_ZIP}) Snapshot ---")
        print(sample[['zip_code', 'total_returns', 'eitc_returns', 'paid_prep_returns', 'paid_preparer_share']].to_string(index=False))

if __name__ == "__main__":
    fetch_and_process_irs_data()