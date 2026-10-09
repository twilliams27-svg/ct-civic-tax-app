import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="CT Civic Tax Access Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
FEATURE_MATRIX_PATH = BASE_DIR / "data" / "processed" / "02_feature_matrix.csv"
LOCAL_GEOJSON_PATH = BASE_DIR / "data" / "raw" / "ct_zctas.geojson"
REMOTE_GEOJSON_URL = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ct_connecticut_zip_codes_geo.min.json"

@st.cache_data
def load_data():
    if not FEATURE_MATRIX_PATH.exists():
        st.error(f"Missing feature dataset at: {FEATURE_MATRIX_PATH}")
        st.stop()
        
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
    
    if not LOCAL_GEOJSON_PATH.exists():
        LOCAL_GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            res = requests.get(REMOTE_GEOJSON_URL, timeout=30)
            if res.status_code == 200:
                with open(LOCAL_GEOJSON_PATH, "wb") as f:
                    f.write(res.content)
            else:
                st.error("Failed to retrieve boundary map file from remote endpoint.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to fetch GeoJSON geometry: {e}")
            st.stop()

    gdf_ct = gpd.read_file(LOCAL_GEOJSON_PATH)
    
    zip_col_candidates = ['ZCTA5CE10', 'ZCTA5CE20', 'ZIP', 'zip', 'ZCTA', 'ZCTA5']
    matched_col = next((c for c in zip_col_candidates if c in gdf_ct.columns), None)
    if matched_col:
        gdf_ct = gdf_ct.rename(columns={matched_col: 'zip_code'})
    else:
        non_geom = [c for c in gdf_ct.columns if c != 'geometry'][0]
        gdf_ct = gdf_ct.rename(columns={non_geom: 'zip_code'})

    gdf_ct['zip_code'] = gdf_ct['zip_code'].astype(str).str.zfill(5)

    gdf_merged = gdf_ct.merge(df, on='zip_code', how='inner')
    
    if gdf_merged.crs != "EPSG:4326":
        gdf_merged = gdf_merged.to_crs("EPSG:4326")
        
    return gdf_merged, df

gdf, df_raw = load_data()


st.sidebar.title("Civic Tax Access")
st.sidebar.markdown("**Connecticut Community Tax Access Index**")

zip_options = ["All Connecticut ZCTAs"] + sorted(gdf['zip_code'].unique().tolist())
selected_zip = st.sidebar.selectbox("Search ZIP Code / ZCTA:", zip_options)

st.sidebar.divider()
st.sidebar.markdown("### Model Parameters")
st.sidebar.info(
    "**Est. Prep Fee:** $250 / return\n\n"
    "**Target ZIP:** 06608 (Bridgeport, CT)\n\n"
    "**Data Sources:** IRS SOI Tax Data (2021) & Census ACS"
)

st.title("Connecticut Civic Tax Access & EITC Leakage Index")
st.markdown(
    "Quantifying commercial tax preparation reliance, EITC capital extraction, "
    "and structural tax assistance gaps across Connecticut's ZCTAs."
)

if selected_zip != "All Connecticut ZCTAs":
    target_data = gdf[gdf['zip_code'] == selected_zip].iloc[0]
    st.subheader(f"Access Metrics for ZIP Code: {selected_zip}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tax Returns", f"{int(target_data['total_returns']):,}")
    m2.metric("EITC Utilization Rate", f"{target_data['eitc_rate_pct']:.1f}%")
    m3.metric("Paid Preparer Reliance", f"{target_data['paid_prep_rate_pct']:.1f}%")
    m4.metric("Est. Capital Leakage", f"${target_data['capital_leakage_dollars']:,.0f}")
else:
    st.subheader("State Level Aggregates (Connecticut)")
    m1, m2, m3, m4 = st.columns(4)
    total_state_returns = df_raw['total_returns'].sum()
    total_state_leakage = df_raw['capital_leakage_dollars'].sum()
    avg_eitc_rate = df_raw['eitc_rate_pct'].mean()
    avg_prep_rate = df_raw['paid_prep_rate_pct'].mean()
    
    m1.metric("Total State Returns", f"{total_state_returns:,}")
    m2.metric("Avg EITC Rate", f"{avg_eitc_rate:.1f}%")
    m3.metric("Avg Paid Prep Rate", f"{avg_prep_rate:.1f}%")
    m4.metric("Total State Capital Leakage", f"${total_state_leakage:,.0f}")

st.divider()

st.subheader("Estimated EITC Capital Leakage ($)")

map_center = [41.6032, -72.6877]
zoom = 9

if selected_zip != "All Connecticut ZCTAs":
    sel_geom = gdf[gdf['zip_code'] == selected_zip].geometry.iloc[0]
    map_center = [sel_geom.centroid.y, sel_geom.centroid.x]
    zoom = 12

m = folium.Map(
    location=map_center, 
    zoom_start=zoom, 
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>"
)

choropleth = folium.Choropleth(
    geo_data=gdf,
    name="Capital Leakage ($)",
    data=gdf,
    columns=["zip_code", "capital_leakage_dollars"],
    key_on="feature.properties.zip_code",
    fill_color="YlOrRd",
    fill_opacity=0.75,
    line_opacity=0.3,
    legend_name="Estimated EITC Capital Leakage ($)",
    highlight=True
).add_to(m)

tooltip = folium.GeoJsonTooltip(
    fields=["zip_code", "total_returns", "eitc_rate_pct", "paid_prep_rate_pct", "capital_leakage_dollars"],
    aliases=["ZIP Code:", "Total Returns:", "EITC Rate (%):", "Paid Prep Rate (%):", "Est. Leakage ($):"],
    localize=True,
    sticky=False
)

folium.GeoJson(
    gdf,
    style_function=lambda x: {'fillColor': '#00000000', 'color': '#333333', 'weight': 0.5},
    tooltip=tooltip
).add_to(m)

if selected_zip != "All Connecticut ZCTAs":
    sel_gdf = gdf[gdf['zip_code'] == selected_zip]
    folium.GeoJson(
        sel_gdf,
        style_function=lambda x: {'fillColor': '#00FFFF', 'color': '#0000FF', 'weight': 3, 'fillOpacity': 0.4}
    ).add_to(m)

st_folium(m, width="100%", height=580, returned_objects=[])

st.divider()
st.subheader("Embed Map on Substack / Policy Reports")
if st.button("Generate HTML Map Export"):
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    m.save(str(reports_dir / "ct_tax_access_map.html"))
    st.success("Saved `reports/ct_tax_access_map.html`!")
