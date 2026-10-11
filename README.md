# CT Civic Tax Access and EITC Capital Leakage Index

Welcome to the repo for the civic data project. The main goal here was to look past broad statewide tax averages and focus on how tax preparation access and the Earned Income tax Credit actually play out at a local level across Connecticut zip codes.

When lower income filers rely a lot on paid commercial tax preparation, a significant portion of their tax credits get chipped away by preparation fees. this project tries to make that clear in numbers, which I call EITC capital leakage.


**The Interactive Streamlit App**
To make the data accessible, I built a live app using Streamlit.
- Users can search or click through various Connecticut ZIP codes to see specific local metrics, such as EITC utilization rates and reliance on paid preparers.
- An interactive map helps visualize geographically across Connecticut ZCTAs, making areas where commercial preparation fees disproportionally impact communities.
- The dashboard groups data into utilization quartiles to show how capital leakage scales across different socioeconomic tiers throughout the state.

**Data Sources and Modeling**
The backend pulls from real world administrative and demographic data:
- IRS SOI Line Item Tax Data (2021) for filing and credit statistics
- US Census ACS 5 year Estimates for local socioeconomic indicators
- Econometric modeling: includes and OLS regression scatter plot to show the relationship between paid preparer reliance and EITC utilization rates. For quality analysis, the model explicitly notes its low R squared and points out the potential omitted variable bias as to not hide behind messy correlations.

## Technology Used
- **Python** for core programming language used for data processing
- **Streamlit** used to build and host the interactive component
- **Pandas and NumPy** for data cleaning, aggregation, and computations
- **Geopandas and Folium** for spatial data processing and interactive mapping for the CT zip code tabulation areas
- **Matplotlib and Seaborn** for statistic data representation, OLS regression plotting, and custom style
