import pandas as pd
import streamlit as st

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
 
def neighbourhood_select():
    nb = './Assets/Neighbourhoods - 4326/Neighbourhoods - 4326.shp'
    regions = gpd.read_file(nb)
    return regions
    # regions['neighbourhood'] = regions['AREA_DE8'].str.replace(' \(.+\)', '').str.lower()
    neighbourhood = st.selectbox(
        'Choose a Neighbourhood',
        regions['AREA_DE8'].sort_values().unique().tolist(),
        index=None,
        placeholder='start typing...'
    )
    
    st.caption("If you don't know your neighbourhood, you can look it up here: [Find Your Neighbourhood](https://www.toronto.ca/city-government/data-research-maps/neighbourhoods-communities/neighbourhood-profiles/find-your-neighbourhood/#location=&lat=&lng=&zoom=)") 
    print("test",neighbourhood)
