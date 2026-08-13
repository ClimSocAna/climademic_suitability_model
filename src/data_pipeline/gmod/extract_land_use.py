# src/data_pipeline/gmod/extract_land_use.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from src.data_pipeline.common import get_file

def extract_land_use_features(directory_global, gmod_df):

    gmod_gdf = gpd.GeoDataFrame(gmod_df, geometry=gpd.points_from_xy(gmod_df.longitude, gmod_df.latitude),
                                crs='EPSG:4326')
    
    #iterate over all years from gmod_gdf
    year_start = gmod_gdf["year"].min()
    year_end =  gmod_gdf["year"].max()

    all_years = []
    for year in np.arange(year_start, year_end + 1):
        #print(year)
        #find a land use file for a chosen yer
        #TODO also allow extracting data from assembeled global dataset, not only for a land use 
        land_use_file = get_file (directory_global, f"{year}_land_use_*")
        #read it as geodataframe
        land_use_df = pd.read_csv(land_use_file)
        land_use_df = land_use_df.dropna(subset=["geometry"]).copy()
        #TODO decide how to work around that: drop nA here, or allow load geomtery as "Nan"
        land_use_df['geometry'] = land_use_df['geometry'].apply(
            lambda x: wkt.loads(x) if pd.notna(x) else None)
        gdf_land_use = gpd.GeoDataFrame(land_use_df, geometry='geometry')
        gdf_land_use.crs = 'epsg:4326'

        #rename columns
        gdf_land_use = gdf_land_use.rename(
            columns={
                "longitude": "polygon_center_lon",
                "latitude":  "polygon_center_lat",
            })
        #filter year
        gmod_year = gmod_gdf.loc[gmod_gdf["year"] == year].copy()
        gmod_year = gmod_year.rename(
            columns={
                "longitude": "obs_longitude",
                "latitude":  "obs_latitude",
            })
        matched = gpd.sjoin(
            gdf_land_use,
            gmod_year.drop(columns="year"),
            how="inner",
            predicate="contains",
            )
        
        matched = (
            matched
            .drop(columns=["index_right"])
        )
    
        all_years.append(matched)

    final_gdf = pd.concat(all_years, ignore_index=True)
    return final_gdf





