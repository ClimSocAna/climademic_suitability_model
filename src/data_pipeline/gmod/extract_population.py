import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from src.data_pipeline.common import get_file

def extract_population_features(directory_global, gmod_df):

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
        population_file = get_file (directory_global, f"{year}_population_*")
        #read it as geodataframe
        population_df = pd.read_csv(population_file)

        population_df['geometry'] = population_df["ghsl_polygon"].apply(
                lambda x: wkt.loads(x) if pd.notna(x) else None)
        gdf_population = gpd.GeoDataFrame(population_df, geometry='geometry')
        gdf_population.crs = 'epsg:4326'

        #rename columns
        gdf_population = gdf_population.rename(
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
            gdf_population,
            gmod_year.drop(columns="year"),
            how="inner",
            predicate="contains",
            )
        
        matched = (
            matched
            .drop(columns=["index_right", 'ghsl_polygon'])
        )
    
        all_years.append(matched)

    final_df = pd.concat(all_years, ignore_index=True)
    return final_df