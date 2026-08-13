# src/data_pipeline/worldwide/build_global_dataset.py
import pandas as pd
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely import wkt

from src.data_pipeline.worldwide.extract_climate import extract_climate_features
from src.data_pipeline.worldwide.extract_land_use import extract_land_use_features, extract_land_use_features_parallel
from src.data_pipeline.worldwide.extract_population import extract_population_features
from src.data_pipeline.common import ensure_geometry

def process_climate(config, year, features=['t2m', 'd2m', 'si10', 'tp'], resolution=0.25):
    paths = config["paths"]
    df_climate = extract_climate_features(paths["raw_data"]["climate"], year, features=features, resolution=resolution)
    fname_climate = Path(paths["processed_data"]["climate_worldwide"], f"{year}_climate_worldwide_res_{resolution}_deg.csv")
    df_climate.to_csv(fname_climate,  index=False, decimal='.', )
    print(f"processed climate data for the {year} is saved as {fname_climate}")
    return df_climate

def process_land_use(config, year, resolution = 0.25):
    paths = config["paths"]
    #df_land_use = extract_land_use_features(paths["raw_data"]["land_use"], year, resolution=resolution)
    df_land_use = extract_land_use_features_parallel(paths["raw_data"]["land_use"], year, resolution=resolution, n_workers=8)
    fname_land_use = Path(paths["processed_data"]["land_use_worldwide"], f"{year}_land_use_worldwide_res_{resolution}_deg.csv")
    df_land_use.to_csv(fname_land_use, sep=',', index=False, decimal='.')#, float_format='%.3f')
    print(f"processed land use data for the {year} is saved as {fname_land_use}")
    return df_land_use

def process_population(config, year, resolution = 0.25):
    paths = config["paths"]
    df_population = extract_population_features(paths["raw_data"]["population"], year, resolution=resolution)
    fname_population = Path(paths["processed_data"]["population_worldwide"], f"{year}_population_worldwide_res_{resolution}_deg.csv")
    df_population.to_csv(fname_population, sep=',', index=False, decimal='.')#), float_format='%.3f')
    print(f"processed population data for the {year} is saved as {fname_population}")
    return df_population

#TODO add possibility to assemble datasets from saved land_use.csv, climate.csv and population.csv files
#TODO add possibility to define the grid limits. (now  its full (-180; 180) for longitude and (-90; 90)for latitude)
def build_global_dataset(config, year, resolution = 0.25):
    paths = config["paths"]
    df_climate = process_climate(config, year)
    df_land_use = process_land_use(config, year)
    df_population = process_population(config, year)

    global_df = assemble_global_dataset(df_climate, df_land_use, df_population)

    fname_global = Path(paths["processed_data"]["global_dataset"], f"{year}_global_clima_lulc_pop_res_{resolution}_deg.csv")
    global_df.to_csv(fname_global, sep=',', index=False, decimal='.')

    return global_df

# Assembles a full dataset from climet, land use and population data
#TODO control columns at the final dataset
def assemble_global_dataset(df_climate, df_land_use, df_population):
    df_land_use['geometry'] = df_land_use['geometry'].apply(ensure_geometry)
    gdf_land_use = gpd.GeoDataFrame(df_land_use, geometry='geometry')
    gdf_land_use.crs = 'epsg:4326'
    
    bounds = gdf_land_use.geometry.bounds
    gdf_land_use["corner_x"] = bounds.maxx.round(6)
    gdf_land_use["corner_y"] = bounds.maxy.round(6)

    df_climate["corner_x"] = df_climate["longitude"].round(6)
    df_climate["corner_y"] = df_climate["latitude"].round(6)

    global_df = gdf_land_use.merge(
        df_climate.drop(columns=["longitude", "latitude"]),
        on=["corner_x", "corner_y", "year"],
        how="inner",
        )
    
    df_population = df_population.drop(columns=["ghsl_polygon"])
    global_df = global_df.merge(df_population,
        on=["longitude", "latitude", "year"],
       how="left",)

    mask = ~((global_df[['Ocean', 'Water']].ne(0.0).any(axis=1)) & (global_df[['Urban', 'Cropland', 'Pasture', 'Forest', 'Shrub', 'Barren']].eq(0.0).all(axis=1)))
    global_df = global_df[mask]
    global_df= global_df.dropna()

    global_df = global_df.drop(columns = ["corner_x", "corner_y"])
        
    global_df["pop_density"] = global_df["ghsl_pop_counts"] / global_df["area_land_km"]

    return global_df

# Tries to load global dataset. if froce_rebuild=True, dataset would be re-created if not found
#TODO some file names are hard-coded. move them to config file
def load_global_dataset(config, year, froce_rebuild=False, resolution = 0.25):
    paths = config["paths"]
    fname_global = Path(paths["processed_data"]["global_dataset"], f"{year}_global_clima_lulc_pop_res_{resolution}_deg.csv")
    if not Path(fname_global).exists():
        if (froce_rebuild):
            dir_name = fname_global.parents[0]
            print(f"No global dataset file file found in directory {dir_name}. New dataset will be constructed...")
            global_dataset = build_global_dataset(config, year, resolution)
        else:
            raise FileNotFoundError(f"No global dataset file file found in directory {dir_name}.")
    else:
        global_dataset =  pd.read_csv(fname_global, low_memory=False)
    return global_dataset