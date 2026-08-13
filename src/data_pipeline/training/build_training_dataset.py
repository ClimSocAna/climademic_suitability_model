# src/data_pipeline/training/build_training_dataset.py
from pathlib import Path
import pandas as pd

from src.data_pipeline.training.observation_mask import build_observation_mask
from src.data_pipeline.training.extract_climate import extract_climate_features
from src.data_pipeline.training.extract_land_use import extract_land_use_features
from src.data_pipeline.training.extract_population import extract_population_features

#TODO here, duplictes are not checked in any way. Add a duplicate check /removal? 

#build dataset from scratch. Currently, its not saving separate dataframes for climate, LULC and population
#TODO add savng separate dataframes for climate, LULC and population on demand
def build_training_dataset(config):
    
    paths = config["paths"]

    #load polygon observation masks 
    mask_gdf = build_observation_mask(paths["raw_data"]["compendium"], 1975, 2014) # we use data from 1975 t0 2014

    climate_gdf = extract_climate_features(paths["raw_data"]["climate"], mask_gdf)
    land_use_gdf = extract_land_use_features(paths["raw_data"]["land_use"], mask_gdf)
    population_df = extract_population_features(paths["raw_data"]["population"], mask_gdf)

    #merge extracted datasets
    merge_on_cols= ["YEAR", "X", "Y", "OCCURRENCE_ID"]
    
    land_use_gdf = land_use_gdf.drop(columns=["geometry","mask_geometry"])
    population_df = population_df.drop(columns=["geometry","mask_geometry"])

    cols_to_add_land_use = merge_on_cols + [
        col for col in land_use_gdf.columns if col not in climate_gdf.columns]
    
    cols_to_add_pop = merge_on_cols + [
        col for col in population_df.columns if col not in climate_gdf.columns]

    training_dataset = (climate_gdf.merge(land_use_gdf[cols_to_add_land_use], on=merge_on_cols, how="left")
                        .merge(population_df[cols_to_add_pop], on=merge_on_cols, how="left"))
    
    #calculate population density 
    training_dataset["pop_density"] = training_dataset["ghsl_pop_counts"] / training_dataset["area_land_km"]

    training_dataset.to_csv(paths["processed_data"]["training_dataset"], sep=',', index=False, decimal='.')

    print(f"training datased is saved as {paths["processed_data"]["training_dataset"]}")

# load training dataset. If the file (path defined in config) does nt exist, 
# setting force_rebuind=True will build file from scratch
def load_training_dataset(config, force_rebuild=False):
    paths = config["paths"]
    fanme = paths["processed_data"]["training_dataset"]
    if not Path(fanme).exists():
        if (force_rebuild):
            dir_name = fanme.parents[0]
            print(f"No training data file found in directory {dir_name}. New trainig file will be constructed...")
            training_dataset = build_training_dataset(config)
        else:
            raise FileNotFoundError(f"No training data file found in directory {dir_name}.")
    else:
        training_dataset =  pd.read_csv(fanme, low_memory=False)
    return training_dataset