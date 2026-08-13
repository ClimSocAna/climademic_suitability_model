# src/data_pipeline/gmod/build_gmod_dataset.py
from pathlib import Path

from src.data_pipeline.gmod.extract_observations import extract_observations
from src.data_pipeline.gmod.extract_climate import extract_climate_features
from src.data_pipeline.gmod.extract_land_use import extract_land_use_features
from src.data_pipeline.gmod.extract_population import extract_population_features

def get_gmod_observations(year_start, year_end, config):
    paths = config["paths"]     
    path_inatiralist = Path(paths["raw_data"]["gmod_inaturalist"])
    path_mosquito_alert = Path(paths["raw_data"]["gmod_mosquito_alert"])
    gmod_df = extract_observations(year_start, year_end, path_inatiralist, path_mosquito_alert)
    gmod_df["observation_id"] = range(len(gmod_df))
    return gmod_df

def process_climate(config, gmod_df):
    paths = config["paths"]
    climate_path = Path(paths["raw_data"]["climate"])
    climate_df = extract_climate_features(climate_path, gmod_df)
    return climate_df

def process_land_use(config, gmod_df):
    paths = config["paths"]
    dir_land_use = Path(paths["processed_data"]["land_use_worldwide"])
    land_use_gdf = extract_land_use_features(dir_land_use, gmod_df)
    return land_use_gdf

def process_population(config, gmod_df):
    paths = config["paths"]
    dir_population = Path(paths["processed_data"]["population_worldwide"])
    dpopulation_gdf = extract_population_features(dir_population, gmod_df)
    return dpopulation_gdf

def build_gmod_dataset(year_start, year_end, config):
    gmod_df = get_gmod_observations(year_start, year_end, config)
    clima_df = process_climate(config, gmod_df)
    land_use_gdf = process_land_use(config, gmod_df)
    population_gdf = process_population(config, gmod_df)


    merge_on_cols = ["observation_id", "year"]
    new_cols_land_use = merge_on_cols + [
        c for c in land_use_gdf.columns
        if c not in clima_df.columns and c not in merge_on_cols]

    final_df = (
        clima_df
        .merge(land_use_gdf[new_cols_land_use], on=merge_on_cols, how="left")
    )

    new_cols_pop = merge_on_cols + [
        c for c in population_gdf.columns
        if c not in final_df.columns and c not in merge_on_cols]
    
    final_df = (
        final_df
        .merge(population_gdf[new_cols_pop], on=merge_on_cols, how="left")
    )

    mask = ~((final_df[['Ocean', 'Water']].ne(0.0).any(axis=1)) & (final_df[['Urban', 'Cropland', 'Pasture', 'Forest', 'Shrub', 'Barren']].eq(0.0).all(axis=1)))
    final_df = final_df[mask]
    final_df = final_df.dropna()

    final_df["pop_density"] = final_df["ghsl_pop_counts"] / final_df["area_land_km"]

    paths = config["paths"]
    fname_gmod_dataset = Path(paths["processed_data"]["gmod_dataset"]) / f"GMOD_climate_land_use_pop_{year_start}-{year_end}.csv"
    final_df.to_csv(fname_gmod_dataset, sep=',', index=False, decimal='.')