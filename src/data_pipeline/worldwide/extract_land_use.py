# src/data_pipeline/worldwide/extract_land_use.py
import numpy as np
import pandas as pd
import rasterio as rs
from shapely import box
import multiprocessing as mp
from functools import partial

from src.data_pipeline.common import get_file, get_grid, GEOD, calculate_area, implace_nan

def calculate_window_bounds(lon, lat, grid_step):
    left = lon - grid_step / 2
    right = lon + grid_step / 2
                
    bottom = lat - grid_step / 2
    top = lat + grid_step/ 2
    return left, bottom, right, top

def process_window(index, lon, lat, year, hilada_raster, categories, grid_resolution):
    #print(index/1036800 * 100, "%") #prints % of processed info
    result = {
            "longitude": lon,
            "latitude": lat,
            "year": year,
            "geometry": None,
            "area_total_km": np.nan,
            "area_land_km": np.nan
        }
    
    valid_cat = categories.keys() - {"None"} #all categories, except "None/NaN"
    for category in valid_cat:
        result[category] = np.nan

    minx, miny, maxx, maxy = calculate_window_bounds(lon, lat, grid_resolution)
    window = rs.windows.from_bounds(minx, miny, maxx, maxy, hilada_raster.transform)
    lulc_data = hilada_raster.read(1, window=window)
    lulc_data = implace_nan(lulc_data, categories["None"]) #replace 99 with nan values

    total_pixels = np.sum(~np.isnan(lulc_data))
    if total_pixels > 0 :
        lulc_polygon = box(minx, miny, maxx, maxy)
        result["geometry"] = lulc_polygon

        area_total, _ = GEOD.geometry_area_perimeter(lulc_polygon)
        result["area_total_km"] = area_total / 1e6
        area_land_km = calculate_area(lulc_data, [categories["Water"], categories["Ocean"]], hilada_raster.transform)

        result["area_land_km"]  = area_land_km#calculate_area(lulc_data, [categories["Water"], categories["Ocean"]], hilada_raster.transform)
        
        for category in valid_cat:
                result[category] = np.sum(lulc_data == categories[category]) / total_pixels
    return result


def extract_land_use_features(lulc_dir_path, year, resolution=0.25):
    hilda_file = get_file (lulc_dir_path, f"*_{year}_*")
    grid = get_grid(-180, 180, -90, 90, resolution) # get coordinates of the centers of the cells

    categories_hilda = {
        "Ocean": 0, 
        "Urban": 11,
        "Cropland": 22, 
        "Pasture":33, 
        "Forest": 44, 
        "Shrub":  55,  
        "Barren": 66, 
        "Water": 77, 
        "None": 99}
    
    with rs.open(hilda_file) as hilada_raster:
        results = [ process_window(index, lon, lat, year, hilada_raster, categories_hilda, resolution)
                   for index, lon, lat in grid ]
        df_lulc = pd.DataFrame(results)
        
    return df_lulc

def process_window_parallel(index, lon, lat, year, hilada_raster_path, categories, grid_resolution):
    with rs.open(hilada_raster_path) as hilada_raster:
        result = process_window(index, lon, lat, year, hilada_raster, categories, grid_resolution)
    return result

# Wrapper for parallel processing: packs multiple arguments into one for imap_unordered.
def process_window_parallel_wrapper(args):
    return process_window_parallel(*args)

# Process the land-use file in parallel using n_workers.
# batch_size defines the number of tasks assigned to a worker at a time.
# If <= 0, it is calculated automatically.

def extract_land_use_features_parallel(lulc_dir_path, year, resolution=0.25, n_workers = 4, batch_size = -1):
    hilda_file = get_file (lulc_dir_path, f"*_{year}_*")
    grid = get_grid(-180, 180, -90, 90, resolution) # get coordinates of the centers of the cells

    categories_hilda = {
        "Ocean": 0, 
        "Urban": 11,
        "Cropland": 22, 
        "Pasture":33, 
        "Forest": 44, 
        "Shrub":  55,  
        "Barren": 66, 
        "Water": 77, 
        "None": 99}


    chunksize = batch_size
    if (chunksize <= 0):
        chunksize = max(1, len(grid) // (n_workers * 10))


    args = [ (index,
            lon,
            lat,
            year,
            hilda_file,
            categories_hilda,
            resolution)
            for index, lon, lat in grid ]


    with mp.Pool(n_workers) as pool:
        results = list(pool.imap_unordered(
                process_window_parallel_wrapper,
                args,
                chunksize=chunksize,
            )
        )
    df_lulc = pd.DataFrame(results)  
    return df_lulc

