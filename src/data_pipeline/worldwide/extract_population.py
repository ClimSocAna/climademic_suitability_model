# src/data_pipeline/worldwide/extract_population.py
import rasterio as rs
import numpy as np
from shapely import box
from pathlib import Path
import pandas as pd
from src.data_pipeline.common import get_file, get_grid, implace_nan

# (lon, lat) are the center coordinates of the window. 
# Compute the bounding coordinates (west, south, east, north) based on the center point and resolution.
def calculate_window_bounds(lon, lat, grid_step):
    left = lon - grid_step / 2
    right = lon + grid_step / 2
                
    bottom = lat - grid_step / 2
    top = lat + grid_step/ 2
    return left, bottom, right, top

# interpolate the data for target_year
def interpolate_pop_data(start_data, end_data, start_year, end_year, target_year):
    ratio = (target_year - start_year) / (end_year - start_year)
    interpolated_data = start_data + ratio * (end_data - start_data)
    return interpolated_data

#TODO calculate total area and populated area (# of populated pixels)?
def process_window(index, lon, lat, year, start_year, end_year, pop_raster_start, pop_raster_end, 
                   interpolate_flag, grid_resolution):
    result = {
        "longitude": lon,
        "latitude": lat,
        "year": year,
        "ghsl_polygon": None,
    }

    minx, miny, maxx, maxy = calculate_window_bounds(lon, lat, grid_resolution)
    ghsl_polygon = box(minx, miny, maxx, maxy)
    result["ghsl_polygon"] = ghsl_polygon
    
    if interpolate_flag:
        window_start = rs.windows.from_bounds(minx, miny, maxx, maxy, pop_raster_start.transform)
        window_end   = rs.windows.from_bounds(minx, miny, maxx, maxy, pop_raster_end.transform)

        start_data = pop_raster_start.read(1, window=window_start)
        start_data = implace_nan(start_data, -200) #replace -200 with nan values
        
        end_data = pop_raster_end.read(1, window=window_end)
        end_data = implace_nan(end_data, -200) #replace -200 with nan values
        
        ghsl_data_year = interpolate_pop_data(start_data, end_data, start_year, end_year, year)
    else:
        window = rs.windows.from_bounds(minx, miny, maxx, maxy, pop_raster_start.transform)
        ghsl_data_year = pop_raster_start.read(1, window=window)
        ghsl_data_year = implace_nan(ghsl_data_year, -200) #replace -200 with nan values

    if ~np.all(np.isnan(ghsl_data_year)):
        result["ghsl_pop_counts"] = np.nansum(ghsl_data_year)
    return result


def extract_population_features(pop_dir_path, year, resolution=0.25):

    pop_dir_path = Path(pop_dir_path)
    
    interpolate_flag = False
    grid = get_grid(-180, 180, -90, 90, resolution)
    
    #if the year corresonds to ghsl step, no interpolation needed 
    if year % 5 == 0:
        file = get_file(pop_dir_path, f"*_E{year}_*")
        with rs.open(file) as pop_raster:
            
            results = [ process_window(index, lon, lat, year, None, None,
                            pop_raster, None, interpolate_flag, resolution)
                            for index, lon, lat in grid ]
    #if the year does not correspond to ghsl step,  interpolation is necessary 
    else: 

        #calculate years before and after the target year, as a source for interpolation
        prev_year = (year // 5) * 5
        next_year = prev_year + 5

        file_start = get_file(pop_dir_path, f"*_E{prev_year}_*")
        file_end   = get_file(pop_dir_path, f"*_E{next_year}_*")
        interpolate_flag = True

        with rs.open(file_start) as pop_raster_start, \
             rs.open(file_end) as pop_raster_end:
            
            results = [ process_window(index, lon, lat, year, prev_year, next_year,
                            pop_raster_start, pop_raster_end, interpolate_flag, resolution)
                            for index, lon, lat in grid ]
        
    df_population = pd.DataFrame(results)
    return df_population