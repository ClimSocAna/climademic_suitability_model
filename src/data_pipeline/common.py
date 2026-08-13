# src/data_pipeline/common.py
from pathlib import Path
import numpy as np
from shapely import Polygon, wkt
import pandas as pd
from shapely.geometry.base import BaseGeometry

import rasterio as rs
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

def get_file (dir_path, mask):
    directory = Path(dir_path)
    file = next(directory.glob(mask), None) 
    if file is None:
        raise FileNotFoundError(f"No file matching {mask} found in {directory}")
    return file

# Get window bounds for the given bounds.
# In this case, the bounds do not perfectly align with the cells of the resolution grid.
# First, we identify the pixels/cells that contain the bounds.
# Then, we use the full extent of those pixels.
# As a result, the window is usually slightly larger than the original bounds.
def get_window(bounds, raster):
 
    minx, miny, maxx, maxy = bounds

    # Define the index slice containing the mask polygon.
    # Rasterio indices start from the upper-left corner,
    # so row_start has a higher y-coordinate than row_end.
    row_start = raster.index(minx, maxy)[0] #define the index of the row containing coordinates (minx, maxy)
    row_end   = raster.index(minx, miny)[0]

    col_start = raster.index(minx, miny) [1]
    col_end   = raster.index(maxx, maxy) [1]

    # In most cases, the window will be slightly larger than the mask polygon,
    # since it fully contains the polygon.
    window = rs.windows.Window.from_slices(rows=(row_start, row_end + 1), cols=(col_start, col_end +1))
    
    return window

def get_grid(grid_lon_min, grid_lon_max, grid_lat_min, grid_lat_max, grid_resolution):
    lon_lat_combinations = [
                    (index, lon, lat)
                    for index, (lon, lat) in enumerate(
                        (lon, lat)
                        for lon in np.arange(grid_lon_min + grid_resolution / 2, grid_lon_max, grid_resolution)
                        for lat in np.arange(grid_lat_min + grid_resolution / 2, grid_lat_max, grid_resolution)
                    ) ] 
    lon_lat_combinations = np.array(lon_lat_combinations)
    return lon_lat_combinations

#calculates the area of pixels not in the excluded_values
def calculate_area(data, excluded_values, transform):

    valid_mask = ~np.isin(data, excluded_values) & ~np.isnan(data)
    rows, cols = np.where(valid_mask)
    total_area_km2 = 0.0
    for r, col in zip(rows, cols):
        # Get the coordinates of the cell corners
        lon1, lat1 = transform * (col, r)  # Top-left corner
        lon2, lat2 = transform * (col + 1, r)  # Top-right corner
        lon3, lat3 = transform * (col + 1, r + 1)  # Bottom-right corner
        lon4, lat4 = transform * (col, r + 1)  # Bottom-left corner
        
        # Calculate the area of the cell
        
        polygon = Polygon([(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4), (lon1, lat1)])
        area, _ = GEOD.geometry_area_perimeter(polygon)
        total_area_km2 += abs(area) / 1e6
    return total_area_km2

def implace_nan(data, default_nan):
    data = np.where(data == default_nan, np.nan, data) #replace default_nan with nan values
    return data

def ensure_geometry(value):
    if isinstance(value, BaseGeometry):
        return value
    if pd.isna(value):
        return None
    return wkt.loads(value)