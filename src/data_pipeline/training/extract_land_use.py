# data_pipeline/training/extract_land_use.py
import rasterio as rs
import numpy as np
from shapely import box, Polygon, MultiPolygon
from pyproj import Geod
import pandas as pd
from pathlib import Path

from src.data_pipeline.common import get_window, get_file, GEOD, calculate_area

def _drop_only_water(dataframe):
    mask = ~((dataframe[['Ocean', 'Water']].ne(0.0).any(axis=1)) & (dataframe[['Urban', 'Cropland', 'Pasture', 'Forest', 'Shrub', 'Barren']].eq(0.0).all(axis=1)))
    dataframe = dataframe[mask]
    return dataframe

# count categories for each row / observation polygon
def _count_categories(row, raster, categories):

    #some observation polygones are crossing the antimeridian ()+/-180 degree longitude, and theregore 
    # getting split into two (Multipolygon). Processing needs to be done for each polygon in the Multipolygon
    geometries = (list(row.mask_geometry.geoms) if row.mask_geometry.geom_type == "MultiPolygon"
                  else [row.mask_geometry])

    valid_cat = categories.keys() - {"None"} #all categories, except "None/NaN"

    area_total_km = 0.0
    area_land_km  = 0.0

    #result = {category: np.nan for category in valid_cat}
    category_counts = {category: 0.0 for category in valid_cat}
    total_valid_pixels = 0.0
    hilda_polygons = []

    for geom in geometries:
        #define the window that fully contains the observation (sub)polygon
        hilda_window = get_window(geom.bounds, raster) 
        
        window_transform = raster.window_transform(hilda_window)  # Get transform for this window
        
        #read only part of the data inside the defined window
        data = raster.read(1, window=hilda_window)

        data = np.where(data == categories["None"], np.nan, data) #replace 99 with np.nan values    
        total_pixels = np.sum(~np.isnan(data)) # calculate the total number of not-Nan data pixels

        total_valid_pixels += total_pixels
        if total_pixels > 0: #if it is 0 - do nothing. default values are already nan
            for category in valid_cat:
                # Calculate the proportion of pixels belonging to each category,
                # # relative to the total number of valid (non-NaN) pixels.
                #TODO calculate the fracture based on the area, not on the amount of pixels
                category_counts[category] += np.sum(data == categories[category])
    
        # define and save the polygon bounds/geomatry, used to calculate the results 
        # (usually slightly  larger than the original bounds.)
        hilda_window_bounds = rs.windows.bounds(hilda_window, raster.transform)
        hilda_polygon = box(*hilda_window_bounds)
        
        hilda_polygons.append(hilda_polygon)
    
        # calculate area of land in km^2. will be used later to calculate population density
        area_total_geom, _ = GEOD.geometry_area_perimeter(hilda_polygon)
        area_total_km += abs(area_total_geom)

        area_land_geom = calculate_area(data, [categories["Water"], categories["Ocean"]], window_transform)
        area_land_km += abs(area_land_geom)

    # Calculate proportions AFTER processing all geometries
    result = { category: (category_counts[category]/ total_valid_pixels if total_valid_pixels > 0 else np.nan)
              for category in valid_cat}

    # Preserve all window polygons
    result["hilda_polygon"] = (hilda_polygons[0] if len(hilda_polygons) == 1
                               else MultiPolygon(hilda_polygons))
    result["area_total_km"] = (area_total_km / 1e6 if total_valid_pixels > 0 else np.nan)
    result["area_land_km"]  = (area_land_km if total_valid_pixels > 0 else np.nan)
    
    return pd.Series(result)

def extract_land_use_features(lulc_dir_path, mask_gdf):

    hilda_gdf = mask_gdf.copy()
    hilda_dir = Path(lulc_dir_path)

    #process only years from mask_gdf
    year_start = hilda_gdf["YEAR"].min()
    year_end = hilda_gdf["YEAR"].max()

    years = np.arange(year_start, year_end + 1)

    categories_hilda = {
        "Ocean": 0, 
        "Urban": 11,
        "Cropland": 22, 
        "Pasture":33, 
        "Forest": 44, 
        "Shrub":  55,  
        "Barren": 66, 
        "Water": 77, 
        "None": 99
    }

    valid_cat = categories_hilda.keys() - {"None"} # all categories, except "None" / "Nan"
    feature_cols = (list(valid_cat) +  ["hilda_polygon", "area_land_km", "area_total_km"])

    for category in valid_cat: 
        hilda_gdf[category] = np.nan 
    hilda_gdf["area_land_km"] = np.nan 
    hilda_gdf["area_total_km"] = np.nan 
    hilda_gdf["hilda_polygon"] = None
    
    for year in years:
        # for a particular year, find corresponding HILDA+ file. 
        # It is assumed that the filename pattern contains the "_{year}_" part
        # and can be uniquely assigned to a particular year.    
        hilda_fname = get_file(hilda_dir, f"*_{year}_*")
    
        print("Processing year: ", year, ", HILDA+ file  ", hilda_fname.name)
        year_mask = (hilda_gdf["YEAR"] == year)

        if (year_mask.any()):
            with rs.open(hilda_fname) as hilda_raster:

                result = hilda_gdf.loc[year_mask].apply(lambda row: _count_categories(row, hilda_raster, categories_hilda), axis=1,)

            hilda_gdf.loc[year_mask, feature_cols] = result[feature_cols]
    hilda_gdf = _drop_only_water(hilda_gdf)
    return hilda_gdf