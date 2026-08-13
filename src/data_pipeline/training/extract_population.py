# data_pipeline/training/extract_population.py
from pathlib import Path
import numpy as np
import rasterio as rs
from shapely import box
import pandas as pd

from src.data_pipeline.common import get_window, get_file

 
def _interpolate_pop_data(start_data, end_data, start_year, end_year, target_year):
    start_data = np.where(start_data == -200, np.nan, start_data) #replace -200 with nan values
    end_data = np.where(end_data == -200, np.nan, end_data) #replace -200 with nan values
    ratio = (target_year - start_year) / (end_year - start_year)
    interpolated_data = start_data + ratio * (end_data - start_data)
    return interpolated_data


def _count_population (row,  ghsl_start_raster, ghsl_end_raster, start_year, end_year, target_year):
    
    ghsl_window = get_window(row.mask_geometry.bounds, ghsl_start_raster)

    ghsl_window_bounds = rs.windows.bounds(ghsl_window, ghsl_start_raster.transform)
    
    ghsl_polygon = box(*ghsl_window_bounds)
    row['ghsl_polygon'] = ghsl_polygon
    
    ghsl_start_data = ghsl_start_raster.read(1, window=ghsl_window)
    ghsl_end_data   = ghsl_end_raster.read(1, window=ghsl_window)
    
    
    interpolated_ghsl_data = _interpolate_pop_data(ghsl_start_data, ghsl_end_data, start_year, end_year, target_year)
    
    #print("pop: ", np.nansum(interpolated_ghsl_data))
    ghsl_pop_counts = np.nan

    if not np.all(np.isnan(interpolated_ghsl_data)):
        ghsl_pop_counts = np.nansum(interpolated_ghsl_data)  #check if not all elements are NaN. if all are NaN - do nothing. default values are already nan
    return pd.Series({
        "ghsl_pop_counts": ghsl_pop_counts,
        "ghsl_polygon": ghsl_polygon,
    })

def extract_population_features(ghsl_dir_path, mask_gdf):

    ghsl_gdf = mask_gdf.copy()


    feature_cols = ["ghsl_pop_counts", "ghsl_polygon"]

    #init columns
    ghsl_gdf["ghsl_pop_counts"] = np.nan
    ghsl_gdf["ghsl_polygon"] = None

    ghsl_dir = Path(ghsl_dir_path)

    #process only years from mask_gdf
    year_start = ghsl_gdf["YEAR"].min()
    year_end = ghsl_gdf["YEAR"].max()
    
    #ghsl is avaliable in 5-year step 1975, 1980, ...
    #calculate right years avaliable for intepolation
    ghsl_year_start = (year_start // 5) * 5
    ghsl_year_end = (year_end // 5) * 5 + 5

    print(ghsl_year_start, ghsl_year_end)

    #years = np.arange(year_start, year_end)
    ghsl_years = np.arange(ghsl_year_start, ghsl_year_end + 1, 5)
    for i in range(len(ghsl_years) - 1):
        
        start_year_block = ghsl_years[i]
        end_year_block   = ghsl_years[i + 1]

        #read files corresponding to selected years
        ghsl_start_name = get_file(ghsl_dir, f"*_E{start_year_block}_*")
        ghsl_end_name = get_file(ghsl_dir, f"*_E{end_year_block}_*")
        
        with rs.open(ghsl_start_name) as ghsl_start_raster, rs.open(ghsl_end_name) as ghsl_end_raster:
            for target_year in np.arange(year_start, year_end + 1):
                print(target_year)
                year_mask = (ghsl_gdf["YEAR"] == target_year)
                if (year_mask.any()):
                    result = ghsl_gdf.loc[year_mask].apply(
                    lambda row: _count_population(
                                row,
                                ghsl_start_raster,
                                ghsl_end_raster,
                                start_year_block,
                                end_year_block,
                                target_year,),
                            axis=1,)
                else:
                    print(f"no observation data for {target_year}")
                ghsl_gdf.loc[year_mask, feature_cols] = result[feature_cols]

    return ghsl_gdf