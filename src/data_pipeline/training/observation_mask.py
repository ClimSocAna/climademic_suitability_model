# data_pipeline/training/observation_mask.py
from shapely.geometry import box, MultiPolygon
import pandas as pd
import geopandas as gpd
import numpy as np

from src.data_pipeline.common import GEOD

#In the compendium, data is represented as polygons with different side lengths. 
# Here, we define the side length based on the parameter values.
def _get_polygon_offset_km(value):

    mapping = {
        "-999": 5,
        "Less than 10km": 10,
        "Less than 25km": 25,
        "Less than 100km": 100,
        "2": 110,
    }

    return mapping.get(str(value), 110)

#Converting kilometers to degrees and getting the coordinate window 
# for a particular polygon centered at x, y.
def _make_geodetic_box(x, y, offset_km):
    
    half_distance_m = offset_km / 2 * 1e3

    north = GEOD.fwd(x, y, 0, half_distance_m)
    south = GEOD.fwd(x, y, 180, half_distance_m)
    east  = GEOD.fwd(x, y, 90, half_distance_m)
    west  = GEOD.fwd(x, y, 270, half_distance_m)

    minx = west[0]
    maxx = east[0]
    miny = south[1]
    maxy = north[1]

    geometry = box(minx, miny, maxx, maxy)

    # Detect crossing of the 0-degree line and split polygons into two.
    # # Otherwise, it would create a polygon spanning the entire latitude.
    if minx > maxx:
        geometry = MultiPolygon([
            box(minx, miny, 180, maxy),
            box(-180, miny, maxx, maxy)
        ])

    return geometry

#extract polygon geometry (in degrees) for a particular observation/row
def _make_geodetic_mask(row):
    x = row.geometry.x
    y = row.geometry.y

    offset_km = _get_polygon_offset_km(str(row["POLYGON_ADMIN"]))

    return _make_geodetic_box(x, y, offset_km)

#load compendium dataset from .csv file. 
def load_compendium_raw(compendium_fpath_csv, year_start, year_end):
    compendium_data = pd.read_csv(compendium_fpath_csv) 
    compendium_data = compendium_data.loc[~compendium_data["YEAR"].isna()] #filter out those without time information
    compendium_data = compendium_data.loc[compendium_data["YEAR"]!='2006-2008'] #filter out strange timing
    compendium_data["YEAR"] = compendium_data["YEAR"].astype(np.int64)
    compendium_data = compendium_data.loc[compendium_data["YEAR"] > year_start - 1] #filter the years
    compendium_data = compendium_data.loc[compendium_data["YEAR"] < year_end + 1]
    compendium_gdf = gpd.GeoDataFrame(compendium_data, geometry=gpd.points_from_xy(compendium_data.X, compendium_data.Y),
                                crs='EPSG:4326') # EPSG:4326 is WGS84 Latitude/Longitude
    return compendium_gdf

#extrat polygom mask for observations
def create_geodetic_mask(compendium_gdf):
    gdf = compendium_gdf.copy()
    gdf["mask_geometry"] = gdf.apply(_make_geodetic_mask, axis=1)
    gdf = gdf.set_geometry("mask_geometry")
    return gdf

# Read the raw .csv file and create a polygon mask.
def build_observation_mask(observation_path, year_start, year_end):
    observation_gdf_raw = load_compendium_raw(observation_path, year_start, year_end)
    observation_gdf_mask = create_geodetic_mask(observation_gdf_raw)
    return observation_gdf_mask