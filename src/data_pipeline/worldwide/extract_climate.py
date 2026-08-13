# src/data_pipeline/worldwide/extract_climate.py
import xarray as xr
import numpy as np

#get the resolution of the dataset
def _get_resolution(dataset):
    lon_res = abs(np.diff(dataset.longitude.values).mean())
    lat_res = abs(np.diff(dataset.latitude.values).mean())

    return lon_res, lat_res

#check if the latitudial and longitudial resolution of the dataset is the same,
#and if it corrsponds to the resolutiion parameter
def _check_resolution(dataset, resolution):
    result = False
    lon_res, lat_res = _get_resolution(dataset)
    if (lon_res != lat_res):
        raise ValueError( f"Expected latitudial resolution ({lat_res}) and \
            longitudial resolution ({lon_res}) does not match")
    elif (lon_res != resolution):
        raise ValueError( f"Expected resolution ({resolution}, got {lon_res}")
    else:
        result = True
    return result

# that how it was done originally - we just rely on the (lon; lat) combinations from th eera5 dataset
# TODO make this function more flexible, in case (lon; lat) coordinates does not match the selected resolution, etc.
def extract_climate_features_to_df(clima_ds, features):

    #detect wich of desired features are in the dataset
    features_present = [feature for feature in features if feature in clima_ds.data_vars]
    
    df = (clima_ds[features_present].to_dataframe()
          .reset_index().sort_values(["longitude", "latitude", "time"]))
    
    result = df.pivot_table(
    index=["longitude", "latitude"],
    columns="time",
    values=features_present
    )

    result = result.sort_index(axis=1)

    result.columns = [
        f"{var}_{time_i + 1}"
        for var, time_i in zip(
            result.columns.get_level_values(0),
            result.T.groupby(level=0).cumcount()
        )
    ]
    
    result = result.reset_index()

    return result

# TODO: Add support for different resolutions 
def extract_climate_features(climate_fpath, year, features = ['t2m', 'd2m', 'si10', 'tp'], resolution=0.25):
    with xr.open_dataset(climate_fpath) as clima_ds: #open era5 file
        if (_check_resolution(clima_ds, resolution)):# chek if era5 resolution corrresponds to desired resolution

            clima_ds = clima_ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))#, expver=1)

            clima_features_df = extract_climate_features_to_df(clima_ds, features)
            clima_features_df["year"] = year
            # Detect 0-360 convention
            if clima_features_df["longitude"].max() > 180:
                clima_features_df["longitude"] = ((clima_features_df["longitude"] + 180) % 360) - 180

            return clima_features_df