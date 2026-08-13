# src/data_pipeline/gmod/extract_climate.py

import xarray as xr
import pandas as pd

def extrac_values_month_year(x, y, year, month, var_name, clima_ds):
    year = int(year)

    time = pd.to_datetime(f"{year}-{month}-01")# we kow, in this particular datset,  the validation time is 1st. of the month

    lon = x
    lat = y

    # df.X is -180..180 but NetCDF is 0..360
    lon = lon if lon >= 0 else lon + 360

    # Select the value from the grid point nearest to the (x, y) location.
    value = clima_ds[var_name].sel(
        time=time,
        longitude=lon,
        latitude=lat,
        method="nearest"
    ).values

    return value

def extract_climate_features(climate_fpath, gmod_df):
    climate_df = gmod_df.copy()
    year_start = climate_df["year"].min()
    year_end = climate_df["year"].max()
    clima_variables = ["t2m", "tp", "d2m", "si10"] #variables #TODO allow selection of variables.
    with xr.open_dataset(climate_fpath) as clima_ds:
       # Only consider years contained in the mask dataset.
        clima_ds = clima_ds.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"), expver=1)
        for var_name in clima_variables:
            extracted = climate_df.apply(lambda row: extrac_values_month_year(row.longitude, row.latitude, row.year, row.month, var_name, clima_ds),axis=1)
            monthly_df = pd.DataFrame(extracted.tolist(),columns=[f"{var_name}"],
                                    index=climate_df.index)
            climate_df = pd.concat([climate_df, monthly_df], axis=1)
    climate_df = climate_df.rename(
            columns={
                "longitude": "obs_longitude",
                "latitude":  "obs_latitude",
            })
    return climate_df


