import pandas as pd

#TODO it is assumed, that gmod raw data is for particukar years,and we do not define 
# the time range whn importing data here

#TODO here, duplictes are not checked in any way. Do we trust GMOD have removed duplicates already?
#  Add a duplicate check /removal?

#TODO incapsulation of these two functions. tehy're quite similar

def extract_mosquito_alert(year_start, year_end, fpath_mosquito_alert, species):
    read_cols = [
        "Indentified by Human",
        "observationResCatObsPheTime",
        "latitude",
        "longitude",
    ]
    mosquito_alert_df = pd.read_csv(fpath_mosquito_alert, usecols=read_cols, low_memory=False)
    #filter only species defined
    mosquito_alert_df = mosquito_alert_df.loc[mosquito_alert_df[ "Indentified by Human"].isin(species)].copy()
    mosquito_alert_df=mosquito_alert_df.dropna()

    time_col = 'observationResCatObsPheTime'

    mosquito_alert_df[time_col] = pd.to_datetime(mosquito_alert_df[time_col], 
                                              format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    mosquito_alert_df = mosquito_alert_df.dropna(subset=[time_col])

    mosquito_alert_df['day'] = mosquito_alert_df[time_col].dt.day
    mosquito_alert_df['month'] = mosquito_alert_df[time_col].dt.month
    mosquito_alert_df['year'] = mosquito_alert_df[time_col].dt.year

    mosquito_alert_df = (
        mosquito_alert_df
        .drop(columns=[time_col])
        .rename(columns={"Indentified by Human": "Specie"})
    )

    years_mask = (mosquito_alert_df["year"] >= year_start) & (mosquito_alert_df["year"] <= year_end)
    mosquito_alert_df = mosquito_alert_df[years_mask]

    return mosquito_alert_df

def extract_inaturalist(year_start, year_end, fpath_inaturalist, species):

    read_cols = [
        "ObsTaxonName",
        "observationResCatObsPheTime",
        "latitude",
        "longitude",
    ]

    inaturalist_df = pd.read_csv(fpath_inaturalist, usecols=read_cols, low_memory=False)

    #filter only species defined
    inaturalist_df = inaturalist_df.loc[inaturalist_df['ObsTaxonName'].isin(species)].copy()
    inaturalist_df=inaturalist_df.dropna()


    time_col = 'observationResCatObsPheTime'
    inaturalist_df[time_col] = pd.to_datetime(inaturalist_df[time_col], 
                                              format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    inaturalist_df = inaturalist_df.dropna(subset=[time_col])

    inaturalist_df['day'] = inaturalist_df[time_col].dt.day
    inaturalist_df['month'] = inaturalist_df[time_col].dt.month
    inaturalist_df['year'] = inaturalist_df[time_col].dt.year

    inaturalist_df = (
        inaturalist_df
        .drop(columns=[time_col])
        .rename(columns={"ObsTaxonName": "Specie"})
    )
    
    years_mask = (inaturalist_df["year"] >= year_start) & (inaturalist_df["year"] <= year_end)
    inaturalist_df = inaturalist_df[years_mask]
    return inaturalist_df

def extract_observations(year_start, year_end, path_inatiralist, path_mosquito_alert):
    #TODO allow specie selection as a parameter    
    species = ["Aedes aegypti", "Aedes albopictus"]
    inaturalist_df = extract_inaturalist(year_start, year_end, path_inatiralist, species)
    mosquito_alert_df = extract_mosquito_alert(year_start, year_end, path_mosquito_alert, species)

    #TODO here, we dont check any crs inconsistency, longitude mismatch (-180 to 180 or 0 to 360), etc. We asume thea are the same
    observations_df = pd.concat([inaturalist_df, mosquito_alert_df], ignore_index=True)
    return observations_df