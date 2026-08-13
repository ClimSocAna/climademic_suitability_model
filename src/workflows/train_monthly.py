import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.monthly_model.monthly_climademic_model import ClimademicMonthlyModel

_columns_list=['t2m_q_0.75', 'd2m_q_0.75', 'tp_q_0.75', 'si10_q_0.75',
               'pop_density', 'Ocean', 'Urban',
               'Cropland', 'Pasture', 'Forest', 'Shrub',
               'Barren', 'Water']

#TODO for all functions define type check
def calculate_quantiles(dataframe, parameters=['t2m', 'd2m', 'tp', 'si10'], quantile_value=0.75):
    quantile_df = dataframe.copy()

    for param in parameters:
        quntile_col_name = param + "_q_" + str(quantile_value)
        col_range = [f"{param}_{i:02d}" for i in range(1, 13)]
        quantile_df[quntile_col_name] = quantile_df[col_range].quantile(quantile_value,axis=1)
    return quantile_df

def load_training_data(training_data_path, specie):
    df_specie = pd.read_csv(training_data_path,low_memory=False)
    df_specie = df_specie.loc[df_specie["VECTOR"] == specie]
    return df_specie.copy()

def prepare_training_dataset(dataframe):
    df = dataframe[_columns_list].copy()
    df = df.dropna(axis=0)

    X = df.copy()
    y = np.ones(len(X))

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=2) #TODO enable the test_size as parameter
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled.tolist(), y_train.tolist(), X_test_scaled.tolist(), y_test.tolist(), scaler

def save_scaler(scaler, path):
    scaler_path = Path(path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)


def get_gmod_file(gmod_dir, year_start, year_end) :

    gmod_file= next(Path(gmod_dir).glob(f"*_{year_start}-{year_end}*"), None)

    if gmod_file is None:
        raise FileNotFoundError(
            f"No GMOD file found for years {year_start}-{year_end} in directory: {gmod_dir}"
        )

    return gmod_file

def prepare_gmod_dataset(gmod_file, specie):
    gmod_df = pd.read_csv(gmod_file)
    gmod_df = gmod_df.loc[gmod_df["Specie"] == specie]
    gmod_df = gmod_df.rename(
        columns={
            "t2m": "t2m_q_0.75",
            "d2m": "d2m_q_0.75",
            "tp": "tp_q_0.75",
            "si10": "si10_q_0.75",
        }
    )
    gmod_df = gmod_df.dropna() 

    features = gmod_df[ _columns_list + ["year"]]
    return  features


def predict_gmod_probabilities(model, scaler, features):
    X_scaled = scaler.transform(features).tolist()

    predicted_labels, _, predicted_values = model.predict(X_scaled, probability=True)

    probabilities = pd.DataFrame(predicted_values, columns=["Prob_1", "Prob_2"])
    probabilities["P_Label"] = predicted_labels

    return np.array(predicted_values)

def update_training_datasets(X_train, y_train, features,
                             predicted_probabilities, scaler, threshold_range: tuple[float, float] = (0.25, 0.5)):

    #define misclassified predictions for given trseshold
    threshold_min, threshold_max = threshold_range

    #check if thresholds are within the allowed probability range
    if not 0 <= threshold_min <= threshold_max <= 1:
        raise ValueError("threshold_range must be between 0 and 1, with min <= max")
    
    mask = ((predicted_probabilities[:, 0] >= threshold_min) &
            (predicted_probabilities[:, 0] < threshold_max))
    misclassified_indices = np.where(mask)[0]

    #add misclassified points to th etraining dataset
    if len(misclassified_indices) > 0:
        misclassified_features = features.iloc[misclassified_indices]
        misclassified_features_scaled = scaler.transform(misclassified_features).tolist()

        X_train.extend(misclassified_features_scaled)
        y_train.extend(np.ones(len(misclassified_features_scaled), dtype=int).tolist())

    return X_train, y_train

def _save_training_dataset(X_train, scaler, path):
    """Save the accumulated training dataset in its original feature scale."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X_train_array = np.asarray(X_train, dtype=float)

    # Convert scaled values back to their original scale.
    X_train_unscaled = scaler.inverse_transform(X_train_array)

    training_df = pd.DataFrame(X_train_unscaled, columns=_columns_list)
    training_df.to_csv(output_path, index=False)

#TODO check years to be correct
def train_climademic_monthly_model(config, specie, gmod_year_start, gmod_year_end, save_intermediate_training=False, intermediate_training_years=[]):
    #TODO check intermediate datastet years against gmod years
    intermediate_training_years = np.array(intermediate_training_years)
    paths = config["paths"]
    vector = f"Aedes {specie}"

    #path for saving trained models
    models_dir = Path(paths["models_directory"])
    
    training_df =load_training_data(paths["processed_data"]["training_dataset"], vector)
    training_df_quantiles = calculate_quantiles(training_df)
    
    #TODO. we split train/test, but never use test ds. maybe ditch splitting completely? 
    # i.e. train on a whole historic dataset?
    X_train, y_train, X_test, y_test, scaler  = prepare_training_dataset(training_df_quantiles)
    save_scaler(scaler, Path(models_dir, specie, "scaler")) #save for future inference

    gamma = config["hyperparameters"][specie]["gamma"]
    nu = config["hyperparameters"][specie]["nu"]

    params = f"-s 2 -t 2 -g {gamma} -n {nu} -b 1 -q"

    model = ClimademicMonthlyModel(params=params)
    model.train(X_train, y_train)
    
    #save base model (1975-2014)
    model_name = f"{specie}/base_ocsvm_{specie}_quantile_75.model'"
    model_path = Path(models_dir, model_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Base model is trained and saved as {model_path}")
    
    #iterate over years of incremental learning
    years = np.arange(gmod_year_start, gmod_year_end + 1)
    gmod_filename = get_gmod_file(paths["processed_data"]["gmod_dataset"], gmod_year_start, gmod_year_end)
    features = prepare_gmod_dataset(gmod_filename, vector)

    print(f"Start incremental learning for years {gmod_year_start} - {gmod_year_end}")
    for year in years:
        #get gmod data for that year
        features_year = features.loc[features["year"] == year].copy()
        features_year = features_year.drop(columns=["year"])
        if (len(features_year) > 0):
        #predict probabilities for given gmod set
            gmod_prob = predict_gmod_probabilities(model, scaler, features_year)

            #add misclassified points to the training dataset
            X_train, y_train = update_training_datasets(X_train, y_train, features_year,
                                gmod_prob, scaler)
        #retrain model
        model.train(X_train, y_train)

        #save  newly trained model
        model_name = f"{specie}/{year}_ocsvm_{specie}_quantile_75.model'"
        model_path = Path(models_dir, model_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        print(f"Model for year {year} is trained and saved as {model_path}")

        if (save_intermediate_training & np.isin(year, intermediate_training_years)):
            interm_fname = f"{year}_{specie}_intermediate_training_ds.csv"
            inerm_fpath = Path(models_dir, specie, "intermediate_training_data", interm_fname)
            _save_training_dataset(X_train, scaler, inerm_fpath)
            print(f"Training data for year {year} is saved as {inerm_fpath}")
    