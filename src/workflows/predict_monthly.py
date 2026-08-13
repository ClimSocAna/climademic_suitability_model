import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.monthly_model.monthly_climademic_model import ClimademicMonthlyModel

_climate_params = ['t2m', 'd2m', 'tp', 'si10']
_lan_use_pop_cols = ['pop_density', 'Ocean', 'Urban',
               'Cropland', 'Pasture', 'Forest', 'Shrub',
               'Barren', 'Water']
#TODO add possibility to predict for each month separately
def predict_monthly(config, year_analysis, specie, model_year):

    paths = config["paths"]
    dataset_path = Path(paths["processed_data"]["global_dataset"], f"{year_analysis}_global_clima_lulc_pop_res_0.25_deg.csv")
    inference_dataset = pd.read_csv(dataset_path)

    print(dataset_path)
    #TODO check the validity of loaded dataset: no NaN values, no inf valuess, etc

    model = ClimademicMonthlyModel()
    model_name = f"{specie}/{model_year}_ocsvm_{specie}_quantile_75.model'"
    model_path = Path(paths["models_directory"], model_name)
    model.load(model_path)
    
    scaler_path = Path(paths["models_directory"], f"{specie}/scaler")
    scaler = joblib.load(scaler_path)

    monthly_fpath_dir = Path(paths["predictions"], f"{specie}", f"{year_analysis}")
    monthly_fpath_dir.mkdir(parents=True, exist_ok=True)
    for month in np.arange(1, 13):
        monthly_result = inference_dataset[["longitude", "latitude", "geometry"]].copy()

        monthly_cols  = [f"{param}_{month}" for param in _climate_params] + _lan_use_pop_cols
        monthly_inference_dataset = inference_dataset[monthly_cols]

        monthly_inference_dataset = monthly_inference_dataset.rename(
                    columns={
                                    f"t2m_{month}": "t2m_q_0.75",
                                    f"d2m_{month}": "d2m_q_0.75",
                                    f"tp_{month}": "tp_q_0.75",
                                    f"si10_{month}": "si10_q_0.75",})

        X_test_scaled = scaler.transform(monthly_inference_dataset)
        predicted_labels, _, predicted_values = model.predict(X_test_scaled.tolist(), probability=True)

        predicted_values = np.asarray(predicted_values)
        monthly_result["Prob_1"] = predicted_values[:, 0]
        monthly_result["Prob_2"] = predicted_values[:, 1]
        monthly_result["P_Label"] = predicted_labels

        monthly_result_filename=str(year_analysis)+'_monthly_mean_'+str(month)+'_ocsvm_'+specie+'_predictions_'+str(model_year)+'_mod.csv'
        monthly_fpath = Path(monthly_fpath_dir, monthly_result_filename)

        monthly_result.to_csv(monthly_fpath, sep=',', index=False, decimal='.')
        print(f"Prediction for {month:02d}.{year_analysis} is saved as {monthly_fpath}")




