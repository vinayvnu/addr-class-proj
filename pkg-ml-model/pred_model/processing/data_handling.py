import os
import pandas as pd
import joblib
from pred_model.config import config

# load dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    return pd.read_csv(filepath)

# serialisation
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f'Model has been saved, {config.MODEL_NAME}')

# deserialisation
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    return joblib.load(save_path)

