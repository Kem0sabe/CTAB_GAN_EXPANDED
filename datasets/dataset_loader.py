

from pathlib import Path
import pandas as pd

def load_demo(dataset_name = "adult"):


    dataset_file = get_dataset_path(dataset_name)


    file_path = Path(__file__).resolve()
    folder_path = file_path.parent
    csv_path = folder_path / dataset_file
    
    if not csv_path.exists():
        return csv_path
        raise FileNotFoundError("Demo file not found")
        
    return pd.read_csv(csv_path)


def display_demo():
    datasets = ["adult", "car", "king"]
    return datasets

def get_dataset_path(dataset_name):
    if dataset_name == "adult": return "Adult.csv"
    if dataset_name == "car": return "car.csv"
    if dataset_name == "king": return "king.csv"
    raise ValueError("Invalid dataset name")
    