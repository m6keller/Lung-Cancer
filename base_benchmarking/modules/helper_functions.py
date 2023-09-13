import pandas as pd

def load_dataset(path_to_dataset):
    df = pd.read_csv(path_to_dataset)
    x = df.drop(["LUNG_CANCER"], axis=1)
    y = df["LUNG_CANCER"]
    return x, y