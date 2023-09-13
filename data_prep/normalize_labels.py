import pandas as pd
from sklearn.preprocessing import MinMaxScaler  

INTEGER_ENCODED_FEATURES = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE ",
    "ALLERGY ",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]

def get_scaled_age(df):
    scaler = MinMaxScaler()
    age_reshaped = df["AGE"].values.reshape(-1, 1)
    scaled_age = scaler.fit_transform(age_reshaped)
    scaled_age = scaled_age.flatten()
    return scaled_age

def main():
    df = pd.read_csv("./survey_lung_cancer.csv")
    
    scaled_df = pd.DataFrame()
    
    scaled_df["AGE"] = get_scaled_age(df)
    scaled_df["GENDER"] = df["GENDER"].apply(lambda x: 1 if x == "M" else 0)
    scaled_df["LUNG_CANCER"] = df["LUNG_CANCER"].apply(lambda x: 1 if x == "YES" else 0)
    for feature in INTEGER_ENCODED_FEATURES:
        scaler = MinMaxScaler()
        feature_reshaped = df[feature].values.reshape(-1, 1)
        scaled_feature = scaler.fit_transform(feature_reshaped)
        scaled_feature = scaled_feature.flatten()
        
        scaled_df[feature] = scaled_feature
        
    scaled_df = scaled_df.to_csv("./scaled_lung_cancer_survey.csv", index=False)
    
if __name__ == "__main__":
    main()
        
    