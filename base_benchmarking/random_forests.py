from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from modules.helper_functions import load_dataset
from datetime import datetime
import pandas as pd
import logging

N_ESTIMATORS = [1, 5, 10, 50, 100, 200, 300]
RANDOM_STATE = 42
PATH_TO_DATASET = "./scaled_lung_cancer_survey.csv"
OUTPUT_FILE = "./base_benchmarking/outputs/random_forest_scores.csv"

def score_for_n_estimators(n_estimators, x_train, y_train, x_test, y_test):
    logging.info(f"Training and predicting for n_estimators = {n_estimators}")
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    random_forest.fit(x_train, y_train)
    score = random_forest.score(x_test, y_test)
    logging.info(f"Accuracy Score: {score}")
    y_pred = random_forest.predict(x_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return score, sensitivity, specificity

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Random Forest Classification")
    logging.info("Starting at:")
    start_time = datetime.now()
    logging.info(start_time)
    x, y = load_dataset(path_to_dataset=PATH_TO_DATASET)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    
    scores_for_n_estimators = []
    sensitivities = []
    specificities = []
    
    for n_estimators in N_ESTIMATORS:
        score, sensitivity, specificity = score_for_n_estimators(n_estimators=n_estimators, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        scores_for_n_estimators.append(score)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    endtime = datetime.now()
    total_time = str(endtime - start_time)
    
    df = pd.DataFrame({
        "n_estimators": N_ESTIMATORS, 
        "score": scores_for_n_estimators,
        "sensitivity": sensitivities,
        "specificity": specificities
    })
    df.to_csv(OUTPUT_FILE)
    
    logging.info("Finished classification for all n_estimators")
    logging.info("Finished at:")
    logging.info(endtime)
    logging.info(f"Took {total_time}")

if __name__ == "__main__":
    main()
