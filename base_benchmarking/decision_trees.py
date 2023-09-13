from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from modules.helper_functions import load_dataset
from sklearn.metrics import confusion_matrix
from datetime import datetime
import pandas as pd
import logging

MAX_DEPTH_VALUES = [None, 10, 20, 30, 40] 
RANDOM_STATE = 42
PATH_TO_DATASET = "./scaled_lung_cancer_survey.csv"
OUTPUT_FILE = "./base_benchmarking/outputs/decision_tree_scores.csv"

def score_for_max_depth(max_depth, x_train, y_train, x_test, y_test):
    logging.info(f"Training and predicting for max_depth = {max_depth}")
    decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    decision_tree.fit(x_train, y_train)
    score = decision_tree.score(x_test, y_test)
    logging.info(f"Accuracy Score: {score}")
    y_pred = decision_tree.predict(x_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return score, sensitivity, specificity

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Decision Tree Classification")
    logging.info("Starting at:")
    start_time = datetime.now()
    logging.info(start_time)
    x, y = load_dataset(path_to_dataset=PATH_TO_DATASET)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    
    scores_for_max_depth = []
    sensitivities = []
    specificities = []
    
    for max_depth in MAX_DEPTH_VALUES:
        score, sensitivity, specificity = score_for_max_depth(max_depth=max_depth, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        scores_for_max_depth.append(score)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    endtime = datetime.now()
    total_time = str(endtime - start_time)
    
    df = pd.DataFrame({
        "max_depth": MAX_DEPTH_VALUES,
        "score": scores_for_max_depth,
        "sensitivity": sensitivities,
        "specificity": specificities
    })
    df.to_csv(OUTPUT_FILE)
    
    logging.info("Finished classification for all max_depth values")
    logging.info("Finished at:")
    logging.info(endtime)
    logging.info(f"Took {total_time}")

if __name__ == "__main__":
    main()
