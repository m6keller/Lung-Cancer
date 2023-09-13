from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from modules.helper_functions import load_dataset
import pandas as pd

OUTPUT_FILE = "./base_benchmarking/outputs/knn_scores.csv"
PATH_TO_DATASET = "./scaled_lung_cancer_survey.csv"
RANDOM_STATE = 42

K_VALUES = [2,3,4,5,10,50,100,150]

def get_scores_for_k(k, x_train, y_train, x_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    
    y_pred = classifier.predict(x_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return score, sensitivity, specificity

def main():
    scores_for_k = []
    specificities_for_k = []
    sensitivities_for_k = []
    x, y = load_dataset(path_to_dataset=PATH_TO_DATASET)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    
    for k in K_VALUES:
        score, sensitivity, specificity = get_scores_for_k(k=k, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        scores_for_k.append(score)
        sensitivities_for_k.append(sensitivity)
        specificities_for_k.append(specificity)
    
    df = pd.DataFrame({"k": K_VALUES, "score": scores_for_k, "sensitivity": sensitivities_for_k, "specificity": specificities_for_k})        
    df.to_csv(OUTPUT_FILE, index=False)
    
if __name__ == "__main__":
    main()
        
    
        

    
        
    