from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from modules.helper_functions import load_dataset
import pandas as pd

OUTPUT_FILE = "./base_benchmarking/outputs/svm_scores_for_c.csv"
PATH_TO_DATASET = "./scaled_lung_cancer_survey.csv"
RANDOM_STATE = 42

C_VALUES = [10**i for i in range(10)]
KERNEL_FUNCTIONS = ["linear", "poly", "rbf", "sigmoid"] 

def create_classifier(c=None, kernel=None):
    if kernel is None:
        classifier = SVC(C=c)
    elif c is None:
        classifier = SVC(kernel=kernel)
    else:
        classifier = SVC(C=c, kernel=kernel)
    return classifier

def get_scores(x_train, y_train, x_test, y_test, c=None, kernel=None):
    classifier = create_classifier(c=c, kernel=kernel)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    
    y_pred = classifier.predict(x_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return score, sensitivity, specificity

def find_and_save_score_for_c_values(c_values, x_train, y_train, x_test, y_test):
    scores_for_c = []
    specificities_for_c = []
    sensitivities_for_c = []
    for c in c_values:
        score, sensitivity, specificity = get_scores(c=c, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        scores_for_c.append(score)
        sensitivities_for_c.append(sensitivity)
        specificities_for_c.append(specificity)
        
    df = pd.DataFrame({
        "c": c_values, 
        "score": scores_for_c, 
        "sensitivity": sensitivities_for_c, 
        "specificity": specificities_for_c
    })        
    df.to_csv(OUTPUT_FILE, index=False)

def find_and_save_score_for_kernel_values(kernel_values, x_train, y_train, x_test, y_test, c=100000):
    scores = []
    specificities = []
    sensitivities = []
    for kernel_value in kernel_values:
        score, sensitivity, specificity = get_scores(kernel=kernel_value, c=c, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        scores.append(score)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
    df = pd.DataFrame({
        "kernel": kernel_values, 
        "score": scores, 
        "sensitivity": sensitivities, 
        "specificity": specificities
    })        
    df.to_csv("./base_benchmarking/outputs/svm_scores_for_kernel.csv", index=False)


def main():
    x, y = load_dataset(path_to_dataset=PATH_TO_DATASET)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    
    find_and_save_score_for_c_values(c_values=C_VALUES, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    find_and_save_score_for_kernel_values(kernel_values=KERNEL_FUNCTIONS, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    

    
   
    
if __name__ == "__main__":
    main()
        
    
        

    
        
    