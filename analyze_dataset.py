
import sys, os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np; seed = 2
DATASETS = {'small':500, 'medium':1200, 'large':11000, 'Salmonella':500}
EXPRESSION_FEATURES = ['Correlation of expression profile', 'Correlation of ICA components', 'Expression profile distance', 'ICA component distance', 'Target and sRNA occur in same ICA component', 'ICA component score']


#########################
#####   FUNCTIONS   #####
#########################

def read_in_and_process_data(training_file, testing_file, use_expression_features):
    # Read in data
    df_train = pd.read_csv(training_file, sep=',', header=0)
    df_test = pd.read_csv(testing_file, sep=',', header=0)
    df_train.drop(columns=['sRNA', 'Target'], inplace=True)
    df_test.drop(columns=['sRNA', 'Target'], inplace=True)

    # Randomly undersample majority class
    undersample_size = 0
    for d in DATASETS:
        if (d in training_file): undersample_size = DATASETS[d]
    if (undersample_size == 0): sys.stderr.write('\nError - input file has unexpected name\n\n'); sys.exit(1)
    LABEL = 'Evinced Interaction'
    df_plus = df_train[df_train[LABEL] == 1]
    df_negative = df_train[df_train[LABEL] == 0]
    df_negative = df_negative.sample(undersample_size, random_state=seed)
    df_temp = pd.concat([df_plus, df_negative], ignore_index=True)
    
    # Pre-process data
    y_train = df_temp.pop(LABEL).to_numpy()
    X_train = df_temp.to_numpy()
    y_test = df_test.pop(LABEL).to_numpy()
    X_test = df_test.to_numpy()
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    feature_names = df_temp.columns
    
    # Remove 6 expression features
    if (not use_expression_features):
        X_train = X_train[:, :-len(EXPRESSION_FEATURES)]
        X_test = X_test[:, :-len(EXPRESSION_FEATURES)]
    return X_train, X_test, y_train, y_test, feature_names


def compute_feature_relationships(X_train, y_train, feature_names):
    mutual_information = mutual_info_classif(X_train, y_train, random_state=seed)
    f_statistics, p_values = f_classif(X_train, y_train)
    relationships = np.vstack((mutual_information, f_statistics, p_values)).T
    df_relationships = pd.DataFrame(relationships, columns=['Mutual Information', 'F-statistic', 'p-value'], index=list(feature_names))
    sys.stdout.write('\n' + df_relationships.to_string() + '\n\n')


def train_and_evaluate_model(X_train, X_test, y_train, y_test, use_expression_features):
    # Train model
    model = GradientBoostingClassifier(random_state=seed, min_samples_leaf=5)
    model.fit(X_train, y_train)
    y_test_preds = model.predict(X_test)
    y_test_probs = model.predict_proba(X_test)[:,1]

    # Evaluate model
    auc = roc_auc_score(y_test, y_test_preds)
    cm = confusion_matrix(y_test, y_test_preds)
    FPR = cm[0][1] / (cm[0][0] + cm[0][1])
    recall = cm[1][1] / (cm[1][0] + cm[1][1])
    if (use_expression_features): sys.stdout.write('*****   USING EXPRESSION FEATURES   *****\n')
    else: sys.stdout.write('*****   NOT USING EXPRESSION FEATURES   *****\n')
    sys.stdout.write('Sensitivity:         \t' + str(recall) + '\n')
    sys.stdout.write('False positive rate: \t' + str(FPR) + '\n')
    sys.stdout.write('Area under ROC curve:\t' + str(auc) + '\n\n')


####################
#####   MAIN   #####
####################

if __name__ == '__main__':
    DATA_DIR = 'data' + os.sep  # Directory to output files
    if (len(sys.argv) < 3):
        sys.stderr.write('\nUSAGE: python analyze_dataset.py ' + DATA_DIR + 'training_small.csv ' + DATA_DIR + 'testing_small.csv\n\n')
        sys.stderr.write('analyze_dataset.py takes two command line arguments, the path to a csv file with training data and the path to a csv file with testing data. For example, the two command line arguments could be ' + DATA_DIR + 'training_small.csv and ' + DATA_DIR + 'testing_small.csv (for E. coli data), or they could be ' + DATA_DIR + 'training_Salmonella.csv and ' + DATA_DIR + 'testing_Salmonella.csv (for Salmonella data). The program performs the following steps:\n')
        sys.stderr.write('\t1) Read in training and testing data\n')
        sys.stderr.write('\t2) Undersample the majority class\n')
        sys.stderr.write('\t3) Scale the data so that values for each feature have a mean of zero and unit variance\n')
        sys.stderr.write('\t4) For each feature, compute the mutual information as well as the ANOVA F-statistic and corresponding p-value between the feature and the dependent class variable indicating interactions and non-interactions\n') 
        sys.stderr.write('\t5) Train two Gradient Boosting Classifiers: one using 9 features (no expression data) and one using 15 features (including 6 features corresponding to expression data)\n')
        sys.stderr.write('\t6) Report the performance (sensitivity, false positive rate, area under ROC curve) of both classifiers on testing data\n\n')
        sys.exit(1)

    X_train_9, X_test_9, y_train_9, y_test_9, feature_names = read_in_and_process_data(sys.argv[1], sys.argv[2], False)
    X_train_15, X_test_15, y_train_15, y_test_15, feature_names = read_in_and_process_data(sys.argv[1], sys.argv[2], True)
    compute_feature_relationships(X_train_15, y_train_15, feature_names)
    train_and_evaluate_model(X_train_9, X_test_9, y_train_9, y_test_9, False)
    train_and_evaluate_model(X_train_15, X_test_15, y_train_15, y_test_15, True)

