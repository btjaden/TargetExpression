# <img src="https://cs.wellesley.edu/~btjaden/TargetExpression/RNA.png" width=100> <img src="https://cs.wellesley.edu/~btjaden/TargetExpression/title.png" width=600> <img src="https://cs.wellesley.edu/~btjaden/TargetExpression/RNA.png" width=100>


==========

## Identifying Targets of sRNA Regulation

### `analyze_dataset.py`

`analyze_dataset.py` trains a machine learning model on a provided dataset (the folder `data` contains several example datasets for _E. coli_ and for _Salmonella_) and reports the model's performance at identifying bacterial sRNA regulatory targets

EXAMPLE USAGE: &nbsp;&nbsp;&nbsp;&nbsp;`python analyze_dataset.py data/training_Salmonella.csv data/testing_Salmonella.csv`<BR>

As input, `analyze_dataset.py` requires a `.csv` file of training data (for training the machine learning model) and a `.csv` file of testing data (for evaluating the model's performance in identifying targets of sRNA regulation). `analyze_dataset.py` performs the following steps:

1. Read in training data and testing data
2. Undersample the majority class
3. Scale the data so that values for each feature have a mean of zero and unit variance
4. For each feature, compute the mutual information as well as the ANOVA F-statistic and corresponding p-value between the feature and the dependent class variable indicating interactions and non-interactions
5. Train two Gradient Boosting Classifiers: one using 9 features (no expression data) and one using 15 features (including 6 features corresponding to expression data)
6. Report the performance (sensitivity, false positive rate, area under ROC curve) of both classifiers on the testing data


==========

## Creating a Custom Dataset

### `ICA.py` and `calculate_feature_values.py`

In order to execute the abovementioned program, `analyze_dataset.py`, to identify sRNA regulatory targets, the user needs a dataset containing information about sRNA and target interactions. Example datasets are provided in the `data` folder. To create your own custom dataset, you must start with a `.csv` file of normalized log TPM values obtained from a set of genome-wide expression experiments, e.g., a set of RNA-seq experiments (example files are provided in the `data` folder). The rows of the `.csv` file correspond to genes and the columns correspond to experiments. Each entry is the normalized log TPM value of a gene in an experiment. `ICA.py` performs ICA (Independent Component Analysis), outputting the source matrix to the file `data/S.csv` and the mixing matrix to the file `data/A.csv`.

EXAMPLE USAGE: &nbsp;&nbsp;&nbsp;&nbsp;`python ICA.py data/TPM.csv`<BR>

Once a source matrix is computed using ICA with the `ICA.py` program, feature values for candidate sRNA and target interactions can be calculated from the source matrix using the program `calculate_feature_values.py`.

EXAMPLE USAGE: &nbsp;&nbsp;&nbsp;&nbsp;`python calculate_feature_values.py Salmonella`<BR>

`calculate_feature_values.py` will output a dataset as a `.csv` file that can be used as input to `analyze_dataset.py` to identify regulatory interactions between sRNAs and targets

