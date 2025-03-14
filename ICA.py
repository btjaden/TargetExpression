
import sys, os
import pandas as pd
from sklearn.decomposition import FastICA


#########################
#####   FUNCTIONS   #####
#########################

def perform_ICA(filename, DATA_DIR, ITERATIONS):
        COMPONENTS = 500
        X = pd.read_csv(filename, sep=',', header=0, index_col=0)
        ica = FastICA(n_components=COMPONENTS, max_iter=ITERATIONS, random_state=0)
        columns = ['C_' + str(v) for v in range(1, COMPONENTS+1)]
        S = pd.DataFrame(ica.fit_transform(X), index=X.index, columns=columns)
        A = pd.DataFrame(ica.mixing_, index=X.columns, columns=columns)
        A.index.name = 'SRA'
        S.to_csv(DATA_DIR + 'S.csv', sep=',')
        sys.stdout.write('Source matrix with dimensions ' + str(S.shape) + ' written to ' + DATA_DIR + 'S.csv\n')
        A.to_csv(DATA_DIR + 'A.csv', sep=',')
        sys.stdout.write('Mixing matrix with dimensions ' + str(A.shape) + ' written to ' + DATA_DIR + 'A.csv\n')


####################
#####   MAIN   #####
####################

if __name__ == '__main__':
        DATA_DIR = 'data' + os.sep  # Directory to output files
        if (len(sys.argv) < 2):
                sys.stderr.write('\nUSAGE: python ICA.py ' + DATA_DIR + 'TPM.csv\n\n')
                sys.stderr.write('ICA.py takes one command line argument, the path to a csv file with normalized log TPM values. The rows of the file correspond to genes and the columns correspond to experiments. Each entry is the normalized log TPM value of a gene in an experiment. The program performs ICA (Independent Component Analysis), outputting the source matrix to the file ' + DATA_DIR + 'S.csv and the mixing matrix to the file ' + DATA_DIR + 'A.csv. The program may take hours to execute, depending on the number of iterations (to speed it up, use a smaller number of iterations).\n\n')
                sys.exit(1)

        ITERATIONS = 1000000
        if (not os.path.isdir(DATA_DIR)): os.makedirs(DATA_DIR)
        perform_ICA(sys.argv[1], DATA_DIR, ITERATIONS)

