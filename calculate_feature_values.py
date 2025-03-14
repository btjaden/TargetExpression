
import sys, os
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import lilliefors


#########################
#####   FUNCTIONS   #####
#########################

# GET LIST OF sRNAs AND GENES
def read_in_genes(FILENAME, ORGANISM):
        sRNAs, genes = [], []
        name1_to_2, name2_to_1, sRNA_to_1 = {}, {}, {}
        with open(FILENAME, 'r') as f:
                line = f.readline().strip()
                while (line != ''):
                        if (not line.startswith('#')):  # Ignore comments
                                parse_line = line.split('\t')
                                gene_type = parse_line[2]
                                misc_info = parse_line[-1].split()
                                name1 = misc_info[1][1:-2]
                                name2 = misc_info[-1][1:-2]
                                if (name2 == '?'): name2 = name1
                                if (gene_type == 'ncRNA'):
                                        if (ORGANISM == 'Ecoli'):
                                                sRNAs.append(name2)
                                                sRNA_to_1[name2] = name1
                                        else:
                                                sRNAs.append(name1)
                                                sRNA_to_1[name1] = name2
                                elif (gene_type == 'CDS'): genes.append(name1)
                                name1_to_2[name1] = name2
                                name2_to_1[name2] = name1
                        line = f.readline().strip()
        return sRNAs, genes, name1_to_2, name2_to_1, sRNA_to_1


# GET FEATURES FOR INTERACTIONS
def get_features(FILENAME, name2_to_1, name1_to_2, ORGANISM):
        features = pd.read_csv(FILENAME, sep=',', header=0, usecols=['sRNA', 'Target', 'Length of target', 'Overlap upstream gene on same strand', 'Nucleotide distance to upstream gene on same strand', 'Overlap upstream gene on either strand', 'Nucleotide distance to upstream gene on either strand', 'Target conservation', 'Number of homologs', 'Seed region of hybridization', 'Hybridization energy'])
        index = 0 if (ORGANISM == 'Ecoli') else -1
        pair_names = []
        for i in range(features.shape[0]):
                sRNA = features.at[i, 'sRNA'].replace("3ETS-leuZ", "3'ETS-leuZ")
                mRNA = features.at[i, 'Target'].replace('(','').replace(')','').split()[index]
                name = name2_to_1[mRNA] if (ORGANISM == 'Ecoli') else mRNA
                pair_names.append(sRNA + '::' + name)
        features['pair_names'] = pair_names
        features.set_index(keys='pair_names', drop=True, inplace=True)
        features.drop(columns=['sRNA', 'Target'], inplace=True)
        return features


# READ IN INTERACTIONS FILE
def read_interactions(sRNAs, name1_to_2, name2_to_1, size, ORGANISM):
        FILENAME = GENE_DIR + 'interactions_' + size + '.csv'
        if (size == 'small'): return read_interactions_small(FILENAME, sRNAs, name1_to_2)
        elif (size == 'medium'): return read_interactions_large(FILENAME, sRNAs, name2_to_1, ORGANISM)
        elif (size == 'large'): return read_interactions_large(FILENAME, sRNAs, name2_to_1, ORGANISM)
        elif (size == 'Salmonella'): return read_interactions_large(FILENAME, sRNAs, name2_to_1, ORGANISM)
        else:
                sys.stderr.write('\nError - could not parse interactions file ' + FILENAME + '\n\n')
                sys.exit(1)

        
# READ IN SMALL LIST OF HIGH QUALITY SRNA:TARGET INTERACTIONS AND NON-INTERACTIONS
def read_interactions_small(FILENAME, sRNAs, name1_to_2):
        interactions = {}
        target = pd.read_csv(FILENAME, sep=',', header=0)
        for i in range(target.shape[0]):
                sRNA = target.at[i, 'sRNA']
                sRNA = sRNA[0].lower() + sRNA[1:]
                mRNA = target.at[i, 'Target']
                if (sRNA in sRNAs):
                        if (sRNA not in interactions): interactions[sRNA] = {}
                        interactions[sRNA][mRNA] = True

        # Generate non-interactions for all genes that are not interactions
        non_interactions = {}
        names = list(name1_to_2.keys())
        for sRNA in interactions.keys():
                if (sRNA not in non_interactions): non_interactions[sRNA] = {}
                for name1 in names:
                        if (name1 not in interactions[sRNA]): non_interactions[sRNA][name1] = True
        return interactions, non_interactions


# READ IN LARGE LIST OF LOW QUALITY SRNA:TARGET INTERACTIONS AND NON-INTERACTIONS
def read_interactions_large(FILENAME, sRNA_to_1, name2_to_1, ORGANISM):
        interactions, non_interactions = {}, {}
        target = pd.read_csv(FILENAME, sep=',', header=0)
        index = 0 if (ORGANISM == 'Ecoli') else -1
        for i in range(target.shape[0]):
                sRNA = target.at[i, 'sRNA'].replace("3ETS-leuZ", "3'ETS-leuZ")
                mRNA = target.at[i, 'Target'].replace('(','').replace(')','').split()[index]
                isInteraction = target.at[i, 'Evinced Interaction']
                if (sRNA in sRNA_to_1):  # Ignore non-sRNAs
                        inters = interactions if (isInteraction == 1) else non_interactions
                        if (sRNA not in inters): inters[sRNA] = {}
                        name = name2_to_1[mRNA] if (ORGANISM == 'Ecoli') else mRNA
                        inters[sRNA][name] = True
        return interactions, non_interactions


# DETERMINE CORRELATIONS OF INTERACTIONS (OR NON-INTERACTIONS)
def get_correlation(MATRIX_FILENAME, sRNA_to_1, name1_to_2, ORGANISM):
        # Compute correlations for all genes
        X = pd.read_csv(MATRIX_FILENAME, sep=',', header=0, index_col=0)
        X['index'] = range(len(X))
        gene_index = X['index'].to_dict()
        X.drop('index', axis=1, inplace=True)
        X = X.to_numpy()
        # Ignore error when computing correlation of constant rows
        with np.errstate(divide='ignore', invalid='ignore'):
                correlation = np.corrcoef(X)  # Correlation
                correlation = np.nan_to_num(correlation)  # Replace NaN with 0.0

        correlations = {}
        for sRNA, name1 in sRNA_to_1.items():
                if (sRNA not in correlations): correlations[sRNA] = {}
                for name in name1_to_2:
                        name_temp = name1 if (ORGANISM == 'Ecoli') else sRNA
                        correlations[sRNA][name] = correlation[gene_index[name_temp]][gene_index[name]]
        return correlations


# DETERMINE DISTANCES OF INTERACTIONS (OR NON-INTERACTIONS)
def get_distances(MATRIX_FILENAME, sRNA_to_1, name1_to_2, ORGANISM):
        # Read in data
        X = pd.read_csv(MATRIX_FILENAME, sep=',', header=0, index_col=0)
        X['index'] = range(len(X))
        gene_index = X['index'].to_dict()
        X.drop('index', axis=1, inplace=True)
        X = X.to_numpy()

        distances = {}
        for sRNA, name1 in sRNA_to_1.items():
                if (sRNA not in distances): distances[sRNA] = {}
                name_temp = name1 if (ORGANISM == 'Ecoli') else sRNA
                distance_values = np.linalg.norm(X-X[gene_index[name_temp]], axis=1)
                for name in name1_to_2:
                        distances[sRNA][name] = distance_values[gene_index[name]]
        return distances


# CALCULATE MEMBERSHIP OF EACH COMPONENT
def analyze_components(sRNA_to_1, name1_to_2, inters, non_inters):
        THRESHOLD = 0.15
        X = pd.read_csv(COMPONENT_FILE, sep=',', header=0, index_col=0)
        components = [{}]
        genes_components = {}
        for i in range(1, X.shape[1]+1):
                components.append({})
                col = 'C_' + str(i)  # Name of component
                component = X[col].abs().sort_values()
                test_stat, pvalue = lilliefors(component)
                while (test_stat > THRESHOLD):
                        name1 = component.index[-1]
                        components[i][name1] = True
                        if (name1 not in genes_components): genes_components[name1] = []
                        genes_components[name1].append(i)
                        component.drop(index=name1, inplace=True)
                        test_stat, pvalue = lilliefors(component)

        # For each sRNA, get set of genes (neighbors) that share a component with the sRNA
        neighbors = {}
        for sRNA, name1 in sRNA_to_1.items():
                neighbors[sRNA] = {}
                if (name1 in genes_components):
                        for c in genes_components[name1]:  # For each of the sRNA's components
                                for g in components[c].keys():  # For each neighbor
                                        if (g != name1):  # Don't add self
                                                if (g not in neighbors[sRNA]):
                                                        neighbors[sRNA][g] = 0.0
                                                neighbors[sRNA][g] = max(neighbors[sRNA][g], abs(X.at[g, 'C_' + str(c)]))
        return neighbors


# MERGE EXPRESSION RELATED FEATURES
def combine_all_features(features, correlations_expression, correlations_ICA, distances_expression, distances_ICA, neighbors):
        sRNAs, mRNAs, corrs_exp, corrs_ICA, dist_exp, dist_ICA, is_neighbor, neighbor_score = [], [], [], [], [], [], [], []
        indices = list(features.index.values)
        for index in indices:
                sRNA, mRNA = index.split('::')
                sRNAs.append(sRNA)
                mRNAs.append(mRNA)
                if (sRNA in correlations_expression) and (mRNA in correlations_expression[sRNA]): corrs_exp.append(correlations_expression[sRNA][mRNA])
                else: corrs_exp.append(0.0)
                if (sRNA in correlations_ICA) and (mRNA in correlations_ICA[sRNA]): corrs_ICA.append(correlations_ICA[sRNA][mRNA])
                else: corrs_ICA.append(0.0)
                if (sRNA in distances_expression) and (mRNA in distances_expression[sRNA]): dist_exp.append(distances_expression[sRNA][mRNA])
                else: dist_exp.append(999999.9)
                if (sRNA in distances_ICA) and (mRNA in distances_ICA[sRNA]): dist_ICA.append(distances_ICA[sRNA][mRNA])
                else: dist_ICA.append(999999.9)
                if (sRNA in neighbors):
                        if (mRNA in neighbors[sRNA]):
                                is_neighbor.append(1)
                                neighbor_score.append(neighbors[sRNA][mRNA])
                        else:
                                is_neighbor.append(0)
                                neighbor_score.append(0.0)
                else:
                        is_neighbor.append(-1)
                        neighbor_score.append(0.0)

        features['Correlation of expression profile'] = corrs_exp
        features['Correlation of ICA components'] = corrs_ICA
        features['Expression profile distance'] = dist_exp
        features['ICA component distance'] = dist_ICA
        features['Target and sRNA occur in same ICA component'] = is_neighbor
        features['ICA component score'] = neighbor_score
        return features


def output_feature_file(all_features, name1_to_2, inters, non_inters, OUT_FILENAME):
        INT_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 13}
        rows, cols = 0, len(list(all_features.columns.values)) + 1
        with open(OUT_FILENAME, 'w') as out_file:
                out_file.write('sRNA' + ',' + 'Target' + ',' + ','.join(list(all_features.columns.values)) + ',' + 'Evinced Interaction' + '\n')
                for sRNA,mRNAs in inters.items():  # Interactions
                        for mRNA in mRNAs.keys():
                                index_name1 = sRNA + '::' + mRNA
                                #index_name2 = sRNA + '::' + name1_to_2[mRNA]
                                if (index_name1 in all_features.index):
                                        #out_file.write(index_name2 + ',' + ','.join([str(f) for f in list(all_features.loc[index_name1])]) + ',' + '1' + '\n')
                                        out_file.write(sRNA + ',' + name1_to_2[mRNA] + ',')
                                        current_row = list(all_features.loc[index_name1])
                                        for i in range(len(current_row)):
                                                if (i in INT_INDICES): out_file.write(str(int(current_row[i])) + ',')
                                                else: out_file.write(str(current_row[i]) + ',')
                                        out_file.write('1' + '\n')
                                        rows += 1
                for sRNA,mRNAs in non_inters.items():  # NON_interactions
                        for mRNA in mRNAs.keys():
                                index_name1 = sRNA + '::' + mRNA
                                #index_name2 = sRNA + '::' + name1_to_2[mRNA]
                                if (index_name1 in all_features.index):
                                        #out_file.write(index_name2 + ',' + ','.join([str(f) for f in list(all_features.loc[index_name1])]) + ',' + '0' + '\n')
                                        out_file.write(sRNA + ',' + name1_to_2[mRNA] + ',')
                                        current_row = list(all_features.loc[index_name1])
                                        for i in range(len(current_row)):
                                                if (i in INT_INDICES): out_file.write(str(int(current_row[i])) + ',')
                                                else: out_file.write(str(current_row[i]) + ',')
                                        out_file.write('0' + '\n')
                                        rows +=1
        sys.stdout.write('Created file ' + OUT_FILENAME + ' with ' + str(rows) + ' candidate interactions\n')


####################
#####   MAIN   #####
####################

if __name__ == '__main__':
        DATA_DIR = 'data' + os.sep  # Directory with data files
        GENE_DIR = 'genome' + os.sep  # Directory with gene files
        ORGANISMS = {'Ecoli':['small', 'medium', 'large'], 'Salmonella':['Salmonella']}
        if (len(sys.argv) < 2):
                sys.stderr.write('\nUSAGE: python calculate_feature_values.py Ecoli\n\n')
                sys.stderr.write('calculate_feature_values.py takes one command line argument, either *Ecoli* or *Salmonella*. For the specified organism, the program uses the source matrix from ICA to calculate feature values for each candidate sRNA and target interaction. Feature values for each candidate interaction are output to a csv file in the directory ' + DATA_DIR + '\n\n')
                sys.exit(1)
        if (sys.argv[1] not in ORGANISMS):
                sys.stderr.write('\nError - command line argument must be either "Ecoli" or "Salmonella"\n\n')
                sys.exit(1)

        ORGANISM = sys.argv[1]
        ANNOTATION_FILE = GENE_DIR + ORGANISM + '.gtf'
        INTERACTIONS_FILE = GENE_DIR + 'interactions_large.csv'
        if (ORGANISM == 'Salmonella'): INTERACTIONS_FILE = GENE_DIR + 'interactions_Salmonella.csv'
        EXPRESSION_FILE = DATA_DIR + 'TPM_' + ORGANISM + '.csv'
        COMPONENT_FILE = DATA_DIR + 'S_' + ORGANISM + '.csv'

        sRNAs, genes, name1_to_2, name2_to_1, sRNA_to_1 = read_in_genes(ANNOTATION_FILE, ORGANISM)
        features = get_features(INTERACTIONS_FILE, name2_to_1, name1_to_2, ORGANISM)
        correlations_expression = get_correlation(EXPRESSION_FILE, sRNA_to_1, name1_to_2, ORGANISM)
        correlations_ICA = get_correlation(COMPONENT_FILE, sRNA_to_1, name1_to_2, ORGANISM)
        distances_expression = get_distances(EXPRESSION_FILE, sRNA_to_1, name1_to_2, ORGANISM)
        distances_ICA = get_distances(COMPONENT_FILE, sRNA_to_1, name1_to_2, ORGANISM)
        for size in ORGANISMS[sys.argv[1]]:
                interactions, non_interactions = read_interactions(sRNAs, name1_to_2, name2_to_1, size, ORGANISM)
                neighbors = analyze_components(sRNA_to_1, name1_to_2, interactions, non_interactions)
                all_features = combine_all_features(features, correlations_expression, correlations_ICA, distances_expression, distances_ICA, neighbors)
                output_feature_file(all_features, name1_to_2, interactions, non_interactions, DATA_DIR + size + '.csv')

