
import pandas as pd
import scipy.stats as ss
import numpy as np

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    # Utility Function to compute the correlation of categorical features
    # Source: https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# Search for categorical variables that are highly correlated
def show_cat_correlation(df, cat_feature_list, threshold=0.80):
    """ Compute the categorical vriables that are highly correlated using Cramer V statistics
"""
    # Utility Function to compute the correlation of categorical features
    # Source: https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix

    print("show_cat_correlation: categorical feature list {}".format(cat_feature_list))
    for i, col_i in enumerate(cat_feature_list):
        for j, col_j in enumerate(cat_feature_list):
            if (i < j):
                cross_tab = pd.crosstab(df[col_i],
                                        df[col_j]).values
                cramer_corr = cramers_corrected_stat(
                    cross_tab)
                if (cramer_corr > threshold):
                    print("Cramer's correlation for {} and {} is {}".format(
                        col_i, col_j, int(cramer_corr * 100)))
