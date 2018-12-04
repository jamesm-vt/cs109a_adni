# CS109A
#
# Authors:
#   Shristi Pandey
#   James May,
#   Zachary Werkhoven
#   Guilherme Braz


import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# print term definitions and codes, and 
def define_terms(data_df, dict_df, table_name=None, columns=None):
    if columns is None:
        columns = data_df.columns
    if table_name is None:
        keys = ["FLDNAME", "TYPE", "TEXT", "CODE"]
    else:
        keys = ["FLDNAME", "TYPE", "TBLNAME", "TEXT", "CODE"]

    term_dicts = []
    for col in columns:

        term_dict = dict.fromkeys(keys)
        if table_name is None:
            loc = (dict_df.FLDNAME == col)
        else:
            loc = (dict_df.FLDNAME == col) & (dict_df.TBLNAME == table_name)
        tmp = dict_df.loc[loc][keys]

        for key in keys:
            if tmp[key].unique().shape[0]:
                term_dict[key] = tmp[key].unique()[0]
            else:
                term_dict[key] = float('nan')

        term_dicts.append(term_dict)

    data_dict = pd.DataFrame.from_dict(term_dicts).reindex(columns=keys)
    return (data_dict)


# function that searches recursively under an input directory for all files of a given extension
def paths_with_ext(directory=None, extension=".csv"):
    if directory is None:
        directory = os.getcwd()

    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                matches.append(os.path.join(root, filename))

    return (matches)


# report num entries, columns, patients, and range of num records and phases covered
def describe_meta_data(df):
    by_patient = df.groupby("RID")
    nRecords = by_patient.apply(len)
    nPatients = nRecords.shape[0]
    if "Phase" in df.columns:
        nPhases = by_patient.apply(lambda x: len(x.Phase.unique()))
        print("Phases:\t {}".format(df.Phase.unique()))
    else:
        nPhases = 0
        print("No phases listed")
    duplicate_in_phase = np.sum(nRecords != nPhases)

    print("Num Entries: %i\nNum Columns: %i" % (df.shape[0], df.shape[1]))
    print("Num Patients: %i\nRecords per Patient: %i-%i" % (nPatients, np.min(nRecords), np.max(nRecords)))
    print("Phases spanned per patient: %i-%i" % (np.min(nPhases), np.max(nPhases)))
    print("Patients w/ Duplicates: %i" % duplicate_in_phase)


# add new patient_data from  by-patient dataframe
def combine_patient_data(pat_df, new_df):
    # intialize empty placeholder columns in patient data
    new_cols = new_df.columns

    # remove duplicate columns
    new_cols = list(set(new_cols) - set(pat_df.columns))
    pat_df = pat_df.reindex(columns=pat_df.columns.tolist() + new_cols)

    # compare list of patient IDs
    old_RIDs = pat_df.index
    new_RIDs = new_df.index
    is_old = np.in1d(new_RIDs, old_RIDs)

    # grab data from existing patient list to insert into pat_df
    insert_mat = new_df.loc[is_old, new_cols].values

    # import pdb;pdb.set_trace()
    pat_df.loc[new_RIDs[is_old], new_cols] = insert_mat

    # append new patient data from RIDs non-existent in pat_df
    pat_df = pat_df.append(new_df.loc[~is_old], sort=True)

    return (pat_df)


# ensure that meta data columns are appended to column output list
def append_meta_cols(df_cols, col_list):
    # ensure col_list is list
    if type(col_list) is not list:
        col_list = list(col_list)

    # define columns to append
    append_cols = ["RID", "VISCODE", "VISCODE2"]
    for col in append_cols:
        if col not in col_list and col in df_cols:
            col_list.append(col)

    # output updated column list
    return (col_list)


def scale_cols(data: pd.DataFrame, cols: list = None, scaler: StandardScaler = None) -> pd.DataFrame:
    """Standardize the columns of data specified by cols.
    
    # Arguments
        data - A DataFrame of data to scale.
        cols - A list of column names to standardize. If None, all dtype 'float64' columns
            will be selected.
        scaler - A previously fit scaler used to transform the data.
            If scaler=None, a new scaler will be created to fit and transform the data.
    
    # Returns
        A copy of data with specified columns standardized.
    """

    data = data.copy()

    if cols is None:
        cols = data.select_dtypes(include=['float64']).columns

    data_to_scale = data[cols]

    # Instantiate and fit a scaler if one wasn't provided
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data_to_scale)

    scaler.fit(data_to_scale)
    data[cols] = scaler.transform(data_to_scale)

    return data


def impute_values_regression(data, cols=None, estimator=None, param_grid={}, random_error=True):
    """Impute missing data using regression and optionally a random error term. If
    cols=None, then all columns having dtype of 'float*' and missing data will be
    considered. For each column with missing data, this function will remove rows
    with null values for that column.
    
    The best model will be used to predict the missing values. The predicted values
    will be added back to the data and will be included in predicting successive columns.
    
    # Arguments
        data: Dataframe of data with missing values
        cols: A list of columns with missing values to impute. If None, columns with
            float dtypes and missing values will be selected.
        classifier: the classifier to use
        param_grid: GridSearchCv params
        random_error: Boolean - whether to add noise from random errors
    
    # Returns
        A copy of data with missing values imputed and a list of scores (one per imputed column).
    """

    # Use KFold object to ensure data is shuffled properly
    folds = 5
    kfold = KFold(folds, shuffle=True)

    data = data.copy()
    scores = []

    # get the columns that have null values
    if cols is None:
        cols = data.columns[data.isnull().any()].tolist()

    for col in cols:

        # Only consider dtype = 'float*' for regression
        if 'float' not in str(data[col].dtype):
            continue

        train = data.copy()

        # Use simple mean imputation on other non-categorical predictors for modeling
        float_data = train.select_dtypes(include=['float64'])
        if col in float_data.columns:
            float_data = float_data.drop(labels=col, axis=1)

        float_data = float_data.fillna(float_data.mean())
        train.update(float_data)

        # Remove rows with missing data in the current col
        data_w_nulls = train[train[col].isnull()]

        # If no missing values, continue to next variable
        if len(data_w_nulls) == 0:
            continue

        # Train on the rest
        data_wo_nulls = train.dropna(subset=[col])

        # Pull out the current col as the reponse variable
        X = data_wo_nulls.drop(labels=col, axis=1)
        y = data_wo_nulls[col]

        gs = GridSearchCV(estimator, param_grid=param_grid, cv=kfold,
                          return_train_score=True, n_jobs=-1)
        gs = gs.fit(X.values, y.values)
        scores.append((gs.best_estimator_, gs.best_score_))

        # Get residuals to add random error
        yhat = gs.predict(X)
        resids = y - yhat

        # Now prepare and predict the missing values
        data_w_nulls = data_w_nulls.drop(labels=col, axis=1)  # drop col we're predicting
        preds = gs.predict(data_w_nulls)

        # Add random errors
        if random_error:
            errors = np.random.choice(resids, len(preds))
            preds = np.add(preds, errors)

        # Update the dataset with predicted values
        data.loc[data[col].isnull(), col] = preds

    # Return imputed data and scores
    return data, cols, scores


def impute_values_classification(data, cols=None, estimator=None, param_grid={},
                                 scoring="accuracy", impute_thresh=0.0):
    """Impute missing data using the given classifier. For each column with
    missing data, this function will remove rows with null values for that columm.
    
    The classifier will be trained on the remaining data and then will be used to
    predict the missing values. The predicted values will be added back into the
    data and will be included in predicting successive columns.
    
    # Arguments
        data: Dataframe of data with missing values
        cols: A list of columns to impute missing values
        estimator: A classifier to use
        param_grid: param grid to be passed to the classifier
        scoring: scoring passed to gridsearchcv
        impute_thresh: float 0:1 - proportion of missing values a categorical column
            is allowed and still be included in the predictors during modeling. If a
            column exceeds this amount it will be removed from modeling until it has been
            modeled itself and predicted. At that time it will be added to the predictors.

    # Returns
        A copy of data with missing values imputed, a list of classifies fearures, scores,
        and columns that could not be imputed due to errors (if any).
    """
    folds = 5
    kfold = StratifiedKFold(folds, shuffle=True)

    data = data.copy()
    scores, errors, classified = [], [], []
    categories = list(cols)

    for col in cols:

        # Remove the category we're modeling/predicting
        categories.pop(categories.index(col))

        train = data.copy()

        # Drop categorical cols from predictors if threshold is exceeded
        train = drop_missing_cols(train, categories, impute_thresh)

        # Identify categorical cols still in the dataset
        subset = list(set(categories) & set(train.columns))

        # Make categorical variables before we fit
        train = pd.get_dummies(train, columns=subset, drop_first=True, dummy_na=True)

        # Use simple mean imputation on other non-categorical predictors for modeling
        float_data = train.select_dtypes(include=['float64'])
        if col in float_data.columns:
            float_data = float_data.drop(labels=col, axis=1)

        float_data = float_data.fillna(float_data.mean())
        train.update(float_data)

        # Remove rows with missing data in the current col
        data_w_nulls = train[train[col].isnull()]

        # If there are no NAs, then make dummies and continue
        if len(data_w_nulls) == 0:
            data = pd.get_dummies(data, columns=[col], drop_first=True, dummy_na=False)
            continue
        
        print(f'Imputing feature {cols.index(col) + 1} of {len(cols)}: {col}')
        
        # Train on the rest
        data_wo_nulls = train.dropna(subset=[col])

        # Pull out the current col as the reponse variable
        X = data_wo_nulls.drop(labels=col, axis=1)
        y = data_wo_nulls[col]
        
        # Make sure we have enough class members to do cv
        if (min(y.value_counts()) < folds) or (len(y.value_counts()) < 2):
            print(f'Not enough class members to impute {col}')
            print(y.value_counts())
            data = data.drop(labels=col, axis=1)
            errors.append(col)
            continue

        gs = GridSearchCV(estimator, param_grid=param_grid, cv=kfold, scoring=scoring,
                          return_train_score=True, n_jobs=-1)
        try:
            gs = gs.fit(X.values, y.values)
        except:
            print(f'Error fitting values for {col}')
            print(y.value_counts())
            data = data.drop(labels=col, axis=1)
            errors.append(col)
            continue
            
        scores.append((gs.best_estimator_, gs.best_score_))

        # Now prepare and predict the missing values
        data_w_nulls = data_w_nulls.drop(labels=col, axis=1)  # drop col we're predicting
        data_w_nulls = data_w_nulls.fillna(data_w_nulls.mean())  # impute NaNs in other predictors
        data_w_nulls = data_w_nulls.fillna(0)  # in case we can't impute the mean
        preds = gs.predict(data_w_nulls.values)

        # Update the dataset with predicted values
        data.loc[data[col].isnull(), col] = preds

        # Make the estimated variable categorical and add back into data
        data = pd.get_dummies(data, columns=[col], drop_first=True, dummy_na=False)
        classified.append(col)

    # Return imputed data and scores
    return data, classified, scores, errors


def drop_missing_cols(data:pd.DataFrame, cols=None, threshold=0.0):
    """Removes columns with missing data beyond the given threshold.

    # Arguments:
        data - pd.DataFrame
        cols - the cols to consider. If 'None', then all will be considered.
        threshold - max allowable % of missing values. If the column has missing
            values at or above this amount, it will be dropped.

    # Returns
        data with columns removed based on cols and threshold
    """

    if cols is None:
        cols = data.columns

    for col in cols:
        missing_pct = len(data[col][data[col].isnull()]) / data.shape[0]
        if missing_pct > threshold:
            data.drop(columns=[col])

    return data


# How much missing data do we have per column?
def calculate_missing_data(data):
    """Calculates the number of and percentage of missing values per column.
    
    data - Dataframe from which to calculate missing data.
    
    Return: A dataframe with count and percentage of missing values per colummn.
    """
    cols_with_nulls = data.columns[data.isnull().any()]

    idx = []
    missing = []
    missing_pct = []
    total_missing = data.isnull().sum().sum()

    for col in cols_with_nulls:
        idx.append(col)
        missing.append(len(data[col][data[col].isnull()]))
        missing_pct.append(100 * len(data[col][data[col]
                                     .isnull()]) / len(data[col]))

    missing_df = pd.DataFrame(np.array([missing, missing_pct]).T,
                              index=idx, columns=['Num Missing', 'Pct. Missing'])
    missing_df = missing_df.sort_values(by=['Pct. Missing'], ascending=False)

    print(f'There are a total of {total_missing} missing values.')
    print(f'Out of {len(data.columns)} features in the dataset, {len(missing_df)} have missing values.')
    print("\nQuartiles of missing data:")
    print(missing_df.quantile([.25, .5, .75, 1]))

    return missing_df
