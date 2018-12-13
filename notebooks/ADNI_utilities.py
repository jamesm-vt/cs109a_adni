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

from glob import glob
from joblib import dump, load
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def define_terms(data_df, dict_df, table_name=None, columns=None):
    """Print human readable definitions for features of a raw data file
    
    # Arguments
        data_df - raw data dataframe with features to define
        dict_df - ADNI dictionary dataframe with feature definitions
        table_name - (optional) ADNI code for the source table of data_df
        columns - (optional) subset of features in the raw data file to define
    
    # Returns
        A dictionary dataframe where each row is a defined term.
    """

    # define all columns if no subset is provided 
    if columns is None:
        columns = data_df.columns

    # remove TBLNAME from the query if no table name is provided
    if table_name is None:
        keys = ["FLDNAME", "TYPE", "TEXT", "CODE"]
    else:
        keys = ["FLDNAME", "TYPE", "TBLNAME", "TEXT", "CODE"]

    # iterate of features and extract definitions and term codes 
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


def paths_with_ext(directory=None, extension=".csv"):
    """Search recursively under input directory for all files of a given extension
    
    # Arguments
        directory - Parent directory to search under.
        extension - target file extension to filter results by
    
    # Returns
        A list of complete file paths for all files of the matching extension under the parent directory.
    """

    # use current working directory if none is provided 
    if directory is None:
        directory = os.getcwd()

    # initialize file paths placeholder and iterate over directories
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        # iterate over files in current directory
        for filename in filenames:
            if filename.endswith(extension):
                # append file paths with matching extension
                matches.append(os.path.join(root, filename))

    return (matches)



def describe_meta_data(df):
    """
    Print summary information for a dataframe including:
        - number of phases covered by the data
        - number of observations in the data
        - number of unique patient IDs in the data
        - number of records per patient ID
        - number of patients with more than one observation
    """
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
    """
    Ensure that the mandatory meta data features are included in a list of features

    # Arguments
        df_cols - all column names in the target dataframe
        col_list - current list of columns to extract

    # Returns
        An updated column list with patient ID (RID) and visit code (VISCODE)
    """

    # ensure col_list is list
    if type(col_list) is not list:
        col_list = list(col_list)

    # define columns to append
    append_cols = ["RID", "VISCODE", "VISCODE2"]
    for col in append_cols:
        # append meta data columns if not already in the list
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
    scores, imputed = [], []

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

        print(f'Imputing feature {cols.index(col) + 1} of {len(cols)}: {col}')
            
        # Train on the rest
        data_wo_nulls = train.dropna(subset=[col])
        
        if data_wo_nulls.shape[0] < folds:
            col_mean = data[col].fillna(data[col].mean())
            data.update(col_mean)
            continue

        # Pull out the current col as the reponse variable
        X = data_wo_nulls.drop(labels=col, axis=1)
        y = data_wo_nulls[col]

        gs = GridSearchCV(estimator, param_grid=param_grid, cv=kfold,
                          return_train_score=True, n_jobs=-1)
        gs = gs.fit(X.values, y.values)
        
        imputed.append(col)
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
    return data, imputed, scores


def impute_values_classification(data, cols=None, estimator=None, param_grid={},
                                 scoring="accuracy", impute_thresh=0.25):
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
    strat_kfold = StratifiedKFold(folds, shuffle=True)
    kfold = KFold(folds, shuffle=True)

    data = data.copy()
    scores, errors, imputed = [], [], []
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

        gs = GridSearchCV(estimator, param_grid=param_grid, cv=strat_kfold, scoring=scoring,
                          return_train_score=True, n_jobs=-1)
        try:
            gs = gs.fit(X.values, y.values)
        except:
            try:
                # If not enough of each class to stratify, try simple KFold
                gs = GridSearchCV(estimator, param_grid=param_grid,
                              cv=kfold, scoring=scoring,
                              return_train_score=True, n_jobs=-1)
                gs = gs.fit(X.values, y.values)
            except:
                print(f'Error fitting values for {col}')
                print(y.value_counts())
                data = data.drop(labels=col, axis=1)
                errors.append(col)
                continue
        
        imputed.append(col)
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

    # Return imputed data and scores
    return data, imputed, scores, errors



def impute_errors(data, cols=None):
    """Imputes classification features. This function does not use
    KFold validation and is only intended for those features that
    cannot be imputed with KFold due to the distribution of the
    classes, i.e. some class counts are less than 'k' and therefore
    an error occus during the fitting.
    """
    data = data.copy()
    scores, errors, imputed = [], [], []
    categories = list(cols)
    impute_thresh = .25

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

        dtc = DecisionTreeClassifier(max_features="auto", class_weight="balanced", min_samples_leaf=3, max_depth=None)
        ada = AdaBoostClassifier(base_estimator=dtc, n_estimators=100)
         
        try:
            ada = ada.fit(X.values, y.values)
        except:
            print(f'Error fitting values for {col}')
            print(y.value_counts())
            data = data.drop(labels=col, axis=1)
            continue
        
        imputed.append(col)
        scores.append(ada.score(X, y))

        # Now prepare and predict the missing values
        data_w_nulls = data_w_nulls.drop(labels=col, axis=1)  # drop col we're predicting
        data_w_nulls = data_w_nulls.fillna(data_w_nulls.mean())  # impute NaNs in other predictors
        data_w_nulls = data_w_nulls.fillna(0)  # in case we can't impute the mean
        preds = ada.predict(data_w_nulls.values)

        # Update the dataset with predicted values
        data.loc[data[col].isnull(), col] = preds

        # Make the estimated variable categorical and add back into data
        data = pd.get_dummies(data, columns=[col], drop_first=True, dummy_na=False)

    # Return imputed data and scores
    return data, imputed, scores, errors

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


def get_feature_names(data_path, design_mat, resp_vars):
    file_w_path = data_path + design_mat
    df = pd.read_csv(file_w_path, index_col='RID')
    
    if 'baseline' in design_mat:
        dummies = ['PTETHCAT', 'PTGENDER', 'PTRACCAT', 'PTMARRY', 'FSVERSION', 'APOE4']
        df = df.drop(columns=['CDRSB','mPACCtrailsB', 'mPACCdigit'])
        df = df.drop(columns=['DX_bl'], axis=1)
        df = pd.get_dummies(df, columns=dummies)
        return list(df.columns)

    if 'modeled' in design_mat:
        df = reverse_one_hot(resp_vars, df)
        
    df = df.drop(resp_vars, axis=1).select_dtypes(['number'])
    df = df.drop(columns=['CDRSB','mPACCtrailsB', 'mPACCdigit'])
    return list(df.columns)


def reverse_one_hot(resp_set, imp_df):
    """This function recreates response variables from one-hot-encoded datasets
    
    # Arguments
        resp_set: contains a list of response variables' prefixes to be rebuilt
        imp_df: contains the imputed dataset with the raw data
    """
    
    col_set = imp_df.columns.tolist()
    is_resp = [any([resp in col for resp in resp_set]) for col in col_set]
    resp_cols = imp_df.columns[is_resp]
    
    # iterate of response prefixes
    resp_data = []
    for resp in resp_set:
    
        # get subset of columns corresponding to currest prefix
        is_subset = [resp in col for col in col_set]
        subset_cols = imp_df.columns[is_subset]
    
        # convert train data to column index of true value
        tmp = np.argmax(imp_df[subset_cols].values,1)+1
        tmp[~imp_df[subset_cols].values.any(1)] = 0
        resp_data.append(tmp)
    
    # drop one-hot response vars and add new features
    imp_df = imp_df.drop(columns=resp_cols, axis=1)
    for col, data in zip(resp_set, resp_data):
        imp_df[col] = pd.Series(np.array(data), index=imp_df.index)
    
    return imp_df

# Load and return models from disk based on glob pattern
def load_models(glob_ptrn):
    model_list = glob(glob_ptrn)
    return [load(model) for model in model_list]


# Return train/test data from the given desing matix
def get_train_test(filename, rm_vars, resp_variable):
    df = pd.read_csv(filename, index_col='RID')
    df = reverse_one_hot(rm_vars, df)
    df_train, df_test = train_test_split(df, test_size=.2, shuffle=True, random_state=42)

    y_train = df_train[resp_variable]
    X_train = df_train.drop(rm_vars, axis=1).select_dtypes(['number'])

    y_test = df_test[resp_variable]
    X_test = df_test.drop(rm_vars, axis=1).select_dtypes(['number'])
    
    return X_train, y_train, X_test, y_test

def get_ADNI_baseline_data(bl_file_path):
    df = pd.read_csv(bl_file_path, index_col='RID')
    dummies = ['PTETHCAT', 'PTGENDER', 'PTRACCAT', 'PTMARRY', 'FSVERSION', 'APOE4']
    df = pd.get_dummies(df, columns=dummies)
    X = df.drop(['DX_bl'], axis=1)
    X = X.drop(columns=['CDRSB','mPACCtrailsB', 'mPACCdigit'])
    y = df['DX_bl']
    return X, y

def get_ADNI_baseline(estimators):
    bl_files = glob('../data/Imputed/baseline*.csv')
    for bl_file_path in bl_files:
        # Read in the file
        bl_file = bl_file_path.split('/')[-1]
        estimators[bl_file] = []
        df = pd.read_csv(bl_file_path, index_col='RID')
        
        dummies = ['PTETHCAT', 'PTGENDER', 'PTRACCAT', 'PTMARRY', 'FSVERSION', 'APOE4']
        df = pd.get_dummies(df, columns=dummies)
        X = df.drop(['DX_bl'], axis=1)
        X = X.drop(columns=['CDRSB','mPACCtrailsB', 'mPACCdigit'])
        y = df['DX_bl']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True, random_state=42)

        # Bagging
        dt_clf = DecisionTreeClassifier()
        bag_clf = BaggingClassifier(dt_clf, n_jobs=-1, n_estimators=100)
        bag_clf_params = {'base_estimator__max_depth':[2, 3, 5, 10, 20],
                  'base_estimator__min_samples_leaf': [1, 2, 4, 6, 20]}

        bag_gs = GridSearchCV(bag_clf, bag_clf_params, iid=False, return_train_score=False, n_jobs=-1, cv=5)
        bag_gs.fit(X_train, y_train)
        estimators[bl_file].append((bag_gs.best_score_, bag_gs.best_estimator_))

        # Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf_clf_params = {'max_depth':[2, 3, 5, 10, 20],
              'min_samples_leaf': [1, 2, 4, 6, 20]}  

        rf_gs = GridSearchCV(rf_clf, rf_clf_params, iid=False, return_train_score=False, n_jobs=-1, cv=5)
        rf_gs.fit(X_train, y_train)
        estimators[bl_file].append((rf_gs.best_score_, rf_gs.best_estimator_))
    return estimators