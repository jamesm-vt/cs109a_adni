title: 
notebook: 
section: 
subsection: 

## Contents
{:.no_toc}
*  
{: toc}


This Notebook's key objective is to compare different models applied on datasets produced with different data imputation strategies.



```python
#importing libraries
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import scale

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# import custom dependencies
import ADNI_utilities
```


## Reading and preparing the data

We start by reading all the datasets from different imputation strategies and choosing the response variable to be used. 

The rationale for hyper parameters choices can be found on the "Parameter finetunning" section.



```python
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
```




```python
#Initializing the data strucuture that will hold the test scores from different models
models = ['Logistic', 'kNN', 'Decision Tree', 'Bagging', 'Boosting', 'Random Forest']
datasets = ['mean - 30pct', 'mean - 50pct', 'model - 30pct', 'model - 50pct', 'model - 100pct']
result_container_multi = pd.DataFrame(index=datasets, columns=models)
result_container_binary = pd.DataFrame(index=datasets, columns=models)

filenames = pd.DataFrame(index=datasets, columns=['filename'])
filenames.at['mean - 30pct','filename'] = 'data_mean_upto_30pct_missing.csv'
filenames.at['mean - 50pct','filename'] = 'data_mean_upto_50pct_missing.csv'
filenames.at['model - 30pct','filename'] = 'data_modeled_upto_30pct_missing.csv'
filenames.at['model - 50pct','filename'] = 'data_modeled_upto_50pct_missing.csv'
filenames.at['model - 100pct','filename'] = 'data_modeled_upto_100pct_missing.csv'
```




```python
#Running the different models and storing results

import warnings
warnings.filterwarnings('ignore')

#Setting up common parameters
data_path = '../data/Imputed/'
resp_variable = 'DX_FINAL'
resp_vars = ['DXCOMB', 'DX_CHANGE', 'DX_FINAL', 'DX_BASE', 'DX_bl']
testsize = 0.2
rs = 42 # set random state so results are repeatable
run_binary = 1 #Set this variable to 1 if models with binary response variable are to be run

#Logistic classifier
for dataset in datasets:
    #Loading data
    filename = data_path + filenames.at[dataset, 'filename']
    print(f'LogisticRegressionCV: fitting {filename}')
    
    df = pd.read_csv(filename)
    if 'model' in dataset:
        df = reverse_one_hot(resp_vars, df)
    df_train, df_test = train_test_split(df, test_size=testsize, shuffle=True, random_state=rs)

    y_train_multi = df_train[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_train_bin = df_train[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])

    y_test_multi = df_test[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_test_bin = df_test[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
    
    #Running the model and storing results
    model_multi = LogisticRegressionCV(multi_class="ovr", cv=4, penalty='l2',
                                       max_iter=1000).fit(X_train, y_train_multi)
    score = model_multi.score(X_test, y_test_multi)
    result_container_multi.loc[dataset,'Logistic'] = score
    
    if (run_binary == 1 and not 'model' in dataset):
        model_binary = LogisticRegressionCV(multi_class="ovr", cv=4, penalty='l2').fit(X_train, y_train_bin)
        result_container_binary.loc[dataset, 'Logistic'] = model_binary.score(X_test, y_test_bin)
```


    LogisticRegressionCV: fitting ../data/Imputed/data_mean_upto_30pct_missing.csv
    LogisticRegressionCV: fitting ../data/Imputed/data_mean_upto_50pct_missing.csv
    LogisticRegressionCV: fitting ../data/Imputed/data_modeled_upto_30pct_missing.csv
    LogisticRegressionCV: fitting ../data/Imputed/data_modeled_upto_50pct_missing.csv
    LogisticRegressionCV: fitting ../data/Imputed/data_modeled_upto_100pct_missing.csv
    



```python
#kNN
for dataset in datasets:
    #Loading data
    filename = data_path + filenames.at[dataset, 'filename']
    print(f'KNeighborsClassifier: fitting {filename}')
    
    df = pd.read_csv(filename)
    if 'model' in dataset:
        df = reverse_one_hot(resp_vars, df)
    df_train, df_test = train_test_split(df, test_size=testsize, shuffle=True, random_state=rs)

    y_train_multi = df_train[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_train_bin = df_train[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])

    y_test_multi = df_test[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_test_bin = df_test[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
    
    #Running the model and storing results
    model_multi = KNeighborsClassifier(n_neighbors=52).fit(X_train, y_train_multi)
    score = model_multi.score(X_test, y_test_multi)
    result_container_multi.loc[dataset,'kNN'] = score
    
    if (run_binary == 1 and not 'model' in dataset):
        model_binary = KNeighborsClassifier(n_neighbors=36).fit(X_train, y_train_bin)
        result_container_binary.loc[dataset, 'kNN'] = model_binary.score(X_test, y_test_bin)
```


    KNeighborsClassifier: fitting ../data/Imputed/data_mean_upto_30pct_missing.csv
    KNeighborsClassifier: fitting ../data/Imputed/data_mean_upto_50pct_missing.csv
    KNeighborsClassifier: fitting ../data/Imputed/data_modeled_upto_30pct_missing.csv
    KNeighborsClassifier: fitting ../data/Imputed/data_modeled_upto_50pct_missing.csv
    KNeighborsClassifier: fitting ../data/Imputed/data_modeled_upto_100pct_missing.csv
    



```python
#Decision Tree
for dataset in datasets:
    #Loading data
    filename = data_path + filenames.at[dataset, 'filename']
    print(f'DecisionTreeClassifier: fitting {filename}')
    
    df = pd.read_csv(filename)
    if 'model' in dataset:
        df = reverse_one_hot(resp_vars, df)
    df_train, df_test = train_test_split(df, test_size=testsize, shuffle=True, random_state=rs)

    y_train_multi = df_train[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_train_bin = df_train[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])

    y_test_multi = df_test[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_test_bin = df_test[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
    
    #Running the model and storing results
    model_multi = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train_multi)
    score = model_multi.score(X_test, y_test_multi)
    result_container_multi.loc[dataset,'Decision Tree'] = score
    
    if (run_binary == 1 and not 'model' in dataset):
        model_binary = DecisionTreeClassifier(max_depth=6).fit(X_train, y_train_bin)
        result_container_binary.loc[dataset, 'Decision Tree'] = model_binary.score(X_test, y_test_bin)
```


    DecisionTreeClassifier: fitting ../data/Imputed/data_mean_upto_30pct_missing.csv
    DecisionTreeClassifier: fitting ../data/Imputed/data_mean_upto_50pct_missing.csv
    DecisionTreeClassifier: fitting ../data/Imputed/data_modeled_upto_30pct_missing.csv
    DecisionTreeClassifier: fitting ../data/Imputed/data_modeled_upto_50pct_missing.csv
    DecisionTreeClassifier: fitting ../data/Imputed/data_modeled_upto_100pct_missing.csv
    



```python
#Bagging
for dataset in datasets:
    #Loading data
    filename = data_path + filenames.at[dataset, 'filename']
    print(f'BaggingClassifier: fitting {filename}')
    
    df = pd.read_csv(filename)
    if 'model' in dataset:
        df = reverse_one_hot(resp_vars, df)
    df_train, df_test = train_test_split(df, test_size=testsize, shuffle=True, random_state=rs)

    y_train_multi = df_train[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_train_bin = df_train[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])

    y_test_multi = df_test[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_test_bin = df_test[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
    
    #Running the model and storing results
    model_multi = BaggingClassifier(DecisionTreeClassifier(max_depth=10),
                                    n_estimators=100).fit(X_train, y_train_multi)
    score = model_multi.score(X_test, y_test_multi)
    result_container_multi.loc[dataset,'Bagging'] = score
    
    if (run_binary == 1 and not 'model' in dataset):
        model_binary = BaggingClassifier(DecisionTreeClassifier(max_depth=4),
                                         n_estimators=10).fit(X_train, y_train_bin)
        result_container_binary.loc[dataset, 'Bagging'] = model_binary.score(X_test, y_test_bin)
```


    BaggingClassifier: fitting ../data/Imputed/data_mean_upto_30pct_missing.csv
    BaggingClassifier: fitting ../data/Imputed/data_mean_upto_50pct_missing.csv
    BaggingClassifier: fitting ../data/Imputed/data_modeled_upto_30pct_missing.csv
    BaggingClassifier: fitting ../data/Imputed/data_modeled_upto_50pct_missing.csv
    BaggingClassifier: fitting ../data/Imputed/data_modeled_upto_100pct_missing.csv
    



```python
#Boosting
for dataset in datasets:
    #Loading data
    filename = data_path + filenames.at[dataset, 'filename']
    print(f'AdaBoostClassifier: fitting {filename}')
    
    df = pd.read_csv(filename)
    if 'model' in dataset:
        df = reverse_one_hot(resp_vars, df)
    df_train, df_test = train_test_split(df, test_size=testsize, shuffle=True, random_state=rs)

    y_train_multi = df_train[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_train_bin = df_train[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])

    y_test_multi = df_test[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_test_bin = df_test[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
    
    #Running the model and storing results
    model_multi = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=767,
                                     learning_rate=0.5).fit(X_train, y_train_multi)
    
    score = model_multi.score(X_test, y_test_multi)
    result_container_multi.loc[dataset,'Boosting'] = score
    
    if (run_binary == 1 and not 'model' in dataset):
        model_binary = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=749 ,
                                          learning_rate=0.5).fit(X_train, y_train_bin)
        result_container_binary.loc[dataset, 'Boosting'] = model_binary.score(X_test, y_test_bin)
```


    AdaBoostClassifier: fitting ../data/Imputed/data_mean_upto_30pct_missing.csv
    AdaBoostClassifier: fitting ../data/Imputed/data_mean_upto_50pct_missing.csv
    AdaBoostClassifier: fitting ../data/Imputed/data_modeled_upto_30pct_missing.csv
    AdaBoostClassifier: fitting ../data/Imputed/data_modeled_upto_50pct_missing.csv
    AdaBoostClassifier: fitting ../data/Imputed/data_modeled_upto_100pct_missing.csv
    



```python
#Random Forest
for dataset in datasets:
    #Loading data
    filename = data_path + filenames.at[dataset, 'filename']
    print(f'RandomForestClassifier: fitting {filename}')
    
    df = pd.read_csv(filename)
    if 'model' in dataset:
        df = reverse_one_hot(resp_vars, df)
    df_train, df_test = train_test_split(df, test_size=testsize, shuffle=True, random_state=rs)

    y_train_multi = df_train[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_train_bin = df_train[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])

    y_test_multi = df_test[resp_variable]
    if (run_binary == 1 and not 'model' in dataset):
        y_test_bin = df_test[resp_variable].apply(lambda x: 1 if x == 3 else 0)
    X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
    
    #Running the model and storing results
    model_multi = RandomForestClassifier(n_estimators=100, max_depth=4).fit(X_train, y_train_multi)
    score = model_multi.score(X_test, y_test_multi)
    result_container_multi.loc[dataset,'Random Forest'] = score
    
    if (run_binary == 1 and not 'model' in dataset):
        model_binary = RandomForestClassifier(n_estimators=50, max_depth=8).fit(X_train, y_train_bin)
        result_container_binary.loc[dataset, 'Random Forest'] = model_binary.score(X_test, y_test_bin)
```


    RandomForestClassifier: fitting ../data/Imputed/data_mean_upto_30pct_missing.csv
    RandomForestClassifier: fitting ../data/Imputed/data_mean_upto_50pct_missing.csv
    RandomForestClassifier: fitting ../data/Imputed/data_modeled_upto_30pct_missing.csv
    RandomForestClassifier: fitting ../data/Imputed/data_modeled_upto_50pct_missing.csv
    RandomForestClassifier: fitting ../data/Imputed/data_modeled_upto_100pct_missing.csv
    

## Perfomance comparison

This section will present the performance differences across the models tested by our group.



```python
#Plotting the results as a heat map

fig, ax = plt.subplots(1,1, figsize=(12,6))

ax = sns.heatmap(result_container_multi.astype('float64'), annot=True, linewidths=0.1, fmt='.3f', cmap='coolwarm')
ax.set_title('Test scores of different models on imputed datasets', fontsize=16);
plt.show()
```



![png](model_comparison_files/model_comparison_12_0.png)


## Dimensionality reduction

We have also explored the possibility of using PCA to reduce the dimensionality of our dataset and reach additional conclusions regarding our dataset and regarding AD diagnosis prediction.



```python
#PCA

#Reading the "strictest" dataset to apply PCA techniques
df = pd.read_csv(data_path + 'data_mean_upto_30pct_missing.csv')

df_train, df_test = train_test_split(df, test_size=testsize, shuffle=True, random_state=rs)
y_train_multi = df_train[resp_variable]
X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])
X_train = scale(X_train)

#y_test_multi = df_test[resp_variable]
#X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
#X_test = scale(X_test)

#Running the PCA routine
pca = PCA()
pca.fit(X_train)

#Plotting the results
fig, ax = plt.subplots(1,1, figsize=(12,6))

plt.plot(np.cumsum(pca.explained_variance_ratio_), label='PCA cumulative explained variance')
plt.plot(np.linspace(0,1,135), 'g--', label='Benchmark "uniform" explained variance')
plt.legend(fontsize=12)
ax.set_xlabel('Number of features - ordered by the most to the least explanatory', fontsize=14)
ax.set_ylabel('Cumulative explained variance', fontsize=14)
ax.set_title('PCA Analysis results', fontsize=14);
```



![png](model_comparison_files/model_comparison_14_0.png)


As one can see in the plot above, there is not a strong explanatory power concentration on any principal component. Out of 133 components on the select dataset, more than 80 are needed to achieve 90% variance explanation. The top 20 components explain less than 50% of the variance. It is interesting to notice that, after the 20th and before the 100th component, the blue line is almost "parallel" to the green benchmark line - showing that there is very limited explanatory power concentration.

For this reason, added to the fact that similar results hold for all imputation strategies, we have decided not to use PCA as a tool to further improve our models.

## Parameter tuning

In this section, one will find the reasoning behind the choice of hyperparameters used in each fitted model. We will use the mean-imputed database with 30-pct threshold as the basis for hyperparameter fine-tunning.



```python
df = pd.read_csv('../data/Imputed/data_mean_upto_30pct_missing.csv')

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=rs)
resp_vars = ['DXCOMB', 'DX_CHANGE', 'DX_FINAL', 'DX_BASE', 'DX_bl']

y_train_multi = df_train['DX_FINAL']
y_train_bin = df_train['DX_FINAL'].apply(lambda x: 1 if x == 3 else 0)
X_train = df_train.drop(resp_vars, axis=1).select_dtypes(['number'])

y_test_multi = df_test['DX_FINAL']
y_test_bin = df_test['DX_FINAL'].apply(lambda x: 1 if x == 3 else 0)
X_test = df_test.drop(resp_vars, axis=1).select_dtypes(['number'])
```


### Logistic classifier



```python
logistic = LogisticRegressionCV(multi_class="ovr", cv=4, penalty='l2').fit(X_train, y_train_multi)
logistic.score(X_test, y_test_multi)
```





    0.7312348668280871





```python
#Repeating the process for the binary response variable
logistic = LogisticRegressionCV(multi_class="ovr", cv=4, penalty='l2').fit(X_train, y_train_bin)
logistic.score(X_test, y_test_bin)
```





    0.8692493946731235



There is no hyper parameter to choose, but training/fit conditions were set.

### kNN



```python
k_set = range(1,70)

scores = []
scores_test = []
for i in k_set:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train_multi)
    scores.append(np.mean(cross_val_score(knn, X_train, y_train_multi, cv=5)))
    scores_test.append(knn.score(X_test, y_test_multi))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(k_set, scores, 'g*-', label = 'Train CV score')
ax.plot(k_set, scores_test, 'b*-', label = 'Test score')
ax.set_xlabel('n_neighbors', size=14)
ax.set_ylabel('Score', size=14)
ax.legend(fontsize=14)
ax.set_title('KNN Train/Test scores as a function of n_neighbors', size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_23_0.png)


In the kNN models for the multi category response variable, `k=52` appears to be the optimal parameter.



```python
#Repeating the process for the binary response variable
k_set = range(1,100)

scores = []
scores_test = []
for i in k_set:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train_bin)
    scores.append(np.mean(cross_val_score(knn, X_train, y_train_bin, cv=5)))
    scores_test.append(knn.score(X_test, y_test_bin))

fig, ax = plt.subplots(1,1, figsize=(12, 6))
    
ax.plot(k_set, scores, 'g*-', label = 'Train CV score')
ax.plot(k_set, scores_test, 'b*-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('n_neighbors', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title('KNN scores (Binary) as a function of n_neighbors', size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_25_0.png)


In the kNN models for the binary response variable, `k=36` appears to be the optimal parameter.

### Simple decision tree



```python
depths = range(1,25)

scores = []
scores_test = []
for i in depths:
    dec = DecisionTreeClassifier(max_depth=i)
    dec.fit(X_train, y_train_multi)
    scores.append(dec.score(X_train, y_train_multi))
    scores_test.append(dec.score(X_test, y_test_multi))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(depths, scores, 'g*-', label = 'Train score')
ax.plot(depths, scores_test, 'b*-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('max_depth', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title("DecisionTree Train/Test Scores - function of max_depth", size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_28_0.png)


Among the Decision Tree models for the multi category response variable, `max_depth=3` appears to be the optimal parameter.



```python
#Repeating the process for the binary decision variable
depths = range(1,25)

scores = []
scores_test = []
for i in depths:
    dec = DecisionTreeClassifier(max_depth=i)
    dec.fit(X_train, y_train_bin)
    scores.append(dec.score(X_train, y_train_bin))
    scores_test.append(dec.score(X_test, y_test_bin))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(depths, scores, 'g*-', label = 'Train score')
ax.plot(depths, scores_test, 'b*-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('max_depth', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title("DecisionTree Scores (Binary) - function of max_depth", size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_30_0.png)


Among the Decision Tree models for the binary response variable, `max_depth=6` appears to be the optimal parameter.

### Bagging decision tree



```python
num_trees = range(1,100)

scores = []
scores_test = []
for i in num_trees:
    bag = BaggingClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=i)
    bag.fit(X_train, y_train_multi)
    scores.append(bag.score(X_train, y_train_multi))
    scores_test.append(bag.score(X_test, y_test_multi))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(num_trees, scores, 'g*-', label = 'Train score')
ax.plot(num_trees, scores_test, 'b*-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('number of estimators',size=14)
ax.set_ylabel('Score', size=14)
ax.set_title('Bagging Train/Test scores - function of num estimators', size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_33_0.png)




```python
scores_test.index(max(scores_test))
```





    9



Among the Bagging Decision Trees for the multi category response variable, `n_estimators=13` seems to be the best hyper parameter choice.



```python
#Repeating the process for the binary response variable
num_trees = range(1,100)

scores = []
scores_test = []
for i in num_trees:
    bag = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=i)
    bag.fit(X_train, y_train_bin)
    scores.append(bag.score(X_train, y_train_bin))
    scores_test.append(bag.score(X_test, y_test_bin))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(num_trees, scores, 'g*-', label = 'Train score')
ax.plot(num_trees, scores_test, 'b*-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('number of estimators', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title('Bagging scores (Binary) - function of num estimators', size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_36_0.png)


Among the Bagging Decision Trees for the binary response variable, `n_estimators=10` seems to be the best hyper parameter choice.

### Boosting decision tree



```python
boo = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=800, learning_rate=0.5)
boo.fit(X_train, y_train_multi)

scores = list(boo.staged_score(X_train, y_train_multi))
scores_test = list(boo.staged_score(X_test, y_test_multi))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(range(1,801), scores, 'g-', label = 'Train score')
ax.plot(range(1,801), scores_test, 'b-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('# of iterations', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title('Boosting Train/Test Scores - function of iterations', size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_39_0.png)




```python
scores_test.index(max(scores_test))
```





    767



The boosting procedure with 8 iterations seems to be our best choice of hyper parameters to predict the multi category response variable.



```python
#Repeating the process for the binary response variable
boo = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=800, learning_rate=0.5)
boo.fit(X_train, y_train_bin)

scores = list(boo.staged_score(X_train, y_train_bin))
scores_test = list(boo.staged_score(X_test, y_test_bin))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(range(1,801), scores, 'g-', label = 'Train score')
ax.plot(range(1,801), scores_test, 'b-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('# of iterations', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title('Boosting Scores (Binary) - function of iterations', size=18)

plt.show()
```



![png](model_comparison_files/model_comparison_42_0.png)




```python
scores_test.index(max(scores_test))
```





    749



The boosting procedure with 595 iterations seems to be our best choice of hyper parameters to predict the binary response variable.

### Random Forest



```python
depths = range(1,25)

scores = []
scores_test = []
for i in depths:
    rfc = RandomForestClassifier(n_estimators=50, max_depth=i)
    rfc.fit(X_train, y_train_multi)
    #scores.append(np.mean(cross_val_score(rfc, X_train, y_train_multi, cv=5)))
    scores.append(rfc.score(X_train, y_train_multi))
    scores_test.append(rfc.score(X_test, y_test_multi))

fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(depths, scores, 'g*-', label = 'Train score')
ax.plot(depths, scores_test, 'b*-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('max_depth', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title('RandomForest Train/Test Scores - function of max_depth')

plt.show()
```



![png](model_comparison_files/model_comparison_46_0.png)


Among the Random Forest models for the multi category response variable, `max_depth=4` appears to be the optimal parameter.



```python
#Repeating the process for the binary response variable
depths = range(1,25)

scores = []
scores_test = []
for i in depths:
    rfc = RandomForestClassifier(n_estimators=50, max_depth=i)
    rfc.fit(X_train, y_train_bin)
    #scores.append(np.mean(cross_val_score(rfc, X_train, y_train_bin, cv=5)))
    scores.append(rfc.score(X_train, y_train_bin))
    scores_test.append(rfc.score(X_test, y_test_bin))
    
fig, ax = plt.subplots(1,1, figsize=(12, 6))

ax.plot(depths, scores, 'g*-', label = 'Train score')
ax.plot(depths, scores_test, 'b*-', label = 'Test score')
ax.legend(fontsize=14)
ax.set_xlabel('max_depth', size=14)
ax.set_ylabel('Score', size=14)
ax.set_title('RandomForest Scores (Binary) - function of max_depth')

plt.show()
```



![png](model_comparison_files/model_comparison_48_0.png)


Among the Random Forest models for the binary response variable, `max_depth=8` appears to be the optimal parameter.