---
title: ADNI Merge
notebook: ..\markdown_notebooks\merged_data_eda.ipynb
section: 2
subsection: 5
---

## Contents
{:.no_toc}
*  
{: toc}


The purpose of this notebook is to explore/clean the ADNI Merged dataset to see how it may be used in building design matrices.



```python
from IPython.core.display import HTML
with open("./project.css") as css:
    styles = css.read()
HTML(styles)
```





<style>
    table {
        display: inline-block
    }
    .rendered_html td, .rendered_html th {text-align: left;}
</style>



## ADNI Data
Before looking at a single observation or feature, there is a lot of information we can glean from reviewing ADNI metadata. There are over 250 datasets in the ADNI data inventory spanning the 4 study phases (ADNI 1, GO, 2, 3) - and this number does not include the archives. These studies are longitudinal. ADNI-1 started in 2004 and ADNI-3 continues today. Although there is potentially a wealth of information, insights, and predictive power in these data, their data collection methods and longitudinal nature present many challenges.

One challenge is that all biometrics within the scope of the study are not collected across all study phases. Also, within each phase, not all participants had all measurements taken. For example, in ADNI-1, $100\%$ of the cohort had a 1.5 Tesla (1.5T) MRI, $50\%$ had a PET scan. Of the $50\%$ that didn't have a PET scan, $25\%$ had a 3T MRI. Finally, only $20\%$ of the ADNI-1 cohort had a lumbar puncture (L.P.) to collect cerebral spinal fluid (CSF).

Other data challenges are related to the longitudinal nature of the studies across the different phases. In each successive phase of the study, participants were rolled over from previous phases while new participants were also added - *(cohort details can be seen in the table above)*. However, existing participants in the study must provide their consent to be included in each subsequent phase. Furthermore, an obvious, but nonetheless real, complication with this population is that a participant could be removed from the study at any time due to significant deterioration in health or death. 

The result is that each phase of the study produces a richer set of longitudinal data than the previous study because of the rollover participants. The downside of this design is the inherent introduction of missingness into the data due to the recently joined participants.

### ADNI Phases
There have been ADNI 4 study phases to date with the following goals:

<!-- Begin ADNI Phase table -->

| Study Phase | Goal | Dates | Cohort |
|:---: |:--- |:--- | --- |
| ADNI 1 | Develop biomarkers as outcome measures for clinical trials | 2004-2009 | 200 elderly controls<br>400 MCI<br>200 AD |
| ADNI GO | Examine biomarkers in earlier stages of disease | 2009-2011 | Existing ADNI-1 +<br>200 early MCI |
| ADNI 2 | Develop biomarkers as predictors of cognitive decline, and as outcome measures | 2011-2016 | Existing ADNI-1 and ADNI-GO +<br>150 elderly controls<br>100 early MCI<br>150 late MCI<br>150 AD |
| ADNI 3 | Study the use of tau PET and functional imaging techniques in clinical trials | 2016 - present | Existing ADNI-1, ADNI-GO, ADNI-2 +<br>133 elderly controls<br>151 MCI<br>87 AD |

<!-- End ADNI phase table -->

### An initial look at the data.
Given the breadth of available data and the challenges mentioned above, deciding where to invest EDA effort is an important consideration. Fortunately, ADNI provides a combined dataset consisting of key ADNI tables merged into a single table based on the patient identifier or `RID`. As is common with most ADNI datasets, each observation represents a single visit for a participant. This means that a single participant (`RID`) may appear multiple times in the dataset. The number of occurrences will generally depend on what phase the participant entered the study.

Let's take an initial look at the merged dataset.



```python
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ADNI_utilities as utils

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
```




```python
adni_merge = pd.read_csv('../data/ADNIMERGE.csv', low_memory=False)
```




```python
print("The shape of the dataset is {}.".format(adni_merge.shape))
print("There are {} unique participants in the dataset."
      .format(len(adni_merge.RID.unique())))
print("There is an average of {:.2f} rows in the data per participant."
      .format(len(adni_merge)/len(adni_merge.RID.unique())))
```


    The shape of the dataset is (13632, 113).
    There are 2081 unique participants in the dataset.
    There is an average of 6.55 rows in the data per participant.
    

As seen above, the dataset contains $2081$ unique study participants spanning $13632$ visits. The data is longitudinal based on participant visits spaced roughly six months apart. The `VISCODE` feature represents the visit in which the measurements and evaluations were captured. The initial evaluation measurements are identified by `VISCODE` = `'bl'`, which stands for baseline. Below are the unique `VISCODE` values in the dataset:



```python
adni_merge.sort_values(by='Month')['VISCODE'].unique()
```





    array(['bl', 'm03', 'm06', 'm12', 'm18', 'm24', 'm36', 'm30', 'm42',
           'm48', 'm54', 'm60', 'm66', 'm72', 'm78', 'm84', 'm90', 'm96',
           'm102', 'm108', 'm114', 'm120', 'm126', 'm132', 'm144', 'm156'],
          dtype=object)



Let's visualize the number of study participants per `VISCODE`.



```python
# M represents months since the last visit (0 = baseline/initial visit)
adni_by_month = adni_merge.groupby(by='M').count()
particpants = adni_by_month['RID']
visits = adni_merge.sort_values(by='M')['VISCODE'].unique()

fig, ax = plt.subplots(1, 1, figsize=(14,8))

ax.set_title('Number of participants per visit', size=17)
ax.set_xticks(range(0, 26, 2))
ax.set_xlabel('Visit Code (indicates months since inital baseline visit)', size=14)
ax.set_ylabel('Participants', size=14)
ax.bar(visits, particpants)

plt.show()
```



![png](merged_data_eda_files/merged_data_eda_11_0.png)


Based on the design of the study as discussed above, we expect there to be a lot of missing data in this data set. Let's take a look.



```python
# Calculate missing data
missing_data = utils.calculate_missing_data(adni_merge)

# Look at the top 10 columns in terms of missing values
missing_data.head(10)
```


    There are a total of 507270 missing values.
    Out of 113 features in the dataset, 93 have missing values.
    
    Quartiles of missing data:
          Num Missing  Pct. Missing
    0.25       3839.0     28.161678
    0.50       6185.0     45.371185
    0.75       7758.0     56.910211
    1.00      13632.0    100.000000
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Num Missing</th>
      <th>Pct. Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FLDSTRENG_bl</th>
      <td>13632.0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>FLDSTRENG</th>
      <td>13632.0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>PIB_bl</th>
      <td>13483.0</td>
      <td>98.906984</td>
    </tr>
    <tr>
      <th>PIB</th>
      <td>13409.0</td>
      <td>98.364143</td>
    </tr>
    <tr>
      <th>PTAU</th>
      <td>11262.0</td>
      <td>82.614437</td>
    </tr>
    <tr>
      <th>ABETA</th>
      <td>11261.0</td>
      <td>82.607101</td>
    </tr>
    <tr>
      <th>TAU</th>
      <td>11261.0</td>
      <td>82.607101</td>
    </tr>
    <tr>
      <th>AV45</th>
      <td>11122.0</td>
      <td>81.587441</td>
    </tr>
    <tr>
      <th>FDG</th>
      <td>10125.0</td>
      <td>74.273768</td>
    </tr>
    <tr>
      <th>DIGITSCOR</th>
      <td>9832.0</td>
      <td>72.124413</td>
    </tr>
  </tbody>
</table>
</div>



We've noticed that the numbers of non-null values for `PTAU`, `ABETA`, and `TAU` are suspiciously close to the number of unique participants. The fact that these features all have almost the exact same number of missing values could be an artifact of how and when these data were collected. Perhaps these were collected on the initial baseline `bl` visit.



```python
missing_proteins = missing_data.loc[['TAU', 'PTAU', 'ABETA']]
missing_proteins['Num Values Present'] = len(adni_merge) - missing_proteins['Num Missing']
missing_proteins['Num Participants'] = len(adni_merge.RID.unique())
missing_proteins
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Num Missing</th>
      <th>Pct. Missing</th>
      <th>Num Values Present</th>
      <th>Num Participants</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TAU</th>
      <td>11261.0</td>
      <td>82.607101</td>
      <td>2371.0</td>
      <td>2081</td>
    </tr>
    <tr>
      <th>PTAU</th>
      <td>11262.0</td>
      <td>82.614437</td>
      <td>2370.0</td>
      <td>2081</td>
    </tr>
    <tr>
      <th>ABETA</th>
      <td>11261.0</td>
      <td>82.607101</td>
      <td>2371.0</td>
      <td>2081</td>
    </tr>
  </tbody>
</table>
</div>



There are many options to deal with the variable number of visits in the merged data set. Instead of vertically stacking the visits as in the merged dataset, we could split on `VISCODE` and stack the data *horizontally* creating wide rows with many more features. However, this is essentially transposing the data and moving the missing values from deep columns to wide rows. Another option is to split the data into multiple subsets of data based on `VISCODE` and deal with them separately. As shown in the "*Participants per visit*" figure, every participant had at least a baseline visit. This subset should provide the most complete and uniform representation of the data.



```python
baseline = adni_merge[adni_merge['VISCODE'] == 'bl'].copy()
print('Shape of the baseline visit subset: ', baseline.shape)

baseline.head()
```


    Shape of the baseline visit subset:  (2081, 113)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>PTID</th>
      <th>VISCODE</th>
      <th>SITE</th>
      <th>COLPROT</th>
      <th>ORIGPROT</th>
      <th>EXAMDATE</th>
      <th>DX_bl</th>
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>...</th>
      <th>TAU_bl</th>
      <th>PTAU_bl</th>
      <th>FDG_bl</th>
      <th>PIB_bl</th>
      <th>AV45_bl</th>
      <th>Years_bl</th>
      <th>Month_bl</th>
      <th>Month</th>
      <th>M</th>
      <th>update_stamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>011_S_0002</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2005-09-08</td>
      <td>CN</td>
      <td>74.3</td>
      <td>Male</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.36665</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2018-10-19 22:51:15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2005-09-12</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>239.7</td>
      <td>22.83</td>
      <td>1.08355</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2018-10-19 22:51:15.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>022_S_0004</td>
      <td>bl</td>
      <td>22</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2005-11-08</td>
      <td>LMCI</td>
      <td>67.5</td>
      <td>Male</td>
      <td>...</td>
      <td>153.1</td>
      <td>13.29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2018-10-19 22:51:15.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>011_S_0005</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2005-09-07</td>
      <td>CN</td>
      <td>73.7</td>
      <td>Male</td>
      <td>...</td>
      <td>337</td>
      <td>33.43</td>
      <td>1.29343</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2018-10-19 22:51:15.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6</td>
      <td>100_S_0006</td>
      <td>bl</td>
      <td>100</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2005-11-29</td>
      <td>LMCI</td>
      <td>80.4</td>
      <td>Female</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2018-10-19 22:51:15.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 113 columns</p>
</div>



Let's examime the missing data from the baseline visit dataset.



```python
# Calculate missing data
missing_data = utils.calculate_missing_data(baseline)
```


    There are a total of 54057 missing values.
    Out of 113 features in the dataset, 91 have missing values.
    
    Quartiles of missing data:
          Num Missing  Pct. Missing
    0.25         22.0      1.057184
    0.50        728.0     34.983181
    0.75        851.0     40.893801
    1.00       2081.0    100.000000
    

## Filling in Missing Data
Based on the study data, we know that not all biometrics are measured at every visit. Therefore we may be able to pull measures together from different visits to help fill in missing data. Of course since we're dealing with longitudinal data with visits month or years apart, we have to make sure that we only consider measures from visits where the diagnosis code is unchanged.



```python
# For every column in the baseline (VISCODE='bl') with missing values, look for
# a value in subsequent visits with the constraint that the DX code must
# not have changed. Take the first (earliest) biometric measure available.

baseline.sort_values(by='RID')
missing_cols = baseline.columns[baseline.isnull().any()]
viscodes = list(adni_merge.sort_values(by='Month')['VISCODE'].unique())
viscodes.pop(0) # Get rid of 'bl'

missing_values = baseline.isnull().sum().sum()
updated_values = 0

print('Searching...')
for col in missing_cols:
    
    for v in viscodes:
        
        # Get the RIDs with missing values in this colummn.
        # Do this for each VISCODE since we are iteratively
        # updating missing values.
        rids = baseline[baseline[col].isnull()].RID
        
        # Create a DataFrame from adni_merge for the current
        # VISCODE, RIDs, & where current col is not null.
        df = adni_merge.loc[(adni_merge.RID.isin(rids))
                            & (adni_merge.VISCODE == v)
                            & (adni_merge[col].notnull()),
                            baseline.columns] 
               
        if df.empty: # if no matches, continue
            continue
            
        df = df.copy()
        df.sort_values(by='RID', inplace=True)
            
        # Find baseline participants who are also in the current VISCODE
        bl = baseline[baseline.RID.isin(df.RID)].copy()
        bl.sort_values(by='RID', inplace=True)
        df.index = bl.index
        
        # Only keep those where the diagnosis is unchanged & col is not null
        df = df[(df.DX == bl.DX) & (df[col].notnull())]

        if df.empty:  # if DX codes don't match, continue
            continue

        # Update null values in the original baseline DF
        baseline.loc[baseline.index.isin(df.index), col] = df[col]
        updated_values += len(df)

print(f'Updated {updated_values} of {missing_values} missing values.')
```


    Searching...
    Updated 4104 of 54057 missing values.
    



```python
# Calculate missing data
missing_data = utils.calculate_missing_data(baseline)

# Look at the top 10 columns in terms of missing values
missing_data.head(10)
```


    There are a total of 49953 missing values.
    Out of 113 features in the dataset, 91 have missing values.
    
    Quartiles of missing data:
          Num Missing  Pct. Missing
    0.25         21.0      1.009130
    0.50        641.0     30.802499
    0.75        832.0     39.980778
    1.00       2081.0    100.000000
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Num Missing</th>
      <th>Pct. Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FLDSTRENG</th>
      <td>2081.0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>FLDSTRENG_bl</th>
      <td>2081.0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>PIB_bl</th>
      <td>2061.0</td>
      <td>99.038924</td>
    </tr>
    <tr>
      <th>PIB</th>
      <td>1990.0</td>
      <td>95.627102</td>
    </tr>
    <tr>
      <th>DIGITSCOR_bl</th>
      <td>1267.0</td>
      <td>60.884190</td>
    </tr>
    <tr>
      <th>DIGITSCOR</th>
      <td>1264.0</td>
      <td>60.740029</td>
    </tr>
    <tr>
      <th>AV45_bl</th>
      <td>1107.0</td>
      <td>53.195579</td>
    </tr>
    <tr>
      <th>AV45</th>
      <td>971.0</td>
      <td>46.660259</td>
    </tr>
    <tr>
      <th>EcogSPOrgan_bl</th>
      <td>891.0</td>
      <td>42.815954</td>
    </tr>
    <tr>
      <th>PTAU_bl</th>
      <td>866.0</td>
      <td>41.614608</td>
    </tr>
  </tbody>
</table>
</div>



Clearly a lot of missing data remain and we will likely have to explore methods to impute these values. Before doing that however, we will explore the data a little closer to see if there are features that should be dropped due to high correlation, lack of information, or other reasons.

`FLDSTRENG` and `FLDSTRENG_bl` are providing absolutely no information so we can drop them.



```python
baseline = baseline.drop(labels=['FLDSTRENG', 'FLDSTRENG_bl'], axis=1)
```


There appears to be a lot of features with similar, if not identical, information such as `TAU`, `TAU_bl`, `AV45`, `AV45_bl`. Let's examine this pattern to see if these pairs are highly correlated.



```python
# Generate a correlation matrix of xxx_bl vs xxx
corr_df = baseline.corr()
cols = baseline.columns
col1, col2, corr = [], [], []

# Specifically check the correlation of xxx_bl to xxx
for col in cols:
    if '_bl' in col.lower(): 
        drop_bl = col[0:-3]
        if (drop_bl in cols):
            if (col in corr_df.index and corr_df.loc[col][drop_bl] > .8):
                col1.append(col)
                col2.append(drop_bl)
                corr.append(corr_df.loc[col][drop_bl])

# Display the results                
bl_corr_df = pd.DataFrame({"Baseline column": col1, 'Alternate column': col2, "Correlation": corr})
bl_corr_df
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline column</th>
      <th>Alternate column</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CDRSB_bl</td>
      <td>CDRSB</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADAS11_bl</td>
      <td>ADAS11</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADAS13_bl</td>
      <td>ADAS13</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADASQ4_bl</td>
      <td>ADASQ4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MMSE_bl</td>
      <td>MMSE</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RAVLT_immediate_bl</td>
      <td>RAVLT_immediate</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RAVLT_learning_bl</td>
      <td>RAVLT_learning</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RAVLT_forgetting_bl</td>
      <td>RAVLT_forgetting</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RAVLT_perc_forgetting_bl</td>
      <td>RAVLT_perc_forgetting</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LDELTOTAL_BL</td>
      <td>LDELTOTAL</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DIGITSCOR_bl</td>
      <td>DIGITSCOR</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TRABSCOR_bl</td>
      <td>TRABSCOR</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FAQ_bl</td>
      <td>FAQ</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mPACCdigit_bl</td>
      <td>mPACCdigit</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>mPACCtrailsB_bl</td>
      <td>mPACCtrailsB</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ventricles_bl</td>
      <td>Ventricles</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hippocampus_bl</td>
      <td>Hippocampus</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>WholeBrain_bl</td>
      <td>WholeBrain</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Entorhinal_bl</td>
      <td>Entorhinal</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Fusiform_bl</td>
      <td>Fusiform</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MidTemp_bl</td>
      <td>MidTemp</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ICV_bl</td>
      <td>ICV</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MOCA_bl</td>
      <td>MOCA</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>EcogPtMem_bl</td>
      <td>EcogPtMem</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>EcogPtLang_bl</td>
      <td>EcogPtLang</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>EcogPtVisspat_bl</td>
      <td>EcogPtVisspat</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>EcogPtPlan_bl</td>
      <td>EcogPtPlan</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>EcogPtOrgan_bl</td>
      <td>EcogPtOrgan</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>EcogPtDivatt_bl</td>
      <td>EcogPtDivatt</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>EcogPtTotal_bl</td>
      <td>EcogPtTotal</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>EcogSPMem_bl</td>
      <td>EcogSPMem</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>EcogSPLang_bl</td>
      <td>EcogSPLang</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>EcogSPVisspat_bl</td>
      <td>EcogSPVisspat</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>EcogSPPlan_bl</td>
      <td>EcogSPPlan</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>EcogSPOrgan_bl</td>
      <td>EcogSPOrgan</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>EcogSPDivatt_bl</td>
      <td>EcogSPDivatt</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>EcogSPTotal_bl</td>
      <td>EcogSPTotal</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>FDG_bl</td>
      <td>FDG</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>PIB_bl</td>
      <td>PIB</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>AV45_bl</td>
      <td>AV45</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



As we can see from the table above, there is perfect correlation between the baseline and non-baseline versions of these fields. Since these pairs of features indeed contain duplicate information, we can drop one of each pair.



```python
baseline = baseline.drop(labels=bl_corr_df['Baseline column'].values, axis=1)
print("The new shape of the baseline subset is {}.".format(baseline.shape))
```


    The new shape of the baseline subset is (2081, 71).
    



```python
# Let's see what features we have now
baseline.columns.sort_values()
```





    Index(['ABETA', 'ABETA_bl', 'ADAS11', 'ADAS13', 'ADASQ4', 'AGE', 'APOE4',
           'AV45', 'CDRSB', 'COLPROT', 'DIGITSCOR', 'DX', 'DX_bl', 'EXAMDATE',
           'EXAMDATE_bl', 'EcogPtDivatt', 'EcogPtLang', 'EcogPtMem', 'EcogPtOrgan',
           'EcogPtPlan', 'EcogPtTotal', 'EcogPtVisspat', 'EcogSPDivatt',
           'EcogSPLang', 'EcogSPMem', 'EcogSPOrgan', 'EcogSPPlan', 'EcogSPTotal',
           'EcogSPVisspat', 'Entorhinal', 'FAQ', 'FDG', 'FSVERSION',
           'FSVERSION_bl', 'Fusiform', 'Hippocampus', 'ICV', 'IMAGEUID',
           'LDELTOTAL', 'M', 'MMSE', 'MOCA', 'MidTemp', 'Month', 'Month_bl',
           'ORIGPROT', 'PIB', 'PTAU', 'PTAU_bl', 'PTEDUCAT', 'PTETHCAT',
           'PTGENDER', 'PTID', 'PTMARRY', 'PTRACCAT', 'RAVLT_forgetting',
           'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting', 'RID',
           'SITE', 'TAU', 'TAU_bl', 'TRABSCOR', 'VISCODE', 'Ventricles',
           'WholeBrain', 'Years_bl', 'mPACCdigit', 'mPACCtrailsB', 'update_stamp'],
          dtype='object')



We still have some columns that look very similar. These may contain non-numeric values such as strings or NaNs, or possibly they are really uncorrelated.



```python
# Columns to check for duplicates
cols = ['ABETA', 'DX', 'EXAMDATE','FSVERSION',  'PTAU', 'TAU']

bl_missing = []
missing = []
matching_vals = []

for col in cols:
    missing.append(baseline[col].isnull().sum())
    bl_missing.append(baseline[col+'_bl'].isnull().sum())
    match = (baseline[col] == baseline[col+'_bl']).sum()
    matching_vals.append((match + min(missing[-1], bl_missing[-1]))/len(baseline) * 100)
                 
# Display the results                
bl_dupes = pd.DataFrame({'Missing Values': missing,
                            'Baseline Missing Values': bl_missing,
                            'Percent Matching': matching_vals}, index=cols)
bl_dupes
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>Baseline Missing Values</th>
      <th>Percent Matching</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ABETA</th>
      <td>832</td>
      <td>866</td>
      <td>98.366170</td>
    </tr>
    <tr>
      <th>DX</th>
      <td>25</td>
      <td>16</td>
      <td>24.939933</td>
    </tr>
    <tr>
      <th>EXAMDATE</th>
      <td>0</td>
      <td>0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>FSVERSION</th>
      <td>346</td>
      <td>357</td>
      <td>99.471408</td>
    </tr>
    <tr>
      <th>PTAU</th>
      <td>832</td>
      <td>866</td>
      <td>98.366170</td>
    </tr>
    <tr>
      <th>TAU</th>
      <td>832</td>
      <td>866</td>
      <td>98.366170</td>
    </tr>
  </tbody>
</table>
</div>



All of the pairs are nearly exact duplicates except `DX` and `DX_bl`, so we can drop one of the duplicate columns. The baseline versions have slightly more missing data, so we'll drop those. Then we'll take a look at `DX` vs.`DX_bl`.



```python
# Get a list of duplicate column names to drop
dupe_cols = [col + '_bl' for col in cols]

# Remove DX_bl until we investigate further
del dupe_cols[dupe_cols.index('DX_bl')]

# Drop the columns
baseline = baseline.drop(labels=dupe_cols, axis=1)
```




```python
# See how DX maps to DX_bl
baseline.drop_duplicates('DX_bl')[['DX_bl', 'DX']]
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DX_bl</th>
      <th>DX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CN</td>
      <td>CN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AD</td>
      <td>Dementia</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LMCI</td>
      <td>MCI</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SMC</td>
      <td>CN</td>
    </tr>
    <tr>
      <th>2848</th>
      <td>EMCI</td>
      <td>MCI</td>
    </tr>
    <tr>
      <th>11389</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Although similar, the diagnoses in `DX_bl` are more specific than those in `DX`. We'll use the more standard diagnosis codes in `DX`. However `DX_bl` has slightly less missingness so we'll keep that column and remap some of the values to match `DX` where needed.



```python
# Drop DX
baseline = baseline.drop(labels=['DX'], axis=1)

# Remap some of the DX_bl values
baseline.DX_bl = baseline.DX_bl.replace('LMCI','MCI')
baseline.DX_bl = baseline.DX_bl.replace('EMCI','MCI')
baseline.DX_bl = baseline.DX_bl.replace('SMC','CN')
```


We also have some participants for which we have no diagnosis code, so these records will not be useful and can be dropped.



```python
missing_dx = len(baseline[baseline['DX_bl'].isnull()])
baseline = baseline.dropna(axis=0, subset=['DX_bl'])
print(f'Removed {missing_dx} participants with no diagnosis code.')
```


    Removed 16 participants with no diagnosis code.
    

`PTID` is also duplicative. It is a combination of `RID` and `SITE`.



```python
baseline = baseline.drop(labels='PTID', axis=1)
```




```python
print("The new shape of the baseline subset is {}.".format(baseline.shape))
utils.calculate_missing_data(baseline)
```


    The new shape of the baseline subset is (2065, 64).
    There are a total of 20821 missing values.
    Out of 64 features in the dataset, 44 have missing values.
    
    Quartiles of missing data:
          Num Missing  Pct. Missing
    0.25        22.75      1.101695
    0.50       541.50     26.222760
    0.75       649.00     31.428571
    1.00      1974.00     95.593220
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Num Missing</th>
      <th>Pct. Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PIB</th>
      <td>1974.0</td>
      <td>95.593220</td>
    </tr>
    <tr>
      <th>DIGITSCOR</th>
      <td>1248.0</td>
      <td>60.435835</td>
    </tr>
    <tr>
      <th>AV45</th>
      <td>956.0</td>
      <td>46.295400</td>
    </tr>
    <tr>
      <th>ABETA</th>
      <td>816.0</td>
      <td>39.515738</td>
    </tr>
    <tr>
      <th>TAU</th>
      <td>816.0</td>
      <td>39.515738</td>
    </tr>
    <tr>
      <th>PTAU</th>
      <td>816.0</td>
      <td>39.515738</td>
    </tr>
    <tr>
      <th>EcogSPOrgan</th>
      <td>670.0</td>
      <td>32.445521</td>
    </tr>
    <tr>
      <th>EcogSPDivatt</th>
      <td>656.0</td>
      <td>31.767554</td>
    </tr>
    <tr>
      <th>EcogSPVisspat</th>
      <td>654.0</td>
      <td>31.670702</td>
    </tr>
    <tr>
      <th>EcogSPPlan</th>
      <td>651.0</td>
      <td>31.525424</td>
    </tr>
    <tr>
      <th>EcogSPTotal</th>
      <td>649.0</td>
      <td>31.428571</td>
    </tr>
    <tr>
      <th>EcogSPMem</th>
      <td>649.0</td>
      <td>31.428571</td>
    </tr>
    <tr>
      <th>EcogSPLang</th>
      <td>648.0</td>
      <td>31.380145</td>
    </tr>
    <tr>
      <th>MOCA</th>
      <td>647.0</td>
      <td>31.331719</td>
    </tr>
    <tr>
      <th>EcogPtOrgan</th>
      <td>644.0</td>
      <td>31.186441</td>
    </tr>
    <tr>
      <th>FDG</th>
      <td>643.0</td>
      <td>31.138015</td>
    </tr>
    <tr>
      <th>EcogPtDivatt</th>
      <td>642.0</td>
      <td>31.089588</td>
    </tr>
    <tr>
      <th>EcogPtLang</th>
      <td>640.0</td>
      <td>30.992736</td>
    </tr>
    <tr>
      <th>EcogPtVisspat</th>
      <td>640.0</td>
      <td>30.992736</td>
    </tr>
    <tr>
      <th>EcogPtPlan</th>
      <td>639.0</td>
      <td>30.944310</td>
    </tr>
    <tr>
      <th>EcogPtMem</th>
      <td>638.0</td>
      <td>30.895884</td>
    </tr>
    <tr>
      <th>EcogPtTotal</th>
      <td>638.0</td>
      <td>30.895884</td>
    </tr>
    <tr>
      <th>Entorhinal</th>
      <td>445.0</td>
      <td>21.549637</td>
    </tr>
    <tr>
      <th>Fusiform</th>
      <td>445.0</td>
      <td>21.549637</td>
    </tr>
    <tr>
      <th>MidTemp</th>
      <td>445.0</td>
      <td>21.549637</td>
    </tr>
    <tr>
      <th>Hippocampus</th>
      <td>398.0</td>
      <td>19.273608</td>
    </tr>
    <tr>
      <th>Ventricles</th>
      <td>358.0</td>
      <td>17.336562</td>
    </tr>
    <tr>
      <th>WholeBrain</th>
      <td>346.0</td>
      <td>16.755448</td>
    </tr>
    <tr>
      <th>APOE4</th>
      <td>340.0</td>
      <td>16.464891</td>
    </tr>
    <tr>
      <th>FSVERSION</th>
      <td>330.0</td>
      <td>15.980630</td>
    </tr>
    <tr>
      <th>IMAGEUID</th>
      <td>330.0</td>
      <td>15.980630</td>
    </tr>
    <tr>
      <th>ICV</th>
      <td>330.0</td>
      <td>15.980630</td>
    </tr>
    <tr>
      <th>TRABSCOR</th>
      <td>26.0</td>
      <td>1.259080</td>
    </tr>
    <tr>
      <th>FAQ</th>
      <td>13.0</td>
      <td>0.629540</td>
    </tr>
    <tr>
      <th>RAVLT_perc_forgetting</th>
      <td>8.0</td>
      <td>0.387409</td>
    </tr>
    <tr>
      <th>ADAS13</th>
      <td>8.0</td>
      <td>0.387409</td>
    </tr>
    <tr>
      <th>ADAS11</th>
      <td>6.0</td>
      <td>0.290557</td>
    </tr>
    <tr>
      <th>LDELTOTAL</th>
      <td>3.0</td>
      <td>0.145278</td>
    </tr>
    <tr>
      <th>RAVLT_forgetting</th>
      <td>3.0</td>
      <td>0.145278</td>
    </tr>
    <tr>
      <th>RAVLT_learning</th>
      <td>3.0</td>
      <td>0.145278</td>
    </tr>
    <tr>
      <th>RAVLT_immediate</th>
      <td>3.0</td>
      <td>0.145278</td>
    </tr>
    <tr>
      <th>ADASQ4</th>
      <td>3.0</td>
      <td>0.145278</td>
    </tr>
    <tr>
      <th>mPACCdigit</th>
      <td>2.0</td>
      <td>0.096852</td>
    </tr>
    <tr>
      <th>mPACCtrailsB</th>
      <td>2.0</td>
      <td>0.096852</td>
    </tr>
  </tbody>
</table>
</div>



In the new dataset, it is clear that `PIB` will not be useful and can be removed. PIB or *PiB* stands for **Pi**ttsburgh Compound-**B** - a synthetic radiotracer developed for use in PET scans to visualize and measure A$\beta$ deposits in the brain.



```python
baseline = baseline.drop(labels='PIB', axis=1)
```


There are other features in the data that we are confindent won't be helpful in predicting AD. These include features such as `SITE`, `update_stamp`, `EXAMDATE`, etc. We will remove them.



```python
baseline = baseline.drop(labels=['update_stamp', 'Years_bl', 'SITE', 'VISCODE', 'COLPROT', 'ORIGPROT',
                                 'Month_bl', 'M', 'EXAMDATE', 'IMAGEUID'], axis=1)
```




```python
# Let's see what features we have now
baseline.columns.sort_values()
```





    Index(['ABETA', 'ADAS11', 'ADAS13', 'ADASQ4', 'AGE', 'APOE4', 'AV45', 'CDRSB',
           'DIGITSCOR', 'DX_bl', 'EcogPtDivatt', 'EcogPtLang', 'EcogPtMem',
           'EcogPtOrgan', 'EcogPtPlan', 'EcogPtTotal', 'EcogPtVisspat',
           'EcogSPDivatt', 'EcogSPLang', 'EcogSPMem', 'EcogSPOrgan', 'EcogSPPlan',
           'EcogSPTotal', 'EcogSPVisspat', 'Entorhinal', 'FAQ', 'FDG', 'FSVERSION',
           'Fusiform', 'Hippocampus', 'ICV', 'LDELTOTAL', 'MMSE', 'MOCA',
           'MidTemp', 'Month', 'PTAU', 'PTEDUCAT', 'PTETHCAT', 'PTGENDER',
           'PTMARRY', 'PTRACCAT', 'RAVLT_forgetting', 'RAVLT_immediate',
           'RAVLT_learning', 'RAVLT_perc_forgetting', 'RID', 'TAU', 'TRABSCOR',
           'Ventricles', 'WholeBrain', 'mPACCdigit', 'mPACCtrailsB'],
          dtype='object')



We have some non-numeric data in `ABETA`, `TAU`, and `PTAU`, such as '>1300' or '<80'. We'll remove the `>` and `<` characters and change the dtype to float64.



```python
# remove < or > 
def remove_gt_lt(val):
    if type(val) == str:
        return float(val.replace('>', '').replace('<', ''))
    else:
        return val
    
    
for col in ['ABETA', 'TAU', 'PTAU']:
    values = baseline[col].values
    baseline[col] = list(map(remove_gt_lt, values))
```




```python
baseline['PTEDUCAT'] = baseline.PTEDUCAT.astype('float')
```


## Combining ADNI Merged and Per-Patient data
There are additional (potentially valuable) data that are not included in the Merged data set. These data have been cleaned and put into a format such that we can join them to the merged data set. We will merge the additional per-patient data with the ADNI Merged data. First we set the index to `RID`.



```python
# First set the baseline index to 'RID'
baseline.index = baseline['RID']
baseline = baseline.drop(labels='RID', axis=1)
```


Read in the per-patient data and identify and remove any duplicate columns.



```python
# Read in the additional patient data. Use RID as the index column
pat_data = pd.read_csv('../data/Per_Patient/patient_firstidx_merge.csv', index_col='RID', na_values='-1')

dupes = list(set(baseline.columns) & set(pat_data.columns))
pat_data = pat_data.drop(labels=dupes, axis=1)
print(f'Dropped duplicate columns: {dupes}')
```


    Dropped duplicate columns: ['PTMARRY', 'PTEDUCAT', 'ABETA', 'TAU', 'PTRACCAT', 'PTETHCAT', 'PTGENDER', 'PTAU']
    

Now we'll merge the ADNI_Merge dataset and cleaned Per_Patient dataset that we've built and curated from raw data files. We'll merge the datasets on `RID`.



```python
# Merge baseline with pat_data
pat_comb = pd.merge(baseline, pat_data, on='RID')

# First drop cols that have only NA values
pat_comb = pat_comb.dropna(axis='columns', thresh=1)
```


Save the combined file.



```python
pat_comb.to_csv('../data/Per_Patient/pat_merge_b4_impute.csv')
```


Our goal is to strike a balance between keeping as much data as possible and reducing noise introduced by imputing columns that are nearly completely void of information. We will explore the relationship between these by imputing with different thresholds for missing data.
