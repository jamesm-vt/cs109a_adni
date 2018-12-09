---
title: Biomarker Exploratory Data Analysis
notebook: ..\markdown_notebooks\biomarker_EDA.ipynb
section: 2
subsection: 2
---

## Contents
{:.no_toc}
*  
{: toc}


Import libraries



```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# import custom dependencies
import sys
sys.path.append('../ADNI_')
from ADNI_utilities import define_terms, describe_meta_data, append_meta_cols
```


Initialize data structures



```python
# import adni dictionary
apo_dict = pd.read_csv("../data/Biomarker Data/APOERES_DICT.csv")

# import data set specific dictionaries
csf_dict = pd.read_csv("../data/Biomarker Data/UPENNBIOMK_MASTER_DICT.csv")
lab_dict = pd.read_csv("../data/Biomarker Data/LABDATA_DICT.csv")
adni_dict_df = pd.read_csv("../data/study info/DATADIC.csv")
```




```python
# define dataframes from the biomarker dataset
apo_df = pd.read_csv("../data/Biomarker Data/APOERES.csv")
csf_df = pd.read_csv("../data/Biomarker Data/UPENNBIOMK_MASTER.csv")
lab_df =  pd.read_csv("../data/Biomarker Data/LABDATA.csv")
```


<div style="background-color:lightgrey">
  <h3>ApoE Measurements</h3>
</div>

The APOERES table contains information about patient alleles for the ApoE gene which has been linked to alzheimers.



```python
# describe data structure
describe_meta_data(apo_df)
```


    Phases:	 ['ADNI1' 'ADNIGO2']
    Num Entries: 2067
    Num Columns: 16
    Num Patients: 2067
    Records per Patient: 1-1
    Phases spanned per patient: 1-1
    Patients w/ Duplicates: 0
    

Looks like there is a single entry for each patient with a single phase spanned. We can take a look at the features to see what we might be interested in.



```python
# define and print terms from apo table
term_defs = define_terms(apo_df, adni_dict_df, "APOERES")
term_defs
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
      <th>FLDNAME</th>
      <th>TYPE</th>
      <th>TBLNAME</th>
      <th>TEXT</th>
      <th>CODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Record ID</td>
      <td>"crfname","ApoE Genotyping - Results","indexes...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Site ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>APOERES</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>APOERES</td>
      <td>Date record created</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>APOERES</td>
      <td>Date record last updated</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>APTESTDT</td>
      <td>D</td>
      <td>APOERES</td>
      <td>Date Test Performed</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>APGEN1</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Genotype - Allele 1</td>
      <td>2..4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>APGEN2</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Genotype - Allele 2</td>
      <td>2..4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>APVOLUME</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Volume of Blood Shipped in Lavendar Top Tube</td>
      <td>1..10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>APRECEIVE</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Sample recieved within 24 hours of blood draw?</td>
      <td>1=Yes;0=No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>APAMBTEMP</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Sample shipped at ambient temperature?</td>
      <td>1=Yes;0=No</td>
    </tr>
    <tr>
      <th>13</th>
      <td>APRESAMP</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Request Resample?</td>
      <td>1=Yes;0=No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>APUSABLE</td>
      <td>N</td>
      <td>APOERES</td>
      <td>Sample Useable?</td>
      <td>1=Yes;0=No</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The two columns of interest here are categorical variables describing the two alleles of ApoE for each patient. The rest of data here is meta data about the patient visit or the sample.



```python
# record the columns of interest
apo_cols = ["APGEN1","APGEN2"]
```




```python
# check to ensure data types are int for categorical data
apo_df[apo_cols].dtypes
```





    APGEN1    int64
    APGEN2    int64
    dtype: object



## Biomarker Master Data

UPENN Biomarker master table contains abeta, tau, and ptau measurements taken from patient CSF. We can repeat the same process above for this table.



```python
describe_meta_data(csf_df)
```


    No phases listed
    Num Entries: 5876
    Num Columns: 14
    Num Patients: 1249
    Records per Patient: 2-26
    Phases spanned per patient: 0-0
    Patients w/ Duplicates: 1249
    



```python
# define and print terms from CSF biomarker master table
term_defs = define_terms(csf_df, csf_dict, "UPENNBIOMK_MASTER")
term_defs
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
      <th>FLDNAME</th>
      <th>TYPE</th>
      <th>TBLNAME</th>
      <th>TEXT</th>
      <th>CODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RID</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VISCODE</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BATCH</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Name of LONI table, corresponding to analytica...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KIT</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Reagents lot number</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>STDS</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Calibrators and Quality Controls lot number</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>RUNDATE</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Date of analytical run</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ABETA</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Result rescaled to UPENNBIOMK</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TAU</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Result rescaled to UPENNBIOMK</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PTAU</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Result rescaled to UPENNBIOMK</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ABETA_RAW</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Original (raw) result, before rescaling</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TAU_RAW</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Original (raw) result, before rescaling</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PTAU_RAW</td>
      <td>-4.0</td>
      <td>UPENNBIOMK_MASTER</td>
      <td>Original (raw) result, before rescaling</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



There is no phase information in this table. Looks like we won't be able to group the data by phase. Taking a quick glance at the features, the most interesting ones look like the re-scaled measurement of ABETA, TAU, and PTAU.



```python
# record columns for later use
csf_cols = ["ABETA","TAU","PTAU"]
```




```python
# check to ensure data type is float for continous variable
csf_df[csf_cols].dtypes
```





    ABETA    float64
    TAU      float64
    PTAU     float64
    dtype: object



## Laboratory Chemical Screenings

The lab master data set contains lab results from a variety of chemical tests performed on patient blood and urine.



```python
# define and print terms from lab data master table
describe_meta_data(lab_df)
```


    Phases:	 ['ADNI1' 'ADNIGO' 'ADNI2']
    Num Entries: 2463
    Num Columns: 131
    Num Patients: 2285
    Records per Patient: 1-3
    Phases spanned per patient: 1-1
    Patients w/ Duplicates: 171
    

As another example of disorganisation and non-standardization of ADNI data, the ADNI dictionary does not contain proper definitions of the lab tests. A separate dictionary of lab codes is provided, but the dictionary is not formatted like other ADNI dictionaries. So we can define another lookup function for the lab codes.



```python
# create a function to extract lab codes from the lab dict (has a different structure from other dictionaries)
def define_labcodes(df, dict_df):
    
    keys=["Test Code","Test Description"]
    term_dicts = []
    for col in df.columns:

        term_dict = dict.fromkeys(keys)
        loc = (dict_df["Test Code"] == col)
        
        if any(loc):
            tmp = dict_df.loc[loc][keys]

            for key in keys:
                if tmp[key].unique().shape[0]:
                    term_dict[key] = tmp[key].unique()[0]
                else:
                    term_dict[key] = float('nan')

            term_dicts.append(term_dict)
            #print("Name: {FLDNAME},\nType: {TYPE},\nTable: {TBLNAME},\nDesc: {TEXT},\nCode:{CODE}\n".format(**term_dict))
    
    data_dict = pd.DataFrame.from_dict(term_dicts).reindex(columns=keys)
    return(data_dict)
```




```python
# extract lab codes and descriptions of each test
lab_codes = define_labcodes(lab_df, lab_dict)
lab_codes.head(10)
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
      <th>Test Code</th>
      <th>Test Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AXT117</td>
      <td>Thyroid Stim. Hormone-QT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BAT126</td>
      <td>Vitamin B12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CMT1</td>
      <td>Color-QT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CMT10</td>
      <td>Urine Nitrite-QT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CMT11</td>
      <td>Leukocyte Esterase-QT</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CMT2</td>
      <td>Specific Gravity-QT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CMT3</td>
      <td>pH-QT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CMT43</td>
      <td>Blood (+)-QT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CMT49</td>
      <td>Urine Protein (3+)-QT</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CMT5</td>
      <td>Urine Glucose-QT</td>
    </tr>
  </tbody>
</table>
</div>



Since we have no apriori hypothesis about which of these lab measurements might be more interesting than others, let's keep measurements from all the tests.



```python
# record all lab codes and use them to extract lab code data from 
lab_cols = lab_codes["Test Code"]
```


Let's move on to missing data for this data set.

From the ADNI website:

>**Laboratory Data**: Screening clinical lab results (i.e. urine, chemistry panel).
Data contains some character coding (i.e. SCC09: No specimen received ), and
they can be treated as missing data. (LABDATA.csv)

Keeping this in mind, we can define a function to replace all alpha-numeric entries with the standard missing value `NaN`. All the data is in a string format by default; so we need to determine which strings are numeric and which are alpha-numeric.



```python
# determine if a string contains any non numeric characters
def is_number(string: str):
    
    # define valid numeric characters 
    # (including decimal and negative sign)
    valid_chars = set(str(np.arange(0,10,1))[1:-1] + '.-')
    is_num = not bool(set(string)-valid_chars)
    return(is_num)
```




```python
# find columns of lab df with strings
str_cols = lab_df[lab_cols].dtypes == object
str_cols = lab_cols[str_cols.values]

# define anonymous function to replace missing data with NaN
str_isnumber = lab_df[str_cols].apply(lambda x: x.apply(is_number))

# convert values with strings to missing val (-1)
str_vals = lab_df[str_cols].values
str_vals[~str_isnumber] = '-1'
num_vals = str_vals.astype(float)

# store new numeric values in dataframe
lab_df[str_cols] = num_vals

# convert missing values to nan
lab_df = lab_df.replace(to_replace=-1, value=np.nan)

# look for columns where all values are missing
# and remove them from the list of columns
all_missing_cols = str_cols[(num_vals==-1).all(0)]
lab_cols = list(set(lab_cols) - set(all_missing_cols))
```




```python
# check to make sure all of our lab test columns are numeric
lab_df[lab_cols].dtypes.unique()
```





    array([dtype('float64')], dtype=object)



## Save the data to file

Before moving on to additional analysis, we can save the dataframes with the missing values updated and the columns restricted to our columns of interest.



```python
# intialize dataframe list and empty placeholder
all_dfs = [apo_df, csf_df, lab_df]
all_df_cols = [apo_cols, csf_cols, lab_cols]
df_names = ["apoe","csf","lab"]

# iterate over dataframes
for i,df in enumerate(all_dfs):
    
    # ensure standardized missing value
    df.replace({np.nan:-1, -4:-1}, inplace=True)
    
    # ensure RID is in column list for indexing
    cols = all_df_cols[i]
    cols = append_meta_cols(df.columns, cols)
    
    # write data to csv
    to_write = df[cols]
    to_write.to_csv("../data/Cleaned/" + df_names[i] + "_clean.csv")
```

