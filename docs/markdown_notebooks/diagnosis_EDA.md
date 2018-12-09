---
title: Diagnostic Response Variable EDA
notebook: ..\markdown_notebooks\diagnosis_EDA.ipynb
section: 2
subsection: 3
---

## Contents
{:.no_toc}
*  
{: toc}


(Zach Werkhoven)

## Import dependencies



```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
%matplotlib inline

# import custom dependencies
from ADNI_utilities import define_terms, describe_meta_data, append_meta_cols
```


## Diagnostic Summary Analysis

ADNI uses diagnostic codes to define patient dianosis for alzheimer's disease at each visit. Common to all ADNI phases are the following:

- Normal or `NL` or `1`
- Mild Cognitive Impairment or `MCI` or `2`
- Alzheimer's Disease or `AD` or `3`

However, some ADNI phases record additional diagnoses (eg. `LMCI` and `EMCI` for early and late mild cognitive impairment) and others only record change in diagnoses relative to the last visit. The purpose of this EDA is to try methods for constructing a single response variable with the format above for each visit. 



```python
# read in data
dx_df = pd.read_csv("../data/Diagnosis/DXSUM_PDXCONV_ADNIALL.csv")

# read in the ADNI dictionary and get summary of terms
adni_dict_df = pd.read_csv("../data/study info/DATADIC.csv")
dx_terms = define_terms(dx_df, adni_dict_df)
```




```python
# print diagnosis dataframe overview
describe_meta_data(dx_df)
```


    Phases:	 ['ADNI1' 'ADNIGO' 'ADNI2' 'ADNI3']
    Num Entries: 11264
    Num Columns: 53
    Num Patients: 2516
    Records per Patient: 1-15
    Phases spanned per patient: 1-4
    Patients w/ Duplicates: 2024
    



```python
# print the summary of terms
dx_terms
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
    </tr>
    <tr>
      <th>1</th>
      <td>ID</td>
      <td>N</td>
      <td>Record ID</td>
      <td>"crfname","","indexes","adni_aal_idx=TBLID,FLD...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>Site ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VISCODE2</td>
      <td>-4</td>
      <td>Translated visit code</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>Date record created</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>Date record last updated</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EXAMDATE</td>
      <td>D</td>
      <td>Examination Date</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DXCHANGE</td>
      <td>N</td>
      <td>1.  Which best describes the participant's cha...</td>
      <td>1=Stable: NL to NL; 2=Stable: MCI to MCI; 3=St...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>DXCURREN</td>
      <td>N</td>
      <td>1. Current Diagnosis</td>
      <td>1=NL;2=MCI;3=AD</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DXCONV</td>
      <td>N</td>
      <td>2. Has there been a conversion or reversion to...</td>
      <td>1=Yes - Conversion;2=Yes - Reversion; 0=No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>DXCONTYP</td>
      <td>N</td>
      <td>If YES - CONVERSION, choose type</td>
      <td>1=Normal Control to MCI; 2=Normal Control to A...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>DXREV</td>
      <td>N</td>
      <td>If YES - REVERSION, choose type</td>
      <td>1=MCI to Normal Control; 2=AD to MCI; 3=AD to ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>DXNORM</td>
      <td>T</td>
      <td>Normal</td>
      <td>1=Yes</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DXNODEP</td>
      <td>T</td>
      <td>Mild Depression</td>
      <td>1=Yes</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DXMCI</td>
      <td>T</td>
      <td>Mild Cognitive Impairment</td>
      <td>1=Yes</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DXMDES</td>
      <td>T</td>
      <td>If Mild Cognitive Impairment, select any that ...</td>
      <td>1=MCI (Memory features); 2=MCI (Non-memory fea...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DXMPTR1</td>
      <td>N</td>
      <td>1. Subjective memory complaint</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>19</th>
      <td>DXMPTR2</td>
      <td>N</td>
      <td>2. Informant memory complaint</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>20</th>
      <td>DXMPTR3</td>
      <td>N</td>
      <td>3. Normal general cognitive function</td>
      <td>1=Yes; 0=No; 2=Marginal</td>
    </tr>
    <tr>
      <th>21</th>
      <td>DXMPTR4</td>
      <td>N</td>
      <td>4. Normal activities of daily living</td>
      <td>1=Yes; 0=No; 2=Marginal</td>
    </tr>
    <tr>
      <th>22</th>
      <td>DXMPTR5</td>
      <td>N</td>
      <td>5. Objective memory impairment for age and edu...</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>23</th>
      <td>DXMPTR6</td>
      <td>N</td>
      <td>6. Not demented by diagnostic criteria</td>
      <td>1=Yes;0=No</td>
    </tr>
    <tr>
      <th>24</th>
      <td>DXMDUE</td>
      <td>N</td>
      <td>If MCI</td>
      <td>1=MCI due to Alzheimer's Disease; 2=MCI due to...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>DXMOTHET</td>
      <td>T</td>
      <td>If MCI due to other etiology, select box(es) t...</td>
      <td>1=Frontal Lobe Dementia; 2=Parkinson's Disease...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>DXMOTHSP</td>
      <td>T</td>
      <td>Other (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>DXDSEV</td>
      <td>N</td>
      <td>3a.  Dementia Severity - Clinician's Impression</td>
      <td>1=Mild; 2=Moderate; 3=Severe</td>
    </tr>
    <tr>
      <th>28</th>
      <td>DXDDUE</td>
      <td>N</td>
      <td>3b.  Suspected cause of dementia</td>
      <td>1=Dementia due to Alzheimer's Disease; 2=Demen...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>DXAD</td>
      <td>T</td>
      <td>Alzheimer's Disease</td>
      <td>1=Yes</td>
    </tr>
    <tr>
      <th>30</th>
      <td>DXADES</td>
      <td>N</td>
      <td>If Alzheimer's Disease, select box that indica...</td>
      <td>1=Mild; 2=Moderate; 3=Severe</td>
    </tr>
    <tr>
      <th>31</th>
      <td>DXAPP</td>
      <td>N</td>
      <td>If Alzheimer's Disease</td>
      <td>1=Probable; 2=Possible</td>
    </tr>
    <tr>
      <th>32</th>
      <td>DXAPROB</td>
      <td>T</td>
      <td>If Probable AD, select box(es) for other sympt...</td>
      <td>1=None;2=Stroke(s);3=Depression;4=Delirium;5=P...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>DXAMETASP</td>
      <td>T</td>
      <td>Metabolic/Toxic Disorder (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>DXAOTHRSP</td>
      <td>T</td>
      <td>Other (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>DXAPOSS</td>
      <td>T</td>
      <td>If Possible AD, select box(es) to indicate rea...</td>
      <td>1=Atypical clinical course or features (specif...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>DXAATYSP</td>
      <td>T</td>
      <td>Atypical clinical course or features (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>DXAMETSP</td>
      <td>T</td>
      <td>Metabolic / Toxic Disorder (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>DXAOTHSP</td>
      <td>T</td>
      <td>Other (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>DXPARK</td>
      <td>T</td>
      <td>Parkinsonism</td>
      <td>1=Yes</td>
    </tr>
    <tr>
      <th>40</th>
      <td>DXPARKSP</td>
      <td>T</td>
      <td>If yes, please describe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>DXPDES</td>
      <td>N</td>
      <td>If Parkinsonism, select box which indicates be...</td>
      <td>1=Parkinsonism without cognitive impairment; 2...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>DXPCOG</td>
      <td>N</td>
      <td>If Parkinsonism with cognitive impairment, dem...</td>
      <td>1=PD;2=PDD;3=DLB;4=PDAD</td>
    </tr>
    <tr>
      <th>43</th>
      <td>DXPATYP</td>
      <td>N</td>
      <td>If Atypical Parkinsonism</td>
      <td>1=PSP;2=CBGD;3=OPCA;4=SND;5=Shy Drager;6=Vascu...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>DXPOTHSP</td>
      <td>T</td>
      <td>Other (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45</th>
      <td>DXDEP</td>
      <td>N</td>
      <td>4a. Depressive symptoms present?</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>46</th>
      <td>DXDEPSP</td>
      <td>T</td>
      <td>If yes, please describe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>47</th>
      <td>DXOTHDEM</td>
      <td>T</td>
      <td>Other Dementia (not Alzheimer's Disease)</td>
      <td>1=Yes</td>
    </tr>
    <tr>
      <th>48</th>
      <td>DXODES</td>
      <td>N</td>
      <td>If Other Dementia, select box which indicates ...</td>
      <td>1=Frontal; 2=Huntington; 3=Alcohol; 4=NPH; 5=M...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>DXOOTHSP</td>
      <td>T</td>
      <td>Other (specify)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50</th>
      <td>DXCONFID</td>
      <td>N</td>
      <td>Physician Confidence in Diagnosis:</td>
      <td>1=Uncertain; 2=Mildly Confident; 3=Moderately ...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>DIAGNOSIS</td>
      <td>N</td>
      <td>Specify diagnostic category:</td>
      <td>1=Cognitively Normal; 5=Significant Memory Con...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We can see from the summary above that there are many diagnostic summaries which take the form of comments on or are conditional on previous categories. These will have to be removed.



```python
# specify non alzheimer's diagnosis columns that may be of interest
dx_cols = ["DXNORM","DXNODEP","DXMPTR1","DXMPTR2","DXMPTR3","DXMPTR4","DXMPTR5","DXPARK","DXDEP","DXOTHDEM"]
```


There seem to be three primary metrics that record alzheimer's diagnosis in the data set: `DXCURREN`, `DXCHANGE`, `DIAGNOSIS`. To determine how to split up the data, let's look at the values for each metric.



```python
# print the values for diagnosis metrics
print("DXCURREN values:\n")
print(dx_terms.loc[dx_terms.FLDNAME=="DXCURREN"].CODE.values)
print("\nDXCHANGE values:\n")
print(dx_terms.loc[dx_terms.FLDNAME=="DXCHANGE"].CODE.values)
print("\nDIAGNOSIS values:\n")
print(dx_terms.loc[dx_terms.FLDNAME=="DIAGNOSIS"].CODE.values)
```


    DXCURREN values:
    
    ['1=NL;2=MCI;3=AD']
    
    DXCHANGE values:
    
    ['1=Stable: NL to NL; 2=Stable: MCI to MCI; 3=Stable: Dementia to Dementia; 4=Conversion: NL to MCI; 5=Conversion: MCI to Dementia; 6=Conversion: NL to Dementia; 7=Reversion: MCI to NL; 8=Reversion: Dementia to MCI; 9=Reversion: Dementia to NL']
    
    DIAGNOSIS values:
    
    ["1=Cognitively Normal; 5=Significant Memory Concern;2=Early MCI; 3=Late MCI; 4=Alzheimer's Disease"]
    

By combining information from the metrics above, we should be able to to get a measure for each patient that falls into one of the three categories defined earlier: `NL`, `MCI`, and `AD`.

Let's see which metrics were recorded during each ADNI phase.



```python
# look at number of diagnostic values for each phase
by_phase = dx_df.groupby("Phase")
n_per_phase = by_phase[["DXCURREN","DXCHANGE","DIAGNOSIS"]].apply(lambda x: x.shape[0]-x.isna().sum())
n_per_phase
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
      <th>DXCURREN</th>
      <th>DXCHANGE</th>
      <th>DIAGNOSIS</th>
    </tr>
    <tr>
      <th>Phase</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADNI1</th>
      <td>3868</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ADNI2</th>
      <td>0</td>
      <td>5638</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ADNI3</th>
      <td>0</td>
      <td>0</td>
      <td>1281</td>
    </tr>
    <tr>
      <th>ADNIGO</th>
      <td>0</td>
      <td>475</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We can tell from the table above that we have different diagnostic summary categories for different phases of ADNI. We can make a more or less complete list of diagnoses by combining these three categories into one.

For the `DIAGNOSIS` metric, we'll use the following conversion rule:

- Normal = NL `1` or SMCI `5`
- MCI = EMCI `2` or LMCI `3`
- AD = AD `4`

The `DXCHANGE` metric diagnosis take the format `from _ to _`. We'll use a rule that just records the current diagnosis as follows:

- Normal = NL to NL `1`, MCI to NL `7`, or AD to NL `9`
- MCI = MCI to MCI `2`, NL to MCI `4`, or AD to MCI `8`
- AD = AD to AD `2`, MCI to AD `5`, or NL to AD `6`



```python
# convert DXCHANGE to DX, NL=(1,7,9), MCI=(2,4,8), AD=(3,5,6)
def combine_dx_measures(dxchange, dxcurr, diagnosis):
    
    # ensure arrays have proper dimensions
    dxchange = dxchange.reshape(-1,1)
    diagnosis = diagnosis.reshape(-1,1)
    
    # adjust DXCHANGE to NL, MCI, and AD
    NL = np.array([1,7,9]).reshape(1,3)
    MCI = np.array([2,4,8]).reshape(1,3)
    AD = np.array([3,5,6]).reshape(1,3)
    is_normal = (dxchange==NL).any(1)
    is_mildcog = (dxchange==MCI).any(1)
    is_alzh = (dxchange==AD).any(1)
    
    # insert into dx summary
    dx_sum = np.full(dxchange.shape,np.nan)
    dx_sum[is_normal]=1
    dx_sum[is_mildcog]=2
    dx_sum[is_alzh]=3
    
    # adjust DIAGNOSIS to NL, MCI, and AD
    NL = np.array([1,5]).reshape(1,2)
    MCI = np.array([2,3]).reshape(1,2)
    is_normal = (diagnosis==NL).any(1)
    is_mildcog = (diagnosis==MCI).any(1)
    is_alzh = diagnosis== 4
    
    # insert into dx summary
    dx_sum[is_normal]=1
    dx_sum[is_mildcog]=2
    dx_sum[is_alzh]=3
    
    # add in dxcurr 
    dx_sum[np.isnan(dx_sum)] = dxcurr[np.isnan(dx_sum).flatten()]
    
    return(dx_sum)
```




```python
# combine diagnostic values across ADNI phases and add to df
dx_comb = combine_dx_measures(dx_df.DXCHANGE.values, dx_df.DXCURREN.values, dx_df.DIAGNOSIS.values)
dx_df["DXCOMB"] = dx_comb

# append our new category to dx column list
dx_cols.append("DXCOMB")
```


Check the unique values in the new combined diagnostic metric `DXCOMB`



```python
# print unique vals
dx_df.DXCOMB.unique()
```





    array([ 1.,  3.,  2., nan])



Looking at the same table of number of entries per phase, we can see that all the original information has been captured by the new metric.



```python
# look at number of diagnostic values for each phase
by_phase = dx_df.groupby("Phase")
n_per_phase = by_phase[["DXCURREN","DXCHANGE","DIAGNOSIS","DXCOMB"]].apply(lambda x: x.shape[0]-x.isna().sum())
n_per_phase
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
      <th>DXCURREN</th>
      <th>DXCHANGE</th>
      <th>DIAGNOSIS</th>
      <th>DXCOMB</th>
    </tr>
    <tr>
      <th>Phase</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADNI1</th>
      <td>3868</td>
      <td>0</td>
      <td>0</td>
      <td>3868</td>
    </tr>
    <tr>
      <th>ADNI2</th>
      <td>0</td>
      <td>5638</td>
      <td>0</td>
      <td>5638</td>
    </tr>
    <tr>
      <th>ADNI3</th>
      <td>0</td>
      <td>0</td>
      <td>1281</td>
      <td>1281</td>
    </tr>
    <tr>
      <th>ADNIGO</th>
      <td>0</td>
      <td>475</td>
      <td>0</td>
      <td>475</td>
    </tr>
  </tbody>
</table>
</div>





```python
# ensure data types are all int for categorical data
dx_df[dx_cols].dtypes.unique()
```





    array([dtype('float64')], dtype=object)





```python
# ensure standardized missing value is compatible with int
dx_df.replace({np.nan:-1, -4:-1}, inplace=True)

# convert to int dtype
dx_df[dx_cols] = dx_df[dx_cols].astype(int)
```


## Output data to file

With the columns selected and the response variable constructed, we can output the data to file for later use.





```python
# ensure RID, VISCODE is in column list
dx_cols = append_meta_cols(dx_df.columns, dx_cols)
    
# write data to file
to_file_df = dx_df[dx_cols]
to_file_df.to_csv("../data/Cleaned/diagnosis_clean.csv")
```

