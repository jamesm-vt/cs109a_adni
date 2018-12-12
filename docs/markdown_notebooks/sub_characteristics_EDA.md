---
title: Subject Characterestics
notebook: ..\markdown_notebooks\sub_characteristics_EDA.ipynb
section: 2
subsection: 7
---

## Contents
{:.no_toc}
*  
{: toc}


The purpose of this notebook is to explore and clean the subject characteristics data available in ADNI.The data in these files covers a broad range of patient information including but not limited to patient education, sex, and family history.

**Import libraries**



```python
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# import custom dependencies
import sys
from ADNI_utilities import define_terms, describe_meta_data, paths_with_ext, append_meta_cols
```




```python
# define figure defaults
mpl.rc('axes', labelsize=10, titlesize=14)
mpl.rc('figure', figsize=[6,4], titlesize=14)
mpl.rc('legend', fontsize=12)
mpl.rc('lines', linewidth=2, color='k')
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)
```


## Data meta analysis

There are multiple files to consider. Before looking in depth at specific data, it may be helpful to inspect some summary information on each data set. To start we can profile the following for each file:

- number of records
- number of patients
- number of duplicate entries per patient
- ADNI phases covered



```python
# import adni dictionary
adni_dict_df = pd.read_csv("../data/study info/DATADIC.csv")
```




```python
# print meta data for each file
csv_paths = paths_with_ext(directory="../data/Subject_Characteristics/")
for path in csv_paths:
    print("\n" + path + "\n")
    df = pd.read_csv(path, low_memory=False)
    describe_meta_data(df)
```


    
    ../data/Subject_Characteristics/FAMXHPAR.csv
    
    Phases:	 ['ADNI3']
    Num Entries: 587
    Num Columns: 19
    Num Patients: 587
    Records per Patient: 1-1
    Phases spanned per patient: 1-1
    Patients w/ Duplicates: 0
    
    ../data/Subject_Characteristics/FAMXHSIB.csv
    
    Phases:	 ['ADNI3']
    Num Entries: 540
    Num Columns: 17
    Num Patients: 540
    Records per Patient: 1-1
    Phases spanned per patient: 1-1
    Patients w/ Duplicates: 0
    
    ../data/Subject_Characteristics/FHQ.csv
    
    No phases listed
    Num Entries: 2952
    Num Columns: 15
    Num Patients: 2677
    Records per Patient: 1-3
    Phases spanned per patient: 0-0
    Patients w/ Duplicates: 2677
    
    ../data/Subject_Characteristics/PTDEMOG.csv
    
    Phases:	 ['ADNI1' 'ADNIGO' 'ADNI2' 'ADNI3']
    Num Entries: 4357
    Num Columns: 32
    Num Patients: 3582
    Records per Patient: 1-6
    Phases spanned per patient: 1-3
    Patients w/ Duplicates: 163
    

It looks like there is some information on family history which is split into a few different .files. ADNI1-ADNI2 are covered under `FHQ.csv` and ADNI3 family history is split into `FAMXHPAR` and `FAMXHSIB`. We'll start with patient demographics since that seems to cover all patients and ADNI phases.

## Patient Demographics

The patient demographics data set covers general information about the patients such has:

- Gender
- Date of birth
- Marital Status
- Handedness
- Primary language

Most of these features should fall under the category of predictors that we have no prior expectation will be linked to Alzheimer's.



```python
# intialize neuroexam results and describe entries
demo_df = pd.read_csv("../data/Subject_Characteristics/PTDEMOG.csv")

# create dictionary_df for NEUROEXM table
demo_dict = define_terms(demo_df, adni_dict_df, table_name="PTDEMOG");
demo_dict
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
      <td>PTDEMOG</td>
      <td>Record ID</td>
      <td>"crfname","Participant Demographic Information...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>Site ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>PTDEMOG</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VISCODE2</td>
      <td>T</td>
      <td>PTDEMOG</td>
      <td>Translated visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>PTDEMOG</td>
      <td>Date record created</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>PTDEMOG</td>
      <td>Date record last updated</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PTSOURCE</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>Information Source</td>
      <td>1=Participant Visit;2=Telephone Call</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PTGENDER</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>1. Participant Gender</td>
      <td>1=Male; 2=Female</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PTDOBMM</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>2a. Participant Month of Birth</td>
      <td>1..12</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PTDOBYY</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>2b. Participant Year of Birth</td>
      <td>1900..1980</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PTHAND</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>3. Participant Handedness</td>
      <td>1=Right;2=Left</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PTMARRY</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>4. Participant Marital Status</td>
      <td>1=Married; 2=Widowed; 3=Divorced; 4=Never marr...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PTEDUCAT</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>5. Participant Education</td>
      <td>0..20</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PTWORKHS</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>5a. Does the participant have a work history s...</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PTWORK</td>
      <td>T</td>
      <td>PTDEMOG</td>
      <td>6a. Primary occupation during most of adult life</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PTWRECNT</td>
      <td>T</td>
      <td>PTDEMOG</td>
      <td>6b. Most recent occupation</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PTNOTRT</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>7. Participant Retired?</td>
      <td>1=Yes;0=No</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PTRTYR</td>
      <td>D</td>
      <td>PTDEMOG</td>
      <td>Retirement Date</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PTHOME</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>8. Type of Participant residence</td>
      <td>1=House; 2=Condo/Co-op (owned); 3=Apartment (r...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PTOTHOME</td>
      <td>T</td>
      <td>PTDEMOG</td>
      <td>If Other, specify:</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PTTLANG</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>9. Language to be used for testing the Partici...</td>
      <td>1=English; 2=Spanish</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PTPLANG</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>10. Participant's Primary Language</td>
      <td>1=English; 2=Spanish; 3=Other (specify)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PTPSPEC</td>
      <td>T</td>
      <td>PTDEMOG</td>
      <td>If Other, specify:</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PTCOGBEG</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>11a. Year of onset of cognitive symptoms (best...</td>
      <td>9999=NA - Cognitively Normal;1985=1985;1986=19...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PTMCIBEG</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>11a. Year of onset of Mild Cognitive Impairmen...</td>
      <td>1985..2012</td>
    </tr>
    <tr>
      <th>27</th>
      <td>PTADBEG</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>11. Year of onset of Alzheimer's disease sympt...</td>
      <td>1985..2007</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PTADDX</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>11b. Year of Alzheimer's Disease diagnosis</td>
      <td>9999=NA - Not Diagnosed with AD;1985=1985;1986...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>PTETHCAT</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>12. Ethnic Category</td>
      <td>1=Hispanic or Latino; 2=Not Hispanic or Latino...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PTRACCAT</td>
      <td>N</td>
      <td>PTDEMOG</td>
      <td>13. Racial Categories</td>
      <td>1=American Indian or Alaskan Native; 2=Asian; ...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Many of the columns above are not likely to be useful because they are dependent on answers in other columns. Some of the columns are likely to be intractable (eg. `Primary Occupation`). We can invesigate the values of some of these further.



```python
# define columns to inspect unique values
cols = ["PTWORK","PTHOME","PTEDUCAT","PTMARRY","PTMCIBEG"]
[print("{0}:\n{1}\n".format(col,demo_df[col].unique())) for col in cols];

```


    PTWORK:
    ['computer progammer' '-4' 'chemical engineer' ... 'analyst network'
     'Cardiac cath lab tech' 'manager (technical)']
    
    PTHOME:
    [ 1.  3.  2.  6.  8.  5.  4. -4.  7. nan]
    
    PTEDUCAT:
    [16. 18. 10. 13. 12.  9. 14. 20. 17.  3. 19. 15.  8. 11.  4.  7.  6. -1.
     -4. nan  5.]
    
    PTMARRY:
    [ 1.  2.  3.  4.  5. -4. nan]
    
    PTMCIBEG:
    [  nan 2006. 2005. 2009. 2008. 2007. 1997. 2000. 1996. 1999. 1985. 2010.
     1993. 2004. 1998. 2002. 1994. 2001. 2003. 1995. 1986. 1991. 1990.]
    
    

We can see from the summary above that `-1`, `-4`, and `NaN` are all used for missing values. We can also see that `PTWORK` has too many unique values to print, meaning that our feature space will explode if we include this category. `PTEDUCAT` is most likely a continuous feature either containing the total number of years of education or some kind of summary index. We can look at the histogram to see if it matches our expectation for number of years (ie. heavily weighted to 10+ with very few values below).



```python
# PTEDUCAT histogram
plt.hist(demo_df.PTEDUCAT.dropna())
plt.xlabel("Education")
plt.title("Patient Education Histogram");
```



![png](sub_characteristics_EDA_files/sub_characteristics_EDA_15_0.png)


This distribution seems to match our expectation for a distribution of number of years of education. Now we can define our columns of interest.



```python
# define columns to keep
demo_cols  = demo_df.columns[[9,11,12,13,14,18,22,29,30]]
demo_cols
```





    Index(['PTGENDER', 'PTDOBYY', 'PTHAND', 'PTMARRY', 'PTEDUCAT', 'PTNOTRT',
           'PTTLANG', 'PTETHCAT', 'PTRACCAT'],
          dtype='object')



Next we should inspect the data types and make sure the format of each column makes sense.



```python
# print data types for each col
demo_df[demo_cols].dtypes
```





    PTGENDER    float64
    PTDOBYY     float64
    PTHAND      float64
    PTMARRY     float64
    PTEDUCAT    float64
    PTNOTRT     float64
    PTTLANG     float64
    PTETHCAT    float64
    PTRACCAT    float64
    dtype: object



All of our columns are stored as floats, but most of them contain categorical variables and will need to be converted to int.



```python
# replace missing values with -1
demo_df.replace({np.nan:-1, -4:-1}, inplace=True)

# convert categorical columns to int
categoricals = demo_cols[[0,2,3,5,6,7,8]]
demo_df[categoricals] = demo_df[categoricals].astype(int)
demo_df[demo_cols].dtypes
```





    PTGENDER      int32
    PTDOBYY     float64
    PTHAND        int32
    PTMARRY       int32
    PTEDUCAT    float64
    PTNOTRT       int32
    PTTLANG       int32
    PTETHCAT      int32
    PTRACCAT      int32
    dtype: object



## Patient Family History

These files contain information about history of dementia in the patient's family. With the patient family history information spread across multiple files, it will be important to determine if the questions asked are consistent enough across phases to combine into a single family history table.



```python
# intialize family history results and describe entries
fhq_df = pd.read_csv("../data/Subject_Characteristics/FHQ.csv")

# create dictionary_df for NEUROEXM table
fhq_dict = define_terms(fhq_df, adni_dict_df, table_name="FHQ");
fhq_dict
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
      <td>FHQ</td>
      <td>Record ID</td>
      <td>"crfname","Family History Questionnaire","inde...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Site ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>FHQ</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>FHQ</td>
      <td>Date record created</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>FHQ</td>
      <td>Date record last updated</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>FHQSOURCE</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Information Souce</td>
      <td>1=Participant Visit;2=Telephone Call</td>
    </tr>
    <tr>
      <th>8</th>
      <td>FHQPROV</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Indicate below who provided the information co...</td>
      <td>1=Participant only; 2=Study Partner only; 3=Bo...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FHQMOM</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Dementia</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FHQMOMAD</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Alzheimer's Disease</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FHQDAD</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Dementia</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FHQDADAD</td>
      <td>N</td>
      <td>FHQ</td>
      <td>Alzheimer's Disease</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FHQSIB</td>
      <td>N</td>
      <td>FHQ</td>
      <td>3. Does the participant have any siblings?</td>
      <td>1=Yes; 0=No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The family history questionnaire used in ADNI1, ADNIGO, and ADNI2 contains only information about the prevalence of dementia and alzheimer's disease in parents. The only information collected about siblings is whether or not the participant has siblings, meaning that most participants will not have medical history information regarding their siblings. For that reason, we will focus on the parent questionnaire only from ADNI3.



```python
# intialize family history results and describe entries
parent_df = pd.read_csv("../data/Subject_Characteristics/FAMXHPAR.csv")

# create dictionary_df for NEUROEXM table
parent_dict = define_terms(parent_df, adni_dict_df, table_name="FAMHXPAR");
parent_dict
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
      <td>FAMHXPAR</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>FAMHXPAR</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>FAMHXPAR</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>FAMHXPAR</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MOTHALIVE</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>Is mother living?</td>
      <td>0=No;1=Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MOTHAGE</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>Current age or age at death?</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MOTHDEM</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>Did/Does the biological mother have dementia?</td>
      <td>0=No;1=Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MOTHAD</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>If yes, did/does biological mother have Alzhei...</td>
      <td>0=No;1=Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MOTHSXAGE</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>At what approximate age did mother's symptoms ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FATHALIVE</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>Is father living?</td>
      <td>0=No;1=Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FATHAGE</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>Current age or age at death?</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FATHDEM</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>Did/Does the biological father have dementia?</td>
      <td>0=No;1=Yes</td>
    </tr>
    <tr>
      <th>15</th>
      <td>FATHAD</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>If yes, did/does biological father have Alzhei...</td>
      <td>0=No;1=Yes</td>
    </tr>
    <tr>
      <th>16</th>
      <td>FATHSXAGE</td>
      <td>N</td>
      <td>FAMHXPAR</td>
      <td>At what approximate age did father's symptoms ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>COMM</td>
      <td>T</td>
      <td>FAMHXPAR</td>
      <td>Comments</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The table above shows that information about the presence of dementia and alzheimer's is contained in both data sets. We will merge these measures along with patient `RID` into a single table.



```python
# concatenate data from both dataframes along rows
rid = np.hstack((fhq_df.RID.values,parent_df.RID.values))
viscode = np.hstack((fhq_df.VISCODE.values,parent_df.VISCODE.values))
momdem = np.hstack((fhq_df.FHQMOM.values,parent_df.MOTHDEM.values))
momad = np.hstack((fhq_df.FHQMOMAD.values,parent_df.MOTHAD.values))
daddem = np.hstack((fhq_df.FHQDAD.values,parent_df.FATHDEM.values))
dadad = np.hstack((fhq_df.FHQDADAD.values,parent_df.FATHAD.values))

# concatenate features along columns
fam_data = np.vstack((rid,viscode,momdem,momad,daddem,dadad)).T

# define a new dataframe
fam_df = pd.DataFrame(columns=["RID","VISCODE","MOMDEM","MOMAD","DADDEM","DADAD"], data=fam_data)
```




```python
# inspect header
fam_df.head()
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
      <th>RID</th>
      <th>VISCODE</th>
      <th>MOMDEM</th>
      <th>MOMAD</th>
      <th>DADDEM</th>
      <th>DADAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>sc</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>f</td>
      <td>0</td>
      <td>-4</td>
      <td>0</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>sc</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>sc</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>sc</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We can see that all the values above are stored as floats. Now we need to ensure that all data types are compatible with categorical data.



```python
# replace missing values with -1
fam_df.replace({np.nan:-1, -4:-1}, inplace=True)
int_cols = ["RID","MOMDEM","MOMAD","DADDEM","DADAD"]
fam_df[int_cols] = fam_df[int_cols].astype(int)

# record columns to keep
fam_cols = fam_df.columns
```


## Save cleaned subject characteristics to file

With the columns from each data set hand-picked, the appropriate data types selected, and the missingness standardized, we can write the new cleaned dataframes to file.



```python
# intialize dataframe list and empty placeholder
all_dfs = [demo_df, fam_df]
all_df_cols = [demo_cols, fam_cols]
df_names = ["demographics","famhist"]

# iterate over dataframes
for i,df in enumerate(all_dfs):
    
    # ensure RID is in column list for indexing
    cols = all_df_cols[i]
    cols = append_meta_cols(df.columns, cols)

    # write data to csv
    to_write = df[cols]
    to_write.to_csv("../data/Cleaned/" + df_names[i] + "_clean.csv")
```

