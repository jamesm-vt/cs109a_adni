---
title: Cognitive Tests
notebook: ..\markdown_notebooks\neuropsych_scores_EDA.ipynb
section: 2
subsection: 6
---

## Contents
{:.no_toc}
*  
{: toc}


The purpose of this notebook is to explore and clean select data from the neuropsychological examinations available in ADNI.

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

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# import custom dependencies
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


Import the ADNI dictionary to get readable definitions of features.



```python
# import adni dictionary
adni_dict_df = pd.read_csv("../data/study info/DATADIC.csv")
```


## Geriatric Depression Scale

The Geriatric Depression scale is calculated from a battery of questioned designed to quantify a patient's depression.



```python
# intialize neuroexam results and describe entries
gds_df = pd.read_csv("../data/Neuropsychological/GDSCALE.csv")

# create dictionary_df for NEUROEXM table
gds_dict = define_terms(gds_df, adni_dict_df, table_name="GDSCALE");
gds_dict
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
      <td>GDSCALE</td>
      <td>Record ID</td>
      <td>"crfname","Geriatric Depression Scale","indexe...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>Site ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>GDSCALE</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VISCODE2</td>
      <td>T</td>
      <td>GDSCALE</td>
      <td>Translated visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>GDSCALE</td>
      <td>Date record created</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>GDSCALE</td>
      <td>Date record last updated</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EXAMDATE</td>
      <td>D</td>
      <td>GDSCALE</td>
      <td>Examination Date</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GDSOURCE</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>Information Source</td>
      <td>1=Participant Visit;2=Telephone Call</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GDUNABL</td>
      <td>T</td>
      <td>GDSCALE</td>
      <td>Check here if:</td>
      <td>1=Participant is unable to complete the GDS ba...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GDUNABSP</td>
      <td>T</td>
      <td>GDSCALE</td>
      <td>If unable, explain:</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GDSATIS</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>1. Are you basically satisfied with your life?</td>
      <td>1=Yes(0); 0=No(1)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GDDROP</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>2. Have you dropped many of your activities an...</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GDEMPTY</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>3. Do you feel that your life is empty?</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GDBORED</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>4. Do you often get bored?</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GDSPIRIT</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>5. Are you in good spirits most of the time?</td>
      <td>1=Yes(0); 0=No(1)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GDAFRAID</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>6. Are you afraid that something bad is going ...</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GDHAPPY</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>7. Do you feel happy most of the time?</td>
      <td>1=Yes(0); 0=No(1)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDHELP</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>8. Do you often feel helpless?</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GDHOME</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>9. Do you prefer to stay at home, rather than ...</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GDMEMORY</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>10. Do you feel you have more problems with me...</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>GDALIVE</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>11. Do you think its wonderful to be alive now?</td>
      <td>1=Yes(0); 0=No(1)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GDWORTH</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>12. Do you feel pretty worthless the way you a...</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GDENERGY</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>13. Do you feel full of energy?</td>
      <td>1=Yes(0); 0=No(1)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GDHOPE</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>14. Do you feel that your situation is hopeless?</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>GDBETTER</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>15. Do you think that most people are better o...</td>
      <td>1=Yes(1); 0=No(0)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>GDTOTAL</td>
      <td>N</td>
      <td>GDSCALE</td>
      <td>Total Score</td>
      <td>0..15</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Most of the features here are raw data answers to the questions. Well just take the `GDTOTAL` score for future use.



```python
# record columns
gds_cols = ["GDTOTAL"]

# standardize missingness and ensure float dtype
gds_df.replace({np.nan:-1, -4:-1}, inplace=True)
gds_df[gds_cols] = gds_df[gds_cols].astype(float)
```


## Mini-Mental State Exam



```python
# intialize neuroexam results and describe entries
mmse_df = pd.read_csv("../data/Neuropsychological/MMSE.csv", low_memory=False)

# create dictionary_df for NEUROEXM table
mmse_dict = define_terms(mmse_df, adni_dict_df, table_name="MMSE");
mmse_dict
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
      <td>MMSE</td>
      <td>Record ID</td>
      <td>"crfname","Mini Mental State Exam","indexes","...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Site ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VISCODE2</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Translated visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>MMSE</td>
      <td>Date record created</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>MMSE</td>
      <td>Date record last updated</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EXAMDATE</td>
      <td>D</td>
      <td>MMSE</td>
      <td>Examination Date</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MMDATE</td>
      <td>N</td>
      <td>MMSE</td>
      <td>1. What is today's date?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MMDATECM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What is today's date?--&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MMYEAR</td>
      <td>N</td>
      <td>MMSE</td>
      <td>2. What is the year?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MMYEARCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What is the year?--&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MMMONTH</td>
      <td>N</td>
      <td>MMSE</td>
      <td>3. What is the month?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MMMNTHCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What is the month?--&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MMDAY</td>
      <td>N</td>
      <td>MMSE</td>
      <td>4. What day of the week is today?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MMDAYCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What day of the week is ...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MMSEASON</td>
      <td>N</td>
      <td>MMSE</td>
      <td>5. What season is it?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MMSESNCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What season is it?--&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MMHOSPIT</td>
      <td>N</td>
      <td>MMSE</td>
      <td>6. What is the name of this hospital (clinic, ...</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MMHOSPCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What is the name of this...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>MMFLOOR</td>
      <td>N</td>
      <td>MMSE</td>
      <td>7. What floor are we on?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MMFLRCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What floor are we on?--&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MMCITY</td>
      <td>N</td>
      <td>MMSE</td>
      <td>8. What town or city are we in?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>24</th>
      <td>MMCITYCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What town or city are we...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MMAREA</td>
      <td>N</td>
      <td>MMSE</td>
      <td>9. What county (district, borough, area) are w...</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>26</th>
      <td>MMAREACM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What county (district, b...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>MMSTATE</td>
      <td>N</td>
      <td>MMSE</td>
      <td>10. What state are we in?</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MMSTCM</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Verbatim response &lt;!--What state are we in?--&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>MMRECALL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Which list was used?</td>
      <td>1=Standard (Ball, Flag, Tree);2=Alternate (App...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>MMBALLDL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>19. Ball</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>45</th>
      <td>MMFLAGDL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>20. Flag</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>46</th>
      <td>MMTREEDL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>21. Tree</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>47</th>
      <td>MMWATCH</td>
      <td>N</td>
      <td>MMSE</td>
      <td>22. Show the participant a wrist watch and ask...</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>48</th>
      <td>MMPENCIL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>23. Repeat for pencil.</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>49</th>
      <td>MMREPEAT</td>
      <td>N</td>
      <td>MMSE</td>
      <td>24. Say, "Repeat after me: no ifs, ands, or bu...</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>50</th>
      <td>MMHAND</td>
      <td>N</td>
      <td>MMSE</td>
      <td>25. Takes paper in right hand.</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>51</th>
      <td>MMFOLD</td>
      <td>N</td>
      <td>MMSE</td>
      <td>26. Folds paper in half.</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>52</th>
      <td>MMONFLR</td>
      <td>N</td>
      <td>MMSE</td>
      <td>27. Puts paper on floor.</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>53</th>
      <td>MMREAD</td>
      <td>N</td>
      <td>MMSE</td>
      <td>28. Present the piece of paper which reads, "C...</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>54</th>
      <td>MMWRITE</td>
      <td>N</td>
      <td>MMSE</td>
      <td>29. Give the participant a blank piece of pape...</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>55</th>
      <td>MMDRAW</td>
      <td>N</td>
      <td>MMSE</td>
      <td>30. Present the participant with the Construct...</td>
      <td>1=Correct; 2=Incorrect</td>
    </tr>
    <tr>
      <th>56</th>
      <td>MMSCORE</td>
      <td>N</td>
      <td>MMSE</td>
      <td>MMSE TOTAL SCORE</td>
      <td>0..30</td>
    </tr>
    <tr>
      <th>57</th>
      <td>DONE</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Was assessment/procedure done?</td>
      <td>0=No;1=Yes</td>
    </tr>
    <tr>
      <th>58</th>
      <td>MMLTR1</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Indicate 1st letter</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>59</th>
      <td>MMLTR2</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Indicate 2nd letter</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60</th>
      <td>MMLTR3</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Indicate 3rd letter</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>61</th>
      <td>MMLTR4</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Indicate 4th letter</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>62</th>
      <td>MMLTR5</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Indicate 5th letter</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>63</th>
      <td>MMLTR6</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Indicate 6th letter (if given)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>64</th>
      <td>MMLTR7</td>
      <td>T</td>
      <td>MMSE</td>
      <td>Indicate 7th letter (if given)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>65</th>
      <td>WORD1</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Ball (alt: Apple)</td>
      <td>0=0 - Incorrect;1=1 - Correct</td>
    </tr>
    <tr>
      <th>66</th>
      <td>WORD1DL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Ball (alt: Apple)</td>
      <td>0=0 - Incorrect;1=1 - Correct</td>
    </tr>
    <tr>
      <th>67</th>
      <td>WORD2</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Flag (alt: Penny)</td>
      <td>0=0 - Incorrect;1=1 - Correct</td>
    </tr>
    <tr>
      <th>68</th>
      <td>WORD2DL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Flag (alt: Penny)</td>
      <td>0=0 - Incorrect;1=1 - Correct</td>
    </tr>
    <tr>
      <th>69</th>
      <td>WORD3</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Tree (alt: Table)</td>
      <td>0=0 - Incorrect;1=1 - Correct</td>
    </tr>
    <tr>
      <th>70</th>
      <td>WORD3DL</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Tree (alt: Table)</td>
      <td>0=0 - Incorrect;1=1 - Correct</td>
    </tr>
    <tr>
      <th>71</th>
      <td>WORDLIST</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Which list was used?</td>
      <td>1=Ball, Flag, Tree;2=Apple, Penny, Table</td>
    </tr>
    <tr>
      <th>72</th>
      <td>WORLDSCORE</td>
      <td>N</td>
      <td>MMSE</td>
      <td>Score:  World Backwards</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>73</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 5 columns</p>
</div>



Most of the features here are raw data answers to the questions. Again we'll just save the total score `MMSCORE`.



```python
# record columns
mmse_cols = ["MMSCORE"]

# standardize missingness and ensure float dtype
mmse_df.replace({np.nan:-1, -4:-1}, inplace=True)
mmse_df[mmse_cols] = mmse_df[mmse_cols].astype(float)
```


## Modified Hachinski Ischemia Scale



```python
# intialize neuroexam results and describe entries
mhach_df = pd.read_csv("../data/Neuropsychological/MODHACH.csv", low_memory=False)

# create dictionary_df for NEUROEXM table
mhach_dict = define_terms(mhach_df, adni_dict_df, table_name="MODHACH");
mhach_dict
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
      <td>MODHACH</td>
      <td>Record ID</td>
      <td>"crfname","Modified Hachinski","indexes","adni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RID</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>Participant roster ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SITEID</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>Site ID</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VISCODE</td>
      <td>T</td>
      <td>MODHACH</td>
      <td>Visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>VISCODE2</td>
      <td>T</td>
      <td>MODHACH</td>
      <td>Translated visit code</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>USERDATE</td>
      <td>S</td>
      <td>MODHACH</td>
      <td>Date record created</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>USERDATE2</td>
      <td>S</td>
      <td>MODHACH</td>
      <td>Date record last updated</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EXAMDATE</td>
      <td>D</td>
      <td>MODHACH</td>
      <td>Examination Date</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HMONSET</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>1. Abrupt Onset of Dementia</td>
      <td>2=Present - 2 points; 0=Absent</td>
    </tr>
    <tr>
      <th>10</th>
      <td>HMSTEPWS</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>2. Stepwise Deterioration of Dementia</td>
      <td>1=Present - 1 point; 0=Absent</td>
    </tr>
    <tr>
      <th>11</th>
      <td>HMSOMATC</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>3. Somatic Complaints</td>
      <td>1=Present - 1 point; 0=Absent</td>
    </tr>
    <tr>
      <th>12</th>
      <td>HMEMOTIO</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>4. Emotional Incontinence</td>
      <td>1=Present - 1 point; 0=Absent</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HMHYPERT</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>5. History of Hypertension</td>
      <td>1=Present - 1 point; 0=Absent</td>
    </tr>
    <tr>
      <th>14</th>
      <td>HMSTROKE</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>6. History of Stroke</td>
      <td>2=Present - 2 points; 0=Absent</td>
    </tr>
    <tr>
      <th>15</th>
      <td>HMNEURSM</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>7. Focal Neurologic Symptoms</td>
      <td>2=Present - 2 points; 0=Absent</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HMNEURSG</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>8. Focal Neurologic Signs</td>
      <td>2=Present - 2 points; 0=Absent</td>
    </tr>
    <tr>
      <th>17</th>
      <td>HMSCORE</td>
      <td>N</td>
      <td>MODHACH</td>
      <td>TOTAL SCORE</td>
      <td>0..12</td>
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



Most of the features here are raw data answers to the questions. Again we'll just save the total score `HMSCORE`.



```python
# record columns
mhach_cols = ["HMSCORE"]

# standardize missingness and ensure float dtype
mhach_df.replace({np.nan:-1, -4:-1}, inplace=True)
mhach_df[mhach_cols] = mhach_df[mhach_cols].astype(float)
```


## Saving Data to File

With the columns from each data set hand-picked, the appropriate data types selected, and the missingness standardized, we can write the new cleaned dataframes to file.



```python
# intialize dataframe list and empty placeholder
all_dfs = [gds_df, mmse_df, mhach_df]
all_df_cols = [gds_cols, mmse_cols, mhach_cols]
df_names = ["depression","mmse","mhach"]

# iterate over dataframes
for i,df in enumerate(all_dfs):
    
    # ensure RID is in column list for indexing
    cols = all_df_cols[i]
    cols = append_meta_cols(df.columns, cols)
    
    # write data to csv
    to_write = df[cols]
    to_write.to_csv("../data/Cleaned/" + df_names[i] + "_clean.csv")
```

