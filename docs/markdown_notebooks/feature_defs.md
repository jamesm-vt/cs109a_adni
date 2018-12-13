title: 
notebook: 
section: 
subsection: 

## Contents
{:.no_toc}
*  
{: toc}



```python
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from ADNI_utilities import define_terms, describe_meta_data, paths_with_ext, append_meta_cols
```




```python
adni_dict_df = pd.read_csv("../data/Study Info/DATADIC.csv")
merge_dict_df = pd.read_csv("../data/Study Info/ADNIMERGE_DICT.csv")
lab_dict = pd.read_csv("../data/Biomarker Data/LABDATA_DICT.csv")
```




```python
# get the feature importance
paths = paths_with_ext("../data/Models/Feature_Importance/")
fnames = []
top_features = []
importance = np.tile(np.nan, (3,176))
i=0
for path in paths:
    
    if "modeled_upto_50" in path and "Log" not in path:
        df=pd.read_csv(path)
        df = df.drop(columns="Unnamed: 0",axis=1)
        mean_importance = df.mean().values
        mean_importance = mean_importance/mean_importance.sum()
        importance[i,:] = mean_importance
        sorted_feats = df.columns[np.argsort(mean_importance)[::-1]]
        top_features.append(sorted_feats[:100])
        fnames.append(os.path.basename(path).split(".")[0])
        i+=1
        
importance_df = pd.DataFrame(data=importance, columns=df.columns, index=fnames)
```




```python
# find common features
common_features = top_features[0]
for feats in top_features:
    common_features = list(set(common_features) & set(feats))
```




```python
# find common features
common_features = top_features[0]
for feats in top_features:
    common_features = list(set(common_features) & set(feats))
```




```python
imp = importance_df.loc["RandomForestClassifier_modeled_upto_50",common_features]
sorted_idx = np.argsort(imp.values)[::-1]
sorted_imp = imp.values[sorted_idx]
sorted_feats = [common_features[i] for i in sorted_idx]
```




```python
df = pd.DataFrame(columns=sorted_feats, index=[0])
terms = define_terms(df, adni_dict_df)
features = df.columns
terms["FLDNAME"] = df.columns

# add in definitions from adnimerge dict
merge_terms = define_terms(df, merge_dict_df)
is_merge = ~merge_terms.FLDNAME.isna()
terms.loc[is_merge,"TEXT"] = merge_terms.loc[is_merge,"TEXT"]
terms.loc[is_merge,"CRFNAME"] = merge_terms.loc[is_merge,"CRFNAME"]

# reverse column order
terms['FEATURE_IMPORTANCE'] = sorted_imp
cols = ['FLDNAME', 'FEATURE_IMPORTANCE', 'TEXT', 'CRFNAME']
terms = terms[cols]

```




```python
terms
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
      <th>FEATURE_IMPORTANCE</th>
      <th>TEXT</th>
      <th>CRFNAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CDRSB</td>
      <td>0.075535</td>
      <td>CDR-SB</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mPACCdigit</td>
      <td>0.059558</td>
      <td>ADNI modified Preclinical Alzheimer's Cognitiv...</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LDELTOTAL</td>
      <td>0.052368</td>
      <td>Logical Memory - Delayed Recall</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mPACCtrailsB</td>
      <td>0.049832</td>
      <td>ADNI modified Preclinical Alzheimer's Cognitiv...</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FAQ</td>
      <td>0.040017</td>
      <td>FAQ</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MOCA</td>
      <td>0.024542</td>
      <td>MOCA</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>6</th>
      <td>EcogSPTotal</td>
      <td>0.024169</td>
      <td>SP ECog - Total</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADASQ4</td>
      <td>0.023870</td>
      <td>ADAS Delayed Word Recall</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ADAS13</td>
      <td>0.022962</td>
      <td>ADAS 13</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FDG</td>
      <td>0.022923</td>
      <td>Average FDG-PET of angular, temporal, and post...</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EcogSPMem</td>
      <td>0.022311</td>
      <td>SP ECog - Mem</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AV45</td>
      <td>0.020883</td>
      <td>Reference region - florbetapir mean of whole c...</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ABETA</td>
      <td>0.020470</td>
      <td>CSF ABETA</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>13</th>
      <td>RAVLT_immediate</td>
      <td>0.019951</td>
      <td>RAVLT Immediate (sum of 5 trials)</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Hippocampus</td>
      <td>0.015370</td>
      <td>UCSF Hippocampus</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ADAS11</td>
      <td>0.013745</td>
      <td>ADAS 11</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MMSE</td>
      <td>0.013678</td>
      <td>MMSE</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>17</th>
      <td>RAVLT_perc_forgetting</td>
      <td>0.013278</td>
      <td>RAVLT Percent Forgetting</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MMSCORE</td>
      <td>0.011901</td>
      <td>MMSE TOTAL SCORE</td>
      <td>Mini Mental State Exam</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EcogSPLang</td>
      <td>0.011883</td>
      <td>SP ECog - Lang</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>20</th>
      <td>EcogSPOrgan</td>
      <td>0.011199</td>
      <td>SP ECog - Organ</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EcogPtTotal</td>
      <td>0.010944</td>
      <td>Pt ECog - Total</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>22</th>
      <td>EcogPtMem</td>
      <td>0.010600</td>
      <td>Pt ECog - Mem</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MidTemp</td>
      <td>0.010519</td>
      <td>UCSF Med Temp</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PTAU</td>
      <td>0.009133</td>
      <td>CSF PTAU</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>25</th>
      <td>EcogSPPlan</td>
      <td>0.009111</td>
      <td>SP ECog - Plan</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>26</th>
      <td>TAU</td>
      <td>0.008353</td>
      <td>CSF TAU</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fusiform</td>
      <td>0.008202</td>
      <td>UCSF Fusiform</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Entorhinal</td>
      <td>0.007759</td>
      <td>UCSF Entorhinal</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>29</th>
      <td>HMT9</td>
      <td>0.007744</td>
      <td>Test HMT9; Lymphocytes</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>RCT1408</td>
      <td>0.005689</td>
      <td>Test RCT1408; LDH</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>51</th>
      <td>RCT19</td>
      <td>0.005674</td>
      <td>Test RCT19; Triglycerides (GPO)</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>52</th>
      <td>HMT40</td>
      <td>0.005654</td>
      <td>Test HMT40; Hemoglobin</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>53</th>
      <td>HMT13</td>
      <td>0.005576</td>
      <td>Test HMT13; Platelets</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>54</th>
      <td>HMT10</td>
      <td>0.005553</td>
      <td>Test HMT10; Monocytes</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>55</th>
      <td>RCT11</td>
      <td>0.005444</td>
      <td>Test RCT11; Serum Glucose</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>56</th>
      <td>RCT5</td>
      <td>0.005417</td>
      <td>Test RCT5; AST (SGOT)</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>57</th>
      <td>EcogPtLang</td>
      <td>0.005396</td>
      <td>Pt ECog - Lang</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>58</th>
      <td>RCT183</td>
      <td>0.005094</td>
      <td>Test RCT183; Calcium (EDTA)</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>59</th>
      <td>RCT20</td>
      <td>0.005087</td>
      <td>Test RCT20; Cholesterol (High Performance)</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>60</th>
      <td>HMT2</td>
      <td>0.005050</td>
      <td>Test HMT2; Hematocrit</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>61</th>
      <td>RCT1</td>
      <td>0.005036</td>
      <td>Test RCT1; Total Bilirubin</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>62</th>
      <td>RCT13</td>
      <td>0.005013</td>
      <td>Test RCT13; Albumin</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>63</th>
      <td>RCT14</td>
      <td>0.004940</td>
      <td>Test RCT14; Creatine Kinase</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>64</th>
      <td>RCT4</td>
      <td>0.004929</td>
      <td>Test RCT4; ALT (SGPT)</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>65</th>
      <td>RCT3</td>
      <td>0.004912</td>
      <td>Test RCT3; GGT</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>66</th>
      <td>EcogPtOrgan</td>
      <td>0.004904</td>
      <td>Pt ECog - Organ</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>67</th>
      <td>HMT3</td>
      <td>0.004824</td>
      <td>Test HMT3; RBC</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>68</th>
      <td>RCT392</td>
      <td>0.004758</td>
      <td>Test RCT392; Creatinine (Rate Blanked)</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>69</th>
      <td>RCT12</td>
      <td>0.004603</td>
      <td>Test RCT12; Total Protein</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>70</th>
      <td>RCT9</td>
      <td>0.004523</td>
      <td>Test RCT9; Phosphorus</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>71</th>
      <td>HMT12</td>
      <td>0.004492</td>
      <td>Test HMT12; Basophils</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>72</th>
      <td>RCT6</td>
      <td>0.004475</td>
      <td>Test RCT6; Urea Nitrogen</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>73</th>
      <td>EcogPtDivatt</td>
      <td>0.004403</td>
      <td>Pt ECog - Div atten</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>74</th>
      <td>GDTOTAL</td>
      <td>0.003887</td>
      <td>Total Score</td>
      <td>Geriatric Depression Scale</td>
    </tr>
    <tr>
      <th>75</th>
      <td>PTEDUCAT</td>
      <td>0.003593</td>
      <td>Education</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>76</th>
      <td>RAVLT_forgetting</td>
      <td>0.003430</td>
      <td>RAVLT Forgetting (trial 5 - delayed)</td>
      <td>Key variables merged into one data table</td>
    </tr>
    <tr>
      <th>77</th>
      <td>FSVERSION_Cross-Sectional FreeSurfer (FreeSurf...</td>
      <td>0.002987</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78</th>
      <td>RCT29</td>
      <td>0.002679</td>
      <td>Test RCT29; Direct Bilirubin</td>
      <td>Laboratory Data</td>
    </tr>
    <tr>
      <th>79</th>
      <td>MH16SMOK_1.0</td>
      <td>0.001032</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 4 columns</p>
</div>





```python
terms.to_csv("../data/Results/top_feature_definitions.csv")
```

