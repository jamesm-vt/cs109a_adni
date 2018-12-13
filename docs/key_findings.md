---
title: Key Findings
notebook:
section: 6
subsection: 1
---

## Goal

To summarize key predictors and findings of the project.

## Summary of Results

<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTiVdZaslqLlstuGXWtYX052rQRkclstZCyBB3tOKuiAb7CL3MY7KaeU8p9Hs3nxUFJ4mFzmJA0WA59/pubhtml?gid=1571893128&amp;single=true&amp;widget=true&amp;headers=false" style="width: 100%; height: 600px"></iframe>

## Summary of Model Comparison:

In addition to the most used ensemble methods, we have decided also to test some traditional classification algorithms to check if the more complex ensemble methods actually improved accuracy. We have ran those methods in 5 datasets built with different imputation strategies. The accuracy of each model on the test sets of each dataset is summarized on the table below:

<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/modelheatmap.png" style="width: 80%" class="image">
    <figcaption class="center_cap" style="text-align: center">
        Figure 1. Model comparison summary.
    </figcaption>
</figure>



It is easy to check that "non-ensemble" methods did perform worse than the ensemble methods - as expected. Among the different ensemble methods, Bagging (using Decision Trees) showed stronger results than Boosting and Random Forests. It is also interesting to notice that the combination of Bagging and model-based imputation has proven to be the best combination to accurately predict Alzheimer's Disease on our patient base.
More details about the deployment of each model can be found on the "Model Comparison" subsection.

## Selected Model:

<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/FeatureImportance_top80Features.svg" class="image">
    <figcaption class="center_cap" style="text-align: center">
        Figure 2. Top Features arranged by Feature Importance as assigned by Random Forest.
    </figcaption>
</figure>

## Most important features identified by our model:

The ensemble classification methods used by our group have identified features provided by ADNI that are capable of predicting AD. Among these are many cognitive tests and clinical test results that are well documented on the literature and support doctors diagnosing the disease. Many cognitive tests showed up as the most important predictors, a finding that is consistent to studies like (5).

In this section, we will present the key features that were selected by our models and that are consistent with findings reported on published literature regarding Alzheimer’s Disease.

#### Logical Memory Delayed Recall (LDELTOTAL)

In the Logical Memory Delayed Recall test, the patient is told a brief story and is requested to retell it just after hearing it. This test is reported by (1) as the top predictor of the Clinical Dementia Ratio (CDR), a variable that is very similar (almost a trivial predictor) to DX_FINAL the response variable chosen by our group as it represents the patient’s diagnostic in her most recent ADNI visit. It is relevant to highlight that (1) performed dimensionality reduction using both an algorithmic approach and specialists opinions. In both scenarios, LDELTOTAL appeared as a top predictor.


#### Functional Assessment Questionnaire (FAQ)

The FAQ is an assessment tool in which the patient is evaluated across 10 dimensions and receives a score ranging from 0 (no impairment) to 3 (severe impairment). The variable used on our model was the total FAQ score. Studies like (1) and (3) show evidence that this score is correlated with AD and (3) even points out which parts of the assessment are more correlated with risks of progressing from cognitive normality to mild cognitive impairment.


#### Montreal Cognitive Assessment (MoCA)

When physicians diagnose cognitive impairments, one possible tool to do so is the MoCA, which consists of a series of tasks that the physician asks the patient to perform. The patient’s performance is graded from 0 to 30, this grade was stored in the variable MOCA, and used as a predictor in our model. (4) showed that not only MOCA results are significant to correctly classify patients between CN, MCI, and AD but also that the results deteriorate over time.


#### Everyday Cognition by Study Partner (ECogSP) and Everyday Cognition by Patient (ECogPt)

During ADNI visits, Everyday Cognition questionnaires are applied both to the Study Partner (a relative or a caregiver) and to the patient herself. Features related to the responses of both questionnaires were identified as relevant predictors of cognitive status. (5) has also reached a similar conclusion.

#### Alzheimer's Disease Assessment Scale (ADAS)

ADAS is one of the most complete and widely-used assessments to diagnose cognitive impairments. ADNI data reports 15 measures related to ADAS and our model has selected two of them as features with relevant predictive power. This finding is consistent with studies such as (1), (4), and (5) that reach similar conclusions using various ADAS-related measures - including the ones selected by our model.

#### Fluoro-Deoxy-Glucose tracer (FDG)

The FDG variable provided by ADNI stores the average FDG tracer for different regions of interest in the human brain obtained through PET exams. Our models chose this variable as one of the top predictors of cognitive status. (6) found out that not only FDG is a indicator of AD, but also a very important tool in preliminary diagnostics of the disease. It is also noticeable that we have found studies like (2) that linked FDG to results obtained in various cognitive assessments like the ones described above.

#### Neuroimaging results

Many ADNI variables (e.g MidTemp, Ventricles, Fusiform) that store volumetric information about the patient’s brain were selected as predictors in our model. Despite the fact that cognitive tests are more important predictors, those variables still played an important role predicting cognitive status. This finding is similar to what was obtained by (5).


## Additional features identified by our model:

In addition to the anatomical and cognitive tests outlined above which could be considered more direct phenotypic features of the disease, our modeling also highlights other indirect biological pathways that have been linked to Alzheimer’s disease. Among the top features uncovered by all three of our ensemble methods are physiological measurements that may be the biological underpinnings of the underlying pathophysiology of the AD.

#### Inflammatory pathway

Numerous biochemical and neuropathological studies have shown a clear evidence for activation of inflammatory pathways in Alzheimer’s disease (7,8). Furthermore, long term use of anti-inflammatory drugs is also linked to a reduction in risk to Alzheimer’s disease(9, 10). In concordance with these findings, we found that a significant number of the top features identified by all of the three ensembl methods (namely percentage of lymphocytes, basophils, monocytes and neutrophils) point towards the activation of the inflammatory pathway.

#### Vitamin B-12

Also among our top 50 features found in all three models is the laboratory measurement of vitamin B12 levels in patients. A number of clinical studies have also investigated the correlation between the plasma vitamin B12 levels and cognitive impairment in AD (11, 12). These studies consistently conclude that vitamin B12 deficiency accelerates the progression of AD.



## Effects of imputations on selected features:

Our approach to imputation was to try both modeling and mean/mode based imputation. Our imputation strategy involved trying and tuning different types of models for both categorical and quantitative features. The other factor that we considered was the cut-off threshold for missing values (30%, 50%, 100%).

#### Imputation method

Our analysis showed that the imputation method (model vs mean) had little effect on the performance of the model. The model based imputation resulted in an average increase of about .005. Imputation method did not have a noticeable impact on feature importance.

#### Missingness threshold

We found the level of missingness that we accepted did have a slight impact on model performance. The more features we kept, regardless of their amount of missingness, the better the models performed. The difference was small but noticeable on both mean and model imputed datasets.

We found that neither imputation method or missingness threshold had a significant impact on feature importance. The most significant factor influencing feature selection was the choice of model.

<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/feat_import_model.png" class="image" style="width: 100%">
    <figcaption class="center_cap" style="text-align: center">
        Figure 3. Impact of model choice on feature importance.
    </figcaption>
</figure>


<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/missingness_Vs_Importance.svg" class="image">
    <figcaption class="center_cap" style="text-align: center">
        Figure 4. Percentage of data imputed among our top 80 common features across three ensemble models.
    </figcaption>
</figure>

#### References: 

1. Battista, P., Salvatore, C., & Castiglioni, I. (2017). Optimizing Neuropsychological Assessments for Cognitive, Behavioral, and Functional Impairment Classification: A Machine Learning Study. Behavioural neurology, 2017, 1850909.

2. Landau, Susan & Harvey, Danielle & Madison, Cindee & A Koeppe, Robert & M Reiman, Eric & L Foster, Norman & Weiner, Michael & J Jagust, William. (2009). Associations between cognitive, functional, and FDG-PET measures of decline in AD and MCI. Neurobiology of aging. 32. 1207-18. 10.1016/j.neurobiolaging.2009.07.002.

3. Marshall, G. A., Zoller, A. S., Lorius, N., Amariglio, R. E., Locascio, J. J., Johnson, K. A., Sperling, R. A., … Rentz, D. M. (2015). Functional Activities Questionnaire Items that Best Discriminate and Predict Progression from Clinically Normal to Mild Cognitive Impairment. Current Alzheimer research, 12(5), 493-502.

4. Chrem Méndez, Patricio & Calandri, Ismael & Nahas, Federico Exequiel & Russo, María & Demey, Ignacio & Eugenia Martín, María & Clarens, Florencia & Harris, Paula & Tapajoz, Fernanda & Campos, Jorge & I. Surace, Ezequiel & Martinetto, Horacio & Ventrice, Fernando & Cohen, Gabriela & Vazquez, Silvia & Romero, Carlos & Guinjoan, Salvador & Allegri, Ricardo & Sevlever, Gustavo. (2018). Argentina-Alzheimer’s disease neuroimaging initiative (Arg-ADNI): Neuropsychological evolution profile after one-year follow up. Arquivos de Neuro-Psiquiatria. 76. 231-240. 10.1590/0004-282x20180025.

5. Li, K., Chan, W., Doody, R. S., Quinn, J., Luo, S., Alzheimer’s Disease Neuroimaging Initiative (2017). Prediction of Conversion to Alzheimer's Disease with Longitudinal Measures and Time-To-Event Data. Journal of Alzheimer's disease : JAD, 58(2), 361-371.

6. Mosconi, L., Berti, V., Glodzik, L., Pupi, A., De Santi, S., & de Leon, M. J. (2010). Pre-clinical detection of Alzheimer's disease using FDG-PET, with or without amyloid imaging. Journal of Alzheimer's disease : JAD, 20(3), 843-54.

7. Wyss-Coray, T. & Rogers, J. Inflammation in Alzheimer disease-a brief review of the basic science and clinical literature. Cold Spring Harb Perspect Med 2, a006346, doi:10.1101/cshperspect.a006346 (2012).

8. Heppner, F. L., Ransohoff, R. M. & Becher, B. Immune attack: the role of inflammation in Alzheimer disease. Nat Rev Neurosci 16, 358-372, doi:10.1038/nrn3880 (2015).

9. Hull, M., Lieb, K. & Fiebich, B. L. Anti-inflammatory drugs: a hope for Alzheimer's disease? Expert Opin Investig Drugs 9, 671-683, doi:10.1517/13543784.9.4.671 (2000)

10. Thal, L. J. Anti-inflammatory drugs and Alzheimer's disease. Neurobiol Aging 21, 449-450; discussion 451-443 (2000).

11. Refsum, H. & Smith, A. D. Low vitamin B-12 status in confirmed Alzheimer's disease as revealed by serum holotranscobalamin. J Neurol Neurosurg Psychiatry 74, 959-961 (2003).

12. Stuerenburg, H. J., Mueller-Thomsen, T. & Methner, A. Vitamin B 12 plasma concentrations in Alzheimer disease. Neuro Endocrinol Lett 25, 176-177 (2004).
