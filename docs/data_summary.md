---
title: ADNI Data Summary
notebook:
section: 1
subsection: 2
---

## Contents
{:.no_toc}
*  
{: toc}

## What is the ADNI data?

The Alzheimer’s Disease Neuroimaging Initiative (ADNI) is a global study that aims to understand factors leading to the Alzheimer’s Disease (AD) and to track disease progression. The study began in 2004 and is currently funded to continue through 2021. Many universities in the US and Canada have contributed to ADNI with data collected from nearly 3,000 patients across 63 clincal sites. ADNI has progressed through 4 phases (ADNI1, ADNI-GO, ADNI-2, and ADNI-3) with distinct research goals (Figure 1). Although the central focus of ADNI is to provide neuroanatomical imaging on patients over time, the study also includes comprehensive information from genetic screens, clinical exam results,patient history, and clinical diagnosis for Alzheimer's Disease. In total, the ADNI data is distributed over **(insert number)** raw data tables.

{:.center}
<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/adni_phase_summary.png" class="image">
    <figcaption style="text-align: left">
        Figure 1. Summary of the measures collected during each ADNI phase (from ADNI website).
    </figcaption>
</figure>

ADNI's data files have a longitudinal format with each observation corresponding to a clinical visit for a single patient. The diversity of measures recorded from patients during the study requires that that they are collected at different clinicians in separate locations on different dates. The end result is that the separate visits often record very different information. Patients periodically have diagnostic visits where they are assigned one of the three diagnostic categories: Cognitively Normal(CN), Mild Cognitive Impairment(MCI), and Alzheimer’s Disease(AD) (Figure 1). ADNI recruits patients on a rolling basis with patients joining and dropping out of the study at each ADNI phase. The initial patient set at the start of the study in ADNI-1 represents the single largest cohort of patients, with a comparable number of patients joining in ADNI-2. A small fraction (~20%) of patients change diagnoses over the course of the study.

{:.center}
<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/Patient_Demographics.svg" class="image">
    <figcaption style="text-align: left">
        Figure 2. ADNI contains longitudinal data collected over multiple phases across many patient visits.
    </figcaption>
</figure>

## Patient Demographics

In additional to the raw data, ADNI provides a data set (adnimerge) that summarizes the most commonly used measures captured throughout the study. This data set compiles 113 features from more than 13,000 visits and 2,080 patients. The data comes from roughly an equal distribution of male and female patients aged between 54 and 91 years old with a mean age of 73 years old. The education level of the patients follows a roughly equal distribution across the three diagnostic categories in the data. It is noteworthy that the distribution of race and ethnicities in the data, however, is largely skewed toward non-hispanic, white patients (Figure 3). 

{:.center}
<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/PatientDemographics.svg" class="image">
    <figcaption style="text-align: center">
        Figure 3. Distributions of ADNI patient demographics.
    </figcaption>
</figure>

## Data Missingness

Due to the length of the study, the diversity of measures recorded, and the cost of the procedures involved,it is hardly surprising that much of the is data missing. Roughly half of the measures are missing 50% or more of the observations (Figure 4). Unfortunately, missingness is so prevalent in the data set that dropping observations or features with missingness will leave us with nothing to model. Filling in missing values with some method of imputation is a critical.

{:.center}
<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/miss_by_feature.svg" class="image" style="width: 650px">
    <figcaption style="text-align: left">
        Figure 4. Histogram of percent missingness among the features present in the dataset.
    </figcaption>
</figure>

Of additional concern is the fact that the missingness of ADNI data is not distributed randomly. A look at the raw data in adnimerge shows large chunks of missing data along both rows and columns (Figure. 5). The structure of the missing data poses an additional challenges in replacing the missing data because it raises the possiblity bias in the remaining data. There are limits to how we can correct for any bias introduced by non-random missingness. So we will have to keep the possibility of bias in mind when interpreting our results. 

{:.center}
<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/adni_merge_missingness.png" class="image" style="width: 650px">
    <figcaption style="text-align: left">
        Figure 5. Missing data overview of the adnimerge dataset. White indicates missing values. Sparkline on the right represents data completeness by row and the least(52) and highest(108) number of features missing in each entry of the database.
    </figcaption>
</figure>


