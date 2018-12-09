---
title: ADNI Data Summary
notebook:
section: 1
subsection: 2
---

## Project Summary

The Alzheimer’s Disease Neuroimaging Initiative (ADNI) is a global study that aims to understand factors leading to the Alzheimer’s Disease (AD) and to track disease progression. The study began in 2004 and is currently funded to continue through 2021. Many universities in the US and Canada have contributed to ADNI with data collected from nearly 3,000 patients across 63 clincal sites. ADNI has progressed through 4 phases (ADNI1, ADNI-GO, ADNI-2, and ADNI-3) with distinct research goals. Although the central focus of ADNI is to provide neuroanatomical imaging on patients over time, the study also includes comprehensive information from genetic screens, clinical exam results,patient history, and clinical diagnosis for Alzheimer's Disease. In total, the ADNI data is distributed over **(insert number)** raw data tables.

ADNI's data files have a longitudinal format with each observation corresponding to a clinical visit for a single patient. Following each visit, patients are assigned one of the three diagnostic categories: Cognitively Normal(CN), Mild Cognitive Impairment(MCI), and Alzheimer’s Disease(AD). 


ADNI contains longitudinal data collected over multiple phases across many patient visits. 

ADNI provides a dataset (adnimerge) that summarizes the most commonly used measures captured throughout the study. This dataset compiles 113 features from more than 13,000 visits and 2,080 patients. The data comes from roughly an equal distribution of male and female patients aged between 54 and 91 years old with a mean age of 73 years old. The education level of the patients follows a roughly equal distribution across the three diagnostic categories in the data. It is noteworthy that the distribution of race and ethnicities in the data, however, is largely skewed towards white and non-hispanic populations (Figure 2). 


{:.center}
![png](index_files/ADNI_logo_vector.png)

## Principal Challenges and Features

The scope of the ADNI study spans 14 years, 63 clincal sites and a diversity of measures including neuroimaging, cognitive tests, genetic screenings, chemical profiling, and patient history. The nature of the data and data collection presents unique challenges in modeling the disease. These challenges and our approaches to solving them are outlined here but are detailed in their respective sections. Briefly, we aimed to address the following in this project:

- **Data Cleaning** - The ADNI data is distributed throughout hundreds of raw data files with inconsistent formats. In the section, we: research the raw data, separate the relevant data from meta data, standardize data types for continuous and categorical data, convert all missing values to a standard indicator

- **Data Aggregation** - The ADNI data is organized in a longitudinal format with a single observation corresponding to a single visit for a particular patient. In this section, we: identify and compare methods for converting the data from a longitudinal format to a format with a single entry per patient.

- **Imputation** - Many of values are missing in the per patient data set above. In this section we compare methods for filtering and imputing the data to generate complete design matrices ready for modeling.

- **Modeling** - With the data cleaned, aggregated by patient, and imputed to replace missing values, we train and compare the performance of Random Forest and ADA Boost models in predicting Alzheimer's Disease diagnoses.
