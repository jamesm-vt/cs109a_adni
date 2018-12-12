---
title: Data Cleaning
notebook:
section: 2
subsection: 1
---

## Contents
{:.no_toc}
*  
{: toc}

## Goal

The goal of this section is to construct a collection of longitudinal data tables with selected features of interest and standard format. The output of this phase of the project is a collection of CSV files that contain only the features we are interested in modeling and a two features of patient meta data: 1) the unique patient ID corresponding to each visit (RID) 2) the visit code that details when the visit occurred relative to baseline (VISCODE).

## Principal Challenges and Features

The scope of the ADNI study poses many challenges to building clean data sets. In this section, we hope to address the following concerns:

- the data is spread over hundreds of files
- most feature names are not human readable
- most files contain substantial missingness
- files frequently contain duplicate observations per patient and visit code
- features are often missing or formatted differently in some ADNI phases
- multiple missing value indicators are used
- data types of measures are not standardized and are often inappropriate for the data
- measures are sometime scaled differently or recorded in different units

## Summary of Approach

ADNI offers 244 data sets to date (Figure 1). The genetic and neuroimaging data sets are among the most expansive and most commonly studied data in ADNI. Due to time constraints and our interest in characterizing some of the less well-studied ADNI data, we narrowed our search by largely excluding the genetic and neuroimaging data. Even after excluding these data sets, we were left with more than 100 raw data files. Among these, we selected 14 data sets for both completeness (i.e. present in all ADNI phases) and relevance, with the number of features ranging from 10-150 per file. Considering that the number of patients (i.e. observations)is approximately 2,000, we aimed to construct a cleaned data set containing no more than 500 features.  Given that the data is spread over so many files, we implemented the following standard approach that could be repeatedly applied to each file of interest:

1. Use dictionaries provided by ADNI to generate human readable definitions for all features in the data set.
2. Based on those definitions, exclude features that meet any of the following criteria: 
    - patient meta data
    - file meta data
    - dependent on other features
    - described by or are duplicate with other features
3. Replace all missing values of the remaining features with a standard missing value indicator (-1).
4. Convert all categorical and continuous variables to integer and float data types, respectively.
5. Convert all observations of continuous variables to metric units (where applicable).
6. Append patient ID (RID) and visit code (VISCODE) meta data to the selected features.
7. Save the cleaned data to a CSV file.

<figure class="center_fig">
    <img src="/cs109a_adni/data_summary_files/data_files_table.png" class="image" style="width: 90%; margin-top: 20px">
    <figcaption class="center_cap">
        Table 1. Raw contents of the ADNI database.
    </figcaption>
</figure>

Many of the specific decisions made in hand-selecting features are not essential to understand the imputation and modeling results but are provided for detailed descriptions of how the eventual feature set was chosen. Of particular note is the  **data aggregation** method used to convert a variety of raw data tables in longitudinal format into a single per patient data set.

<div style="margin-top: 100px"></div>