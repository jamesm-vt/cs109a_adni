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
- most files contain substantial missingness
- files frequently contain duplicate observations per patient and visit code
- features are often missing or formatted differently in some ADNI phases
- multiple missing value indicators are used
- data types of measures are not standardized and are often inappropriate for the data
- measures are sometime scaled differently or recorded in different units

## Summary of Approach