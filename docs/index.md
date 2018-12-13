---
title: Project Overview
notebook:
section: 1
subsection: 1
---

## Goal

The aim of this project is to model Alzheimer's Disease with the ultimate goal of predicting whether or not a patient has or will ever develop the disease from the diverse data available in the Alzheimer’s Disease Neuroimaging Initiative (ADNI). More specifically, we hope to answer the question:

> What are the key clinical, genetic, and biospecimen-related features that can predict whether an individual has developed or will develop Alzheimer’s Disease at later point in time?

## Resources

The data used for this project can all be found in the <a href="https://ida.loni.usc.edu/login.jsp?project=ADNI&page=HOME">ADNI database</a>. All code used for analysis and visualization can be found on the<a href="https://github.com/jamesm-vt/cs109a_adni">project github repository</a>. Most of the code and and step-by-step rationale is formatted in easily followable jupyter notebooks.

## Motivation

A 2015 report by the World Health Origanization (WHO) estimated 29.8 million people with Alzheimer's Disease. Between 2000 and 2016, Alzheimer's Disease and other forms of dementia rose from the 14th to 5th leading cause of death, killing nearly 2 million people globally in 2016 alone (Figure 1). Alzheimer's has increasingly become a focus of research due to its impact on human health. 

Since the onset of the study in 2004, the purpose of ADNI has provided a standard framework and set of experimental protocols for researchers at separate institutions to study ALzheimer's under collective guidelines. The ADNI data set appears in the analyses of more than 1000 publications. As such, other studies have well-characterized many of the key biological correlates of Alzheimer's Disease. We therefore took a more wholistic approach, attempting characterize the complete data set with an additional emphasis on lesser-known or weaker predictors of the disease.

<figure class="center_fig">
    <img src="/cs109a_adni/index_files/global_cause_of_death.jpg" class="image" style="width: 100%">
    <figcaption class="center_cap" style="text-align: center">
        Figure 1. Alzheimer's Disease is a leading cause of death in the world.
    </figcaption>
</figure>

## Expected Outcomes

Although much about Alzheimer's Disease is unknown, many clinical studies support amyloid hypothesis for Alzheimer's pathogenesis. According to this hypothesis, the disease is primarily caused by increase extracellular buildup of the peptide amyloid-A-beta (A\beta) in protein aggregates called senile plaques. In addition to A\beta buildup, increased levels of the phosophorylated isoform of the Tau protein (pTau) is thought to be linked to Alzheimer's by disrupting cellular transport in neurons. For these reasons, we expect measures of A\beta and pTau in the cerebrospinal fluid should be important indicators of the disease. 

In addition to these biomarkers, we expect other well-known correlates of Alzheimer's available in the ADNI data to provide predictive power. ADNI provides a wealth of neuroimaging measurements of metabolism, brain volume, and neuronal activity as well as cognitive batteries designed to assist in detection and diagnosesof Alzheimer's. We expect that many of these measures will be among the top predictors of Alzheimer's in our models. 

Although these features may drive most of the predictive power of our models, we also anticipate that the ADNI data is rich enough to generate models that allow us to identify many other significant risk factors of Azlheimer's Disease.

<figure class="center_fig">
    <img src="/cs109a_adni/index_files/ADNI_logo_vector.png" class="image" style="width: 25%; padding: 100px;">
</figure>



## Principal Challenges and Features

The scope of the ADNI study spans 14 years, 63 clincal sites and a diversity of measures including neuroimaging, cognitive tests, genetic screenings, chemical profiling, and patient history. The nature of the data and data collection presents unique challenges in modeling the disease. These challenges and our approaches to solving them are outlined here but are detailed in their respective sections. Briefly, we aimed to address the following in this project:

- **Data Cleaning** - The ADNI data is distributed throughout hundreds of raw data files with inconsistent formats. In the section, we: research the raw data, separate the relevant data from meta data, standardize data types for continuous and categorical data, convert all missing values to a standard indicator

- **Data Aggregation** - The ADNI data is organized in a longitudinal format with a single observation corresponding to a single visit for a particular patient. In this section, we: identify and compare methods for converting the data from a longitudinal format to a format with a single entry per patient.

- **Imputation** - Many of values are missing in the per patient data set above. In this section we compare methods for filtering and imputing the data to generate complete design matrices ready for modeling.

- **Modeling** - With the data cleaned, aggregated by patient, and imputed to replace missing values, we train and compare the performance of Random Forest and ADA Boost models in predicting Alzheimer's Disease diagnoses.

<div style="margin-top: 100px"></div>
