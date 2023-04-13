---
title: Imputation
notebook:
section: 4
subsection: 1
---

## Goal

The goal of this section is to explore methods to replace missing values in our per patient data sets and, ultimately, produce complete design matrices using the best method. Considering the substantial missingness of the ADNI data, we anticipate that this phase may be of critical importance for modeling downstream.

## Imputation method selection

We considered the following general strategies for imputing the data:

1. Drop rows and columns with missing values from the data
2. Replace missing values with the mean (continuous variables) or mode (categorical variables) of each column
3. Replace missing values by modeling each feature with missingness iteratively as a function of the features with no missingness 

A quick look at the data showed that too many values were missing for the first strategy to work. In additional to being unfeasible, this approach is unattractive for two reasons: 1) it reduces the statistical power of our models by decreasing the sampling 2) the face that our data is not missing at random puts us at risk of introducing bias in the remaining data set. Mean/mode imputation is simple and has low computational cost. Unlike the modeling-based apporach, this strategy also offers the benefit of being able to impute each feature independently. If the features are completely linearly independent, we can still reasonably impute the data. Model-based imputation offers the most promise of adding signal back to the data, but we must be able to reasonably model our features with one another in order to do so. This approach also poses a few practical and computational challenges

To assess whether we can expect model-based imputation to add signal to the data, we can characterize the prevalence of multi-colinearity in the data. A simple way of characterizing multi-colinearity in a holistic was is to compute the correlation matrix for the data set and visualize the distribution of correlation coefficients (Figure 1). To serve as a baseline for comparison, we can generate a null-model data set of the same size from randomly distributed data. Visualizing the data this way, we can see that our data is slightly more distributed around zero (no correlation) than the random data. There is some hope that model imputation may add some limited signal to the data. We ultimately decided to proceed with model-based imputation using mean/mode imputation as a basis for comparison.

<figure class="center_fig">
    <img src="/docs/imputation_overview_files/corrcoef_hist.svg" class="image" style="width: 100%">
    <figcaption class="center_cap">
        Figure 1. The features of the per patient data set are slightly more correlated to one another than expected by random chance.
    </figcaption>
</figure>

One concern of model-based imputation is that the various models we use to generate predictions for our missing values do not reflect the error in our models. To account for this, we add noise to our predictions by randomly sampling the residuals of our models.

To positively bias our method toward signal present in the unimputed data, we ranked the features by missingness from lowest to highest and iterated over them in the following way:

1. Select the highest ranked feature with any missingness.
2. Fit a model to the selected feature using all features with no missingness as predictors.
3. Replace missing values of the selected feature by generating predictions from the model.
4. Repeat steps 1-3 until no features with missingness remain.

With these design choices in place, the base model to use is the most important remaining choice. The choice of model is non-trivial and is considered in detail in the following notebook.

<div style="margin-top: 100px"></div>
