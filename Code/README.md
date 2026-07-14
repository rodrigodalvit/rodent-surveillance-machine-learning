
---

## Improved `code/README.md`

This is the polished version of the README you already had.

```markdown
# Image Classification Pipeline Using Moment Features, PCA/ICA, and k-NN

This folder contains MATLAB scripts for the image-processing and machine-learning pipeline used to classify grayscale tracking-plate images.

The workflow combines image preprocessing, feature extraction using Legendre Moments, PCA and ICA, and classification using the k-nearest neighbors algorithm.

## Pipeline overview

The pipeline has five main stages.

---

## 1. Image preprocessing

**Script:** `image_preprocessing.m`

### Purpose

This script prepares the grayscale image data for feature extraction and classification.

Main steps:

- applies a global threshold to grayscale images;
- reduces intraclass variation in the image set;
- divides each image into 25 equal-sized sub-images using a 5 × 5 grid;
- saves each sub-image individually for later feature extraction.

### Main output

- preprocessed sub-images saved as `.jpg` files.

---

## 2. Feature extraction using Legendre Moments

**Scripts:**  
`extract_legendre_moments.m`  
`LM.m`

### Purpose

These scripts compute Legendre Moment features from the preprocessed image regions.

Main steps:

- computes Legendre Moments up to user-defined orders;
- extracts numerical feature vectors from each image or sub-image;
- appends class labels to the feature matrix;
- saves the resulting features in `.mat` files.

### Main output

- Legendre Moment feature matrices;
- feature vectors with associated class labels.

---

## 3. Feature extraction using PCA and ICA

**Script:** `pca_ica_feature_extraction.m`

### Purpose

This script applies dimensionality reduction and feature extraction using Principal Component Analysis and Independent Component Analysis.

Main steps:

- computes PCA-based image features;
- retains either all variance or 95% of variance, depending on the selected configuration;
- computes ICA-based features;
- extracts 100 independent components;
- saves PCA scores, ICA components, and explained variance information.

### Main output

- PCA feature matrices;
- ICA feature matrices;
- explained variance information;
- saved `.mat` files for downstream classification.

---

## 4. k-NN classification using varying numbers of features

**Script:** `knn_classification_features.m`

### Purpose

This script evaluates classification performance using different numbers of features.

Main steps:

- performs k-nearest neighbors classification;
- compares ICA-, PCA-, and Legendre Moment-derived features;
- iterates over different numbers of selected features;
- computes classification performance metrics.

### Metrics generated

- accuracy;
- sensitivity;
- specificity;
- classification loss;
- mean squared error.

### Main output

- performance tables;
- classification plots;
- figure corresponding to the feature-number analysis.

This script generates the results associated with **Figure 4** of the study.

---

## 5. k-NN classification using varying training sizes

**Script:** `knn_classification_trainingsize.m`

### Purpose

This script evaluates model performance across different training-set sizes.

Main steps:

- performs k-nearest neighbors classification using different proportions of training data;
- evaluates performance across training-size configurations;
- generates ROC curves at 80% training size;
- compares classification performance across feature-extraction methods.

### Metrics generated

- accuracy;
- sensitivity;
- specificity;
- classification loss;
- mean squared error;
- ROC curves.

### Main output

- performance tables;
- ROC curves;
- classification figures.

This script generates the results associated with **Figures 5 and 6** of the study.

---

## Suggested script order

Run the scripts in the following order:

```matlab
image_preprocessing
extract_legendre_moments
pca_ica_feature_extraction
knn_classification_features
knn_classification_trainingsize
