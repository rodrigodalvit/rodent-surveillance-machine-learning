# Machine Learning for Rodent Surveillance Using Tracking Plates

This repository contains code and documentation associated with the article:

**Souza, F. N., Awoniyi, A. M., da Silva, R. D. C., Nery Jr, N., Oliveira, M. V. M., Zeppelini, C. G., The, G. A. P., Hacker, K., Eyre, M. T., Argibay, H. D., Ko, A., Costa, F., & Khalil, H. (2025). _Incorporating Machine Learning Techniques to Enhance Rodent Surveillance in Marginalized Urban Communities_. Ecology and Evolution, 15(11), e72382. https://doi.org/10.1002/ece3.72382**

## Overview

Effective rodent surveillance is important for public health, disease ecology, and rodent-borne zoonoses control. Traditional rodent surveillance methods, including trapping and manual interpretation of tracking plates, can be time-consuming, labor-intensive, and dependent on specialized technical expertise.

This project applies machine-learning and image-processing techniques to evaluate rodent tracking plates and support more efficient rodent surveillance. The analytical workflow includes image preprocessing, thresholding, dimensionality reduction, feature extraction, and classification.

## Main methods

The workflow includes:

- RGB-to-grayscale image conversion;
- Otsu-based thresholding;
- binary image generation;
- extraction of regions of interest from tracking plate images;
- dimensionality reduction using Principal Component Analysis;
- Independent Component Analysis;
- Legendre Moments;
- k-nearest neighbors classification;
- comparison with conventional human-interpreted tracking plates.

## Repository purpose

This repository is intended to support reproducibility and transparency by organizing the code, data-access instructions, and analysis workflow associated with the published study.

The official archived data and code are available on Zenodo:

**Zenodo record:** https://zenodo.org/records/17256286  
**DOI:** https://doi.org/10.5281/zenodo.17256286

## Suggested workflow

1. Clone this repository.

```bash
git clone https://github.com/YOUR-USERNAME/rodent-surveillance-machine-learning.git
cd rodent-surveillance-machine-learning
