# AMD Graphical Models

Please see our paper: ["A Graphical Method for Identifying Gene Clusters from RNA Sequencing Data"](https://arxiv.org/abs/2511.09590) for details.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The identification of disease-gene associations is instrumental in understanding the mechanisms of diseases and developing novel treatments. Besides identifying genes from RNA-Seq datasets, it is often necessary to identify gene clusters that have relationships with a disease. In this work, we propose a graph-based method for using an RNA-Seq dataset with known genes related to a disease and perform a robust clustering analysis to identify clusters of genes. Our method involves the construction of a gene co-expression network, followed by the computation of gene embeddings leveraging Node2Vec+, an algorithm applying weighted biased random walks and skipgram with negative sampling to compute node embeddings from undirected graphs with weighted edges. Finally, we perform spectral clustering to identify clusters of genes. All processes in our entire method are jointly optimized for stability, robustness, and optimality by applying Tree-structured Parzen Estimator. Our method was applied to an RNA-Seq dataset of known genes that have associations with Age-related Macular Degeneration (AMD). We also performed tests to validate and verify the robustness and statistical significance of our methods due to the stochastic nature of the involved processes. Our results show that our method is capable of generating consistent and robust clustering results. Our method can be seamlessly applied to other RNA-Seq datasets due to our process of joint optimization, ensuring the stability and optimality of the several steps in our method, including the construction of a gene co-expression network, computation of gene embeddings, and clustering of genes. Our work will aid in the discovery of natural structures in the RNA-Seq data, and understanding gene regulation and gene functions not just for AMD but for any disease in general. 

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/arkobarman/AMD_graphical_models.git
cd AMD_graphical_models
pip install -r requirements.txt
```

## Usage

Detailed usage instructions and examples can be found in the `pipeline.ipynb` file.
