# Causal considerations can deterimine the utility of machine learning assisted GWAS

**Abstract:** Machine Learning (ML) is increasingly employed to generate health related traits (phenotypes) for genetic discovery, either by imputing existing phenotypes into larger cohorts or by creating novel phenotypes. While these ML-derived phenotypes can significantly increase sample size, and thereby empower genetic discovery, they can also inflate the false discovery rate (FDR). Recent research has focused on developing estimators that leverage both true and machine-learned phenotypes to properly control false positives. Our work complements these efforts by exploring how the true positive rate (TPR) and FDR depend on the causal relationships among the inputs to the ML model, the true phenotypes, and the environment.

Using a simulation-based framework, we study causal architectures in which the machine-learned proxy phenotype is derived from biomarkers (i.e. ML model input features) either causally upstream or downstream of the target phenotype (ML model output). We show that no inflation of the false discovery rate occurs when the proxy phenotype is generated from upstream biomarkers, but that false discoveries can occur when the proxy phenotype is generated from downstream biomarkers. Next, we show that power to detect genetic variants truly associated with the target trait depends on its genetic component and correlation with the proxy trait. However, the source of the correlation is key to evaluating a proxy phenotype’s utility for genetic discovery. We demonstrate that evaluating machine-learned proxy phenotypes using out-of-sample predictive performance (e.g. test $R^2$) provides a poor lens on utility. This is because overall predictive performance does not differentiate between genetic and environmental correlation. In addition to parsing these properties of machine-learned phenotypes via simulations, we further illustrate them using real-world data from the UK Biobank.

[Biorxiv Pre-print Link](https://www.biorxiv.org/content/10.1101/2024.12.16.628604v1)

## Simulation code
This repository contains the simulation code used in the pre-print. Each notebook contains the simulation code used in a sub-section of the **Results** section of the paper.

1. [Downstream vs. Upstream Biomarkers for imputation - which to use?](https://github.com/insitro/causal_considerations_ml_assisted_gwas/blob/main/downstream_vs_upstream_biomarkers.ipynb) 

2. [Recovery of true positive variants depends on target phenotype heritability and proxy phenotype correlation](https://github.com/insitro/causal_considerations_ml_assisted_gwas/blob/main/effect_of_env_and_heritability.ipynb)

3. [Why phenotypic correlation is a misleading indicator of utility for genetic discovery](https://github.com/insitro/causal_considerations_ml_assisted_gwas/blob/main/effects_env_ml_phenos.ipynb)

