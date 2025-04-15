## Efficient Discovery of Parkinson’s Drug Candidates using Graph Neural Networks ##

Authors: Jake Inger

### Abstract

Parkinson's Disease (PD) is a debilitating neurodegenerative disorder affecting over 10 million people worldwide. Despite the availability of symptomatic treatments such as levodopa, there are no disease-modifying therapies that halt or slow the progression of PD. The traditional drug discovery process is time-consuming, expensive, and prone to failure, with nearly 90% of candidate drugs failing in clinical trials. The vastness of chemical space (estimated at over 10^60 potential compounds) makes brute-force screening infeasible.

To address these challenges, we present a novel pipeline that integrates deep generative models with predictive graph neural networks (GNNs) for efficient discovery of potential PD drug candidates. Our approach combines a Variational Graph Autoencoder (VGAE) for molecule generation and a GraphSAGE model for binding affinity prediction. Both models are trained on curated datasets from ChEMBL35, focusing on PD-relevant protein targets, and are applied to screen molecules from the ZINC20 database. We further refine promising candidates via latent space optimization to improve predicted potency. This pipeline offers a scalable, data-driven approach for de novo drug design, producing novel and chemically valid molecules that can serve as potential leads for Parkinson's Disease.

### Introduction

Parkinson’s Disease is a chronic, progressive neurodegenerative disorder characterized by the loss of dopaminergic neurons. It leads to debilitating symptoms including tremors, rigidity, and slowed movement. While current treatments can manage symptoms, they do not affect the underlying neurodegenerative processes. Developing effective, disease-modifying therapies remains a critical unmet need.

The complexity of PD pathology, coupled with the immense size of chemical space, creates significant obstacles for drug discovery. Traditional high-throughput screening and docking-based virtual screening methods are limited by speed and computational resources. Moreover, they often yield false positives due to limitations in scoring functions. Recently, GNNs and deep generative models have emerged as powerful tools for drug discovery, capable of learning molecular and interaction patterns directly from graph-structured data.

Our work builds on these developments by designing an integrated GNN pipeline specifically for PD drug discovery. The pipeline uses a VGAE to generate valid, novel molecules and a GraphSAGE predictor to estimate their binding affinity to known PD targets. By iteratively optimizing generated molecules in latent space, we can enhance their predicted efficacy while enforcing drug-likeness and synthetic accessibility constraints. This method allows exploration of novel chemotypes and structural motifs beyond the training data, accelerating the discovery process.

### Methods

#### Datasets

ChEMBL v35: Curated subset of molecules with measured binding affinity against PD targets such as MAO-B and A2A receptor.

ZINC20: Large-scale chemical library filtered for drug-like properties and used for virtual screening.

#### Data Preprocessing

Molecules were represented as undirected graphs using RDKit. Atom and bond features were encoded to capture chemical properties. Protein targets were processed into residue graphs from PDB structures. Interaction graphs between ligand atoms and protein residues were constructed for affinity prediction.

Variational Graph Autoencoder (VGAE)

Encoder: 3-layer Graph Convolutional Network outputs latent mean and variance vectors.

Decoder: Reconstructs molecular graph from latent vector by predicting atom types and bonds.

Training: Combined ChEMBL and ZINC20 data (~100k molecules) with reconstruction and KL divergence loss.

GraphSAGE Predictor

Input: Combined ligand-protein interaction graphs.

Architecture: 4-layer GraphSAGE with feature aggregation and fully connected output.

Output: Predicted pIC50 (binding affinity) and classification of activity.

Latent Space Optimization

Selected top VGAE-generated candidates.

Performed gradient ascent in latent space to increase predicted binding affinity.

Applied drug-likeness and synthetic filters (QED, MW, SA score, PAINS).

#### Experimental Results

Prediction Model

RMSE: 0.54

Pearson R: 0.82

ROC-AUC: 0.94

EF@5%: 12.3

GraphSAGE outperformed both docking scores and fingerprint-based QSAR baselines in predictive accuracy and virtual screening efficiency.

Generative Model

Validity: 97.8%

Uniqueness: 99.5%

Novelty: 95%

Generated molecules covered diverse scaffolds and met drug-likeness criteria.

Screening ZINC20

Screened 1M molecules with GraphSAGE.

Top 0.5% enriched in MAO-B binders.

Docking validated predicted high-affinity candidates.

Latent Optimization

Improved pIC50 by 0.5–1.5 units.

Retained drug-like properties post-optimization.

Final candidates included novel scaffolds with high synthetic accessibility.

### Discussion

Our results demonstrate that combining VGAEs and GNN predictors enables efficient traversal of chemical space for PD drug discovery. The generative model produces novel, valid molecules, while the predictor accurately identifies high-affinity candidates. The latent optimization process mimics lead optimization in medicinal chemistry, guided by learned structure–activity relationships.

Compared to traditional virtual screening, our method achieves faster candidate generation and prioritization. The approach is scalable and generalizable to other targets or disease areas. Limitations include reliance on training data diversity, decoder accuracy, and the lack of explicit modeling for off-target effects or ADMET properties.

Future work includes experimental testing of top candidates, multi-objective optimization (e.g., dual-target drugs), integration with retrosynthesis models, and expansion to other neurodegenerative diseases.

Statement of Contributions

Jake Inger: Conceptualized and implemented the pipeline, prepared and curated datasets, trained models, conducted screening and optimization, and authored the manuscript.

Conclusion

We presented a GNN-based framework for the discovery of novel PD drug candidates. Using a combination of generative and predictive models, we efficiently explored chemical space and identified drug-like molecules with high predicted potency. This pipeline accelerates early-stage drug discovery and provides a blueprint for applying AI-driven methods to neurodegenerative diseases.

References

Getchell et al., NeurologyLive, 2023

Mayo Clinic, 2022

Margolis, NewsRx, 2024

Jensen et al., DrugDiscovery.net, 2020

Loza et al., J Med Chem, 2018

Luttens et al., Nature Comp Sci, 2025

Wasilewska et al., IJMS, 2024

Li et al., Front Pharmacol, 2022

Simonovsky & Komodakis, ICANN, 2018

Jin et al., ICML, 2018

Hamilton et al., NeurIPS, 2017

Zhu et al., arXiv:2202.09212

Irwin et al., J Chem Inf Model, 2020

Elton et al., Mol Syst Des Eng, 2019

McCloskey et al., arXiv, 2020

Appendix

Source code: src/models/

Data scripts: data_processing/

Notebooks: notebooks/

Logs and checkpoints: logs/, checkpoints/

Figures and molecule visualizations available in supplementary materials.

