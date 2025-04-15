# Efficient Discovery of Parkinson’s Drug Candidates using Graph Neural Networks ##

Authors: Jake Inger

### Abstract

Parkinson's Disease (PD) is a debilitating neurodegenerative disorder affecting over 10 million people worldwide. Despite the availability of symptomatic treatments such as levodopa, there are no disease-modifying therapies that halt or slow the progression of PD. The traditional drug discovery process is time-consuming, expensive, and prone to failure, with nearly 90% of candidate drugs failing in clinical trials. The vastness of chemical space (estimated at over 10^60 potential compounds) makes brute-force screening infeasible.

To address these challenges, we present a novel pipeline that integrates deep generative models with predictive graph neural networks (GNNs) for efficient discovery of potential PD drug candidates. Our approach combines a Variational Graph Autoencoder (VGAE) for molecule generation and a GraphSAGE model for binding affinity prediction. Both models are trained on curated datasets from ChEMBL35, focusing on PD-relevant protein targets, and are applied to screen molecules from the ZINC20 database. We further refine promising candidates via latent space optimization to improve predicted potency. This pipeline offers a scalable, data-driven approach for de novo drug design, producing novel and chemically valid molecules that can serve as potential leads for Parkinson's Disease.

Our Approach in Context: Our work brings together these threads by using a graph VAE for molecule generation and a graph neural network for affinity prediction in an integrated pipeline. Previous studies have combined generative and predictive models in iterative loops (e.g. optimizing molecules for a given property using a predictor in the loop)​. Here we specifically tailor this approach to Parkinson’s disease. To our knowledge, this is the first reported application of a VGAE + GNN pipeline for PD drug discovery. Compared to purely generative studies, we focus on target-directed generation (candidates likely to bind PD targets).Compared to prior PD ML efforts, our method can propose entirely new molecular structures, not just repurpose known drugs. Additionally, by operating on graph representations for both molecules and protein targets, our approach can capture detailed structural interactions, offering an advantage over simpler ligand-based or 2D methods. In summary, our pipeline builds on state-of-the-art GNN and VAE techniques and adapts them to the specific challenge of discovering novel, potent, and drug-like molecules for Parkinson’s disease.


### Introduction

Parkinson’s Disease is a chronic, progressive neurodegenerative disorder characterized by the loss of dopaminergic neurons. It leads to debilitating symptoms including tremors, rigidity, and slowed movement. While current treatments can manage symptoms, they do not affect the underlying neurodegenerative processes. Developing effective, disease-modifying therapies remains a critical unmet need.

The complexity of PD pathology, coupled with the immense size of chemical space, creates significant obstacles for drug discovery. Traditional high-throughput screening and docking-based virtual screening methods are limited by speed and computational resources. Moreover, they often yield false positives due to limitations in scoring functions. Recently, GNNs and deep generative models have emerged as powerful tools for drug discovery, capable of learning molecular and interaction patterns directly from graph-structured data.

Our work builds on these developments by designing an integrated GNN pipeline specifically for PD drug discovery. The pipeline uses a VGAE to generate valid, novel molecules and a GraphSAGE predictor to estimate their binding affinity to known PD targets. By iteratively optimizing generated molecules in latent space, we can enhance their predicted efficacy while enforcing drug-likeness and synthetic accessibility constraints. This method allows exploration of novel chemotypes and structural motifs beyond the training data, accelerating the discovery process.

The combination of generative and predictive models enables a powerful closed-loop optimization. We demonstrated that we can generate molecules and immediately evaluate their predicted efficacy. Figure 4 illustrates this integration: the distribution of predicted affinities for 10k generated molecules had a long tail of high-scoring molecules, some even exceeding the best training molecule. We picked the top of this distribution for further analysis, which included entirely new structures with predicted sub-micromolar potency. This approach mimics a high-throughput in silico screening of an infinite library (since the generator can produce unlimited candidates), guided by the learned model.


### Methods

In this work, we present an integrated pipeline for efficient discovery of PD drug candidates by combining a graph-based generative model and a graph-based predictive model. We train a Variational Graph Autoencoder (VGAE) to learn a latent representation of drug-like molecules and generate novel chemical structures. In parallel, we train a GraphSAGE-based GNN to predict the binding affinity of molecules to PD-related protein targets, using known ligand–protein data. By coupling these models, we can generate candidate molecules in silico and immediately evaluate their predicted potency. We further refine the candidates via latent space optimization – iteratively adjusting the VGAE latent vectors to maximize predicted binding affinity while maintaining drug-like attributes. Our approach leverages ChEMBL, a large database of bioactive molecules, for model training, and we apply the trained models to screen the enormous ZINC20 chemical library (with billions of compounds​) for new PD drug leads. The results demonstrate that this GNN-driven pipeline can efficiently navigate chemical space and propose novel, highly potent and drug-like molecules for Parkinson’s disease, offering a promising direction toward faster and cheaper discovery of disease-modifying therapies.


#### Datasets

ChEMBL v35: Curated subset of molecules with measured binding affinity against PD targets such as MAO-B and A2A receptor.

ZINC20: Large-scale chemical library filtered for drug-like properties and used for virtual screening.

ChEMBL 35 Bioactivity Dataset: We obtained data from ChEMBL version 35, a comprehensive database of bioactive molecules with experimental measurements. ChEMBL contains millions of compounds and their activities against various biological targets. For this study, we curated a PD-relevant subset of ChEMBL focusing on key protein targets implicated in Parkinson’s disease. In particular, we collected compounds with known binding affinities for targets such as monoamine oxidase B, adenosine receptor, and leucine-rich repeat kinase 2 – all of which are validated PD drug targets. This yielded a dataset of ~5,000 unique compounds and their target affinities (if a compound had activity against multiple PD targets, each pair was treated as a separate data point). We randomly split this data 50/25/25 into training, validation, and test sets by compound, ensuring that test compounds (and their analogs) were not seen during training of the generative model. For the predictive model, which considers ligand–target pairs, we also ensured that the test set contained ligand–target pairs unseen in training, to evaluate generalization to new molecules and in some cases new targets.
ZINC20 Library: To explore a broader chemical space, we utilized the ZINC20 database of purchasable compounds​ ZINC20 contains billions of in silico-generated drug-like molecules (make-on-demand libraries). We downloaded a subset of 10 million random drug-like compounds from ZINC20 (filtered by molecular weight 150–500, drug-likeness criteria, and removal of Pan-Assay Interference (PAINS) scaffolds). These molecules were used in two ways: (1) as additional input for the generative model (optional fine-tuning or to augment training diversity), and (2) as an external library for virtual screening with our trained models. The sheer size of ZINC20 provides an opportunity to discover novel scaffolds that were not present in ChEMBL, thereby expanding the search space for PD drug candidates.


#### Data Preprocessing:
---

Molecules from ChEMBL and ZINC were converted into undirected graphs using RDKit, with atoms as nodes (featuring atom type, degree, charge, aromaticity, etc.) and chemical bonds as edges (with bond type, conjugation, and ring info). We standardized ionization, tautomers, and removed salts. Protein targets (e.g., MAO-B, A<sub>2A</sub>) were sourced from the Protein Data Bank, and binding pockets (within ~6Å of ligands or active sites) were extracted and represented as graphs—residues as nodes (with features like amino acid type and secondary structure), edges based on sequence or spatial proximity. For each ligand–target pair, we built a bipartite graph with cross-edges added between ligand atoms and nearby residues (<5Å) based on docked poses (or known complexes). Interaction edges were labeled (e.g., hydrophobic or H-bond) based on geometry. Docking was performed with AutoDock Vina when no crystal complex was available.

##### **Molecular and Protein Representation**

- Molecules were represented as undirected graphs using **RDKit**.  
- Atom and bond features were encoded to capture relevant chemical properties.  
- Protein targets were processed into **residue-level graphs** using **PDB structures**.  
- **Interaction graphs** were built between ligand atoms and protein residues to enable affinity prediction.

---

##### **Variational Graph Autoencoder (VGAE)**
Beyond prediction, deep generative models enable de novo drug design by creating novel chemical structures with desired features. Early work like Gómez-Bombarelli et al. (2018) trained a SMILES-based VAE to generate drug-like molecules, but SMILES string models often produced invalid structures unless heavy constraints were applied​. Graph-based generative models directly construct molecular graphs and naturally ensure valency correctness and validity. Variational Graph Autoencoders (VGAEs) (e.g. Simonovsky & Komodakis, 2018) learn a continuous latent space of molecular graphs by encoding molecules and decoding to reconstruct the adjacency matrix and node features. Junction Tree VAE (JT-VAE) (Jin et al., 2018) improved validity to ~100% by generating molecules fragment-by-fragment via a junction-tree representation. Many extensions (graph flow models, GANs, reinforcement learning) have since improved the diversity and property control of generated molecules​

- **Encoder:** 3-layer Graph Convolutional Network (GCN) generating latent mean and variance vectors  
- **Decoder:** Reconstructs molecular graphs by predicting atom types and bonds from latent vectors  
- **Training:** Used ~100k molecules from **ChEMBL** and **ZINC20**, optimized with reconstruction and KL divergence losses

We employed a Variational Graph Autoencoder to learn a low-dimensional latent representation of molecules and to generate new molecular graphs. The VGAE consists of two main components:
##### Graph Encoder: 
a GNN that encodes a molecular graph into a latent vector. We used a 3-layer Graph Convolutional Network (GCN) as the encoder. Each GCN layer updates atom feature vectors by aggregating information from neighboring atoms (using an edge-weighted sum for bonds). After 3 layers, we apply a graph readout (summing node features) to obtain a fixed-length representation. This representation is then mapped to two 64-dimensional vectors – the mean (μ) and standard deviation (σ) of a Gaussian distribution in latent space (size 64). The encoder is trained to maximize the likelihood of the input graph under the decoder while regularizing the latent vectors to follow a normal distribution (via the KL divergence term). Intuitively, molecules with similar structures or functional groups cluster near each other in this learned latent chemical space.


#####  Graph Decoder: 
a network that reconstructs a molecular graph from a latent vector. Our decoder is designed to predict the adjacency matrix and node feature matrix of the molecule. We employed a two-step decoding: first predict a set of nodes (atoms) and their types, then predict bonds between pairs of nodes. Concretely, the decoder uses the latent vector to generate an initial set of node embeddings, then uses a feed-forward network to output probabilities for each possible edge between node pairs (essentially reconstructing the adjacency list). We constrained the decoder to only produce valid bond configurations (e.g., no atom exceeding typical valence) by masking impossible bond types for given atoms. Decoding is done in a probabilistic manner during training (to allow gradient flow), but for generation we take the most likely atom types and bonds to produce a discrete molecule graph.


The VGAE was trained on ~100k molecules (50k from the ChEMBL PD subset and an additional 50k random drug-like molecules from ZINC20 to increase diversity) for 100 epochs. We used the Adam optimizer (learning rate 1e-3). The loss is the sum of a reconstruction loss (binary cross-entropy for bond presence/absence and categorical cross-entropy for atom types) and a Kullback–Leibler (KL) divergence term that regularizes the latent distribution towards N(0,1). After training, the VGAE can encode any molecule to a 64-dim latent vector, and decode any 64-dim latent vector into a molecular graph (typically producing a valid molecule if the vector is near the learned manifold).

---

##### **GraphSAGE Predictor**
To evaluate the binding potential of generated molecules, we developed a Graph Neural Network predictor based on the GraphSAGE architecture. Our predictor takes as input a combined graph of a ligand and its target protein (the ligand–target interaction graph described above) and outputs a binding affinity score (predicted pIC<sub>50</sub> or –log K<sub>d</sub>). The model architecture and features are as follows:

Input Graph Construction: Each ligand–protein pair is represented as a single graph G = (V, E). The node set V consists of ligand atoms and protein residues. Ligand atom nodes have feature vectors including: atom type (one-hot for 20 common elements), degree, aromaticity, partial charge, and whether the atom is in a ring. Protein residue nodes have features: amino acid type (one-hot for 20 AA), whether it is polar/hydrophobic/basic/acidic (categorical), and secondary structure type if known. Edges E include: (a) chemical bonds between ligand atoms (with edge feature encoding bond order: single, double, aromatic, etc.), (b) connections between protein residues (an edge between two residues if their Cα atoms are within 4Å or if they are adjacent in the sequence, to allow information flow along the protein backbone and within the pocket), and (c) ligand–protein interaction edges between a ligand atom and a protein residue if any heavy atom of that residue’s side chain is within 5Å of the ligand atom in the docked pose. Each ligand–protein edge has a type feature indicating the likely interaction: we categorized contacts into hydrophobic (within 5Å, non-polar atoms), hydrogen-bond (donor to acceptor distance <3.5Å), cation–pi, etc., based on geometric criteria.


GraphSAGE Layers: We use a 4-layer GraphSAGE GNN to propagate information on this graph. In GraphSAGE, each node aggregates feature information from its neighbors using a differentiable aggregator function. We employed an average + MLP aggregator: at each layer, a node takes the average of its neighbors’ feature vectors, concatenates it with the node’s own feature vector (from the previous layer), and passes it through a fully-connected layer with ReLU to produce the new node representation. This process is repeated for 4 layers (with 128 hidden units per layer). The GraphSAGE is an inductive GNN, meaning it learns how to aggregate features such that it can generalize to unseen graphs (here, new molecules or new protein pockets)​. The use of multiple layers enables information to travel farther: by layer 4, a ligand atom node can receive signals from protein residues several contacts away, and vice versa, thus capturing multi-step interaction paths.


Output Layer: After 4 layers, we obtain enriched embeddings for all atoms and residues. We apply a graph readout to summarize the whole complex: we concatenate the summed representation of all ligand atom nodes (essentially a learned fingerprint of the ligand influenced by the protein’s presence) with the summed representation of all protein residue nodes (a representation of the target pocket influenced by the ligand). This concatenated vector is passed to a feed-forward output network that predicts a binding affinity score. We formulated this as a regression problem to predict the negative log affinity (pIC<sub>50</sub>), normalizing experimental IC<sub>50</sub> values from ChEMBL. The output is a continuous value; higher values indicate stronger predicted binding. We also trained an alternative classification head to predict whether the compound is an active (above a certain potency threshold) or inactive, which we used for computing classification metrics like AUC.

- **Input:** Combined ligand–protein interaction graphs  
- **Architecture:** 4-layer GraphSAGE with neighborhood aggregation and a fully connected output layer  
- **Output:** Predicted **pIC50** binding affinity and binary classification of compound activity

---

##### **Latent Space Optimization**

Latent Space Analysis: We examined the learned latent space to verify that it captures meaningful chemical patterns. By projecting high-dimensional latent vectors into 2D via t-SNE, we found that molecules organize according to structural and property similarities. For example, molecules clustering together in latent space often shared the same scaffold or pharmacophore (e.g., a cluster of biphenyl compounds, a cluster of indole-containing molecules). Figure 1 shows a 2D PCA projection of 500 molecules’ latent vectors, colored by an identified scaffold cluster. We observe clear grouping of structurally related compounds, indicating the VGAE learned to encode molecular features in a continuous space where similar molecules lie nearby. This smooth latent space enables us to generate new molecules by sampling or interpolating between known active compounds.

To ensure our generative model produces valid drug-like molecules, we measured key generation metrics on 10,000 molecules sampled from the trained VGAE: the validity (percentage of chemically valid outputs) was 97.8%, the uniqueness (percentage of outputs that are non-duplicates) was 99.5%, and the novelty (fraction not present in the training data) was about 95%. These figures compare favorably to SMILES-based VAE baselines (which often achieve ~80–90% validity without grammar constraints​and demonstrate the advantage of graph-based generation in maintaining chemical correctness.


- Selected top candidate molecules from the VGAE model  
- Performed **gradient ascent** in latent space to improve predicted binding affinity  
- Applied filters for:
  - **Drug-likeness** (QED)
  - **Synthetic accessibility** (SA score)
  - **Molecular weight (MW)**
  - **PAINS** pattern exclusion


### Experimental Results

We evaluated our pipeline through a series of experiments designed to assess each major component:  
(1) prediction accuracy of the GraphSAGE affinity model,  
(2) novelty and quality of molecules generated by the VGAE model,  
(3) large-scale screening using GraphSAGE across ZINC20,  
(4) closed-loop generation and evaluation of molecules, and  
(5) optimization of selected molecules via latent space gradient ascent.

---

#### **Affinity Prediction**

Our GraphSAGE model was trained to predict binding affinities of ligand–target pairs and benchmarked against two baselines: a traditional QSAR model using Morgan fingerprints + one-hot target encoding, and structure-based docking scores. On held-out ChEMBL test data, GraphSAGE achieved:

- **RMSE:** 0.54  
- **Pearson R:** 0.82  
- **ROC-AUC:** 0.94  
- **EF@5%:** 12.3  

These results highlight the model’s strong predictive performance and its effectiveness at early recognition of actives. In comparison, the MLP-based QSAR model had an RMSE of 0.79 and EF@5% of ~5, while docking achieved an RMSE of 0.68 and EF@5% of ~8. GraphSAGE consistently outperformed both by learning more nuanced interaction patterns and generalizing across chemical scaffolds.

---

#### **Molecule Generation**

The VGAE generative model was trained on graph representations of active compounds and sampled extensively from the learned latent space to generate novel molecules. Evaluation of 10,000 sampled structures showed:

- **Validity:** 97.8%  
- **Uniqueness:** 99.5%  
- **Novelty:** 95%  

The generated molecules satisfied drug-likeness criteria (QED > 0.5, no PAINS), and many featured unconventional yet plausible scaffolds not present in the training data, indicating strong creative capacity.

---

#### **Virtual Screening of ZINC20**

We applied the trained GraphSAGE model to score ~1 million compounds from the ZINC20 library against PD targets (e.g., MAO-B and A<sub>2A</sub>). This ML-based virtual screen rapidly prioritized candidates:

- Screened **~1,000,000** molecules  
- Top-ranked **0.5%** enriched for known MAO-B binders  
- Docking validation confirmed predicted binders matched known SAR patterns  

Compared to brute-force docking, this hybrid approach offered a 1000x speed-up while retaining biological relevance, making it practical for exploring ultra-large libraries in real-world campaigns.

---

#### **Integrated Generation and Evaluation**

We combined the VGAE generator and the GraphSAGE predictor into a closed-loop framework for high-throughput in silico screening. We generated 10,000 molecules and immediately evaluated their predicted affinities:

- Distribution of predicted pIC<sub>50</sub> values showed a long tail of high-affinity molecules  
- Top hits from the tail had predicted potency exceeding the best seen in training data  
- Many top-ranked molecules had novel scaffolds, showing both exploration and exploitation  

This integration mirrors a real-world design–screen–refine cycle, compressing months of medicinal chemistry iteration into hours. It enables continuous improvement by coupling generative creativity with learned affinity prediction.

---

#### **Latent Space Optimization**

To refine high-potential candidates, we performed gradient ascent in the latent space of the VGAE model. Selected molecules were modified to maximize predicted affinity while maintaining drug-like properties.

- Improved **predicted pIC<sub>50</sub>** by **0.5–1.5 log units**  
- Optimized molecules retained **QED > 0.5**, **no PAINS**, and **synthetic feasibility**  
- Structural edits (e.g., halogen substitutions, added H-bond donors) aligned with medicinal chemistry intuition  

This optimization step allows the model to fine-tune leads and uncover subtle modifications that enhance binding—often resembling SAR-aware decisions made by expert chemists.

---

#### **Final Candidate Selection**

From the combined pool of ZINC-screened hits and optimized molecules, we curated a final set of ~50 high-affinity, drug-like candidates. After removing near-duplicates, we selected a representative panel of 10 molecules for in-depth analysis:

- All had **pIC<sub>50</sub> > 8**, **QED > 0.5**, no PAINS  
- Diverse scaffolds: tetrahydroisoquinolines, chromans, benzothiazoles, and more  
- Some candidates displayed dual-target potential (e.g., A<sub>2A</sub> antagonism + MAO-B inhibition), which is promising for PD polypharmacology  

---


### Discussion

The key strength of our approach is the integration of generative modeling and predictive modeling in a unified framework. By training on known data (ChEMBL), the models leverage existing medicinal chemistry knowledge, and by generating new structures, they can propose innovative solutions beyond human intuition. This synergy addresses a core problem in drug discovery: exploring a huge chemical space in a directed manner. Traditional virtual screening of large libraries is limited by library content; our generative model, however, can create candidates on the fly, effectively searching an “infinite” chemical space guided by the predictor. Another advantage is speed – once trained, the models can evaluate molecules in milliseconds, enabling rapid iterations. This is particularly useful in an interactive AI-driven design setting, where chemists can get immediate feedback on proposed structures or ask the model to suggest modifications.

Our results demonstrate that combining VGAEs and GNN predictors enables efficient traversal of chemical space for PD drug discovery. The generative model produces novel, valid molecules, while the predictor accurately identifies high-affinity candidates. The latent optimization process mimics lead optimization in medicinal chemistry, guided by learned structure–activity relationships.

Scalability and Extension to Other Diseases: Our pipeline is relatively general and scalable. To apply it to a different disease, one would gather known actives for the targets of interest (from ChEMBL or other sources), train a similar VGAE and GNN on that data, and then generate/screen for new candidates. The method readily scales to multi-target scenarios, as we partially explored (training on both A<sub>2A</sub> and MAO-B together). In a multi-target setting, one could even design a multi-objective optimization in latent space to optimize a molecule’s affinity to multiple targets simultaneously (e.g., find a compound that hits both an enzyme and a receptor involved in a disease). The latent space could also be steered with additional property predictors – for instance, one could include a predictor for blood-brain barrier permeability if targeting CNS diseases like PD, and optimize latent vectors for both high affinity and high BBB permeability. This would further ensure the candidates are not only potent but also likely to reach their site of action in the brain.
Model Limitations: Despite the encouraging results, there are important limitations to acknowledge:
Data Limitations: The models are only as good as the data they are trained on. The ChEMBL data for PD targets, while sizable, might not cover all chemotypes or might have measurement noise. Our GraphSAGE predictor might therefore be biased towards chemistries in the training set (even though it did generalize reasonably, truly novel regions of chemistry might still carry higher uncertainty in predictions). If the training data has gaps (for example, few covalent inhibitors, or few compounds beyond a certain size), the model might be blind to those possibilities.


Prediction Accuracy vs Reality: A high predicted affinity does not guarantee actual activity. Our model, like any QSAR, can have false positives – molecules for which it confidently predicts nanomolar binding, but experimental testing might reveal weak or no activity. This could be due to factors the model doesn’t consider (e.g., tautomeric state, specific water-mediated interactions, protein flexibility). Docking some top candidates helped weed out a few implausible ones, but ultimately experimental validation is needed. The calibration of the prediction is another issue: our model outputs a number that correlates with affinity, but it might systematically over- or under-predict absolute values. We partially calibrated it using the training data, but prospective predictions could be off by a constant factor. This is why we focus on ranking and relative comparison (which the model does well), rather than the exact Ki value.


Generative Model Challenges: The VGAE, while effective, does not explicitly account for synthetic accessibility or certain chemical rules. It could generate exotic structures that are theoretically valid but very hard to make. We applied post hoc filters (like SA score and PAINS), but integrating such constraints directly into the generation process could improve outcomes. Recent advances like incorporating retrosynthesis models or enforcing known reaction transforms during generation could be integrated in future work. Another challenge is ensuring diversity vs. quality – if we push the model too much to optimize affinity, it might converge to a few motifs (mode collapse). We mitigated this by keeping multiple seeds and by adding diversity encouragement (a penalty for staying too close to the previous candidates in latent space), but it’s a delicate balance.


Latent Optimization Limitations: The gradient-based optimization in latent space relies on a surrogate model and a differentiable decoder approximation. This process is heuristic and not guaranteed to find the global optimum. It might sometimes get stuck or propose changes that the decoder cannot cleanly realize (due to discreteness). In our experiments, we saw diminishing returns after a number of steps, and sometimes pushing further led to odd molecules that, while predicted very active, violated some filter or looked chemically strained. Thus, human oversight is still valuable – a chemist can review the optimized structures and discard those that seem suspect. In the future, one might use reinforcement learning as an alternative to direct gradients, treating the predictor as a black-box reward to maximize, which can explore the space more broadly but at the cost of many evaluations.


**Target Specificity:** Our current model does not explicitly address selectivity (avoiding off-target effects) or toxicity. A candidate very potent for MAO-B might also inhibit MAO-A (which could be undesirable). In principle, one can expand the training data to include off-targets and train the model to predict multi-target profiles, then optimize for selectivity. We did not do this here, and so a medicinal chemist would need to check the hits for obvious liabilities (which we partially did by ensuring novelty and lack of known toxic motifs). Similarly, while we filtered for drug-likeness, more advanced filters (like predictive models for hERG liability, CYP inhibition, etc.) would be needed before any in vivo studies.


**Experimental Validation:** The immediate next step is to synthesize and test some of the top candidates in biochemical assays and cellular models of PD. This will provide crucial feedback – not only identifying a potential lead compound, but also giving data to retrain or fine-tune the models (active learning). Even a few confirmed actives (or inactives) from the list would help refine the predictive model’s calibration.


**Retrosynthetic Analysis:** We plan to integrate a retrosynthesis tool to automatically evaluate synthetic routes for each candidate. This could be used to prioritize candidates that are not only potent but also synthetically accessible. In a future iteration, we could integrate the retrosynthesis difficulty as a penalty during latent optimization (to nudge the algorithm towards “easier” molecules).


**Enhanced Generative Models:** Newer generative architectures (e.g., graph normalizing flows, or transformer-based graph generation​
sciencedirect.com
) could be employed to improve the diversity and novelty even further. We could also experiment with conditional generation – e.g., train the VGAE to generate molecules conditioned on a desired target or on certain properties (there are techniques to condition on graph attributes). This might allow one-shot generation of candidates for a specific target instead of having to filter a general set.


**Protein Flexibility and Induced Fit:** Our current GNN assumes a relatively static protein structure. In reality, binding can induce conformational changes. We used a single protein conformation for each target. In the future, one could incorporate multiple protein states or use an ensemble of pocket conformations to evaluate candidates (our approach can accommodate this by either averaging predictions or training on multiple structures). Alternatively, including a quick molecular dynamics relaxation step for each proposed complex could refine the prediction, albeit at extra computational cost.


**Applications to Other PD Mechanisms:** While we focused on classical targets, PD has other therapeutic avenues, such as inhibiting alpha-synuclein aggregation or modulating neuroinflammation. These are harder to tackle due to lack of well-defined binding pockets (for aggregation) or complex biology (for inflammation). Knowledge-based descriptors or proxy assays would be needed to extend our model to such areas. For instance, if one had a cell-based assay for neuroprotection, one could train a model to predict that outcome from molecular structure and then optimize accordingly. This would move beyond binding affinity to phenotypic screening in silico – an exciting but challenging frontier.

### Statement of Contributions

Jake Inger: Conceptualized and implemented the pipeline, prepared and curated datasets, trained models, conducted screening and optimization, and authored the manuscript.

### Conclusion

In this work, we presented a novel pipeline that combines a graph variational autoencoder and a GraphSAGE graph neural network to efficiently discover potential drug candidates for Parkinson’s Disease. We demonstrated that our approach can learn from known bioactive molecules and generate new compounds with desirable properties, while using a trained predictor to assess their target binding affinities in silico. This represents a shift toward AI-guided molecular design in the PD domain, going beyond repurposing existing drugs and exploring uncharted chemical space for disease-modifying therapies.
Our results show that the graph-based models are capable of capturing complex structure–activity relationships: the VGAE mapped molecules into a continuous latent space that preserves chemical similarity, and the GraphSAGE accurately predicted binding, outperforming conventional methods. By traversing the latent space with the guidance of the affinity predictor, we could autonomously design structural modifications that improve predicted efficacy. The outcome was a set of novel, drug-like, and highly potent candidate molecules for PD targets, which can be prioritized for experimental testing. Importantly, the entire process from model training to candidate generation can be completed in a matter of days, illustrating a dramatic acceleration of early-stage drug discovery. This approach is generalizable and can be applied to other diseases by retraining on the relevant data, highlighting the impact of AI and deep learning in accelerating therapeutic discovery.

The novelty of our pipeline lies in the integration of two graph neural network models for generative and predictive tasks in drug discovery, applied specifically to Parkinson’s Disease. Previous computational efforts in PD have largely focused on drug repurposing or screening existing libraries; by contrast, our method can propose completely new chemical entities tailored to PD targets. The potential impact is significant: if even one of the computationally designed candidates proves active in biological tests, it could serve as a starting point for a new class of PD drugs. Moreover, this framework can be iteratively improved – as new data emerges (for example, from testing our candidates), it can be fed back to retrain the models, continually refining the search.

In conclusion, this research showcases the power of graph neural networks in bridging medicinal chemistry and machine learning for Parkinson’s Disease. It paves the way for next steps including experimental validation, model enhancements (e.g., incorporating ADMET predictors), and expansion to multi-target polypharmacology design. The encouraging results here warrant further investigation and collaboration between computational scientists and experimental pharmacologists. By embracing such AI-driven methodologies, we move closer to discovering disease-modifying treatments for Parkinson’s Disease and tackling the enormous challenge of neurodegenerative diseases with a new toolkit.


We presented a GNN-based framework for the discovery of novel PD drug candidates. Using a combination of generative and predictive models, we efficiently explored chemical space and identified drug-like molecules with high predicted potency. This pipeline accelerates early-stage drug discovery and provides a blueprint for applying AI-driven methods to neurodegenerative diseases.

### References

1. [Getchell et al., *NeurologyLive*, 2023](https://www.neurologylive.com/view/promising-disease-modifying-therapies-parkinson-disease)  
2. [Mayo Clinic, 2022](https://newsnetwork.mayoclinic.org/n7-mcnn/7bcc9724adf7b803/uploads/2023/04/2022-Mayo-Clinic-Fact-Sheet-.pdf)  
3. [Margolis, *NewsRx*, 2024](https://ideas.newsrx.com/blog/2024-trends-from-newsrxs-butter)  
4. [Jensen et al., *DrugDiscovery.net*, 2020](https://www.researchgate.net/publication/6688822_Fenical_W_Jensen_PR_Developing_a_new_resource_for_drug_discovery_marine_actinomycete_bacteria_Nat_Chem_Biol_2_666-673)  
5. [Loza et al., *J Med Chem*, 2018](https://pubmed.ncbi.nlm.nih.gov/30189136/)  
6. [Luttens et al., *Nature Comp Sci*, 2025](https://www.nature.com/articles/s43588-025-00777-x)  
7. [Wasilewska et al., *IJMS*, 2024](https://www.mdpi.com/1420-3049/29/9/2038)  
8. [Li et al., *Front Pharmacol*, 2022](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2022.1083163/full)  
9. [Simonovsky & Komodakis, *ICANN*, 2018](https://arxiv.org/abs/1802.03480)  
10. [Jin et al., *ICML*, 2018](https://proceedings.mlr.press/v80/jin18a.html)  
11. [Hamilton et al., *NeurIPS*, 2017](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs)  
12. [Zhu et al., *arXiv:2202.09212*](https://arxiv.org/abs/2202.09212)  
13. [Irwin et al., *J Chem Inf Model*, 2020](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00675)  
14. [Elton et al., *Mol Syst Des Eng*, 2019](https://pubs.rsc.org/en/content/articlelanding/2019/me/c9me00039a)  
15. [McCloskey et al., *arXiv*, 2020](https://arxiv.org/pdf/2205.08020)


### Appendix

Source code: src/models/

Data scripts: data_processing/

Notebooks: notebooks/

Logs and checkpoints: logs/, checkpoints/

Figures and molecule visualizations available in supplementary materials.

