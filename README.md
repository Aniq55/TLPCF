## Rethinking Evaluation for Temporal Link Prediction through Counterfactual Analysis

In response to critiques of existing evaluation methods for Temporal Link Prediction (TLP) models, we propose a novel approach to verify if these models truly capture temporal patterns in the data. Our method involves a sanity check formulated as a counterfactual question: 
> What if a TLP model is tested on a temporally distorted version of the data instead of the real data? 

Ideally, a TLP model that effectively learns temporal patterns should perform worse on temporally distorted data compared to real data. We provide an in-depth analysis of this hypothesis and introduce two data distortion techniques to assess well-known TLP models.

Our contributions are threefold: (1) We introduce simple techniques to distort temporal patterns within a graph, generating temporally distorted test splits of well-known datasets for sanity checks. These distortion methods are applicable to any temporal graph dataset. (2) We perform counterfactual analysis on TLP models such as `JODIE`, `TGAT`, `TGN`, `CAWN`, `GraphMixer` and `DyGFormer` to evaluate their capability in capturing temporal patterns across different datasets. (3) We propose an alternative evaluation strategy for TLP, addressing the limitations of binary classification and ranking methods, and introduce two metrics - average time difference (ATD) and average count difference (ACD) - to provide a comprehensive measure of a model's predictive performance. 

#### Data Distortion
To generate the `Intense` and `Shuffle` distorted samples, please run 
```
bash src/generate.sh
```

#### Source Code of the TLP Models:
- `JODIE` and `TGN` are available at: https://github.com/twitter-research/tgn
- `TGAT` is available at: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
- `CAWN` is available at: https://github.com/snap-stanford/CAW
- `GraphMixer` and `DyGFormer` are available at: https://github.com/yule-BUAA/DyGLib

For each model, the code can be run on the real and distorted test data by running:
```
bash run_all.sh
```