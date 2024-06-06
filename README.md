### Rethinking Evaluation Strategy for Temporal Link Prediction through Counterfactual Analysis
Aniq Ur Rahman, Alexander Modell, Justin P. Coon

In response to critiques of existing evaluation methods for Temporal Link Prediction (TLP) models, we propose a novel approach to verify if these models truly capture temporal patterns in the data. Our method involves a sanity check formulated as a counterfactual question: 
> What if a TLP model is tested on a temporally distorted version of the data instead of the real data? 

Ideally, a TLP model that effectively learns temporal patterns should perform worse on temporally distorted data compared to real data. We provide an in-depth analysis of this hypothesis and introduce two data distortion techniques to assess well-known TLP models.

Our contributions are threefold: (1) We introduce simple techniques to distort temporal patterns within a graph, generating temporally distorted test splits of well-known datasets for sanity checks. These distortion methods are applicable to any temporal graph dataset. (2) We perform counterfactual analysis on TLP models such as `JODIE` [1], `TGAT` [2], `TGN` [3], and `CAWN` [4] to evaluate their capability in capturing temporal patterns across different datasets. (3) We propose an alternative evaluation strategy for TLP, addressing the limitations of binary classification and ranking methods, and introduce two metrics -- average time difference (ATD) and average count difference (ACD) -- to provide a comprehensive measure of a model's predictive performance. 


#### References:
[1] Kumar, S., Zhang, X., & Leskovec, J. (2019, July). Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1269-1278).

[2] Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020). Inductive representation learning on temporal graphs. arXiv preprint arXiv:2002.07962.

[3] Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637.

[4] Wang, Y., Chang, Y. Y., Liu, Y., Leskovec, J., & Li, P. (2020, October). Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. In International Conference on Learning Representations.


#### Source Code of the TLP Models:
- `JODIE` and `TGN` are available in the fork: https://github.com/Aniq55/tgn.git
- `TGAT` is available at the fork: https://github.com/Aniq55/TGAT.git
- `CAWN` is available at the fork: https://github.com/Aniq55/CAW.git