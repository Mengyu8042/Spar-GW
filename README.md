# Efficient Approximation of Gromov-Wasserstein Distance using Importance Sparsification
This repository includes the implementation of our work **"Efficient Approximation of Gromov-Wasserstein Distance using Importance Sparsification"** [https://arxiv.org/abs/2205.13573].

![spar-gw](figures/spar-gw.PNG)

If you use this toolbox in your research and find it useful, please cite:
```
@article{li2022efficient,
  title={Efficient Approximation of Gromov-Wasserstein Distance using Importance Sparsification},
  author={Li, Mengyu and Yu, Jun and Xu, Hongteng and Meng, Cheng},
  journal={arXiv preprint arXiv:2205.13573},
  year={2022}
}
```

## Introduction
A brief introduction about the folders and files:
* `methods/`: the proposed method and baselines;
    * `LinearGromov/`: implementation of LR-GW for approximating GW;
    * `gromov_funcs.py`: implementation of EGW-based methods (EGW, PGA-GW, and EMD-GW), AE, SaGroW, and **Spar-GW** for approximating GW (and FGW);
    * `GromovWassersteinFramework.py`, `GromovWassersteinGraphToolkit.py`: implementation of S-GWL for approximating GW;
    * `unbalanced_gromov_funcs.py`: implementation of EUGW, PGA-UGW, SaGroW, and **Spar-UGW** for UGW.
* `results/`: precomputed results;
    * `SYNTHETIC_dist_mat_spargw.mat`: a precomputed FGW distance matrix of the SYNTHETIC dataset, approximated by Spar-GW.
* `data_simulators.py`: generate the used synthetic datasets, i.e., "Moon", "Graph", "Gaussian", or "Spiral".
* `demo_gw_distance.ipynb`: an example of approximating the GW distance using different methods.
* `demo_graph_analysis.ipynb`: an example of graph clustering and graph classification.

You can run `demo_gw_distance.ipynb` and `demo_graph_analysis.ipynb` to test our method for GW distance approximation and graph clustering/classification, respectively.

## Environments
Python: 3.8.8

Install the following dependencies using the `pip` command:

| Name | Version |
| ------ | ------ |
| matplotlib | 3.5.2 |
| ipykernel | 5.3.4 |
| ipython | 7.22.0 |
| networkx | 2.6.3 |
| numpy | 1.21.5 |
| pandas | 1.2.4 |
| POT | 0.8.1.0 |
| scikit-learn | 0.24.2 |
| scipy | 1.8.0 |
| torch | 1.10.1 |
| torch-geometric | 2.0.4 |


## Acknowledgements

This toolbox has been created and is maintained by

* [Mengyu Li](https://github.com/Mengyu8042): limengyu516@ruc.edu.cn
* Jun Yu: yujunbeta@bit.edu.cn
* [Hongteng Xu](https://github.com/HongtengXu): hongtengxu@ruc.edu.cn
* [Cheng Meng](https://github.com/ChengzijunAixiaoli): chengmeng@ruc.edu.cn

Feel free to contact us if any questions.

## Main References
* https://pythonot.github.io/index.html
* https://github.com/meyerscetbon/LinearGromov
* https://github.com/HongtengXu/s-gwl
* https://github.com/Hv0nnus/Sampled-Gromov-Wasserstein
* https://github.com/thibsej/unbalanced_gromov_wasserstein

