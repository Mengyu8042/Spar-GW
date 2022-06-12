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
    * `gromov_funcs.py`: implementation of EGW-based methods (EGW, PGA-GW, and EMD-GW), AE, SaGroW, and **Spar-GW** for approximating GW (and FGW);
    * `GromovWassersteinFramework.py`, `GromovWassersteinGraphToolkit.py`: implementation of S-GWL and LR-GW for approximating GW;
    * `unbalanced_gromov_funcs.py`: implementation of EUGW, PGA-UGW, SaGroW, and **Spar-UGW** for UGW.
* `data_simulators.py`: generate the used synthetic datasets, i.e., "Moon", "Graph", "Gaussian", or "Spiral".
* `demo_gromov.ipynb`: an example of approximating the GW distance using different methods.
* `graph_analysis.py`: an example of graph clustering and graph classification.

You can run `demo_gromov.ipynb` or `graph_analysis.py` to test our method.

## Main Dependencies
Install the following requirements using the `pip` command:
* matplotlib
* networkx
* numpy
* pandas
* POT
* random
* scipy
* sklearn
* torch
* torch_geometric


## Acknowledgements

This toolbox has been created and is maintained by

* [Mengyu Li](https://github.com/Mengyu8042): limengyu516@ruc.edu.cn
* Jun Yu: yujunbeta@bit.edu.cn
* [Hongteng Xu](https://github.com/HongtengXu): hongtengxu@ruc.edu.cn
* [Cheng Meng](https://github.com/ChengzijunAixiaoli): chengmeng@ruc.edu.cn

Feel free to contact us if any question.

## Main References
* https://pythonot.github.io/index.html
* https://github.com/HongtengXu/s-gwl
* https://github.com/Hv0nnus/Sampled-Gromov-Wasserstein
* https://github.com/thibsej/unbalanced_gromov_wasserstein

