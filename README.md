# Positivity-Preserving Adaptive Runge-Kutta Methods

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/262750089.svg)](https://zenodo.org/badge/latestdoi/262750089)

This repository contains some code used in the article
```
@online{nusslein2020positivity,
  title={Positivity-Preserving Adaptive {R}unge--{K}utta Methods},
  author={N\"u\ss lein, Stephan and Ranocha, Hendrik and Ketcheson, David I},
  year={2020},
  month={05},
  eprint={2005.XXXXX},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

> Many important differential equations model quantities whose value must remain positive or stay in some bounded interval. These bounds may not be preserved when the model is solved numerically. We propose to ensure positivity or other bounds by applying Runge-Kutta integration in which the method weights are adapted in order to enforce the bounds. The weights are chosen at each step after calculating the stage derivatives, in a way that also preserves (when possible) the order of accuracy of the method. The choice of weights is given by the solution of a linear program. We investigate different approaches to choosing the weights by considering adding further constraints. We also provide some analysis of the properties
of Runge--Kutta methods with perturbed weights. Numerical examples demonstrate the effectiveness of the approach, including application to both stiff and non-stiff problems.

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please cite this repository as
```
@misc{nusslein2020positivityRepro,
  title={{Positive\_RK\_Reproducibility}.
         {P}ositivity-Preserving Adaptive {R}unge--{K}utta Methods},
  author={N\"u{\ss}lein, Stephan and Ranocha, Hendrik and Ketcheson, David I},
  year={2020},
  month={05},
  howpublished={\url{https://github.com/ketch/Positive\_RK\_Reproducibility}},
  doi={10.5281/zenodo.3819791}
}
```


## Installation instructions

To reproduce the numerical experiments, you need to install the following software.

- Python 3 (version 3.8.2)
- [Jupyter](https://jupyter.org/) (jupyter-notebook version 6.0.3)
- [Numba](http://numba.pydata.org/) (version 0.48.0)
  - either `pip3 install --user numba`
  - or `sudo apt install python3-numba`
- [Numpy](https://numpy.org/) (version 1.17.4)
  - either `pip3 install --user numpy`
  - or `sudo apt install python3-numpy`
- [Scipy](https://www.scipy.org/) (version 1.3.3)
  - either `pip3 install --user scipy`
  - or `sudo apt install python3-scipy`
- [Matplotlib](https://matplotlib.org/) (version 3.1.2)
  - either `pip3 install --user matplotlib`
  - or `sudo apt install python3-matplotlib`
- [CVXPY](https://www.cvxpy.org/) (version 1.0.31)
  - `pip3 install --user cvxpy`
- [Mosek](https://www.mosek.com/) (version 9.2.5)
  - `pip3 install --user Mosek`
  - Get and install a license, e.g. a free academic license
- [Nodepy](https://github.com/ketch/nodepy) (version 0.9)
  - `pip3 install --user nodepy`

Then, open a `jupyter notebook` and run the examples. The numerical examples were run with recent software available at the beginning of May 2020 in Kubuntu 20.04.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
