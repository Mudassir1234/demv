# DEMV : Debiaser for Multiple Variables

![GitHub last commit](https://img.shields.io/github/last-commit/giordanoDaloisio/demv2022?style=for-the-badge) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)

## Table of contents

- [Citation request](#citation-request)
- [Project structure](#project-structure)
- [General info](#general-info)
- [DEMV class description](#demv-class-description)
  - [Attributes](#attributes)
  - [Methods](#methods)
  - [Example usage](#example-usage)

## Citation request

Please cite our papers if you use DEMV in your experiments:

_Giordano d’Aloisio, Andrea D’Angelo, Antinisca Di Marco, Giovanni Stilo, Debiaser for Multiple Variables to enhance fairness in classification tasks, Information Processing & Management,
Volume 60, Issue 2, 2023, 103226, ISSN 0306-4573, <https://doi.org/10.1016/j.ipm.2022.103226>_

```bibtex
@article{daloisio_debiaser_2023,
title = {Debiaser for Multiple Variables to enhance fairness in classification tasks},
journal = {Information Processing & Management},
volume = {60},
number = {2},
pages = {103226},
year = {2023},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2022.103226},
url = {https://www.sciencedirect.com/science/article/pii/S0306457322003272},
author = {Giordano d’Aloisio and Andrea D’Angelo and Antinisca {Di Marco} and Giovanni Stilo},
keywords = {Machine learning, Bias and Fairness, Multi-class classification, Preprocessing algorithm, Equality},
}
```

_d’Aloisio, G., Stilo, G., Di Marco, A., D’Angelo, A. (2022). Enhancing Fairness in Classification Tasks with Multiple Variables: A Data- and Model-Agnostic Approach. In: Boratto, L., Faralli, S., Marras, M., Stilo, G. (eds) Advances in Bias and Fairness in Information Retrieval. BIAS 2022. Communications in Computer and Information Science, vol 1610. Springer, Cham. <https://doi.org/10.1007/978-3-031-09316-6_11>_

```bibtex
@inproceedings{d2022enhancing,
  title={Enhancing Fairness in Classification Tasks with Multiple Variables: A Data-and Model-Agnostic Approach},
  author={d’Aloisio, Giordano and Stilo, Giovanni and Di Marco, Antinisca and D’Angelo, Andrea},
  booktitle={International Workshop on Algorithmic Bias in Search and Recommendation},
  pages={117--129},
  year={2022},
  organization={Springer}
}
```

## Project structure

This repository is organized as follows:

1. The source code of DEMV is in the `demv` folder.

## General info

DEMV is a Debiaser for Multiple Variables that aims to increase Fairness in any given dataset, both binary and categorical, with one or more sensitive variables, while keeping the accuracy of the classifier as high as possible.
The main idea behind the proposed method is that to enhance the classifier’s fairness during pre-processing effectively is necessary to consider all possible combinations of the values of the sensitive variables and the label’s values for the definition of the so-called _sensitive groups_.

We approach the problem by recursively identifying all the possible groups given by combining all the values of the sensible variables with the belonging label (class). Next, for each group, we compute its expected (𝑊𝑒𝑥𝑝) and observed (𝑊𝑜𝑏𝑠) sizes and look at the ratio among these two values. If 𝑊𝑒𝑥𝑝/𝑊𝑜𝑏𝑠 = 1, it implies that the group is fully balanced. Otherwise, if the ratio is less than one, the group size is larger than expected, so we must remove an
element from the considered group accordingly to a chosen deletion strategy. Finally, if the ratio is greater than one, the group is smaller than expected, so we have to add another item accordingly to a generation strategy. For each group, we recursively repeat this balancing operation until 𝑊𝑒𝑥𝑝/𝑊𝑜𝑏𝑠 converge to one. It is worth noting that, in order to keep a high level of accuracy, the new items added to a group should be coherent in their values with the already existing ones.

The papers describing our work are available at:

- <https://doi.org/10.1016/j.ipm.2022.103226>
- <http://dx.doi.org/10.1007/978-3-031-09316-6_11> ([pdf](https://www.researchgate.net/profile/Giordano-Daloisio/publication/361406303_Enhancing_Fairness_in_Classification_Tasks_with_Multiple_Variables_A_Data-_and_Model-Agnostic_Approach/links/6357a1ca8d4484154a32cf02/Enhancing-Fairness-in-Classification-Tasks-with-Multiple-Variables-A-Data-and-Model-Agnostic-Approach.pdf)).

## DEMV class description

### Attributes

- `round_level : float`

  Tolerance value to balance the sensitive groups

- `debug : bool`

  Prints w_exp/w_obs, useful for debugging

- `stop : int`

  Maximum number of balance iterations

- `strategy: string`

  Balancing strategy to use. Must be one of `smote`, `adasyn` and `uniform` (default is `uniform`)

- `iter : int`

  Maximum number of iterations

### Methods

- `__init__(self, round_level=None, debug=False, stop=-1, strategy='uniform')`

        Creates a new DEMV instance

        Parameters
        ----------
        round_level : float, optional
            Tolerance value to balance the sensitive groups (default is None)
        debug : bool, optional
            Prints w_exp/w_obs, useful for debugging (default is False)
        stop : int, optional
            Maximum number of balance iterations (default is -1)
        strategy: string, optional
            Balancing strategy to use. Must be one of `smote`, `adasyn` and `uniform` (default is `uniform`)

- `fit_transform(dataset: pandas.DataFrame, protected_attrs: list, label_name: str)`

        Balances the dataset's sensitive groups

        Parameters
        ----------
        dataset : pandas.DataFrame
            Dataset to be balanced
        protected_attrs : list
            List of protected attribute names
        label_name : str
            Label name

        Returns
        -------
        pandas.DataFrame :
            Balanced dataset

- `get_iters()`

      Gets the maximum number of iterations

        Returns
        -------
        int:
            maximum number of iterations

### Example usage

In the following we show an example usage of our algorithm:

```python
from demv import DEMV
import pandas as pd

df = pd.read_csv('some_data.csv')
protected_attrs = ['s1','s2']
label = 'l'

demv = DEMV(round_level=1)
df_bal = demv.fit_transform(df, protected_attrs, label)
print('Maximum number of iterations: ',demv.get_iters())
```

## Credits

The original paper was written by Giordano d'Aloisio, Giovanni Stilo, Antinisca di Marco and Andrea D'Angelo.
This work is partially supported by Territori Aperti a project funded by Fondo Territori Lavoro e Conoscenza CGIL CISL UIL, by SoBigData-PlusPlus H2020-INFRAIA-2019-1 EU project, contract number 871042 and by “FAIR-EDU: Promote FAIRness in EDUcation institutions” a project founded by the University of L’Aquila. All the numerical simulations have been realized mostly on the Linux HPC cluster Caliban of the High-Performance Computing Laboratory of the Department of Information Engineering, Computer Science and Mathematics (DISIM) at the University of L’Aquila.

## License

This work is licensed under AGPL 3.0 license.
