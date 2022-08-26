# DEMV : Debiaser for Multiple Variables

![GitHub last commit](https://img.shields.io/github/last-commit/giordanoDaloisio/demv2022?style=for-the-badge) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)

## Table of contents

- [General info](#general-info)
- [Citation request](#citation-request)
- [Project structure](#project-structure)
- [Datasets and methods](#datasets-and-methods)
- [Experiment replication](#experiment-replication)
- [Credits](#credits)
- [License](#license)

## General info

DEMV is a Debiaser for Multiple Variables that aims to increase Fairness in any given dataset, both binary and categorical, with one or more sensitive variables, while keeping the accuracy of the classifier as high as possible.
The main idea behind the proposed method is that to enhance the classifier’s fairness during pre-processing effectively is necessary to consider all possible combinations of the values of the sensitive variables and the label’s values for the definition of the so-called _sensitive groups_.

We approach the problem by recursively identifying all the possible groups given by combining all the values of the sensible variables with the belonging label (class). Next, for each group, we compute its expected (𝑊𝑒𝑥𝑝) and observed (𝑊𝑜𝑏𝑠) sizes and look at the ratio among these two values. If 𝑊𝑒𝑥𝑝/𝑊𝑜𝑏𝑠 = 1, it implies that the group is fully balanced. Otherwise, if the ratio is less than one, the group size is larger than expected, so we must remove an
element from the considered group accordingly to a chosen deletion strategy. Finally, if the ratio is greater than one, the group is smaller than expected, so we have to add another item accordingly to a generation strategy. For each group, we recursively repeat this balancing operation until 𝑊𝑒𝑥𝑝/𝑊𝑜𝑏𝑠 converge to one. It is worth noting that, in order to keep a high level of accuracy, the new items added to a group should be coherent in their values with the already existing ones.

The paper describing our work is available at <http://dx.doi.org/10.1007/978-3-031-09316-6_11>.

## Citation request

Please cite our paper if you use DEMV in your experiments:

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

1. Useful files to create and manage the environment are listed in the main repository (requirements.txt, environment.yml). See [Experiment Replication](#experiment-replication) for more informations.
2. The DEMV algorithm is contained within the demv.py file. This file contains all the necessary functions to generate metrics with demv and other methods.
3. The scripts used to actually generate the metrics are generatemetrics.py and getdataset.py. To replicate tests, see the [Experiment Replication](#experiment-replication) section.
4. The data folder contains all the datasets listed in section [Datasets and methods](#datasets-and-methods).

## Datasets and methods

DEMV was tested with numerous datasets and methods. 
Please check the aforementioned paper for more information on the datasets and their preprocessing.

### Datasets

The included datasets are:

| Dataset  | Full Name | Type | Description  | Sensitive variables  |
|---|---|---|---| ---|
| ADULT  | Adult income | Binary | The goal is to predict if a person has an income higher than 50k a year.   |  Sex, race, bachelors |
| COMPAS | ProPublica Recidivism | Binary | The goal is to predict if a person will recidivate in the next two years. |Sex, race, age  |
| GERMAN| German credit | Binary | The goal is to predict if a person will recidivate in the next two years. | Sex, age, investment_as_income_percentage |
|CMC| Contraceptive Method Choice | Multiclass | This multi-class dataset comprises 1,473 instances and ten columns about women’s contraceptive method choice. | wife_religion, wife_work, wife_edu |
| CRIME | Communities and Crime | Multiclass | This multi-class dataset is made of 1,994 instances by 100 attributes and contains information about the per-capita violent crimes in a community. | black_people, hisp_people, MedRent |
| DRUG | Drug Usage | Multiclass | This multi-class dataset has 1,885 instances and 15 attributes about the frequency of drugs consumption. | race, gender, age  |
| LAW | Law School Admission | Multiclass | This multi-class dataset comprises 20,694 samples by 14 attributes and contains information about the bar passage data of Law School students.| race, gender, age |
| PARK | Parkinson's Telemonitoring | Multiclass |  This multi-class dataset comprises 5875 items and 19 features about Unified Parkinson’s Disease Rating Scale (UPDRS) score classification. | age, sex, PPE  |
| WINE | Wine Quality | Multiclass | This multi-class dataset comprises 6,438 instances and 13 attributes about wine quality (variable quality). The classes are four increasing values indicating quality. | alcohol, type, density |

## Experiment replication

### Environment setup

1. **Conda users**

   Create the conda environment using the following command:

   ```shell
   conda env create -f environment.yml
   ```

2. **Pip users**

   Create the virtual environment using the following commands:

   ```shell
   virtualenv <env_name>
   source <env_name>/bin/activate
   pip install -r requirements.txt
   ```

3. **Manual setup**

   If the previous ways to not work just install the following libraries manually:

   - [pandas](https://pandas.pydata.org/)
   - [numpy](https://numpy.org/)
   - [imblearn](https://imbalanced-learn.org/stable/)
   - [scikit-learn](https://scikit-learn.org/stable/)
   - [fairlearn](https://fairlearn.org/)

### Run script

In order to replicate those tests, once the project directory has been downloaded, please open the terminal and move to the project folder within the terminal.

Now you can type

`python generatemetrics.py -h`

to receive help and information on the requested parameters. Shortly put, you should provide the dataset, the method, the number of sensitive variables to consider, and optionally the classifier you want to use (it defaults to Logistic Regression).
Here is the output of the help command, which explains the parameters:

```shell
usage: generatemetrics.py [-h] [--classifier [{logistic,gradient,svc,mlp}]] [--cm | --no-cm]
                          {adult,cmc,law,compas,crime,drugs,german,obesity,park,wine,all} {biased,eg,grid,uniform,smote,adasyn} {1,2,3,4}

Metrics generator for DEMV testing.

positional arguments:
  {adult,cmc,law,compas,crime,drugs,german,obesity,park,wine,all}
                        Required argument: Chosen dataset to generate metrics for. Availability of datasets changes according to the chosen method.
                        All available datasets are: adult, cmc, compas, crime, drugs, german, obesity, park, wine.
  {biased,eg,grid,uniform,smote,adasyn}
                        Required argument: Chosen method to generate metrics for. Can be biased, eg, grid, uniform, smote, adasyn.
  {1,2,3,4}             Required argument: Number of sensitive features in the dataset to consider, up to 4.

optional arguments:
  -h, --help            show this help message and exit
  --classifier [{logistic,gradient,svc,mlp}]
                        Optional argument: classifier to use. Possible options are logistic, gradient, svc and mlp. Defaults to Logistic Regression
                        (logistic).
  --cm, --no-cm         Optional argument: only generate Confusion Matrices for the selected dataset.

Example usage: python generatemetrics.py cmc biased 3 --classifier svc
```

Please note that not all datasets, given their properties, have 4 available number_of_features or can be run with any method. Please contact us if you notice some unusual error. All the testing done by us was documented in the paper.

For instance:

`python generatemetrics.py cmc biased 3 --classifier svc`

or

`python generatemetrics.py crime uniform 2`

Results will then be saved in the folder "ris" inside the folder generatemetrics, according to how many sensitive variables you have chosen to consider, up to 4. In particular, the output file will have the following structure:

`ris/[number_of_features]features/metrics_[DATASET]_[METHOD]_[NUMBER_OF_FEATURES]_[CLASSIFIER].csv`

A temporary discard_eval.csv file will also be created, but can be removed at any time and will always be overwritten by the subsequent execution.

Another use case can be generating the confusion matrices for a dataset instead of the related metrics. In order to do that, just add the option --cm. In this case the number of features and method will be ignored, and several confusion matrices will be generated. In particular, the command

`python generatemetrics.py cmc demv 3 --cm`

will ignore the "demv" and "3" attributes and generate the following confusion matrices:

1. Confusion matrix for the sensible groups in the dataset:
    1. Without any debiaser
    2. Pre-processed with DEMV
    3. Processed with Exponentiated Gradient
    4. Processed with Grid Search
2. Confusion matrix for the non-sensible groups in the dataset:
    1. Without any debiaser
    2. Pre-processed with DEMV
    3. Processed with Exponentiated Gradient
    4. Processed with Grid Search.
  

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
