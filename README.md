# Debiaser for Multiple Variables

This repository contains the code used in the paper: _Debiaser for Multiple Variables: a data and model agnostic bias mitigaton method for classification problems_

## Project structure

The project is organized in the following way:

- `data`: contains all the datasets used in the analysis.
- `ris`: contains the metrics related to _DEMV_ evaluation in a `csv` format. The file names follow this pattern: _<dataset_name>\_eval_. In each file we store the stop value and the value of each metric for each iteration.
- `environment.yml and requirements.txt`: dependency files for conda and pip respectively.
- `demv.py`: this file contains the implementation of the _DEMV_ algorithm.
- `utils.py`: this file contains some utility functions used in the notebooks.
- `jupyter notebooks`: each notebook contains the analysis applied to a specific dataset and is named with the same name of the dataset. For the analysis applied with a single sensitive variable the file is named as: _<dataset_name>Singlevar.ipynb_.

## Environment setup

### Conda users

Create the conda environment using the following command:

```shell
$conda env create -f environment.yml
```

### Pip users

Create virtual environment using the following command:

```shell
$virtualenv <env_name>
$source <env_name>/bin/activate
$pip install -r requirements.txt
```

To reproduce the analysis simply run the corresponding notebook.
