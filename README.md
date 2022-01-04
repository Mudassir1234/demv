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
conda env create -f environment.yml
```

### Pip users

Create the virtual environment using the following commands:

```shell
virtualenv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt
```

To reproduce the analysis simply run the corresponding notebooks.

## License
<p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p>
