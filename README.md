# Debiaser for Multiple Variables

This repository contains the implementation of _DEMV_ algorithm described in the paper: _Enhancing Fairness in Classification Tasks with Multiple Variables: a Data- and Model-Agnostic Approach_

## Project structure

The project is organized in the following way:

- `environment.yml and requirements.txt`: dependency files for conda and pip respectively.
- `demv.py`: this file contains the implementation of the _DEMV_ algorithm.

## Environment setup

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

## Class description

### Attributes

- `round_level : float`
      Tolerance value to balance the sensitive groups

- `debug : bool`
      Prints w_exp/w_obs, useful for debugging

- `stop : int`
      Maximum number of balance iterations, after which the algorithm stops

- `iter : int`
      Maximum number of iterations computed by the algorithm to balance the groups

### Methods

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

## License

This work is licensed under AGPL 3.0 license
