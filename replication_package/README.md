# Experiment replication

This folder contains the code to reproduce the experiments reported in the paper. All the scripts are available in the `src` folder.

## Datasets and methods

DEMV was tested with numerous datasets and methods.
Please check the aforementioned paper for more information on the datasets and their preprocessing.

### Datasets

The included datasets are:

| Dataset | Full Name                   | Type       | Description                                                                                                                                                            | Sensitive variables                       |
| ------- | --------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| ADULT   | Adult income                | Binary     | The goal is to predict if a person has an income higher than 50k a year.                                                                                               | Sex, race, bachelors                      |
| COMPAS  | ProPublica Recidivism       | Binary     | The goal is to predict if a person will recidivate in the next two years.                                                                                              | Sex, race, age                            |
| GERMAN  | German credit               | Binary     | The goal is to predict if a person will recidivate in the next two years.                                                                                              | Sex, age, investment_as_income_percentage |
| CMC     | Contraceptive Method Choice | Multiclass | This multi-class dataset comprises 1,473 instances and ten columns about women’s contraceptive method choice.                                                          | wife_religion, wife_work, wife_edu        |
| CRIME   | Communities and Crime       | Multiclass | This multi-class dataset is made of 1,994 instances by 100 attributes and contains information about the per-capita violent crimes in a community.                     | black_people, hisp_people, MedRent        |
| DRUG    | Drug Usage                  | Multiclass | This multi-class dataset has 1,885 instances and 15 attributes about the frequency of drugs consumption.                                                               | race, gender, age                         |
| LAW     | Law School Admission        | Multiclass | This multi-class dataset comprises 20,694 samples by 14 attributes and contains information about the bar passage data of Law School students.                         | race, gender, age                         |
| PARK    | Parkinson's Telemonitoring  | Multiclass | This multi-class dataset comprises 5875 items and 19 features about Unified Parkinson’s Disease Rating Scale (UPDRS) score classification.                             | age, sex, PPE                             |
| WINE    | Wine Quality                | Multiclass | This multi-class dataset comprises 6,438 instances and 13 attributes about wine quality (variable quality). The classes are four increasing values indicating quality. | alcohol, type, density                    |

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
   - [scikit-learn](https://scikit-learn.org/stable/)
   - [fairlearn](https://fairlearn.org/)

### Run script

In order to replicate those tests, once the project directory has been downloaded, please open the terminal and move to the project folder within the terminal.

Now you can type

`python generatemetrics.py -h`

to receive help and information on the requested parameters. Shortly put, you should provide the dataset, the method, the number of sensitive variables to consider, and optionally the classifier you want to use (it defaults to Logistic Regression).
Here is the output of the help command, which explains the parameters:

```shell
usage: generatemetrics.py [-h] [--classifier [{logistic,gradient,svc,mlp}]] [--cm | --no-cm] [--sensitivefeature SENSITIVEFEATURE]
                          {adult,cmc,law,compas,crime,drug,german,obesity,park,wine,all} {biased,eg,grid,uniform,smote,adasyn} {1,2,3,4}

Metrics generator for DEMV testing.

positional arguments:
  {adult,cmc,law,compas,crime,drug,german,obesity,park,wine,all}
                        Required argument: Chosen dataset to generate metrics for. Availability of datasets changes according to the
                        chosen method. All available datasets are: adult, cmc, compas, crime, drugs, german, obesity, park, wine.
  {biased,eg,grid,uniform,smote,adasyn}
                        Required argument: Chosen method to generate metrics for. Can be biased, eg, grid, uniform, smote, adasyn.
  {1,2,3,4}             Required argument: Number of sensitive features in the dataset to consider, up to 3. If "1" is chosen, two
                        datasets will be generated, one for each canonical sensitive feature (as described in literature for that
                        dataset)

optional arguments:
  -h, --help            show this help message and exit
  --classifier [{logistic,gradient,svc,mlp}]
                        Optional argument: classifier to use. Possible options are logistic, gradient, svc and mlp. Defaults to Logistic
                        Regression (logistic).
  --cm, --no-cm         Optional argument: only generate Confusion Matrices for the selected dataset.
  --sensitivefeature SENSITIVEFEATURE
                        Optional argument: force sensitive feature to be considered (Only works if number of features is 1)

Example usage: python generatemetrics.py cmc biased 3 --classifier svc
```

Please note that not all datasets, given their properties, have 4 available number_of_features or can be run with any method. Please contact us if you notice some unusual error. All the testing done by us was documented in the paper.

For instance:

`python generatemetrics.py cmc biased 3 --classifier svc`

or

`python generatemetrics.py crime uniform 2`

Results will then be saved in the folder "ris" inside the folder generatemetrics, according to how many sensitive variables you have chosen to consider, up to 3. In particular, the output file will have the following structure:

`ris/[number_of_features]features/metrics_[DATASET]_[METHOD]_[NUMBER_OF_FEATURES]_[CLASSIFIER].csv`

A temporary discard_eval.csv file will also be created, but can be removed at any time and will always be overwritten by the subsequent execution.

### Optional arguments

We will now describe in greater detail the optional arguments.

1. `classifier`

It is possible to choose between multiple classifiers to generate the metrics. The default one is Logistic Regression. However, with the --classifier argument, it is possible to choose between Logistic Regression, Support Vector Classifier, Multilayer perceptron and Gradient Boosting Classifier.

2. `sensitivefeature`

You can also specify the parameter --sensitivefeature if you want to choose a specific sensitive feature to generate metrics for, between those listed in the above table. If the dataset allows for it, metrics will be generated considering those specific features as sensitive.
The number of specified sensitive features must be equal to the number_of_feature parameter. For instance:

`python generatemetrics.py cmc biased 1 --sensitivefeature wife_religion`

or

`python generatemetrics.py cmc biased 1 --sensitivefeature wife_religion,wife_work`

Multiple sensitive features specified this way must be separated by a comma.

3. `cm (confusion matrices)`

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
